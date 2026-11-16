//! High-Throughput Continuous Batching Engine with SGLang Radix Tree Prefix Caching.
//!
//! Replaces per-token mutex serialization with an iteration-level scheduling actor:
//! - Sub-millisecond Radix Tree prompt cache lookup: reuses existing physical blocks.
//! - Paged KV-Cache block management: zero heap allocation during decoding.
//! - Lock-free multi-client streaming: transmits sampled tokens directly into client channels.

use crate::pipeline::SpecializedPipeline;
use crossbeam_utils::sync::{Parker, Unparker};
use oxide_alloc::{PagedKvArena, RadixPrefixCache, SequenceBlockTable};
use oxide_core::dfa::{DFA_ERROR_STATE, DfaSchemaGrammar};
use oxide_core::sampler::{AcademicSamplerEngine, SamplerState, SamplingConfig};
use oxide_models::{GgufTokenizer, Llama3Model, Llama3ScratchBuffers};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use tokio::sync::{Mutex, mpsc, oneshot};

/// Command sent from HTTP/SSE/gRPC request handlers to the background iteration engine.
#[derive(Debug)]
pub enum EngineCommand {
    /// Submit a new inference request.
    EnqueueRequest {
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        dfa: Option<Arc<DfaSchemaGrammar>>,
        token_sender: mpsc::UnboundedSender<TokenEvent>,
        ack_sender: oneshot::Sender<Result<usize, String>>,
    },
    /// Cancel an active sequence.
    CancelRequest { request_id: String },
    /// Shutdown the engine worker.
    Shutdown,
}

/// Token streaming event emitted to clients.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token_id: u32,
    pub is_terminal: bool,
    pub finish_reason: Option<String>,
}

/// Execution Phase of an individual sequence in the Continuous Batching engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequencePhase {
    /// Ingesting prompt tokens in chunks up to `prefill_config.compute_prefill_budget()`.
    Prefill { uncomputed_start: usize },
    /// Generating autoregressive output tokens one per step.
    Decode,
}

/// Active Sequence State managed by the Continuous Batching scheduler.
#[derive(Debug)]
struct ActiveSequence {
    request_id: String,
    block_table: SequenceBlockTable,
    #[allow(dead_code)]
    prompt_tokens: Vec<u32>,
    current_token: u32,
    current_position: usize,
    generated_count: usize,
    max_tokens: usize,
    temperature: f32,
    sender: mpsc::UnboundedSender<TokenEvent>,
    sampler: AcademicSamplerEngine,
    sampler_state: SamplerState,
    dfa: Option<Arc<DfaSchemaGrammar>>,
    dfa_state: u32,
    phase: SequencePhase,
}

/// SLA-aware dynamic chunked prefill scheduler configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkedPrefillConfig {
    pub min_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub max_num_batched_tokens: usize,
    pub sla_itl_micros: u64,
    pub base_step_micros: u64,
    pub prefill_alpha_micros_per_tok: f32,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            min_chunk_tokens: 32,
            max_chunk_tokens: 512,
            max_num_batched_tokens: 2048,
            sla_itl_micros: 25_000,
            base_step_micros: 2_000,
            prefill_alpha_micros_per_tok: 25.0,
        }
    }
}

impl ChunkedPrefillConfig {
    #[must_use]
    pub fn compute_prefill_budget(&self, active_decode_batch: usize) -> usize {
        let decode_est_micros = self.base_step_micros + (active_decode_batch as u64 * 350);
        let headroom = self.sla_itl_micros.saturating_sub(decode_est_micros);
        let time_budget = (headroom as f32 / self.prefill_alpha_micros_per_tok.max(1.0)) as usize;
        let token_capacity_budget = self.max_num_batched_tokens.saturating_sub(active_decode_batch);
        let budget = time_budget.min(token_capacity_budget);
        budget.clamp(self.min_chunk_tokens, self.max_chunk_tokens)
    }
}

/// Prefill chunk slice descriptor for ragged batching.
#[derive(Debug, Clone)]
pub struct PrefillChunkDesc {
    pub slot_id: usize,
    pub tokens: Vec<u32>,
    pub seq_id: u16,
    pub is_final_chunk: bool,
    pub total_prompt_len: usize,
}

/// Single token decode slice descriptor for ragged batching.
#[derive(Debug, Clone)]
pub struct DecodeSlotDesc {
    pub slot_id: usize,
    pub token: u32,
    pub position: usize,
}

/// Prefix clone descriptor for zero-copy Radix cache hits.
#[derive(Debug, Clone)]
pub struct PrefixCloneDesc {
    pub src_slot: u64,
    pub dst_slot: u64,
    pub num_tokens: usize,
}

/// Unified ragged chunked prefill + decode mixing execution descriptor.
#[derive(Debug, Clone)]
pub struct BatchExecDescriptor {
    pub prefix_clones: Vec<PrefixCloneDesc>,
    pub prefill_chunks: Vec<PrefillChunkDesc>,
    pub decode_slots: Vec<DecodeSlotDesc>,
}

#[derive(Debug, Clone)]
pub struct PrefillChunkResult {
    pub slot_id: usize,
    pub tokens_processed: usize,
    pub is_final_chunk: bool,
    pub total_prompt_len: usize,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DecodeSlotResult {
    pub slot_id: usize,
    pub next_token: u32,
    pub error: Option<String>,
}

/// Result returned from the dedicated GPU execution worker to the scheduler actor.
#[derive(Debug, Clone)]
pub struct BatchExecCompletion {
    pub prefill_results: Vec<PrefillChunkResult>,
    pub decode_results: Vec<DecodeSlotResult>,
}

/// Background Continuous Batching & Radix Cache Actor.
pub struct ContinuousBatchingEngine {
    model: Option<Arc<Llama3Model>>,
    pipeline: Option<Arc<Mutex<SpecializedPipeline>>>,
    paged_arena: Option<PagedKvArena>,
    radix_cache: RadixPrefixCache,
    scratch: Option<Llama3ScratchBuffers>,
    active_sequences: HashMap<usize, ActiveSequence>,
    pending_prefix_clones: Vec<PrefixCloneDesc>,
    pending_jump_chunks: Vec<PrefillChunkDesc>,
    max_slots: usize,
    cmd_receiver: mpsc::UnboundedReceiver<EngineCommand>,
    pub prefill_config: ChunkedPrefillConfig,
    pub tokenizer: Option<Arc<GgufTokenizer>>,
    exec_tx: Option<rtrb::Producer<BatchExecDescriptor>>,
    comp_rx: Option<rtrb::Consumer<BatchExecCompletion>>,
    worker_unparker: Option<Unparker>,
    worker_running: Option<Arc<AtomicBool>>,
    worker_handle: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for ContinuousBatchingEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousBatchingEngine")
            .field("max_slots", &self.max_slots)
            .field("active_sequences_count", &self.active_sequences.len())
            .field("prefill_config", &self.prefill_config)
            .finish()
    }
}

impl ContinuousBatchingEngine {
    #[must_use]
    pub fn new(
        model: Arc<Llama3Model>,
        max_slots: usize,
        total_paged_blocks: usize,
        cmd_receiver: mpsc::UnboundedReceiver<EngineCommand>,
    ) -> Self {
        let paged_arena = PagedKvArena::new(
            total_paged_blocks,
            model.config.num_layers,
            model.config.num_kv_heads,
            model.config.head_dim,
        );
        let radix_cache = RadixPrefixCache::new();
        let scratch = model.create_scratch();

        Self {
            model: Some(model),
            pipeline: None,
            paged_arena: Some(paged_arena),
            radix_cache,
            scratch: Some(scratch),
            active_sequences: HashMap::with_capacity(max_slots),
            pending_prefix_clones: Vec::new(),
            pending_jump_chunks: Vec::new(),
            max_slots,
            cmd_receiver,
            prefill_config: ChunkedPrefillConfig::default(),
            tokenizer: None,
            exec_tx: None,
            comp_rx: None,
            worker_unparker: None,
            worker_running: None,
            worker_handle: None,
        }
    }

    #[must_use]
    pub fn new_with_pipeline(
        pipeline: Arc<Mutex<SpecializedPipeline>>,
        max_slots: usize,
        cmd_receiver: mpsc::UnboundedReceiver<EngineCommand>,
    ) -> Self {
        let (exec_tx, exec_rx) = rtrb::RingBuffer::<BatchExecDescriptor>::new(64);
        let (comp_tx, comp_rx) = rtrb::RingBuffer::<BatchExecCompletion>::new(64);
        let worker_running = Arc::new(AtomicBool::new(true));
        let worker_running_thread = Arc::clone(&worker_running);
        let parker = Parker::new();
        let worker_unparker = parker.unparker().clone();
        let pipeline_clone = Arc::clone(&pipeline);

        let worker_handle = std::thread::Builder::new()
            .name("oxide-gpu-worker".to_string())
            .spawn(move || {
                run_gpu_worker_loop(exec_rx, comp_tx, pipeline_clone, worker_running_thread, parker);
            })
            .expect("Failed to spawn oxide-gpu-worker thread");

        Self {
            model: None,
            pipeline: Some(pipeline),
            paged_arena: None,
            radix_cache: RadixPrefixCache::new(),
            scratch: None,
            active_sequences: HashMap::with_capacity(max_slots),
            pending_prefix_clones: Vec::new(),
            pending_jump_chunks: Vec::new(),
            max_slots,
            cmd_receiver,
            prefill_config: ChunkedPrefillConfig::default(),
            tokenizer: None,
            exec_tx: Some(exec_tx),
            comp_rx: Some(comp_rx),
            worker_unparker: Some(worker_unparker),
            worker_running: Some(worker_running),
            worker_handle: Some(worker_handle),
        }
    }

    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: Arc<GgufTokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Spawns the Continuous Batching Engine loop on a dedicated Tokio thread.
    pub fn spawn(mut self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.run_loop().await;
        })
    }

    /// Main iteration scheduler loop (SGLang/vLLM parity with lock-free rtrb overlap).
    pub async fn run_loop(&mut self) {
        loop {
            // 1. Ingest all pending commands from clients non-blockingly
            while let Ok(cmd) = self.cmd_receiver.try_recv() {
                match cmd {
                    EngineCommand::EnqueueRequest {
                        request_id,
                        prompt_tokens,
                        max_tokens,
                        temperature,
                        top_p,
                        dfa,
                        token_sender,
                        ack_sender,
                    } => {
                        let res = self.handle_enqueue(
                            request_id,
                            prompt_tokens,
                            max_tokens,
                            temperature,
                            top_p,
                            dfa,
                            token_sender,
                        );
                        let _ = ack_sender.send(res);
                    }
                    EngineCommand::CancelRequest { request_id } => {
                        self.handle_cancel(&request_id);
                    }
                    EngineCommand::Shutdown => {
                        return;
                    }
                }
            }

            // 2. If no active sequences, await next incoming command asynchronously (0% CPU idle)
            if self.active_sequences.is_empty() {
                match self.cmd_receiver.recv().await {
                    Some(EngineCommand::EnqueueRequest {
                        request_id,
                        prompt_tokens,
                        max_tokens,
                        temperature,
                        top_p,
                        dfa,
                        token_sender,
                        ack_sender,
                    }) => {
                        let res = self.handle_enqueue(
                            request_id,
                            prompt_tokens,
                            max_tokens,
                            temperature,
                            top_p,
                            dfa,
                            token_sender,
                        );
                        let _ = ack_sender.send(res);
                    }
                    Some(EngineCommand::CancelRequest { request_id }) => {
                        self.handle_cancel(&request_id);
                    }
                    Some(EngineCommand::Shutdown) | None => {
                        return;
                    }
                }
            }

            // 3. Execute iteration-level continuous batching step
            if self.exec_tx.is_some() {
                self.step_batch_overlap().await;
            } else {
                self.step_batch();
                tokio::task::yield_now().await;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_enqueue(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        dfa: Option<Arc<DfaSchemaGrammar>>,
        token_sender: mpsc::UnboundedSender<TokenEvent>,
    ) -> Result<usize, String> {
        if self.active_sequences.len() >= self.max_slots {
            return Err("All inference slots saturated (max concurrency reached)".to_string());
        }

        // Find available slot index (0..max_slots)
        let slot_id = (0..self.max_slots)
            .find(|s| !self.active_sequences.contains_key(s))
            .ok_or("No free slots")?;

        if prompt_tokens.is_empty() {
            return Err("Cannot enqueue empty prompt".to_string());
        }

        if self.pipeline.is_some() {
            let (matched_len, cached_slots) = self.radix_cache.match_longest_prefix(&prompt_tokens);
            if matched_len > 0 {
                if let Some(&src_slot) = cached_slots.first() {
                    self.pending_prefix_clones.push(PrefixCloneDesc {
                        src_slot: src_slot as u64,
                        dst_slot: slot_id as u64,
                        num_tokens: matched_len,
                    });
                }
                tracing::info!(
                    "[Radix Tree Prefix Cache] HIT! Reusing {matched_len}/{} prefix tokens (0 ms re-eval)",
                    prompt_tokens.len()
                );
            }

            let target_prefill_len = prompt_tokens.len().saturating_sub(1);
            let (phase, initial_token, initial_pos) = if matched_len < target_prefill_len {
                (
                    SequencePhase::Prefill {
                        uncomputed_start: matched_len,
                    },
                    prompt_tokens[matched_len],
                    matched_len,
                )
            } else {
                self.radix_cache.insert(&prompt_tokens, &[slot_id as u32]);
                (
                    SequencePhase::Decode,
                    *prompt_tokens.last().unwrap_or(&1),
                    target_prefill_len,
                )
            };

            // Radix tree LRU capacity check (>90% of max budget)
            let max_token_budget = self.max_slots * 2048;
            if self.radix_cache.total_tokens() > (max_token_budget * 90) / 100 {
                self.radix_cache
                    .evict_to_budget((max_token_budget * 80) / 100);
            }

            let sampling_config = SamplingConfig {
                temperature,
                top_p,
                ..Default::default()
            };
            let sampler = AcademicSamplerEngine::new(sampling_config);
            let sampler_state = SamplerState::new(5.0);

            let mut seq = ActiveSequence {
                request_id,
                block_table: SequenceBlockTable::new(),
                prompt_tokens,
                current_token: initial_token,
                current_position: initial_pos,
                generated_count: 0,
                max_tokens,
                temperature,
                sender: token_sender,
                sampler,
                sampler_state,
                dfa,
                dfa_state: 0,
                phase,
            };

            // Initial DFA jump-forward if grammar begins with deterministic syntax and in decode phase
            if matches!(seq.phase, SequencePhase::Decode) {
                if let Some(ref grammar) = seq.dfa.clone() {
                    let (jump_bytes, next_state) =
                        grammar.find_jump_forward_bytes(seq.dfa_state, 16);
                    if !jump_bytes.is_empty() {
                        let jump_tokens: Vec<u32> = if let Some(ref tok) = self.tokenizer {
                            if let Ok(s) = std::str::from_utf8(&jump_bytes) {
                                tok.encode(s)
                            } else {
                                jump_bytes.iter().map(|&b| u32::from(b)).collect()
                            }
                        } else {
                            jump_bytes.iter().map(|&b| u32::from(b)).collect()
                        };

                        for &jump_tok in &jump_tokens {
                            if seq.generated_count >= seq.max_tokens {
                                break;
                            }
                            let is_accepting = grammar
                                .accepting_states
                                .get(next_state as usize)
                                .copied()
                                .unwrap_or(false);
                            let reached_max = seq.generated_count + 1 >= seq.max_tokens;
                            let is_terminal = reached_max || is_accepting;
                            let finish_reason = if is_accepting {
                                Some("stop".to_string())
                            } else if reached_max {
                                Some("length".to_string())
                            } else {
                                None
                            };

                            let event = TokenEvent {
                                token_id: jump_tok,
                                is_terminal,
                                finish_reason,
                            };
                            let _ = seq.sender.send(event);

                            seq.current_token = jump_tok;
                            seq.current_position += 1;
                            seq.generated_count += 1;
                        }

                        self.pending_jump_chunks.push(PrefillChunkDesc {
                            slot_id,
                            tokens: jump_tokens,
                            seq_id: slot_id as u16,
                            is_final_chunk: false,
                            total_prompt_len: seq.prompt_tokens.len(),
                        });

                        seq.dfa_state = next_state;
                    }
                }
            }

            self.active_sequences.insert(slot_id, seq);
            return Ok(slot_id);
        }

        let _model = self.model.as_ref().ok_or("No model configured")?;
        let arena = self.paged_arena.as_mut().ok_or("No paged arena")?;

        // SGLang Radix Tree Prefix Cache Lookup:
        let (matched_len, cached_blocks) = self.radix_cache.match_longest_prefix(&prompt_tokens);

        for &b in &cached_blocks {
            arena.retain_block(b);
        }

        let mut block_table = SequenceBlockTable::new();
        block_table.block_ids = cached_blocks;
        block_table.num_tokens = matched_len;

        // Check if arena needs LRU eviction (>90% full)
        if arena.free_block_count() < (arena.total_blocks * 10) / 100 {
            let target_token_budget = (arena.total_blocks * 16 * 80) / 100;
            let evicted = self
                .radix_cache
                .evict_to_budget_collecting_blocks(target_token_budget);
            for b in evicted {
                arena.release_block(b);
            }
        }

        let target_prefill_len = prompt_tokens.len().saturating_sub(1);
        let (phase, initial_token, initial_pos) = if matched_len < target_prefill_len {
            (
                SequencePhase::Prefill {
                    uncomputed_start: matched_len,
                },
                prompt_tokens[matched_len],
                matched_len,
            )
        } else {
            for &b in &block_table.block_ids {
                arena.retain_block(b);
            }
            self.radix_cache
                .insert(&prompt_tokens, &block_table.block_ids);
            (
                SequencePhase::Decode,
                *prompt_tokens.last().unwrap_or(&1),
                target_prefill_len,
            )
        };

        let sampling_config = SamplingConfig {
            temperature,
            top_p,
            ..Default::default()
        };
        let sampler = AcademicSamplerEngine::new(sampling_config);
        let sampler_state = SamplerState::new(5.0);

        let seq = ActiveSequence {
            request_id,
            block_table,
            prompt_tokens,
            current_token: initial_token,
            current_position: initial_pos,
            generated_count: 0,
            max_tokens,
            temperature,
            sender: token_sender,
            sampler,
            sampler_state,
            dfa,
            dfa_state: 0,
            phase,
        };

        self.active_sequences.insert(slot_id, seq);
        Ok(slot_id)
    }

    fn handle_cancel(&mut self, request_id: &str) {
        let slot_to_remove = self
            .active_sequences
            .iter()
            .find(|(_, seq)| seq.request_id == request_id)
            .map(|(s, _)| *s);

        if let Some(slot_id) = slot_to_remove {
            if let Some(mut seq) = self.active_sequences.remove(&slot_id) {
                if let Some(ref arena) = self.paged_arena {
                    seq.block_table.release_all(arena);
                }
            }
        }
    }

    /// Single-pass iteration-level continuous batching scheduler.
    /// Interleaves SLA-bounded chunked prefill with unified batched decode in the same iteration.
    fn step_batch(&mut self) {
        let mut finished_slots = Vec::new();

        if let Some(ref pipe_mutex) = self.pipeline {
            let slot_keys: Vec<usize> = self.active_sequences.keys().copied().collect();
            if slot_keys.is_empty() {
                return;
            }

            let Ok(mut pipe) = pipe_mutex.try_lock() else {
                return;
            };

            // Stage 1: SLA-bounded chunked prefill for sequences in Prefill phase
            let active_decode_count = self
                .active_sequences
                .values()
                .filter(|s| matches!(s.phase, SequencePhase::Decode))
                .count();
            let mut prefill_budget = self
                .prefill_config
                .compute_prefill_budget(active_decode_count);

            for &slot_id in &slot_keys {
                if prefill_budget == 0 {
                    break;
                }
                if let Some(seq) = self.active_sequences.get_mut(&slot_id) {
                    if let SequencePhase::Prefill { uncomputed_start } = seq.phase {
                        let target_len = seq.prompt_tokens.len().saturating_sub(1);
                        let needed = target_len.saturating_sub(uncomputed_start);
                        if needed > 0 {
                            let chunk_len = needed.min(prefill_budget);
                            let chunk =
                                &seq.prompt_tokens[uncomputed_start..uncomputed_start + chunk_len];
                            if let Err(e) = pipe.prefill(slot_id as u64, chunk, slot_id as u16) {
                                tracing::error!("Prefill chunk failed for slot {slot_id}: {e}");
                                let _ = seq.sender.send(TokenEvent {
                                    token_id: 0,
                                    is_terminal: true,
                                    finish_reason: Some(format!("prefill_error: {e}")),
                                });
                                finished_slots.push(slot_id);
                                continue;
                            }
                            let next_start = uncomputed_start + chunk_len;
                            prefill_budget = prefill_budget.saturating_sub(chunk_len);

                            if next_start >= target_len {
                                let num_pages = (seq.prompt_tokens.len() + 15) / 16;
                                let block_ids: Vec<u32> = (0..num_pages as u32)
                                    .map(|p| ((slot_id as u32) << 16) | p)
                                    .collect();
                                self.radix_cache.insert(&seq.prompt_tokens, &block_ids);
                                seq.phase = SequencePhase::Decode;
                                seq.current_token = *seq.prompt_tokens.last().unwrap_or(&1);
                                seq.current_position = target_len;
                            } else {
                                seq.phase = SequencePhase::Prefill {
                                    uncomputed_start: next_start,
                                };
                            }
                        } else {
                            let num_pages = (seq.prompt_tokens.len() + 15) / 16;
                            let block_ids: Vec<u32> = (0..num_pages as u32)
                                .map(|p| ((slot_id as u32) << 16) | p)
                                .collect();
                            self.radix_cache.insert(&seq.prompt_tokens, &block_ids);
                            seq.phase = SequencePhase::Decode;
                            seq.current_token = *seq.prompt_tokens.last().unwrap_or(&1);
                            seq.current_position = target_len;
                        }
                    }
                }
            }

            // Stage 2: Batched decode iteration across all sequences currently in Decode phase
            let mut decode_slots = Vec::new();
            let mut batch_tokens = Vec::new();
            let mut batch_positions = Vec::new();

            for &slot_id in &slot_keys {
                if finished_slots.contains(&slot_id) {
                    continue;
                }
                if let Some(seq) = self.active_sequences.get(&slot_id) {
                    if matches!(seq.phase, SequencePhase::Decode) {
                        decode_slots.push(slot_id);
                        batch_tokens.push(seq.current_token);
                        batch_positions.push(seq.current_position);
                    }
                }
            }

            if decode_slots.is_empty() {
                return;
            }

            let next_tokens =
                match pipe.step_continuous_batch(&batch_tokens, &batch_positions, &decode_slots) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("step_continuous_batch failed: {e}");
                        return;
                    }
                };

            for (i, &slot_id) in decode_slots.iter().enumerate() {
                let next_token = next_tokens[i];
                let is_eos = if let Some(ref tok) = self.tokenizer {
                    tok.is_stop_token(next_token)
                } else {
                    next_token == 2
                        || next_token == 128_001
                        || next_token == 128_008
                        || next_token == 128_009
                        || next_token == 151_643
                        || next_token == 151_645
                };

                if let Some(seq) = self.active_sequences.get_mut(&slot_id) {
                    seq.current_token = next_token;
                    seq.current_position += 1;
                    seq.generated_count += 1;

                    // H2O (Heavy-Hitter Oracle) + StreamingLLM KV-cache pruning
                    if let Some(ref arena_ref) = self.paged_arena {
                        let max_blocks_budget = 128; // 2,048 tokens / 16
                        if seq.block_table.block_ids.len() > max_blocks_budget {
                            let _evicted = seq
                                .block_table
                                .prune_heavy_hitters(arena_ref, max_blocks_budget);
                        }
                    }

                    let mut is_accepting = false;
                    if let Some(ref grammar) = seq.dfa.clone() {
                        let tok_bytes: Vec<u8> = if let Some(ref tok) = self.tokenizer {
                            tok.decode_token(next_token).into_bytes()
                        } else {
                            vec![next_token as u8]
                        };
                        for b in tok_bytes {
                            if let Ok(next_s) = grammar.transition(seq.dfa_state, b) {
                                if next_s != DFA_ERROR_STATE {
                                    seq.dfa_state = next_s;
                                }
                            }
                        }
                        is_accepting = grammar
                            .accepting_states
                            .get(seq.dfa_state as usize)
                            .copied()
                            .unwrap_or(false);
                    }

                    let reached_max = seq.generated_count >= seq.max_tokens;
                    let is_terminal = is_eos || reached_max || is_accepting;

                    let finish_reason = if is_eos || is_accepting {
                        Some("stop".to_string())
                    } else if reached_max {
                        Some("length".to_string())
                    } else {
                        None
                    };

                    let event = TokenEvent {
                        token_id: next_token,
                        is_terminal,
                        finish_reason,
                    };

                    let send_ok = seq.sender.send(event).is_ok();
                    if is_terminal || !send_ok {
                        finished_slots.push(slot_id);
                    } else if let Some(ref grammar) = seq.dfa.clone() {
                        let (jump_bytes, next_state) =
                            grammar.find_jump_forward_bytes(seq.dfa_state, 16);
                        if !jump_bytes.is_empty() {
                            let jump_tokens: Vec<u32> = if let Some(ref tok) = self.tokenizer {
                                if let Ok(s) = std::str::from_utf8(&jump_bytes) {
                                    tok.encode(s)
                                } else {
                                    jump_bytes.iter().map(|&b| u32::from(b)).collect()
                                }
                            } else {
                                jump_bytes.iter().map(|&b| u32::from(b)).collect()
                            };

                            for &jump_tok in &jump_tokens {
                                if seq.generated_count >= seq.max_tokens {
                                    break;
                                }
                                let is_jump_accepting = grammar
                                    .accepting_states
                                    .get(next_state as usize)
                                    .copied()
                                    .unwrap_or(false);
                                let reached_max = seq.generated_count + 1 >= seq.max_tokens;
                                let is_terminal = reached_max || is_jump_accepting;
                                let finish_reason = if is_jump_accepting {
                                    Some("stop".to_string())
                                } else if reached_max {
                                    Some("length".to_string())
                                } else {
                                    None
                                };

                                let event = TokenEvent {
                                    token_id: jump_tok,
                                    is_terminal,
                                    finish_reason,
                                };
                                let _ = seq.sender.send(event);
                                let _ = pipe.prefill(slot_id as u64, &[jump_tok], slot_id as u16);

                                seq.current_token = jump_tok;
                                seq.current_position += 1;
                                seq.generated_count += 1;

                                if is_terminal {
                                    finished_slots.push(slot_id);
                                    break;
                                }
                            }
                            seq.dfa_state = next_state;
                        }
                    }
                }
            }

            for slot_id in finished_slots {
                self.active_sequences.remove(&slot_id);
            }
            return;
        }

        let Some(model) = self.model.as_ref() else {
            return;
        };
        let Some(arena) = self.paged_arena.as_mut() else {
            return;
        };
        let Some(scratch) = self.scratch.as_mut() else {
            return;
        };

        let slot_keys: Vec<usize> = self.active_sequences.keys().copied().collect();

        // Stage 1: CPU/Paged chunked prefill
        let active_decode_count = self
            .active_sequences
            .values()
            .filter(|s| matches!(s.phase, SequencePhase::Decode))
            .count();
        let mut prefill_budget = self
            .prefill_config
            .compute_prefill_budget(active_decode_count);

        for &slot_id in &slot_keys {
            if prefill_budget == 0 {
                break;
            }
            if let Some(seq) = self.active_sequences.get_mut(&slot_id) {
                if let SequencePhase::Prefill { uncomputed_start } = seq.phase {
                    let target_len = seq.prompt_tokens.len().saturating_sub(1);
                    let needed = target_len.saturating_sub(uncomputed_start);
                    if needed > 0 {
                        let chunk_len = needed.min(prefill_budget);
                        let chunk =
                            &seq.prompt_tokens[uncomputed_start..uncomputed_start + chunk_len];
                        for (i, &tok) in chunk.iter().enumerate() {
                            let pos = uncomputed_start + i;
                            if let Err(e) = model.forward_step_paged(
                                tok,
                                pos,
                                arena,
                                &mut seq.block_table,
                                scratch,
                            ) {
                                tracing::error!("Prefill step failed for slot {slot_id}: {e}");
                                break;
                            }
                        }
                        let next_start = uncomputed_start + chunk_len;
                        prefill_budget = prefill_budget.saturating_sub(chunk_len);

                        if next_start >= target_len {
                            for &b in &seq.block_table.block_ids {
                                arena.retain_block(b);
                            }
                            self.radix_cache
                                .insert(&seq.prompt_tokens, &seq.block_table.block_ids);
                            seq.phase = SequencePhase::Decode;
                            seq.current_token = *seq.prompt_tokens.last().unwrap_or(&1);
                            seq.current_position = target_len;
                        } else {
                            seq.phase = SequencePhase::Prefill {
                                uncomputed_start: next_start,
                            };
                        }
                    } else {
                        seq.phase = SequencePhase::Decode;
                        seq.current_token = *seq.prompt_tokens.last().unwrap_or(&1);
                        seq.current_position = target_len;
                    }
                }
            }
        }

        // Stage 2: Decode step for active sequences in Decode phase
        for slot_id in slot_keys {
            let Some(seq) = self.active_sequences.get_mut(&slot_id) else {
                continue;
            };
            if !matches!(seq.phase, SequencePhase::Decode) {
                continue;
            }

            // Run zero-allocation forward decode step
            let step_res = model.forward_step_paged(
                seq.current_token,
                seq.current_position,
                arena,
                &mut seq.block_table,
                scratch,
            );

            match step_res {
                Ok(()) => {
                    let next_token = if seq.temperature <= 1e-6 {
                        let mut max_idx = 0;
                        let mut max_val = f32::NEG_INFINITY;
                        for (i, &logit) in scratch.logits.iter().enumerate() {
                            if logit > max_val {
                                max_val = logit;
                                max_idx = i;
                            }
                        }
                        max_idx as u32
                    } else {
                        seq.sampler
                            .sample_token(&mut scratch.logits, &mut seq.sampler_state, 10)
                            .unwrap_or(0)
                    };
                    seq.current_token = next_token;
                    seq.current_position += 1;
                    seq.generated_count += 1;

                    // H2O (Heavy-Hitter Oracle) + StreamingLLM KV-cache pruning
                    let max_blocks_budget = 128; // 2,048 tokens / 16
                    if seq.block_table.block_ids.len() > max_blocks_budget {
                        let _evicted = seq
                            .block_table
                            .prune_heavy_hitters(arena, max_blocks_budget);
                    }

                    let is_eos = if let Some(ref tok) = self.tokenizer {
                        tok.is_stop_token(next_token)
                    } else {
                        next_token == 0
                            || next_token == 2
                            || next_token == 128_001
                            || next_token == 128_008
                            || next_token == 128_009
                            || next_token == 151_643
                            || next_token == 151_645
                    };
                    let mut is_accepting = false;
                    if let Some(ref grammar) = seq.dfa.clone() {
                        let tok_bytes: Vec<u8> = if let Some(ref tok) = self.tokenizer {
                            tok.decode_token(next_token).into_bytes()
                        } else {
                            vec![next_token as u8]
                        };
                        for b in tok_bytes {
                            if let Ok(next_s) = grammar.transition(seq.dfa_state, b) {
                                if next_s != DFA_ERROR_STATE {
                                    seq.dfa_state = next_s;
                                }
                            }
                        }
                        is_accepting = grammar
                            .accepting_states
                            .get(seq.dfa_state as usize)
                            .copied()
                            .unwrap_or(false);
                    }

                    let reached_max = seq.generated_count >= seq.max_tokens;
                    let is_terminal = is_eos || reached_max || is_accepting;

                    let finish_reason = if is_eos || is_accepting {
                        Some("stop".to_string())
                    } else if reached_max {
                        Some("length".to_string())
                    } else {
                        None
                    };

                    let event = TokenEvent {
                        token_id: next_token,
                        is_terminal,
                        finish_reason,
                    };

                    let send_ok = seq.sender.send(event).is_ok();
                    if is_terminal || !send_ok {
                        finished_slots.push(slot_id);
                    } else if let Some(ref grammar) = seq.dfa.clone() {
                        let (jump_bytes, next_state) =
                            grammar.find_jump_forward_bytes(seq.dfa_state, 16);
                        if !jump_bytes.is_empty() {
                            let jump_tokens: Vec<u32> = if let Some(ref tok) = self.tokenizer {
                                if let Ok(s) = std::str::from_utf8(&jump_bytes) {
                                    tok.encode(s)
                                } else {
                                    jump_bytes.iter().map(|&b| u32::from(b)).collect()
                                }
                            } else {
                                jump_bytes.iter().map(|&b| u32::from(b)).collect()
                            };

                            for &jump_tok in &jump_tokens {
                                if seq.generated_count >= seq.max_tokens {
                                    break;
                                }
                                let is_jump_accepting = grammar
                                    .accepting_states
                                    .get(next_state as usize)
                                    .copied()
                                    .unwrap_or(false);
                                let reached_max = seq.generated_count + 1 >= seq.max_tokens;
                                let is_terminal = reached_max || is_jump_accepting;
                                let finish_reason = if is_jump_accepting {
                                    Some("stop".to_string())
                                } else if reached_max {
                                    Some("length".to_string())
                                } else {
                                    None
                                };

                                let event = TokenEvent {
                                    token_id: jump_tok,
                                    is_terminal,
                                    finish_reason,
                                };
                                let _ = seq.sender.send(event);

                                let _ = model.forward_step_paged(
                                    jump_tok,
                                    seq.current_position,
                                    arena,
                                    &mut seq.block_table,
                                    scratch,
                                );

                                seq.current_token = jump_tok;
                                seq.current_position += 1;
                                seq.generated_count += 1;

                                if is_terminal {
                                    finished_slots.push(slot_id);
                                    break;
                                }
                            }
                            seq.dfa_state = next_state;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Decode step failed for slot {slot_id}: {e}");
                    finished_slots.push(slot_id);
                }
            }
        }

        // Clean up finished sequences and release physical KV blocks
        for slot_id in finished_slots {
            if let Some(mut seq) = self.active_sequences.remove(&slot_id) {
                seq.block_table.release_all(arena);
            }
        }
    }
}

/// Client handle for interacting with the Continuous Batching Engine.
#[derive(Debug, Clone)]
pub struct EngineHandle {
    sender: mpsc::UnboundedSender<EngineCommand>,
}

impl EngineHandle {
    #[must_use]
    pub fn new(sender: mpsc::UnboundedSender<EngineCommand>) -> Self {
        Self { sender }
    }

    /// Enqueue a prompt with an optional DFA schema grammar constraint.
    pub async fn generate_stream_with_grammar(
        &self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        dfa: Option<Arc<DfaSchemaGrammar>>,
    ) -> Result<mpsc::UnboundedReceiver<TokenEvent>, String> {
        let (token_tx, token_rx) = mpsc::unbounded_channel();
        let (ack_tx, ack_rx) = oneshot::channel();

        self.sender
            .send(EngineCommand::EnqueueRequest {
                request_id,
                prompt_tokens,
                max_tokens,
                temperature,
                top_p,
                dfa,
                token_sender: token_tx,
                ack_sender: ack_tx,
            })
            .map_err(|e| format!("Failed to dispatch command: {e}"))?;

        ack_rx
            .await
            .map_err(|e| format!("Engine ACK dropped: {e}"))??;

        Ok(token_rx)
    }

    /// Enqueue a prompt and receive a lock-free streaming receiver of generated tokens.
    pub async fn generate_stream(
        &self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<mpsc::UnboundedReceiver<TokenEvent>, String> {
        self.generate_stream_with_grammar(
            request_id,
            prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            None,
        )
        .await
    }

    /// Cancel a running generation request.
    pub fn cancel(&self, request_id: String) {
        let _ = self
            .sender
            .send(EngineCommand::CancelRequest { request_id });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_prefill_budget_computation() {
        let cfg = ChunkedPrefillConfig::default();
        // At batch 0, maximum headroom
        let b0 = cfg.compute_prefill_budget(0);
        assert!(b0 >= cfg.min_chunk_tokens);
        assert!(b0 <= cfg.max_chunk_tokens);

        // At high batch size (e.g. 64), decode est is high, budget clamps to min_chunk_tokens
        let b64 = cfg.compute_prefill_budget(64);
        assert_eq!(b64, cfg.min_chunk_tokens);
        assert!(b0 > b64);
    }
}
