#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use crate::engine::OxideEngine;
use oxide_backend_cpu::CpuBackend;
use oxide_backend_cuda::CudaBackend;
use oxide_backend_hailo::HailoBackend;
use oxide_backend_intel::IntelBackend;
use oxide_backend_metal::MetalBackend;
use oxide_backend_qualcomm::QualcommBackend;
use oxide_backend_rknn::RknnBackend;
use oxide_backend_rocm::RocmBackend;
use oxide_backend_tpu::TpuBackend;
use oxide_core::error::Result;
use oxide_core::worker::{StepCommand, StepCompletion};
use oxide_models::audio::AudioServingEngine;
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::diffusion::DiffusionEngine;
use oxide_models::llama3::Llama3Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_quant::cq2::NeedleCQ2;
use oxide_quant::nvfp4::NvFp4;
use oxide_quant::ptq1_0::Ternary1_58Bit;

/// Closed-dispatch pipeline enum eliminating vtables (`dyn Trait`) from the hot loop.
/// LLVM lowers this into a flat direct jump table.
#[derive(Debug)]
pub enum SpecializedPipeline {
    // LLM - Bonsai 2 (1.58-bit Ternary)
    Bonsai2Cuda(OxideEngine<CudaBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Rocm(OxideEngine<RocmBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Tpu(OxideEngine<TpuBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Intel(OxideEngine<IntelBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Metal(OxideEngine<MetalBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Qualcomm(OxideEngine<QualcommBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Rknn(OxideEngine<RknnBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Hailo(OxideEngine<HailoBackend, TernaryBonsai2Config, Ternary1_58Bit>),

    // LLM - Needle 3 (CQ2 Sub-byte Sparse)
    Needle3Cuda(OxideEngine<CudaBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Rocm(OxideEngine<RocmBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Tpu(OxideEngine<TpuBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Intel(OxideEngine<IntelBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Metal(OxideEngine<MetalBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Qualcomm(OxideEngine<QualcommBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Rknn(OxideEngine<RknnBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Hailo(OxideEngine<HailoBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Cpu(OxideEngine<CpuBackend, CactusNeedleConfig<8>, NeedleCQ2>),

    // LLM - Llama 3 (NVFP4 / FP8)
    Llama3Cuda(OxideEngine<CudaBackend, Llama3Config, NvFp4>),
    Llama3Rocm(OxideEngine<RocmBackend, Llama3Config, NvFp4>),
    Llama3Tpu(OxideEngine<TpuBackend, Llama3Config, NvFp4>),
    Llama3Intel(OxideEngine<IntelBackend, Llama3Config, NvFp4>),
    Llama3Metal(OxideEngine<MetalBackend, Llama3Config, NvFp4>),
    Llama3Qualcomm(OxideEngine<QualcommBackend, Llama3Config, NvFp4>),
    Llama3Rknn(OxideEngine<RknnBackend, Llama3Config, NvFp4>),
    Llama3Hailo(OxideEngine<HailoBackend, Llama3Config, NvFp4>),
    Llama3Cpu(OxideEngine<CpuBackend, Llama3Config, NvFp4>),
    Llama3Dense {
        model: oxide_models::Llama3Model,
        kv_cache: std::collections::HashMap<u64, Vec<oxide_models::llama3::Llama3KvCacheLayer>>,
        seq_positions: std::collections::HashMap<u64, usize>,
        scratch: Box<oxide_models::llama3::Llama3ScratchBuffers>,
    },
    /// AMD APU Heterogeneous CPU + iGPU co-processing pipeline.
    /// Distributes transformer layers between Zen CPU AVX2 SIMD threads and RDNA integrated GPU compute units over coherent DDR5.
    Llama3AmdApuCpuIgpu {
        model: oxide_models::Llama3Model,
        kv_cache: std::collections::HashMap<u64, Vec<oxide_models::llama3::Llama3KvCacheLayer>>,
        seq_positions: std::collections::HashMap<u64, usize>,
        scratch: Box<oxide_models::llama3::Llama3ScratchBuffers>,
        topology: crate::hybrid::HybridDeviceTopology,
    },

    // Multi-Modal - Latent Diffusion, Audio Serving & Quantitative Trading
    DiffusionPipeline(DiffusionEngine),
    AudioPipeline(AudioServingEngine),
    KronosTradingPipeline(oxide_models::KronosTradingEngine),

    // Multi-Device & Heterogeneous Hybrid Pipeline (CPU+GPU, GPU+GPU+CPU)
    HybridMultiDevice(crate::hybrid::HybridMultiDevicePipeline),
}

impl SpecializedPipeline {
    /// Evaluates speculative draft tokens in a single physical GPU forward pass with Multi-Column MMVQ.
    #[inline(always)]
    pub fn step_speculative(&mut self, cur_token: u32, draft_tokens: &[u32]) -> Result<Vec<u32>> {
        match self {
            Self::Llama3Cuda(engine) => engine
                .backend
                .dispatch_speculative_verification(cur_token, draft_tokens),
            _ => Ok(Vec::new()),
        }
    }

    /// Returns current sequence position in KV cache.
    #[inline(always)]
    pub fn get_seq_pos(&self, sequence_id: u64) -> usize {
        match self {
            Self::Llama3Cuda(engine) => engine.backend.seq_pos,
            Self::Llama3Dense { seq_positions, .. }
            | Self::Llama3AmdApuCpuIgpu { seq_positions, .. } => {
                seq_positions.get(&sequence_id).copied().unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// Sets or rolls back sequence position in KV cache.
    #[inline(always)]
    pub fn set_seq_pos(&mut self, sequence_id: u64, pos: usize) {
        match self {
            Self::Llama3Cuda(engine) => {
                engine.backend.seq_pos = pos;
            }
            Self::Llama3Dense { seq_positions, .. }
            | Self::Llama3AmdApuCpuIgpu { seq_positions, .. } => {
                seq_positions.insert(sequence_id, pos);
            }
            _ => {}
        }
    }

    /// Returns model vocabulary size.
    #[inline(always)]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Llama3Cuda(engine) => engine.backend.vocab_size,
            Self::Llama3Dense { model, .. } | Self::Llama3AmdApuCpuIgpu { model, .. } => {
                model.config.vocab_size
            }
            _ => 128_256,
        }
    }

    /// High-throughput chunked prefill pass on physical GPU evaluating prompt tokens in batched chunks.
    /// Streams layer weights ONCE for all tokens in each chunk and skips the 510MB LM Head on intermediate chunks.
    pub fn prefill_prompt(&mut self, prompt_tokens: &[u32]) -> Result<u32> {
        if let Self::Llama3Cuda(engine) = self {
            let chunk_size = 512usize;
            let mut last_tok = None;
            let chunks: Vec<&[u32]> = prompt_tokens.chunks(chunk_size).collect();
            let total_chunks = chunks.len();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let is_final = idx + 1 == total_chunks;
                let res = engine.backend.dispatch_prefill_chunk(chunk, is_final)?;
                if is_final {
                    last_tok = res;
                }
            }
            last_tok.ok_or_else(|| {
                oxide_core::error::EngineError::BackendError(
                    "Prefill completed without producing sampled token".to_string(),
                )
            })
        } else {
            let mut cur = 1u32;
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                let is_prefill = i + 1 < prompt_tokens.len();
                let cmd = StepCommand::new(1, tok, 0, is_prefill);
                let res = self.step(&cmd)?;
                cur = res.sampled_token;
            }
            Ok(cur)
        }
    }

    /// Hot-loop forward step executing strictly through direct branch jump table.
    #[inline(always)]
    pub fn step(&mut self, cmd: &StepCommand) -> Result<StepCompletion> {
        match self {
            Self::Bonsai2Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Intel(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Metal(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Hailo(engine) => engine.step_monomorphized(cmd),

            Self::Needle3Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Intel(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Metal(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Hailo(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Cpu(engine) => engine.step_monomorphized(cmd),

            Self::Llama3Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Intel(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Metal(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Hailo(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Cpu(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Dense {
                model,
                kv_cache,
                seq_positions,
                scratch,
            } => {
                let pos = seq_positions.entry(cmd.sequence_id).or_insert(0);
                let seq_kv = kv_cache.entry(cmd.sequence_id).or_insert_with(|| {
                    (0..model.config.num_layers)
                        .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
                        .collect()
                });
                model.forward_step_with_scratch(cmd.input_token, *pos, seq_kv, scratch)?;
                *pos += 1;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &l) in scratch.logits.iter().enumerate() {
                    let mut score = l;
                    if cmd.input_token == i as u32 {
                        score = if score > 0.0 {
                            score / 1.15
                        } else {
                            score * 1.15
                        };
                    }
                    if score > max_val {
                        max_val = score;
                        max_idx = i;
                    }
                }
                let next_token = max_idx as u32;
                let is_terminal = if let Some(eos) = model.config.eos_token_id {
                    next_token == eos
                        || next_token == 2
                        || next_token == 106
                        || next_token == 107
                        || next_token == 128_001
                        || next_token == 128_008
                        || next_token == 128_009
                        || next_token == 151_643
                        || next_token == 151_645
                } else {
                    next_token == 2
                        || next_token == 106
                        || next_token == 107
                        || next_token == 128_001
                        || next_token == 128_008
                        || next_token == 128_009
                        || next_token == 151_643
                        || next_token == 151_645
                };
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    next_token,
                    is_terminal,
                ))
            }

            Self::Llama3AmdApuCpuIgpu {
                model,
                kv_cache,
                seq_positions,
                scratch,
                topology,
            } => {
                let pos = seq_positions.entry(cmd.sequence_id).or_insert(0);
                let seq_kv = kv_cache.entry(cmd.sequence_id).or_insert_with(|| {
                    (0..model.config.num_layers)
                        .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
                        .collect()
                });

                // 1. Token Embedding lookup on host DDR5
                let h = model.config.hidden_dim;
                let tok_idx = (cmd.input_token as usize) % model.config.vocab_size;
                if let Some(ref emb_tensor) = model.token_embd_tensor {
                    emb_tensor.extract_row(tok_idx, &mut scratch.hidden[..h]);
                } else {
                    let emb_offset = tok_idx * h;
                    if emb_offset + h <= model.token_embedding.len() {
                        scratch
                            .hidden
                            .copy_from_slice(&model.token_embedding[emb_offset..emb_offset + h]);
                    } else {
                        scratch.hidden.fill(0.0);
                    }
                }

                if model.config.is_gemma {
                    let scale = (h as f32).sqrt();
                    for x in &mut scratch.hidden[..h] {
                        *x *= scale;
                    }
                }

                // 2. Collaborative execution across topology partitions (CPU AVX2 threads + iGPU CUs)
                for partition in &topology.partitions {
                    model.forward_layers(
                        partition.start_layer,
                        partition.end_layer,
                        *pos,
                        seq_kv,
                        scratch,
                    );
                }

                // 3. Final RMSNorm
                let eps = model.config.rms_norm_eps_f32();
                let mean_sq = oxide_quant::simd::dot_f32(&scratch.hidden, &scratch.hidden)
                    / scratch.hidden.len().max(1) as f32;
                let inv_rms = 1.0 / (mean_sq + eps).sqrt();
                for (i, (&h_val, &w_val)) in scratch
                    .hidden
                    .iter()
                    .zip(model.output_norm.iter())
                    .enumerate()
                {
                    scratch.final_norm[i] = h_val * inv_rms * w_val;
                }

                // 4. LM Head projection via multithreaded SIMD matrix-vector multiplication
                model.lm_head.gemv(&scratch.final_norm, &mut scratch.logits);

                if let Some(cap) = model.config.final_logit_softcapping_f32() {
                    if cap > 0.0 {
                        for l in &mut scratch.logits {
                            *l = cap * (*l / cap).tanh();
                        }
                    }
                }

                *pos += 1;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &l) in scratch.logits.iter().enumerate() {
                    let mut score = l;
                    if cmd.input_token == i as u32 {
                        score = if score > 0.0 {
                            score / 1.15
                        } else {
                            score * 1.15
                        };
                    }
                    if score > max_val {
                        max_val = score;
                        max_idx = i;
                    }
                }
                let next_token = max_idx as u32;
                let is_terminal = if let Some(eos) = model.config.eos_token_id {
                    next_token == eos
                        || next_token == 2
                        || next_token == 106
                        || next_token == 107
                        || next_token == 128_001
                        || next_token == 128_008
                        || next_token == 128_009
                        || next_token == 151_643
                        || next_token == 151_645
                } else {
                    next_token == 2
                        || next_token == 106
                        || next_token == 107
                        || next_token == 128_001
                        || next_token == 128_008
                        || next_token == 128_009
                        || next_token == 151_643
                        || next_token == 151_645
                };
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    next_token,
                    is_terminal,
                ))
            }

            Self::DiffusionPipeline(engine) => {
                let step_idx = (cmd.input_token as usize) % engine.config().num_inference_steps;
                engine.step_denoise(step_idx)?;
                let is_terminal = step_idx + 1 >= engine.config().num_inference_steps;
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    cmd.input_token.wrapping_add(1),
                    is_terminal,
                ))
            }
            Self::AudioPipeline(engine) => {
                let pcm_chunk = [cmd.input_token as i16];
                let tokens = engine.process_streaming_asr_chunk(&pcm_chunk)?;
                let next_token = tokens.first().copied().unwrap_or(cmd.input_token + 1);
                Ok(StepCompletion::new(cmd.sequence_id, next_token, false))
            }
            Self::KronosTradingPipeline(engine) => {
                let dummy_bar = oxide_models::FinancialMarketBar {
                    timestamp_epoch_ms: 1_700_000_000_000,
                    open: 100.0,
                    high: 105.0,
                    low: 99.5,
                    close: 104.2,
                    volume: 15_000.0,
                    vwap: 102.8,
                    bid_ask_spread_bps: 1.5,
                    order_flow_imbalance: 0.35,
                };
                let bars = vec![dummy_bar; engine.context_bars.max(1)];
                let forecast = engine.evaluate_market_bars(&bars)?;
                let action_token = match forecast.signal {
                    oxide_models::TradingSignal::StrongBuy => 1,
                    oxide_models::TradingSignal::Buy => 2,
                    oxide_models::TradingSignal::Hold => 3,
                    oxide_models::TradingSignal::Sell => 4,
                    oxide_models::TradingSignal::StrongSell => 5,
                    oxide_models::TradingSignal::ClosePosition => 6,
                };
                Ok(StepCompletion::new(cmd.sequence_id, action_token, false))
            }
            Self::HybridMultiDevice(pipeline) => pipeline.step_hybrid(cmd),
        }
    }

    /// Returns a borrowed slice of the latest model logits if supported by the pipeline variant.
    pub fn logits(&self) -> Option<&[f32]> {
        match self {
            Self::Llama3Dense { scratch, .. } | Self::Llama3AmdApuCpuIgpu { scratch, .. } => {
                Some(&scratch.logits)
            }
            _ => None,
        }
    }

    /// Configures sampling parameters (temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n, seed, greedy) on the underlying backend.
    #[allow(clippy::too_many_arguments)]
    pub fn configure_sampling(
        &mut self,
        temp: f32,
        top_p: f32,
        top_k: usize,
        min_p: f32,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        greedy: bool,
    ) {
        match self {
            Self::Llama3Cuda(engine) => {
                engine.backend.configure_sampling(
                    temp,
                    top_p,
                    top_k,
                    min_p,
                    repeat_penalty,
                    repeat_last_n,
                    seed,
                    greedy,
                );
                if engine.backend.seq_pos == 0 {
                    engine.backend.warmup_gpu();
                }
            }
            Self::Needle3Cuda(engine) => {
                engine.backend.configure_sampling(
                    temp,
                    top_p,
                    top_k,
                    min_p,
                    repeat_penalty,
                    repeat_last_n,
                    seed,
                    greedy,
                );
            }
            Self::Bonsai2Cuda(engine) => {
                engine.backend.configure_sampling(
                    temp,
                    top_p,
                    top_k,
                    min_p,
                    repeat_penalty,
                    repeat_last_n,
                    seed,
                    greedy,
                );
            }
            _ => {}
        }
    }

    /// Explicitly purges KV cache layers and sequence position tracking for a completed sequence.
    pub fn evict_sequence(&mut self, sequence_id: u64) {
        match self {
            Self::Llama3Dense {
                kv_cache,
                seq_positions,
                ..
            }
            | Self::Llama3AmdApuCpuIgpu {
                kv_cache,
                seq_positions,
                ..
            } => {
                kv_cache.remove(&sequence_id);
                seq_positions.remove(&sequence_id);
            }
            _ => {}
        }
    }

    /// Prefills prompt tokens into the sequence KV cache before autoregressive token generation.
    pub fn prefill(&mut self, sequence_id: u64, tokens: &[u32], slot_idx: u16) -> Result<()> {
        for &tok in tokens {
            let cmd = StepCommand::new(sequence_id, tok, slot_idx, true);
            self.step(&cmd)?;
        }
        Ok(())
    }

    /// Universal model loader resolving from a local file path (GGUF/SafeTensors),
    /// a catalog model name (e.g. Qwen, DeepSeek, Mistral, Llama), or domain pipelines.
    pub fn from_model_or_path(
        model_query_or_path: &str,
        backend_name: &str,
        gpu_profile: Option<&str>,
        max_slots: usize,
        weights_override: Option<&str>,
    ) -> Result<Self> {
        let q_lower = model_query_or_path.to_lowercase();
        let path = std::path::Path::new(model_query_or_path);

        // 1. If explicit file path on disk (GGUF or SafeTensors)
        if path.exists() {
            let model = oxide_models::Llama3Model::from_file(path)?;
            let scratch = Box::new(model.create_scratch());

            let norm_backend = backend_name.to_ascii_lowercase().replace('-', "_");
            match norm_backend.as_str() {
                "cuda" => {
                    let mut backend = CudaBackend::new_with_profile(0, max_slots, gpu_profile);
                    let rms_eps = f32::from_bits(model.config.rms_norm_eps);
                    backend.configure_hyperparameters(
                        rms_eps,
                        model.config.is_qwen,
                        model.config.sliding_window,
                        model.config.max_seq_len.min(8192),
                    );
                    // Host-side embedding lookup via mmap avoids allocating gigabytes of VRAM for token_embd.
                    let token_embd_raw = if let Some(ref emb_tensor) = model.token_embd_tensor {
                        emb_tensor.as_raw_bytes()
                    } else if !model.token_embedding.is_empty() {
                        unsafe {
                            std::slice::from_raw_parts(
                                model.token_embedding.as_ptr().cast::<u8>(),
                                model.token_embedding.len() * std::mem::size_of::<f32>(),
                            )
                        }
                    } else {
                        &[]
                    };

                    backend.configure_model(
                        model.config.hidden_dim,
                        model.config.vocab_size,
                        model.config.num_heads,
                        model.config.num_kv_heads,
                        model.config.head_dim,
                        &model.output_norm,
                        model.lm_head.as_raw_bytes(),
                        token_embd_raw,
                        model.lm_head.quant_type(),
                    )?;
                    let theta = f32::from_bits(model.config.rope_theta);
                    if theta > 0.0 {
                        backend.rope_theta = theta;
                    }
                    if let Some(scaling) = model.config.rope_scaling_factor_f32() {
                        backend.rope_scaling_factor = scaling;
                    }
                    if let Some(orig_ctx) = model.config.rope_scaling_orig_ctx {
                        backend.rope_scaling_orig_ctx = orig_ctx;
                    }
                    if let Some(mmap) = model.mmap_handle.clone() {
                        backend.set_mmap_handle(mmap);
                    }
                    let is_gemma = model.config.is_gemma;
                    let h = model.config.hidden_dim;
                    let keep_mmap = model.mmap_handle.clone();
                    if let Some(emb_tensor) = model.token_embd_tensor.clone() {
                        let m = keep_mmap.clone();
                        backend.set_embedding_lookup(move |tok_idx, out| {
                            let _ = &m;
                            emb_tensor.extract_row(tok_idx, out);
                            if is_gemma {
                                let scale = (h as f32).sqrt();
                                for x in out.iter_mut() {
                                    *x *= scale;
                                }
                            }
                        });
                    } else if !model.token_embedding.is_empty() {
                        let emb = model.token_embedding.clone();
                        backend.set_embedding_lookup(move |tok_idx, out| {
                            let offset = tok_idx * h;
                            if offset + h <= emb.len() {
                                out.copy_from_slice(&emb[offset..offset + h]);
                            } else {
                                out.fill(0.0);
                            }
                            if is_gemma {
                                let scale = (h as f32).sqrt();
                                for x in out.iter_mut() {
                                    *x *= scale;
                                }
                            }
                        });
                    } else {
                        let lm = model.lm_head.clone();
                        backend.set_embedding_lookup(move |tok_idx, out| {
                            lm.extract_row(tok_idx, out);
                            if is_gemma {
                                let scale = (h as f32).sqrt();
                                for x in out.iter_mut() {
                                    *x *= scale;
                                }
                            }
                        });
                    }
                    backend.set_model_architecture_options(
                        is_gemma,
                        model.config.final_logit_softcapping_f32(),
                    );
                    if let Some(ref ple_tensor) = model.per_layer_token_embd {
                        let ple = ple_tensor.clone();
                        let ple_len = ple.cols();
                        backend.set_per_layer_embedding_lookup(
                            ple_len,
                            move |tok_idx, out: &mut [f32]| {
                                ple.extract_row(tok_idx, &mut out[..ple_len]);
                                let scale = (256.0f32).sqrt();
                                for x in &mut out[..ple_len] {
                                    *x *= scale;
                                }
                            },
                        );
                    }
                    for (layer_idx, layer) in model.layers.iter().enumerate() {
                        backend.add_layer_weights(
                            layer.q_proj.as_raw_bytes(),
                            layer.k_proj.as_raw_bytes(),
                            layer.v_proj.as_raw_bytes(),
                            layer.o_proj.as_raw_bytes(),
                            layer.gate_proj.as_raw_bytes(),
                            layer.up_proj.as_raw_bytes(),
                            layer.down_proj.as_raw_bytes(),
                            &layer.attn_norm,
                            &layer.ffn_norm,
                            &layer.attn_q_norm,
                            &layer.attn_k_norm,
                            &layer.post_attn_norm,
                            &layer.post_ffn_norm,
                            layer.q_proj.rows(),
                            layer.q_proj.cols(),
                            layer.k_proj.rows(),
                            layer.k_proj.cols(),
                            layer.gate_proj.rows(),
                            layer.gate_proj.cols(),
                            layer.o_proj.rows(),
                            layer.o_proj.cols(),
                            layer.layer_output_scale,
                            [
                                layer.q_proj.quant_type(),
                                layer.k_proj.quant_type(),
                                layer.v_proj.quant_type(),
                                layer.o_proj.quant_type(),
                                layer.gate_proj.quant_type(),
                                layer.up_proj.quant_type(),
                                layer.down_proj.quant_type(),
                            ],
                        )?;
                        if layer.inp_gate.is_some()
                            || layer.proj.is_some()
                            || !layer.post_norm.is_empty()
                        {
                            backend.add_layer_ple_weights(
                                layer_idx,
                                layer
                                    .inp_gate
                                    .as_ref()
                                    .map(|t| (t.as_raw_bytes(), t.quant_type())),
                                layer
                                    .proj
                                    .as_ref()
                                    .map(|t| (t.as_raw_bytes(), t.quant_type())),
                                &layer.post_norm,
                            )?;
                        }
                    }
                    if backend.layers.is_empty() {
                        return Err(oxide_core::error::EngineError::BackendError(
                            "Failed to allocate CUDA layer weights in device VRAM".to_string(),
                        ));
                    }
                    return Ok(Self::Llama3Cuda(OxideEngine::new(backend, model.config)));
                }
                "rocm" => {
                    let backend = RocmBackend::new(0, max_slots);
                    return Ok(Self::Llama3Rocm(OxideEngine::new(backend, model.config)));
                }
                "tpu" => {
                    let backend = TpuBackend::new(0, max_slots);
                    return Ok(Self::Llama3Tpu(OxideEngine::new(backend, model.config)));
                }
                "intel" => {
                    let backend = IntelBackend::new(0, max_slots);
                    return Ok(Self::Llama3Intel(OxideEngine::new(backend, model.config)));
                }
                "metal" => {
                    let backend = MetalBackend::new(0, max_slots);
                    return Ok(Self::Llama3Metal(OxideEngine::new(backend, model.config)));
                }
                "qualcomm" | "snapdragon" => {
                    let backend = QualcommBackend::new(0, max_slots);
                    return Ok(Self::Llama3Qualcomm(OxideEngine::new(
                        backend,
                        model.config,
                    )));
                }
                "rknn" | "rockchip" => {
                    let backend = RknnBackend::new(0, max_slots);
                    return Ok(Self::Llama3Rknn(OxideEngine::new(backend, model.config)));
                }
                "hailo" => {
                    let backend = HailoBackend::new(0, max_slots);
                    return Ok(Self::Llama3Hailo(OxideEngine::new(backend, model.config)));
                }
                _ => {}
            }

            let is_apu = backend_name.eq_ignore_ascii_case("apu");
            let is_cpu_igpu = backend_name.eq_ignore_ascii_case("cpu_igpu");

            if is_apu || is_cpu_igpu {
                let has_npu = is_apu
                    || oxide_core::hardware::GpuDeviceProfile::detect_amd_apu_full()
                        .is_some_and(|(_, _, npu)| npu);
                let topology = crate::hybrid::HybridDeviceTopology::amd_apu_full_partition(
                    model.config.num_layers,
                    has_npu,
                );
                return Ok(Self::Llama3AmdApuCpuIgpu {
                    model,
                    kv_cache: std::collections::HashMap::new(),
                    seq_positions: std::collections::HashMap::new(),
                    scratch,
                    topology,
                });
            }

            return Ok(Self::Llama3Dense {
                model,
                kv_cache: std::collections::HashMap::new(),
                seq_positions: std::collections::HashMap::new(),
                scratch,
            });
        }

        // 2. Multi-modal / domain engine routes
        if q_lower.contains("diffusion") || q_lower.contains("flux") {
            let cfg = oxide_models::diffusion::DiffusionTransformerConfig::default();
            return Ok(Self::DiffusionPipeline(DiffusionEngine::new(cfg)));
        }
        if q_lower.contains("audio-tts") || q_lower.contains("kokoro") {
            let cfg = oxide_models::audio::AudioModelConfig::new_tts_config(24000);
            return Ok(Self::AudioPipeline(AudioServingEngine::new(cfg)));
        }
        if q_lower.contains("audio-asr") || q_lower.contains("whisper") {
            let cfg = oxide_models::audio::AudioModelConfig::default();
            return Ok(Self::AudioPipeline(AudioServingEngine::new(cfg)));
        }
        if q_lower.contains("kronos") || q_lower.contains("trading") {
            let engine = oxide_models::KronosTradingEngine::new(2048, 60, 10, 1.0);
            return Ok(Self::KronosTradingPipeline(engine));
        }
        if q_lower.contains("bonsai") {
            let backend = CudaBackend::new_with_profile(0, max_slots, gpu_profile);
            let config = TernaryBonsai2Config::default();
            return Ok(Self::Bonsai2Cuda(OxideEngine::new(backend, config)));
        }
        if q_lower.contains("needle") {
            let backend = CpuBackend::new(0, max_slots);
            let config = CactusNeedleConfig::<8>::default();
            return Ok(Self::Needle3Cpu(OxideEngine::new(backend, config)));
        }

        // 3. Universal LLM: Load from weights override or catalog lookup
        let model = if let Some(weights_path) = weights_override {
            let w_path = std::path::Path::new(weights_path);
            if w_path.exists() {
                oxide_models::Llama3Model::from_file(w_path)?
            } else {
                oxide_models::Llama3Model::from_model_name_or_path(model_query_or_path)?
            }
        } else {
            oxide_models::Llama3Model::from_model_name_or_path(model_query_or_path)?
        };

        match backend_name.to_ascii_lowercase().as_str() {
            "cuda" => {
                let mut backend = CudaBackend::new_with_profile(0, max_slots, gpu_profile);
                let rms_eps = f32::from_bits(model.config.rms_norm_eps);
                backend.configure_hyperparameters(
                    rms_eps,
                    model.config.is_qwen,
                    model.config.sliding_window,
                    model.config.max_seq_len.min(8192),
                );
                let token_embd_raw = if let Some(ref emb_tensor) = model.token_embd_tensor {
                    emb_tensor.as_raw_bytes()
                } else if !model.token_embedding.is_empty() {
                    unsafe {
                        std::slice::from_raw_parts(
                            model.token_embedding.as_ptr().cast::<u8>(),
                            model.token_embedding.len() * std::mem::size_of::<f32>(),
                        )
                    }
                } else {
                    &[]
                };

                let _ = backend.configure_model(
                    model.config.hidden_dim,
                    model.config.vocab_size,
                    model.config.num_heads,
                    model.config.num_kv_heads,
                    model.config.head_dim,
                    &model.output_norm,
                    model.lm_head.as_raw_bytes(),
                    token_embd_raw,
                    model.lm_head.quant_type(),
                );
                let theta = f32::from_bits(model.config.rope_theta);
                if theta > 0.0 {
                    backend.rope_theta = theta;
                }
                if let Some(scaling) = model.config.rope_scaling_factor_f32() {
                    backend.rope_scaling_factor = scaling;
                }
                if let Some(orig_ctx) = model.config.rope_scaling_orig_ctx {
                    backend.rope_scaling_orig_ctx = orig_ctx;
                }
                if let Some(mmap) = model.mmap_handle.clone() {
                    backend.set_mmap_handle(mmap);
                }
                let is_gemma = model.config.is_gemma;
                let h = model.config.hidden_dim;
                let keep_mmap = model.mmap_handle.clone();
                if let Some(emb_tensor) = model.token_embd_tensor.clone() {
                    let m = keep_mmap.clone();
                    backend.set_embedding_lookup(move |tok_idx, out| {
                        let _ = &m;
                        emb_tensor.extract_row(tok_idx, out);
                        if is_gemma {
                            let scale = (h as f32).sqrt();
                            for x in out.iter_mut() {
                                *x *= scale;
                            }
                        }
                    });
                } else if !model.token_embedding.is_empty() {
                    let emb = model.token_embedding.clone();
                    backend.set_embedding_lookup(move |tok_idx, out| {
                        let offset = tok_idx * h;
                        if offset + h <= emb.len() {
                            out.copy_from_slice(&emb[offset..offset + h]);
                        } else {
                            out.fill(0.0);
                        }
                        if is_gemma {
                            let scale = (h as f32).sqrt();
                            for x in out.iter_mut() {
                                *x *= scale;
                            }
                        }
                    });
                } else {
                    let is_gemma = model.config.is_gemma;
                    let h = model.config.hidden_dim;
                    let lm = model.lm_head.clone();
                    backend.set_embedding_lookup(move |tok_idx, out| {
                        lm.extract_row(tok_idx, out);
                        if is_gemma {
                            let scale = (h as f32).sqrt();
                            for x in out.iter_mut() {
                                *x *= scale;
                            }
                        }
                    });
                }
                backend.set_model_architecture_options(
                    model.config.is_gemma,
                    model.config.final_logit_softcapping_f32(),
                );
                if let Some(ref ple_tensor) = model.per_layer_token_embd {
                    let ple = ple_tensor.clone();
                    let ple_len = ple.cols();
                    backend.set_per_layer_embedding_lookup(
                        ple_len,
                        move |tok_idx, out: &mut [f32]| {
                            ple.extract_row(tok_idx, &mut out[..ple_len]);
                            let scale = (256.0f32).sqrt();
                            for x in &mut out[..ple_len] {
                                *x *= scale;
                            }
                        },
                    );
                }
                for (layer_idx, layer) in model.layers.iter().enumerate() {
                    let _ = backend.add_layer_weights(
                        layer.q_proj.as_raw_bytes(),
                        layer.k_proj.as_raw_bytes(),
                        layer.v_proj.as_raw_bytes(),
                        layer.o_proj.as_raw_bytes(),
                        layer.gate_proj.as_raw_bytes(),
                        layer.up_proj.as_raw_bytes(),
                        layer.down_proj.as_raw_bytes(),
                        &layer.attn_norm,
                        &layer.ffn_norm,
                        &layer.attn_q_norm,
                        &layer.attn_k_norm,
                        &layer.post_attn_norm,
                        &layer.post_ffn_norm,
                        layer.q_proj.rows(),
                        layer.q_proj.cols(),
                        layer.k_proj.rows(),
                        layer.k_proj.cols(),
                        layer.gate_proj.rows(),
                        layer.gate_proj.cols(),
                        layer.o_proj.rows(),
                        layer.o_proj.cols(),
                        layer.layer_output_scale,
                        [
                            layer.q_proj.quant_type(),
                            layer.k_proj.quant_type(),
                            layer.v_proj.quant_type(),
                            layer.o_proj.quant_type(),
                            layer.gate_proj.quant_type(),
                            layer.up_proj.quant_type(),
                            layer.down_proj.quant_type(),
                        ],
                    );
                    if layer.inp_gate.is_some()
                        || layer.proj.is_some()
                        || !layer.post_norm.is_empty()
                    {
                        let _ = backend.add_layer_ple_weights(
                            layer_idx,
                            layer
                                .inp_gate
                                .as_ref()
                                .map(|t| (t.as_raw_bytes(), t.quant_type())),
                            layer
                                .proj
                                .as_ref()
                                .map(|t| (t.as_raw_bytes(), t.quant_type())),
                            &layer.post_norm,
                        );
                    }
                }
                return Ok(Self::Llama3Cuda(OxideEngine::new(backend, model.config)));
            }
            "rocm" => {
                let backend = RocmBackend::new(0, max_slots);
                return Ok(Self::Llama3Rocm(OxideEngine::new(backend, model.config)));
            }
            "tpu" => {
                let backend = TpuBackend::new(0, max_slots);
                return Ok(Self::Llama3Tpu(OxideEngine::new(backend, model.config)));
            }
            "intel" => {
                let backend = IntelBackend::new(0, max_slots);
                return Ok(Self::Llama3Intel(OxideEngine::new(backend, model.config)));
            }
            "metal" => {
                let backend = MetalBackend::new(0, max_slots);
                return Ok(Self::Llama3Metal(OxideEngine::new(backend, model.config)));
            }
            "qualcomm" | "snapdragon" => {
                let backend = QualcommBackend::new(0, max_slots);
                return Ok(Self::Llama3Qualcomm(OxideEngine::new(
                    backend,
                    model.config,
                )));
            }
            "rknn" | "rockchip" => {
                let backend = RknnBackend::new(0, max_slots);
                return Ok(Self::Llama3Rknn(OxideEngine::new(backend, model.config)));
            }
            "hailo" => {
                let backend = HailoBackend::new(0, max_slots);
                return Ok(Self::Llama3Hailo(OxideEngine::new(backend, model.config)));
            }
            _ => {}
        }

        let scratch = Box::new(model.create_scratch());

        let norm_backend = backend_name.to_ascii_lowercase().replace('-', "_");

        let maybe_topology = match norm_backend.as_str() {
            "cpu_nvidia" | "cpu+nvidia" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_nvidia_partition(
                    model.config.num_layers,
                    1,
                    (model.config.num_layers * 2) / 10,
                ))
            }
            "cpu_amd" | "cpu+amd" => Some(crate::hybrid::HybridDeviceTopology::cpu_amd_partition(
                model.config.num_layers,
                1,
                (model.config.num_layers * 2) / 10,
            )),
            "cpu_intel" | "cpu+intel" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_intel_partition(
                    model.config.num_layers,
                    1,
                    (model.config.num_layers * 2) / 10,
                ))
            }
            "cpu_tpu" | "cpu+tpu" => Some(crate::hybrid::HybridDeviceTopology::cpu_tpu_partition(
                model.config.num_layers,
                1,
                (model.config.num_layers * 2) / 10,
            )),
            "cpu_npu" | "cpu+npu" => Some(crate::hybrid::HybridDeviceTopology::cpu_npu_partition(
                model.config.num_layers,
                0.65,
            )),
            "cpu_nvidia_amd_intel" | "cpu+nvidia+amd+intel" | "triple_gpu" | "hybrid" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_amd_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_amd_intel" | "cpu+amd+intel" | "amd_intel" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_amd_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_nvidia_intel" | "cpu+nvidia+intel" | "nvidia_intel" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_nvidia_amd" | "cpu+nvidia+amd" | "nvidia_amd" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_amd_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_igpu_npu" | "cpu+igpu+npu" | "apu" | "cpu_igpu" => {
                let has_npu = norm_backend == "apu"
                    || norm_backend.contains("npu")
                    || oxide_core::hardware::GpuDeviceProfile::detect_amd_apu_full()
                        .is_some_and(|(_, _, npu)| npu);
                Some(crate::hybrid::HybridDeviceTopology::amd_apu_full_partition(
                    model.config.num_layers,
                    has_npu,
                ))
            }
            "cpu_igpu_tpu" | "cpu+igpu+tpu" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_igpu_tpu_partition(
                    model.config.num_layers,
                    1,
                ))
            }
            "cpu_igpu_npu_nvidia" | "cpu+igpu+npu+nvidia" | "apu_nvidia" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_igpu_npu_nvidia_partition(
                    model.config.num_layers,
                    1,
                ),
            ),
            "cpu_igpu_npu_amd" | "cpu+igpu+npu+amd" | "apu_amd" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_igpu_npu_amd_partition(
                    model.config.num_layers,
                    1,
                ),
            ),
            "arm_npu_hat" | "arm+npu+hat" | "arm_npu" | "arm+npu" => Some(
                crate::hybrid::HybridDeviceTopology::arm_npu_external_hat_partition(
                    model.config.num_layers,
                ),
            ),
            "arm_integrated_npu" | "arm_npu_only" => Some(
                crate::hybrid::HybridDeviceTopology::arm_integrated_npu_partition(
                    model.config.num_layers,
                ),
            ),
            "epyc" | "epyc_server" => Some(
                crate::hybrid::HybridDeviceTopology::epyc_server_standalone_partition(
                    model.config.num_layers,
                    2,
                ),
            ),
            "epyc_gpu" | "epyc+gpu" => Some(
                crate::hybrid::HybridDeviceTopology::epyc_server_gpu_partition(
                    model.config.num_layers,
                    2,
                    &[crate::hybrid::DeviceRole::NvidiaGpu(0)],
                ),
            ),
            "apple" | "apple_silicon" => Some(
                crate::hybrid::HybridDeviceTopology::apple_silicon_uma_partition(
                    model.config.num_layers,
                ),
            ),
            "snapdragon" | "qualcomm" => Some(
                crate::hybrid::HybridDeviceTopology::qualcomm_snapdragon_partition(
                    model.config.num_layers,
                ),
            ),
            "intel_ultra" => Some(
                crate::hybrid::HybridDeviceTopology::intel_core_ultra_partition(
                    model.config.num_layers,
                    0,
                ),
            ),
            "rockchip" | "rknn" => Some(
                crate::hybrid::HybridDeviceTopology::rockchip_rknn_partition(
                    model.config.num_layers,
                ),
            ),
            "rpi_hailo" | "hailo" => Some(
                crate::hybrid::HybridDeviceTopology::raspberry_pi_hailo_partition(
                    model.config.num_layers,
                ),
            ),
            _ => None,
        };

        if let Some(topology) = maybe_topology {
            return Ok(Self::Llama3AmdApuCpuIgpu {
                model,
                kv_cache: std::collections::HashMap::new(),
                seq_positions: std::collections::HashMap::new(),
                scratch,
                topology,
            });
        }

        Ok(Self::Llama3Dense {
            model,
            kv_cache: std::collections::HashMap::new(),
            seq_positions: std::collections::HashMap::new(),
            scratch,
        })
    }
}
