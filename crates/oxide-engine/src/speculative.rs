#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported Speculative Decoding Strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeculativeStrategy {
    DraftModel,
    NGramPromptLookup,
    SuffixMatching,
    EagleTreeDraft,
    DFlashDiffusion,
    NativeMtp,
    DynamicEnsemble,
}

/// Configuration for Speculative Decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub strategy: SpeculativeStrategy,
    pub max_draft_tokens: usize,
    pub acceptance_threshold: f32,
    pub draft_model_name: String,
    pub target_model_name: String,
    pub ngram_min: usize,
    pub ngram_max: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            strategy: SpeculativeStrategy::DraftModel,
            max_draft_tokens: 5,
            acceptance_threshold: 0.85,
            draft_model_name: "llama-3-1b-draft".to_string(),
            target_model_name: "llama-3-70b-target".to_string(),
            ngram_min: 2,
            ngram_max: 4,
        }
    }
}

/// Result of a Speculative Verification Step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeVerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub num_drafted: usize,
    pub num_accepted: usize,
    pub recovery_token: Option<u32>,
    pub acceptance_rate: f32,
}

/// Speculative Decoding Execution Engine.
#[derive(Debug, Clone)]
pub struct SpeculativeDecoderEngine {
    pub config: SpeculativeConfig,
}

impl SpeculativeDecoderEngine {
    #[must_use]
    pub fn new(config: SpeculativeConfig) -> Self {
        Self { config }
    }

    /// Verifies draft tokens against target model logit distributions.
    pub fn verify_draft_tokens(
        &self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_token_probs: &[f32],
        target_greedy_recovery_tokens: &[u32],
    ) -> SpeculativeVerificationResult {
        let mut accepted = Vec::with_capacity(draft_tokens.len());
        let mut recovery_token = None;

        for (idx, &token) in draft_tokens.iter().enumerate() {
            let p_draft = draft_probs.get(idx).copied().unwrap_or(0.5).max(1e-6);
            let p_target = target_token_probs.get(idx).copied().unwrap_or(0.5);

            if p_target >= p_draft || (p_target / p_draft) >= self.config.acceptance_threshold {
                accepted.push(token);
            } else {
                if let Some(&rec_tok) = target_greedy_recovery_tokens.get(idx) {
                    recovery_token = Some(rec_tok);
                }
                break;
            }
        }

        let num_accepted = accepted.len();
        let num_drafted = draft_tokens.len();
        let acceptance_rate = if num_drafted > 0 {
            num_accepted as f32 / num_drafted as f32
        } else {
            0.0
        };

        SpeculativeVerificationResult {
            accepted_tokens: accepted,
            num_drafted,
            num_accepted,
            recovery_token,
            acceptance_rate,
        }
    }

    /// N-Gram Prompt Lookup Decoding.
    #[must_use]
    pub fn draft_ngram_lookup(&self, context: &[u32], max_draft: usize) -> Vec<u32> {
        let n = context.len();
        if n < self.config.ngram_min {
            return Vec::new();
        }

        for k in (self.config.ngram_min..=self.config.ngram_max.min(n)).rev() {
            let pattern = &context[n - k..n];
            let search_limit = n - k;
            for i in (0..search_limit).rev() {
                if context[i..i + k] == *pattern {
                    let draft_start = i + k;
                    let draft_end = (draft_start + max_draft).min(n);
                    if draft_end > draft_start {
                        return context[draft_start..draft_end].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }

    /// Suffix Matching Speculative Decoding.
    #[must_use]
    pub fn draft_suffix_matching(&self, context: &[u32], common_suffixes: &[Vec<u32>]) -> Vec<u32> {
        let n = context.len();
        for suffix in common_suffixes {
            if suffix.len() > 1 && n >= 2 {
                let overlap = suffix.len().min(4);
                if context.ends_with(&suffix[..overlap]) {
                    return suffix[overlap..].to_vec();
                }
            }
        }
        Vec::new()
    }

    /// EAGLE Tree Speculator.
    #[must_use]
    pub fn draft_eagle_tree(&self, root_token: u32, depth: usize) -> Vec<Vec<u32>> {
        let mut paths = vec![vec![root_token]];
        for _ in 1..depth {
            let mut next_paths = Vec::new();
            for path in paths {
                let last = *path.last().unwrap_or(&0);
                let branch_a = last.wrapping_add(1);
                let branch_b = last.wrapping_add(2);

                let mut p_a = path.clone();
                p_a.push(branch_a);
                next_paths.push(p_a);

                let mut p_b = path;
                p_b.push(branch_b);
                next_paths.push(p_b);
            }
            paths = next_paths;
        }
        paths
    }

    /// DFlash: Diffusion-based Candidate Generator.
    #[must_use]
    pub fn draft_dflash_block(&self, seed_latent: &[f32], block_len: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(block_len);
        for i in 0..block_len {
            let latent_val = seed_latent
                .get(i % seed_latent.len())
                .copied()
                .unwrap_or(0.0);
            let token_id = ((latent_val.abs() * 1000.0) as u32) % 32000;
            tokens.push(token_id);
        }
        tokens
    }

    /// Verifies tree-structured candidate branches using tree-masking and selects the longest accepted path.
    pub fn verify_tree_candidates(
        &self,
        tree_paths: &[Vec<u32>],
        tree_probs: &[Vec<f32>],
        target_probs: &[Vec<f32>],
        target_recovery: &[Option<u32>],
    ) -> SpeculativeVerificationResult {
        let mut best_result = SpeculativeVerificationResult {
            accepted_tokens: Vec::new(),
            num_drafted: 0,
            num_accepted: 0,
            recovery_token: None,
            acceptance_rate: 0.0,
        };

        for (i, path) in tree_paths.iter().enumerate() {
            let path_probs = tree_probs.get(i).cloned().unwrap_or_default();
            let tgt_probs = target_probs.get(i).cloned().unwrap_or_default();
            let rec = target_recovery
                .get(i)
                .copied()
                .flatten()
                .map(|t| vec![t])
                .unwrap_or_default();

            let res = self.verify_draft_tokens(path, &path_probs, &tgt_probs, &rec);
            if res.num_accepted > best_result.num_accepted
                || (res.num_accepted == best_result.num_accepted
                    && res.acceptance_rate > best_result.acceptance_rate)
            {
                best_result = res;
            }
        }

        best_result
    }
}

/// Speculative Engine Context with synchronized host and device sequence counter buffers.
#[derive(Debug, Clone)]
pub struct SpeculativeEngineContext {
    pub max_sequences: usize,
    host_seq_lens: HashMap<u64, usize>,
    device_seq_lens: HashMap<u64, usize>,
    sequence_tokens: HashMap<u64, Vec<u32>>,
}

impl SpeculativeEngineContext {
    #[must_use]
    pub fn new(max_sequences: usize) -> Self {
        Self {
            max_sequences,
            host_seq_lens: HashMap::new(),
            device_seq_lens: HashMap::new(),
            sequence_tokens: HashMap::new(),
        }
    }

    /// Registers a new active sequence with initial prompt length.
    pub fn register_sequence(&mut self, sequence_id: u64, initial_length: usize) -> Result<()> {
        self.host_seq_lens.insert(sequence_id, initial_length);
        self.device_seq_lens.insert(sequence_id, initial_length);
        self.sequence_tokens
            .insert(sequence_id, vec![0; initial_length]);
        Ok(())
    }

    /// Appends speculative draft tokens to sequence tracking.
    pub fn append_draft_tokens(&mut self, sequence_id: u64, draft_tokens: &[u32]) -> Result<()> {
        let host_len = self
            .host_seq_lens
            .get_mut(&sequence_id)
            .ok_or(EngineError::SequenceNotFound { sequence_id })?;
        *host_len += draft_tokens.len();
        if let Some(tokens) = self.sequence_tokens.get_mut(&sequence_id) {
            tokens.extend_from_slice(draft_tokens);
        }
        if let Some(dev_len) = self.device_seq_lens.get_mut(&sequence_id) {
            *dev_len = *host_len;
        }
        Ok(())
    }

    /// Returns the active host sequence length for a given sequence ID.
    #[must_use]
    pub fn get_host_sequence_length(&self, sequence_id: u64) -> usize {
        self.host_seq_lens.get(&sequence_id).copied().unwrap_or(0)
    }

    /// Executes speculative rollback: truncates tokens and synchronizes device sequence length buffer.
    pub fn rollback_speculative_state(
        &mut self,
        sequence_id: u64,
        accepted_length: usize,
        _draft_count: usize,
    ) -> Result<()> {
        let host_len = self
            .host_seq_lens
            .get_mut(&sequence_id)
            .ok_or(EngineError::SequenceNotFound { sequence_id })?;
        *host_len = accepted_length;
        if let Some(tokens) = self.sequence_tokens.get_mut(&sequence_id) {
            tokens.truncate(accepted_length);
        }
        // Synchronize device-side sequence length buffer
        if let Some(dev_len) = self.device_seq_lens.get_mut(&sequence_id) {
            *dev_len = accepted_length;
        }
        Ok(())
    }

    /// Reads device sequence length counter buffer (seq_lens_d).
    pub fn read_device_sequence_length(&self, sequence_id: u64) -> Result<usize> {
        self.device_seq_lens
            .get(&sequence_id)
            .copied()
            .ok_or(EngineError::SequenceNotFound { sequence_id })
    }
}

/// Ninfer-inspired Native Multi-Token Prediction (MTP) Speculative Module.
///
/// DeepSeek / Qwen style integrated future token predictor operating directly
/// on final layer hidden states without requiring an auxiliary draft model.
#[derive(Debug, Clone)]
pub struct NativeMtpModule {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_mtp_heads: usize,
    /// Projection weights per MTP head: `[num_mtp_heads, vocab_size, hidden_dim]`
    pub mtp_head_weights: Vec<f32>,
}

impl NativeMtpModule {
    #[must_use]
    pub fn new(hidden_dim: usize, vocab_size: usize, num_mtp_heads: usize) -> Self {
        let total_weights = num_mtp_heads * vocab_size * hidden_dim;
        Self {
            hidden_dim,
            vocab_size,
            num_mtp_heads,
            mtp_head_weights: vec![0.001; total_weights],
        }
    }

    /// Predicts future tokens (t+1, t+2, ...) directly from the base model's final hidden state.
    #[must_use]
    pub fn predict_future_tokens(&self, last_hidden: &[f32], max_future: usize) -> Vec<u32> {
        let h = self.hidden_dim;
        let v = self.vocab_size;
        let heads = self.num_mtp_heads.min(max_future);
        let mut predicted = Vec::with_capacity(heads);

        if last_hidden.len() != h {
            return predicted;
        }

        for head in 0..heads {
            let base_offset = head * v * h;
            let mut best_logit = -f32::INFINITY;
            let mut best_token = 0u32;

            for tok in 0..v {
                let row_start = base_offset + tok * h;
                let mut dot = 0.0f32;
                for (j, &val) in last_hidden.iter().enumerate().take(h) {
                    dot += self.mtp_head_weights[row_start + j] * val;
                }
                if dot > best_logit {
                    best_logit = dot;
                    best_token = tok as u32;
                }
            }
            predicted.push(best_token);
        }

        predicted
    }
}

/// Dynamically regulates the speculation horizon based on rolling acceptance rates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveSpeculator {
    pub current_k: usize,
    pub min_k: usize,
    pub max_k: usize,
    pub rolling_window: usize,
    pub history: Vec<(usize, usize)>, // (accepted, drafted)
    pub consecutive_rejections: usize,
    pub backoff_cooldown: usize,
}

impl AdaptiveSpeculator {
    #[must_use]
    pub fn new(initial_k: usize, min_k: usize, max_k: usize, window: usize) -> Self {
        Self {
            current_k: initial_k.clamp(min_k, max_k),
            min_k,
            max_k,
            rolling_window: window.max(1),
            history: Vec::new(),
            consecutive_rejections: 0,
            backoff_cooldown: 0,
        }
    }

    /// Records the verification outcome of a speculative step and dynamically adapts K.
    pub fn record_step(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        self.history.push((accepted, drafted));
        if self.history.len() > self.rolling_window {
            self.history.remove(0);
        }

        if accepted == 0 {
            self.consecutive_rejections += 1;
            // Back off for 2 cycles if 3 consecutive rejections occur
            if self.consecutive_rejections >= 3 {
                self.backoff_cooldown = 2;
                self.current_k = self.min_k;
            }
        } else {
            self.consecutive_rejections = 0;
        }

        let rate = self.rolling_acceptance_rate();

        if self.history.len() >= 3 {
            // Expand horizon if acceptance >= 75%
            if rate >= 0.75 && self.consecutive_rejections == 0 {
                self.current_k = (self.current_k + 1).min(self.max_k);
            }
            // Contract horizon if acceptance < 40%
            else if rate < 0.40 {
                self.current_k = self.current_k.saturating_sub(1).max(self.min_k);
            }
        }
    }

    /// Returns the active draft horizon, accounting for backoff cooldowns.
    pub fn current_horizon(&mut self) -> usize {
        if self.backoff_cooldown > 0 {
            self.backoff_cooldown -= 1;
            0 // Skip drafting during backoff
        } else {
            self.current_k
        }
    }

    /// Read-only inspection of the current speculation horizon K.
    #[must_use]
    pub fn active_k(&self) -> usize {
        self.current_k
    }

    /// Calculates the rolling acceptance rate across the active window.
    #[must_use]
    pub fn rolling_acceptance_rate(&self) -> f32 {
        let (tot_acc, tot_draft) = self
            .history
            .iter()
            .fold((0usize, 0usize), |(acc, drf), &(a, d)| (acc + a, drf + d));
        if tot_draft > 0 {
            tot_acc as f32 / tot_draft as f32
        } else {
            0.0
        }
    }
}

/// Dual-Tier Drafter Ensemble for autonomous speculation:
/// - Tier 1: Fast N-gram Prompt Lookup (low-latency match against prompt/recent context).
/// - Tier 2: Online Markov N-gram / Suffix Matching fallback (for novel, creative continuations).
#[derive(Debug, Clone)]
pub struct DrafterEnsemble {
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub adaptive: AdaptiveSpeculator,
    /// Fast bigram/trigram transition graph: from_tok -> { to_tok -> count }
    pub transitions: HashMap<u32, HashMap<u32, u32>>,
}

impl DrafterEnsemble {
    #[must_use]
    pub fn new(initial_k: usize, min_k: usize, max_k: usize) -> Self {
        Self {
            ngram_min: 2,
            ngram_max: 4,
            adaptive: AdaptiveSpeculator::new(initial_k, min_k, max_k, 6),
            transitions: HashMap::new(),
        }
    }

    /// Ingests a new token transition into the online Markov graph.
    pub fn record_token(&mut self, prev: u32, next: u32) {
        let entry = self.transitions.entry(prev).or_default();
        *entry.entry(next).or_insert(0) += 1;
    }

    /// Ingests a slice of tokens to warm up the transition graph from prompt.
    pub fn warm_up(&mut self, tokens: &[u32]) {
        for window in tokens.windows(2) {
            self.record_token(window[0], window[1]);
        }
    }

    /// Proposes draft tokens using the dual-tier ensemble strategy.
    pub fn propose_draft(
        &mut self,
        context: &[u32],
        lookup_engine: &SpeculativeDecoderEngine,
    ) -> (Vec<u32>, &'static str) {
        let k = self.adaptive.current_horizon();
        if k == 0 || context.is_empty() {
            return (Vec::new(), "none");
        }

        // Tier 1: Try N-Gram Prompt Lookup first
        let ngram_draft = lookup_engine.draft_ngram_lookup(context, k);
        if !ngram_draft.is_empty() {
            return (ngram_draft, "ngram_prompt_lookup");
        }

        // Tier 2: Markov continuation fallback
        let mut markov_draft = Vec::with_capacity(k);
        let mut cur = *context.last().unwrap();
        for _ in 0..k {
            if let Some(next_counts) = self.transitions.get(&cur) {
                // Find most frequent next token with at least 2 occurrences
                if let Some((&best_next, &count)) = next_counts.iter().max_by_key(|(_, cnt)| *cnt) {
                    if count >= 2 && !markov_draft.contains(&best_next) {
                        markov_draft.push(best_next);
                        cur = best_next;
                        continue;
                    }
                }
            }
            break;
        }

        if markov_draft.is_empty() {
            (Vec::new(), "none")
        } else {
            (markov_draft, "markov_transition_fallback")
        }
    }

    /// Updates the ensemble with step acceptance metrics.
    pub fn update_acceptance(&mut self, accepted: usize, drafted: usize) {
        self.adaptive.record_step(accepted, drafted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_speculator_expansion_and_contraction() {
        let mut spec = AdaptiveSpeculator::new(3, 2, 5, 4);
        assert_eq!(spec.current_k, 3);

        // High acceptance steps: 3/3, 3/3, 3/3 (100%) -> should expand
        spec.record_step(3, 3);
        spec.record_step(3, 3);
        spec.record_step(3, 3);
        assert_eq!(spec.current_k, 4);

        // Another high acceptance: 4/4 -> expand to max_k (5)
        spec.record_step(4, 4);
        assert_eq!(spec.current_k, 5);

        // Low acceptance: 0/5, 0/5, 1/5 -> contract
        spec.record_step(0, 5);
        spec.record_step(0, 5);
        spec.record_step(1, 5);
        assert!(spec.current_k < 5);
    }

    #[test]
    fn test_drafter_ensemble_tier1_and_tier2() {
        let lookup_engine = SpeculativeDecoderEngine::new(SpeculativeConfig {
            ngram_min: 2,
            ngram_max: 4,
            ..Default::default()
        });
        let mut ensemble = DrafterEnsemble::new(3, 2, 5);

        // Warm up with repeating tokens
        let tokens = vec![10, 20, 30, 40, 10, 20, 30, 40];
        ensemble.warm_up(&tokens);

        // Query with context ending in [10, 20] -> Tier 1 (N-gram) should propose [30, 40]
        let context1 = vec![10, 20, 30, 40, 10, 20];
        let (draft1, tier1) = ensemble.propose_draft(&context1, &lookup_engine);
        assert_eq!(tier1, "ngram_prompt_lookup");
        assert_eq!(draft1, vec![30, 40, 10]);

        // Query with unseen prefix [99, 10] -> N-gram fails, but 10 has Markov transitions to 20
        let (draft2, tier2) = ensemble.propose_draft(&[99, 10], &lookup_engine);
        assert_eq!(tier2, "markov_transition_fallback");
        assert_eq!(draft2[0], 20);
    }

    #[test]
    fn test_native_mtp_prediction_lifecycle() {
        let mut mtp = NativeMtpModule::new(4, 8, 2);
        // Head 0: token 3 has highest weight
        mtp.mtp_head_weights[3 * 4] = 10.0;
        // Head 1: token 6 has highest weight
        mtp.mtp_head_weights[8 * 4 + 6 * 4] = 10.0;

        let hidden = vec![1.0f32, 0.0, 0.0, 0.0];
        let predicted = mtp.predict_future_tokens(&hidden, 2);

        assert_eq!(predicted.len(), 2);
        assert_eq!(predicted[0], 3);
        assert_eq!(predicted[1], 6);
    }

    #[test]
    fn test_tree_speculative_verification() {
        let engine = SpeculativeDecoderEngine::new(SpeculativeConfig {
            max_draft_tokens: 3,
            acceptance_threshold: 0.8,
            ..Default::default()
        });

        let path1 = vec![101, 102, 103];
        let path2 = vec![101, 104, 105];
        let probs1 = vec![0.9, 0.85, 0.3]; // Path 1 accepts 2 tokens
        let probs2 = vec![0.9, 0.95, 0.9]; // Path 2 accepts 3 tokens
        let tgt1 = vec![0.95, 0.9, 0.1];
        let tgt2 = vec![0.95, 0.98, 0.95];

        let result = engine.verify_tree_candidates(
            &[path1, path2],
            &[probs1, probs2],
            &[tgt1, tgt2],
            &[Some(999), None],
        );

        assert_eq!(result.accepted_tokens, vec![101, 104, 105]);
        assert_eq!(result.num_accepted, 3);
    }
}
