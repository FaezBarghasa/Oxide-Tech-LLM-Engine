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
            for i in 0..search_limit {
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
                for j in 0..h {
                    dot += self.mtp_head_weights[row_start + j] * last_hidden[j];
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
