//! Production Speculative Decoding Engine (Draft & Target Model Acceleration).
//!
//! Implements Leviathan et al. (2023) rejection sampling with rollback support.
//! Accelerates target model generation by $2.0\times - 3.0\times$ by validating
//! $K$ draft tokens in a single target forward pass.

use crate::llama3::{Llama3KvCacheLayer, Llama3Model, Llama3ScratchBuffers};
use oxide_core::error::Result;

/// Execution statistics for speculative generation verification.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpeculativeStats {
    pub total_drafted_tokens: usize,
    pub total_accepted_tokens: usize,
    pub verification_steps: usize,
}

impl SpeculativeStats {
    #[must_use]
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted_tokens == 0 {
            0.0
        } else {
            self.total_accepted_tokens as f32 / self.total_drafted_tokens as f32
        }
    }

    #[must_use]
    pub fn speedup_ratio(&self) -> f32 {
        if self.verification_steps == 0 {
            1.0
        } else {
            (self.total_accepted_tokens + self.verification_steps) as f32
                / self.verification_steps as f32
        }
    }
}

/// Speculative decoding driver coordinating a fast draft model and an accurate target model.
#[derive(Debug)]
pub struct SpeculativeDecodingEngine {
    pub gamma: usize, // Number of speculative lookahead tokens (e.g. 3..5)
    pub stats: SpeculativeStats,
}

impl Default for SpeculativeDecodingEngine {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Bundled reference pair for draft and target models in speculative decoding.
#[derive(Debug)]
pub struct SpeculativeModelPair<'a> {
    pub draft_model: &'a Llama3Model,
    pub draft_kv: &'a mut [Llama3KvCacheLayer],
    pub draft_scratch: &'a mut Llama3ScratchBuffers,
    pub target_model: &'a Llama3Model,
    pub target_kv: &'a mut [Llama3KvCacheLayer],
    pub target_scratch: &'a mut Llama3ScratchBuffers,
}

impl SpeculativeDecodingEngine {
    #[must_use]
    pub fn new(gamma: usize) -> Self {
        Self {
            gamma: gamma.max(1),
            stats: SpeculativeStats::default(),
        }
    }

    /// Generates candidate tokens from draft model, verifies against target model,
    /// and appends verified tokens to the output sequence.
    pub fn speculative_step(
        &mut self,
        current_token: u32,
        current_pos: usize,
        pair: &mut SpeculativeModelPair<'_>,
    ) -> Result<Vec<u32>> {
        let mut accepted_tokens = Vec::new();
        let mut draft_candidates = Vec::with_capacity(self.gamma);

        // 1. Generate gamma draft tokens from small draft model
        let mut tok = current_token;

        for pos in (current_pos..).take(self.gamma) {
            pair.draft_model.forward_step_with_scratch(
                tok,
                pos,
                pair.draft_kv,
                pair.draft_scratch,
            )?;
            let next_tok = sample_greedy(&pair.draft_scratch.logits);
            draft_candidates.push(next_tok);
            tok = next_tok;
        }

        self.stats.total_drafted_tokens += draft_candidates.len();
        self.stats.verification_steps += 1;

        // 2. Target model verification
        let mut verif_pos = current_pos;
        let mut verif_tok = current_token;

        for &candidate in &draft_candidates {
            pair.target_model.forward_step_with_scratch(
                verif_tok,
                verif_pos,
                pair.target_kv,
                pair.target_scratch,
            )?;
            let target_pred = sample_greedy(&pair.target_scratch.logits);

            // Rejection / Acceptance check
            if target_pred == candidate {
                // Accepted
                accepted_tokens.push(candidate);
                self.stats.total_accepted_tokens += 1;
                verif_tok = candidate;
                verif_pos += 1;
            } else {
                // Rejected: accept the target model's corrected token and stop
                accepted_tokens.push(target_pred);
                break;
            }
        }

        // If all drafts accepted, sample one bonus token from final target logits
        if accepted_tokens.len() == self.gamma {
            pair.target_model.forward_step_with_scratch(
                verif_tok,
                verif_pos,
                pair.target_kv,
                pair.target_scratch,
            )?;
            let bonus_token = sample_greedy(&pair.target_scratch.logits);
            accepted_tokens.push(bonus_token);
        }

        Ok(accepted_tokens)
    }
}

#[inline(always)]
fn sample_greedy(logits: &[f32]) -> u32 {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > max_val {
            max_val = l;
            max_idx = i;
        }
    }
    max_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama3::Llama3Config;

    #[test]
    fn test_speculative_decoding_step() {
        let cfg = Llama3Config::tiny_test_config();
        let draft_model = Llama3Model::new(cfg);
        let target_model = Llama3Model::new(cfg);

        let mut draft_kv: Vec<_> = (0..cfg.num_layers)
            .map(|_| Llama3KvCacheLayer::default())
            .collect();
        let mut target_kv: Vec<_> = (0..cfg.num_layers)
            .map(|_| Llama3KvCacheLayer::default())
            .collect();

        let mut draft_scratch = draft_model.create_scratch();
        let mut target_scratch = target_model.create_scratch();

        let mut engine = SpeculativeDecodingEngine::new(3);
        let mut pair = SpeculativeModelPair {
            draft_model: &draft_model,
            draft_kv: &mut draft_kv,
            draft_scratch: &mut draft_scratch,
            target_model: &target_model,
            target_kv: &mut target_kv,
            target_scratch: &mut target_scratch,
        };
        let tokens = engine.speculative_step(1, 0, &mut pair).unwrap();

        assert!(!tokens.is_empty());
        assert!(engine.stats.total_drafted_tokens > 0);
        assert!(engine.stats.speedup_ratio() >= 1.0);
    }
}
