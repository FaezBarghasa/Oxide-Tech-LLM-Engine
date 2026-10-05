#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Configuration for Speculative Decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    pub max_draft_tokens: usize,
    pub acceptance_threshold: f32,
    pub draft_model_name: String,
    pub target_model_name: String,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: 5,
            acceptance_threshold: 0.85,
            draft_model_name: "llama-3-1b-draft".to_string(),
            target_model_name: "llama-3-70b-target".to_string(),
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
    /// Uses standard speculative acceptance criterion: accept if `p_target(x) >= p_draft(x)` or with probability `p_target(x) / p_draft(x)`.
    pub fn verify_draft_tokens(
        &self,
        draft_tokens: &[u32],
        draft_probs: &[f32],
        target_token_probs: &[f32], // Target probability of each drafted token
        target_greedy_recovery_tokens: &[u32], // The token target model would pick at each step
    ) -> SpeculativeVerificationResult {
        let mut accepted = Vec::with_capacity(draft_tokens.len());
        let mut recovery_token = None;

        for (idx, &token) in draft_tokens.iter().enumerate() {
            let p_draft = draft_probs.get(idx).copied().unwrap_or(0.5).max(1e-6);
            let p_target = target_token_probs.get(idx).copied().unwrap_or(0.5);

            if p_target >= p_draft || (p_target / p_draft) >= self.config.acceptance_threshold {
                accepted.push(token);
            } else {
                // Draft rejected at this token; pick recovery token from target model
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
}
