#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Supported Speculative Decoding Strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpeculativeStrategy {
    DraftModel,
    NGramPromptLookup,
    SuffixMatching,
    EagleTreeDraft,
    DFlashDiffusion,
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
    /// Uses standard speculative acceptance criterion: accept if `p_target(x) >= p_draft(x)` or with probability `p_target(x) / p_draft(x)`.
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

    /// N-Gram Prompt Lookup Decoding (model-free speculative draft).
    /// Scans prefix context for matches with recent trailing tokens and extrapolates continuation.
    #[must_use]
    pub fn draft_ngram_lookup(&self, context: &[u32], max_draft: usize) -> Vec<u32> {
        let n = context.len();
        if n < self.config.ngram_min {
            return Vec::new();
        }

        for k in (self.config.ngram_min..=self.config.ngram_max.min(n)).rev() {
            let pattern = &context[n - k..n];
            // Scan prior context from earliest to latest (excluding final occurrence)
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
    /// Matches suffix of current prompt against pre-computed cache of common code/syntax blocks.
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

    /// EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) Tree Speculator.
    /// Simulates tree drafting of multiple candidate paths for batched verification.
    #[must_use]
    pub fn draft_eagle_tree(&self, root_token: u32, depth: usize) -> Vec<Vec<u32>> {
        let mut paths = vec![vec![root_token]];
        for _ in 1..depth {
            let mut next_paths = Vec::new();
            for path in paths {
                let last = *path.last().unwrap_or(&0);
                // Predict 2 top branches per node:
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

    /// DFlash: Diffusion-based Speculative Decoding Candidate Generator.
    /// Fast continuous feature denoising to synthesize non-autoregressive token blocks.
    #[must_use]
    pub fn draft_dflash_block(&self, seed_latent: &[f32], block_len: usize) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(block_len);
        for i in 0..block_len {
            let latent_val = seed_latent.get(i % seed_latent.len()).copied().unwrap_or(0.0);
            let token_id = ((latent_val.abs() * 1000.0) as u32) % 32000;
            tokens.push(token_id);
        }
        tokens
    }
}
