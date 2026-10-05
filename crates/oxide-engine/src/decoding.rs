//! Advanced High-Throughput Decoding Algorithms (Parallel Sampling, Beam Search).
//!
//! Provides enterprise decoding strategies:
//! - Parallel Sampling (Best-of-N candidate exploration with log-probability / reward ranking)
//! - Beam Search (Width K, length-normalized log-probability scoring, and EOS hypothesis pruning)

use serde::{Deserialize, Serialize};

/// Beam Hypothesis State.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub cumulative_log_prob: f32,
    pub is_finished: bool,
}

impl BeamHypothesis {
    #[must_use]
    pub fn new(prompt: Vec<u32>) -> Self {
        Self {
            tokens: prompt,
            cumulative_log_prob: 0.0,
            is_finished: false,
        }
    }

    /// Evaluates length-normalized score:
    /// score = cumulative_log_prob / ((5 + length)^alpha / (5 + 1)^alpha)
    #[must_use]
    pub fn normalized_score(&self, alpha: f32) -> f32 {
        let length = self.tokens.len() as f32;
        let penalty = ((5.0 + length) / 6.0).powf(alpha);
        if penalty > 0.0 {
            self.cumulative_log_prob / penalty
        } else {
            self.cumulative_log_prob
        }
    }
}

/// Configuration for Beam Search Decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BeamSearchConfig {
    pub beam_width: usize,
    pub max_tokens: usize,
    pub length_penalty_alpha: f32,
    pub eos_token_id: u32,
    pub early_stopping: bool,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_tokens: 128,
            length_penalty_alpha: 0.6,
            eos_token_id: 128_001,
            early_stopping: true,
        }
    }
}

/// Beam Search Execution Engine.
#[derive(Debug, Clone)]
pub struct BeamSearchEngine {
    pub config: BeamSearchConfig,
}

impl BeamSearchEngine {
    #[must_use]
    pub fn new(config: BeamSearchConfig) -> Self {
        Self { config }
    }

    /// Advances active beams by 1 decoding step given top candidate tokens and their log-probabilities.
    /// `candidates`: for each active beam, a list of `(token_id, log_prob)`.
    #[must_use]
    pub fn step(
        &self,
        active_beams: &[BeamHypothesis],
        candidates: &[Vec<(u32, f32)>],
    ) -> Vec<BeamHypothesis> {
        let mut expanded: Vec<BeamHypothesis> = Vec::new();

        for (beam_idx, beam) in active_beams.iter().enumerate() {
            if beam.is_finished {
                expanded.push(beam.clone());
                continue;
            }

            if let Some(beam_cands) = candidates.get(beam_idx) {
                for &(tok, log_p) in beam_cands {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(tok);
                    let is_finished = tok == self.config.eos_token_id;
                    expanded.push(BeamHypothesis {
                        tokens: new_tokens,
                        cumulative_log_prob: beam.cumulative_log_prob + log_p,
                        is_finished,
                    });
                }
            }
        }

        // Rank by normalized score
        expanded.sort_by(|a, b| {
            let sa = a.normalized_score(self.config.length_penalty_alpha);
            let sb = b.normalized_score(self.config.length_penalty_alpha);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        expanded.truncate(self.config.beam_width);
        expanded
    }
}

/// Parallel Sampling Result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelSampleCandidate {
    pub candidate_id: usize,
    pub tokens: Vec<u32>,
    pub total_log_prob: f32,
}

/// Parallel Sampling (Best-of-N) Engine.
#[derive(Debug, Clone)]
pub struct ParallelSamplingEngine {
    pub n_samples: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl ParallelSamplingEngine {
    #[must_use]
    pub fn new(n_samples: usize, temperature: f32, top_p: f32) -> Self {
        Self {
            n_samples,
            temperature,
            top_p,
        }
    }

    /// Ranks candidates by descending total log probability.
    pub fn rank_candidates(candidates: &mut [ParallelSampleCandidate]) {
        candidates.sort_by(|a, b| {
            b.total_log_prob
                .partial_cmp(&a.total_log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}
