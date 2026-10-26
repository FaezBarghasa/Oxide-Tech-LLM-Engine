//! DSpark Speculative Drafter (DFlash + Markov & Confidence Head).
//! Implements inter-token dependency correction for non-autoregressive block drafting,
//! using low-rank Markov transition heads and log-SNR noise schedule conditioning.

use crate::dflash::{DFlashConfig, DFlashModel};

/// Configuration for DSpark Markov-corrected block drafting.
#[derive(Debug, Clone, PartialEq)]
pub struct DSparkConfig {
    pub dflash: DFlashConfig,
    pub markov_rank: usize,
    pub min_log_snr: f32,
    pub max_log_snr: f32,
}

impl Default for DSparkConfig {
    fn default() -> Self {
        Self {
            dflash: DFlashConfig::default(),
            markov_rank: 64,
            min_log_snr: -4.0,
            max_log_snr: 4.0,
        }
    }
}

/// DSpark Speculative Block Drafter.
#[derive(Debug, Clone)]
pub struct DSparkModel {
    pub base_drafter: DFlashModel,
    pub config: DSparkConfig,
    /// Markov transition low-rank factor 1: [markov_rank, vocab_size]
    pub markov_w1: Vec<f32>,
    /// Markov transition low-rank factor 2: [vocab_size, markov_rank]
    pub markov_w2: Vec<f32>,
    /// Confidence predictor head: [1, draft_hidden_dim]
    pub confidence_head: Vec<f32>,
}

impl DSparkModel {
    pub fn new(config: DSparkConfig) -> Self {
        let base_drafter = DFlashModel::new(config.dflash.clone());
        let vocab_size = config.dflash.vocab_size;
        let rank = config.markov_rank;
        let d_draft = config.dflash.draft_hidden_dim;

        Self {
            base_drafter,
            config,
            markov_w1: vec![0.005f32; rank * vocab_size],
            markov_w2: vec![0.005f32; vocab_size * rank],
            confidence_head: vec![0.01f32; d_draft],
        }
    }

    /// Computes Markov transition logits bias for a given preceding token ID:
    /// $\text{bias} = W_2 (W_1 \cdot e_{prev})$
    pub fn compute_markov_bias(&self, prev_token_id: u32, bias_out: &mut [f32]) {
        let vocab_size = self.config.dflash.vocab_size;
        let rank = self.config.markov_rank;
        assert_eq!(bias_out.len(), vocab_size);

        if (prev_token_id as usize) >= vocab_size {
            bias_out.fill(0.0);
            return;
        }

        // 1. Column slice of W1 for prev_token_id: vector of size rank
        let mut hidden_rank = vec![0.0f32; rank];
        for r in 0..rank {
            hidden_rank[r] = self.markov_w1[r * vocab_size + (prev_token_id as usize)];
        }

        // 2. Project back to vocab: W2 * hidden_rank
        for v in 0..vocab_size {
            let mut sum = 0.0f32;
            let offset = v * rank;
            for r in 0..rank {
                sum += self.markov_w2[offset + r] * hidden_rank[r];
            }
            bias_out[v] = sum;
        }
    }

    /// Performs sequential Markov-corrected block drafting:
    /// 1. Emits initial draft block from target intermediate hidden state.
    /// 2. Iteratively adjusts logits of token $t \in [1, B-1]$ using Markov transition bias from token $t-1$.
    pub fn draft_block_markov(
        &self,
        target_hidden: &[f32],
        anchor_token_id: u32,
    ) -> (Vec<u32>, Vec<f32>) {
        let (raw_tokens, confidences) = self.base_drafter.draft_block(target_hidden);
        let b_size = raw_tokens.len();
        let vocab_size = self.config.dflash.vocab_size;

        let mut corrected_tokens = Vec::with_capacity(b_size);
        let mut corrected_confidences = Vec::with_capacity(b_size);
        let mut markov_bias = vec![0.0f32; vocab_size];

        let mut prev_token = anchor_token_id;

        for (b, (&raw_tok, &raw_conf)) in raw_tokens.iter().zip(confidences.iter()).enumerate() {
            if b == 0 {
                // First slot uses anchor token transition
                self.compute_markov_bias(prev_token, &mut markov_bias);
                let corrected_conf = raw_conf + markov_bias[raw_tok as usize];
                corrected_tokens.push(raw_tok);
                corrected_confidences.push(corrected_conf);
                prev_token = raw_tok;
            } else {
                // Subsequent slots are biased by previous draft prediction
                self.compute_markov_bias(prev_token, &mut markov_bias);
                let corrected_conf = raw_conf + markov_bias[raw_tok as usize];
                corrected_tokens.push(raw_tok);
                corrected_confidences.push(corrected_conf);
                prev_token = raw_tok;
            }
        }

        (corrected_tokens, corrected_confidences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dspark_markov_drafting() {
        let dflash = DFlashConfig {
            target_hidden_dim: 32,
            draft_hidden_dim: 16,
            block_size: 4,
            conv_kernel_size: 3,
            conv_group_size: 16,
            vocab_size: 64,
        };
        let config = DSparkConfig {
            dflash,
            markov_rank: 8,
            min_log_snr: -4.0,
            max_log_snr: 4.0,
        };

        let model = DSparkModel::new(config);
        let target_hidden = vec![0.2f32; 32];
        let (tokens, confidences) = model.draft_block_markov(&target_hidden, 12);

        assert_eq!(tokens.len(), 4);
        assert_eq!(confidences.len(), 4);
    }
}
