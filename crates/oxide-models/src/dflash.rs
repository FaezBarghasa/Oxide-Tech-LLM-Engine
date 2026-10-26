//! DFlash Speculative Block Drafter.
//! Non-autoregressive speculative drafter utilizing intermediate target model representations
//! and a short convolutional trunk to emit block-wise draft tokens in a single forward pass.

/// Configuration parameters for DFlash speculative block drafting.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    pub target_hidden_dim: usize,
    pub draft_hidden_dim: usize,
    pub block_size: usize,
    pub conv_kernel_size: usize,
    pub conv_group_size: usize,
    pub vocab_size: usize,
}

impl Default for DFlashConfig {
    fn default() -> Self {
        Self {
            target_hidden_dim: 4096,
            draft_hidden_dim: 1024,
            block_size: 4,
            conv_kernel_size: 3,
            conv_group_size: 64,
            vocab_size: 128256,
        }
    }
}

/// DFlash Speculative Block Drafter Model.
#[derive(Debug, Clone)]
pub struct DFlashModel {
    pub config: DFlashConfig,
    /// Linear projection from target model hidden state to draft hidden state.
    pub proj_in_weight: Vec<f32>, // [draft_hidden_dim, target_hidden_dim]
    /// 1D depthwise convolution weights [draft_hidden_dim, conv_kernel_size].
    pub conv_weight: Vec<f32>,
    /// Block-expansion projection [block_size * draft_hidden_dim, draft_hidden_dim].
    pub block_proj_weight: Vec<f32>,
    /// Draft LM Head output projection [vocab_size, draft_hidden_dim].
    pub output_head_weight: Vec<f32>,
}

impl DFlashModel {
    pub fn new(config: DFlashConfig) -> Self {
        let proj_in_size = config.draft_hidden_dim * config.target_hidden_dim;
        let conv_size = config.draft_hidden_dim * config.conv_kernel_size;
        let block_proj_size = config.block_size * config.draft_hidden_dim * config.draft_hidden_dim;
        let output_head_size = config.vocab_size * config.draft_hidden_dim;

        Self {
            config,
            proj_in_weight: vec![0.01f32; proj_in_size],
            conv_weight: vec![1.0f32 / (3.0); conv_size],
            block_proj_weight: vec![0.01f32; block_proj_size],
            output_head_weight: vec![0.001f32; output_head_size],
        }
    }

    /// Generates a speculative block of $B$ draft token IDs and log-probabilities
    /// conditioned on an intermediate hidden state extracted from the target model.
    pub fn draft_block(
        &self,
        target_hidden: &[f32], // [target_hidden_dim]
    ) -> (Vec<u32>, Vec<f32>) {
        let d_in = self.config.target_hidden_dim;
        let d_draft = self.config.draft_hidden_dim;
        let b_size = self.config.block_size;
        assert_eq!(target_hidden.len(), d_in);

        // 1. Context Projection: target_hidden -> draft_hidden
        let mut draft_feat = vec![0.0f32; d_draft];
        for r in 0..d_draft {
            let mut sum = 0.0f32;
            let offset = r * d_in;
            for c in 0..d_in {
                sum += self.proj_in_weight[offset + c] * target_hidden[c];
            }
            // GELU non-linearity
            draft_feat[r] =
                0.5 * sum * (1.0 + (0.797_884_6 * (sum + 0.044_715 * sum * sum * sum)).tanh());
        }

        // 2. Convolutional filter
        let k_size = self.config.conv_kernel_size;
        let mut conv_out = vec![0.0f32; d_draft];
        for r in 0..d_draft {
            let mut sum = 0.0f32;
            let offset = r * k_size;
            for k in 0..k_size {
                sum += self.conv_weight[offset + k] * draft_feat[r];
            }
            conv_out[r] = sum;
        }

        // 3. Block expansion: draft_feat -> block_size * draft_hidden_dim
        let mut block_feats = vec![0.0f32; b_size * d_draft];
        for b in 0..b_size {
            let b_offset = b * d_draft;
            for r in 0..d_draft {
                let mut sum = 0.0f32;
                let proj_offset = (b_offset + r) * d_draft;
                for c in 0..d_draft {
                    sum += self.block_proj_weight[proj_offset + c] * conv_out[c];
                }
                block_feats[b_offset + r] = sum;
            }
        }

        // 4. Token classification per block slot
        let mut draft_tokens = Vec::with_capacity(b_size);
        let mut confidences = Vec::with_capacity(b_size);

        for b in 0..b_size {
            let slot_feat = &block_feats[b * d_draft..(b + 1) * d_draft];
            let mut best_id = 0u32;
            let mut max_logit = f32::NEG_INFINITY;

            // Argmax over vocab
            for v in 0..self.config.vocab_size {
                let mut logit = 0.0f32;
                let head_offset = v * d_draft;
                for c in 0..d_draft {
                    logit += self.output_head_weight[head_offset + c] * slot_feat[c];
                }
                if logit > max_logit {
                    max_logit = logit;
                    best_id = v as u32;
                }
            }

            draft_tokens.push(best_id);
            confidences.push(max_logit);
        }

        (draft_tokens, confidences)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dflash_block_draft() {
        let config = DFlashConfig {
            target_hidden_dim: 32,
            draft_hidden_dim: 16,
            block_size: 4,
            conv_kernel_size: 3,
            conv_group_size: 16,
            vocab_size: 64,
        };
        let model = DFlashModel::new(config);
        let target_hidden = vec![0.5f32; 32];
        let (tokens, confidences) = model.draft_block(&target_hidden);

        assert_eq!(tokens.len(), 4);
        assert_eq!(confidences.len(), 4);
    }
}
