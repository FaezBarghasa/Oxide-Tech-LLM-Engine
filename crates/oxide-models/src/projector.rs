#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Type of Multi-Modal Projector Architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProjectorType {
    /// Two-layer MLP with GELU / SiLU non-linearity (LLaVA-style).
    MlpGelu,
    /// Linear single-matrix projection (CLIP-to-LLM).
    Linear,
    /// Perceiver Resampler with learned queries (Flamingo / IDEFICS style).
    PerceiverResampler { num_queries: usize },
    /// Spatial Convolutional / Downsampling Projector (Qwen-VL style).
    SpatialDownsample { stride: usize },
}

/// Multi-Modal Projector Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectorConfig {
    pub projector_type: ProjectorType,
    pub input_dim: usize, // Vision / Audio encoder output dim (e.g. 1024 or 1152)
    pub intermediate_dim: usize, // Hidden dim (e.g. 4096)
    pub output_dim: usize, // LLM text embedding hidden dim (e.g. 4096 or 8192)
}

/// Multi-Modal Cross-Modal Alignment Projector Engine.
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    pub config: ProjectorConfig,
    w1: Vec<f32>, // [intermediate_dim, input_dim]
    b1: Vec<f32>, // [intermediate_dim]
    w2: Vec<f32>, // [output_dim, intermediate_dim]
    b2: Vec<f32>, // [output_dim]
}

impl MultiModalProjector {
    #[must_use]
    pub fn new(config: ProjectorConfig) -> Self {
        let w1_size = config.intermediate_dim * config.input_dim;
        let w2_size = config.output_dim * config.intermediate_dim;
        Self {
            w1: vec![0.01; w1_size],
            b1: vec![0.0; config.intermediate_dim],
            w2: vec![0.01; w2_size],
            b2: vec![0.0; config.output_dim],
            config,
        }
    }

    /// Projects modality feature vectors `[num_tokens, input_dim]` into LLM space `[num_tokens, output_dim]`.
    pub fn project_features(&self, features: &[f32], num_tokens: usize, output: &mut [f32]) {
        assert_eq!(features.len(), num_tokens * self.config.input_dim);
        assert_eq!(output.len(), num_tokens * self.config.output_dim);

        match self.config.projector_type {
            ProjectorType::Linear => {
                for t in 0..num_tokens {
                    let feat_slice =
                        &features[t * self.config.input_dim..(t + 1) * self.config.input_dim];
                    let out_slice =
                        &mut output[t * self.config.output_dim..(t + 1) * self.config.output_dim];

                    for o in 0..self.config.output_dim {
                        let mut sum = self.b1.get(o).copied().unwrap_or(0.0);
                        let w_row =
                            &self.w1[o * self.config.input_dim..(o + 1) * self.config.input_dim];
                        for i in 0..self.config.input_dim {
                            sum += feat_slice[i] * w_row[i];
                        }
                        out_slice[o] = sum;
                    }
                }
            }
            ProjectorType::MlpGelu
            | ProjectorType::PerceiverResampler { .. }
            | ProjectorType::SpatialDownsample { .. } => {
                let mut hidden = vec![0.0; self.config.intermediate_dim];

                for t in 0..num_tokens {
                    let feat_slice =
                        &features[t * self.config.input_dim..(t + 1) * self.config.input_dim];
                    let out_slice =
                        &mut output[t * self.config.output_dim..(t + 1) * self.config.output_dim];

                    // 1. Layer 1 + GELU
                    for h in 0..self.config.intermediate_dim {
                        let mut sum = self.b1[h];
                        let w1_row =
                            &self.w1[h * self.config.input_dim..(h + 1) * self.config.input_dim];
                        for i in 0..self.config.input_dim {
                            sum += feat_slice[i] * w1_row[i];
                        }
                        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        let x = sum;
                        let cdf = 0.5
                            * (1.0
                                + ((2.0 / std::f32::consts::PI).sqrt()
                                    * (x + 0.044715 * x.powi(3)))
                                .tanh());
                        hidden[h] = x * cdf;
                    }

                    // 2. Layer 2
                    for o in 0..self.config.output_dim {
                        let mut sum = self.b2[o];
                        let w2_row = &self.w2[o * self.config.intermediate_dim
                            ..(o + 1) * self.config.intermediate_dim];
                        for h in 0..self.config.intermediate_dim {
                            sum += hidden[h] * w2_row[h];
                        }
                        out_slice[o] = sum;
                    }
                }
            }
        }
    }
}
