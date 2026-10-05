#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// RoPE (Rotary Position Embedding) Scaling Strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RopeScalingType {
    /// Standard unscaled RoPE.
    Default,
    /// Linear position interpolation (PI).
    Linear { factor: f32 },
    /// YaRN (Yet another RoPE extensioN) with frequency band ramps.
    Yarn {
        factor: f32,
        original_max_position: u32,
        beta_fast: f32,
        beta_slow: f32,
        attn_factor: f32,
    },
    /// LongRoPE non-uniform evolutionary search scaling factors.
    LongRope {
        short_factor: f32,
        long_factor: f32,
        original_max_position: u32,
    },
    /// Llama 3 frequency-based dynamic adjustment.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: u32,
    },
}

/// Configuration for Rotary Position Embedding calculations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeConfig {
    pub head_dim: usize,
    pub base_theta: f32,
    pub scaling: RopeScalingType,
    pub max_position_embeddings: usize,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            head_dim: 128,
            base_theta: 10000.0,
            scaling: RopeScalingType::Default,
            max_position_embeddings: 8192,
        }
    }
}

/// Rotary Position Embedding Scaling Engine.
#[derive(Debug, Clone)]
pub struct RopeScalingEngine {
    pub config: RopeConfig,
    inv_frequencies: Vec<f32>,
}

impl RopeScalingEngine {
    #[must_use]
    pub fn new(config: RopeConfig) -> Self {
        let half_dim = config.head_dim / 2;
        let mut inv_frequencies = Vec::with_capacity(half_dim);

        for i in 0..half_dim {
            let dim_fraction = (2 * i) as f32 / config.head_dim as f32;
            let base_inv_freq = 1.0 / config.base_theta.powf(dim_fraction);

            let scaled_inv_freq = match config.scaling {
                RopeScalingType::Default => base_inv_freq,
                RopeScalingType::Linear { factor } => base_inv_freq / factor.max(1.0),
                RopeScalingType::Yarn {
                    factor,
                    original_max_position,
                    beta_fast,
                    beta_slow,
                    attn_factor,
                } => {
                    let low_freq_wavelen = original_max_position as f32 / beta_slow;
                    let high_freq_wavelen = original_max_position as f32 / beta_fast;
                    let wavelen = 2.0 * std::f32::consts::PI / base_inv_freq;

                    if wavelen < high_freq_wavelen {
                        base_inv_freq
                    } else if wavelen > low_freq_wavelen {
                        base_inv_freq / factor * attn_factor
                    } else {
                        let smooth = (original_max_position as f32 / wavelen - beta_fast)
                            / (beta_slow - beta_fast);
                        ((1.0 - smooth) * base_inv_freq + smooth * (base_inv_freq / factor))
                            * attn_factor
                    }
                }
                RopeScalingType::LongRope {
                    short_factor,
                    long_factor,
                    original_max_position,
                } => {
                    let factor = if config.max_position_embeddings > original_max_position as usize {
                        long_factor
                    } else {
                        short_factor
                    };
                    base_inv_freq / factor.max(1.0)
                }
                RopeScalingType::Llama3 {
                    factor,
                    low_freq_factor,
                    high_freq_factor,
                    original_max_position,
                } => {
                    let low_freq_wavelen = original_max_position as f32 / low_freq_factor;
                    let high_freq_wavelen = original_max_position as f32 / high_freq_factor;
                    let wavelen = 2.0 * std::f32::consts::PI / base_inv_freq;

                    if wavelen < high_freq_wavelen {
                        base_inv_freq
                    } else if wavelen > low_freq_wavelen {
                        base_inv_freq / factor
                    } else {
                        let smooth = (original_max_position as f32 / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * base_inv_freq / factor + smooth * base_inv_freq
                    }
                }
            };

            inv_frequencies.push(scaled_inv_freq);
        }

        Self {
            config,
            inv_frequencies,
        }
    }

    /// Computes (cos, sin) cache for a specific token position.
    #[must_use]
    pub fn compute_position_cos_sin(&self, position: usize) -> (Vec<f32>, Vec<f32>) {
        let half_dim = self.config.head_dim / 2;
        let mut cos = Vec::with_capacity(half_dim);
        let mut sin = Vec::with_capacity(half_dim);

        for &inv_freq in &self.inv_frequencies {
            let angle = position as f32 * inv_freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }

        (cos, sin)
    }

    /// Applies rotary embedding in-place to query or key slice of shape `[head_dim]`.
    pub fn apply_rotary_in_place(&self, vector: &mut [f32], position: usize) {
        let half_dim = self.config.head_dim / 2;
        if vector.len() < self.config.head_dim {
            return;
        }

        let (cos, sin) = self.compute_position_cos_sin(position);

        for i in 0..half_dim {
            let v0 = vector[i];
            let v1 = vector[i + half_dim];
            let c = cos[i];
            let s = sin[i];

            vector[i] = v0 * c - v1 * s;
            vector[i + half_dim] = v0 * s + v1 * c;
        }
    }
}
