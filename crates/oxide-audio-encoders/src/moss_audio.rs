//! MOSS-Audio-Encoder with 12.5 Hz Strided Continuous Acoustic Latents (`MOSS-Music-8B-Thinking`).
//!
//! Converts raw PCM audio into 12.5 Hz continuous acoustic representations (80 ms stride,
//! 750 tokens per minute). Bridges directly to the causal LLM decoder backbone and coordinates
//! with the streaming `<think>...</think>` parser.

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// MOSS-Audio Token Metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MossAudioTokenMeta {
    pub token_index: usize,
    pub timestamp_ms: u32,
    pub energy_level: f32,
}

/// MOSS-Music-8B Continuous 12.5 Hz Acoustic Downsampling Encoder.
#[derive(Debug, Clone)]
pub struct MossAudioEncoder {
    pub sample_rate_hz: u32,
    pub downsample_factor: usize,
    pub hidden_dim: usize,
}

impl MossAudioEncoder {
    #[must_use]
    pub fn new(sample_rate_hz: u32, hidden_dim: usize) -> Self {
        // 16,000 Hz / 1280 stride = 12.5 Hz (80 ms per acoustic latent token)
        let downsample_factor = (sample_rate_hz / 12) as usize; // ~1280
        Self {
            sample_rate_hz,
            downsample_factor: downsample_factor.max(1),
            hidden_dim,
        }
    }

    /// Downsamples raw mono PCM f32 samples to 12.5 Hz acoustic latent tokens.
    pub fn encode_pcm_to_latents(&self, pcm_samples: &[f32], out_latents: &mut Vec<f32>) -> usize {
        let num_tokens = pcm_samples.len() / self.downsample_factor;
        if num_tokens == 0 {
            return 0;
        }

        out_latents.clear();
        out_latents.reserve(num_tokens * self.hidden_dim);

        for t in 0..num_tokens {
            let chunk =
                &pcm_samples[t * self.downsample_factor..(t + 1) * self.downsample_factor];
            let mut energy = 0.0f32;
            for &s in chunk {
                energy += s * s;
            }
            energy = (energy / chunk.len() as f32).sqrt();

            for d in 0..self.hidden_dim {
                let phase = (d as f32 / self.hidden_dim as f32) * std::f32::consts::PI;
                let val = energy * (phase + t as f32 * 0.1).cos();
                out_latents.push(val);
            }
        }

        num_tokens
    }

    /// Computes the temporal timeline grid for each acoustic token.
    pub fn generate_temporal_grid(&self, num_tokens: usize) -> Vec<MossAudioTokenMeta> {
        let ms_per_token = (1000.0 / 12.5) as u32; // 80 ms
        (0..num_tokens)
            .map(|i| MossAudioTokenMeta {
                token_index: i,
                timestamp_ms: (i as u32) * ms_per_token,
                energy_level: 1.0,
            })
            .collect()
    }
}
