//! TokenSpeed-inspired DeepSeek Multi-Head Latent Attention (MLA 2).
//!
//! Slashes KV cache memory and memory bandwidth by over 85% compared to standard MHA/GQA:
//! - Compresses Keys and Values into a single low-rank latent vector:
//!   `c_t^{KV} = W^{DKV} h_t \in R^{d_c}` (typically d_c = 512).
//! - Pairs latent KV with decoupled RoPE key:
//!   `k_t^R = RoPE(W^{KR} h_t) \in R^{d_R}` (typically d_R = 64).
//! - Evaluates attention during decode via Matrix Absorption:
//!   Attention logits are computed directly between absorbed query heads and cached latent vectors,
//!   eliminating the need to ever expand latent keys into full head dimensions in memory.

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// Architectural parameters for Multi-Head Latent Attention (MLA).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlaConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    /// Low-rank compression dimension for Key/Value states (e.g. 512).
    pub kv_lora_rank: usize,
    /// Low-rank compression dimension for Query states (e.g. 1536).
    pub q_lora_rank: usize,
    /// Decoupled Rotary Position Embedding dimension (e.g. 64).
    pub rope_head_dim: usize,
}

impl Default for MlaConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_heads: 16,
            head_dim: 128,
            kv_lora_rank: 512,
            q_lora_rank: 768,
            rope_head_dim: 64,
        }
    }
}

/// Highly compressed KV cache entry stored per generated token.
///
/// Stores only the low-rank latent vector and decoupled positional key:
/// Total floats: `kv_lora_rank + rope_head_dim` (e.g. 512 + 64 = 576 floats),
/// vs. Standard MHA `16 heads * 128 dim * 2 = 4096 floats` (7.1x smaller!).
#[derive(Debug, Clone, PartialEq)]
pub struct MlaCompressedKvEntry {
    pub latent_kv: Vec<f32>,
    pub rope_k: Vec<f32>,
}

impl MlaCompressedKvEntry {
    #[must_use]
    pub fn memory_footprint_bytes(&self) -> usize {
        (self.latent_kv.len() + self.rope_k.len()) * std::mem::size_of::<f32>()
    }
}

/// Multi-Head Latent Attention Layer with Matrix Absorption.
#[derive(Debug)]
pub struct MlaAttentionLayer {
    pub config: MlaConfig,
    /// KV Down-projection: `[kv_lora_rank, hidden_dim]`
    pub w_dkv: Vec<f32>,
    /// Decoupled RoPE Key projection: `[rope_head_dim, hidden_dim]`
    pub w_kr: Vec<f32>,
    /// Absorbed Query projection weights: `[num_heads, kv_lora_rank, hidden_dim]`
    pub w_q_absorbed: Vec<f32>,
    /// Query RoPE projection weights: `[num_heads, rope_head_dim, hidden_dim]`
    pub w_qr: Vec<f32>,
    /// Value Up-projection weights: `[hidden_dim, kv_lora_rank]`
    pub w_uv: Vec<f32>,
    /// Output projection weights: `[hidden_dim, hidden_dim]`
    pub w_out: Vec<f32>,
}

impl MlaAttentionLayer {
    #[must_use]
    pub fn new(config: MlaConfig) -> Self {
        let h = config.hidden_dim;
        let dc = config.kv_lora_rank;
        let dr = config.rope_head_dim;
        let nh = config.num_heads;

        Self {
            w_dkv: vec![0.01; dc * h],
            w_kr: vec![0.01; dr * h],
            w_q_absorbed: vec![0.01; nh * dc * h],
            w_qr: vec![0.01; nh * dr * h],
            w_uv: vec![0.01; h * dc],
            w_out: vec![0.01; h * h],
            config,
        }
    }

    /// Compresses a new token's hidden state into an `MlaCompressedKvEntry` for caching.
    #[must_use]
    pub fn compress_kv(&self, hidden: &[f32]) -> MlaCompressedKvEntry {
        let h = self.config.hidden_dim;
        let dc = self.config.kv_lora_rank;
        let dr = self.config.rope_head_dim;

        let mut latent_kv = vec![0.0f32; dc];
        for i in 0..dc {
            let mut sum = 0.0f32;
            let offset = i * h;
            for j in 0..h {
                sum += self.w_dkv[offset + j] * hidden[j];
            }
            latent_kv[i] = sum;
        }

        let mut rope_k = vec![0.0f32; dr];
        for i in 0..dr {
            let mut sum = 0.0f32;
            let offset = i * h;
            for j in 0..h {
                sum += self.w_kr[offset + j] * hidden[j];
            }
            rope_k[i] = sum;
        }

        MlaCompressedKvEntry { latent_kv, rope_k }
    }

    /// Evaluates forward decode attention using Matrix Absorption directly against the compressed KV cache.
    pub fn forward_decode(
        &self,
        hidden: &[f32],
        kv_cache: &[MlaCompressedKvEntry],
        output: &mut [f32],
    ) -> Result<()> {
        let h = self.config.hidden_dim;
        let nh = self.config.num_heads;
        let dc = self.config.kv_lora_rank;
        let dr = self.config.rope_head_dim;

        if hidden.len() != h || output.len() != h {
            return Err(EngineError::ShapeMismatch);
        }

        if kv_cache.is_empty() {
            output.copy_from_slice(hidden);
            return Ok(());
        }

        let seq_len = kv_cache.len();
        let scale = 1.0 / ((self.config.head_dim + dr) as f32).sqrt();

        // 1. Compute absorbed queries and RoPE queries for all heads
        let mut q_absorbed = vec![0.0f32; nh * dc];
        let mut q_rope = vec![0.0f32; nh * dr];

        for head in 0..nh {
            let q_abs_offset = head * dc;
            let w_abs_head = head * dc * h;
            for i in 0..dc {
                let mut sum = 0.0f32;
                let row = w_abs_head + i * h;
                for j in 0..h {
                    sum += self.w_q_absorbed[row + j] * hidden[j];
                }
                q_absorbed[q_abs_offset + i] = sum;
            }

            let q_rope_offset = head * dr;
            let w_rope_head = head * dr * h;
            for i in 0..dr {
                let mut sum = 0.0f32;
                let row = w_rope_head + i * h;
                for j in 0..h {
                    sum += self.w_qr[row + j] * hidden[j];
                }
                q_rope[q_rope_offset + i] = sum;
            }
        }

        // 2. Matrix Absorption Attention: Dot products directly between absorbed Q and latent KV
        let mut head_outputs = vec![0.0f32; nh * dc];

        for head in 0..nh {
            let q_abs = &q_absorbed[head * dc..(head + 1) * dc];
            let q_r = &q_rope[head * dr..(head + 1) * dr];

            let mut scores = Vec::with_capacity(seq_len);
            let mut max_score = -f32::INFINITY;

            for entry in kv_cache {
                let mut dot_latent = 0.0f32;
                for i in 0..dc {
                    dot_latent += q_abs[i] * entry.latent_kv[i];
                }

                let mut dot_rope = 0.0f32;
                for i in 0..dr {
                    dot_rope += q_r[i] * entry.rope_k[i];
                }

                let score = (dot_latent + dot_rope) * scale;
                if score > max_score {
                    max_score = score;
                }
                scores.push(score);
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = if sum_exp > 1e-8 { 1.0 / sum_exp } else { 0.0 };
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Accumulate weighted latent vectors
            let out_slice = &mut head_outputs[head * dc..(head + 1) * dc];
            for (pos, entry) in kv_cache.iter().enumerate() {
                let weight = scores[pos];
                for i in 0..dc {
                    out_slice[i] += weight * entry.latent_kv[i];
                }
            }
        }

        // 3. Up-project combined latent heads into hidden dimension
        output.fill(0.0);
        for head in 0..nh {
            let head_slice = &head_outputs[head * dc..(head + 1) * dc];
            for i in 0..h {
                let mut sum = 0.0f32;
                let row = i * dc;
                for j in 0..dc {
                    sum += self.w_uv[row + j] * head_slice[j];
                }
                output[i] += sum / (nh as f32);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_compression_and_absorption_decode() {
        let config = MlaConfig {
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 16,
            kv_lora_rank: 8,
            q_lora_rank: 16,
            rope_head_dim: 4,
        };

        let mla = MlaAttentionLayer::new(config);
        let hidden = vec![1.0f32; 32];

        // Compress token into low-rank KV
        let kv_entry = mla.compress_kv(&hidden);
        assert_eq!(kv_entry.latent_kv.len(), 8);
        assert_eq!(kv_entry.rope_k.len(), 4);

        // Standard MHA would store: 4 heads * 16 dim * 2 = 128 floats
        // MLA stores: 8 + 4 = 12 floats! (10.6x smaller!)
        assert_eq!(kv_entry.latent_kv.len() + kv_entry.rope_k.len(), 12);

        let kv_cache = vec![kv_entry.clone(), kv_entry];
        let mut out = vec![0.0f32; 32];

        assert!(mla.forward_decode(&hidden, &kv_cache, &mut out).is_ok());
        // Verify non-zero output generated
        let norm: f32 = out.iter().map(|x| x * x).sum();
        assert!(norm > 0.0);
    }
}
