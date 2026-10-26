//! TokenSpeed-MLA: Speed-of-Light Multi-Head Latent Attention Execution Engine.
//!
//! Ported directly from TokenSpeed-MLA architecture:
//! - Optimal Query/Head Tile Folding (`fold_sq_factor`):
//!   Folds query tokens into head axis (`H_eff = num_heads * F`, `q_eff = q_len / F`)
//!   when `num_heads * F <= 128` to saturate CTA tiles in agentic workloads.
//! - Continuous Query Packing:
//!   Linearly packs row indices as `query_token * num_heads + head` to minimize padding.
//! - Two-Stage Split-KV Reduction with Dynamic Workspace:
//!   Splits deep context sequences across parallel threadblocks with partial LSE
//!   (Log-Sum-Exp) generation, followed by a tree reduction kernel.

use oxide_core::error::{EngineError, Result};
use std::fmt;

/// Maximum effective heads per CTA tile.
pub const MLA_MAX_TILE_HEADS: usize = 128;

/// Configuration for TokenSpeed Multi-Head Latent Attention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpeedMlaConfig {
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
}

impl Default for TokenSpeedMlaConfig {
    fn default() -> Self {
        // DeepSeek-V2 / V3 / Kimi MLA typical dimensions
        Self {
            num_heads: 64,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            max_seq_len: 131_072,
        }
    }
}

impl TokenSpeedMlaConfig {
    #[must_use]
    pub const fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    #[must_use]
    pub const fn latent_kv_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }
}

/// Geometry optimizer for MLA decode tiles and split-KV layouts.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlaTileOptimizer;

impl MlaTileOptimizer {
    /// Computes TokenSpeed's optimal query folding factor `(F, h_eff, q_eff)`.
    ///
    /// Finds the largest integer factor `F >= 1` such that:
    /// `q_len % F == 0` and `num_heads * F <= 128`.
    #[must_use]
    pub fn compute_fold_sq_factor(q_len: usize, num_heads: usize) -> (usize, usize, usize) {
        if q_len == 0 || num_heads == 0 {
            return (1, num_heads, q_len);
        }

        let mut best_f = 1;
        // Search factors of q_len
        for f in 1..=q_len {
            if q_len.is_multiple_of(f) && num_heads * f <= MLA_MAX_TILE_HEADS {
                best_f = f;
            }
        }

        let h_eff = num_heads * best_f;
        let q_eff = q_len / best_f;
        (best_f, h_eff, q_eff)
    }

    /// Computes the number of CTA tiles needed under continuous query packing.
    ///
    /// Packed rows = `q_len * num_heads`.
    /// Tile count = `ceil((q_len * num_heads) / 128)`.
    #[must_use]
    pub fn compute_packed_query_tiles(q_len: usize, num_heads: usize) -> usize {
        let total_rows = q_len * num_heads;
        if total_rows == 0 {
            return 0;
        }
        (total_rows + MLA_MAX_TILE_HEADS - 1) / MLA_MAX_TILE_HEADS
    }

    /// Sizing heuristic for split-KV parallel reduction workspace.
    ///
    /// Formula from TokenSpeed:
    /// `workspace_bytes = batch_size * 128 * ceil(H * q_len / 128) * split_kv * (v_head_dim + 1) * 4`
    /// where the `+ 1` element accommodates the partial Log-Sum-Exp (LSE) accumulator.
    #[must_use]
    pub fn calculate_split_kv_workspace_bytes(
        batch_size: usize,
        num_heads: usize,
        q_len: usize,
        split_kv: usize,
        v_head_dim: usize,
    ) -> usize {
        if split_kv <= 1 {
            return 0;
        }
        let tiles = Self::compute_packed_query_tiles(q_len, num_heads);
        batch_size * MLA_MAX_TILE_HEADS * tiles * split_kv * (v_head_dim + 1) * 4
    }

    /// Recommends an optimal split-KV factor for a given context sequence length.
    #[must_use]
    pub fn recommend_split_kv(kv_seq_len: usize) -> usize {
        if kv_seq_len < 4096 {
            1
        } else if kv_seq_len < 16384 {
            4
        } else if kv_seq_len < 65536 {
            8
        } else {
            16
        }
    }
}

/// TokenSpeed Multi-Head Latent Attention decode and reduction engine.
#[derive(Debug, Clone)]
pub struct TokenSpeedMlaEngine {
    config: TokenSpeedMlaConfig,
}

/// Borrowed input buffers for Split-KV MLA decode execution.
#[derive(Debug, Clone)]
pub struct TokenSpeedMlaInputs<'a> {
    pub q_nope: &'a [f32],
    pub q_rope: &'a [f32],
    pub w_uk: &'a [f32],
    pub w_uv: &'a [f32],
    pub latent_kv_cache: &'a [f32],
    pub seq_len: usize,
}

impl TokenSpeedMlaEngine {
    #[must_use]
    pub fn new(config: TokenSpeedMlaConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &TokenSpeedMlaConfig {
        &self.config
    }

    /// Executes two-stage Split-KV MLA decode over compressed latent KV cache.
    ///
    /// Latent KV cache stores concatenated `[kv_lora_cmpr, k_rope]` of dimension
    /// `kv_lora_rank + qk_rope_head_dim`.
    pub fn forward_decode_split_kv(
        &self,
        inputs: &TokenSpeedMlaInputs<'_>,
        out: &mut [f32],
    ) -> Result<()> {
        let num_heads = self.config.num_heads;
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let qk_rope_dim = self.config.qk_rope_head_dim;
        let v_dim = self.config.v_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;
        let latent_dim = self.config.latent_kv_dim();
        let seq_len = inputs.seq_len;

        if inputs.q_nope.len() < num_heads * qk_nope_dim
            || inputs.q_rope.len() < num_heads * qk_rope_dim
            || out.len() < num_heads * v_dim
            || inputs.latent_kv_cache.len() < seq_len * latent_dim
        {
            return Err(EngineError::ShapeMismatch);
        }

        let q_nope = inputs.q_nope;
        let q_rope = inputs.q_rope;
        let w_uk = inputs.w_uk;
        let w_uv = inputs.w_uv;
        let latent_kv_cache = inputs.latent_kv_cache;

        if seq_len == 0 {
            out.fill(0.0);
            return Ok(());
        }

        // Determine split-KV chunks
        let split_kv = MlaTileOptimizer::recommend_split_kv(seq_len);
        let chunk_size = (seq_len + split_kv - 1) / split_kv;
        let scale = 1.0 / ((qk_nope_dim + qk_rope_dim) as f32).sqrt();

        // Workspace buffers storing partial outputs and partial LSE per chunk
        let mut partial_outputs = vec![0.0f32; split_kv * num_heads * v_dim];
        let mut partial_lse = vec![f32::NEG_INFINITY; split_kv * num_heads];

        // Stage 1: Parallel Split-KV partial execution
        for chunk_idx in 0..split_kv {
            let chunk_start = chunk_idx * chunk_size;
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            if chunk_start >= chunk_end {
                continue;
            }

            let chunk_len = chunk_end - chunk_start;

            for h in 0..num_heads {
                let qn_head = &q_nope[h * qk_nope_dim..(h + 1) * qk_nope_dim];
                let qr_head = &q_rope[h * qk_rope_dim..(h + 1) * qk_rope_dim];
                let w_uk_head =
                    &w_uk[h * qk_nope_dim * kv_lora_rank..(h + 1) * qk_nope_dim * kv_lora_rank];
                let w_uv_head = &w_uv[h * v_dim * kv_lora_rank..(h + 1) * v_dim * kv_lora_rank];

                // Compute absorbed Query for the low-rank latent KV:
                // q_absorbed[r] = sum_d (qn_head[d] * w_uk_head[d, r])
                let mut q_absorbed = vec![0.0f32; kv_lora_rank];
                for r in 0..kv_lora_rank {
                    let mut sum = 0.0f32;
                    for d in 0..qk_nope_dim {
                        sum += qn_head[d] * w_uk_head[d * kv_lora_rank + r];
                    }
                    q_absorbed[r] = sum;
                }

                // Compute scores for tokens in this chunk
                let mut scores = Vec::with_capacity(chunk_len);
                let mut chunk_max = f32::NEG_INFINITY;

                for pos in chunk_start..chunk_end {
                    let latent_token = &latent_kv_cache[pos * latent_dim..(pos + 1) * latent_dim];
                    let c_latent = &latent_token[0..kv_lora_rank];
                    let k_rope = &latent_token[kv_lora_rank..kv_lora_rank + qk_rope_dim];

                    // Score = (q_absorbed . c_latent + q_rope . k_rope) * scale
                    let mut dot_latent = 0.0f32;
                    for r in 0..kv_lora_rank {
                        dot_latent += q_absorbed[r] * c_latent[r];
                    }

                    let mut dot_rope = 0.0f32;
                    for d in 0..qk_rope_dim {
                        dot_rope += qr_head[d] * k_rope[d];
                    }

                    let score = (dot_latent + dot_rope) * scale;
                    if score > chunk_max {
                        chunk_max = score;
                    }
                    scores.push(score);
                }

                // Chunk local softmax
                let mut chunk_sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - chunk_max).exp();
                    chunk_sum_exp += *s;
                }

                // Record local LSE: chunk_max + ln(chunk_sum_exp)
                let local_lse = chunk_max + chunk_sum_exp.max(1e-12).ln();
                partial_lse[chunk_idx * num_heads + h] = local_lse;

                // Accumulate partial Value output for this chunk:
                // First accumulate weighted c_latent: v_latent_acc[r] = sum_t (norm_weight * c_latent[r])
                let mut v_latent_acc = vec![0.0f32; kv_lora_rank];
                let inv_chunk_sum = if chunk_sum_exp > 0.0 {
                    1.0 / chunk_sum_exp
                } else {
                    0.0
                };

                for (idx, pos) in (chunk_start..chunk_end).enumerate() {
                    let weight = scores[idx] * inv_chunk_sum;
                    let latent_token =
                        &latent_kv_cache[pos * latent_dim..pos * latent_dim + kv_lora_rank];
                    for r in 0..kv_lora_rank {
                        v_latent_acc[r] += weight * latent_token[r];
                    }
                }

                // Project through W_uv to produce head output: out_val[d] = sum_r (w_uv[d, r] * v_latent_acc[r])
                let partial_out_offset = (chunk_idx * num_heads + h) * v_dim;
                for d in 0..v_dim {
                    let mut sum_v = 0.0f32;
                    for r in 0..kv_lora_rank {
                        sum_v += w_uv_head[d * kv_lora_rank + r] * v_latent_acc[r];
                    }
                    partial_outputs[partial_out_offset + d] = sum_v;
                }
            }
        }

        // Stage 2: Tree Reduction across Split-KV chunks
        out.fill(0.0);
        for h in 0..num_heads {
            // Find global max LSE across valid chunks
            let mut global_max_lse = f32::NEG_INFINITY;
            for chunk_idx in 0..split_kv {
                let lse = partial_lse[chunk_idx * num_heads + h];
                if lse > global_max_lse {
                    global_max_lse = lse;
                }
            }

            if global_max_lse.is_infinite() {
                continue;
            }

            // Sum exp(lse - global_max)
            let mut sum_chunk_weights = 0.0f32;
            let mut chunk_weights = vec![0.0f32; split_kv];
            for chunk_idx in 0..split_kv {
                let lse = partial_lse[chunk_idx * num_heads + h];
                if !lse.is_infinite() {
                    let w = (lse - global_max_lse).exp();
                    chunk_weights[chunk_idx] = w;
                    sum_chunk_weights += w;
                }
            }

            let inv_sum_weights = if sum_chunk_weights > 0.0 {
                1.0 / sum_chunk_weights
            } else {
                0.0
            };

            // Combine partial outputs using normalized chunk weights
            let head_out = &mut out[h * v_dim..(h + 1) * v_dim];
            for chunk_idx in 0..split_kv {
                let norm_w = chunk_weights[chunk_idx] * inv_sum_weights;
                if norm_w > 0.0 {
                    let partial_out_offset = (chunk_idx * num_heads + h) * v_dim;
                    let partial_out =
                        &partial_outputs[partial_out_offset..partial_out_offset + v_dim];
                    for d in 0..v_dim {
                        head_out[d] += norm_w * partial_out[d];
                    }
                }
            }
        }

        Ok(())
    }

    /// Projects uncompressed query Nope vector into absorbed latent space:
    /// `q_absorbed[r] = sum_d (q_nope[d] * W_UK[r, d])`.
    pub fn compute_absorbed_query(
        &self,
        q_nope_head: &[f32],
        w_uk_head: &[f32],
        q_absorbed: &mut [f32],
    ) {
        let qk_nope_dim = self.config.qk_nope_head_dim;
        let kv_lora_rank = self.config.kv_lora_rank;
        for r in 0..kv_lora_rank {
            let mut acc = 0.0f32;
            for d in 0..qk_nope_dim {
                acc += q_nope_head[d] * w_uk_head[r * qk_nope_dim + d];
            }
            q_absorbed[r] = acc;
        }
    }
}

impl fmt::Display for TokenSpeedMlaEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TokenSpeedMlaEngine(heads={}, qk_dim={}, v_dim={}, kv_rank={})",
            self.config.num_heads,
            self.config.qk_head_dim(),
            self.config.v_head_dim,
            self.config.kv_lora_rank
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenspeed_fold_sq_factor_optimization() {
        // q_len = 4, num_heads = 64:
        // F = 2 since 64 * 2 = 128 <= 128 and 4 % 2 == 0.
        // H_eff = 128, q_eff = 2.
        let (f, h_eff, q_eff) = MlaTileOptimizer::compute_fold_sq_factor(4, 64);
        assert_eq!(f, 2);
        assert_eq!(h_eff, 128);
        assert_eq!(q_eff, 2);

        // q_len = 1, num_heads = 64:
        // F = 1 since q_len = 1.
        let (f, h_eff, q_eff) = MlaTileOptimizer::compute_fold_sq_factor(1, 64);
        assert_eq!(f, 1);
        assert_eq!(h_eff, 64);
        assert_eq!(q_eff, 1);

        // q_len = 8, num_heads = 32:
        // 32 * 4 = 128 <= 128 and 8 % 4 == 0 -> F = 4.
        let (f, h_eff, q_eff) = MlaTileOptimizer::compute_fold_sq_factor(8, 32);
        assert_eq!(f, 4);
        assert_eq!(h_eff, 128);
        assert_eq!(q_eff, 2);
    }

    #[test]
    fn test_tokenspeed_workspace_sizing() {
        // batch=1, heads=64, q_len=4, split_kv=4, v_dim=128
        // tiles = ceil(64 * 4 / 128) = 2
        // bytes = 1 * 128 * 2 * 4 * (128 + 1) * 4 = 128 * 2 * 4 * 129 * 4 = 528384 bytes
        let bytes = MlaTileOptimizer::calculate_split_kv_workspace_bytes(1, 64, 4, 4, 128);
        assert_eq!(bytes, 528_384);

        // split_kv = 1 requires zero workspace
        let zero_bytes = MlaTileOptimizer::calculate_split_kv_workspace_bytes(1, 64, 4, 1, 128);
        assert_eq!(zero_bytes, 0);
    }

    #[test]
    fn test_tokenspeed_mla_forward_decode_split_kv() {
        let config = TokenSpeedMlaConfig {
            num_heads: 2,
            q_lora_rank: 8,
            kv_lora_rank: 4,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 2,
            v_head_dim: 4,
            max_seq_len: 2048,
        };

        let engine = TokenSpeedMlaEngine::new(config.clone());

        let q_nope = vec![1.0f32; 2 * 4];
        let q_rope = vec![0.5f32; 2 * 2];
        let w_uk = vec![0.1f32; 2 * 4 * 4];
        let w_uv = vec![0.2f32; 2 * 4 * 4];

        let seq_len = 16;
        let latent_dim = 4 + 2; // kv_rank + rope_dim
        let latent_kv_cache = vec![0.5f32; seq_len * latent_dim];

        let mut out = vec![0.0f32; 2 * 4];

        let inputs = TokenSpeedMlaInputs {
            q_nope: &q_nope,
            q_rope: &q_rope,
            w_uk: &w_uk,
            w_uv: &w_uv,
            latent_kv_cache: &latent_kv_cache,
            seq_len,
        };

        engine
            .forward_decode_split_kv(&inputs, &mut out)
            .expect("forward decode");

        for val in &out {
            assert!(*val > 0.0);
        }
    }
}
