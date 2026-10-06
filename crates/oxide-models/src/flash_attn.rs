#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Flash Attention Engine Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlashAttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub block_size_m: usize, // Query tile size (e.g. 64 or 128)
    pub block_size_n: usize, // Key/Value tile size (e.g. 64 or 128)
    pub is_causal: bool,
    pub softmax_scale: f32,
}

impl FlashAttentionConfig {
    #[must_use]
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, is_causal: bool) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            block_size_m: 64,
            block_size_n: 64,
            is_causal,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

/// Flash Attention Online Softmax State per Query Tile.
#[derive(Debug, Clone)]
struct OnlineSoftmaxTile {
    max_score: f32,
    sum_exp: f32,
}

/// In-Memory / Host Tiled Flash Attention Implementation with Exact Online Softmax.
#[derive(Debug, Clone)]
pub struct FlashAttentionEngine {
    pub config: FlashAttentionConfig,
}

impl FlashAttentionEngine {
    #[must_use]
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Computes Flash Attention forward pass for a single head:
    /// `Q`: shape `[seq_len_q, head_dim]`
    /// `K`: shape `[seq_len_kv, head_dim]`
    /// `V`: shape `[seq_len_kv, head_dim]`
    /// `Output`: shape `[seq_len_q, head_dim]`
    pub fn forward_head(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len_q: usize,
        seq_len_kv: usize,
        output: &mut [f32],
    ) {
        let head_dim = self.config.head_dim;
        assert_eq!(q.len(), seq_len_q * head_dim);
        assert_eq!(k.len(), seq_len_kv * head_dim);
        assert_eq!(v.len(), seq_len_kv * head_dim);
        assert_eq!(output.len(), seq_len_q * head_dim);

        output.fill(0.0);

        let br = self.config.block_size_m;
        let bc = self.config.block_size_n;

        let num_q_blocks = (seq_len_q + br - 1) / br;
        let num_kv_blocks = (seq_len_kv + bc - 1) / bc;

        let mut tile_states = vec![
            OnlineSoftmaxTile {
                max_score: f32::NEG_INFINITY,
                sum_exp: 0.0,
            };
            br
        ];
        let mut s_row = vec![0.0f32; bc];

        for q_blk in 0..num_q_blocks {
            let q_start = q_blk * br;
            let q_end = (q_start + br).min(seq_len_q);
            let q_len = q_end - q_start;

            for state in tile_states.iter_mut().take(q_len) {
                state.max_score = f32::NEG_INFINITY;
                state.sum_exp = 0.0;
            }

            for kv_blk in 0..num_kv_blocks {
                let kv_start = kv_blk * bc;
                let kv_end = (kv_start + bc).min(seq_len_kv);
                let kv_len = kv_end - kv_start;

                for (i, row_state) in tile_states.iter_mut().enumerate().take(q_len) {
                    let global_q_idx = q_start + i;
                    let q_vec = &q[global_q_idx * head_dim..(global_q_idx + 1) * head_dim];

                    let mut tile_local_max = f32::NEG_INFINITY;

                    for j in 0..kv_len {
                        let global_kv_idx = kv_start + j;
                        if self.config.is_causal && global_kv_idx > global_q_idx {
                            s_row[j] = f32::NEG_INFINITY;
                            continue;
                        }

                        let k_vec = &k[global_kv_idx * head_dim..(global_kv_idx + 1) * head_dim];
                        let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum();
                        let score = dot * self.config.softmax_scale;
                        if score > tile_local_max {
                            tile_local_max = score;
                        }
                        s_row[j] = score;
                    }

                    if tile_local_max == f32::NEG_INFINITY {
                        continue;
                    }

                    let new_max = row_state.max_score.max(tile_local_max);
                    let scale_old = (row_state.max_score - new_max).exp();

                    let mut tile_sum_exp = 0.0;
                    for score in &mut s_row[..kv_len] {
                        if *score != f32::NEG_INFINITY {
                            let p = (*score - new_max).exp();
                            *score = p;
                            tile_sum_exp += p;
                        } else {
                            *score = 0.0;
                        }
                    }

                    let out_slice =
                        &mut output[global_q_idx * head_dim..(global_q_idx + 1) * head_dim];

                    // Rescale previous accumulated output
                    if row_state.sum_exp > 0.0 {
                        for val in out_slice.iter_mut() {
                            *val *= scale_old;
                        }
                    }

                    // Accumulate new P * V
                    for (j, &p) in s_row.iter().enumerate().take(kv_len) {
                        if p > 0.0 {
                            let global_kv_idx = kv_start + j;
                            let v_vec =
                                &v[global_kv_idx * head_dim..(global_kv_idx + 1) * head_dim];
                            for d in 0..head_dim {
                                out_slice[d] += p * v_vec[d];
                            }
                        }
                    }

                    row_state.sum_exp = row_state.sum_exp * scale_old + tile_sum_exp;
                    row_state.max_score = new_max;
                }
            }

            // Final normalization by sum_exp
            for (i, row_state) in tile_states.iter().enumerate().take(q_len) {
                let global_q_idx = q_start + i;
                let out_slice = &mut output[global_q_idx * head_dim..(global_q_idx + 1) * head_dim];
                if row_state.sum_exp > 0.0 {
                    let inv_sum = 1.0 / row_state.sum_exp;
                    for val in out_slice.iter_mut() {
                        *val *= inv_sum;
                    }
                }
            }
        }
    }

    /// Computes incremental single-token decoding attention against historical KV cache.
    /// `q_head`: Query vector for this head `[head_dim]`
    /// `k_cache`: All cached keys across layers/tokens `[seq_len, num_kv_heads * head_dim]`
    /// `v_cache`: All cached values across layers/tokens `[seq_len, num_kv_heads * head_dim]`
    /// `kv_head_idx`: Index of the assigned KV head (for GQA)
    /// `num_kv_heads`: Total KV heads in model
    /// `context_len`: Total tokens in cache including current token
    /// `output`: Destination slice for this head `[head_dim]`
    pub fn forward_decode_gqa(
        &self,
        q_head: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        kv_head_idx: usize,
        num_kv_heads: usize,
        context_len: usize,
        output: &mut [f32],
    ) {
        let head_dim = self.config.head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let kv_head_offset = kv_head_idx * head_dim;

        output.fill(0.0);
        if context_len == 0 {
            return;
        }

        let mut max_score = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for t in 0..context_len {
            let t_offset = t * kv_dim + kv_head_offset;
            if t_offset + head_dim > k_cache.len() || t_offset + head_dim > v_cache.len() {
                break;
            }
            let k_vec = &k_cache[t_offset..t_offset + head_dim];
            let v_vec = &v_cache[t_offset..t_offset + head_dim];

            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_vec[d];
            }
            let score = dot * self.config.softmax_scale;

            let new_max = max_score.max(score);
            let alpha = (max_score - new_max).exp();
            let beta = (score - new_max).exp();

            max_score = new_max;
            sum_exp = sum_exp * alpha + beta;

            for d in 0..head_dim {
                output[d] = output[d] * alpha + beta * v_vec[d];
            }
        }

        if sum_exp > 0.0 {
            let inv_sum = 1.0 / sum_exp;
            for val in output.iter_mut() {
                *val *= inv_sum;
            }
        }
    }
}
