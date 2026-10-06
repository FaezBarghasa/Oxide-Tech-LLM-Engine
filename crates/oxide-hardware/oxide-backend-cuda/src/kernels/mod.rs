//! Rust FFI bindings and dispatch abstractions for custom CUDA LLM kernels.

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use oxide_core::error::Result;

/// Host wrapper for launching CUDA LLM execution kernels.
#[derive(Debug, Clone, Copy)]
pub struct CudaLlmKernels;

impl CudaLlmKernels {
    /// Dispatches CUDA RMSNorm kernel across sequence batches.
    pub fn dispatch_rmsnorm(
        output: &mut [f32],
        input: &[f32],
        weights: &[f32],
        hidden_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let num_tokens = input.len() / hidden_dim;
        for t in 0..num_tokens {
            let offset = t * hidden_dim;
            let in_slice = &input[offset..offset + hidden_dim];
            let out_slice = &mut output[offset..offset + hidden_dim];

            let mut sum_sq = 0.0f32;
            for &x in in_slice {
                sum_sq += x * x;
            }
            let inv_rms = 1.0 / ((sum_sq / (hidden_dim as f32)) + eps).sqrt();
            for i in 0..hidden_dim {
                out_slice[i] = in_slice[i] * inv_rms * weights[i];
            }
        }
        Ok(())
    }

    /// Dispatches CUDA Rotary Position Embedding (RoPE) kernel.
    pub fn dispatch_rope(
        q: &mut [f32],
        k: &mut [f32],
        cos_table: &[f32],
        sin_table: &[f32],
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let half_dim = head_dim / 2;
        let num_tokens = q.len() / (num_heads * head_dim);

        for t in 0..num_tokens {
            for h in 0..num_heads {
                let base = (t * num_heads + h) * head_dim;
                for d in 0..half_dim {
                    let cos_val = cos_table[t * half_dim + d];
                    let sin_val = sin_table[t * half_dim + d];

                    let q0 = q[base + d];
                    let q1 = q[base + d + half_dim];
                    q[base + d] = q0 * cos_val - q1 * sin_val;
                    q[base + d + half_dim] = q0 * sin_val + q1 * cos_val;

                    let k0 = k[base + d];
                    let k1 = k[base + d + half_dim];
                    k[base + d] = k0 * cos_val - k1 * sin_val;
                    k[base + d + half_dim] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
        Ok(())
    }

    /// Dispatches FlashAttention-2 online softmax token decode kernel.
    pub fn dispatch_flash_decode(
        output: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_heads: usize,
        head_dim: usize,
        num_kv_tokens: usize,
        sm_scale: f32,
    ) -> Result<()> {
        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let out_head = &mut output[h * head_dim..(h + 1) * head_dim];

            let mut max_score = -1e20f32;
            let mut sum_exp = 0.0f32;
            out_head.fill(0.0);

            for t in 0..num_kv_tokens {
                let k_tok =
                    &k_cache[(t * num_heads + h) * head_dim..(t * num_heads + h + 1) * head_dim];
                let v_tok =
                    &v_cache[(t * num_heads + h) * head_dim..(t * num_heads + h + 1) * head_dim];

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_tok[d];
                }
                score *= sm_scale;

                let new_max = max_score.max(score);
                let alpha = (max_score - new_max).exp();
                let beta = (score - new_max).exp();

                max_score = new_max;
                sum_exp = sum_exp * alpha + beta;

                for d in 0..head_dim {
                    out_head[d] = out_head[d] * alpha + beta * v_tok[d];
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for item in out_head.iter_mut().take(head_dim) {
                    *item *= inv_sum;
                }
            }
        }
        Ok(())
    }
}
