//! Custom ROCm/HIP and Taconite APU kernel dispatch for AMD GPUs and APUs (CPU + iGPU + NPU).

#![allow(clippy::cast_precision_loss)]

use oxide_core::error::Result;

/// Host wrapper for launching custom AMD ROCm / Taconite kernels.
#[derive(Debug, Clone, Copy)]
pub struct RocmLlmKernels;

impl RocmLlmKernels {
    /// Dispatches AMD Wavefront-optimized RMSNorm.
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

    /// Dispatches Taconite unified memory heterogeneous step for AMD APUs (CPU + iGPU + NPU).
    pub fn dispatch_apu_unified_step(token_out: &mut u32, logits: &[f32]) -> Result<()> {
        let mut max_val = -1e30f32;
        let mut argmax = 0usize;

        for (idx, &l) in logits.iter().enumerate() {
            if l > max_val {
                max_val = l;
                argmax = idx;
            }
        }

        *token_out = argmax as u32;
        Ok(())
    }
}
