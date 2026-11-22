//! Mode A: Dequant-Time LoRA Weight Fusion.
//!
//! Fuses fine-tuned weights directly into base dequantized model weights upon load:
//! W_fused = Dequant(W_base) + (alpha / r) * B * A
//!
//! Yields 0.0 ms runtime inference overhead.

use crate::loader::UnslothTensorPair;

impl UnslothTensorPair {
    /// Computes and accumulates the weight delta: Delta_W = (alpha / r) * B * A
    pub fn compute_weight_delta(&self, delta_w: &mut [f32]) {
        assert_eq!(
            delta_w.len(),
            self.out_dim * self.in_dim,
            "delta_w size mismatch"
        );
        delta_w.fill(0.0);

        for o in 0..self.out_dim {
            let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            for (r, &b_elem) in b_row.iter().enumerate() {
                let b_val = b_elem * self.scaling;
                let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
                let out_offset = o * self.in_dim;
                for (i, &a_elem) in a_row.iter().enumerate() {
                    delta_w[out_offset + i] += b_val * a_elem;
                }
            }
        }
    }

    /// Fuses delta directly into target FP32 weight matrix in-place.
    pub fn fuse_into_weights(&self, target_weights: &mut [f32]) {
        assert_eq!(
            target_weights.len(),
            self.out_dim * self.in_dim,
            "target_weights size mismatch"
        );
        for o in 0..self.out_dim {
            let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            for (r, &b_elem) in b_row.iter().enumerate() {
                let b_val = b_elem * self.scaling;
                let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
                let out_offset = o * self.in_dim;
                for (i, &a_elem) in a_row.iter().enumerate() {
                    target_weights[out_offset + i] += b_val * a_elem;
                }
            }
        }
    }
}
