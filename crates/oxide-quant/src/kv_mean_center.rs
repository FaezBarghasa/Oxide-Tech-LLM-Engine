//! Softmax-Invariant KV-Cache Mean Centering (`kv_mean_center`).
//! Subtracts a per-(head, channel) mean bias vector from Key states prior to KV-cache quantization.
//!
//! # Mathematical Principle:
//! For query vector $q \in \mathbb{R}^d$ and key vectors $k_j \in \mathbb{R}^d$:
//! $$q \cdot (k_j - \mu_k) = q \cdot k_j - q \cdot \mu_k$$
//! Because $q \cdot \mu_k$ is constant across all key positions $j \in [1, T]$ for a given query $q$,
//! it cancels out identically in the Softmax normalization:
//! $$\text{Softmax}\left(\frac{q \cdot k_j - c}{\sqrt{d}}\right) = \text{Softmax}\left(\frac{q \cdot k_j}{\sqrt{d}}\right)$$
//! Centering $k_j$ around zero removes the DC bias, substantially compressing activation dynamic range
//! and eliminating quantization truncation error for Q4_0, Q8_0, and PTQ1_0 caches.

/// Per-layer Key mean-centering bias state.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerKeyBias {
    pub n_heads: usize,
    pub head_dim: usize,
    /// Channel biases of shape `[n_heads * head_dim]`.
    pub biases: Vec<f32>,
}

impl LayerKeyBias {
    /// Creates a new zeroed bias vector for a layer.
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self {
            n_heads,
            head_dim,
            biases: vec![0.0f32; n_heads * head_dim],
        }
    }

    /// Creates a bias state from an existing slice.
    pub fn from_slice(n_heads: usize, head_dim: usize, biases: &[f32]) -> Self {
        assert_eq!(n_heads * head_dim, biases.len());
        Self {
            n_heads,
            head_dim,
            biases: biases.to_vec(),
        }
    }

    /// In-place subtraction of the mean bias from a Key token vector of shape `[n_heads * head_dim]`.
    #[inline(always)]
    pub fn apply_in_place(&self, k_token: &mut [f32]) {
        assert_eq!(k_token.len(), self.biases.len());
        for (k, &b) in k_token.iter_mut().zip(self.biases.iter()) {
            *k -= b;
        }
    }

    /// In-place subtraction across multiple tokens `[n_tokens, n_heads * head_dim]`.
    pub fn apply_batch_in_place(&self, k_tokens: &mut [f32], n_tokens: usize) {
        let dim = self.biases.len();
        assert_eq!(k_tokens.len(), n_tokens * dim);
        for token_chunk in k_tokens.chunks_exact_mut(dim) {
            for (k, &b) in token_chunk.iter_mut().zip(self.biases.iter()) {
                *k -= b;
            }
        }
    }
}

/// Dynamic online accumulator to estimate the Key mean bias during calibration passes.
#[derive(Debug, Clone, Default)]
pub struct OnlineMeanCalibrator {
    pub count: usize,
    pub sum: Vec<f64>,
}

impl OnlineMeanCalibrator {
    pub fn new(dim: usize) -> Self {
        Self {
            count: 0,
            sum: vec![0.0; dim],
        }
    }

    /// Observes a token vector and updates running sums.
    pub fn observe(&mut self, k_token: &[f32]) {
        if self.sum.is_empty() {
            self.sum = vec![0.0; k_token.len()];
        }
        assert_eq!(self.sum.len(), k_token.len());
        for (s, &v) in self.sum.iter_mut().zip(k_token.iter()) {
            *s += v as f64;
        }
        self.count += 1;
    }

    /// Finalizes the calibration into a `LayerKeyBias`.
    pub fn finalize(&self, n_heads: usize, head_dim: usize) -> LayerKeyBias {
        assert_eq!(n_heads * head_dim, self.sum.len());
        let inv_count = if self.count > 0 {
            1.0 / (self.count as f64)
        } else {
            0.0
        };

        let biases: Vec<f32> = self.sum.iter().map(|&s| (s * inv_count) as f32).collect();
        LayerKeyBias {
            n_heads,
            head_dim,
            biases,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_invariance() {
        let n_heads = 2;
        let head_dim = 4;
        let dim = n_heads * head_dim;

        let q = [1.0f32, -0.5, 2.0, 0.1, 0.0, 1.2, -1.0, 0.5];
        let mut k = [0.5f32, 1.5, -0.2, 0.8, -1.0, 0.3, 0.9, -0.4];

        let bias = LayerKeyBias::from_slice(
            n_heads,
            head_dim,
            &[0.2, -0.1, 0.5, 0.3, -0.4, 0.1, 0.2, -0.3],
        );

        // Dot product before centering
        let dot_raw: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        // Dot product of q with bias vector
        let dot_bias: f32 = q.iter().zip(bias.biases.iter()).map(|(a, b)| a * b).sum();

        // Apply bias centering
        bias.apply_in_place(&mut k);
        let dot_centered: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        // Verify mathematical equivalence: dot_raw = dot_centered + dot_bias
        let diff = (dot_raw - (dot_centered + dot_bias)).abs();
        assert!(
            diff < 1e-5,
            "Softmax invariance identity violated: diff = {}",
            diff
        );
    }
}
