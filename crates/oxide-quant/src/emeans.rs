//! Meta FAIR E-Means: High-Performance Vector Quantization & Clustering Engine.
//!
//! Implements the e-means clustering algorithm from Meta FAIR & École des Ponts
//! (*"e-means: outperforming k-means (in one-tenth the epochs)"*, arXiv:2611.09501).
//!
//! Solves the classic vector quantization objective:
//!   \min_{C} \sum_{x \in X} \min_{j \in [k]} \|x - c_j\|^2
//!
//! Unlike Lloyd's algorithm (full-batch updates with slow convergence), e-means
//! updates sufficient statistics (running count $N_j$ and running sum $S_j$) via an
//! Exponential Moving Average (EMA) with mini-batch updates and cosine annealing.
//! It reaches lower MSE in ~1/10th the epochs with 10-45x speedup.

use crate::simd::dot_f32;

/// Configuration parameters for the E-Means clustering optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct EMeansConfig {
    /// Number of clusters (codebook size, e.g. 256 for 8-bit, 16 for 4-bit, 4 for 2-bit).
    pub k: usize,
    /// Vector dimensionality.
    pub dim: usize,
    /// Number of training epochs (e.g. 5 to 10 epochs are sufficient for e-means).
    pub epochs: usize,
    /// Mini-batch size for EMA updates (e.g. 256 or 512).
    pub batch_size: usize,
    /// Initial learning rate / step size alpha_0 (default 1.0).
    pub alpha_0: f32,
    /// Minimum relative learning rate epsilon at final epoch (default 0.1).
    pub epsilon: f32,
}

impl Default for EMeansConfig {
    fn default() -> Self {
        Self {
            k: 256,
            dim: 128,
            epochs: 5,
            batch_size: 256,
            alpha_0: 1.0,
            epsilon: 0.1,
        }
    }
}

/// E-Means Vector Quantization Codebook.
#[derive(Debug, Clone, PartialEq)]
pub struct EMeansCodebook {
    /// Number of clusters.
    pub k: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Centroid vectors stored contiguously: [k * dim].
    pub centroids: Vec<f32>,
    /// Running cluster counts: [k].
    pub counts: Vec<f32>,
    /// Running cluster sums: [k * dim].
    pub sums: Vec<f32>,
    /// Precomputed squared L2 norms of centroids: [k].
    pub centroid_norms_sq: Vec<f32>,
}

impl EMeansCodebook {
    /// Creates an uninitialized codebook for `k` centroids of dimension `dim`.
    #[must_use]
    pub fn new(k: usize, dim: usize) -> Self {
        assert!(k > 0, "k must be positive");
        assert!(dim > 0, "dim must be positive");
        Self {
            k,
            dim,
            centroids: vec![0.0; k * dim],
            counts: vec![1.0; k],
            sums: vec![0.0; k * dim],
            centroid_norms_sq: vec![0.0; k],
        }
    }

    /// Recomputes the squared L2 norms for all centroids.
    pub fn update_centroid_norms(&mut self) {
        for j in 0..self.k {
            let c = &self.centroids[j * self.dim..(j + 1) * self.dim];
            self.centroid_norms_sq[j] = dot_f32(c, c);
        }
    }

    /// Initializes centroids from training data using deterministic uniform striding.
    pub fn init_from_samples(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        assert!(n >= self.k, "Data samples ({n}) must be >= k ({})", self.k);

        let step = n / self.k;
        for j in 0..self.k {
            let sample_idx = (j * step).min(n - 1);
            let src = &data[sample_idx * self.dim..(sample_idx + 1) * self.dim];
            let dst = &mut self.centroids[j * self.dim..(j + 1) * self.dim];
            dst.copy_from_slice(src);
            self.sums[j * self.dim..(j + 1) * self.dim].copy_from_slice(src);
            self.counts[j] = 1.0;
        }
        self.update_centroid_norms();
    }

    /// Finds the nearest centroid for a single vector `x`.
    ///
    /// Computes $\arg\min_j \|x - c_j\|^2 = \arg\max_j (2 x \cdot c_j - \|c_j\|^2)$.
    /// Returns `(nearest_cluster_index, squared_distance)`.
    #[inline]
    #[must_use]
    pub fn find_nearest_centroid(&self, x: &[f32]) -> (usize, f32) {
        debug_assert_eq!(x.len(), self.dim);
        let x_norm_sq = dot_f32(x, x);
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for j in 0..self.k {
            let c = &self.centroids[j * self.dim..(j + 1) * self.dim];
            let dot = dot_f32(x, c);
            // ||x - c||^2 = ||x||^2 - 2 * (x . c) + ||c||^2
            let dist_sq = (x_norm_sq - 2.0 * dot + self.centroid_norms_sq[j]).max(0.0);
            if dist_sq < best_dist {
                best_dist = dist_sq;
                best_idx = j;
            }
        }

        (best_idx, best_dist)
    }

    /// Quantizes a single input vector into its nearest cluster index.
    #[inline]
    #[must_use]
    pub fn quantize(&self, x: &[f32]) -> usize {
        self.find_nearest_centroid(x).0
    }

    /// Dequantizes a cluster index into reconstructed vector values in `out`.
    #[inline]
    pub fn dequantize(&self, cluster_idx: usize, out: &mut [f32]) {
        assert!(cluster_idx < self.k, "Cluster index out of bounds");
        assert!(out.len() >= self.dim, "Output buffer too small");
        let c = &self.centroids[cluster_idx * self.dim..(cluster_idx + 1) * self.dim];
        out[..self.dim].copy_from_slice(c);
    }

    /// Trains the codebook on `data` using Meta FAIR's E-Means algorithm.
    ///
    /// - Mini-batch assignments
    /// - Sufficient statistics EMA updates
    /// - Cosine annealing step size schedule
    pub fn train(&mut self, data: &[f32], config: &EMeansConfig) {
        let n = data.len() / self.dim;
        if n < self.k {
            return;
        }

        self.init_from_samples(data);

        let batch_size = config.batch_size.max(1).min(n);
        let batches_per_epoch = (n + batch_size - 1) / batch_size;
        let total_steps = (config.epochs * batches_per_epoch).max(1);

        let mut step = 0;
        let mut batch_sums = vec![0.0f32; self.k * self.dim];
        let mut batch_counts = vec![0.0f32; self.k];

        for _epoch in 0..config.epochs {
            for b in 0..batches_per_epoch {
                let start_idx = b * batch_size;
                let end_idx = (start_idx + batch_size).min(n);
                let cur_batch_len = end_idx - start_idx;
                if cur_batch_len == 0 {
                    continue;
                }

                // Step size with cosine annealing:
                // alpha(t) = alpha_0 * (eps + (1 - eps) * 0.5 * (1 + cos(pi * t / M)))
                let t_frac = (step as f32) / (total_steps as f32);
                let cosine = 0.5 * (1.0 + (std::f32::consts::PI * t_frac).cos());
                let alpha = config.alpha_0 * (config.epsilon + (1.0 - config.epsilon) * cosine);

                batch_sums.fill(0.0);
                batch_counts.fill(0.0);

                // Assign each vector in mini-batch to nearest centroid
                for i in start_idx..end_idx {
                    let x = &data[i * self.dim..(i + 1) * self.dim];
                    let (cluster_idx, _) = self.find_nearest_centroid(x);
                    batch_counts[cluster_idx] += 1.0;

                    let b_sum =
                        &mut batch_sums[cluster_idx * self.dim..(cluster_idx + 1) * self.dim];
                    for d in 0..self.dim {
                        b_sum[d] += x[d];
                    }
                }

                // EMA update for clusters that observed samples
                let one_minus_alpha = 1.0 - alpha;
                for j in 0..self.k {
                    let n_j = batch_counts[j];
                    if n_j > 0.0 {
                        self.counts[j] = one_minus_alpha * self.counts[j] + alpha * n_j;
                        let inv_count = 1.0 / self.counts[j].max(1e-6);

                        let s_start = j * self.dim;
                        let s_end = s_start + self.dim;
                        let b_s = &batch_sums[s_start..s_end];
                        let sum_j = &mut self.sums[s_start..s_end];
                        let c_j = &mut self.centroids[s_start..s_end];

                        for d in 0..self.dim {
                            sum_j[d] = one_minus_alpha * sum_j[d] + alpha * b_s[d];
                            c_j[d] = sum_j[d] * inv_count;
                        }
                    }
                }

                self.update_centroid_norms();
                step += 1;
            }
        }
    }

    /// Computes the Mean Squared Error (MSE) loss across a dataset.
    #[must_use]
    pub fn compute_mse(&self, data: &[f32]) -> f32 {
        let n = data.len() / self.dim;
        if n == 0 {
            return 0.0;
        }

        let mut total_dist = 0.0f32;
        for i in 0..n {
            let x = &data[i * self.dim..(i + 1) * self.dim];
            let (_, dist_sq) = self.find_nearest_centroid(x);
            total_dist += dist_sq;
        }

        total_dist / (n as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emeans_clustering_convergence() {
        // Create 4 distinct 2D Gaussian clusters:
        // Cluster 0: (-5, -5)
        // Cluster 1: (5, -5)
        // Cluster 2: (-5, 5)
        // Cluster 3: (5, 5)
        let centers = [[-5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [5.0, 5.0]];
        let mut data = Vec::with_capacity(400 * 2);

        for c in &centers {
            for i in 0..100 {
                let jitter_x = (i as f32 % 10.0 - 5.0) * 0.05;
                let jitter_y = (i as f32 / 10.0 - 5.0) * 0.05;
                data.push(c[0] + jitter_x);
                data.push(c[1] + jitter_y);
            }
        }

        let mut codebook = EMeansCodebook::new(4, 2);
        let config = EMeansConfig {
            k: 4,
            dim: 2,
            epochs: 5,
            batch_size: 50,
            alpha_0: 0.9,
            epsilon: 0.1,
        };

        let initial_mse = codebook.compute_mse(&data);
        codebook.train(&data, &config);
        let final_mse = codebook.compute_mse(&data);

        assert!(
            final_mse < initial_mse,
            "E-Means should reduce MSE: final ({final_mse}) < initial ({initial_mse})"
        );
        assert!(
            final_mse < 1.0,
            "E-Means should accurately cluster tightly grouped data, got MSE {final_mse}"
        );

        // Test dequantization
        let mut reconstructed = [0.0f32; 2];
        for (idx, c) in centers.iter().enumerate() {
            let q = codebook.quantize(c);
            codebook.dequantize(q, &mut reconstructed);
            let dist = ((reconstructed[0] - c[0]).powi(2) + (reconstructed[1] - c[1]).powi(2)).sqrt();
            assert!(
                dist < 0.5,
                "Center {idx} reconstruction error {dist} should be < 0.5"
            );
        }
    }
}
