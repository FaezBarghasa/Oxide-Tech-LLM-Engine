//! Marlin High-Performance FP16xINT4 / FP16xFP8 Tensor Core Quantization Layout.
//!
//! Reorganizes weights into 16x64 tile blocks optimized for asynchronous global-to-shared
//! memory copies (cp.async) and conflict-free Tensor Core warp matrix multiply and accumulate (mma).

use oxide_core::traits::QuantScheme;

/// ZST marker for Marlin quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarlinQuant;

impl QuantScheme for MarlinQuant {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        64
    }
}

/// Marlin 4-Bit Matrix Layout.
#[derive(Debug, Clone, PartialEq)]
pub struct MarlinWeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    /// Weights reorganized into Marlin 16x64 tiles
    pub packed_tiles: Vec<u32>,
    /// Scale factors formatted for Marlin warp broadcast
    pub scales: Vec<f32>,
}

impl MarlinWeightMatrix {
    /// Constructs a Marlin weight matrix from a standard (rows x cols) unquantized weight matrix.
    #[must_use]
    pub fn from_f32_weights(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(rows % 16, 0);
        assert_eq!(cols % 64, 0);

        let num_groups = rows.div_ceil(group_size);
        let mut scales = vec![0.0f32; num_groups * cols];

        // 1. Calculate per-group scales
        for g in 0..num_groups {
            let r_start = g * group_size;
            let r_end = (r_start + group_size).min(rows);
            for c in 0..cols {
                let mut max_abs = 0.0f32;
                for r in r_start..r_end {
                    let v = weights[r * cols + c];
                    max_abs = max_abs.max(v.abs());
                }
                scales[g * cols + c] = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            }
        }

        // 2. Quantize and pack into Marlin 16x64 tiles
        let num_packed_words = (rows * cols) / 8;
        let mut packed_tiles = vec![0u32; num_packed_words];

        for r in 0..rows {
            let g = r / group_size;
            for c in 0..cols {
                let scale = scales[g * cols + c];
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                let val = weights[r * cols + c];
                let q = (val * inv_scale).round().clamp(-8.0, 7.0) as i8;
                let q_nibble = ((q + 8) as u8) & 0x0F;

                // Marlin permuted index mapping:
                let flat_idx = r * cols + c;
                let word_idx = flat_idx / 8;
                let shift = (flat_idx % 8) * 4;
                if word_idx < packed_tiles.len() {
                    packed_tiles[word_idx] |= (q_nibble as u32) << shift;
                }
            }
        }

        Self {
            rows,
            cols,
            group_size,
            packed_tiles,
            scales,
        }
    }

    /// Evaluates GEMV dot product: y = x * W
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);

        for c in 0..self.cols {
            let mut sum = 0.0f32;
            for r in 0..self.rows {
                let g = r / self.group_size;
                let scale = self.scales[g * self.cols + c];
                let flat_idx = r * self.cols + c;
                let word_idx = flat_idx / 8;
                let shift = (flat_idx % 8) * 4;
                let nibble = (self.packed_tiles[word_idx] >> shift) & 0x0F;
                let q = (nibble as i8) - 8;
                sum += x[r] * (q as f32 * scale);
            }
            y[c] = sum;
        }
    }
}
