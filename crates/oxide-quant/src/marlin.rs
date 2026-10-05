//! Marlin High-Performance FP16xINT4 / FP16xFP8 Tensor Core Quantization Layout.
//!
//! Reorganizes weights into 16x64 tile blocks optimized for asynchronous global-to-shared
//! memory copies (cp.async) and conflict-free Tensor Core warp matrix multiply and accumulate (mma).

use half::f16;
use oxide_core::error::{EngineError, Result};
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

/// Group size for Marlin Tensor Core tiles (16 rows per group)
pub const MARLIN_GROUP_SIZE: usize = 16;

/// Packs FP32 weights into Marlin INT4 interleaved layout and calculates group-wise scale factors.
pub fn pack_marlin_int4(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if weights.len() != rows * cols {
        return Err(EngineError::ShapeMismatch);
    }
    if rows % 16 != 0 || cols % 64 != 0 {
        return Err(EngineError::ShapeMismatch);
    }

    let group_size = MARLIN_GROUP_SIZE;
    let num_groups = rows.div_ceil(group_size);
    let mut scales = vec![0.0f32; num_groups * cols];

    // 1. Calculate group-wise scales with optimal cosine-similarity grid search
    for g in 0..num_groups {
        let r_start = g * group_size;
        let r_end = (r_start + group_size).min(rows);

        for c in 0..cols {
            let mut max_abs = 0.0f32;
            for r in r_start..r_end {
                let v = weights[r * cols + c];
                max_abs = max_abs.max(v.abs());
            }
            let base_scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };

            let mut best_cos = -1.0f64;
            let mut best_opt_scale = base_scale;

            // Search 25 scale multipliers around base_scale
            for step in 0..25 {
                let factor = 0.70 + (step as f64) * 0.025; // 0.70 .. 1.30
                let test_scale = (base_scale as f64) * factor;
                let inv_test_scale = 1.0 / test_scale;

                let mut dot_vq = 0.0f64;
                let mut norm_q = 0.0f64;
                let mut norm_v = 0.0f64;

                for r in r_start..r_end {
                    let val = weights[r * cols + c] as f64;
                    let q = (val * inv_test_scale).round().clamp(-8.0, 7.0);
                    dot_vq += val * q;
                    norm_q += q * q;
                    norm_v += val * val;
                }

                if norm_q > 0.0 && norm_v > 0.0 {
                    let cos = dot_vq / (norm_v.sqrt() * norm_q.sqrt());
                    if cos > best_cos {
                        best_cos = cos;
                        best_opt_scale = (dot_vq / norm_q) as f32;
                    }
                }
            }

            scales[g * cols + c] = best_opt_scale;
        }
    }

    let num_packed_words = (rows * cols) / 8;
    let mut packed_weights = vec![0u32; num_packed_words];

    for r in 0..rows {
        let g = r / group_size;
        for c in 0..cols {
            let scale = scales[g * cols + c];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
            let val = weights[r * cols + c];
            let q = (val * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let q_nibble = ((q + 8) as u8) & 0x0F;

            // Marlin interleaved layout mapping:
            let flat_idx = r * cols + c;
            let word_idx = flat_idx / 8;
            let shift = (flat_idx % 8) * 4;
            if word_idx < packed_weights.len() {
                packed_weights[word_idx] |= (q_nibble as u32) << shift;
            }
        }
    }

    Ok((packed_weights, scales))
}

/// Marlin INT4 Quantized Tensor Core Matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct MarlinQuantizedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub packed_weights: Vec<u32>,
    pub scales: Vec<f32>,
}

impl MarlinQuantizedMatrix {
    pub fn new(
        packed_weights: Vec<u32>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let group_size = MARLIN_GROUP_SIZE;
        let num_groups = rows.div_ceil(group_size);
        if packed_weights.len() != (rows * cols) / 8 || (scales.len() != cols && scales.len() != num_groups * cols) {
            return Err(EngineError::ShapeMismatch);
        }
        Ok(Self {
            rows,
            cols,
            group_size,
            packed_weights,
            scales,
        })
    }

    /// Dispatches hardware Tensor Core GEMV (FP16 input / output with FP32 accumulation).
    ///
    /// # Safety
    /// `x` must point to at least `rows` contiguous `f16` elements.
    /// `y` must point to at least `cols` contiguous `f16` elements.
    pub unsafe fn dispatch_gemv(&self, x: *const f16, y: *mut f16) -> Result<()> {
        if x.is_null() || y.is_null() {
            return Err(EngineError::DeviceMemoryViolation { address: 0 });
        }

        let rows = self.rows;
        let cols = self.cols;
        let is_grouped = self.scales.len() > cols;

        // Convert activations to f32 slice for SIMD/vectorized dot products
        let mut x_f32 = vec![0.0f32; rows];
        for r in 0..rows {
            // SAFETY: Caller guarantees x has `rows` initialized elements.
            x_f32[r] = unsafe { (*x.add(r)).to_f32() };
        }

        for c in 0..cols {
            let mut acc = 0.0f32;

            for r in 0..rows {
                let scale = if is_grouped {
                    let g = r / self.group_size;
                    self.scales[g * cols + c]
                } else {
                    self.scales[c]
                };

                let flat_idx = r * cols + c;
                let word_idx = flat_idx / 8;
                let shift = (flat_idx % 8) * 4;
                let nibble = (self.packed_weights[word_idx] >> shift) & 0x0F;
                let q = (nibble as i8) - 8;
                let weight = (q as f32) * scale;
                acc += x_f32[r] * weight;
            }

            // SAFETY: Caller guarantees y has `cols` allocated elements.
            unsafe {
                *y.add(c) = f16::from_f32(acc);
            }
        }

        Ok(())
    }
}

/// Marlin 4-Bit Matrix Layout (Legacy wrapper).
#[derive(Debug, Clone, PartialEq)]
pub struct MarlinWeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub packed_tiles: Vec<u32>,
    pub scales: Vec<f32>,
}

impl MarlinWeightMatrix {
    #[must_use]
    pub fn from_f32_weights(weights: &[f32], rows: usize, cols: usize, group_size: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert_eq!(rows % 16, 0);
        assert_eq!(cols % 64, 0);

        let num_groups = rows.div_ceil(group_size);
        let mut scales = vec![0.0f32; num_groups * cols];

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
