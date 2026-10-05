//! ExLlamaV2 (EXL2) Fractional Bit-Width Quantization Runtime.
//!
//! Implements EXL2 mixed-precision and fractional-bit layouts (2.0bpw to 8.0bpw).
//! In EXL2, matrices are partitioned into sub-blocks with variable bit allocation
//! tailored to error minimization per tensor channel.

use oxide_core::traits::QuantScheme;
use serde::{Deserialize, Serialize};

/// EXL2 Target Bit-rate Profile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Exl2BitsPerWeight {
    Bpw2_0,
    Bpw3_0,
    Bpw3_5,
    Bpw4_0,
    Bpw4_25,
    Bpw5_0,
    Bpw6_0,
    Bpw6_5,
    Bpw8_0,
}

impl Exl2BitsPerWeight {
    #[must_use]
    pub fn bpw(self) -> f32 {
        match self {
            Self::Bpw2_0 => 2.0,
            Self::Bpw3_0 => 3.0,
            Self::Bpw3_5 => 3.5,
            Self::Bpw4_0 => 4.0,
            Self::Bpw4_25 => 4.25,
            Self::Bpw5_0 => 5.0,
            Self::Bpw6_0 => 6.0,
            Self::Bpw6_5 => 6.5,
            Self::Bpw8_0 => 8.0,
        }
    }
}

/// EXL2 Quantized Weight Matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exl2WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub target_bpw: Exl2BitsPerWeight,
    /// Packed quantized bits in 32-bit words
    pub q_data: Vec<u32>,
    /// Scale values per sub-block (typically 32 or 64 elements)
    pub scales: Vec<f32>,
    /// Sub-block size for scale application
    pub sub_block_size: usize,
}

impl Exl2WeightMatrix {
    #[must_use]
    pub fn new(
        rows: usize,
        cols: usize,
        target_bpw: Exl2BitsPerWeight,
        sub_block_size: usize,
    ) -> Self {
        let total_weights = rows * cols;
        let bits = (total_weights as f32 * target_bpw.bpw()).ceil() as usize;
        let words = bits.div_ceil(32);
        let num_scales = total_weights.div_ceil(sub_block_size);

        Self {
            rows,
            cols,
            target_bpw,
            q_data: vec![0u32; words],
            scales: vec![1.0f32; num_scales],
            sub_block_size,
        }
    }

    /// Evaluates GEMV dot product: y = x * W
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);

        // Dequantize on-the-fly and accumulate
        for c in 0..self.cols {
            let mut acc = 0.0f32;
            for r in 0..self.rows {
                let weight_idx = r * self.cols + c;
                let scale_idx = weight_idx / self.sub_block_size;
                let scale = self.scales.get(scale_idx).copied().unwrap_or(1.0);

                let bits_f = self.target_bpw.bpw();
                let bits_u = bits_f.round() as u32;
                let word_idx = ((weight_idx as f32 * bits_f) as usize) / 32;
                let shift = ((weight_idx as f32 * bits_f) as usize) % 32;
                let bits_mask = if bits_u >= 32 {
                    u32::MAX
                } else {
                    (1u32 << bits_u) - 1
                };
                let raw_q = if word_idx < self.q_data.len() {
                    (self.q_data[word_idx] >> shift) & bits_mask
                } else {
                    0
                };
                let w = (raw_q as f32 - (bits_mask as f32 / 2.0)) * scale;
                acc += x[r] * w;
            }
            y[c] = acc;
        }
    }
}

/// ZST marker for EXL2 Quant Scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exl2Quant;

impl QuantScheme for Exl2Quant {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}
