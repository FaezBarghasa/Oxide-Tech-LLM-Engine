//! BitsAndBytes Quantization Formats (NF4 and FP4 from QLoRA).
//!
//! Provides the NormalFloat4 (NF4) data type, an information-theoretically optimal
//! quantile quantization for zero-mean, unit-variance normally distributed weights.

use oxide_core::traits::QuantScheme;

/// NormalFloat4 (NF4) 16 empirical normal distribution quantiles.
pub const NF4_TABLE: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.3949175,
    -0.2844414,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070783,
    0.562617,
    0.72295684,
    1.0,
];

/// NormalFloat 4 (NF4) marker and block container (64 weights per block with FP32/FP16 scale).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockNf4_64 {
    pub scale: f32,
    pub qs: [u8; 32], // 64 nibbles = 32 bytes
}

impl QuantScheme for BlockNf4_64 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5 // 4 bits + scale overhead
    }

    #[inline(always)]
    fn block_size() -> usize {
        64
    }
}

impl BlockNf4_64 {
    #[inline(always)]
    fn quantize_nf4(val: f32) -> u8 {
        let mut best_idx = 0u8;
        let mut min_diff = f32::MAX;
        for (i, &cand) in NF4_TABLE.iter().enumerate() {
            let diff = (val - cand).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i as u8;
            }
        }
        best_idx
    }

    #[must_use]
    pub fn quantize(input: &[f32; 64]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        let inv_scale = 1.0 / scale;

        let mut qs = [0u8; 32];
        for i in 0..32 {
            let q0 = Self::quantize_nf4(input[i] * inv_scale);
            let q1 = Self::quantize_nf4(input[i + 32] * inv_scale);
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self { scale, qs }
    }

    pub fn dequantize(&self, output: &mut [f32; 64]) {
        let s = self.scale;
        for i in 0..32 {
            let byte = self.qs[i];
            let q0 = (byte & 0x0F) as usize;
            let q1 = ((byte >> 4) & 0x0F) as usize;
            output[i] = NF4_TABLE[q0] * s;
            output[i + 32] = NF4_TABLE[q1] * s;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 64]) -> f32 {
        let mut deq = [0.0f32; 64];
        self.dequantize(&mut deq);
        let mut sum = 0.0f32;
        for i in 0..64 {
            sum += deq[i] * activations[i];
        }
        sum
    }
}
