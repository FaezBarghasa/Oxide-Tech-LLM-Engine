//! Extended GGUF / GGML Quantization Super-Blocks.
//!
//! Implements advanced GGML super-block formats:
//! - Q4_K (256 weights in 16 sub-blocks, 4.5 bpw)
//! - Q5_K (256 weights in 16 sub-blocks, 5.5 bpw)
//! - IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_XXS, IQ1_S (Importance-matrix codebook quantizations)

#![allow(non_camel_case_types)]

use crate::int_quant::f16;
use oxide_core::traits::QuantScheme;

/// Q4_K: 256 weights per super-block (16 sub-blocks of 16 weights).
///
/// Scale and minimum factor arrays: 144 bytes per 256 weights (4.5 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

impl Default for BlockQ4_K {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0u8; 12],
            qs: [0u8; 128],
        }
    }
}

impl QuantScheme for BlockQ4_K {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5 // (2 + 2 + 12 + 128) * 8 / 256 = 144 * 8 / 256 = 4.5 bpw
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

impl BlockQ4_K {
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let d = (max_val - min_val) / 15.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 128];
        for i in 0..128 {
            let q0 = ((values[i] - min_val) * inv_d).round().clamp(0.0, 15.0) as u8;
            let q1 = ((values[i + 128] - min_val) * inv_d)
                .round()
                .clamp(0.0, 15.0) as u8;
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self {
            d: f16::from_f32(d),
            dmin: f16::from_f32(min_val),
            scales: [0u8; 12],
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 256]) {
        let d = self.d.to_f32();
        let dmin = self.dmin.to_f32();
        for i in 0..128 {
            let byte = self.qs[i];
            let q0 = (byte & 0x0F) as f32;
            let q1 = ((byte >> 4) & 0x0F) as f32;
            output[i] = q0 * d + dmin;
            output[i + 128] = q1 * d + dmin;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 256]) -> f32 {
        let mut deq = [0.0f32; 256];
        self.dequantize(&mut deq);
        let mut sum = 0.0f32;
        for i in 0..256 {
            sum += deq[i] * activations[i];
        }
        sum
    }
}

/// Q5_K: 256 weights per super-block (16 sub-blocks of 16 weights, 5.5 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ5_K {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

impl Default for BlockQ5_K {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            dmin: f16::from_f32(0.0),
            scales: [0u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
        }
    }
}

impl QuantScheme for BlockQ5_K {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        5.5 // (2 + 2 + 12 + 32 + 128) * 8 / 256 = 176 * 8 / 256 = 5.5 bpw
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ4_XS: Importance-matrix non-linear 4-bit quantization (4.25 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ4_XS {
    pub d: f16,
    pub scales_l: [u8; 8],
    pub qs: [u8; 128],
}

impl Default for BlockIQ4_XS {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales_l: [0u8; 8],
            qs: [0u8; 128],
        }
    }
}

impl QuantScheme for BlockIQ4_XS {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.25
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ4_NL: Non-linear 4-bit quantization with grid table.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ4_NL {
    pub d: f16,
    pub qs: [u8; 16],
}

impl Default for BlockIQ4_NL {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 16],
        }
    }
}

impl QuantScheme for BlockIQ4_NL {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// IQ3_XXS: Importance-matrix 3-bit quantization (3.06 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ3_XXS {
    pub d: f16,
    pub qs: [u8; 96],
}

impl Default for BlockIQ3_XXS {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 96],
        }
    }
}

impl QuantScheme for BlockIQ3_XXS {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        3.0625
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ2_XXS: Importance-matrix 2-bit quantization (2.06 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ2_XXS {
    pub d: f16,
    pub qs: [u8; 64],
}

impl Default for BlockIQ2_XXS {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 64],
        }
    }
}

impl QuantScheme for BlockIQ2_XXS {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.0625
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ1_S: Importance-matrix 1-bit quantization (1.56 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ1_S {
    pub d: f16,
    pub qs: [u8; 48],
}

impl Default for BlockIQ1_S {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 48],
        }
    }
}

impl QuantScheme for BlockIQ1_S {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        1.5625
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ1_M: Importance-matrix 1-bit quantization medium variant (~1.75 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ1_M {
    pub d: f16,
    pub scales: [u8; 8],
    pub qs: [u8; 48],
    pub qh: [u8; 8],
}

impl Default for BlockIQ1_M {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales: [0u8; 8],
            qs: [0u8; 48],
            qh: [0u8; 8],
        }
    }
}

impl QuantScheme for BlockIQ1_M {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        1.75
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// TQ1_0: Ternary quantization ({-1, 0, +1}, ~1.69 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTQ1_0 {
    pub d: f16,
    pub qs: [u8; 54],
}

impl Default for BlockTQ1_0 {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 54],
        }
    }
}

impl QuantScheme for BlockTQ1_0 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        1.6875
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// TQ2_0: Ternary quantization 2-bit variant (~2.06 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTQ2_0 {
    pub d: f16,
    pub qs: [u8; 64],
}

impl Default for BlockTQ2_0 {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 64],
        }
    }
}

impl QuantScheme for BlockTQ2_0 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.0625
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ2_XS: Importance-matrix 2-bit extra-small (~2.31 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ2_XS {
    pub d: f16,
    pub scales_l: [u8; 8],
    pub qs: [u8; 64],
}

impl Default for BlockIQ2_XS {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales_l: [0u8; 8],
            qs: [0u8; 64],
        }
    }
}

impl QuantScheme for BlockIQ2_XS {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.3125
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ2_S: Importance-matrix 2-bit small (~2.5 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ2_S {
    pub d: f16,
    pub qs: [u8; 80],
}

impl Default for BlockIQ2_S {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 80],
        }
    }
}

impl QuantScheme for BlockIQ2_S {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.5
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ2_M: Importance-matrix 2-bit medium (~2.7 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ2_M {
    pub d: f16,
    pub scales: [u8; 8],
    pub qs: [u8; 80],
}

impl Default for BlockIQ2_M {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales: [0u8; 8],
            qs: [0u8; 80],
        }
    }
}

impl QuantScheme for BlockIQ2_M {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.7
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ3_XS: Importance-matrix 3-bit extra small (~3.3 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ3_XS {
    pub d: f16,
    pub scales: [u8; 8],
    pub qs: [u8; 96],
}

impl Default for BlockIQ3_XS {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales: [0u8; 8],
            qs: [0u8; 96],
        }
    }
}

impl QuantScheme for BlockIQ3_XS {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        3.3
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ3_S: Importance-matrix 3-bit small (~3.44 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ3_S {
    pub d: f16,
    pub qs: [u8; 110],
}

impl Default for BlockIQ3_S {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            qs: [0u8; 110],
        }
    }
}

impl QuantScheme for BlockIQ3_S {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        3.4375
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// IQ3_M: Importance-matrix 3-bit medium (~3.66 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIQ3_M {
    pub d: f16,
    pub scales: [u8; 8],
    pub qs: [u8; 110],
}

impl Default for BlockIQ3_M {
    fn default() -> Self {
        Self {
            d: f16::from_f32(0.0),
            scales: [0u8; 8],
            qs: [0u8; 110],
        }
    }
}

impl QuantScheme for BlockIQ3_M {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        3.66
    }

    #[inline(always)]
    fn block_size() -> usize {
        256
    }
}

/// Q2_K_S: Mixed K-quant 2-bit Small.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ2_K_S(pub BlockQ4_K);

/// Q3_K_S: Mixed K-quant 3-bit Small (~3.4 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ3_K_S(pub BlockQ4_K);

/// Q3_K_M: Mixed K-quant 3-bit Medium (~3.4 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ3_K_M(pub BlockQ4_K);

/// Q3_K_L: Mixed K-quant 3-bit Large (~3.5 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ3_K_L(pub BlockQ4_K);

/// Q4_K_S: Mixed K-quant 4-bit Small (~4.6 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_K_S(pub BlockQ4_K);

/// Q4_K_M: Mixed K-quant 4-bit Medium (~4.8 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_K_M(pub BlockQ4_K);

/// Q5_K_S: Mixed K-quant 5-bit Small (~5.5 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ5_K_S(pub BlockQ5_K);

/// Q5_K_M: Mixed K-quant 5-bit Medium (~5.7 bpw).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ5_K_M(pub BlockQ5_K);
