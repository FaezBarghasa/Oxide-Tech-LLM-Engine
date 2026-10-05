//! OCP Microscaling (MX) Formats Specification v1.0.
//!
//! Provides hardware-efficient microscopic block-floating-point representations:
//! - MXFP8 (E4M3 / E5M2 with 8-bit shared scale per 32 elements, 8.25 bpw)
//! - MXFP4 (E2M1 with 8-bit shared scale per 32 elements, 4.25 bpw)
//! - MXFP6 (E3M2 / E2M3 with 8-bit shared scale per 32 elements, 6.25 bpw)
//! - MXINT8 (Signed int8 with 8-bit shared scale per 32 elements, 8.25 bpw)

use crate::fp8::Fp8E4M3;
use oxide_core::traits::QuantScheme;

/// E8M0 8-bit Scale Factor.
///
/// Encodes an unbiased power of 2: scale = 2^(exponent - 127).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct E8M0Scale(pub u8);

impl E8M0Scale {
    #[must_use]
    pub fn from_f32(scale: f32) -> Self {
        if scale <= 0.0 {
            return Self(0);
        }
        // Round to nearest power of 2
        let exp = scale.log2().round() as i32;
        let biased = (exp + 127).clamp(0, 255) as u8;
        Self(biased)
    }

    #[must_use]
    pub fn to_f32(self) -> f32 {
        if self.0 == 0 {
            return 0.0;
        }
        let exp = self.0 as i32 - 127;
        2.0f32.powi(exp)
    }
}

/// OCP MXFP8: 32 elements of FP8 E4M3 with a shared E8M0 8-bit scale factor (8.25 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMxFp8 {
    pub scale: E8M0Scale,
    pub values: [Fp8E4M3; 32],
}

impl QuantScheme for BlockMxFp8 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.25 // 32 * 8 bits + 8 bits scale = 264 bits / 32 = 8.25 bpw
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

impl BlockMxFp8 {
    #[must_use]
    pub fn quantize(input: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let raw_scale = if max_abs > 0.0 {
            max_abs / Fp8E4M3::MAX_VAL
        } else {
            1.0
        };
        let scale = E8M0Scale::from_f32(raw_scale);
        let eff_scale = scale.to_f32();
        let inv_scale = if eff_scale > 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };

        let mut values = [Fp8E4M3::ZERO; 32];
        for i in 0..32 {
            values[i] = Fp8E4M3::from_f32(input[i] * inv_scale);
        }

        Self { scale, values }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let s = self.scale.to_f32();
        for i in 0..32 {
            output[i] = self.values[i].to_f32() * s;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += self.values[i].to_f32() * activations[i];
        }
        sum * self.scale.to_f32()
    }
}

/// OCP MXFP4: 32 elements of E2M1 FP4 with a shared E8M0 8-bit scale factor (4.25 bpw).
///
/// Weights are packed 2 per byte (16 bytes payload + 1 byte scale = 17 bytes).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMxFp4 {
    pub scale: E8M0Scale,
    pub qs: [u8; 16],
}

impl QuantScheme for BlockMxFp4 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.25 // 32 * 4 bits + 8 bits scale = 136 bits / 32 = 4.25 bpw
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// E2M1 lookup table: 4 bits -> float magnitude.
///
/// Bit 3: Sign (0 = positive, 1 = negative)
/// Bits 2..0: Magnitude
/// 000: 0.0
/// 001: 0.5
/// 010: 1.0
/// 011: 1.5
/// 100: 2.0
/// 101: 3.0
/// 110: 4.0
/// 111: 6.0
const E2M1_TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

impl BlockMxFp4 {
    #[inline(always)]
    fn quantize_e2m1(val: f32) -> u8 {
        let sign = if val < 0.0 { 0x08 } else { 0x00 };
        let abs_val = val.abs();

        let mut best_idx = 0u8;
        let mut min_diff = f32::MAX;
        for (i, &cand) in E2M1_TABLE.iter().enumerate() {
            let diff = (abs_val - cand).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i as u8;
            }
        }
        sign | best_idx
    }

    #[inline(always)]
    fn dequantize_e2m1(nibble: u8) -> f32 {
        let mag = E2M1_TABLE[(nibble & 0x07) as usize];
        if (nibble & 0x08) != 0 { -mag } else { mag }
    }

    #[must_use]
    pub fn quantize(input: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let raw_scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
        let scale = E8M0Scale::from_f32(raw_scale);
        let eff_scale = scale.to_f32();
        let inv_scale = if eff_scale > 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };

        let mut qs = [0u8; 16];
        for i in 0..16 {
            let q0 = Self::quantize_e2m1(input[i] * inv_scale);
            let q1 = Self::quantize_e2m1(input[i + 16] * inv_scale);
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self { scale, qs }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let s = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            output[i] = Self::dequantize_e2m1(byte & 0x0F) * s;
            output[i + 16] = Self::dequantize_e2m1((byte >> 4) & 0x0F) * s;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let mut deq = [0.0f32; 32];
        self.dequantize(&mut deq);
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += deq[i] * activations[i];
        }
        sum
    }
}

/// OCP MXFP6: 32 elements of E3M2 FP6 with shared 8-bit scale factor (6.25 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMxFp6 {
    pub scale: E8M0Scale,
    pub packed: [u8; 24], // 32 * 6 bits = 192 bits = 24 bytes
}

impl QuantScheme for BlockMxFp6 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        6.25
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// OCP MXINT8: 32 elements of signed INT8 with shared 8-bit scale factor (8.25 bpw).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockMxInt8 {
    pub scale: E8M0Scale,
    pub values: [i8; 32],
}

impl QuantScheme for BlockMxInt8 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.25
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

impl BlockMxInt8 {
    #[must_use]
    pub fn quantize(input: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let raw_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let scale = E8M0Scale::from_f32(raw_scale);
        let eff_scale = scale.to_f32();
        let inv_scale = if eff_scale > 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };

        let mut values = [0i8; 32];
        for i in 0..32 {
            values[i] = (input[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }

        Self { scale, values }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let s = self.scale.to_f32();
        for i in 0..32 {
            output[i] = (self.values[i] as f32) * s;
        }
    }
}
