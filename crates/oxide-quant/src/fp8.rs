//! FP8 Quantization Formats (OCP E4M3 and E5M2).
//!
//! Standardized 8-bit floating point representations utilized by NVIDIA Ada Lovelace,
//! Hopper, Blackwell, and AMD CDNA3 architectures.

use oxide_core::traits::QuantScheme;

/// FP8 E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7).
///
/// Dynamic range up to 448.0. Standard format for weights and activations.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp8E4M3(pub u8);

impl QuantScheme for Fp8E4M3 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        1
    }
}

impl Fp8E4M3 {
    pub const ZERO: Self = Self(0);
    pub const MAX_VAL: f32 = 448.0;

    /// Converts standard IEEE 754 f32 into FP8 E4M3.
    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        if val == 0.0 {
            return Self::ZERO;
        }

        let bits = val.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = bits & 0x007F_FFFF;

        let abs_val = val.abs();
        if abs_val > Self::MAX_VAL {
            // Saturate to max representable finite value
            return Self((sign << 7) | 0x7E);
        }

        let target_exp = exp + 7;
        if target_exp <= 0 {
            // Subnormal representation
            let shift = (1 - target_exp) as u32;
            let full_mantissa = (1 << 23) | mantissa;
            let sub_m = full_mantissa >> (20 + shift);
            Self((sign << 7) | (sub_m as u8 & 0x07))
        } else if target_exp >= 15 {
            // Saturate to 0x7E (max non-NaN value)
            Self((sign << 7) | 0x7E)
        } else {
            // Normal number
            let m = (mantissa >> 20) as u8;
            Self((sign << 7) | ((target_exp as u8 & 0x0F) << 3) | (m & 0x07))
        }
    }

    /// Converts FP8 E4M3 into standard IEEE 754 f32.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = (self.0 >> 3) & 0x0F;
        let mantissa = self.0 & 0x07;

        if exp == 0 {
            if mantissa == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal: (-1)^sign * 2^(-6) * (mantissa / 8)
            let sign_factor = if sign == 1 { -1.0f32 } else { 1.0f32 };
            return sign_factor * 0.015625 * (mantissa as f32 / 8.0);
        }

        if exp == 15 && mantissa == 7 {
            // Canonical NaN in E4M3
            return f32::NAN;
        }

        let sign_factor = if sign == 1 { -1.0f32 } else { 1.0f32 };
        let e = exp as i32 - 7;
        let scale = 2.0f32.powi(e);
        let m = 1.0f32 + (mantissa as f32 / 8.0);
        sign_factor * scale * m
    }
}

/// FP8 E5M2 (1 sign bit, 5 exponent bits, 2 mantissa bits, bias = 15).
///
/// Dynamic range up to 57344.0. Standard format for gradients and intermediate activations.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp8E5M2(pub u8);

impl QuantScheme for Fp8E5M2 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        1
    }
}

impl Fp8E5M2 {
    pub const ZERO: Self = Self(0);
    pub const MAX_VAL: f32 = 57344.0;

    /// Converts standard IEEE 754 f32 into FP8 E5M2.
    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        if val == 0.0 {
            return Self::ZERO;
        }

        let bits = val.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = bits & 0x007F_FFFF;

        let abs_val = val.abs();
        if abs_val > Self::MAX_VAL {
            return Self((sign << 7) | 0x7B);
        }

        let target_exp = exp + 15;
        if target_exp <= 0 {
            // Subnormal
            let shift = (1 - target_exp) as u32;
            let full_mantissa = (1 << 23) | mantissa;
            let sub_m = full_mantissa >> (21 + shift);
            Self((sign << 7) | (sub_m as u8 & 0x03))
        } else if target_exp >= 31 {
            Self((sign << 7) | 0x7B)
        } else {
            let m = (mantissa >> 21) as u8;
            Self((sign << 7) | ((target_exp as u8 & 0x1F) << 2) | (m & 0x03))
        }
    }

    /// Converts FP8 E5M2 into standard IEEE 754 f32.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let sign = (self.0 >> 7) & 1;
        let exp = (self.0 >> 2) & 0x1F;
        let mantissa = self.0 & 0x03;

        if exp == 0 {
            if mantissa == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            let sign_factor = if sign == 1 { -1.0f32 } else { 1.0f32 };
            return sign_factor * 2.0f32.powi(-14) * (mantissa as f32 / 4.0);
        }

        if exp == 31 {
            return if mantissa == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }

        let sign_factor = if sign == 1 { -1.0f32 } else { 1.0f32 };
        let e = exp as i32 - 15;
        let scale = 2.0f32.powi(e);
        let m = 1.0f32 + (mantissa as f32 / 4.0);
        sign_factor * scale * m
    }
}

/// Block of 32 FP8 E4M3 values with shared per-block f32 scale.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockFp8E4M3 {
    pub scale: f32,
    pub values: [Fp8E4M3; 32],
}

impl QuantScheme for BlockFp8E4M3 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        9.0 // 8 bits per weight + 32-bit scale per 32 weights
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

impl BlockFp8E4M3 {
    #[must_use]
    pub fn quantize(input: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let scale = if max_abs > 0.0 {
            max_abs / Fp8E4M3::MAX_VAL
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;

        let mut values = [Fp8E4M3::ZERO; 32];
        for i in 0..32 {
            values[i] = Fp8E4M3::from_f32(input[i] * inv_scale);
        }

        Self { scale, values }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        for i in 0..32 {
            output[i] = self.values[i].to_f32() * self.scale;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += self.values[i].to_f32() * activations[i];
        }
        sum * self.scale
    }
}
