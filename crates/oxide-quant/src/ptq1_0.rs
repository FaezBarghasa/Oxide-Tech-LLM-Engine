use bytemuck::{Pod, Zeroable};
use oxide_core::traits::QuantScheme;

/// Ternary 1.58-bit quantization scheme marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ternary1_58Bit;

impl QuantScheme for Ternary1_58Bit {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        1.58
    }

    #[inline(always)]
    fn block_size() -> usize {
        128
    }
}

/// 128-element block of 2-bit ternary weights with an FP16 scale.
/// Total size: 32 bytes (256 bits for 128 weights) + 2 bytes FP16 scale = 34 bytes (aligned to 2 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TernaryBlock128 {
    pub scale_fp16: u16,
    pub packed_weights: [u8; 32],
}

impl TernaryBlock128 {
    /// Branchless unpack of a single weight at index `[0, 127]`.
    /// 00_2 -> 0.0, 01_2 -> +1.0, 10_2 -> -1.0
    #[inline(always)]
    #[must_use]
    pub fn unpack_sign(&self, idx: usize) -> f32 {
        let byte_idx = idx >> 2; // idx / 4
        let bit_shift = (idx & 3) << 1; // (idx % 4) * 2
        let code = (self.packed_weights[byte_idx] >> bit_shift) & 0x03;

        let is_pos = (code == 1) as i32;
        let is_neg = (code == 2) as i32;
        (is_pos - is_neg) as f32
    }

    /// Converts FP16 scale to standard f32.
    #[inline(always)]
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.scale_fp16)
    }

    /// Computes the dot product between this 128-element quantized block and an f32 activation slice.
    #[inline(always)]
    pub fn dot_product_128(&self, activations: &[f32; 128]) -> f32 {
        let mut accum: f32 = 0.0;

        for (byte_idx, &byte) in self.packed_weights.iter().enumerate() {
            let base_idx = byte_idx << 2;

            let code0 = byte & 0x03;
            let sign0 = ((code0 == 1) as i32 - (code0 == 2) as i32) as f32;
            accum += activations[base_idx] * sign0;

            let code1 = (byte >> 2) & 0x03;
            let sign1 = ((code1 == 1) as i32 - (code1 == 2) as i32) as f32;
            accum += activations[base_idx + 1] * sign1;

            let code2 = (byte >> 4) & 0x03;
            let sign2 = ((code2 == 1) as i32 - (code2 == 2) as i32) as f32;
            accum += activations[base_idx + 2] * sign2;

            let code3 = (byte >> 6) & 0x03;
            let sign3 = ((code3 == 1) as i32 - (code3 == 2) as i32) as f32;
            accum += activations[base_idx + 3] * sign3;
        }

        accum * self.scale()
    }
}

/// Helper converting an IEEE 754 half-precision binary representation to f32.
#[inline(always)]
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ff;
            let exp_f32 = 127 - 15 + 1 - e;
            f32::from_bits((sign << 31) | (exp_f32 << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Infinity or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13))
    } else {
        // Normalized
        let exp_f32 = exp + 127 - 15;
        f32::from_bits((sign << 31) | (exp_f32 << 23) | (mant << 13))
    }
}
