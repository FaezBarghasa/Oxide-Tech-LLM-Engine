use bytemuck::{Pod, Zeroable};
use oxide_core::traits::QuantScheme;

use crate::int_quant::{BlockQ8_0, f16};

/// Ternary 1.58-bit (PTQ1_0) quantization scheme marker.
/// 128 weights packed into 28 bytes (1.75 bits per weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ternary1_58Bit;

impl QuantScheme for Ternary1_58Bit {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        1.75
    }

    #[inline(always)]
    fn block_size() -> usize {
        128
    }
}

/// Prism True Ternary PTQ1_0 Block: 128 weights per block.
/// Memory layout:
/// - `qs: [u8; 24]` -> 120 elements stored in base-3 (5 trits per byte, 3^5 = 243 <= 255)
/// - `qh: [u8; 2]`  -> 8 elements stored in base-3 (4 trits per byte, 3^4 = 81 <= 255)
/// - `d: f16`       -> 16-bit FP16 block scale
///
/// Total size: 24 + 2 + 2 = 28 bytes (zero padding, exact C ABI compatibility).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockPTQ1_0 {
    pub qs: [u8; 24],
    pub qh: [u8; 2],
    pub d: f16,
}

// Backward-compatible alias for existing callers
pub type TernaryBlock128 = BlockPTQ1_0;

// SAFETY: BlockPTQ1_0 is repr(C) and contains only primitive arrays and f16 with zero padding.
unsafe impl Zeroable for BlockPTQ1_0 {}
// SAFETY: All 28-byte bit patterns are valid representations for BlockPTQ1_0.
unsafe impl Pod for BlockPTQ1_0 {}

impl BlockPTQ1_0 {
    pub const QK: usize = 128;

    /// Extract the ternary coefficient in `{-1, 0, 1}` for element `e` in `[0, 127]`.
    /// Follows the canonical base-3 decoding mapping verified against Prism ML.
    #[inline(always)]
    #[must_use]
    pub fn trit(&self, e: usize) -> i32 {
        let (b, n) = if e < 80 {
            (self.qs[e & 15], e >> 4)
        } else if e < 120 {
            let t = e - 80;
            (self.qs[16 + (t & 7)], t >> 3)
        } else {
            let t = e - 120;
            (self.qh[t & 1], t >> 1)
        };

        let mut v = b as u32;
        for _ in 0..n {
            v = (v * 3) & 0xFF;
        }
        (((v * 3) >> 8) as i32) - 1
    }

    /// Unpack sign as a float (`-1.0`, `0.0`, or `1.0`).
    #[inline(always)]
    #[must_use]
    pub fn unpack_sign(&self, idx: usize) -> f32 {
        self.trit(idx) as f32
    }

    /// Returns the block scale as standard f32.
    #[inline(always)]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.d.to_f32()
    }

    /// Quantize a slice of 128 f32 weights into a `BlockPTQ1_0`.
    #[must_use]
    pub fn quantize(x: &[f32; 128]) -> Self {
        let mut amax = 0.0f32;
        for &v in x {
            amax = amax.max(v.abs());
        }

        let d = amax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 24];
        let mut qh = [0u8; 2];

        // 120 elements in qs via stage 16 then stage 8
        let stages = [16usize, 8usize];
        let mut x_offset = 0;
        let mut j = 0;

        for &c in &stages {
            for m in 0..c {
                let mut q: u16 = 0;
                for n in 0..5 {
                    let val = x[x_offset + m + n * c] * id;
                    let xi = (val.round() as i32 + 1).clamp(0, 2) as u16;
                    q = q * 3 + xi;
                }
                // Ceiling division: (q * 256).div_ceil(243)
                let byte = (q as u32 * 256).div_ceil(243) as u8;
                qs[j + m] = byte;
            }
            j += c;
            x_offset += 5 * c;
        }

        // 8 elements in qh: 2 bytes, 4 elements each
        for h in 0..2 {
            let mut q: u16 = 0;
            for m in 0..4 {
                let val = x[x_offset + h + m * 2] * id;
                let xi = (val.round() as i32 + 1).clamp(0, 2) as u16;
                q = q * 3 + xi;
            }
            q *= 3; // Shift to most significant trit
            let byte = (q as u32 * 256).div_ceil(243) as u8;
            qh[h] = byte;
        }

        Self {
            qs,
            qh,
            d: f16::from_f32(d),
        }
    }

    /// Dequantize 128 elements into output buffer.
    #[inline(always)]
    pub fn dequantize(&self, output: &mut [f32; 128]) {
        let d = self.scale();
        for i in 0..128 {
            output[i] = self.trit(i) as f32 * d;
        }
    }

    /// Computes the dot product between this 128-element quantized block and an f32 activation slice.
    #[inline(always)]
    #[must_use]
    pub fn dot_product_128(&self, activations: &[f32; 128]) -> f32 {
        let mut accum = 0.0f32;
        for i in 0..128 {
            let t = self.trit(i);
            if t != 0 {
                accum += activations[i] * (t as f32);
            }
        }
        accum * self.scale()
    }

    /// Computes the dot product against 4 quantized `BlockQ8_0` activation sub-blocks (4 * 32 = 128).
    #[inline(always)]
    #[must_use]
    pub fn dot_product_q8_0(&self, q8_subblocks: &[BlockQ8_0; 4]) -> f32 {
        let d0 = self.scale();
        let mut total = 0.0f32;

        for (k, subblock) in q8_subblocks.iter().enumerate() {
            let d1 = subblock.scale.to_f32();
            let base_idx = k * 32;
            let mut sumi = 0i32;

            for b in 0..32 {
                let t = self.trit(base_idx + b);
                sumi += t * (subblock.qs[b] as i32);
            }

            total += d0 * d1 * (sumi as f32);
        }

        total
    }
}

/// Helper converting an IEEE 754 half-precision binary representation to f32.
#[inline(always)]
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    f16(h).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptq1_0_size() {
        assert_eq!(std::mem::size_of::<BlockPTQ1_0>(), 28);
        assert_eq!(std::mem::align_of::<BlockPTQ1_0>(), 2);
    }

    #[test]
    fn test_ptq1_0_quant_dequant_roundtrip() {
        let mut input = [0.0f32; 128];
        for (i, v) in input.iter_mut().enumerate() {
            let val = (i % 3) as f32 - 1.0; // -1, 0, 1
            *v = val * 0.75;
        }

        let block = BlockPTQ1_0::quantize(&input);
        let mut out = [0.0f32; 128];
        block.dequantize(&mut out);

        for i in 0..128 {
            let diff = (out[i] - input[i]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at {}: expected {}, got {}",
                i,
                input[i],
                out[i]
            );
        }
    }
}
