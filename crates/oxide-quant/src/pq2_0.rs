use bytemuck::{Pod, Zeroable};
use oxide_core::traits::QuantScheme;

use crate::int_quant::{BlockQ8_0, f16};

/// Prism Quaternary 2-bit (PQ2_0) quantization scheme marker.
/// 128 weights packed into 34 bytes (2.125 bits per weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQuant2_0;

impl QuantScheme for PackedQuant2_0 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.125
    }

    #[inline(always)]
    fn block_size() -> usize {
        128
    }
}

/// Prism Quaternary PQ2_0 Block: 128 weights per block.
/// Memory layout:
/// - `d: f16`       -> 16-bit FP16 block scale (2 bytes)
/// - `qs: [u8; 32]` -> 128 elements packed at 2 bits each (32 bytes)
///
/// Total size: 2 + 32 = 34 bytes (zero padding, exact C ABI compatibility).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockPQ2_0 {
    pub d: f16,
    pub qs: [u8; 32],
}

// SAFETY: BlockPQ2_0 is repr(C) and contains only f16 and [u8; 32] with zero padding.
unsafe impl Zeroable for BlockPQ2_0 {}
// SAFETY: All 34-byte bit patterns are valid representations for BlockPQ2_0.
unsafe impl Pod for BlockPQ2_0 {}

impl BlockPQ2_0 {
    pub const QK: usize = 128;

    /// Extract the 2-bit quaternary coefficient in `{-1, 0, 1, 2}` for element `j` in `[0, 127]`.
    #[inline(always)]
    #[must_use]
    pub fn coef(&self, j: usize) -> i32 {
        let byte_index = j >> 2;
        let bit_offset = (j & 3) << 1;
        let q = (self.qs[byte_index] >> bit_offset) & 0x03;
        (q as i32) - 1
    }

    /// Returns the block scale as standard f32.
    #[inline(always)]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.d.to_f32()
    }

    /// Quantize a slice of 128 f32 weights into a `BlockPQ2_0`.
    #[must_use]
    pub fn quantize(x: &[f32; 128]) -> Self {
        let mut amax = 0.0f32;
        for &v in x {
            amax = amax.max(v.abs());
        }

        let d = amax;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 32];
        for (j, &val) in x.iter().enumerate() {
            let q = ((val * id).round() as i32 + 1).clamp(0, 3) as u8;
            let byte_index = j >> 2;
            let bit_offset = (j & 3) << 1;
            qs[byte_index] |= q << bit_offset;
        }

        Self {
            d: f16::from_f32(d),
            qs,
        }
    }

    /// Dequantize 128 elements into output buffer.
    #[inline(always)]
    pub fn dequantize(&self, output: &mut [f32; 128]) {
        let d = self.scale();
        for (j, out) in output.iter_mut().enumerate() {
            *out = (self.coef(j) as f32) * d;
        }
    }

    /// Computes the dot product between this 128-element quantized block and an f32 activation slice.
    #[inline(always)]
    #[must_use]
    pub fn dot_product_128(&self, activations: &[f32; 128]) -> f32 {
        let mut accum = 0.0f32;
        for (j, &act) in activations.iter().enumerate() {
            let c = self.coef(j);
            if c != 0 {
                accum += act * (c as f32);
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
                let c = self.coef(base_idx + b);
                sumi += c * (subblock.qs[b] as i32);
            }

            total += d0 * d1 * (sumi as f32);
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq2_0_size() {
        assert_eq!(std::mem::size_of::<BlockPQ2_0>(), 34);
        assert_eq!(std::mem::align_of::<BlockPQ2_0>(), 2);
    }

    #[test]
    fn test_pq2_0_quant_dequant_roundtrip() {
        let mut input = [0.0f32; 128];
        for (i, v) in input.iter_mut().enumerate() {
            let val = (i % 3) as f32 - 1.0; // -1, 0, 1
            *v = val * 0.5;
        }

        let block = BlockPQ2_0::quantize(&input);
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
