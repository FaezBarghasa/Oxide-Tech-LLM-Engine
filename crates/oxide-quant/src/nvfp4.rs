//! NVIDIA Blackwell NVFP4 Micro-Atom Floating Point Quantization.
//!
//! Blackwell Tensor Cores introduce native hardware acceleration for 4-bit floating point (E2M1)
//! with dual-level scaling: a global tensor scale (FP32 or FP8) combined with a fine-grained
//! microscopic block scale (FP8 E4M3 or E8M0) across every 16 or 32 weights.

use crate::fp8::Fp8E4M3;
use oxide_core::traits::QuantScheme;

/// NVIDIA FP4 (E2M1) Microscopic Floating Point Quantization marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvFp4;

impl QuantScheme for NvFp4 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5 // 4-bit weights + FP8 micro-scale per 16 elements
    }

    #[inline(always)]
    fn block_size() -> usize {
        16
    }
}

/// Blackwell NVFP4 Micro-Block: 16 weights in 8 bytes + 1x FP8 E4M3 micro-scale (8.5 bytes total).
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockNvFp4_16 {
    pub scale: Fp8E4M3,
    pub qs: [u8; 8], // 16 nibbles = 8 bytes
}

impl QuantScheme for BlockNvFp4_16 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5 // (16 * 4 + 8) / 16 = 4.5 bpw
    }

    #[inline(always)]
    fn block_size() -> usize {
        16
    }
}

/// Blackwell E2M1 magnitude lookup table:
/// 000: 0.0, 001: 0.5, 010: 1.0, 011: 1.5, 100: 2.0, 101: 3.0, 110: 4.0, 111: 6.0
const NVFP4_E2M1_TABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

impl BlockNvFp4_16 {
    #[inline(always)]
    fn quantize_nibble(val: f32) -> u8 {
        let sign = if val < 0.0 { 0x08 } else { 0x00 };
        let abs_val = val.abs();

        let mut best_idx = 0u8;
        let mut min_diff = f32::MAX;
        for (i, &cand) in NVFP4_E2M1_TABLE.iter().enumerate() {
            let diff = (abs_val - cand).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = i as u8;
            }
        }
        sign | best_idx
    }

    #[inline(always)]
    fn dequantize_nibble(nibble: u8) -> f32 {
        let mag = NVFP4_E2M1_TABLE[(nibble & 0x07) as usize];
        if (nibble & 0x08) != 0 {
            -mag
        } else {
            mag
        }
    }

    /// Quantizes 16 floating point values with an optional global tensor scale.
    #[must_use]
    pub fn quantize(input: &[f32; 16], global_scale: f32) -> Self {
        let effective_global = if global_scale > 0.0 { global_scale } else { 1.0 };
        let mut max_abs = 0.0f32;
        for &v in input {
            let normalized = (v / effective_global).abs();
            max_abs = max_abs.max(normalized);
        }

        let raw_micro_scale = if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 };
        let scale = Fp8E4M3::from_f32(raw_micro_scale);
        let eff_micro = scale.to_f32() * effective_global;
        let inv_scale = if eff_micro > 0.0 { 1.0 / eff_micro } else { 0.0 };

        let mut qs = [0u8; 8];
        for i in 0..8 {
            let q0 = Self::quantize_nibble(input[i] * inv_scale);
            let q1 = Self::quantize_nibble(input[i + 8] * inv_scale);
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self { scale, qs }
    }

    /// Dequantizes 16 floating point values given the global tensor scale.
    pub fn dequantize(&self, output: &mut [f32; 16], global_scale: f32) {
        let s = self.scale.to_f32() * global_scale;
        for i in 0..8 {
            let byte = self.qs[i];
            output[i] = Self::dequantize_nibble(byte & 0x0F) * s;
            output[i + 8] = Self::dequantize_nibble((byte >> 4) & 0x0F) * s;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 16], global_scale: f32) -> f32 {
        let mut deq = [0.0f32; 16];
        self.dequantize(&mut deq, global_scale);
        let mut sum = 0.0f32;
        for i in 0..16 {
            sum += deq[i] * activations[i];
        }
        sum
    }
}
