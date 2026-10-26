//! Reflex-inspired Quantized-Resident (QR) Tensors and In-Register GEMV.
//!
//! Enforces zero-copy memory-mapped slicing:
//! - Weights remain stored in their native quantized representation (`BlockQ4_K`, `BlockQ8_0`) in memory or VRAM.
//! - Eliminates 4x memory expansion caused by dequantizing weights to FP16/FP32 buffers.
//! - Performs inline dequantization directly inside SIMD / vector registers during GEMV.

use crate::gguf_quants::BlockQ4_K;
use crate::int_quant::BlockQ8_0;
use crate::simd::{dot_q4_k, dot_q8_0};
use rayon::prelude::*;

/// Supported resident quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrQuantKind {
    Q4K,
    Q80,
}

/// Zero-copy Quantized-Resident Tensor referencing memory-mapped or device-mapped weight blocks.
#[derive(Debug, Clone, Copy)]
pub enum QuantizedResidentTensor<'a> {
    /// 4-bit K-quant resident blocks (256 elements per 144-byte block).
    Q4K {
        rows: usize,
        cols: usize,
        blocks: &'a [BlockQ4_K],
    },
    /// 8-bit standard resident blocks (32 elements per 34-byte block).
    Q80 {
        rows: usize,
        cols: usize,
        blocks: &'a [BlockQ8_0],
    },
}

impl<'a> QuantizedResidentTensor<'a> {
    /// Creates a zero-copy Q4_K resident tensor from a raw byte slice.
    pub fn from_q4_k_bytes(
        rows: usize,
        cols: usize,
        raw_bytes: &'a [u8],
    ) -> Result<Self, &'static str> {
        if !cols.is_multiple_of(256) {
            return Err("Cols must be a multiple of 256 for Q4_K");
        }
        let blocks_per_row = cols / 256;
        let total_blocks = rows * blocks_per_row;
        let expected_bytes = total_blocks * std::mem::size_of::<BlockQ4_K>();

        if raw_bytes.len() < expected_bytes {
            return Err("Byte slice too small for specified Q4_K dimensions");
        }

        let slice = &raw_bytes[..expected_bytes];
        let blocks: &'a [BlockQ4_K] = bytemuck::try_cast_slice(slice)
            .map_err(|_| "Failed to cast raw bytes to BlockQ4_K slice (alignment mismatch)")?;

        Ok(Self::Q4K { rows, cols, blocks })
    }

    /// Creates a zero-copy Q8_0 resident tensor from a raw byte slice.
    pub fn from_q8_0_bytes(
        rows: usize,
        cols: usize,
        raw_bytes: &'a [u8],
    ) -> Result<Self, &'static str> {
        if !cols.is_multiple_of(32) {
            return Err("Cols must be a multiple of 32 for Q8_0");
        }
        let blocks_per_row = cols / 32;
        let total_blocks = rows * blocks_per_row;
        let expected_bytes = total_blocks * std::mem::size_of::<BlockQ8_0>();

        if raw_bytes.len() < expected_bytes {
            return Err("Byte slice too small for specified Q8_0 dimensions");
        }

        let slice = &raw_bytes[..expected_bytes];
        let blocks: &'a [BlockQ8_0] = bytemuck::try_cast_slice(slice)
            .map_err(|_| "Failed to cast raw bytes to BlockQ8_0 slice (alignment mismatch)")?;

        Ok(Self::Q80 { rows, cols, blocks })
    }

    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        match *self {
            Self::Q4K { rows, cols, .. } | Self::Q80 { rows, cols, .. } => (rows, cols),
        }
    }

    #[must_use]
    pub fn quant_kind(&self) -> QrQuantKind {
        match *self {
            Self::Q4K { .. } => QrQuantKind::Q4K,
            Self::Q80 { .. } => QrQuantKind::Q80,
        }
    }

    /// Physical byte footprint of the resident weights in memory.
    #[must_use]
    pub fn memory_footprint_bytes(&self) -> usize {
        match *self {
            Self::Q4K { blocks, .. } => std::mem::size_of_val(blocks),
            Self::Q80 { blocks, .. } => std::mem::size_of_val(blocks),
        }
    }

    /// Equivalent uncompressed FP16/FP32 memory footprint for comparison.
    #[must_use]
    pub fn uncompressed_f16_bytes(&self) -> usize {
        let (r, c) = self.shape();
        r * c * 2
    }

    /// Memory savings ratio achieved by keeping weights resident in quantized form.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.uncompressed_f16_bytes() as f32 / self.memory_footprint_bytes() as f32
    }

    /// Rayon-accelerated in-register matrix-vector multiplication ($y = A \cdot x$).
    ///
    /// Weights are decoded directly into SIMD accumulator registers without creating
    /// any intermediate unpacked FP16/FP32 weight matrices in memory.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) -> Result<(), &'static str> {
        let (rows, cols) = self.shape();
        if x.len() != cols {
            return Err("Input vector length does not match tensor columns");
        }
        if y.len() != rows {
            return Err("Output vector length does not match tensor rows");
        }

        match *self {
            Self::Q4K { blocks, .. } => {
                let blocks_per_row = cols / 256;
                y.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                    let mut row_sum = 0.0f32;
                    for b in 0..blocks_per_row {
                        let block = &blocks[r * blocks_per_row + b];
                        let act_start = b * 256;
                        if let Ok(act_chunk) = x[act_start..act_start + 256].try_into() {
                            row_sum += dot_q4_k(
                                &block.qs,
                                act_chunk,
                                block.d.to_f32(),
                                block.dmin.to_f32(),
                            );
                        }
                    }
                    *out_val = row_sum;
                });
            }
            Self::Q80 { blocks, .. } => {
                let blocks_per_row = cols / 32;
                y.par_iter_mut().enumerate().for_each(|(r, out_val)| {
                    let mut row_sum = 0.0f32;
                    for b in 0..blocks_per_row {
                        let block = &blocks[r * blocks_per_row + b];
                        let act_start = b * 32;
                        if let Ok(act_chunk) = x[act_start..act_start + 32].try_into() {
                            row_sum += dot_q8_0(&block.qs, act_chunk, block.scale.to_f32());
                        }
                    }
                    *out_val = row_sum;
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_tensor_q8_0_lifecycle() {
        let rows = 4;
        let cols = 64; // 2 blocks per row
        let blocks_total = rows * (cols / 32);
        let mut raw_bytes = vec![0u8; blocks_total * std::mem::size_of::<BlockQ8_0>()];

        // Populate scale factor for block 0 (first 34 bytes)
        let f16_scale = half::f16::from_f32(2.0).to_bits();
        raw_bytes[0..2].copy_from_slice(&f16_scale.to_le_bytes());
        // Set first byte of qs to 3
        raw_bytes[2] = 3;

        let qr = QuantizedResidentTensor::from_q8_0_bytes(rows, cols, &raw_bytes)
            .expect("cast succeeds");

        assert_eq!(qr.shape(), (4, 64));
        assert_eq!(qr.quant_kind(), QrQuantKind::Q80);
        assert!(qr.compression_ratio() > 1.8);

        let x = vec![1.0f32; 64];
        let mut y = vec![0.0f32; 4];

        assert!(qr.matvec(&x, &mut y).is_ok());
        // First element: 2.0 * 3.0 * 1.0 = 6.0
        assert_eq!(y[0], 6.0);
    }
}
