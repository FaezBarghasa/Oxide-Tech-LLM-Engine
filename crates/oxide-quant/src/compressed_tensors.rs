//! Compressed-Tensors Runtime Format (Neural Magic / vLLM / Hugging Face).
//!
//! Standardized quantization specification format for LLMs across modern open-source serving
//! engines, supporting packed integer, floating point (FP8), and Marlin tensor layouts.

use oxide_core::traits::QuantScheme;
use serde::{Deserialize, Serialize};

/// High-level compression strategy format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompressionFormat {
    PackQuantized,
    FloatQuantized,
    Marlin,
    IntQuantized,
    NaiveQuantized,
}

/// Precision specification for weights and activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressedQuantType {
    W8A8Int,
    W4A16Int,
    W8A16Int,
    W8A8Fp8,
    Fp8Dynamic,
    W4A8Fp8,
    W4A4Int,
}

impl CompressedQuantType {
    #[must_use]
    pub fn bits_per_weight(&self) -> f32 {
        match *self {
            Self::W4A16Int | Self::W4A8Fp8 | Self::W4A4Int => 4.0,
            Self::W8A8Int | Self::W8A16Int | Self::W8A8Fp8 | Self::Fp8Dynamic => 8.0,
        }
    }

    #[must_use]
    pub fn block_size(&self) -> usize {
        32
    }
}

/// ZST marker for Compressed-Tensors W4A16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressedW4A16;

impl QuantScheme for CompressedW4A16 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// ZST marker for Compressed-Tensors W8A8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressedW8A8;

impl QuantScheme for CompressedW8A8 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// Granularity of scale and zero-point parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantizationStrategy {
    Tensor,
    Channel,
    Group,
}

/// Parsed `quantization_config` metadata dictionary from Hugging Face `config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressedTensorsConfig {
    pub format: CompressionFormat,
    pub quant_type: CompressedQuantType,
    pub strategy: QuantizationStrategy,
    pub group_size: Option<usize>,
    pub symmetric: bool,
    pub act_quant: bool,
}

impl CompressedTensorsConfig {
    /// Creates a default configuration for W4A16 packed INT4.
    #[must_use]
    pub fn w4a16_default() -> Self {
        Self {
            format: CompressionFormat::PackQuantized,
            quant_type: CompressedQuantType::W4A16Int,
            strategy: QuantizationStrategy::Group,
            group_size: Some(128),
            symmetric: true,
            act_quant: false,
        }
    }

    /// Creates a default configuration for W8A8 FP8 inference.
    #[must_use]
    pub fn fp8_default() -> Self {
        Self {
            format: CompressionFormat::FloatQuantized,
            quant_type: CompressedQuantType::W8A8Fp8,
            strategy: QuantizationStrategy::Tensor,
            group_size: None,
            symmetric: true,
            act_quant: true,
        }
    }

    /// Parses compressed-tensors configuration from raw JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

/// Dequantizes compressed-tensors packed weights given configuration.
pub fn dequantize_compressed_slice(
    packed: &[u8],
    scales: &[f32],
    zeros: Option<&[i32]>,
    config: &CompressedTensorsConfig,
    output: &mut [f32],
) {
    let group_size = config.group_size.unwrap_or(usize::MAX);
    match config.quant_type {
        CompressedQuantType::W4A16Int | CompressedQuantType::W4A8Fp8 | CompressedQuantType::W4A4Int => {
            for i in 0..packed.len() {
                let out_idx0 = i * 2;
                let out_idx1 = i * 2 + 1;
                if out_idx0 >= output.len() {
                    break;
                }

                let g0 = if group_size == usize::MAX { 0 } else { out_idx0 / group_size };
                let g1 = if group_size == usize::MAX { 0 } else { out_idx1 / group_size };

                let s0 = scales.get(g0).copied().unwrap_or(1.0);
                let s1 = scales.get(g1).copied().unwrap_or(1.0);

                let z0 = zeros.and_then(|z| z.get(g0).copied()).unwrap_or(0);
                let z1 = zeros.and_then(|z| z.get(g1).copied()).unwrap_or(0);

                let byte = packed[i];
                let q0 = (byte & 0x0F) as i32 - z0;
                let q1 = ((byte >> 4) & 0x0F) as i32 - z1;

                output[out_idx0] = q0 as f32 * s0;
                if out_idx1 < output.len() {
                    output[out_idx1] = q1 as f32 * s1;
                }
            }
        }
        CompressedQuantType::W8A8Int | CompressedQuantType::W8A16Int => {
            for (i, &byte) in packed.iter().enumerate() {
                if i >= output.len() {
                    break;
                }
                let g = if group_size == usize::MAX { 0 } else { i / group_size };
                let s = scales.get(g).copied().unwrap_or(1.0);
                let z = zeros.and_then(|z| z.get(g).copied()).unwrap_or(0);
                output[i] = (byte as i8 as i32 - z) as f32 * s;
            }
        }
        CompressedQuantType::W8A8Fp8 | CompressedQuantType::Fp8Dynamic => {
            use crate::fp8::Fp8E4M3;
            for (i, &byte) in packed.iter().enumerate() {
                if i >= output.len() {
                    break;
                }
                let g = if group_size == usize::MAX { 0 } else { i / group_size };
                let s = scales.get(g).copied().unwrap_or(1.0);
                output[i] = Fp8E4M3(byte).to_f32() * s;
            }
        }
    }
}
