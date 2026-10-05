//! PyTorch Architecture Optimization (TorchAO) Runtime Formats.
//!
//! Implements torchao's modern quantization primitives including sub-byte integers (int1 to int8),
//! microscopic float6 (FP6_E3M2, FP6_E2M3), float5 (FP5), and dynamic activation float8 quantization.

use oxide_core::traits::QuantScheme;
use serde::{Deserialize, Serialize};

/// TorchAO quantization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TorchAoQuantType {
    Int4WeightOnly,
    Int8WeightOnly,
    Int8DynamicActivationInt8Weight,
    Float8WeightOnly,
    Float8DynamicActivationFloat8Weight,
    Fp6E3M2,
    Fp6E2M3,
    Fp5,
    UIntXWeightOnly(u8),
}

impl TorchAoQuantType {
    #[must_use]
    pub fn bits_per_weight(&self) -> f32 {
        match *self {
            Self::Int4WeightOnly => 4.0,
            Self::Int8WeightOnly
            | Self::Int8DynamicActivationInt8Weight
            | Self::Float8WeightOnly
            | Self::Float8DynamicActivationFloat8Weight => 8.0,
            Self::Fp6E3M2 | Self::Fp6E2M3 => 6.0,
            Self::Fp5 => 5.0,
            Self::UIntXWeightOnly(bits) => bits as f32,
        }
    }

    #[must_use]
    pub fn block_size(&self) -> usize {
        32
    }
}

/// ZST marker for TorchAO INT4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorchAoInt4;

impl QuantScheme for TorchAoInt4 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// ZST marker for TorchAO FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TorchAoFloat8;

impl QuantScheme for TorchAoFloat8 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// TorchAO Linear Layer Descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TorchAoLinearDescriptor {
    pub quant_type: TorchAoQuantType,
    pub in_features: usize,
    pub out_features: usize,
    pub group_size: Option<usize>,
    pub has_bias: bool,
}

impl TorchAoLinearDescriptor {
    #[must_use]
    pub fn int4_wo_default(in_features: usize, out_features: usize) -> Self {
        Self {
            quant_type: TorchAoQuantType::Int4WeightOnly,
            in_features,
            out_features,
            group_size: Some(128),
            has_bias: false,
        }
    }

    #[must_use]
    pub fn fp8_wo_default(in_features: usize, out_features: usize) -> Self {
        Self {
            quant_type: TorchAoQuantType::Float8WeightOnly,
            in_features,
            out_features,
            group_size: None,
            has_bias: false,
        }
    }
}

/// FP6 E3M2 dequantization lookup table (64 values).
/// 1 sign bit, 3 exponent bits, 2 mantissa bits.
#[must_use]
pub fn dequantize_fp6_e3m2(bits: u8) -> f32 {
    let sign = (bits >> 5) & 1;
    let exp = (bits >> 2) & 0x07;
    let mantissa = bits & 0x03;

    let sign_factor = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        // Subnormal: 2^(-2) * (mantissa / 4)
        sign_factor * 0.25 * (mantissa as f32 / 4.0)
    } else {
        // Normal: 2^(exp - 3) * (1 + mantissa / 4)
        let scale = 2.0f32.powi(exp as i32 - 3);
        let m = 1.0f32 + (mantissa as f32 / 4.0);
        sign_factor * scale * m
    }
}
