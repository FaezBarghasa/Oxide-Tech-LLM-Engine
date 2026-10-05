//! NVIDIA ModelOpt (Model Optimizer / TensorRT-LLM) Runtime Format.
//!
//! Provides support for NVIDIA ModelOpt post-training quantization, including
//! FP8, NVFP4, SmoothQuant (W8A8), AWQ, and INT4-FP8 mixed-precision schemes.

use oxide_core::traits::QuantScheme;
use serde::{Deserialize, Serialize};

/// NVIDIA ModelOpt quantization algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOptAlgorithm {
    Fp8,
    Nvfp4,
    SmoothQuant,
    Int4Awq,
    W4a16Gptq,
    Int8Uniform,
}

impl ModelOptAlgorithm {
    #[must_use]
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Nvfp4 => 4.5,
            Self::Int4Awq | Self::W4a16Gptq => 4.0,
            Self::Fp8 | Self::SmoothQuant | Self::Int8Uniform => 8.0,
        }
    }

    #[must_use]
    pub fn block_size(&self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            Self::Int4Awq | Self::W4a16Gptq => 128,
            Self::Fp8 | Self::SmoothQuant | Self::Int8Uniform => 32,
        }
    }
}

/// ZST marker for ModelOpt NVFP4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOptNvfp4;

impl QuantScheme for ModelOptNvfp4 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.5
    }

    #[inline(always)]
    fn block_size() -> usize {
        16
    }
}

/// ZST marker for ModelOpt FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOptFp8;

impl QuantScheme for ModelOptFp8 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}

/// ModelOpt quantization configuration descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOptConfig {
    pub algorithm: ModelOptAlgorithm,
    pub group_size: Option<usize>,
    pub kv_cache_quant: Option<String>,
    pub export_format: String,
}

impl ModelOptConfig {
    /// Creates a default ModelOpt config for Blackwell NVFP4.
    #[must_use]
    pub fn nvfp4_default() -> Self {
        Self {
            algorithm: ModelOptAlgorithm::Nvfp4,
            group_size: Some(16),
            kv_cache_quant: Some("fp8".to_string()),
            export_format: "tensorrt_llm".to_string(),
        }
    }

    /// Creates a default ModelOpt config for Hopper FP8.
    #[must_use]
    pub fn fp8_default() -> Self {
        Self {
            algorithm: ModelOptAlgorithm::Fp8,
            group_size: None,
            kv_cache_quant: Some("fp8".to_string()),
            export_format: "tensorrt_llm".to_string(),
        }
    }

    /// Creates a SmoothQuant configuration.
    #[must_use]
    pub fn smoothquant_default() -> Self {
        Self {
            algorithm: ModelOptAlgorithm::SmoothQuant,
            group_size: None,
            kv_cache_quant: Some("int8".to_string()),
            export_format: "tensorrt_llm".to_string(),
        }
    }
}

/// Applies SmoothQuant activation smoothing transform:
/// W' = diag(s) * W, X' = X * diag(s)^(-1)
pub fn apply_smoothquant_weights(weights: &mut [f32], smoothing_factors: &[f32], cols: usize) {
    for (i, w) in weights.iter_mut().enumerate() {
        let col = i % cols;
        let s = smoothing_factors.get(col).copied().unwrap_or(1.0);
        *w *= s;
    }
}
