#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::similar_names
)]

use serde::{Deserialize, Serialize};

/// Statistical profile of a live layer activation tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationTelemetry {
    pub tensor_name: String,
    pub num_elements: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
    pub sparsity_percentage: f32,
    pub has_nan: bool,
    pub has_inf: bool,
}

/// Anomaly detected during live tensor debugging.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorAnomaly {
    NanDetected {
        tensor_name: String,
        count: usize,
    },
    InfDetected {
        tensor_name: String,
        count: usize,
    },
    GradientExplosion {
        tensor_name: String,
        norm: f32,
        threshold: f32,
    },
    GradientVanishing {
        tensor_name: String,
        norm: f32,
        threshold: f32,
    },
}

/// Real-Time Tensor Debugger & Live Observability Probe.
#[derive(Debug, Clone)]
pub struct TensorDebugger;

impl TensorDebugger {
    /// Inspects and calculates comprehensive statistics for a tensor slice.
    #[must_use]
    pub fn inspect(tensor_name: impl Into<String>, slice: &[f32]) -> ActivationTelemetry {
        let name = tensor_name.into();
        let n = slice.len();
        if n == 0 {
            return ActivationTelemetry {
                tensor_name: name,
                num_elements: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                l2_norm: 0.0,
                sparsity_percentage: 0.0,
                has_nan: false,
                has_inf: false,
            };
        }

        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut zeros = 0usize;
        let mut has_nan = false;
        let mut has_inf = false;

        for &val in slice {
            if val.is_nan() {
                has_nan = true;
            } else if val.is_infinite() {
                has_inf = true;
            } else {
                sum += val;
                sum_sq += val * val;
                min = min.min(val);
                max = max.max(val);
                if val.abs() < 1e-7 {
                    zeros += 1;
                }
            }
        }

        let mean = sum / (n as f32);
        let variance = (sum_sq / (n as f32) - mean * mean).max(0.0);
        let std_dev = variance.sqrt();
        let l2_norm = sum_sq.sqrt();
        let sparsity = (zeros as f32 * 100.0) / (n as f32);

        ActivationTelemetry {
            tensor_name: name,
            num_elements: n,
            mean,
            std_dev,
            min: if min.is_infinite() { 0.0 } else { min },
            max: if max.is_infinite() { 0.0 } else { max },
            l2_norm,
            sparsity_percentage: sparsity,
            has_nan,
            has_inf,
        }
    }

    /// Verifies tensor values against numerical safety bounds.
    pub fn assert_safe(tensor_name: &str, slice: &[f32]) -> Result<(), TensorAnomaly> {
        let mut nan_count = 0;
        let mut inf_count = 0;

        for &val in slice {
            if val.is_nan() {
                nan_count += 1;
            } else if val.is_infinite() {
                inf_count += 1;
            }
        }

        if nan_count > 0 {
            return Err(TensorAnomaly::NanDetected {
                tensor_name: tensor_name.to_string(),
                count: nan_count,
            });
        }
        if inf_count > 0 {
            return Err(TensorAnomaly::InfDetected {
                tensor_name: tensor_name.to_string(),
                count: inf_count,
            });
        }

        Ok(())
    }
}

/// Comparison metric between floating-point golden baseline and candidate quantized/pruned outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftSummary {
    pub tensor_name: String,
    pub cosine_similarity: f32,
    pub mean_squared_error: f32,
    pub max_absolute_diff: f32,
    pub relative_l2_error: f32,
}

/// Detects numerical degradation and representation drift across model layers.
#[derive(Debug, Clone)]
pub struct DriftDetector;

impl DriftDetector {
    /// Compares two activation slices and computes numerical divergence metrics.
    #[must_use]
    pub fn compare(
        tensor_name: impl Into<String>,
        golden: &[f32],
        candidate: &[f32],
    ) -> DriftSummary {
        let name = tensor_name.into();
        let n = golden.len().min(candidate.len());
        if n == 0 {
            return DriftSummary {
                tensor_name: name,
                cosine_similarity: 1.0,
                mean_squared_error: 0.0,
                max_absolute_diff: 0.0,
                relative_l2_error: 0.0,
            };
        }

        let mut dot = 0.0f32;
        let mut norm_g_sq = 0.0f32;
        let mut norm_c_sq = 0.0f32;
        let mut sum_sq_diff = 0.0f32;
        let mut max_abs_diff = 0.0f32;

        for i in 0..n {
            let g = golden[i];
            let c = candidate[i];
            dot += g * c;
            norm_g_sq += g * g;
            norm_c_sq += c * c;

            let diff = (g - c).abs();
            sum_sq_diff += diff * diff;
            max_abs_diff = max_abs_diff.max(diff);
        }

        let denom = (norm_g_sq.sqrt() * norm_c_sq.sqrt()).max(1e-12);
        let cos_sim = (dot / denom).clamp(-1.0, 1.0);
        let mse = sum_sq_diff / (n as f32);
        let rel_l2 = sum_sq_diff.sqrt() / norm_g_sq.sqrt().max(1e-12);

        DriftSummary {
            tensor_name: name,
            cosine_similarity: cos_sim,
            mean_squared_error: mse,
            max_absolute_diff: max_abs_diff,
            relative_l2_error: rel_l2,
        }
    }
}

/// Perplexity Auditor measuring validation loss and prediction quality.
#[derive(Debug, Clone)]
pub struct PerplexityAuditor;

impl PerplexityAuditor {
    /// Computes perplexity: PPL = exp(mean CrossEntropyLoss) across sequence logits and targets.
    #[must_use]
    pub fn evaluate_ppl(sequence_logits: &[Vec<f32>], sequence_targets: &[usize]) -> f32 {
        let n = sequence_logits.len().min(sequence_targets.len());
        if n == 0 {
            return 1.0;
        }

        let mut total_loss = 0.0f32;
        for i in 0..n {
            let (loss, _) = crate::training::LossComputer::compute_cross_entropy(
                &sequence_logits[i],
                sequence_targets[i],
                0.0,
            );
            total_loss += loss;
        }

        let mean_loss = total_loss / (n as f32);
        mean_loss.exp()
    }
}

/// Telemetry profile capturing static arena headroom, peak VRAM, and forward latency jitter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryAndJitterProfile {
    pub arena_capacity_bytes: usize,
    pub arena_used_bytes: usize,
    pub arena_headroom_bytes: usize,
    pub peak_vram_bytes: usize,
    pub kv_cache_block_count: usize,
    pub kv_cache_used_blocks: usize,
    pub ttft_micros: u64,
    pub mean_decode_token_micros: u64,
    pub p99_decode_token_micros: u64,
}
