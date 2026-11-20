//! Roofline Sanity Model and Enforcement
//!
//! Enforces epistemic honesty in benchmarks:
//! - MBU = (Bytes_model_and_kv * tok/s) / BW_measured
//! - MFU = (2 * P_active * tok/s) / F_peak
//! Aborts with `BenchmarkIntegrityViolation` if measured exceeds physical ceiling (> 1.05).

use crate::error::{EngineError, Result};

/// Hardware utilization and Roofline receipt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RooflineReceipt {
    pub tokens_per_sec: f64,
    pub bytes_per_token: usize,
    pub active_params: usize,
    pub measured_bw_gbps: f64,
    pub peak_flops_tflops: f64,
    pub mbu_ratio: f64,
    pub mfu_ratio: f64,
    pub is_prefill: bool,
}

impl RooflineReceipt {
    /// Creates a verified Roofline receipt.
    ///
    /// # Errors
    /// Returns `BenchmarkIntegrityViolation` if measured throughput exceeds
    /// physical hardware capabilities by more than 5% (noise/cache margin).
    pub fn new(
        tokens_per_sec: f64,
        bytes_per_token: usize,
        active_params: usize,
        measured_bw_gbps: f64,
        peak_flops_tflops: f64,
        is_prefill: bool,
    ) -> Result<Self> {
        let mbu_ratio = if measured_bw_gbps > 0.0 {
            (bytes_per_token as f64 * tokens_per_sec) / (measured_bw_gbps * 1e9)
        } else {
            0.0
        };

        let mfu_ratio = if peak_flops_tflops > 0.0 {
            (2.0 * active_params as f64 * tokens_per_sec) / (peak_flops_tflops * 1e12)
        } else {
            0.0
        };

        let ratio = if is_prefill { mfu_ratio } else { mbu_ratio };
        if ratio > 1.05 {
            return Err(EngineError::BenchmarkIntegrityViolation {
                measured: ratio as f32,
                ceiling: 1.05,
            });
        }

        Ok(Self {
            tokens_per_sec,
            bytes_per_token,
            active_params,
            measured_bw_gbps,
            peak_flops_tflops,
            mbu_ratio,
            mfu_ratio,
            is_prefill,
        })
    }
}
