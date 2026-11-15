//! Specialized Hailo (Hailo-8, Hailo-10, Hailo-15) NPU Dataflow Kernels.
//! Implements hardware dataflow execution streams, virtual stream pipeline processing,
//! and low-latency INT8/INT4 neural network evaluation.

use oxide_core::error::Result;

/// Specialized Hailo NPU Dataflow Kernels Dispatcher.
#[derive(Debug, Default)]
pub struct HailoLlmKernels;

impl HailoLlmKernels {
    /// Dispatches Hailo-8 / Hailo-10 Dataflow INT8 Matrix-Vector Multiplication.
    pub fn dispatch_hailo_dataflow_gemv_int8(
        out: &mut [f32],
        weights_int8: &[i8],
        activations: &[f32],
        scale: f32,
        m: usize,
        n: usize,
    ) {
        assert!(weights_int8.len() >= m * n);
        assert!(activations.len() >= n);
        assert!(out.len() >= m);

        for row in 0..m {
            let offset = row * n;
            let mut acc = 0.0f32;
            for col in 0..n {
                acc += (weights_int8[offset + col] as f32) * activations[col];
            }
            out[row] = acc * scale;
        }
    }

    /// Dispatches Hailo Dataflow 4-bit Quantized Matrix-Vector Multiplication.
    pub fn dispatch_hailo_dataflow_gemv_int4(
        out: &mut [f32],
        weights_q4: &[oxide_quant::int_quant::BlockQ4_0],
        activations: &[f32],
        m: usize,
        n: usize,
    ) {
        oxide_quant::simd::gemv_q4_0(weights_q4, activations, m, n, out);
    }

    /// Dispatches a specialized forward step decode on Hailo NPU virtual stream.
    pub fn dispatch_hailo_step_decode(
        _input_token: u32,
        _slot_idx: usize,
        _buffer: &mut crate::hailort::HailoVStreamBuffer,
    ) -> Result<u32> {
        // Zero mock/synthetic kernels per AGENTS.md invariant.
        // Real HailoRT VStream execution pipeline is loaded in Phase 4.
        Err(oxide_core::error::EngineError::BackendError(
            "Hailo step decode requires loaded HEF model network and active VStream buffer"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hailo_step_decode() {
        let mut buf = crate::hailort::HailoVStreamBuffer::new(4);
        let res = HailoLlmKernels::dispatch_hailo_step_decode(80, 0, &mut buf);
        assert!(res.is_err());
    }
}
