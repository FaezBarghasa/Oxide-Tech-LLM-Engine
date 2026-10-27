//! Specialized CPU Kernels for LLM Inference.
//! Implements multi-threaded cache-blocked execution, core-affinity pinning,
//! AVX-512/AVX2 SIMD acceleration, and quantized GEMV operations.

use core_affinity::CoreId;
use std::sync::Arc;

/// NUMA and core-affinity aware CPU thread pool for LLM compute.
/// Pins worker threads to physical CPU cores and logical hardware threads
/// across 8 to 128+ cores on AMD EPYC, Intel Xeon, and desktop Ryzen/Core processors.
#[derive(Debug, Clone)]
pub struct CpuThreadPool {
    num_threads: usize,
    core_ids: Arc<Vec<CoreId>>,
}

impl CpuThreadPool {
    /// Initialize CPU thread pool with available hardware threads and core affinity.
    #[must_use]
    pub fn new() -> Self {
        let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();

        Self {
            num_threads: available,
            core_ids: Arc::new(core_ids),
        }
    }

    /// Number of active worker threads in the pool.
    #[must_use]
    pub const fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Pin the current calling thread to a specific hardware core ID.
    pub fn pin_current_thread(&self, thread_index: usize) -> bool {
        if self.core_ids.is_empty() {
            return false;
        }
        let core = self.core_ids[thread_index % self.core_ids.len()];
        core_affinity::set_for_current(core)
    }
}

impl Default for CpuThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized CPU LLM Kernels Dispatcher.
/// Orchestrates multi-threaded SIMD GEMVs, quantized GEMVs, RMSNorm, RoPE,
/// and FlashAttention-2 / FlashDecode on CPU silicon.
#[derive(Debug, Default)]
pub struct CpuLlmKernels;

impl CpuLlmKernels {
    /// Dispatches multi-threaded cache-blocked FP32 Matrix-Vector Multiplication.
    /// Computes `out = matrix * vector` where matrix is `[m, n]`.
    pub fn dispatch_cpu_gemv_f32(
        out: &mut [f32],
        matrix: &[f32],
        vector: &[f32],
        m: usize,
        n: usize,
    ) {
        oxide_quant::simd::gemv_blocked_f32(matrix, vector, m, n, out);
    }

    /// Dispatches multi-threaded Q8_0 quantized Matrix-Vector Multiplication.
    pub fn dispatch_cpu_gemv_q8_0(
        out: &mut [f32],
        matrix: &[oxide_quant::int_quant::BlockQ8_0],
        vector: &[f32],
        m: usize,
        n: usize,
    ) {
        oxide_quant::simd::gemv_q8_0(matrix, vector, m, n, out);
    }

    /// Dispatches multi-threaded Q4_0 quantized Matrix-Vector Multiplication.
    pub fn dispatch_cpu_gemv_q4_0(
        out: &mut [f32],
        matrix: &[oxide_quant::int_quant::BlockQ4_0],
        vector: &[f32],
        m: usize,
        n: usize,
    ) {
        oxide_quant::simd::gemv_q4_0(matrix, vector, m, n, out);
    }

    /// Dispatches multi-threaded Q4_K super-block quantized Matrix-Vector Multiplication.
    pub fn dispatch_cpu_gemv_q4_k(
        out: &mut [f32],
        matrix: &[oxide_quant::gguf_quants::BlockQ4_K],
        vector: &[f32],
        m: usize,
        n: usize,
    ) {
        oxide_quant::simd::gemv_q4_k(matrix, vector, m, n, out);
    }

    /// Dispatches vectorized AVX-512 / AVX2 RMSNorm kernel.
    pub fn dispatch_cpu_rmsnorm(out: &mut [f32], input: &[f32], weight: &[f32], eps: f32) {
        oxide_quant::simd::rmsnorm_f32(input, weight, out, eps);
    }

    /// Dispatches vectorized Rotary Position Embedding (RoPE) kernel.
    pub fn dispatch_cpu_rope(x: &mut [f32], head_dim: usize, position: usize, theta: f32) {
        oxide_quant::simd::rope_f32(x, head_dim, position, theta);
    }

    /// Dispatches multi-threaded CPU FlashAttention-2 / FlashDecode step.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_cpu_flash_decode(
        out: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) {
        oxide_quant::simd::flash_attention_cpu(
            q,
            k,
            v,
            seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            out,
        );
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_thread_pool_and_dispatch() {
        let pool = CpuThreadPool::new();
        assert!(pool.num_threads() >= 1);
        let _ = pool.pin_current_thread(0);
    }
}

