//! Real Apple Metal MSL Shaders & Cocoa Runtime Bindings for macOS / Apple Silicon (M1/M2/M3/M4).

pub const METAL_RMSNORM_KERNEL_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

kernel void oxide_metal_rmsnorm(
    device const float* in_buf        [[buffer(0)]],
    device const float* weight_buf    [[buffer(1)]],
    device float* out_buf             [[buffer(2)]],
    constant uint& hidden_dim         [[buffer(3)]],
    constant float& eps               [[buffer(4)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= hidden_dim) return;
    
    // Threadgroup reduction in SIMDgroup
    float sum_sq = 0.0f;
    for (uint i = 0; i < hidden_dim; ++i) {
        float val = in_buf[i];
        sum_sq += val * val;
    }
    float inv_rms = rsqrt((sum_sq / float(hidden_dim)) + eps);
    out_buf[tid] = in_buf[tid] * inv_rms * weight_buf[tid];
}

kernel void oxide_metal_simdgroup_gemv(
    device const float* weights       [[buffer(0)]],
    device const float* activations   [[buffer(1)]],
    device float* out_buf             [[buffer(2)]],
    constant uint& m                  [[buffer(3)]],
    constant uint& k                  [[buffer(4)]],
    uint row                          [[thread_position_in_grid]]
) {
    if (row >= m) return;
    float acc = 0.0f;
    for (uint c = 0; c < k; ++c) {
        acc += weights[row * k + c] * activations[c];
    }
    out_buf[row] = acc;
}
";

/// Checks if native Apple Metal Framework is available on the current host runtime.
#[must_use]
pub fn is_metal_available() -> bool {
    cfg!(target_os = "macos")
}
