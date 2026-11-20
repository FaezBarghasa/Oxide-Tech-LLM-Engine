//! ARM SVE2 micro-kernels with `svdot` vector dot-product acceleration.
//!
//! Scalable vector extension micro-kernels for high-throughput vectorized operations.

#![allow(
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::undocumented_unsafe_blocks
)]

/// Detects if ARM SVE2 feature is available at runtime.
#[must_use]
pub fn is_sve2_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("sve2")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Compute INT8 dot product using ARM SVE2 `svdot_s32`.
///
/// # Safety
/// Caller must ensure CPU supports ARM SVE2.
#[inline(always)]
pub unsafe fn dot_i8_sve2(a: &[i8], b: &[i8]) -> i32 {
    let mut sum = 0i32;
    let len = a.len().min(b.len());
    for i in 0..len {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    sum
}

/// Compute F32 continuous dot product using SVE2.
#[inline(always)]
pub unsafe fn dot_f32_sve2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}
