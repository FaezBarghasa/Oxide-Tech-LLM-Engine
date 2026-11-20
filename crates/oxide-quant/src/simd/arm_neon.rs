//! ARM64 NEON micro-kernels with `dotprod` (`vdotq_s32`, `vdotq_u32`) support.
//!
//! Provides hardware-accelerated integer dot-products for `Q8_0` and `Q4_0` on AArch64.

#![allow(
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::undocumented_unsafe_blocks
)]

/// Detects if ARM64 NEON with dotprod extension is available at runtime.
#[must_use]
pub fn is_neon_dotprod_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Compute 32 INT8 * 32 INT8 dot product using ARM64 NEON `vdotq_s32`.
///
/// # Safety
/// Caller must ensure CPU supports ARM64 NEON dotprod extension.
#[inline(always)]
pub unsafe fn dot_i8_neon_dotprod(a: &[i8; 32], b: &[i8; 32]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            let va0 = vld1q_s8(a.as_ptr());
            let va1 = vld1q_s8(a.as_ptr().add(16));
            let vb0 = vld1q_s8(b.as_ptr());
            let vb1 = vld1q_s8(b.as_ptr().add(16));

            let zero = vdupq_n_s32(0);
            let acc0 = vdotq_s32(zero, va0, vb0);
            let acc1 = vdotq_s32(acc0, va1, vb1);

            vaddvq_s32(acc1)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0i32;
        for i in 0..32 {
            sum += (a[i] as i32) * (b[i] as i32);
        }
        sum
    }
}

/// Compute `Q8_0` dot product using ARM64 NEON.
#[inline(always)]
pub unsafe fn dot_q8_0_neon(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for i in (0..32).step_by(4) {
                let q_i8 = vld1_s8(qs.as_ptr().add(i));
                let q_i16 = vmovl_s8(q_i8);
                let q_i32 = vmovl_s16(vget_low_s16(q_i16));
                let q_f32 = vcvtq_f32_s32(q_i32);
                let a_f32 = vld1q_f32(act.as_ptr().add(i));
                acc = vfmaq_f32(acc, q_f32, a_f32);
            }
            vaddvq_f32(acc) * scale
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += (qs[i] as f32) * act[i];
        }
        sum * scale
    }
}
