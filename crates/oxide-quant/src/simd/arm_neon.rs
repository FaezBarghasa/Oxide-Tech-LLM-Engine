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

/// Compute `Q4_K` by `Q8_K` block dot-product using ARM64 NEON `dotprod` (`vdotq_s32`).
/// Processes a 256-element block.
#[inline(always)]
pub unsafe fn dot_q4_k_q8_k_block_neon(
    w: &crate::gguf_quants::BlockQ4_K,
    y: &crate::int_quant::BlockQ8_K,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        unsafe {
            let d0 = y.d * w.d.to_f32();
            let dmin0 = -y.d * w.dmin.to_f32();

            let mut sum_acc = vdupq_n_s32(0);
            let mut min_acc = 0i32;

            let act_ptr = y.qs.as_ptr();
            let q_ptr = w.qs.as_ptr();

            for chunk in 0..4 {
                let is = chunk * 2;
                let (sc1, m1) = crate::gguf_quants::get_scale_min_k4(is, &w.scales);
                let (sc2, m2) = crate::gguf_quants::get_scale_min_k4(is + 1, &w.scales);

                let sum_a1 = (y.bsums[chunk * 4] as i32) + (y.bsums[chunk * 4 + 1] as i32);
                let sum_a2 = (y.bsums[chunk * 4 + 2] as i32) + (y.bsums[chunk * 4 + 3] as i32);
                min_acc += (m1 as i32) * sum_a1 + (m2 as i32) * sum_a2;

                let act_base = chunk * 64;
                let q_base = chunk * 32;

                // Load 32 bytes of packed weights
                let q_bytes = vld1q_u8(q_ptr.add(q_base));
                let m4 = vdupq_n_u8(0x0F);
                let q1 = vandq_u8(q_bytes, m4);
                let q2 = vandq_u8(vshrq_n_u8(q_bytes, 4), m4);

                let a1_0 = vld1q_s8(act_ptr.add(act_base).cast());
                let a1_1 = vld1q_s8(act_ptr.add(act_base + 16).cast());
                let a2_0 = vld1q_s8(act_ptr.add(act_base + 32).cast());
                let a2_1 = vld1q_s8(act_ptr.add(act_base + 48).cast());

                let mut p1 = vdupq_n_s32(0);
                p1 = vdotq_s32(
                    p1,
                    vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(q1), vget_low_u8(q1))),
                    a1_0,
                );
                p1 = vdotq_s32(
                    p1,
                    vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(q1), vget_high_u8(q1))),
                    a1_1,
                );

                let mut p2 = vdupq_n_s32(0);
                p2 = vdotq_s32(
                    p2,
                    vreinterpretq_s8_u8(vcombine_u8(vget_low_u8(q2), vget_low_u8(q2))),
                    a2_0,
                );
                p2 = vdotq_s32(
                    p2,
                    vreinterpretq_s8_u8(vcombine_u8(vget_high_u8(q2), vget_high_u8(q2))),
                    a2_1,
                );

                let sc1_v = vdupq_n_s32(sc1 as i32);
                let sc2_v = vdupq_n_s32(sc2 as i32);
                sum_acc = vmlaq_s32(sum_acc, p1, sc1_v);
                sum_acc = vmlaq_s32(sum_acc, p2, sc2_v);
            }

            let bsum = vaddvq_s32(sum_acc);
            d0 * (bsum as f32) + dmin0 * (min_acc as f32)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        crate::simd::scalar::dot_q4_k_q8_k_row_portable(
            std::slice::from_ref(w),
            std::slice::from_ref(y),
        )
    }
}
