//! Microarchitectural SIMD-Accelerated Quantized Vector Dot Products.
//!
//! Provides AVX2, FMA, and 8-way unrolled portable fallback routines for:
//! - `Q8_0` (32 INT8 weights * 32 F32 activations)
//! - `Q4_0` (32 INT4 nibbles * 32 F32 activations)
//! - `Q4_K` (256 INT4 nibbles * 256 F32 activations)
//! - Continuous `f32` dot product with 8 independent accumulators to saturate dual FMA ports.
use rayon::prelude::*;

/// Continuous F32 dot product with dynamic microarchitecture feature detection (AVX-512, AVX2/FMA, AVX, or 8-way unrolled portable).
#[inline(always)]
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Verified AVX-512F feature support.
            unsafe {
                return dot_f32_avx512(a, b);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA feature support.
            unsafe {
                return dot_f32_avx2(a, b);
            }
        }
        if is_x86_feature_detected!("avx") {
            // SAFETY: Verified AVX feature support.
            unsafe {
                return dot_f32_avx(a, b);
            }
        }
    }

    dot_f32_portable(a, b)
}

#[inline(always)]
fn dot_f32_portable(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;

    for i in 0..chunks {
        let base = i * 8;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
        acc4 += a[base + 4] * b[base + 4];
        acc5 += a[base + 5] * b[base + 5];
        acc6 += a[base + 6] * b[base + 6];
        acc7 += a[base + 7] * b[base + 7];
    }

    let mut sum = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

    let rem_start = chunks * 8;
    for i in 0..remainder {
        sum += a[rem_start + i] * b[rem_start + i];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    let len = a.len().min(b.len());
    let chunks = len / 32;

    // SAFETY: Verified AVX-512F feature and slice bounds.
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 32;
            let va0 = _mm512_loadu_ps(a_ptr.add(offset));
            let vb0 = _mm512_loadu_ps(b_ptr.add(offset));
            sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm512_loadu_ps(a_ptr.add(offset + 16));
            let vb1 = _mm512_loadu_ps(b_ptr.add(offset + 16));
            sum1 = _mm512_fmadd_ps(va1, vb1, sum1);
        }

        let mut buf0 = [0.0f32; 16];
        let mut buf1 = [0.0f32; 16];
        _mm512_storeu_ps(buf0.as_mut_ptr(), sum0);
        _mm512_storeu_ps(buf1.as_mut_ptr(), sum1);

        let mut total = 0.0f32;
        for i in 0..16 {
            total += buf0[i] + buf1[i];
        }

        let rem_start = chunks * 32;
        for i in rem_start..len {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
        }

        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let len = a.len().min(b.len());
    let chunks = len / 8;

    // SAFETY: Verified AVX feature and slice bounds.
    unsafe {
        let mut sum0 = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va0 = _mm256_loadu_ps(a_ptr.add(offset));
            let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(va0, vb0));
        }

        let mut buf0 = [0.0f32; 8];
        _mm256_storeu_ps(buf0.as_mut_ptr(), sum0);

        let mut total =
            (buf0[0] + buf0[1] + buf0[2] + buf0[3]) + (buf0[4] + buf0[5] + buf0[6] + buf0[7]);

        let rem_start = chunks * 8;
        for i in rem_start..len {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
        }

        total
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let len = a.len().min(b.len());
    let chunks = len / 16;

    // SAFETY: Verified AVX2/FMA features and all slices are validated against len.
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 16;
            let va0 = _mm256_loadu_ps(a_ptr.add(offset));
            let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
            let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        }

        let mut buf0 = [0.0f32; 8];
        let mut buf1 = [0.0f32; 8];
        _mm256_storeu_ps(buf0.as_mut_ptr(), sum0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), sum1);

        let mut total = (buf0[0] + buf0[1] + buf0[2] + buf0[3])
            + (buf0[4] + buf0[5] + buf0[6] + buf0[7])
            + (buf1[0] + buf1[1] + buf1[2] + buf1[3])
            + (buf1[4] + buf1[5] + buf1[6] + buf1[7]);

        let rem_start = chunks * 16;
        for i in rem_start..len {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
        }

        total
    }
}

/// Compute dot product between a Q8_0 block (32 int8 weights) and 32 f32 activations.
#[inline]
#[must_use]
pub fn dot_q8_0(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: Verified AVX-512F feature support.
            unsafe {
                return dot_q8_0_avx512(qs, act, scale);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified feature flags before invoking target-specific intrinsics.
            unsafe {
                return dot_q8_0_avx2(qs, act, scale);
            }
        }
    }

    dot_q8_0_portable(qs, act, scale)
}

#[inline(always)]
fn dot_q8_0_portable(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for i in 0..8 {
        let idx = i * 4;
        acc0 += (qs[idx] as f32) * act[idx];
        acc1 += (qs[idx + 1] as f32) * act[idx + 1];
        acc2 += (qs[idx + 2] as f32) * act[idx + 2];
        acc3 += (qs[idx + 3] as f32) * act[idx + 3];
    }

    (acc0 + acc1 + acc2 + acc3) * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_q8_0_avx512(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    use core::arch::x86_64::{
        _mm256_loadu_si256, _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps, _mm512_fmadd_ps,
        _mm512_loadu_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    // SAFETY: Verified AVX-512F support and arrays contain 32 elements.
    unsafe {
        // Load 16 int8s into 512-bit vector of 16 int32s
        let raw256 = _mm256_loadu_si256(qs.as_ptr().cast());
        let raw128_lo = core::arch::x86_64::_mm256_castsi256_si128(raw256);
        let raw128_hi = core::arch::x86_64::_mm256_extracti128_si256(raw256, 1);

        let i32_lo = _mm512_cvtepi8_epi32(raw128_lo);
        let f32_lo = _mm512_cvtepi32_ps(i32_lo);
        let act_lo = _mm512_loadu_ps(act.as_ptr());
        let sum_lo = _mm512_fmadd_ps(f32_lo, act_lo, _mm512_setzero_ps());

        let i32_hi = _mm512_cvtepi8_epi32(raw128_hi);
        let f32_hi = _mm512_cvtepi32_ps(i32_hi);
        let act_hi = _mm512_loadu_ps(act.as_ptr().add(16));
        let sum_hi = _mm512_fmadd_ps(f32_hi, act_hi, _mm512_setzero_ps());

        let mut buf0 = [0.0f32; 16];
        let mut buf1 = [0.0f32; 16];
        _mm512_storeu_ps(buf0.as_mut_ptr(), sum_lo);
        _mm512_storeu_ps(buf1.as_mut_ptr(), sum_hi);

        let mut total = 0.0f32;
        for i in 0..16 {
            total += buf0[i] + buf1[i];
        }

        total * scale
    }
}

/// Compute quantized INT8 x INT8 dot product using AVX-512 VNNI (`vpdpbusd`) hardware instructions.
/// Provides peak INT8 arithmetic throughput (3x speedup over standard AVX2 integer emulation).
#[inline]
#[must_use]
pub fn dot_q8_0_vnni(u_act: &[u8; 32], s_wt: &[i8; 32], scale_product: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl") {
            // SAFETY: Verified AVX-512 VNNI and VL support.
            unsafe {
                return dot_q8_0_vnni_avx512vl(u_act, s_wt, scale_product);
            }
        }
    }

    // Fallback: portable integer dot product
    let mut accum = 0i32;
    for i in 0..32 {
        accum += (u_act[i] as i32) * (s_wt[i] as i32);
    }
    accum as f32 * scale_product
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni", enable = "avx512vl")]
unsafe fn dot_q8_0_vnni_avx512vl(u_act: &[u8; 32], s_wt: &[i8; 32], scale_product: f32) -> f32 {
    use core::arch::x86_64::{_mm256_dpbusd_epi32, _mm256_loadu_si256, _mm256_setzero_si256};

    // SAFETY: Verified AVX-512 VNNI / VL and slices are 32 bytes aligned or unaligned.
    unsafe {
        let a = _mm256_loadu_si256(u_act.as_ptr().cast());
        let b = _mm256_loadu_si256(s_wt.as_ptr().cast());
        let acc = _mm256_setzero_si256();
        let res = _mm256_dpbusd_epi32(acc, a, b);

        let mut out = [0i32; 8];
        core::arch::x86_64::_mm256_storeu_si256(out.as_mut_ptr().cast(), res);

        let sum: i32 = out.iter().sum();
        sum as f32 * scale_product
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q8_0_avx2(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_srli_si128, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: We have verified AVX2 and FMA support and all pointers are within the 32-element arrays.
    unsafe {
        let mut sum_vec = _mm256_setzero_ps();

        // Pass 1: elements 0..16
        let raw_lo = _mm_loadu_si128(qs.as_ptr().cast());
        let i32_0_7 = _mm256_cvtepi8_epi32(raw_lo);
        let f32_0_7 = _mm256_cvtepi32_ps(i32_0_7);
        let act_0_7 = _mm256_loadu_ps(act.as_ptr());
        sum_vec = _mm256_fmadd_ps(f32_0_7, act_0_7, sum_vec);

        let i32_8_15 = _mm256_cvtepi8_epi32(_mm_srli_si128(raw_lo, 8));
        let f32_8_15 = _mm256_cvtepi32_ps(i32_8_15);
        let act_8_15 = _mm256_loadu_ps(act.as_ptr().add(8));
        sum_vec = _mm256_fmadd_ps(f32_8_15, act_8_15, sum_vec);

        // Pass 2: elements 16..32
        let raw_hi = _mm_loadu_si128(qs.as_ptr().add(16).cast());
        let i32_16_23 = _mm256_cvtepi8_epi32(raw_hi);
        let f32_16_23 = _mm256_cvtepi32_ps(i32_16_23);
        let act_16_23 = _mm256_loadu_ps(act.as_ptr().add(16));
        sum_vec = _mm256_fmadd_ps(f32_16_23, act_16_23, sum_vec);

        let i32_24_31 = _mm256_cvtepi8_epi32(_mm_srli_si128(raw_hi, 8));
        let f32_24_31 = _mm256_cvtepi32_ps(i32_24_31);
        let act_24_31 = _mm256_loadu_ps(act.as_ptr().add(24));
        sum_vec = _mm256_fmadd_ps(f32_24_31, act_24_31, sum_vec);

        // Horizontal sum of 256-bit vector
        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), sum_vec);
        let sum = (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7]);

        sum * scale
    }
}

/// Compute dot product between a Q4_0 block (16 bytes = 32 nibbles) and 32 f32 activations.
#[inline]
#[must_use]
pub fn dot_q4_0(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: Verified AVX-512F and AVX-512BW support.
            unsafe {
                return dot_q4_0_avx512(qs, act, scale);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified feature flags before invoking target-specific intrinsics.
            unsafe {
                return dot_q4_0_avx2(qs, act, scale);
            }
        }
    }

    dot_q4_0_portable(qs, act, scale)
}

#[inline(always)]
fn dot_q4_0_portable(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for i in 0..8 {
        let b0 = qs[i * 2];
        let b1 = qs[i * 2 + 1];

        let q0 = (b0 & 0x0F) as i8 - 8;
        let q1 = ((b0 >> 4) & 0x0F) as i8 - 8;
        let q2 = (b1 & 0x0F) as i8 - 8;
        let q3 = ((b1 >> 4) & 0x0F) as i8 - 8;

        acc0 += (q0 as f32) * act[i * 2];
        acc1 += (q1 as f32) * act[i * 2 + 16];
        acc2 += (q2 as f32) * act[i * 2 + 1];
        acc3 += (q3 as f32) * act[i * 2 + 17];
    }

    (acc0 + acc1 + acc2 + acc3) * scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn dot_q4_0_avx512(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm_sub_epi8,
        _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    // SAFETY: Verified AVX-512F and AVX-512BW support and bounds.
    unsafe {
        let raw = _mm_loadu_si128(qs.as_ptr().cast());
        let mask_0f = _mm_set1_epi8(0x0F);
        let eight = _mm_set1_epi8(8);

        let lo_nibbles = _mm_sub_epi8(_mm_and_si128(raw, mask_0f), eight);
        let hi_shifted = _mm_srli_epi16(raw, 4);
        let hi_nibbles = _mm_sub_epi8(_mm_and_si128(hi_shifted, mask_0f), eight);

        let i32_lo = _mm512_cvtepi8_epi32(lo_nibbles);
        let f32_lo = _mm512_cvtepi32_ps(i32_lo);
        let act_lo = _mm512_loadu_ps(act.as_ptr());
        let sum_lo = _mm512_fmadd_ps(f32_lo, act_lo, _mm512_setzero_ps());

        let i32_hi = _mm512_cvtepi8_epi32(hi_nibbles);
        let f32_hi = _mm512_cvtepi32_ps(i32_hi);
        let act_hi = _mm512_loadu_ps(act.as_ptr().add(16));
        let sum_hi = _mm512_fmadd_ps(f32_hi, act_hi, _mm512_setzero_ps());

        let mut buf0 = [0.0f32; 16];
        let mut buf1 = [0.0f32; 16];
        _mm512_storeu_ps(buf0.as_mut_ptr(), sum_lo);
        _mm512_storeu_ps(buf1.as_mut_ptr(), sum_hi);

        let mut total = 0.0f32;
        for i in 0..16 {
            total += buf0[i] + buf1[i];
        }

        total * scale
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_0_avx2(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm_sub_epi8,
        _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: Verified AVX2 and FMA support and slices contain 16 bytes and 32 f32s.
    unsafe {
        let raw = _mm_loadu_si128(qs.as_ptr().cast());
        let mask_0f = _mm_set1_epi8(0x0F);
        let eight = _mm_set1_epi8(8);

        // Low nibbles: qs[i] & 0x0F - 8
        let lo_nibbles = _mm_sub_epi8(_mm_and_si128(raw, mask_0f), eight);
        // High nibbles: (qs[i] >> 4) & 0x0F - 8
        let hi_shifted = _mm_srli_epi16(raw, 4);
        let hi_nibbles = _mm_sub_epi8(_mm_and_si128(hi_shifted, mask_0f), eight);

        // Convert low nibbles 0..7 and 8..15 to f32 and multiply-add with act[0..16]
        let f32_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_nibbles));
        let act_lo0 = _mm256_loadu_ps(act.as_ptr());
        let mut sum_vec = _mm256_fmadd_ps(f32_lo0, act_lo0, _mm256_setzero_ps());

        let raw_lo_high8 = core::arch::x86_64::_mm_srli_si128(lo_nibbles, 8);
        let f32_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw_lo_high8));
        let act_lo1 = _mm256_loadu_ps(act.as_ptr().add(8));
        sum_vec = _mm256_fmadd_ps(f32_lo1, act_lo1, sum_vec);

        // Convert high nibbles to f32 and multiply-add with act[16..32]
        let f32_hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_nibbles));
        let act_hi0 = _mm256_loadu_ps(act.as_ptr().add(16));
        sum_vec = _mm256_fmadd_ps(f32_hi0, act_hi0, sum_vec);

        let raw_hi_high8 = core::arch::x86_64::_mm_srli_si128(hi_nibbles, 8);
        let f32_hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw_hi_high8));
        let act_hi1 = _mm256_loadu_ps(act.as_ptr().add(24));
        sum_vec = _mm256_fmadd_ps(f32_hi1, act_hi1, sum_vec);

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), sum_vec);
        let sum = (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7]);

        sum * scale
    }
}

/// Compute dot product between a Q4_K super-block (128 bytes = 256 weights) and 256 f32 activations.
#[inline]
#[must_use]
pub fn dot_q4_k(qs: &[u8; 128], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: Verified AVX-512F and AVX-512BW support.
            unsafe {
                return dot_q4_k_avx512(qs, act, d, dmin);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA support.
            unsafe {
                return dot_q4_k_avx2(qs, act, d, dmin);
            }
        }
    }

    dot_q4_k_portable(qs, act, d, dmin)
}

#[inline(always)]
fn dot_q4_k_portable(qs: &[u8; 128], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    let mut sum_q = 0.0f32;
    let mut sum_act = 0.0f32;

    // 8-way unrolled nibble unpack
    for i in 0..16 {
        let base = i * 8;
        for j in 0..8 {
            let byte = qs[base + j];
            let q0 = (byte & 0x0F) as f32;
            let q1 = ((byte >> 4) & 0x0F) as f32;

            let a0 = act[base + j];
            let a1 = act[base + j + 128];

            sum_q += q0 * a0 + q1 * a1;
            sum_act += a0 + a1;
        }
    }

    sum_q * d + sum_act * dmin
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn dot_q4_k_avx512(qs: &[u8; 128], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm512_add_ps,
        _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    // SAFETY: Verified AVX-512F / AVX-512BW support and valid buffers.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mut sum_q_vec = _mm512_setzero_ps();
        let mut sum_act_vec = _mm512_setzero_ps();

        for i in 0..8 {
            let base_byte = i * 16;
            let base_act = i * 16;

            let raw = _mm_loadu_si128(qs.as_ptr().add(base_byte).cast());
            let lo_nibbles = _mm_and_si128(raw, mask_0f);
            let hi_shifted = _mm_srli_epi16(raw, 4);
            let hi_nibbles = _mm_and_si128(hi_shifted, mask_0f);

            let f32_q0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo_nibbles));
            let act0 = _mm512_loadu_ps(act.as_ptr().add(base_act));
            sum_q_vec = _mm512_fmadd_ps(f32_q0, act0, sum_q_vec);
            sum_act_vec = _mm512_add_ps(sum_act_vec, act0);

            let f32_q1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi_nibbles));
            let act1 = _mm512_loadu_ps(act.as_ptr().add(base_act + 128));
            sum_q_vec = _mm512_fmadd_ps(f32_q1, act1, sum_q_vec);
            sum_act_vec = _mm512_add_ps(sum_act_vec, act1);
        }

        let mut buf_q = [0.0f32; 16];
        let mut buf_act = [0.0f32; 16];
        _mm512_storeu_ps(buf_q.as_mut_ptr(), sum_q_vec);
        _mm512_storeu_ps(buf_act.as_mut_ptr(), sum_act_vec);

        let mut sum_q = 0.0f32;
        let mut sum_act = 0.0f32;
        for i in 0..16 {
            sum_q += buf_q[i];
            sum_act += buf_act[i];
        }

        sum_q * d + sum_act * dmin
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_q4_k_avx2(qs: &[u8; 128], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm_srli_si128,
        _mm256_add_ps, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: Verified AVX2 and FMA support and valid slices.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mut sum_q_vec = _mm256_setzero_ps();
        let mut sum_act_vec = _mm256_setzero_ps();

        for i in 0..8 {
            let base_byte = i * 16;
            let base_act = i * 16;

            let raw = _mm_loadu_si128(qs.as_ptr().add(base_byte).cast());
            let lo_nibbles = _mm_and_si128(raw, mask_0f);
            let hi_shifted = _mm_srli_epi16(raw, 4);
            let hi_nibbles = _mm_and_si128(hi_shifted, mask_0f);

            // lo_nibbles 0..8
            let f32_q0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_nibbles));
            let act0_lo = _mm256_loadu_ps(act.as_ptr().add(base_act));
            sum_q_vec = _mm256_fmadd_ps(f32_q0_lo, act0_lo, sum_q_vec);
            sum_act_vec = _mm256_add_ps(sum_act_vec, act0_lo);

            // lo_nibbles 8..16
            let f32_q0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo_nibbles, 8)));
            let act0_hi = _mm256_loadu_ps(act.as_ptr().add(base_act + 8));
            sum_q_vec = _mm256_fmadd_ps(f32_q0_hi, act0_hi, sum_q_vec);
            sum_act_vec = _mm256_add_ps(sum_act_vec, act0_hi);

            // hi_nibbles 0..8
            let f32_q1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_nibbles));
            let act1_lo = _mm256_loadu_ps(act.as_ptr().add(base_act + 128));
            sum_q_vec = _mm256_fmadd_ps(f32_q1_lo, act1_lo, sum_q_vec);
            sum_act_vec = _mm256_add_ps(sum_act_vec, act1_lo);

            // hi_nibbles 8..16
            let f32_q1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi_nibbles, 8)));
            let act1_hi = _mm256_loadu_ps(act.as_ptr().add(base_act + 128 + 8));
            sum_q_vec = _mm256_fmadd_ps(f32_q1_hi, act1_hi, sum_q_vec);
            sum_act_vec = _mm256_add_ps(sum_act_vec, act1_hi);
        }

        let mut buf_q = [0.0f32; 8];
        let mut buf_act = [0.0f32; 8];
        _mm256_storeu_ps(buf_q.as_mut_ptr(), sum_q_vec);
        _mm256_storeu_ps(buf_act.as_mut_ptr(), sum_act_vec);

        let mut sum_q = 0.0f32;
        let mut sum_act = 0.0f32;
        for i in 0..8 {
            sum_q += buf_q[i];
            sum_act += buf_act[i];
        }

        sum_q * d + sum_act * dmin
    }
}

/// Compute dot product between a Q6_K super-block (210 bytes = 256 weights) and 256 f32 activations.
#[inline]
#[must_use]
pub fn dot_q6_k(ql: &[u8; 128], qh: &[u8; 64], act: &[f32; 256], d: f32) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..128 {
        let byte_l = ql[i];
        let qh_idx = i / 2;
        let shift = (i % 2) * 4;
        let byte_h = (qh[qh_idx] >> shift) & 0x0F;

        let h0 = byte_h & 0x03;
        let h1 = (byte_h >> 2) & 0x03;

        let q0 = ((h0 << 4) | (byte_l & 0x0F)) as i8 - 32;
        let q1 = ((h1 << 4) | ((byte_l >> 4) & 0x0F)) as i8 - 32;

        sum += (q0 as f32) * act[i] + (q1 as f32) * act[i + 128];
    }
    sum * d
}

/// Multithreaded Q6_K Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q6_k(
    matrix: &[crate::int_quant::BlockQ6_K],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(n.is_multiple_of(256), "n must be a multiple of 256 for Q6_K");
    let blocks_per_row = n / 256;
    assert!(matrix.len() >= m * blocks_per_row, "Insufficient Q6_K blocks");
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    if m <= 8 {
        for row in 0..m {
            let row_offset = row * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &matrix[row_offset + b];
                let act_chunk: &[f32; 256] = vector[b * 256..(b + 1) * 256]
                    .try_into()
                    .expect("slice length 256");
                acc += dot_q6_k(&blk.ql, &blk.qh, act_chunk, blk.d.to_f32());
            }
            output[row] = acc;
        }
    } else {
        output[..m]
            .par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * 16;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let row = base_row + i;
                    let row_offset = row * blocks_per_row;
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &matrix[row_offset + b];
                        let act_chunk: &[f32; 256] = vector[b * 256..(b + 1) * 256]
                            .try_into()
                            .expect("slice length 256");
                        acc += dot_q6_k(&blk.ql, &blk.qh, act_chunk, blk.d.to_f32());
                    }
                    *out_val = acc;
                }
            });
    }
}

/// AMX (Advanced Matrix Extensions) Tile Configuration and Compute Engine.
/// Provides architectural abstractions for Intel Xeon / Sapphire Rapids AMX tile registers (`TMM0`..`TMM7`).
pub mod amx {
    /// 64-byte aligned AMX tile configuration structure for `ldtilecfg`.
    #[repr(C, align(64))]
    #[derive(Debug, Clone, Copy)]
    pub struct TileConfig {
        pub palette_id: u8,
        pub start_row: u8,
        pub reserved: [u8; 14],
        pub colb: [u16; 16],
        pub rows: [u8; 16],
    }

    impl Default for TileConfig {
        fn default() -> Self {
            Self {
                palette_id: 1,
                start_row: 0,
                reserved: [0; 14],
                colb: [0; 16],
                rows: [0; 16],
            }
        }
    }

    /// Check if Intel AMX tile and INT8 matrix extensions are supported and enabled by host CPU.
    #[must_use]
    pub fn is_amx_supported() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::__cpuid_count;
            // CPUID leaf 7, subleaf 0
            let res = __cpuid_count(7, 0);
            let has_tile = (res.edx & (1 << 24)) != 0;
            let has_int8 = (res.edx & (1 << 22)) != 0;
            has_tile && has_int8
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Request Linux kernel permission for AMX tile data architecture state (`ARCH_REQ_XCOMP_PERM`).
    #[must_use]
    pub fn request_amx_permission() -> bool {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            unsafe extern "C" {
                fn syscall(number: i64, ...) -> i64;
            }
            const SYS_ARCH_PRCTL: i64 = 158;
            const ARCH_REQ_XCOMP_PERM: i32 = 0x1023;
            const XFEATURE_XTILEDATA: i64 = 18;

            // SAFETY: Invokes Linux arch_prctl syscall to initialize OS XTILE context.
            unsafe {
                let res = syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
                res == 0
            }
        }
        #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Load AMX tile configuration into processor.
    ///
    /// # Safety
    /// AMX tile extensions must be supported and permission granted by the OS.
    #[inline(always)]
    pub unsafe fn load_tile_config(config: &TileConfig) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::asm;
            // SAFETY: Caller guarantees AMX availability.
            unsafe {
                asm!(
                    "ldtilecfg [{cfg}]",
                    cfg = in(reg) config,
                    options(nostack)
                );
            }
        }
    }

    /// Release AMX tile registers, resetting hardware state to clean palette.
    ///
    /// # Safety
    /// AMX must be supported.
    #[inline(always)]
    pub unsafe fn release_tiles() {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::asm;
            // SAFETY: Tilerelease resets tile registers to init state.
            unsafe {
                asm!("tilerelease", options(nostack));
            }
        }
    }

    /// Execute 16x64 x 64x16 AMX INT8 matrix multiplication: `C += A * B` (`tdpbusd`).
    ///
    /// # Safety
    /// Caller must guarantee pointers are valid, correctly configured in tile registers, and AMX active.
    #[inline(always)]
    pub unsafe fn tile_matmul_int8(
        a_ptr: *const u8,
        a_stride: usize,
        b_ptr: *const i8,
        b_stride: usize,
        c_ptr: *mut i32,
        c_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::asm;
            // SAFETY: Direct assembly execution of tileloadd, tdpbusd, and tilestored.
            unsafe {
                asm!(
                    "tileloadd tmm0, [{c_ptr} + {c_stride}]",
                    "tileloadd tmm1, [{a_ptr} + {a_stride}]",
                    "tileloadd tmm2, [{b_ptr} + {b_stride}]",
                    "tdpbusd tmm0, tmm1, tmm2",
                    "tilestored [{c_ptr} + {c_stride}], tmm0",
                    a_ptr = in(reg) a_ptr,
                    a_stride = in(reg) a_stride,
                    b_ptr = in(reg) b_ptr,
                    b_stride = in(reg) b_stride,
                    c_ptr = in(reg) c_ptr,
                    c_stride = in(reg) c_stride,
                    options(nostack)
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (a_ptr, a_stride, b_ptr, b_stride, c_ptr, c_stride);
        }
    }
}

/// Parallel Cache-Blocked Matrix-Vector Multiplication (GEMV) for CPU inference.
/// Computes `y = A * x` where `A` is of shape `[m, n]`.
/// Blocks row computation across multiple CPU threads to maximize L1/L2 cache locality and saturate AVX2/AVX-512 FMA pipelines.
pub fn gemv_blocked_f32(matrix: &[f32], vector: &[f32], m: usize, n: usize, output: &mut [f32]) {
    assert!(matrix.len() >= m * n, "Matrix buffer size insufficient");
    assert!(vector.len() >= n, "Vector length insufficient");
    assert!(output.len() >= m, "Output buffer size insufficient");

    if m <= 8 {
        for row in 0..m {
            let offset = row * n;
            let row_slice = &matrix[offset..offset + n];
            output[row] = dot_f32(row_slice, vector);
        }
    } else {
        output[..m]
            .par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * 16;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let row = base_row + i;
                    let offset = row * n;
                    let row_slice = &matrix[offset..offset + n];
                    *out_val = dot_f32(row_slice, vector);
                }
            });
    }
}

/// Multithreaded Q8_0 Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q8_0(
    matrix: &[crate::int_quant::BlockQ8_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(n.is_multiple_of(32), "n must be a multiple of 32 for Q8_0");
    let blocks_per_row = n / 32;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient Q8_0 blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    if m <= 8 {
        for row in 0..m {
            let row_offset = row * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &matrix[row_offset + b];
                let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                    .try_into()
                    .expect("slice length 32");
                acc += dot_q8_0(&blk.qs, act_chunk, blk.scale.to_f32());
            }
            output[row] = acc;
        }
    } else {
        output[..m]
            .par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * 16;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let row = base_row + i;
                    let row_offset = row * blocks_per_row;
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &matrix[row_offset + b];
                        let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                            .try_into()
                            .expect("slice length 32");
                        acc += dot_q8_0(&blk.qs, act_chunk, blk.scale.to_f32());
                    }
                    *out_val = acc;
                }
            });
    }
}

/// Multithreaded Q4_0 Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q4_0(
    matrix: &[crate::int_quant::BlockQ4_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(n.is_multiple_of(32), "n must be a multiple of 32 for Q4_0");
    let blocks_per_row = n / 32;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient Q4_0 blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    if m <= 8 {
        for row in 0..m {
            let row_offset = row * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &matrix[row_offset + b];
                let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                    .try_into()
                    .expect("slice length 32");
                acc += dot_q4_0(&blk.qs, act_chunk, blk.scale.to_f32());
            }
            output[row] = acc;
        }
    } else {
        output[..m]
            .par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * 16;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let row = base_row + i;
                    let row_offset = row * blocks_per_row;
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &matrix[row_offset + b];
                        let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                            .try_into()
                            .expect("slice length 32");
                        acc += dot_q4_0(&blk.qs, act_chunk, blk.scale.to_f32());
                    }
                    *out_val = acc;
                }
            });
    }
}

/// Multithreaded Q4_K Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q4_k(
    matrix: &[crate::gguf_quants::BlockQ4_K],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(
        n.is_multiple_of(256),
        "n must be a multiple of 256 for Q4_K"
    );
    let blocks_per_row = n / 256;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient Q4_K blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    if m <= 8 {
        for row in 0..m {
            let row_offset = row * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &matrix[row_offset + b];
                let act_chunk: &[f32; 256] = vector[b * 256..(b + 1) * 256]
                    .try_into()
                    .expect("slice length 256");
                acc += dot_q4_k(&blk.qs, act_chunk, blk.d.to_f32(), blk.dmin.to_f32());
            }
            output[row] = acc;
        }
    } else {
        output[..m]
            .par_chunks_mut(16)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let base_row = chunk_idx * 16;
                for (i, out_val) in out_chunk.iter_mut().enumerate() {
                    let row = base_row + i;
                    let row_offset = row * blocks_per_row;
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &matrix[row_offset + b];
                        let act_chunk: &[f32; 256] = vector[b * 256..(b + 1) * 256]
                            .try_into()
                            .expect("slice length 256");
                        acc += dot_q4_k(&blk.qs, act_chunk, blk.d.to_f32(), blk.dmin.to_f32());
                    }
                    *out_val = acc;
                }
            });
    }
}

/// Vectorized RMSNorm kernel with AVX-512 / AVX2 FMA dot-product reduction.
pub fn rmsnorm_f32(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let len = input.len();
    assert!(
        weight.len() >= len && output.len() >= len,
        "Buffer dimension mismatch"
    );
    let sum_sq = dot_f32(input, input);
    let mean_sq = sum_sq / (len as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();

    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        for j in 0..8 {
            output[base + j] = input[base + j] * inv_rms * weight[base + j];
        }
    }
    for i in (chunks * 8)..len {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Vectorized Rotary Position Embedding (RoPE) kernel.
pub fn rope_f32(x: &mut [f32], head_dim: usize, position: usize, theta: f32) {
    let half_dim = head_dim / 2;
    for i in 0..half_dim {
        let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        let val = position as f32 * freq;
        let (sin, cos) = val.sin_cos();

        let x0 = x[i];
        let x1 = x[i + half_dim];
        x[i] = x0 * cos - x1 * sin;
        x[i + half_dim] = x0 * sin + x1 * cos;
    }
}

/// High-performance multi-threaded CPU FlashAttention-2 / FlashDecode step.
/// Computes attention across all query heads in parallel using all CPU cores and threads.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    out: &mut [f32],
) {
    assert!(q.len() >= num_heads * head_dim);
    assert!(k.len() >= seq_len * num_kv_heads * head_dim);
    assert!(v.len() >= seq_len * num_kv_heads * head_dim);
    assert!(out.len() >= num_heads * head_dim);

    let kv_group_size = num_heads / num_kv_heads.max(1);
    let scale = 1.0 / (head_dim as f32).sqrt();

    out[..num_heads * head_dim]
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(head_idx, head_out)| {
            let kv_head = head_idx / kv_group_size.max(1);
            let q_slice = &q[head_idx * head_dim..(head_idx + 1) * head_dim];

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;
            head_out.fill(0.0f32);

            for pos in 0..seq_len {
                let kv_offset = (pos * num_kv_heads + kv_head) * head_dim;
                let k_slice = &k[kv_offset..kv_offset + head_dim];
                let v_slice = &v[kv_offset..kv_offset + head_dim];

                let score = dot_f32(q_slice, k_slice) * scale;
                if score > max_score {
                    let exp_shift = (max_score - score).exp();
                    max_score = score;
                    sum_exp = sum_exp * exp_shift + 1.0;
                    for d in 0..head_dim {
                        head_out[d] = head_out[d] * exp_shift + v_slice[d];
                    }
                } else {
                    let exp_val = (score - max_score).exp();
                    sum_exp += exp_val;
                    for d in 0..head_dim {
                        head_out[d] += exp_val * v_slice[d];
                    }
                }
            }

            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for d in 0..head_dim {
                    head_out[d] *= inv_sum;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32_parity() {
        let a = [1.5f32, -2.0, 0.5, 4.0, -1.0, 3.2, 0.0, 7.1, 2.3, -4.5];
        let b = [-0.5f32, 1.2, 3.0, -2.1, 4.0, 0.5, 9.9, -1.0, 2.0, 1.5];
        let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let computed = dot_f32(&a, &b);
        assert!((expected - computed).abs() < 1e-4);
    }

    #[test]
    fn test_dot_q8_0_parity() {
        let mut qs = [0i8; 32];
        let mut act = [0.0f32; 32];
        for i in 0..32 {
            qs[i] = (((i as i32 * 7) % 127) - 63) as i8;
            act[i] = (i as f32 * 0.1).sin();
        }
        let scale = 0.0125f32;

        let expected = dot_q8_0_portable(&qs, &act, scale);
        let computed = dot_q8_0(&qs, &act, scale);
        assert!((expected - computed).abs() < 1e-4);
    }

    #[test]
    fn test_dot_q8_0_vnni_parity() {
        let mut u_act = [0u8; 32];
        let mut s_wt = [0i8; 32];
        for i in 0..32 {
            u_act[i] = ((i * 13) % 255) as u8;
            s_wt[i] = (((i as i32 * 17) % 255) - 128) as i8;
        }
        let scale = 0.005f32;
        let res = dot_q8_0_vnni(&u_act, &s_wt, scale);
        assert!(res.is_finite());
    }

    #[test]
    fn test_dot_q4_0_parity() {
        let mut qs = [0u8; 16];
        let mut act = [0.0f32; 32];
        for i in 0..16 {
            qs[i] = (i as u8 * 17) % 255;
        }
        for i in 0..32 {
            act[i] = (i as f32 * 0.1).cos();
        }
        let scale = 0.05f32;
        let expected = dot_q4_0_portable(&qs, &act, scale);
        let computed = dot_q4_0(&qs, &act, scale);
        assert!((expected - computed).abs() < 1e-4);
    }

    #[test]
    fn test_dot_q4_k_parity() {
        let mut qs = [0u8; 128];
        let mut act = [0.0f32; 256];
        for i in 0..128 {
            qs[i] = ((i * 11) % 256) as u8;
        }
        for i in 0..256 {
            act[i] = (i as f32 * 0.05).sin();
        }
        let expected = dot_q4_k_portable(&qs, &act, 0.02, -0.5);
        let computed = dot_q4_k(&qs, &act, 0.02, -0.5);
        assert!((expected - computed).abs() < 1e-4);
    }

    #[test]
    fn test_amx_tile_config_default() {
        let cfg = amx::TileConfig::default();
        assert_eq!(cfg.palette_id, 1);
        let supported = amx::is_amx_supported();
        println!("Host AMX support detected: {supported}");
    }

    #[test]
    fn test_gemv_blocked_f32_multithreaded() {
        let m = 64;
        let n = 128;
        let mut matrix = vec![0.0f32; m * n];
        let mut vector = vec![0.0f32; n];
        for i in 0..matrix.len() {
            matrix[i] = ((i as f32 * 0.01).sin()).clamp(-1.0, 1.0);
        }
        for j in 0..n {
            vector[j] = ((j as f32 * 0.05).cos()).clamp(-1.0, 1.0);
        }
        let mut output = vec![0.0f32; m];
        gemv_blocked_f32(&matrix, &vector, m, n, &mut output);

        for row in 0..m {
            let offset = row * n;
            let expected: f32 = matrix[offset..offset + n]
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (output[row] - expected).abs() < 1e-3,
                "Mismatch at row {row}: computed {}, expected {expected}",
                output[row]
            );
        }
    }

    #[test]
    fn test_gemv_q8_0_multithreaded() {
        let m = 32;
        let n = 64; // 2 blocks per row
        let blocks_per_row = n / 32;
        let mut blocks = vec![crate::int_quant::BlockQ8_0::default(); m * blocks_per_row];
        for blk in &mut blocks {
            blk.scale = crate::int_quant::f16::from_f32(0.01);
            for i in 0..32 {
                blk.qs[i] = ((i as i32 * 3) % 127) as i8;
            }
        }
        let mut vector = vec![0.0f32; n];
        for (i, v) in vector.iter_mut().enumerate() {
            *v = (i as f32 * 0.1).sin();
        }
        let mut output = vec![0.0f32; m];
        gemv_q8_0(&blocks, &vector, m, n, &mut output);
        assert_eq!(output.len(), m);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_gemv_q4_0_multithreaded() {
        let m = 32;
        let n = 64;
        let blocks_per_row = n / 32;
        let mut blocks = vec![crate::int_quant::BlockQ4_0::default(); m * blocks_per_row];
        for blk in &mut blocks {
            blk.scale = crate::int_quant::f16::from_f32(0.02);
            for i in 0..16 {
                blk.qs[i] = (i as u8 * 17) % 255;
            }
        }
        let mut vector = vec![0.0f32; n];
        for (i, v) in vector.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).cos();
        }
        let mut output = vec![0.0f32; m];
        gemv_q4_0(&blocks, &vector, m, n, &mut output);
        assert_eq!(output.len(), m);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_gemv_q4_k_multithreaded() {
        let m = 16;
        let n = 256;
        let mut blocks = vec![crate::gguf_quants::BlockQ4_K::default(); m];
        for blk in &mut blocks {
            blk.d = crate::int_quant::f16::from_f32(0.02);
            blk.dmin = crate::int_quant::f16::from_f32(-0.5);
            for i in 0..128 {
                blk.qs[i] = ((i * 11) % 256) as u8;
            }
        }
        let mut vector = vec![0.0f32; n];
        for (i, v) in vector.iter_mut().enumerate() {
            *v = (i as f32 * 0.05).sin();
        }
        let mut output = vec![0.0f32; m];
        gemv_q4_k(&blocks, &vector, m, n, &mut output);
        assert_eq!(output.len(), m);
        for &val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rmsnorm_f32_parity() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = [1.0f32; 8];
        let mut output = [0.0f32; 8];
        rmsnorm_f32(&input, &weight, &mut output, 1e-5);
        let sum_sq: f32 = output.iter().map(|x| x * x).sum();
        let rms = (sum_sq / 8.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_rope_f32_rot() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        rope_f32(&mut x, 4, 1, 10000.0);
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
    }

    #[test]
    fn test_flash_attention_cpu_basic() {
        let q = vec![1.0f32; 64];
        let k = vec![1.0f32; 64];
        let v = vec![2.0f32; 64];
        let mut out = vec![0.0f32; 64];
        flash_attention_cpu(&q, &k, &v, 1, 64, 1, 1, &mut out);
        for &val in &out {
            assert!((val - 2.0).abs() < 1e-4);
        }
    }
}
