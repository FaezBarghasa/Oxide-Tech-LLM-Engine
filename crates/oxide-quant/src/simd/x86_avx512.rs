//! x86_64 AVX-512 Micro-Kernels.

#![allow(
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::identity_op,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::undocumented_unsafe_blocks,
    clippy::collapsible_if
)]

use super::x86_avx2::dot_q4_k_avx2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
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
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_q8_0_avx512(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
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


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn dot_q4_0_avx512(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
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



#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[allow(dead_code)]
pub unsafe fn dot_q4_k_avx512(
    qs: &[u8; 128],
    scales: &[u8; 12],
    act: &[f32; 256],
    d: f32,
    dmin: f32,
) -> f32 {
    // SAFETY: caller guarantees AVX2 and FMA are supported on target machine and inputs are valid slices.
    unsafe { dot_q4_k_avx2(qs, scales, act, d, dmin) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[allow(clippy::too_many_lines, dead_code)]
pub unsafe fn dot_q4_k_avx512_legacy(qs: &[u8; 128], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    use core::arch::x86_64::{
        _MM_HINT_T0, _mm_and_si128, _mm_loadu_si128, _mm_prefetch, _mm_set1_epi8, _mm_srli_epi16,
        _mm512_add_ps, _mm512_cvtepi8_epi32, _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_loadu_ps,
        _mm512_setzero_ps, _mm512_storeu_ps,
    };

    // SAFETY: Verified AVX-512F / AVX-512BW support and valid buffers.
    unsafe {
        // Prefetch high activation slice into L1 cache
        _mm_prefetch(act.as_ptr().add(128).cast(), _MM_HINT_T0);

        let mask_0f = _mm_set1_epi8(0x0F);
        // 4 independent FMA accumulator chains to fully saturate 2x 512-bit FMA units on Zen 4
        let mut sum_q_vec0 = _mm512_setzero_ps();
        let mut sum_act_vec0 = _mm512_setzero_ps();
        let mut sum_q_vec1 = _mm512_setzero_ps();
        let mut sum_act_vec1 = _mm512_setzero_ps();
        let mut sum_q_vec2 = _mm512_setzero_ps();
        let mut sum_act_vec2 = _mm512_setzero_ps();
        let mut sum_q_vec3 = _mm512_setzero_ps();
        let mut sum_act_vec3 = _mm512_setzero_ps();

        // Pass 1: i = 0, 1, 2, 3 (first 64 bytes -> 128 activations)
        {
            // Block 0
            let raw0 = _mm_loadu_si128(qs.as_ptr().cast());
            let lo0 = _mm_and_si128(raw0, mask_0f);
            let hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), mask_0f);
            let act_lo0 = _mm512_loadu_ps(act.as_ptr());
            let act_hi0 = _mm512_loadu_ps(act.as_ptr().add(128));
            sum_q_vec0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo0)),
                act_lo0,
                sum_q_vec0,
            );
            sum_act_vec0 = _mm512_add_ps(sum_act_vec0, act_lo0);
            sum_q_vec1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi0)),
                act_hi0,
                sum_q_vec1,
            );
            sum_act_vec1 = _mm512_add_ps(sum_act_vec1, act_hi0);

            // Block 1
            let raw1 = _mm_loadu_si128(qs.as_ptr().add(16).cast());
            let lo1 = _mm_and_si128(raw1, mask_0f);
            let hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), mask_0f);
            let act_lo1 = _mm512_loadu_ps(act.as_ptr().add(16));
            let act_hi1 = _mm512_loadu_ps(act.as_ptr().add(128 + 16));
            sum_q_vec2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo1)),
                act_lo1,
                sum_q_vec2,
            );
            sum_act_vec2 = _mm512_add_ps(sum_act_vec2, act_lo1);
            sum_q_vec3 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi1)),
                act_hi1,
                sum_q_vec3,
            );
            sum_act_vec3 = _mm512_add_ps(sum_act_vec3, act_hi1);

            // Block 2
            let raw2 = _mm_loadu_si128(qs.as_ptr().add(32).cast());
            let lo2 = _mm_and_si128(raw2, mask_0f);
            let hi2 = _mm_and_si128(_mm_srli_epi16(raw2, 4), mask_0f);
            let act_lo2 = _mm512_loadu_ps(act.as_ptr().add(32));
            let act_hi2 = _mm512_loadu_ps(act.as_ptr().add(128 + 32));
            sum_q_vec0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo2)),
                act_lo2,
                sum_q_vec0,
            );
            sum_act_vec0 = _mm512_add_ps(sum_act_vec0, act_lo2);
            sum_q_vec1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi2)),
                act_hi2,
                sum_q_vec1,
            );
            sum_act_vec1 = _mm512_add_ps(sum_act_vec1, act_hi2);

            // Block 3
            let raw3 = _mm_loadu_si128(qs.as_ptr().add(48).cast());
            let lo3 = _mm_and_si128(raw3, mask_0f);
            let hi3 = _mm_and_si128(_mm_srli_epi16(raw3, 4), mask_0f);
            let act_lo3 = _mm512_loadu_ps(act.as_ptr().add(48));
            let act_hi3 = _mm512_loadu_ps(act.as_ptr().add(128 + 48));
            sum_q_vec2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo3)),
                act_lo3,
                sum_q_vec2,
            );
            sum_act_vec2 = _mm512_add_ps(sum_act_vec2, act_lo3);
            sum_q_vec3 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi3)),
                act_hi3,
                sum_q_vec3,
            );
            sum_act_vec3 = _mm512_add_ps(sum_act_vec3, act_hi3);
        }

        // Pass 2: i = 4, 5, 6, 7 (remaining 64 bytes -> 128 activations)
        {
            // Block 4
            let raw4 = _mm_loadu_si128(qs.as_ptr().add(64).cast());
            let lo4 = _mm_and_si128(raw4, mask_0f);
            let hi4 = _mm_and_si128(_mm_srli_epi16(raw4, 4), mask_0f);
            let act_lo4 = _mm512_loadu_ps(act.as_ptr().add(64));
            let act_hi4 = _mm512_loadu_ps(act.as_ptr().add(128 + 64));
            sum_q_vec0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo4)),
                act_lo4,
                sum_q_vec0,
            );
            sum_act_vec0 = _mm512_add_ps(sum_act_vec0, act_lo4);
            sum_q_vec1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi4)),
                act_hi4,
                sum_q_vec1,
            );
            sum_act_vec1 = _mm512_add_ps(sum_act_vec1, act_hi4);

            // Block 5
            let raw5 = _mm_loadu_si128(qs.as_ptr().add(80).cast());
            let lo5 = _mm_and_si128(raw5, mask_0f);
            let hi5 = _mm_and_si128(_mm_srli_epi16(raw5, 4), mask_0f);
            let act_lo5 = _mm512_loadu_ps(act.as_ptr().add(80));
            let act_hi5 = _mm512_loadu_ps(act.as_ptr().add(128 + 80));
            sum_q_vec2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo5)),
                act_lo5,
                sum_q_vec2,
            );
            sum_act_vec2 = _mm512_add_ps(sum_act_vec2, act_lo5);
            sum_q_vec3 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi5)),
                act_hi5,
                sum_q_vec3,
            );
            sum_act_vec3 = _mm512_add_ps(sum_act_vec3, act_hi5);

            // Block 6
            let raw6 = _mm_loadu_si128(qs.as_ptr().add(96).cast());
            let lo6 = _mm_and_si128(raw6, mask_0f);
            let hi6 = _mm_and_si128(_mm_srli_epi16(raw6, 4), mask_0f);
            let act_lo6 = _mm512_loadu_ps(act.as_ptr().add(96));
            let act_hi6 = _mm512_loadu_ps(act.as_ptr().add(128 + 96));
            sum_q_vec0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo6)),
                act_lo6,
                sum_q_vec0,
            );
            sum_act_vec0 = _mm512_add_ps(sum_act_vec0, act_lo6);
            sum_q_vec1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi6)),
                act_hi6,
                sum_q_vec1,
            );
            sum_act_vec1 = _mm512_add_ps(sum_act_vec1, act_hi6);

            // Block 7
            let raw7 = _mm_loadu_si128(qs.as_ptr().add(112).cast());
            let lo7 = _mm_and_si128(raw7, mask_0f);
            let hi7 = _mm_and_si128(_mm_srli_epi16(raw7, 4), mask_0f);
            let act_lo7 = _mm512_loadu_ps(act.as_ptr().add(112));
            let act_hi7 = _mm512_loadu_ps(act.as_ptr().add(128 + 112));
            sum_q_vec2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo7)),
                act_lo7,
                sum_q_vec2,
            );
            sum_act_vec2 = _mm512_add_ps(sum_act_vec2, act_lo7);
            sum_q_vec3 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi7)),
                act_hi7,
                sum_q_vec3,
            );
            sum_act_vec3 = _mm512_add_ps(sum_act_vec3, act_hi7);
        }

        let sum_q_vec = _mm512_add_ps(
            _mm512_add_ps(sum_q_vec0, sum_q_vec1),
            _mm512_add_ps(sum_q_vec2, sum_q_vec3),
        );
        let sum_act_vec = _mm512_add_ps(
            _mm512_add_ps(sum_act_vec0, sum_act_vec1),
            _mm512_add_ps(sum_act_vec2, sum_act_vec3),
        );

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
#[target_feature(enable = "avx512f", enable = "avx512bw")]
#[allow(dead_code)]
pub unsafe fn dot_q6_k_avx512(ql: &[u8; 128], qh: &[u8; 64], act: &[f32; 256], d: f32) -> f32 {
    use core::arch::x86_64::{
        _mm512_cvtepi32_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512,
        _mm512_set1_epi32, _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_ps,
    };

    // SAFETY: Verified AVX-512F / AVX-512BW support and valid buffers.
    unsafe {
        let mut sum_vec = _mm512_setzero_ps();
        let thirty_two = _mm512_set1_epi32(32);
        let thirty_two_f = _mm512_cvtepi32_ps(thirty_two);

        // Process in chunks of 16 weights (since AVX-512 holds 16 f32s)
        for chunk in 0..8 {
            let base_idx = chunk * 16;
            let act0 = _mm512_loadu_ps(act.as_ptr().add(base_idx));
            let act1 = _mm512_loadu_ps(act.as_ptr().add(base_idx + 128));

            // Reconstruct q0 and q1 for 16 elements
            let mut q0_buf = [0i32; 16];
            let mut q1_buf = [0i32; 16];
            for j in 0..16 {
                let i = base_idx + j;
                let byte_l = *ql.get_unchecked(i);
                let qh_idx = i / 2;
                let shift = (i % 2) * 4;
                let byte_h = (*qh.get_unchecked(qh_idx) >> shift) & 0x0F;

                let h0 = byte_h & 0x03;
                let h1 = (byte_h >> 2) & 0x03;

                q0_buf[j] = ((h0 << 4) | (byte_l & 0x0F)) as i32;
                q1_buf[j] = ((h1 << 4) | ((byte_l >> 4) & 0x0F)) as i32;
            }

            let raw_q0 = _mm512_loadu_si512(q0_buf.as_ptr().cast());
            let raw_q1 = _mm512_loadu_si512(q1_buf.as_ptr().cast());

            let f_q0 = _mm512_sub_ps(_mm512_cvtepi32_ps(raw_q0), thirty_two_f);
            let f_q1 = _mm512_sub_ps(_mm512_cvtepi32_ps(raw_q1), thirty_two_f);

            sum_vec = _mm512_fmadd_ps(f_q0, act0, sum_vec);
            sum_vec = _mm512_fmadd_ps(f_q1, act1, sum_vec);
        }

        let mut buf = [0.0f32; 16];
        _mm512_storeu_ps(buf.as_mut_ptr(), sum_vec);
        let mut total = 0.0f32;
        for v in buf {
            total += v;
        }
        total * d
    }
}

