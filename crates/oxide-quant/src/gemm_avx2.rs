//! Cache-Blocked AVX2 Tile GEMM for CPU Prefill Acceleration.
//!
//! Evaluates $W \in \mathbb{R}^{M \times K} \times X \in \mathbb{R}^{K \times B}$ by accumulating
//! across 4 parallel token vectors simultaneously in 16 YMM registers. Weight blocks are loaded
//! into L1/L2 cache and registers exactly ONCE per row and reused across all batch tokens,
//! maximizing arithmetic intensity and eliminating DRAM memory bandwidth bottlenecks.

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

use crate::gguf_quants::BlockQ4_K;
use crate::int_quant::{BlockQ6_K, BlockQ8_K};

static K_SHUFFLE_K4: [u8; 256] = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
];

static K_SHUFFLE_K6: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
    11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
    14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
];

const KMASK1: u32 = 0x3f3f3f3f;
const KMASK2: u32 = 0x0f0f0f0f;
const KMASK3: u32 = 0x03030303;

#[inline(always)]
fn get_team_threads() -> usize {
    if let Ok(val) = std::env::var("OXIDE_NUM_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    8
}

/// Evaluates a single row of Q4_K weights across 4 tokens simultaneously in AVX2 registers.
///
/// # Safety
/// Caller must ensure CPU supports AVX2 and FMA extensions and slices have valid aligned data.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_k_q8_k_4tokens_avx2(
    row_weights: &[BlockQ4_K],
    tok0: &[BlockQ8_K],
    tok1: &[BlockQ8_K],
    tok2: &[BlockQ8_K],
    tok3: &[BlockQ8_K],
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::{
        _mm_cvtepi32_ps, _mm_fmadd_ps, _mm_hadd_epi16, _mm_madd_epi16, _mm_set_epi32, _mm_set1_ps,
        _mm_setzero_ps, _mm_storeu_ps, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256,
        _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_fmadd_ps,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_ps,
    };

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut acc_m0 = _mm_setzero_ps();
        let mut acc_m1 = _mm_setzero_ps();
        let mut acc_m2 = _mm_setzero_ps();
        let mut acc_m3 = _mm_setzero_ps();

        let num_blocks = row_weights
            .len()
            .min(tok0.len())
            .min(tok1.len())
            .min(tok2.len())
            .min(tok3.len());

        for b in 0..num_blocks {
            let x = &row_weights[b];
            let y0 = &tok0[b];
            let y1 = &tok1[b];
            let y2 = &tok2[b];
            let y3 = &tok3[b];

            let x_d = x.d.to_f32();
            let x_dmin = x.dmin.to_f32();

            let d0 = y0.d * x_d;
            let dmin0 = -y0.d * x_dmin;
            let d1 = y1.d * x_d;
            let dmin1 = -y1.d * x_dmin;
            let d2 = y2.d * x_d;
            let dmin2 = -y2.d * x_dmin;
            let d3 = y3.d * x_d;
            let dmin3 = -y3.d * x_dmin;

            // Unpack scales for weight block x ONCE
            let mut utmp = [0u32; 4];
            std::ptr::copy_nonoverlapping(x.scales.as_ptr(), utmp.as_mut_ptr().cast::<u8>(), 12);
            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));
            let mins_half = _mm256_extracti128_si256(mins_and_scales, 1);

            // Token 0 bsums
            let q8sums0 = _mm256_loadu_si256(y0.bsums.as_ptr().cast());
            let q8s0 = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums0, 0),
                _mm256_extracti128_si256(q8sums0, 1),
            );
            acc_m0 = _mm_fmadd_ps(
                _mm_set1_ps(dmin0),
                _mm_cvtepi32_ps(_mm_madd_epi16(mins_half, q8s0)),
                acc_m0,
            );

            // Token 1 bsums
            let q8sums1 = _mm256_loadu_si256(y1.bsums.as_ptr().cast());
            let q8s1 = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums1, 0),
                _mm256_extracti128_si256(q8sums1, 1),
            );
            acc_m1 = _mm_fmadd_ps(
                _mm_set1_ps(dmin1),
                _mm_cvtepi32_ps(_mm_madd_epi16(mins_half, q8s1)),
                acc_m1,
            );

            // Token 2 bsums
            let q8sums2 = _mm256_loadu_si256(y2.bsums.as_ptr().cast());
            let q8s2 = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums2, 0),
                _mm256_extracti128_si256(q8sums2, 1),
            );
            acc_m2 = _mm_fmadd_ps(
                _mm_set1_ps(dmin2),
                _mm_cvtepi32_ps(_mm_madd_epi16(mins_half, q8s2)),
                acc_m2,
            );

            // Token 3 bsums
            let q8sums3 = _mm256_loadu_si256(y3.bsums.as_ptr().cast());
            let q8s3 = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums3, 0),
                _mm256_extracti128_si256(q8sums3, 1),
            );
            acc_m3 = _mm_fmadd_ps(
                _mm_set1_ps(dmin3),
                _mm_cvtepi32_ps(_mm_madd_epi16(mins_half, q8s3)),
                acc_m3,
            );

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

            let mut sumi0 = _mm256_setzero_si256();
            let mut sumi1 = _mm256_setzero_si256();
            let mut sumi2 = _mm256_setzero_si256();
            let mut sumi3 = _mm256_setzero_si256();

            let mut q4_ptr = x.qs.as_ptr();
            let mut q8_ptr0 = y0.qs.as_ptr();
            let mut q8_ptr1 = y1.qs.as_ptr();
            let mut q8_ptr2 = y2.qs.as_ptr();
            let mut q8_ptr3 = y3.qs.as_ptr();

            for j in 0..4 {
                let shuf_l = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j) * 32).cast());
                let shuf_h = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j + 1) * 32).cast());

                let scale_l = _mm256_shuffle_epi8(scales, shuf_l);
                let scale_h = _mm256_shuffle_epi8(scales, shuf_h);

                // Load 32 weight bytes ONCE for all 4 tokens
                let q4bits = _mm256_loadu_si256(q4_ptr.cast());
                q4_ptr = q4_ptr.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                // Token 0
                let q8l0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);
                let q8h0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);
                let mut p16l0 = _mm256_maddubs_epi16(q4l, q8l0);
                p16l0 = _mm256_madd_epi16(scale_l, p16l0);
                let mut p16h0 = _mm256_maddubs_epi16(q4h, q8h0);
                p16h0 = _mm256_madd_epi16(scale_h, p16h0);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16l0, p16h0));

                // Token 1
                let q8l1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);
                let q8h1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);
                let mut p16l1 = _mm256_maddubs_epi16(q4l, q8l1);
                p16l1 = _mm256_madd_epi16(scale_l, p16l1);
                let mut p16h1 = _mm256_maddubs_epi16(q4h, q8h1);
                p16h1 = _mm256_madd_epi16(scale_h, p16h1);
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p16l1, p16h1));

                // Token 2
                let q8l2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);
                let q8h2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);
                let mut p16l2 = _mm256_maddubs_epi16(q4l, q8l2);
                p16l2 = _mm256_madd_epi16(scale_l, p16l2);
                let mut p16h2 = _mm256_maddubs_epi16(q4h, q8h2);
                p16h2 = _mm256_madd_epi16(scale_h, p16h2);
                sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p16l2, p16h2));

                // Token 3
                let q8l3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);
                let q8h3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);
                let mut p16l3 = _mm256_maddubs_epi16(q4l, q8l3);
                p16l3 = _mm256_madd_epi16(scale_l, p16l3);
                let mut p16h3 = _mm256_maddubs_epi16(q4h, q8h3);
                p16h3 = _mm256_madd_epi16(scale_h, p16h3);
                sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(p16l3, p16h3));
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(sumi2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(sumi3), acc3);
        }

        let mut buf_m = [0.0f32; 16];
        _mm_storeu_ps(buf_m.as_mut_ptr(), acc_m0);
        _mm_storeu_ps(buf_m.as_mut_ptr().add(4), acc_m1);
        _mm_storeu_ps(buf_m.as_mut_ptr().add(8), acc_m2);
        _mm_storeu_ps(buf_m.as_mut_ptr().add(12), acc_m3);
        let sum_m0 = buf_m[0] + buf_m[1] + buf_m[2] + buf_m[3];
        let sum_m1 = buf_m[4] + buf_m[5] + buf_m[6] + buf_m[7];
        let sum_m2 = buf_m[8] + buf_m[9] + buf_m[10] + buf_m[11];
        let sum_m3 = buf_m[12] + buf_m[13] + buf_m[14] + buf_m[15];

        let mut buf = [0.0f32; 32];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf.as_mut_ptr().add(8), acc1);
        _mm256_storeu_ps(buf.as_mut_ptr().add(16), acc2);
        _mm256_storeu_ps(buf.as_mut_ptr().add(24), acc3);

        let sum_acc0 = (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7]);
        let sum_acc1 =
            (buf[8] + buf[9] + buf[10] + buf[11]) + (buf[12] + buf[13] + buf[14] + buf[15]);
        let sum_acc2 =
            (buf[16] + buf[17] + buf[18] + buf[19]) + (buf[20] + buf[21] + buf[22] + buf[23]);
        let sum_acc3 =
            (buf[24] + buf[25] + buf[26] + buf[27]) + (buf[28] + buf[29] + buf[30] + buf[31]);

        (
            sum_acc0 + sum_m0,
            sum_acc1 + sum_m1,
            sum_acc2 + sum_m2,
            sum_acc3 + sum_m3,
        )
    }
}

/// Evaluates a single row of Q6_K weights across 4 tokens simultaneously in AVX2 registers.
///
/// # Safety
/// Caller must ensure CPU supports AVX2 and FMA extensions and slices have valid aligned data.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q6_k_q8_k_4tokens_avx2(
    row_weights: &[BlockQ6_K],
    tok0: &[BlockQ8_K],
    tok1: &[BlockQ8_K],
    tok2: &[BlockQ8_K],
    tok3: &[BlockQ8_K],
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_shuffle_epi8, _mm256_add_epi32, _mm256_and_si256,
        _mm256_cvtepi8_epi16, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_or_si256, _mm256_set1_epi8, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_slli_epi32,
        _mm256_srli_epi16, _mm256_storeu_ps, _mm256_sub_epi32,
    };

    let nb = row_weights
        .len()
        .min(tok0.len())
        .min(tok1.len())
        .min(tok2.len())
        .min(tok3.len());

    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m12 = _mm256_set1_epi8(12);
        let m15 = _mm256_set1_epi8(15);
        let m48 = _mm256_set1_epi8(48);
        let m_c0 = _mm256_set1_epi8(-64); // 0xC0

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in 0..nb {
            let x = &row_weights[i];
            let y0 = &tok0[i];
            let y1 = &tok1[i];
            let y2 = &tok2[i];
            let y3 = &tok3[i];

            let x_d = x.d.to_f32();
            let d0 = y0.d * x_d;
            let d1 = y1.d * x_d;
            let d2 = y2.d * x_d;
            let d3 = y3.d * x_d;

            let scales = _mm_loadu_si128(x.scales.as_ptr().cast());
            let scales_16 = _mm256_cvtepi8_epi16(scales);

            let q8sums0 = _mm256_loadu_si256(y0.bsums.as_ptr().cast());
            let q8sclsub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums0, scales_16), 5);

            let q8sums1 = _mm256_loadu_si256(y1.bsums.as_ptr().cast());
            let q8sclsub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums1, scales_16), 5);

            let q8sums2 = _mm256_loadu_si256(y2.bsums.as_ptr().cast());
            let q8sclsub2 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums2, scales_16), 5);

            let q8sums3 = _mm256_loadu_si256(y3.bsums.as_ptr().cast());
            let q8sclsub3 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums3, scales_16), 5);

            let mut sumi0 = _mm256_setzero_si256();
            let mut sumi1 = _mm256_setzero_si256();
            let mut sumi2 = _mm256_setzero_si256();
            let mut sumi3 = _mm256_setzero_si256();

            let mut q4_ptr = x.ql.as_ptr();
            let mut qh_ptr = x.qh.as_ptr();
            let mut q8_ptr0 = y0.qs.as_ptr();
            let mut q8_ptr1 = y1.qs.as_ptr();
            let mut q8_ptr2 = y2.qs.as_ptr();
            let mut q8_ptr3 = y3.qs.as_ptr();

            for j in 0..2 {
                let q4bits1 = _mm256_loadu_si256(q4_ptr.cast());
                q4_ptr = q4_ptr.add(32);
                let q4bits2 = _mm256_loadu_si256(q4_ptr.cast());
                q4_ptr = q4_ptr.add(32);
                let q4bits_h = _mm256_loadu_si256(qh_ptr.cast());
                qh_ptr = qh_ptr.add(32);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m3), 4);
                let q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m12), 2);
                let q4h_2 = _mm256_and_si256(q4bits_h, m48);
                let q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h, m_c0), 2);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                let is_base = j * 4;
                let sc0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE_K6.as_ptr().add(is_base * 16).cast()),
                ));
                let sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE_K6.as_ptr().add((is_base + 1) * 16).cast()),
                ));
                let sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE_K6.as_ptr().add((is_base + 2) * 16).cast()),
                ));
                let sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE_K6.as_ptr().add((is_base + 3) * 16).cast()),
                ));

                // Token 0
                let q8_0_0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);
                let q8_1_0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);
                let q8_2_0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);
                let q8_3_0 = _mm256_loadu_si256(q8_ptr0.cast());
                q8_ptr0 = q8_ptr0.add(32);

                let p0_0 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, q8_0_0));
                let p1_0 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, q8_1_0));
                let p2_0 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, q8_2_0));
                let p3_0 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, q8_3_0));
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p0_0, p1_0));
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p2_0, p3_0));

                // Token 1
                let q8_0_1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);
                let q8_1_1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);
                let q8_2_1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);
                let q8_3_1 = _mm256_loadu_si256(q8_ptr1.cast());
                q8_ptr1 = q8_ptr1.add(32);

                let p0_1 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, q8_0_1));
                let p1_1 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, q8_1_1));
                let p2_1 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, q8_2_1));
                let p3_1 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, q8_3_1));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p0_1, p1_1));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p2_1, p3_1));

                // Token 2
                let q8_0_2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);
                let q8_1_2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);
                let q8_2_2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);
                let q8_3_2 = _mm256_loadu_si256(q8_ptr2.cast());
                q8_ptr2 = q8_ptr2.add(32);

                let p0_2 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, q8_0_2));
                let p1_2 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, q8_1_2));
                let p2_2 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, q8_2_2));
                let p3_2 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, q8_3_2));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p0_2, p1_2));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p2_2, p3_2));

                // Token 3
                let q8_0_3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);
                let q8_1_3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);
                let q8_2_3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);
                let q8_3_3 = _mm256_loadu_si256(q8_ptr3.cast());
                q8_ptr3 = q8_ptr3.add(32);

                let p0_3 = _mm256_madd_epi16(sc0, _mm256_maddubs_epi16(q4_0, q8_0_3));
                let p1_3 = _mm256_madd_epi16(sc1, _mm256_maddubs_epi16(q4_1, q8_1_3));
                let p2_3 = _mm256_madd_epi16(sc2, _mm256_maddubs_epi16(q4_2, q8_2_3));
                let p3_3 = _mm256_madd_epi16(sc3, _mm256_maddubs_epi16(q4_3, q8_3_3));
                sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(p0_3, p1_3));
                sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(p2_3, p3_3));
            }

            sumi0 = _mm256_sub_epi32(sumi0, q8sclsub0);
            sumi1 = _mm256_sub_epi32(sumi1, q8sclsub1);
            sumi2 = _mm256_sub_epi32(sumi2, q8sclsub2);
            sumi3 = _mm256_sub_epi32(sumi3, q8sclsub3);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(sumi2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(sumi3), acc3);
        }

        let mut buf = [0.0f32; 32];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf.as_mut_ptr().add(8), acc1);
        _mm256_storeu_ps(buf.as_mut_ptr().add(16), acc2);
        _mm256_storeu_ps(buf.as_mut_ptr().add(24), acc3);

        let sum0 = (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7]);
        let sum1 = (buf[8] + buf[9] + buf[10] + buf[11]) + (buf[12] + buf[13] + buf[14] + buf[15]);
        let sum2 =
            (buf[16] + buf[17] + buf[18] + buf[19]) + (buf[20] + buf[21] + buf[22] + buf[23]);
        let sum3 =
            (buf[24] + buf[25] + buf[26] + buf[27]) + (buf[28] + buf[29] + buf[30] + buf[31]);

        (sum0, sum1, sum2, sum3)
    }
}

/// High-throughput cache-blocked Q4_K Matrix Multiplication for prefill.
/// Multiplies matrix W [m, n] by a batch [b, n] of pre-quantized BlockQ8_K activations.
/// Dispatches 4-way vector accumulators over row-parallel worker threads.
pub fn gemm_q4_k_tiled_prefill(
    matrix: &[BlockQ4_K],
    batch_q8: &[Vec<BlockQ8_K>],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    let b = batch_q8.len();
    if b == 0 {
        return;
    }
    if b == 1 {
        crate::simd::gemv_q4_k_prequantized(matrix, &batch_q8[0], m, n, output);
        return;
    }

    let blocks_per_row = n / 256;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let out_ptr = output.as_mut_ptr() as usize;

    team.dispatch_rows(m, |start, end| {
        for row in start..end {
            let row_offset = row * blocks_per_row;
            let row_blocks = &matrix[row_offset..row_offset + blocks_per_row];

            let mut t = 0usize;
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                // 4-token parallel register tile
                while t + 4 <= b {
                    let (v0, v1, v2, v3) = unsafe {
                        dot_q4_k_q8_k_4tokens_avx2(
                            row_blocks,
                            &batch_q8[t],
                            &batch_q8[t + 1],
                            &batch_q8[t + 2],
                            &batch_q8[t + 3],
                        )
                    };
                    unsafe {
                        let base_out = out_ptr as *mut f32;
                        *base_out.add(t * m + row) = v0;
                        *base_out.add((t + 1) * m + row) = v1;
                        *base_out.add((t + 2) * m + row) = v2;
                        *base_out.add((t + 3) * m + row) = v3;
                    }
                    t += 4;
                }
            }

            // Remainder tokens
            while t < b {
                #[cfg(target_arch = "x86_64")]
                let val = if has_avx2 {
                    unsafe { crate::simd::dot_q4_k_q8_k_row_avx2(row_blocks, &batch_q8[t]) }
                } else {
                    crate::simd::dot_q4_k_q8_k_row_portable(row_blocks, &batch_q8[t])
                };
                #[cfg(not(target_arch = "x86_64"))]
                let val = crate::simd::dot_q4_k_q8_k_row_portable(row_blocks, &batch_q8[t]);

                unsafe {
                    let out_elem = (out_ptr as *mut f32).add(t * m + row);
                    *out_elem = val;
                }
                t += 1;
            }
        }
    });
}

/// High-throughput cache-blocked Q6_K Matrix Multiplication for prefill.
/// Multiplies matrix W [m, n] by a batch [b, n] of pre-quantized BlockQ8_K activations.
/// Dispatches 4-way vector accumulators over row-parallel worker threads.
pub fn gemm_q6_k_tiled_prefill(
    matrix: &[BlockQ6_K],
    batch_q8: &[Vec<BlockQ8_K>],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    let b = batch_q8.len();
    if b == 0 {
        return;
    }
    if b == 1 {
        crate::simd::gemv_q6_k_prequantized(matrix, &batch_q8[0], m, n, output);
        return;
    }

    let blocks_per_row = n / 256;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let out_ptr = output.as_mut_ptr() as usize;

    team.dispatch_rows(m, |start, end| {
        for row in start..end {
            let row_offset = row * blocks_per_row;
            let row_blocks = &matrix[row_offset..row_offset + blocks_per_row];

            let mut t = 0usize;
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                // 4-token parallel register tile
                while t + 4 <= b {
                    let (v0, v1, v2, v3) = unsafe {
                        dot_q6_k_q8_k_4tokens_avx2(
                            row_blocks,
                            &batch_q8[t],
                            &batch_q8[t + 1],
                            &batch_q8[t + 2],
                            &batch_q8[t + 3],
                        )
                    };
                    unsafe {
                        let base_out = out_ptr as *mut f32;
                        *base_out.add(t * m + row) = v0;
                        *base_out.add((t + 1) * m + row) = v1;
                        *base_out.add((t + 2) * m + row) = v2;
                        *base_out.add((t + 3) * m + row) = v3;
                    }
                    t += 4;
                }
            }

            // Remainder tokens
            while t < b {
                #[cfg(target_arch = "x86_64")]
                let val = if has_avx2 {
                    unsafe { crate::simd::dot_q6_k_q8_k_row_avx2(row_blocks, &batch_q8[t]) }
                } else {
                    crate::simd::dot_q6_k_q8_k_row_portable(row_blocks, &batch_q8[t])
                };
                #[cfg(not(target_arch = "x86_64"))]
                let val = crate::simd::dot_q6_k_q8_k_row_portable(row_blocks, &batch_q8[t]);

                unsafe {
                    let out_elem = (out_ptr as *mut f32).add(t * m + row);
                    *out_elem = val;
                }
                t += 1;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_q4_k_4tokens_parity() {
        let mut blk = BlockQ4_K {
            d: crate::int_quant::f16::from_f32(0.5),
            dmin: crate::int_quant::f16::from_f32(0.1),
            ..Default::default()
        };
        for i in 0..128 {
            blk.qs[i] = (i as u8) % 255;
        }

        let mut b0 = vec![BlockQ8_K::default()];
        let mut b1 = vec![BlockQ8_K::default()];
        let mut b2 = vec![BlockQ8_K::default()];
        let mut b3 = vec![BlockQ8_K::default()];

        b0[0].d = 0.05;
        b1[0].d = 0.08;
        b2[0].d = 0.12;
        b3[0].d = 0.02;

        for i in 0..256 {
            b0[0].qs[i] = (i as i8).wrapping_mul(3);
            b1[0].qs[i] = (i as i8).wrapping_mul(5);
            b2[0].qs[i] = (i as i8).wrapping_mul(7);
            b3[0].qs[i] = (i as i8).wrapping_mul(11);
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let blk_slice = std::slice::from_ref(&blk);
            let (v0, v1, v2, v3) =
                unsafe { dot_q4_k_q8_k_4tokens_avx2(blk_slice, &b0, &b1, &b2, &b3) };
            let single0 = unsafe { crate::simd::dot_q4_k_q8_k_row_avx2(blk_slice, &b0) };
            let single1 = unsafe { crate::simd::dot_q4_k_q8_k_row_avx2(blk_slice, &b1) };
            let single2 = unsafe { crate::simd::dot_q4_k_q8_k_row_avx2(blk_slice, &b2) };
            let single3 = unsafe { crate::simd::dot_q4_k_q8_k_row_avx2(blk_slice, &b3) };

            assert!((v0 - single0).abs() < 1e-3, "v0 {v0} != single0 {single0}");
            assert!((v1 - single1).abs() < 1e-3, "v1 {v1} != single1 {single1}");
            assert!((v2 - single2).abs() < 1e-3, "v2 {v2} != single2 {single2}");
            assert!((v3 - single3).abs() < 1e-3, "v3 {v3} != single3 {single3}");
        }
    }

    #[test]
    fn test_gemm_q6_k_4tokens_parity() {
        let mut blk = BlockQ6_K {
            d: crate::int_quant::f16::from_f32(0.5),
            ..Default::default()
        };
        for i in 0..128 {
            blk.ql[i] = (i as u8) % 255;
        }
        for i in 0..64 {
            blk.qh[i] = ((i as u8) * 3) % 255;
        }
        for i in 0..16 {
            blk.scales[i] = (i as i8) - 8;
        }

        let mut b0 = vec![BlockQ8_K::default()];
        let mut b1 = vec![BlockQ8_K::default()];
        let mut b2 = vec![BlockQ8_K::default()];
        let mut b3 = vec![BlockQ8_K::default()];

        b0[0].d = 0.05;
        b1[0].d = 0.08;
        b2[0].d = 0.12;
        b3[0].d = 0.02;

        for i in 0..256 {
            b0[0].qs[i] = (i as i8).wrapping_mul(3);
            b1[0].qs[i] = (i as i8).wrapping_mul(5);
            b2[0].qs[i] = (i as i8).wrapping_mul(7);
            b3[0].qs[i] = (i as i8).wrapping_mul(11);
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let blk_slice = std::slice::from_ref(&blk);
            let (v0, v1, v2, v3) =
                unsafe { dot_q6_k_q8_k_4tokens_avx2(blk_slice, &b0, &b1, &b2, &b3) };
            let single0 = unsafe { crate::simd::dot_q6_k_q8_k_row_avx2(blk_slice, &b0) };
            let single1 = unsafe { crate::simd::dot_q6_k_q8_k_row_avx2(blk_slice, &b1) };
            let single2 = unsafe { crate::simd::dot_q6_k_q8_k_row_avx2(blk_slice, &b2) };
            let single3 = unsafe { crate::simd::dot_q6_k_q8_k_row_avx2(blk_slice, &b3) };

            assert!((v0 - single0).abs() < 1e-3, "v0 {v0} != single0 {single0}");
            assert!((v1 - single1).abs() < 1e-3, "v1 {v1} != single1 {single1}");
            assert!((v2 - single2).abs() < 1e-3, "v2 {v2} != single2 {single2}");
            assert!((v3 - single3).abs() < 1e-3, "v3 {v3} != single3 {single3}");
        }
    }
}
