//! x86_64 AVX2 & FMA Block-Quantized Row Dot-Products (Q4_K and Q6_K).

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q6_k_q8_k_row_avx2(
    row_weights: &[crate::int_quant::BlockQ6_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> f32 {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_shuffle_epi8, _mm256_add_epi32, _mm256_and_si256,
        _mm256_cvtepi8_epi16, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_or_si256, _mm256_set1_epi8, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_slli_epi32,
        _mm256_srli_epi16, _mm256_storeu_ps, _mm256_sub_epi32,
    };

    let nb = row_weights.len();
    // SAFETY: Microarchitecture targets x86_64 AVX2 and FMA extensions checked by caller.
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m12 = _mm256_set1_epi8(12);
        let m15 = _mm256_set1_epi8(15);
        let m48 = _mm256_set1_epi8(48);
        let m_c0 = _mm256_set1_epi8(-64); // 0xC0

        static K_SHUFFLE: [u8; 128] = [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
            7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10,
            10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13,
            13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        ];

        let mut acc = _mm256_setzero_ps();

        for i in 0..nb {
            let x = &row_weights[i];
            let y = &act_q8_k[i];

            let d = y.d * x.d.to_f32();

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());
            let scales = _mm_loadu_si128(x.scales.as_ptr().cast());
            let scales_16 = _mm256_cvtepi8_epi16(scales);
            let q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

            let mut sumi = _mm256_setzero_si256();

            let mut q4_ptr = x.ql.as_ptr();
            let mut qh_ptr = x.qh.as_ptr();
            let mut q8_ptr = y.qs.as_ptr();

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

                let q8_0 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_1 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_2 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_3 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);

                let mut p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let mut p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let mut p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let mut p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let is_base = j * 4;
                let scale_0 = _mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add(is_base * 16).cast()),
                );
                let scale_1 = _mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 1) * 16).cast()),
                );
                let scale_2 = _mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 2) * 16).cast()),
                );
                let scale_3 = _mm_shuffle_epi8(
                    scales,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 3) * 16).cast()),
                );

                p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }

            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            let d_vec = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(d_vec, _mm256_cvtepi32_ps(sumi), acc);
        }

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q6_k_q8_k_2rows_avx2(
    row0_weights: &[crate::int_quant::BlockQ6_K],
    row1_weights: &[crate::int_quant::BlockQ6_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> (f32, f32) {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm_shuffle_epi8, _mm256_add_epi32, _mm256_and_si256,
        _mm256_cvtepi8_epi16, _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_or_si256, _mm256_set1_epi8, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_slli_epi32,
        _mm256_srli_epi16, _mm256_storeu_ps, _mm256_sub_epi32,
    };

    static K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];

    // SAFETY: caller guarantees avx2 and fma features are enabled and slices have valid aligned data.
    unsafe {
        let nb = row0_weights
            .len()
            .min(row1_weights.len())
            .min(act_q8_k.len());
        let m15 = _mm256_set1_epi8(15);
        let m3 = _mm256_set1_epi8(3);
        let m12 = _mm256_set1_epi8(12);
        let m48 = _mm256_set1_epi8(48);
        let m_c0 = _mm256_set1_epi8(-64); // 0xC0

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        for i in 0..nb {
            let x0 = &row0_weights[i];
            let x1 = &row1_weights[i];
            let y = &act_q8_k[i];

            let d0 = y.d * x0.d.to_f32();
            let d1 = y.d * x1.d.to_f32();

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());

            let scales0 = _mm_loadu_si128(x0.scales.as_ptr().cast());
            let scales_16_0 = _mm256_cvtepi8_epi16(scales0);
            let q8sclsub0 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16_0), 5);

            let scales1 = _mm_loadu_si128(x1.scales.as_ptr().cast());
            let scales_16_1 = _mm256_cvtepi8_epi16(scales1);
            let q8sclsub1 = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16_1), 5);

            let mut sumi0 = _mm256_setzero_si256();
            let mut sumi1 = _mm256_setzero_si256();

            let mut q4_ptr0 = x0.ql.as_ptr();
            let mut qh_ptr0 = x0.qh.as_ptr();
            let mut q4_ptr1 = x1.ql.as_ptr();
            let mut qh_ptr1 = x1.qh.as_ptr();
            let mut q8_ptr = y.qs.as_ptr();

            for j in 0..2 {
                // Row 0
                let q4bits1_0 = _mm256_loadu_si256(q4_ptr0.cast());
                q4_ptr0 = q4_ptr0.add(32);
                let q4bits2_0 = _mm256_loadu_si256(q4_ptr0.cast());
                q4_ptr0 = q4_ptr0.add(32);
                let q4bits_h_0 = _mm256_loadu_si256(qh_ptr0.cast());
                qh_ptr0 = qh_ptr0.add(32);

                let q4h_0_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h_0, m3), 4);
                let q4h_1_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h_0, m12), 2);
                let q4h_2_0 = _mm256_and_si256(q4bits_h_0, m48);
                let q4h_3_0 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h_0, m_c0), 2);

                let q4_0_0 = _mm256_or_si256(_mm256_and_si256(q4bits1_0, m15), q4h_0_0);
                let q4_1_0 = _mm256_or_si256(_mm256_and_si256(q4bits2_0, m15), q4h_1_0);
                let q4_2_0 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits1_0, 4), m15),
                    q4h_2_0,
                );
                let q4_3_0 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits2_0, 4), m15),
                    q4h_3_0,
                );

                // Row 1
                let q4bits1_1 = _mm256_loadu_si256(q4_ptr1.cast());
                q4_ptr1 = q4_ptr1.add(32);
                let q4bits2_1 = _mm256_loadu_si256(q4_ptr1.cast());
                q4_ptr1 = q4_ptr1.add(32);
                let q4bits_h_1 = _mm256_loadu_si256(qh_ptr1.cast());
                qh_ptr1 = qh_ptr1.add(32);

                let q4h_0_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h_1, m3), 4);
                let q4h_1_1 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h_1, m12), 2);
                let q4h_2_1 = _mm256_and_si256(q4bits_h_1, m48);
                let q4h_3_1 = _mm256_srli_epi16(_mm256_and_si256(q4bits_h_1, m_c0), 2);

                let q4_0_1 = _mm256_or_si256(_mm256_and_si256(q4bits1_1, m15), q4h_0_1);
                let q4_1_1 = _mm256_or_si256(_mm256_and_si256(q4bits2_1, m15), q4h_1_1);
                let q4_2_1 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits1_1, 4), m15),
                    q4h_2_1,
                );
                let q4_3_1 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(q4bits2_1, 4), m15),
                    q4h_3_1,
                );

                // Load 128 activation bytes ONCE for both rows
                let q8_0 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_1 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_2 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8_3 = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);

                // Multiply Row 0
                let is_base = j * 4;
                let sc0_0 = _mm_shuffle_epi8(
                    scales0,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add(is_base * 16).cast()),
                );
                let sc1_0 = _mm_shuffle_epi8(
                    scales0,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 1) * 16).cast()),
                );
                let sc2_0 = _mm_shuffle_epi8(
                    scales0,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 2) * 16).cast()),
                );
                let sc3_0 = _mm_shuffle_epi8(
                    scales0,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 3) * 16).cast()),
                );

                let p16_0_0 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc0_0),
                    _mm256_maddubs_epi16(q4_0_0, q8_0),
                );
                let p16_1_0 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc1_0),
                    _mm256_maddubs_epi16(q4_1_0, q8_1),
                );
                let p16_2_0 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc2_0),
                    _mm256_maddubs_epi16(q4_2_0, q8_2),
                );
                let p16_3_0 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc3_0),
                    _mm256_maddubs_epi16(q4_3_0, q8_3),
                );

                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16_0_0, p16_1_0));
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16_2_0, p16_3_0));

                // Multiply Row 1
                let sc0_1 = _mm_shuffle_epi8(
                    scales1,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add(is_base * 16).cast()),
                );
                let sc1_1 = _mm_shuffle_epi8(
                    scales1,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 1) * 16).cast()),
                );
                let sc2_1 = _mm_shuffle_epi8(
                    scales1,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 2) * 16).cast()),
                );
                let sc3_1 = _mm_shuffle_epi8(
                    scales1,
                    _mm_loadu_si128(K_SHUFFLE.as_ptr().add((is_base + 3) * 16).cast()),
                );

                let p16_0_1 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc0_1),
                    _mm256_maddubs_epi16(q4_0_1, q8_0),
                );
                let p16_1_1 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc1_1),
                    _mm256_maddubs_epi16(q4_1_1, q8_1),
                );
                let p16_2_1 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc2_1),
                    _mm256_maddubs_epi16(q4_2_1, q8_2),
                );
                let p16_3_1 = _mm256_madd_epi16(
                    _mm256_cvtepi8_epi16(sc3_1),
                    _mm256_maddubs_epi16(q4_3_1, q8_3),
                );

                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p16_0_1, p16_1_1));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p16_2_1, p16_3_1));
            }

            sumi0 = _mm256_sub_epi32(sumi0, q8sclsub0);
            sumi1 = _mm256_sub_epi32(sumi1, q8sclsub1);

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
        }

        let mut buf0 = [0.0f32; 8];
        let mut buf1 = [0.0f32; 8];
        _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
        (
            (buf0[0] + buf0[1] + buf0[2] + buf0[3]) + (buf0[4] + buf0[5] + buf0[6] + buf0[7]),
            (buf1[0] + buf1[1] + buf1[2] + buf1[3]) + (buf1[4] + buf1[5] + buf1[6] + buf1[7]),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_k_q8_k_row_avx2(
    row_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> f32 {
    use core::arch::x86_64::{
        _mm_cvtepi32_ps, _mm_fmadd_ps, _mm_hadd_epi16, _mm_madd_epi16, _mm_set_epi32, _mm_set1_ps,
        _mm_setzero_ps, _mm_storeu_ps, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256,
        _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_fmadd_ps,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_ps,
    };

    static K_SHUFFLE_K4: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // SAFETY: caller guarantees avx2 and fma features are enabled and slices have valid aligned data.
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in row_weights.iter().zip(act_q8_k.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

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

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128), sc128, 1);

            let mut sumi = _mm256_setzero_si256();
            let mut q4_ptr = x.qs.as_ptr();
            let mut q8_ptr = y.qs.as_ptr();

            for j in 0..4 {
                let scale_l = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j) * 32).cast()),
                );
                let scale_h = _mm256_shuffle_epi8(
                    scales,
                    _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j + 1) * 32).cast()),
                );

                let q4bits = _mm256_loadu_si256(q4_ptr.cast());
                q4_ptr = q4_ptr.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let mut p16l = _mm256_maddubs_epi16(q4l, q8l);
                p16l = _mm256_madd_epi16(scale_l, p16l);

                let q8h = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let mut p16h = _mm256_maddubs_epi16(q4h, q8h);
                p16h = _mm256_madd_epi16(scale_h, p16h);

                let sumj = _mm256_add_epi32(p16l, p16h);
                sumi = _mm256_add_epi32(sumi, sumj);
            }

            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }

        let mut buf_m = [0.0f32; 4];
        _mm_storeu_ps(buf_m.as_mut_ptr(), acc_m);
        let sum_m = buf_m[0] + buf_m[1] + buf_m[2] + buf_m[3];

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        let sum_acc = (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7]);

        sum_acc + sum_m
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_k_q8_k_2rows_avx2(
    row0_weights: &[crate::gguf_quants::BlockQ4_K],
    row1_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> (f32, f32) {
    use core::arch::x86_64::{
        _mm_cvtepi32_ps, _mm_fmadd_ps, _mm_hadd_epi16, _mm_madd_epi16, _mm_set_epi32, _mm_set1_ps,
        _mm_setzero_ps, _mm_storeu_ps, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256,
        _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_fmadd_ps,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_ps,
    };

    static K_SHUFFLE_K4: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // SAFETY: caller guarantees avx2 and fma features are enabled and slices have valid aligned data.
    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc_m0 = _mm_setzero_ps();
        let mut acc_m1 = _mm_setzero_ps();

        let num_blocks = row0_weights
            .len()
            .min(row1_weights.len())
            .min(act_q8_k.len());

        for b in 0..num_blocks {
            let x0 = &row0_weights[b];
            let x1 = &row1_weights[b];
            let y = &act_q8_k[b];

            let d0 = y.d * x0.d.to_f32();
            let dmin0 = -y.d * x0.dmin.to_f32();
            let d1 = y.d * x1.d.to_f32();
            let dmin1 = -y.d * x1.dmin.to_f32();

            // Unpack scales for row 0
            let mut utmp0 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x0.scales.as_ptr(), utmp0.as_mut_ptr().cast::<u8>(), 12);
            utmp0[3] = ((utmp0[2] >> 4) & KMASK2) | (((utmp0[1] >> 6) & KMASK3) << 4);
            let uaux0 = utmp0[1] & KMASK1;
            utmp0[1] = (utmp0[2] & KMASK2) | (((utmp0[0] >> 6) & KMASK3) << 4);
            utmp0[2] = uaux0;
            utmp0[0] &= KMASK1;

            let mins_and_scales0 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp0[3] as i32,
                utmp0[2] as i32,
                utmp0[1] as i32,
                utmp0[0] as i32,
            ));

            // Unpack scales for row 1
            let mut utmp1 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x1.scales.as_ptr(), utmp1.as_mut_ptr().cast::<u8>(), 12);
            utmp1[3] = ((utmp1[2] >> 4) & KMASK2) | (((utmp1[1] >> 6) & KMASK3) << 4);
            let uaux1 = utmp1[1] & KMASK1;
            utmp1[1] = (utmp1[2] & KMASK2) | (((utmp1[0] >> 6) & KMASK3) << 4);
            utmp1[2] = uaux1;
            utmp1[0] &= KMASK1;

            let mins_and_scales1 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp1[3] as i32,
                utmp1[2] as i32,
                utmp1[1] as i32,
                utmp1[0] as i32,
            ));

            // Quantized activation bsums computed once for y
            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );

            let prod0 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales0, 1), q8s);
            acc_m0 = _mm_fmadd_ps(_mm_set1_ps(dmin0), _mm_cvtepi32_ps(prod0), acc_m0);

            let prod1 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales1, 1), q8s);
            acc_m1 = _mm_fmadd_ps(_mm_set1_ps(dmin1), _mm_cvtepi32_ps(prod1), acc_m1);

            let sc128_0 = _mm256_extracti128_si256(mins_and_scales0, 0);
            let scales0 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_0), sc128_0, 1);

            let sc128_1 = _mm256_extracti128_si256(mins_and_scales1, 0);
            let scales1 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_1), sc128_1, 1);

            let mut sumi0 = _mm256_setzero_si256();
            let mut sumi1 = _mm256_setzero_si256();
            let mut q4_ptr0 = x0.qs.as_ptr();
            let mut q4_ptr1 = x1.qs.as_ptr();
            let mut q8_ptr = y.qs.as_ptr();

            for j in 0..4 {
                let shuf_l = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j) * 32).cast());
                let shuf_h = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j + 1) * 32).cast());

                let scale_l0 = _mm256_shuffle_epi8(scales0, shuf_l);
                let scale_h0 = _mm256_shuffle_epi8(scales0, shuf_h);
                let scale_l1 = _mm256_shuffle_epi8(scales1, shuf_l);
                let scale_h1 = _mm256_shuffle_epi8(scales1, shuf_h);

                // Load 64 activation bytes ONCE for both rows
                let q8l = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8h = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);

                // Row 0
                let q4bits0 = _mm256_loadu_si256(q4_ptr0.cast());
                q4_ptr0 = q4_ptr0.add(32);
                let q4l0 = _mm256_and_si256(q4bits0, m4);
                let q4h0 = _mm256_and_si256(_mm256_srli_epi16(q4bits0, 4), m4);

                let mut p16l0 = _mm256_maddubs_epi16(q4l0, q8l);
                p16l0 = _mm256_madd_epi16(scale_l0, p16l0);
                let mut p16h0 = _mm256_maddubs_epi16(q4h0, q8h);
                p16h0 = _mm256_madd_epi16(scale_h0, p16h0);
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16l0, p16h0));

                // Row 1
                let q4bits1 = _mm256_loadu_si256(q4_ptr1.cast());
                q4_ptr1 = q4_ptr1.add(32);
                let q4l1 = _mm256_and_si256(q4bits1, m4);
                let q4h1 = _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4);

                let mut p16l1 = _mm256_maddubs_epi16(q4l1, q8l);
                p16l1 = _mm256_madd_epi16(scale_l1, p16l1);
                let mut p16h1 = _mm256_maddubs_epi16(q4h1, q8h);
                p16h1 = _mm256_madd_epi16(scale_h1, p16h1);
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p16l1, p16h1));
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
        }

        let mut buf_m0 = [0.0f32; 4];
        let mut buf_m1 = [0.0f32; 4];
        _mm_storeu_ps(buf_m0.as_mut_ptr(), acc_m0);
        _mm_storeu_ps(buf_m1.as_mut_ptr(), acc_m1);
        let sum_m0 = buf_m0[0] + buf_m0[1] + buf_m0[2] + buf_m0[3];
        let sum_m1 = buf_m1[0] + buf_m1[1] + buf_m1[2] + buf_m1[3];

        let mut buf0 = [0.0f32; 8];
        let mut buf1 = [0.0f32; 8];
        _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
        let sum_acc0 =
            (buf0[0] + buf0[1] + buf0[2] + buf0[3]) + (buf0[4] + buf0[5] + buf0[6] + buf0[7]);
        let sum_acc1 =
            (buf1[0] + buf1[1] + buf1[2] + buf1[3]) + (buf1[4] + buf1[5] + buf1[6] + buf1[7]);

        (sum_acc0 + sum_m0, sum_acc1 + sum_m1)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_k_q8_k_4rows_avx2(
    row0_weights: &[crate::gguf_quants::BlockQ4_K],
    row1_weights: &[crate::gguf_quants::BlockQ4_K],
    row2_weights: &[crate::gguf_quants::BlockQ4_K],
    row3_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::{
        _mm_cvtepi32_ps, _mm_fmadd_ps, _mm_hadd_epi16, _mm_madd_epi16, _mm_set_epi32, _mm_set1_ps,
        _mm_setzero_ps, _mm_storeu_ps, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi128_si256,
        _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_fmadd_ps,
        _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_set1_epi8, _mm256_set1_ps, _mm256_setzero_ps, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_ps,
    };

    static K_SHUFFLE_K4: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];

    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    // SAFETY: caller guarantees avx2 and fma features are enabled and slices have valid aligned data.
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

        let num_blocks = row0_weights
            .len()
            .min(row1_weights.len())
            .min(row2_weights.len())
            .min(row3_weights.len())
            .min(act_q8_k.len());

        for b in 0..num_blocks {
            let x0 = &row0_weights[b];
            let x1 = &row1_weights[b];
            let x2 = &row2_weights[b];
            let x3 = &row3_weights[b];
            let y = &act_q8_k[b];

            let d0 = y.d * x0.d.to_f32();
            let dmin0 = -y.d * x0.dmin.to_f32();
            let d1 = y.d * x1.d.to_f32();
            let dmin1 = -y.d * x1.dmin.to_f32();
            let d2 = y.d * x2.d.to_f32();
            let dmin2 = -y.d * x2.dmin.to_f32();
            let d3 = y.d * x3.d.to_f32();
            let dmin3 = -y.d * x3.dmin.to_f32();

            // Unpack scales for 4 rows
            let mut utmp0 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x0.scales.as_ptr(), utmp0.as_mut_ptr().cast::<u8>(), 12);
            utmp0[3] = ((utmp0[2] >> 4) & KMASK2) | (((utmp0[1] >> 6) & KMASK3) << 4);
            let uaux0 = utmp0[1] & KMASK1;
            utmp0[1] = (utmp0[2] & KMASK2) | (((utmp0[0] >> 6) & KMASK3) << 4);
            utmp0[2] = uaux0;
            utmp0[0] &= KMASK1;

            let mins_and_scales0 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp0[3] as i32,
                utmp0[2] as i32,
                utmp0[1] as i32,
                utmp0[0] as i32,
            ));

            let mut utmp1 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x1.scales.as_ptr(), utmp1.as_mut_ptr().cast::<u8>(), 12);
            utmp1[3] = ((utmp1[2] >> 4) & KMASK2) | (((utmp1[1] >> 6) & KMASK3) << 4);
            let uaux1 = utmp1[1] & KMASK1;
            utmp1[1] = (utmp1[2] & KMASK2) | (((utmp1[0] >> 6) & KMASK3) << 4);
            utmp1[2] = uaux1;
            utmp1[0] &= KMASK1;

            let mins_and_scales1 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp1[3] as i32,
                utmp1[2] as i32,
                utmp1[1] as i32,
                utmp1[0] as i32,
            ));

            let mut utmp2 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x2.scales.as_ptr(), utmp2.as_mut_ptr().cast::<u8>(), 12);
            utmp2[3] = ((utmp2[2] >> 4) & KMASK2) | (((utmp2[1] >> 6) & KMASK3) << 4);
            let uaux2 = utmp2[1] & KMASK1;
            utmp2[1] = (utmp2[2] & KMASK2) | (((utmp2[0] >> 6) & KMASK3) << 4);
            utmp2[2] = uaux2;
            utmp2[0] &= KMASK1;

            let mins_and_scales2 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp2[3] as i32,
                utmp2[2] as i32,
                utmp2[1] as i32,
                utmp2[0] as i32,
            ));

            let mut utmp3 = [0u32; 4];
            std::ptr::copy_nonoverlapping(x3.scales.as_ptr(), utmp3.as_mut_ptr().cast::<u8>(), 12);
            utmp3[3] = ((utmp3[2] >> 4) & KMASK2) | (((utmp3[1] >> 6) & KMASK3) << 4);
            let uaux3 = utmp3[1] & KMASK1;
            utmp3[1] = (utmp3[2] & KMASK2) | (((utmp3[0] >> 6) & KMASK3) << 4);
            utmp3[2] = uaux3;
            utmp3[0] &= KMASK1;

            let mins_and_scales3 = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp3[3] as i32,
                utmp3[2] as i32,
                utmp3[1] as i32,
                utmp3[0] as i32,
            ));

            // Activation bsums loaded ONCE for all 4 rows
            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr().cast());
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );

            let prod0 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales0, 1), q8s);
            acc_m0 = _mm_fmadd_ps(_mm_set1_ps(dmin0), _mm_cvtepi32_ps(prod0), acc_m0);

            let prod1 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales1, 1), q8s);
            acc_m1 = _mm_fmadd_ps(_mm_set1_ps(dmin1), _mm_cvtepi32_ps(prod1), acc_m1);

            let prod2 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales2, 1), q8s);
            acc_m2 = _mm_fmadd_ps(_mm_set1_ps(dmin2), _mm_cvtepi32_ps(prod2), acc_m2);

            let prod3 = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales3, 1), q8s);
            acc_m3 = _mm_fmadd_ps(_mm_set1_ps(dmin3), _mm_cvtepi32_ps(prod3), acc_m3);

            let sc128_0 = _mm256_extracti128_si256(mins_and_scales0, 0);
            let scales0 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_0), sc128_0, 1);

            let sc128_1 = _mm256_extracti128_si256(mins_and_scales1, 0);
            let scales1 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_1), sc128_1, 1);

            let sc128_2 = _mm256_extracti128_si256(mins_and_scales2, 0);
            let scales2 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_2), sc128_2, 1);

            let sc128_3 = _mm256_extracti128_si256(mins_and_scales3, 0);
            let scales3 = _mm256_inserti128_si256(_mm256_castsi128_si256(sc128_3), sc128_3, 1);

            let mut sumi0 = _mm256_setzero_si256();
            let mut sumi1 = _mm256_setzero_si256();
            let mut sumi2 = _mm256_setzero_si256();
            let mut sumi3 = _mm256_setzero_si256();

            let mut q4_ptr0 = x0.qs.as_ptr();
            let mut q4_ptr1 = x1.qs.as_ptr();
            let mut q4_ptr2 = x2.qs.as_ptr();
            let mut q4_ptr3 = x3.qs.as_ptr();
            let mut q8_ptr = y.qs.as_ptr();

            for j in 0..4 {
                let shuf_l = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j) * 32).cast());
                let shuf_h = _mm256_loadu_si256(K_SHUFFLE_K4.as_ptr().add((2 * j + 1) * 32).cast());

                let scale_l0 = _mm256_shuffle_epi8(scales0, shuf_l);
                let scale_h0 = _mm256_shuffle_epi8(scales0, shuf_h);
                let scale_l1 = _mm256_shuffle_epi8(scales1, shuf_l);
                let scale_h1 = _mm256_shuffle_epi8(scales1, shuf_h);
                let scale_l2 = _mm256_shuffle_epi8(scales2, shuf_l);
                let scale_h2 = _mm256_shuffle_epi8(scales2, shuf_h);
                let scale_l3 = _mm256_shuffle_epi8(scales3, shuf_l);
                let scale_h3 = _mm256_shuffle_epi8(scales3, shuf_h);

                // Load 64 activation bytes ONCE for all 4 rows
                let q8l = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);
                let q8h = _mm256_loadu_si256(q8_ptr.cast());
                q8_ptr = q8_ptr.add(32);

                // Row 0
                let q4bits0 = _mm256_loadu_si256(q4_ptr0.cast());
                q4_ptr0 = q4_ptr0.add(32);
                let q4l0 = _mm256_and_si256(q4bits0, m4);
                let q4h0 = _mm256_and_si256(_mm256_srli_epi16(q4bits0, 4), m4);
                let p16l0 = _mm256_madd_epi16(scale_l0, _mm256_maddubs_epi16(q4l0, q8l));
                let p16h0 = _mm256_madd_epi16(scale_h0, _mm256_maddubs_epi16(q4h0, q8h));
                sumi0 = _mm256_add_epi32(sumi0, _mm256_add_epi32(p16l0, p16h0));

                // Row 1
                let q4bits1 = _mm256_loadu_si256(q4_ptr1.cast());
                q4_ptr1 = q4_ptr1.add(32);
                let q4l1 = _mm256_and_si256(q4bits1, m4);
                let q4h1 = _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4);
                let p16l1 = _mm256_madd_epi16(scale_l1, _mm256_maddubs_epi16(q4l1, q8l));
                let p16h1 = _mm256_madd_epi16(scale_h1, _mm256_maddubs_epi16(q4h1, q8h));
                sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p16l1, p16h1));

                // Row 2
                let q4bits2 = _mm256_loadu_si256(q4_ptr2.cast());
                q4_ptr2 = q4_ptr2.add(32);
                let q4l2 = _mm256_and_si256(q4bits2, m4);
                let q4h2 = _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4);
                let p16l2 = _mm256_madd_epi16(scale_l2, _mm256_maddubs_epi16(q4l2, q8l));
                let p16h2 = _mm256_madd_epi16(scale_h2, _mm256_maddubs_epi16(q4h2, q8h));
                sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p16l2, p16h2));

                // Row 3
                let q4bits3 = _mm256_loadu_si256(q4_ptr3.cast());
                q4_ptr3 = q4_ptr3.add(32);
                let q4l3 = _mm256_and_si256(q4bits3, m4);
                let q4h3 = _mm256_and_si256(_mm256_srli_epi16(q4bits3, 4), m4);
                let p16l3 = _mm256_madd_epi16(scale_l3, _mm256_maddubs_epi16(q4l3, q8l));
                let p16h3 = _mm256_madd_epi16(scale_h3, _mm256_maddubs_epi16(q4h3, q8h));
                sumi3 = _mm256_add_epi32(sumi3, _mm256_add_epi32(p16l3, p16h3));
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(sumi2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(sumi3), acc3);
        }

        let mut buf_m0 = [0.0f32; 4];
        let mut buf_m1 = [0.0f32; 4];
        let mut buf_m2 = [0.0f32; 4];
        let mut buf_m3 = [0.0f32; 4];
        _mm_storeu_ps(buf_m0.as_mut_ptr(), acc_m0);
        _mm_storeu_ps(buf_m1.as_mut_ptr(), acc_m1);
        _mm_storeu_ps(buf_m2.as_mut_ptr(), acc_m2);
        _mm_storeu_ps(buf_m3.as_mut_ptr(), acc_m3);
        let sum_m0 = buf_m0[0] + buf_m0[1] + buf_m0[2] + buf_m0[3];
        let sum_m1 = buf_m1[0] + buf_m1[1] + buf_m1[2] + buf_m1[3];
        let sum_m2 = buf_m2[0] + buf_m2[1] + buf_m2[2] + buf_m2[3];
        let sum_m3 = buf_m3[0] + buf_m3[1] + buf_m3[2] + buf_m3[3];

        let mut buf0 = [0.0f32; 8];
        let mut buf1 = [0.0f32; 8];
        let mut buf2 = [0.0f32; 8];
        let mut buf3 = [0.0f32; 8];
        _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
        _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
        _mm256_storeu_ps(buf2.as_mut_ptr(), acc2);
        _mm256_storeu_ps(buf3.as_mut_ptr(), acc3);
        let sum_acc0 =
            (buf0[0] + buf0[1] + buf0[2] + buf0[3]) + (buf0[4] + buf0[5] + buf0[6] + buf0[7]);
        let sum_acc1 =
            (buf1[0] + buf1[1] + buf1[2] + buf1[3]) + (buf1[4] + buf1[5] + buf1[6] + buf1[7]);
        let sum_acc2 =
            (buf2[0] + buf2[1] + buf2[2] + buf2[3]) + (buf2[4] + buf2[5] + buf2[6] + buf2[7]);
        let sum_acc3 =
            (buf3[0] + buf3[1] + buf3[2] + buf3[3]) + (buf3[4] + buf3[5] + buf3[6] + buf3[7]);

        (
            sum_acc0 + sum_m0,
            sum_acc1 + sum_m1,
            sum_acc2 + sum_m2,
            sum_acc3 + sum_m3,
        )
    }
}
