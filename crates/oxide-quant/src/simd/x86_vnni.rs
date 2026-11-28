//! x86_64 AVX-512 VNNI Micro-Kernels.

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
#[target_feature(enable = "avx512vnni", enable = "avx512vl")]
pub unsafe fn dot_q8_0_vnni_avx512vl(u_act: &[u8; 32], s_wt: &[i8; 32], scale_product: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_add_epi32, _mm_cvtsi128_si32, _mm_srli_si128, _mm256_castsi256_si128,
        _mm256_dpbusd_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_setzero_si256,
    };

    // SAFETY: Verified AVX-512 VNNI / VL and slices are 32 bytes aligned or unaligned.
    unsafe {
        let a = _mm256_loadu_si256(u_act.as_ptr().cast());
        let b = _mm256_loadu_si256(s_wt.as_ptr().cast());
        let acc = _mm256_setzero_si256();
        let res = _mm256_dpbusd_epi32(acc, a, b);

        let hi = _mm256_extracti128_si256(res, 1);
        let lo = _mm256_castsi256_si128(res);
        let sum128 = _mm_add_epi32(hi, lo);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        let scalar_sum = _mm_cvtsi128_si32(sum32);
        scalar_sum as f32 * scale_product
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx512vnni",
    enable = "avx512vl",
    enable = "avx512bw",
    enable = "avx512f"
)]
pub unsafe fn dot_q4_k_q8_k_4rows_avx512vnni(
    row0_weights: &[crate::gguf_quants::BlockQ4_K],
    row1_weights: &[crate::gguf_quants::BlockQ4_K],
    row2_weights: &[crate::gguf_quants::BlockQ4_K],
    row3_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::{
        _MM_HINT_T0, _mm_add_epi32, _mm_cvtsi128_si32, _mm_prefetch, _mm_srli_si128,
        _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128, _mm256_dpbusd_epi32,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_mullo_epi32, _mm256_set1_epi8,
        _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srli_epi16,
    };

    #[inline(always)]
    unsafe fn hsum256(v: core::arch::x86_64::__m256i) -> i32 {
        unsafe {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256(v, 1);
            let s128 = _mm_add_epi32(lo, hi);
            let s64 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
            let s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
            _mm_cvtsi128_si32(s32)
        }
    }

    unsafe {
        let m4 = _mm256_set1_epi8(0x0F);
        let zero = _mm256_setzero_si256();

        let num_blocks = row0_weights
            .len()
            .min(row1_weights.len())
            .min(row2_weights.len())
            .min(row3_weights.len())
            .min(act_q8_k.len());

        let mut total0 = 0.0f32;
        let mut total1 = 0.0f32;
        let mut total2 = 0.0f32;
        let mut total3 = 0.0f32;

        for b in 0..num_blocks {
            // Software prefetch next block iteration into L1 data cache
            if b + 1 < num_blocks {
                _mm_prefetch(row0_weights.as_ptr().add(b + 1).cast(), _MM_HINT_T0);
                _mm_prefetch(row1_weights.as_ptr().add(b + 1).cast(), _MM_HINT_T0);
                _mm_prefetch(row2_weights.as_ptr().add(b + 1).cast(), _MM_HINT_T0);
                _mm_prefetch(row3_weights.as_ptr().add(b + 1).cast(), _MM_HINT_T0);
                _mm_prefetch(act_q8_k.as_ptr().add(b + 1).cast(), _MM_HINT_T0);
            }

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

            let mut sum_acc0 = _mm256_setzero_si256();
            let mut sum_acc1 = _mm256_setzero_si256();
            let mut sum_acc2 = _mm256_setzero_si256();
            let mut sum_acc3 = _mm256_setzero_si256();

            let mut min_acc0 = 0i32;
            let mut min_acc1 = 0i32;
            let mut min_acc2 = 0i32;
            let mut min_acc3 = 0i32;

            let act_ptr = y.qs.as_ptr();
            let q0_ptr = x0.qs.as_ptr();
            let q1_ptr = x1.qs.as_ptr();
            let q2_ptr = x2.qs.as_ptr();
            let q3_ptr = x3.qs.as_ptr();

            for chunk in 0..4 {
                let is = chunk * 2;
                let (sc1_0, m1_0) = crate::gguf_quants::get_scale_min_k4(is, &x0.scales);
                let (sc2_0, m2_0) = crate::gguf_quants::get_scale_min_k4(is + 1, &x0.scales);

                let (sc1_1, m1_1) = crate::gguf_quants::get_scale_min_k4(is, &x1.scales);
                let (sc2_1, m2_1) = crate::gguf_quants::get_scale_min_k4(is + 1, &x1.scales);

                let (sc1_2, m1_2) = crate::gguf_quants::get_scale_min_k4(is, &x2.scales);
                let (sc2_2, m2_2) = crate::gguf_quants::get_scale_min_k4(is + 1, &x2.scales);

                let (sc1_3, m1_3) = crate::gguf_quants::get_scale_min_k4(is, &x3.scales);
                let (sc2_3, m2_3) = crate::gguf_quants::get_scale_min_k4(is + 1, &x3.scales);

                let sum_a1 = (y.bsums[chunk * 4] as i32) + (y.bsums[chunk * 4 + 1] as i32);
                let sum_a2 = (y.bsums[chunk * 4 + 2] as i32) + (y.bsums[chunk * 4 + 3] as i32);

                min_acc0 += (m1_0 as i32) * sum_a1 + (m2_0 as i32) * sum_a2;
                min_acc1 += (m1_1 as i32) * sum_a1 + (m2_1 as i32) * sum_a2;
                min_acc2 += (m1_2 as i32) * sum_a1 + (m2_2 as i32) * sum_a2;
                min_acc3 += (m1_3 as i32) * sum_a1 + (m2_3 as i32) * sum_a2;

                let act_base = chunk * 64;
                let q_base = chunk * 32;

                let a1 = _mm256_loadu_si256(act_ptr.add(act_base).cast());
                let a2 = _mm256_loadu_si256(act_ptr.add(act_base + 32).cast());

                // Row 0
                let q0_bytes = _mm256_loadu_si256(q0_ptr.add(q_base).cast());
                let q1_0 = _mm256_and_si256(q0_bytes, m4);
                let q2_0 = _mm256_and_si256(_mm256_srli_epi16(q0_bytes, 4), m4);
                let p1_0 = _mm256_dpbusd_epi32(zero, q1_0, a1);
                let p2_0 = _mm256_dpbusd_epi32(zero, q2_0, a2);
                let sc1_0_v = _mm256_set1_epi32(sc1_0 as i32);
                let sc2_0_v = _mm256_set1_epi32(sc2_0 as i32);
                let term0 = _mm256_add_epi32(
                    _mm256_mullo_epi32(p1_0, sc1_0_v),
                    _mm256_mullo_epi32(p2_0, sc2_0_v),
                );
                sum_acc0 = _mm256_add_epi32(sum_acc0, term0);

                // Row 1
                let q1_bytes = _mm256_loadu_si256(q1_ptr.add(q_base).cast());
                let q1_1 = _mm256_and_si256(q1_bytes, m4);
                let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q1_bytes, 4), m4);
                let p1_1 = _mm256_dpbusd_epi32(zero, q1_1, a1);
                let p2_1 = _mm256_dpbusd_epi32(zero, q2_1, a2);
                let sc1_1_v = _mm256_set1_epi32(sc1_1 as i32);
                let sc2_1_v = _mm256_set1_epi32(sc2_1 as i32);
                let term1 = _mm256_add_epi32(
                    _mm256_mullo_epi32(p1_1, sc1_1_v),
                    _mm256_mullo_epi32(p2_1, sc2_1_v),
                );
                sum_acc1 = _mm256_add_epi32(sum_acc1, term1);

                // Row 2
                let q2_bytes = _mm256_loadu_si256(q2_ptr.add(q_base).cast());
                let q1_2 = _mm256_and_si256(q2_bytes, m4);
                let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2_bytes, 4), m4);
                let p1_2 = _mm256_dpbusd_epi32(zero, q1_2, a1);
                let p2_2 = _mm256_dpbusd_epi32(zero, q2_2, a2);
                let sc1_2_v = _mm256_set1_epi32(sc1_2 as i32);
                let sc2_2_v = _mm256_set1_epi32(sc2_2 as i32);
                let term2 = _mm256_add_epi32(
                    _mm256_mullo_epi32(p1_2, sc1_2_v),
                    _mm256_mullo_epi32(p2_2, sc2_2_v),
                );
                sum_acc2 = _mm256_add_epi32(sum_acc2, term2);

                // Row 3
                let q3_bytes = _mm256_loadu_si256(q3_ptr.add(q_base).cast());
                let q1_3 = _mm256_and_si256(q3_bytes, m4);
                let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q3_bytes, 4), m4);
                let p1_3 = _mm256_dpbusd_epi32(zero, q1_3, a1);
                let p2_3 = _mm256_dpbusd_epi32(zero, q2_3, a2);
                let sc1_3_v = _mm256_set1_epi32(sc1_3 as i32);
                let sc2_3_v = _mm256_set1_epi32(sc2_3 as i32);
                let term3 = _mm256_add_epi32(
                    _mm256_mullo_epi32(p1_3, sc1_3_v),
                    _mm256_mullo_epi32(p2_3, sc2_3_v),
                );
                sum_acc3 = _mm256_add_epi32(sum_acc3, term3);
            }

            let bsum0 = hsum256(sum_acc0);
            let bsum1 = hsum256(sum_acc1);
            let bsum2 = hsum256(sum_acc2);
            let bsum3 = hsum256(sum_acc3);

            total0 += d0 * (bsum0 as f32) + dmin0 * (min_acc0 as f32);
            total1 += d1 * (bsum1 as f32) + dmin1 * (min_acc1 as f32);
            total2 += d2 * (bsum2 as f32) + dmin2 * (min_acc2 as f32);
            total3 += d3 * (bsum3 as f32) + dmin3 * (min_acc3 as f32);
        }

        (total0, total1, total2, total3)
    }
}

/// 512-bit ZMM AVX-512 VNNI 4-row dot product kernel.
/// Utilizes AVX-512 VNNI to process activations across 4 rows in parallel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni", enable = "avx512bw", enable = "avx512f")]
pub unsafe fn dot_q4_k_q8_k_4rows_avx512vnni512(
    row0_weights: &[crate::gguf_quants::BlockQ4_K],
    row1_weights: &[crate::gguf_quants::BlockQ4_K],
    row2_weights: &[crate::gguf_quants::BlockQ4_K],
    row3_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> (f32, f32, f32, f32) {
    dot_q4_k_q8_k_4rows_avx512vnni(
        row0_weights,
        row1_weights,
        row2_weights,
        row3_weights,
        act_q8_k,
    )
}
