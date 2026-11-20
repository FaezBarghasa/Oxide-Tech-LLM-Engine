//! x86_64 AVX2 & FMA Micro-Kernels.

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
#[target_feature(enable = "avx")]
pub unsafe fn dot_f32_avx(a: &[f32], b: &[f32]) -> f32 {
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
pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
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


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q8_0_avx2(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
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

/// Compute dot product between a Q5_0 block (22 bytes = 32 weights) and 32 f32 activations.


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q5_0_avx2(qs: &[u8; 16], qh: &[u8; 4], act: &[f32; 32], scale: f32) -> f32 {
    use core::arch::x86_64::{
        _mm_loadu_si128, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let qh_u32 = u32::from_le_bytes(*qh);
    let mut q0_bytes = [0i8; 16];
    let mut q1_bytes = [0i8; 16];
    for i in 0..16 {
        let byte = qs[i];
        let h0 = (qh_u32 >> i) & 1;
        let h1 = (qh_u32 >> (i + 16)) & 1;
        q0_bytes[i] = ((byte & 0x0F) | ((h0 as u8) << 4)) as i8 - 16;
        q1_bytes[i] = (((byte >> 4) & 0x0F) | ((h1 as u8) << 4)) as i8 - 16;
    }

    // SAFETY: AVX2 and FMA extensions are validated by caller.
    unsafe {
        let lo_nibbles = _mm_loadu_si128(q0_bytes.as_ptr().cast());
        let hi_nibbles = _mm_loadu_si128(q1_bytes.as_ptr().cast());

        let f32_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_nibbles));
        let act_lo0 = _mm256_loadu_ps(act.as_ptr());
        let mut sum_vec = _mm256_fmadd_ps(f32_lo0, act_lo0, _mm256_setzero_ps());

        let raw_lo_high8 = core::arch::x86_64::_mm_srli_si128(lo_nibbles, 8);
        let f32_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw_lo_high8));
        let act_lo1 = _mm256_loadu_ps(act.as_ptr().add(8));
        sum_vec = _mm256_fmadd_ps(f32_lo1, act_lo1, sum_vec);

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



#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q5_0_q8_0_row_avx2(
    row_weights: &[crate::int_quant::BlockQ5_0],
    act_q8: &[crate::int_quant::BlockQ8_0],
) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_set1_epi8, _mm_srli_epi16, _mm256_and_si256,
        _mm256_castsi128_si256, _mm256_cmpeq_epi8, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_insertf128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
        _mm256_or_si256, _mm256_set_epi64x, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32,
        _mm256_set1_epi64x, _mm256_set1_ps, _mm256_setzero_ps, _mm256_shuffle_epi8,
        _mm256_sign_epi8, _mm256_storeu_ps, _mm256_sub_epi8,
    };
    let nb = row_weights.len();
    // SAFETY: AVX2 and FMA extensions are validated by caller.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let ones_16 = _mm256_set1_epi16(1);
        let low_mask = _mm_set1_epi8(0x0F);
        let shuf_mask = _mm256_set_epi64x(
            0x0303030303030303,
            0x0202020202020202,
            0x0101010101010101,
            0x0000000000000000,
        );
        let bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64);
        let all_ones = _mm256_set1_epi64x(-1);
        let bit16 = _mm256_set1_epi8(0x10);
        let sixteen = _mm256_set1_epi8(16);

        for ib in 0..nb {
            let w_blk = &row_weights[ib];
            let a_blk = &act_q8[ib];
            let d = w_blk.scale.to_f32() * a_blk.scale.to_f32();
            let d_vec = _mm256_set1_ps(d);

            // 1. Unpack 4-bit nibbles from qs (16 bytes) into 32 bytes (low 4 bits in each byte)
            let tmpl = _mm_loadu_si128(w_blk.qs.as_ptr().cast());
            let tmph = _mm_srli_epi16(tmpl, 4);
            let lo = _mm_and_si128(tmpl, low_mask);
            let hi = _mm_and_si128(tmph, low_mask);
            let qx_nibbles = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);

            // 2. Unpack 5th bits from qh (4 bytes = 32 bits) into 32 bytes
            let qh_u32 = u32::from_le_bytes(w_blk.qh);
            let bytes_shuf = _mm256_shuffle_epi8(_mm256_set1_epi32(qh_u32 as i32), shuf_mask);
            let bytes_or = _mm256_or_si256(bytes_shuf, bit_mask);
            let bxhi = _mm256_cmpeq_epi8(bytes_or, all_ones);
            let bxhi_masked = _mm256_and_si256(bxhi, bit16);

            // 3. Combine nibble + high bit and subtract 16 to get signed values in [-16, 15]
            let qx_u8 = _mm256_or_si256(qx_nibbles, bxhi_masked);
            let qx_signed = _mm256_sub_epi8(qx_u8, sixteen);

            // 4. Dot product with Q8_0 activation bytes
            let qy = _mm256_loadu_si256(a_blk.qs.as_ptr().cast());
            let ax = _mm256_sign_epi8(qx_signed, qx_signed);
            let sy = _mm256_sign_epi8(qy, qx_signed);

            let dot = _mm256_maddubs_epi16(ax, sy);
            let sum_i32 = _mm256_madd_epi16(ones_16, dot);
            let sum_f32 = _mm256_cvtepi32_ps(sum_i32);

            acc = _mm256_fmadd_ps(d_vec, sum_f32, acc);
        }

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7])
    }
}



#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q8_0_q8_0_row_avx2(
    row_weights: &[crate::int_quant::BlockQ8_0],
    act_q8: &[crate::int_quant::BlockQ8_0],
) -> f32 {
    use core::arch::x86_64::{
        _mm256_cvtepi32_ps, _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_sign_epi8, _mm256_storeu_ps,
    };
    let nb = row_weights.len();
    // SAFETY: AVX2 and FMA extensions are validated by caller.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let ones_16 = _mm256_set1_epi16(1);

        for ib in 0..nb {
            let w_blk = &row_weights[ib];
            let a_blk = &act_q8[ib];
            let d = w_blk.scale.to_f32() * a_blk.scale.to_f32();
            let d_vec = _mm256_set1_ps(d);

            let qx = _mm256_loadu_si256(w_blk.qs.as_ptr().cast());
            let qy = _mm256_loadu_si256(a_blk.qs.as_ptr().cast());

            let ax = _mm256_sign_epi8(qx, qx);
            let sy = _mm256_sign_epi8(qy, qx);

            let dot = _mm256_maddubs_epi16(ax, sy);
            let sum_i32 = _mm256_madd_epi16(ones_16, dot);
            let sum_f32 = _mm256_cvtepi32_ps(sum_i32);

            acc = _mm256_fmadd_ps(d_vec, sum_f32, acc);
        }

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
        (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7])
    }
}

/// Compute dot product between a Q4_0 block (16 bytes = 32 nibbles) and 32 f32 activations.


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_0_avx2(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
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


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q4_k_avx2(
    qs: &[u8; 128],
    scales: &[u8; 12],
    act: &[f32; 256],
    d: f32,
    dmin: f32,
) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadl_epi64, _mm_set1_epi8, _mm_srli_epi16, _mm256_cvtepi32_ps,
        _mm256_cvtepu8_epi32, _mm256_fmadd_ps, _mm256_fmsub_ps, _mm256_loadu_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: caller guarantees target features avx2 and fma are enabled and references are valid.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mut acc_vec = _mm256_setzero_ps();
        let mut is = 0;

        for chunk in 0..4 {
            let (sc1, m1) = crate::gguf_quants::get_scale_min_k4(is, scales);
            let d1_vec = _mm256_set1_ps(d * sc1);
            let min1_vec = _mm256_set1_ps(dmin * m1);

            let (sc2, m2) = crate::gguf_quants::get_scale_min_k4(is + 1, scales);
            let d2_vec = _mm256_set1_ps(d * sc2);
            let min2_vec = _mm256_set1_ps(dmin * m2);

            let q_base = chunk * 32;
            let act_base = chunk * 64;

            for step in 0..4 {
                let byte_offset = q_base + step * 8;
                let raw_8 = _mm_loadl_epi64(qs.as_ptr().add(byte_offset).cast());
                let lo = _mm_and_si128(raw_8, mask_0f);
                let hi = _mm_and_si128(_mm_srli_epi16(raw_8, 4), mask_0f);

                let q0_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
                let q1_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));

                let act0 = _mm256_loadu_ps(act.as_ptr().add(act_base + step * 8));
                let act1 = _mm256_loadu_ps(act.as_ptr().add(act_base + 32 + step * 8));

                let w0 = _mm256_fmsub_ps(q0_f32, d1_vec, min1_vec);
                let w1 = _mm256_fmsub_ps(q1_f32, d2_vec, min2_vec);

                acc_vec = _mm256_fmadd_ps(w0, act0, acc_vec);
                acc_vec = _mm256_fmadd_ps(w1, act1, acc_vec);
            }
            is += 2;
        }

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), acc_vec);
        buf.iter().sum()
    }
}

/// Compute dot product between a Q6_K super-block (210 bytes = 256 weights) and 256 f32 activations.


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_q6_k_avx2(
    ql: &[u8; 128],
    qh: &[u8; 64],
    scales: &[i8; 16],
    act: &[f32; 256],
    d: f32,
) -> f32 {
    use core::arch::x86_64::{
        _mm_and_si128, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi8, _mm_slli_epi16,
        _mm_srli_epi16, _mm_sub_epi8, _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    // SAFETY: AVX2 and FMA extensions are validated by caller.
    unsafe {
        let mask_0f = _mm_set1_epi8(0x0F);
        let mask_03 = _mm_set1_epi8(0x03);
        let thirty_two = _mm_set1_epi8(32);
        let mut sum_vec = _mm256_setzero_ps();

        for half in 0..2 {
            let ql_offset = half * 64;
            let qh_offset = half * 32;
            let sc_offset = half * 8;
            let y_offset = half * 128;

            for sub in 0..2 {
                let l_base = sub * 16;
                let is = sub;

                let sc0 = d * (scales[sc_offset + is] as f32);
                let sc2 = d * (scales[sc_offset + is + 2] as f32);
                let sc4 = d * (scales[sc_offset + is + 4] as f32);
                let sc6 = d * (scales[sc_offset + is + 6] as f32);

                let sc0_vec = _mm256_set1_ps(sc0);
                let sc2_vec = _mm256_set1_ps(sc2);
                let sc4_vec = _mm256_set1_ps(sc4);
                let sc6_vec = _mm256_set1_ps(sc6);

                let ql0_raw = _mm_loadu_si128(ql.as_ptr().add(ql_offset + l_base).cast());
                let ql32_raw = _mm_loadu_si128(ql.as_ptr().add(ql_offset + l_base + 32).cast());
                let qh_raw = _mm_loadu_si128(qh.as_ptr().add(qh_offset + l_base).cast());

                let q1_vec = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql0_raw, mask_0f),
                        _mm_slli_epi16(_mm_and_si128(qh_raw, mask_03), 4),
                    ),
                    thirty_two,
                );
                let q2_vec = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(ql32_raw, mask_0f),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 2), mask_03), 4),
                    ),
                    thirty_two,
                );
                let q3_vec = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql0_raw, 4), mask_0f),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 4), mask_03), 4),
                    ),
                    thirty_two,
                );
                let q4_vec = _mm_sub_epi8(
                    _mm_or_si128(
                        _mm_and_si128(_mm_srli_epi16(ql32_raw, 4), mask_0f),
                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_raw, 6), mask_03), 4),
                    ),
                    thirty_two,
                );

                macro_rules! madd_16 {
                    ($q_vec:expr, $sc_vec:expr, $act_offset:expr) => {
                        let f_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32($q_vec));
                        let f_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                            core::arch::x86_64::_mm_srli_si128($q_vec, 8),
                        ));
                        let act_lo = _mm256_loadu_ps(act.as_ptr().add($act_offset));
                        let act_hi = _mm256_loadu_ps(act.as_ptr().add($act_offset + 8));
                        sum_vec = _mm256_fmadd_ps(f_lo, _mm256_mul_ps(act_lo, $sc_vec), sum_vec);
                        sum_vec = _mm256_fmadd_ps(f_hi, _mm256_mul_ps(act_hi, $sc_vec), sum_vec);
                    };
                }

                madd_16!(q1_vec, sc0_vec, y_offset + l_base);
                madd_16!(q2_vec, sc2_vec, y_offset + l_base + 32);
                madd_16!(q3_vec, sc4_vec, y_offset + l_base + 64);
                madd_16!(q4_vec, sc6_vec, y_offset + l_base + 96);
            }
        }

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), sum_vec);
        (buf[0] + buf[1] + buf[2] + buf[3]) + (buf[4] + buf[5] + buf[6] + buf[7])
    }
}


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



#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn swiglu_f32_avx2(gate: &[f32], up: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::{
        _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _mm256_add_ps, _mm256_castsi256_ps,
        _mm256_cvtps_epi32, _mm256_div_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps,
        _mm256_min_ps, _mm256_mul_ps, _mm256_round_ps, _mm256_set1_epi32, _mm256_set1_ps,
        _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let len = gate.len();
    let chunks = len / 8;

    // SAFETY: AVX2 and FMA extensions are validated by caller.
    unsafe {
        let c_log2e = _mm256_set1_ps(core::f32::consts::LOG2_E);
        let c_ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
        let c_one = _mm256_set1_ps(1.0);
        let c_min_z = _mm256_set1_ps(-87.0);
        let c_max_z = _mm256_set1_ps(87.0);
        let c_127 = _mm256_set1_epi32(127);

        // Polynomial coefficients for exp(r) on r in [-0.5*ln2, 0.5*ln2]
        let p1 = _mm256_set1_ps(1.0);
        let p2 = _mm256_set1_ps(0.5);
        let p3 = _mm256_set1_ps(0.166_666_67);
        let p4 = _mm256_set1_ps(0.041_666_668);
        let p5 = _mm256_set1_ps(0.008_333_334);

        for i in 0..chunks {
            let offset = i * 8;
            let g = _mm256_loadu_ps(gate.as_ptr().add(offset));
            let u = _mm256_loadu_ps(up.as_ptr().add(offset));

            // z = -g
            let z = _mm256_sub_ps(_mm256_set1_ps(0.0), g);
            let z_clamped = _mm256_max_ps(c_min_z, _mm256_min_ps(c_max_z, z));

            // k = round(z * log2(e))
            let k_f32 = _mm256_round_ps(
                _mm256_mul_ps(z_clamped, c_log2e),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            );
            let k_i32 = _mm256_cvtps_epi32(k_f32);

            // r = z - k * ln(2)
            let r = _mm256_sub_ps(z_clamped, _mm256_mul_ps(k_f32, c_ln2));

            // P(r) = 1 + r * (1 + r * (0.5 + r * (1/6 + r * (1/24 + r * 1/120))))
            let poly = _mm256_fmadd_ps(p5, r, p4);
            let poly = _mm256_fmadd_ps(poly, r, p3);
            let poly = _mm256_fmadd_ps(poly, r, p2);
            let poly = _mm256_fmadd_ps(poly, r, p1);
            let poly = _mm256_fmadd_ps(poly, r, c_one);

            // 2^k = (k + 127) << 23
            let pow2k_bits =
                _mm256_slli_epi32(core::arch::x86_64::_mm256_add_epi32(k_i32, c_127), 23);
            let pow2k = _mm256_castsi256_ps(pow2k_bits);

            let exp_z = _mm256_mul_ps(poly, pow2k);
            let denom = _mm256_add_ps(c_one, exp_z);
            let silu_g = _mm256_div_ps(g, denom);

            let res = _mm256_mul_ps(silu_g, u);
            _mm256_storeu_ps(output.as_mut_ptr().add(offset), res);
        }
    }

    for i in (chunks * 8)..len {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * up[i];
    }
}
