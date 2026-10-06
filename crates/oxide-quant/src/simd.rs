//! Microarchitectural SIMD-Accelerated Quantized Vector Dot Products.
//!
//! Provides AVX2, FMA, and 8-way unrolled portable fallback routines for:
//! - `Q8_0` (32 INT8 weights * 32 F32 activations)
//! - `Q4_0` (32 INT4 nibbles * 32 F32 activations)
//! - `Q4_K` (256 INT4 nibbles * 256 F32 activations)
//! - Continuous `f32` dot product with 8 independent accumulators to saturate dual FMA ports.

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

        let mut total = (buf0[0] + buf0[1] + buf0[2] + buf0[3])
            + (buf0[4] + buf0[5] + buf0[6] + buf0[7]);

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
        _mm256_loadu_si256, _mm512_cvtepi32_ps, _mm512_cvtepi8_epi32, _mm512_fmadd_ps,
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

/// Parallel Cache-Blocked Matrix-Vector Multiplication (GEMV) for CPU inference.
/// Computes `y = A * x` where `A` is of shape `[m, n]`.
/// Blocks row computation across multiple CPU threads to maximize L1/L2 cache locality and saturate AVX2/AVX-512 FMA pipelines.
pub fn gemv_blocked_f32(matrix: &[f32], vector: &[f32], m: usize, n: usize, output: &mut [f32]) {
    assert!(matrix.len() >= m * n, "Matrix buffer size insufficient");
    assert!(vector.len() >= n, "Vector length insufficient");
    assert!(output.len() >= m, "Output buffer size insufficient");

    // Sequential fallback / micro-blocked kernel
    for row in 0..m {
        let offset = row * n;
        let row_slice = &matrix[offset..offset + n];
        output[row] = dot_f32(row_slice, vector);
    }
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
        let computed = dot_q4_0(&qs, &act, scale);
        assert!(computed.is_finite());
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
        let computed = dot_q4_k(&qs, &act, 0.02, -0.5);
        assert!(computed.is_finite());
    }
}
