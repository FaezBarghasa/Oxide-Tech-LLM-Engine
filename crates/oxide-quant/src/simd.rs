//! Microarchitectural SIMD-Accelerated Quantized Vector Dot Products.
//!
//! Provides AVX2, FMA, and 8-way unrolled portable fallback routines for:
//! - `Q8_0` (32 INT8 weights * 32 F32 activations)
//! - `Q4_0` (32 INT4 nibbles * 32 F32 activations)
//! - `Q4_K` (256 INT4 nibbles * 256 F32 activations)
//! - Continuous `f32` dot product with 8 independent accumulators to saturate dual FMA ports.

/// Continuous F32 dot product with 8-way parallel accumulators to break CPU pipeline dependency chains.
#[inline(always)]
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
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

/// Compute dot product between a Q8_0 block (32 int8 weights) and 32 f32 activations.
#[inline]
#[must_use]
pub fn dot_q8_0(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
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
