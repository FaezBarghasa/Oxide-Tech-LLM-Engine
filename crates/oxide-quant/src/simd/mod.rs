//! Microarchitectural SIMD-Accelerated Quantized Vector Dot Products.
//!
//! Provides AVX-512, AVX2, FMA, ARM NEON/SVE2, and portable fallback routines.

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

pub mod arm_neon;
pub mod arm_sve2;
pub mod scalar;
pub mod x86_amx;
pub mod x86_avx2;
pub mod x86_avx512;
pub mod x86_vnni;

pub use scalar::*;
pub use x86_avx2::*;
pub use x86_avx512::*;
pub use x86_vnni::*;
pub use x86_amx as amx;

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



#[must_use]
pub fn dot_q5_0(qs: &[u8; 16], qh: &[u8; 4], act: &[f32; 32], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified feature flags before invoking target-specific intrinsics.
            unsafe {
                return dot_q5_0_avx2(qs, qh, act, scale);
            }
        }
    }

    dot_q5_0_portable(qs, qh, act, scale)
}

#[inline(always)]


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


#[must_use]
pub fn dot_q4_k(qs: &[u8; 128], scales: &[u8; 12], act: &[f32; 256], d: f32, dmin: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA support.
            unsafe {
                return dot_q4_k_avx2(qs, scales, act, d, dmin);
            }
        }
    }

    dot_q4_k_portable(qs, scales, act, d, dmin)
}

#[inline(always)]


#[must_use]
pub fn dot_q6_k(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], act: &[f32; 256], d: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified feature flags before invoking target-specific intrinsics.
            unsafe {
                return dot_q6_k_avx2(ql, qh, scales, act, d);
            }
        }
    }

    dot_q6_k_portable(ql, qh, scales, act, d)
}



#[inline]
fn get_team_threads() -> usize {
    if let Ok(val) = std::env::var("OXIDE_NUM_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    if let Ok(val) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = val.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    8
}

/// Quantize an f32 vector into pre-allocated BlockQ8_K blocks.
#[inline]
pub fn quantize_row_q8_k(vector: &[f32], q8_k_act: &mut Vec<crate::int_quant::BlockQ8_K>) {
    let blocks_per_row = vector.len() / 256;
    q8_k_act.clear();
    if q8_k_act.capacity() < blocks_per_row {
        q8_k_act.reserve(blocks_per_row - q8_k_act.capacity());
    }
    for b in 0..blocks_per_row {
        let act_chunk: &[f32; 256] = vector[b * 256..(b + 1) * 256]
            .try_into()
            .expect("slice length 256");
        q8_k_act.push(crate::int_quant::BlockQ8_K::quantize(act_chunk));
    }
}

/// Multithreaded Q6_K Matrix-Vector Multiplication using pre-quantized BlockQ8_K activations.
pub fn gemv_q6_k_prequantized(
    matrix: &[crate::int_quant::BlockQ6_K],
    q8_k_act: &[crate::int_quant::BlockQ8_K],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    let blocks_per_row = n / 256;
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;

    if m <= 8 {
        let mut row = 0;
        while row + 1 < m {
            let row_offset0 = row * blocks_per_row;
            let row_offset1 = (row + 1) * blocks_per_row;
            let r0 = &matrix[row_offset0..row_offset0 + blocks_per_row];
            let r1 = &matrix[row_offset1..row_offset1 + blocks_per_row];
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                let (v0, v1) = unsafe { dot_q6_k_q8_k_2rows_avx2(r0, r1, q8_k_act) };
                output[row] = v0;
                output[row + 1] = v1;
                row += 2;
                continue;
            }
            output[row] = dot_q6_k_q8_k_row_portable(r0, q8_k_act);
            output[row + 1] = dot_q6_k_q8_k_row_portable(r1, q8_k_act);
            row += 2;
        }
        if row < m {
            let row_offset = row * blocks_per_row;
            let row_weights = &matrix[row_offset..row_offset + blocks_per_row];
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                output[row] = unsafe { dot_q6_k_q8_k_row_avx2(row_weights, q8_k_act) };
            } else {
                output[row] = dot_q6_k_q8_k_row_portable(row_weights, q8_k_act);
            }
        }
        return;
    }

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let out_ptr = output.as_mut_ptr() as usize;

    team.dispatch_rows(m, |start, end| {
        let local_out = unsafe {
            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
        };
        let mut row = start;
        let mut i = 0;
        while row + 1 < end {
            let row_offset0 = row * blocks_per_row;
            let row_offset1 = (row + 1) * blocks_per_row;
            let r0 = &matrix[row_offset0..row_offset0 + blocks_per_row];
            let r1 = &matrix[row_offset1..row_offset1 + blocks_per_row];

            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                let (v0, v1) = unsafe { dot_q6_k_q8_k_2rows_avx2(r0, r1, q8_k_act) };
                local_out[i] = v0;
                local_out[i + 1] = v1;
                row += 2;
                i += 2;
                continue;
            }

            local_out[i] = dot_q6_k_q8_k_row_portable(r0, q8_k_act);
            local_out[i + 1] = dot_q6_k_q8_k_row_portable(r1, q8_k_act);
            row += 2;
            i += 2;
        }
        if row < end {
            let row_offset = row * blocks_per_row;
            let row_weights = &matrix[row_offset..row_offset + blocks_per_row];

            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                local_out[i] = unsafe { dot_q6_k_q8_k_row_avx2(row_weights, q8_k_act) };
            } else {
                local_out[i] = dot_q6_k_q8_k_row_portable(row_weights, q8_k_act);
            }
        }
    });
}

/// Multithreaded Q6_K Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q6_k(
    matrix: &[crate::int_quant::BlockQ6_K],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(
        n.is_multiple_of(256),
        "n must be a multiple of 256 for Q6_K"
    );
    let blocks_per_row = n / 256;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient Q6_K blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    let mut q8_k_act = Vec::with_capacity(blocks_per_row);
    quantize_row_q8_k(vector, &mut q8_k_act);
    gemv_q6_k_prequantized(matrix, &q8_k_act, m, n, output);
}

/// Multithreaded Q5_0 Matrix-Vector Multiplication across all CPU cores and threads.
pub fn gemv_q5_0(
    matrix: &[crate::int_quant::BlockQ5_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(n.is_multiple_of(32), "n must be a multiple of 32 for Q5_0");
    let blocks_per_row = n / 32;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient Q5_0 blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        let mut act_q8 = Vec::with_capacity(blocks_per_row);
        for b in 0..blocks_per_row {
            let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                .try_into()
                .expect("slice length 32");
            act_q8.push(crate::int_quant::BlockQ8_0::quantize(act_chunk));
        }

        if m <= 8 {
            for row in 0..m {
                let row_offset = row * blocks_per_row;
                // SAFETY: Target architecture supports AVX2 and FMA.
                output[row] = unsafe {
                    dot_q5_0_q8_0_row_avx2(
                        &matrix[row_offset..row_offset + blocks_per_row],
                        &act_q8,
                    )
                };
            }
        } else {
            let team = oxide_core::thread_team::global_thread_team(get_team_threads());
            let out_ptr = output.as_mut_ptr() as usize;

            team.dispatch_rows(m, |start, end| {
                let local_out = unsafe {
                    std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
                };
                for (i, row) in (start..end).enumerate() {
                    let row_offset = row * blocks_per_row;
                    // SAFETY: Target architecture supports AVX2 and FMA.
                    local_out[i] = unsafe {
                        dot_q5_0_q8_0_row_avx2(
                            &matrix[row_offset..row_offset + blocks_per_row],
                            &act_q8,
                        )
                    };
                }
            });
        }
        return;
    }

    if m <= 8 {
        for row in 0..m {
            let row_offset = row * blocks_per_row;
            let mut acc = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &matrix[row_offset + b];
                let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                    .try_into()
                    .expect("slice length 32");
                acc += dot_q5_0(&blk.qs, &blk.qh, act_chunk, blk.scale.to_f32());
            }
            output[row] = acc;
        }
    } else {
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                let row_offset = row * blocks_per_row;
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk = &matrix[row_offset + b];
                    let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                        .try_into()
                        .expect("slice length 32");
                    acc += dot_q5_0(&blk.qs, &blk.qh, act_chunk, blk.scale.to_f32());
                }
                local_out[i] = acc;
            }
        });
    }
}

/// AMX (Advanced Matrix Extensions) Tile Configuration and Compute Engine.
/// Provides architectural abstractions for Intel Xeon / Sapphire Rapids AMX tile registers (`TMM0`..`TMM7`).


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
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                let offset = row * n;
                let row_slice = &matrix[offset..offset + n];
                local_out[i] = dot_f32(row_slice, vector);
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

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        let mut act_q8 = Vec::with_capacity(blocks_per_row);
        for b in 0..blocks_per_row {
            let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                .try_into()
                .expect("slice length 32");
            act_q8.push(crate::int_quant::BlockQ8_0::quantize(act_chunk));
        }

        if m <= 8 {
            for row in 0..m {
                let row_offset = row * blocks_per_row;
                // SAFETY: Target architecture supports AVX2 and FMA.
                output[row] = unsafe {
                    dot_q8_0_q8_0_row_avx2(
                        &matrix[row_offset..row_offset + blocks_per_row],
                        &act_q8,
                    )
                };
            }
        } else {
            let team = oxide_core::thread_team::global_thread_team(get_team_threads());
            let out_ptr = output.as_mut_ptr() as usize;

            team.dispatch_rows(m, |start, end| {
                let local_out = unsafe {
                    std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
                };
                for (i, row) in (start..end).enumerate() {
                    let row_offset = row * blocks_per_row;
                    // SAFETY: Target architecture supports AVX2 and FMA.
                    local_out[i] = unsafe {
                        dot_q8_0_q8_0_row_avx2(
                            &matrix[row_offset..row_offset + blocks_per_row],
                            &act_q8,
                        )
                    };
                }
            });
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx512 = is_x86_feature_detected!("avx512f");

    let compute_row = |row: usize| -> f32 {
        let row_offset = row * blocks_per_row;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let blk = &matrix[row_offset + b];
            let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                .try_into()
                .expect("slice length 32");
            #[cfg(target_arch = "x86_64")]
            if use_avx512 {
                // SAFETY: Target feature AVX-512 has been checked via is_x86_feature_detected!.
                acc += unsafe { dot_q8_0_avx512(&blk.qs, act_chunk, blk.scale.to_f32()) };
                continue;
            }
            acc += dot_q8_0_portable(&blk.qs, act_chunk, blk.scale.to_f32());
        }
        acc
    };

    if m <= 8 {
        for row in 0..m {
            output[row] = compute_row(row);
        }
    } else {
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                local_out[i] = compute_row(row);
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
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                let row_offset = row * blocks_per_row;
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk = &matrix[row_offset + b];
                    let act_chunk: &[f32; 32] = vector[b * 32..(b + 1) * 32]
                        .try_into()
                        .expect("slice length 32");
                    acc += dot_q4_0(&blk.qs, act_chunk, blk.scale.to_f32());
                }
                local_out[i] = acc;
            }
        });
    }
}




/// Multithreaded Q4_K Matrix-Vector Multiplication using pre-quantized BlockQ8_K activations.
pub fn gemv_q4_k_prequantized(
    matrix: &[crate::gguf_quants::BlockQ4_K],
    q8_k_act: &[crate::int_quant::BlockQ8_K],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    let blocks_per_row = n / 256;
    #[cfg(target_arch = "x86_64")]
    let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl");
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let (has_vnni, has_avx2) = (false, false);

    if m <= 8 {
        let mut row = 0;
        #[cfg(target_arch = "x86_64")]
        if has_vnni {
            while row + 3 < m {
                let r0 = &matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let r1 = &matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let r2 = &matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let r3 = &matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (v0, v1, v2, v3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                output[row] = v0;
                output[row + 1] = v1;
                output[row + 2] = v2;
                output[row + 3] = v3;
                row += 4;
            }
        } else if has_avx2 {
            while row + 3 < m {
                let r0 = &matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let r1 = &matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let r2 = &matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let r3 = &matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (v0, v1, v2, v3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                output[row] = v0;
                output[row + 1] = v1;
                output[row + 2] = v2;
                output[row + 3] = v3;
                row += 4;
            }
        }
        while row + 1 < m {
            let row_offset0 = row * blocks_per_row;
            let row_offset1 = (row + 1) * blocks_per_row;
            let r0 = &matrix[row_offset0..row_offset0 + blocks_per_row];
            let r1 = &matrix[row_offset1..row_offset1 + blocks_per_row];
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                let (v0, v1) = unsafe { dot_q4_k_q8_k_2rows_avx2(r0, r1, q8_k_act) };
                output[row] = v0;
                output[row + 1] = v1;
                row += 2;
                continue;
            }
            output[row] = dot_q4_k_q8_k_row_portable(r0, q8_k_act);
            output[row + 1] = dot_q4_k_q8_k_row_portable(r1, q8_k_act);
            row += 2;
        }
        if row < m {
            let row_offset = row * blocks_per_row;
            let row_blocks = &matrix[row_offset..row_offset + blocks_per_row];
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                output[row] = unsafe { dot_q4_k_q8_k_row_avx2(row_blocks, q8_k_act) };
            } else {
                output[row] = dot_q4_k_q8_k_row_portable(row_blocks, q8_k_act);
            }
        }
        return;
    }

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let out_ptr = output.as_mut_ptr() as usize;

    team.dispatch_rows(m, |start, end| {
        let local_out = unsafe {
            std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
        };
        let mut row = start;
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if has_vnni {
            while row + 3 < end {
                let r0 = &matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let r1 = &matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let r2 = &matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let r3 = &matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (v0, v1, v2, v3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                local_out[i] = v0;
                local_out[i + 1] = v1;
                local_out[i + 2] = v2;
                local_out[i + 3] = v3;
                row += 4;
                i += 4;
            }
        } else if has_avx2 {
            while row + 3 < end {
                let r0 = &matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let r1 = &matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let r2 = &matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let r3 = &matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (v0, v1, v2, v3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                local_out[i] = v0;
                local_out[i + 1] = v1;
                local_out[i + 2] = v2;
                local_out[i + 3] = v3;
                row += 4;
                i += 4;
            }
        }
        while row + 1 < end {
            let row_offset0 = row * blocks_per_row;
            let row_offset1 = (row + 1) * blocks_per_row;
            let r0 = &matrix[row_offset0..row_offset0 + blocks_per_row];
            let r1 = &matrix[row_offset1..row_offset1 + blocks_per_row];

            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                let (v0, v1) = unsafe { dot_q4_k_q8_k_2rows_avx2(r0, r1, q8_k_act) };
                local_out[i] = v0;
                local_out[i + 1] = v1;
                row += 2;
                i += 2;
                continue;
            }

            local_out[i] = dot_q4_k_q8_k_row_portable(r0, q8_k_act);
            local_out[i + 1] = dot_q4_k_q8_k_row_portable(r1, q8_k_act);
            row += 2;
            i += 2;
        }
        if row < end {
            let row_offset = row * blocks_per_row;
            let row_blocks = &matrix[row_offset..row_offset + blocks_per_row];

            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                local_out[i] = unsafe { dot_q4_k_q8_k_row_avx2(row_blocks, q8_k_act) };
            } else {
                local_out[i] = dot_q4_k_q8_k_row_portable(row_blocks, q8_k_act);
            }
        }
    });
}

/// Fused Q, K, V Matrix-Vector Multiplication in a single ThreadTeam dispatch pass.
#[allow(clippy::too_many_arguments)]
pub fn gemv_qkv_q4_k_prequantized(
    q_matrix: &[crate::gguf_quants::BlockQ4_K],
    m_q: usize,
    k_matrix: &[crate::gguf_quants::BlockQ4_K],
    m_k: usize,
    v_matrix: &[crate::gguf_quants::BlockQ4_K],
    m_v: usize,
    n: usize,
    q8_k_act: &[crate::int_quant::BlockQ8_K],
    out_q: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
) {
    let blocks_per_row = n / 256;
    let m_total = m_q + m_k + m_v;

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let q_ptr = out_q.as_mut_ptr() as usize;
    let k_ptr = out_k.as_mut_ptr() as usize;
    let v_ptr = out_v.as_mut_ptr() as usize;

    team.dispatch_rows(m_total, |start, end| {
        let mut row = start;
        #[cfg(target_arch = "x86_64")]
        let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl");
        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let (has_vnni, has_avx2) = (false, false);

        #[cfg(target_arch = "x86_64")]
        if has_vnni {
            while row + 3 < end {
                if row + 3 < m_q {
                    let r0 = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                    let r1 = &q_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                    let r2 = &q_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                    let r3 = &q_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (q_ptr as *mut f32).add(row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q && row + 3 < m_q + m_k {
                    let k_row = row - m_q;
                    let r0 = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                    let r1 = &k_matrix[(k_row + 1) * blocks_per_row..(k_row + 2) * blocks_per_row];
                    let r2 = &k_matrix[(k_row + 2) * blocks_per_row..(k_row + 3) * blocks_per_row];
                    let r3 = &k_matrix[(k_row + 3) * blocks_per_row..(k_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (k_ptr as *mut f32).add(k_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q + m_k && row + 3 < m_total {
                    let v_row = row - (m_q + m_k);
                    let r0 = &v_matrix[v_row * blocks_per_row..(v_row + 1) * blocks_per_row];
                    let r1 = &v_matrix[(v_row + 1) * blocks_per_row..(v_row + 2) * blocks_per_row];
                    let r2 = &v_matrix[(v_row + 2) * blocks_per_row..(v_row + 3) * blocks_per_row];
                    let r3 = &v_matrix[(v_row + 3) * blocks_per_row..(v_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (v_ptr as *mut f32).add(v_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                }
                break;
            }
        } else if has_avx2 {
            while row + 3 < end {
                if row + 3 < m_q {
                    let r0 = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                    let r1 = &q_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                    let r2 = &q_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                    let r3 = &q_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (q_ptr as *mut f32).add(row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q && row + 3 < m_q + m_k {
                    let k_row = row - m_q;
                    let r0 = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                    let r1 = &k_matrix[(k_row + 1) * blocks_per_row..(k_row + 2) * blocks_per_row];
                    let r2 = &k_matrix[(k_row + 2) * blocks_per_row..(k_row + 3) * blocks_per_row];
                    let r3 = &k_matrix[(k_row + 3) * blocks_per_row..(k_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (k_ptr as *mut f32).add(k_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q + m_k && row + 3 < m_total {
                    let v_row = row - (m_q + m_k);
                    let r0 = &v_matrix[v_row * blocks_per_row..(v_row + 1) * blocks_per_row];
                    let r1 = &v_matrix[(v_row + 1) * blocks_per_row..(v_row + 2) * blocks_per_row];
                    let r2 = &v_matrix[(v_row + 2) * blocks_per_row..(v_row + 3) * blocks_per_row];
                    let r3 = &v_matrix[(v_row + 3) * blocks_per_row..(v_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (v_ptr as *mut f32).add(v_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                }
                break;
            }
        }

        while row < end {
            if row < m_q {
                let r = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q4_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q4_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((q_ptr as *mut f32).add(row)) = val;
                }
            } else if row < m_q + m_k {
                let k_row = row - m_q;
                let r = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q4_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q4_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((k_ptr as *mut f32).add(k_row)) = val;
                }
            } else {
                let v_row = row - (m_q + m_k);
                let r = &v_matrix[v_row * blocks_per_row..(v_row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q4_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q4_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((v_ptr as *mut f32).add(v_row)) = val;
                }
            }
            row += 1;
        }
    });
}

/// Unified Fused Q(Q4_K) + K(Q4_K) + V(Q6_K) Matrix-Vector Multiplication in a single ThreadTeam pass.
#[allow(clippy::too_many_arguments, clippy::if_not_else)]
pub fn gemv_qkv_q4_k_q6_k_prequantized(
    q_matrix: &[crate::gguf_quants::BlockQ4_K],
    m_q: usize,
    k_matrix: &[crate::gguf_quants::BlockQ4_K],
    m_k: usize,
    v_matrix: &[crate::int_quant::BlockQ6_K],
    m_v: usize,
    n: usize,
    q8_k_act: &[crate::int_quant::BlockQ8_K],
    out_q: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
) {
    let blocks_per_row = n / 256;
    let m_total = m_q + m_k + m_v;

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let q_ptr = out_q.as_mut_ptr() as usize;
    let k_ptr = out_k.as_mut_ptr() as usize;
    let v_ptr = out_v.as_mut_ptr() as usize;

    team.dispatch_rows(m_total, |start, end| {
        let mut row = start;
        #[cfg(target_arch = "x86_64")]
        let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl");
        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let (has_vnni, has_avx2) = (false, false);

        #[cfg(target_arch = "x86_64")]
        if has_vnni {
            while row + 3 < end {
                if row + 3 < m_q {
                    let r0 = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                    let r1 = &q_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                    let r2 = &q_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                    let r3 = &q_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (q_ptr as *mut f32).add(row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q && row + 3 < m_q + m_k {
                    let k_row = row - m_q;
                    let r0 = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                    let r1 = &k_matrix[(k_row + 1) * blocks_per_row..(k_row + 2) * blocks_per_row];
                    let r2 = &k_matrix[(k_row + 2) * blocks_per_row..(k_row + 3) * blocks_per_row];
                    let r3 = &k_matrix[(k_row + 3) * blocks_per_row..(k_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx512vnni(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (k_ptr as *mut f32).add(k_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                }
                break;
            }
        } else if has_avx2 {
            while row + 3 < end {
                if row + 3 < m_q {
                    let r0 = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                    let r1 = &q_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                    let r2 = &q_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                    let r3 = &q_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (q_ptr as *mut f32).add(row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                } else if row >= m_q && row + 3 < m_q + m_k {
                    let k_row = row - m_q;
                    let r0 = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                    let r1 = &k_matrix[(k_row + 1) * blocks_per_row..(k_row + 2) * blocks_per_row];
                    let r2 = &k_matrix[(k_row + 2) * blocks_per_row..(k_row + 3) * blocks_per_row];
                    let r3 = &k_matrix[(k_row + 3) * blocks_per_row..(k_row + 4) * blocks_per_row];
                    let (v0, v1, v2, v3) =
                        unsafe { dot_q4_k_q8_k_4rows_avx2(r0, r1, r2, r3, q8_k_act) };
                    unsafe {
                        let dst = (k_ptr as *mut f32).add(k_row);
                        *dst = v0;
                        *dst.add(1) = v1;
                        *dst.add(2) = v2;
                        *dst.add(3) = v3;
                    }
                    row += 4;
                    continue;
                }
                break;
            }
        }

        // Q6_K 2-row fast path and remaining rows
        while row + 1 < end && row >= m_q + m_k {
            let v_row = row - (m_q + m_k);
            let r0 = &v_matrix[v_row * blocks_per_row..(v_row + 1) * blocks_per_row];
            let r1 = &v_matrix[(v_row + 1) * blocks_per_row..(v_row + 2) * blocks_per_row];
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                let (v0, v1) = unsafe { dot_q6_k_q8_k_2rows_avx2(r0, r1, q8_k_act) };
                unsafe {
                    let dst = (v_ptr as *mut f32).add(v_row);
                    *dst = v0;
                    *dst.add(1) = v1;
                }
                row += 2;
                continue;
            }
            let v0 = dot_q6_k_q8_k_row_portable(r0, q8_k_act);
            let v1 = dot_q6_k_q8_k_row_portable(r1, q8_k_act);
            unsafe {
                let dst = (v_ptr as *mut f32).add(v_row);
                *dst = v0;
                *dst.add(1) = v1;
            }
            row += 2;
        }

        while row < end {
            if row < m_q {
                let r = &q_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q4_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q4_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((q_ptr as *mut f32).add(row)) = val;
                }
            } else if row < m_q + m_k {
                let k_row = row - m_q;
                let r = &k_matrix[k_row * blocks_per_row..(k_row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q4_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q4_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((k_ptr as *mut f32).add(k_row)) = val;
                }
            } else {
                let v_row = row - (m_q + m_k);
                let r = &v_matrix[v_row * blocks_per_row..(v_row + 1) * blocks_per_row];
                let val = if has_avx2 {
                    unsafe { dot_q6_k_q8_k_row_avx2(r, q8_k_act) }
                } else {
                    dot_q6_k_q8_k_row_portable(r, q8_k_act)
                };
                unsafe {
                    *((v_ptr as *mut f32).add(v_row)) = val;
                }
            }
            row += 1;
        }
    });
}

/// Fused Gate + Up + SwiGLU Matrix-Vector Multiplication in a single ThreadTeam dispatch pass.
#[allow(clippy::too_many_arguments, clippy::if_not_else)]
pub fn gemv_gate_up_swiglu_q4_k_prequantized(
    gate_matrix: &[crate::gguf_quants::BlockQ4_K],
    up_matrix: &[crate::gguf_quants::BlockQ4_K],
    m: usize,
    n: usize,
    q8_k_act: &[crate::int_quant::BlockQ8_K],
    gate_out: &mut [f32],
    up_out: &mut [f32],
    activated_out: &mut [f32],
    is_gemma: bool,
) {
    let blocks_per_row = n / 256;
    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let g_ptr = gate_out.as_mut_ptr() as usize;
    let u_ptr = up_out.as_mut_ptr() as usize;
    let act_ptr = activated_out.as_mut_ptr() as usize;

    team.dispatch_rows(m, |start, end| {
        let mut row = start;
        #[cfg(target_arch = "x86_64")]
        let has_vnni = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl");
        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let (has_vnni, has_avx2) = (false, false);

        #[cfg(target_arch = "x86_64")]
        if has_vnni {
            while row + 3 < end {
                let gr0 = &gate_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let gr1 = &gate_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let gr2 = &gate_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let gr3 = &gate_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (g0, g1, g2, g3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx512vnni(gr0, gr1, gr2, gr3, q8_k_act) };

                let ur0 = &up_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let ur1 = &up_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let ur2 = &up_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let ur3 = &up_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (u0, u1, u2, u3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx512vnni(ur0, ur1, ur2, ur3, q8_k_act) };

                unsafe {
                    let g_dst = (g_ptr as *mut f32).add(row);
                    let u_dst = (u_ptr as *mut f32).add(row);
                    let a_dst = (act_ptr as *mut f32).add(row);

                    *g_dst = g0;
                    *g_dst.add(1) = g1;
                    *g_dst.add(2) = g2;
                    *g_dst.add(3) = g3;
                    *u_dst = u0;
                    *u_dst.add(1) = u1;
                    *u_dst.add(2) = u2;
                    *u_dst.add(3) = u3;

                    if !is_gemma {
                        let silu0 = g0 / (1.0 + (-g0).exp());
                        let silu1 = g1 / (1.0 + (-g1).exp());
                        let silu2 = g2 / (1.0 + (-g2).exp());
                        let silu3 = g3 / (1.0 + (-g3).exp());
                        *a_dst = silu0 * u0;
                        *a_dst.add(1) = silu1 * u1;
                        *a_dst.add(2) = silu2 * u2;
                        *a_dst.add(3) = silu3 * u3;
                    } else {
                        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                        for (idx, (g, u)) in
                            [(g0, u0), (g1, u1), (g2, u2), (g3, u3)].iter().enumerate()
                        {
                            let inner = SQRT_2_OVER_PI * (g + 0.044_715 * g * g * g);
                            let gelu = 0.5 * g * (1.0 + inner.tanh());
                            *a_dst.add(idx) = gelu * u;
                        }
                    }
                }

                row += 4;
            }
        } else if has_avx2 {
            while row + 3 < end {
                let gr0 = &gate_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let gr1 = &gate_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let gr2 = &gate_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let gr3 = &gate_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (g0, g1, g2, g3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx2(gr0, gr1, gr2, gr3, q8_k_act) };

                let ur0 = &up_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
                let ur1 = &up_matrix[(row + 1) * blocks_per_row..(row + 2) * blocks_per_row];
                let ur2 = &up_matrix[(row + 2) * blocks_per_row..(row + 3) * blocks_per_row];
                let ur3 = &up_matrix[(row + 3) * blocks_per_row..(row + 4) * blocks_per_row];
                let (u0, u1, u2, u3) =
                    unsafe { dot_q4_k_q8_k_4rows_avx2(ur0, ur1, ur2, ur3, q8_k_act) };

                unsafe {
                    let g_dst = (g_ptr as *mut f32).add(row);
                    let u_dst = (u_ptr as *mut f32).add(row);
                    let a_dst = (act_ptr as *mut f32).add(row);

                    *g_dst = g0;
                    *g_dst.add(1) = g1;
                    *g_dst.add(2) = g2;
                    *g_dst.add(3) = g3;
                    *u_dst = u0;
                    *u_dst.add(1) = u1;
                    *u_dst.add(2) = u2;
                    *u_dst.add(3) = u3;

                    if !is_gemma {
                        let silu0 = g0 / (1.0 + (-g0).exp());
                        let silu1 = g1 / (1.0 + (-g1).exp());
                        let silu2 = g2 / (1.0 + (-g2).exp());
                        let silu3 = g3 / (1.0 + (-g3).exp());
                        *a_dst = silu0 * u0;
                        *a_dst.add(1) = silu1 * u1;
                        *a_dst.add(2) = silu2 * u2;
                        *a_dst.add(3) = silu3 * u3;
                    } else {
                        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                        for (idx, (g, u)) in
                            [(g0, u0), (g1, u1), (g2, u2), (g3, u3)].iter().enumerate()
                        {
                            let inner = SQRT_2_OVER_PI * (g + 0.044_715 * g * g * g);
                            let gelu = 0.5 * g * (1.0 + inner.tanh());
                            *a_dst.add(idx) = gelu * u;
                        }
                    }
                }

                row += 4;
            }
        }

        while row < end {
            let gr = &gate_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
            let ur = &up_matrix[row * blocks_per_row..(row + 1) * blocks_per_row];
            let g = if has_avx2 {
                unsafe { dot_q4_k_q8_k_row_avx2(gr, q8_k_act) }
            } else {
                dot_q4_k_q8_k_row_portable(gr, q8_k_act)
            };
            let u = if has_avx2 {
                unsafe { dot_q4_k_q8_k_row_avx2(ur, q8_k_act) }
            } else {
                dot_q4_k_q8_k_row_portable(ur, q8_k_act)
            };
            unsafe {
                *((g_ptr as *mut f32).add(row)) = g;
                *((u_ptr as *mut f32).add(row)) = u;
                if !is_gemma {
                    let silu = g / (1.0 + (-g).exp());
                    *((act_ptr as *mut f32).add(row)) = silu * u;
                } else {
                    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                    let inner = SQRT_2_OVER_PI * (g + 0.044_715 * g * g * g);
                    let gelu = 0.5 * g * (1.0 + inner.tanh());
                    *((act_ptr as *mut f32).add(row)) = gelu * u;
                }
            }
            row += 1;
        }
    });
}

/// Multithreaded Q4_K Matrix-Vector Multiplication across all CPU cores and threads.
#[allow(clippy::too_many_lines)]
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

    let mut q8_k_act = Vec::with_capacity(blocks_per_row);
    quantize_row_q8_k(vector, &mut q8_k_act);
    gemv_q4_k_prequantized(matrix, &q8_k_act, m, n, output);
}

/// Multithreaded Batched Q4_K Matrix Multiplication for Prefill.
/// Multiplies matrix W [m, n] by batch [b, n] of pre-quantized BlockQ8_K activations.
/// Weight blocks are loaded ONCE per row from RAM/cache and multiplied across all B batch tokens.
pub fn gemm_q4_k_prequantized_batch(
    matrix: &[crate::gguf_quants::BlockQ4_K],
    batch_q8: &[Vec<crate::int_quant::BlockQ8_K>],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    crate::gemm_avx2::gemm_q4_k_tiled_prefill(matrix, batch_q8, m, n, output);
}

/// Multithreaded Batched Q6_K Matrix Multiplication for Prefill.
/// Multiplies matrix W [m, n] by batch [b, n] of pre-quantized BlockQ8_K activations.
/// Weight blocks are loaded ONCE per row from RAM/cache and multiplied across all B batch tokens.
pub fn gemm_q6_k_prequantized_batch(
    matrix: &[crate::int_quant::BlockQ6_K],
    batch_q8: &[Vec<crate::int_quant::BlockQ8_K>],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    crate::gemm_avx2::gemm_q6_k_tiled_prefill(matrix, batch_q8, m, n, output);
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

/// Vectorized RMSNorm kernel with (1.0 + weight) offset scaling for Gemma 2 / Gemma 4 models.
pub fn rmsnorm_f32_offset(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
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
            output[base + j] = input[base + j] * inv_rms * (1.0 + weight[base + j]);
        }
    }
    for i in (chunks * 8)..len {
        output[i] = input[i] * inv_rms * (1.0 + weight[i]);
    }
}

/// Vectorized SwiGLU activation kernel: output[i] = silu(gate[i]) * up[i].
pub fn swiglu_f32(gate: &[f32], up: &[f32], output: &mut [f32]) {
    let len = gate.len().min(up.len()).min(output.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Target features AVX2 and FMA are detected.
            unsafe {
                swiglu_f32_avx2(&gate[..len], &up[..len], &mut output[..len]);
                return;
            }
        }
    }

    // Portable fallback
    for i in 0..len {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        output[i] = silu_g * up[i];
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

    let team = oxide_core::thread_team::global_thread_team(get_team_threads());
    let out_ptr = out.as_mut_ptr() as usize;

    team.dispatch_rows(num_heads, |start_head, end_head| {
        for head_idx in start_head..end_head {
            let head_out = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(head_idx * head_dim),
                    head_dim,
                )
            };
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
        }
    });
}

/// Computes the dot product between a PQ2_0 block's 32 packed bytes (128 quaternary weights) and f32 activations.
#[inline(always)]
/// Multithreaded PTQ1_0 (Ternary 1.58-bit) Matrix-Vector Multiplication across CPU cores.
pub fn gemv_ptq1_0(
    matrix: &[crate::ptq1_0::BlockPTQ1_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(
        n.is_multiple_of(128),
        "n must be a multiple of 128 for PTQ1_0"
    );
    let blocks_per_row = n / 128;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient PTQ1_0 blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    let compute_row = |row: usize| -> f32 {
        let row_offset = row * blocks_per_row;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let blk = &matrix[row_offset + b];
            let act_chunk: &[f32; 128] = vector[b * 128..(b + 1) * 128]
                .try_into()
                .expect("slice length 128");
            acc += blk.dot_product_128(act_chunk);
        }
        acc
    };

    if m <= 8 {
        for row in 0..m {
            output[row] = compute_row(row);
        }
    } else {
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                local_out[i] = compute_row(row);
            }
        });
    }
}

/// Multithreaded PQ2_0 (Quaternary 2-bit) Matrix-Vector Multiplication across CPU cores.
pub fn gemv_pq2_0(
    matrix: &[crate::pq2_0::BlockPQ2_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    assert!(
        n.is_multiple_of(128),
        "n must be a multiple of 128 for PQ2_0"
    );
    let blocks_per_row = n / 128;
    assert!(
        matrix.len() >= m * blocks_per_row,
        "Insufficient PQ2_0 blocks"
    );
    assert!(vector.len() >= n, "Insufficient vector length");
    assert!(output.len() >= m, "Insufficient output length");

    let compute_row = |row: usize| -> f32 {
        let row_offset = row * blocks_per_row;
        let mut acc = 0.0f32;
        for b in 0..blocks_per_row {
            let blk = &matrix[row_offset + b];
            let act_chunk: &[f32; 128] = vector[b * 128..(b + 1) * 128]
                .try_into()
                .expect("slice length 128");
            acc += dot_pq2_0(&blk.qs, act_chunk, blk.scale());
        }
        acc
    };

    if m <= 8 {
        for row in 0..m {
            output[row] = compute_row(row);
        }
    } else {
        let team = oxide_core::thread_team::global_thread_team(get_team_threads());
        let out_ptr = output.as_mut_ptr() as usize;

        team.dispatch_rows(m, |start, end| {
            let local_out = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start), end - start)
            };
            for (i, row) in (start..end).enumerate() {
                local_out[i] = compute_row(row);
            }
        });
    }
}


#[cfg(test)]
mod tests;
