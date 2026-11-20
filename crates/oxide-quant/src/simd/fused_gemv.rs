//! Fused Multi-Projection GEMV Micro-Kernels (QKV and Gate-Up-SwiGLU).

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

use super::*;
use rayon::prelude::*;

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
