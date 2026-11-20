//! Reference Scalar and Portable SIMD Fallbacks.

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

#[inline(always)]
pub fn dot_f32_portable(a: &[f32], b: &[f32]) -> f32 {
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



pub fn dot_q8_0_portable(qs: &[i8; 32], act: &[f32; 32], scale: f32) -> f32 {
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



pub fn dot_q5_0_portable(qs: &[u8; 16], qh: &[u8; 4], act: &[f32; 32], scale: f32) -> f32 {
    let mut acc = 0.0f32;
    let qh_u32 = u32::from_le_bytes(*qh);
    for i in 0..16 {
        let byte = qs[i];
        let h0 = (qh_u32 >> i) & 1;
        let h1 = (qh_u32 >> (i + 16)) & 1;

        let q0 = ((byte & 0x0F) | ((h0 as u8) << 4)) as i8 - 16;
        let q1 = (((byte >> 4) & 0x0F) | ((h1 as u8) << 4)) as i8 - 16;

        acc += (q0 as f32) * act[i] + (q1 as f32) * act[i + 16];
    }
    acc * scale
}



pub fn dot_q4_0_portable(qs: &[u8; 16], act: &[f32; 32], scale: f32) -> f32 {
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



pub fn dot_q4_k_portable(
    qs: &[u8; 128],
    scales: &[u8; 12],
    act: &[f32; 256],
    d: f32,
    dmin: f32,
) -> f32 {
    let mut sum = 0.0f32;
    let mut is = 0;
    for chunk in 0..4 {
        let (sc1, m1) = crate::gguf_quants::get_scale_min_k4(is, scales);
        let d1 = d * sc1;
        let min1 = dmin * m1;

        let (sc2, m2) = crate::gguf_quants::get_scale_min_k4(is + 1, scales);
        let d2 = d * sc2;
        let min2 = dmin * m2;

        let q_base = chunk * 32;
        let act_base = chunk * 64;

        for l in 0..32 {
            let byte = qs[q_base + l];
            let q0 = (byte & 0x0F) as f32;
            let q1 = (byte >> 4) as f32;

            let w0 = d1 * q0 - min1;
            let w1 = d2 * q1 - min2;

            sum += w0 * act[act_base + l] + w1 * act[act_base + 32 + l];
        }
        is += 2;
    }
    sum
}

#[cfg(target_arch = "x86_64")]


pub fn dot_q6_k_portable(
    ql: &[u8; 128],
    qh: &[u8; 64],
    scales: &[i8; 16],
    act: &[f32; 256],
    d: f32,
) -> f32 {
    let mut sum = 0.0f32;
    for half in 0..2 {
        let ql_offset = half * 64;
        let qh_offset = half * 32;
        let sc_offset = half * 8;
        let y_offset = half * 128;

        for l in 0..32 {
            let is = l / 16;
            let qh_byte = qh[qh_offset + l];
            let ql0 = ql[ql_offset + l];
            let ql32 = ql[ql_offset + l + 32];

            let q1 = ((ql0 & 0x0F) | ((qh_byte & 3) << 4)) as i8 - 32;
            let q2 = ((ql32 & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8 - 32;

            let sc0 = scales[sc_offset + is] as f32;
            let sc2 = scales[sc_offset + is + 2] as f32;
            let sc4 = scales[sc_offset + is + 4] as f32;
            let sc6 = scales[sc_offset + is + 6] as f32;

            sum += sc0 * (q1 as f32) * act[y_offset + l]
                + sc2 * (q2 as f32) * act[y_offset + l + 32]
                + sc4 * (q3 as f32) * act[y_offset + l + 64]
                + sc6 * (q4 as f32) * act[y_offset + l + 96];
        }
    }
    sum * d
}



pub fn dot_q6_k_q8_k_row_portable(
    row_weights: &[crate::int_quant::BlockQ6_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> f32 {
    let mut sum = 0.0f32;
    for (blk, act_blk) in row_weights.iter().zip(act_q8_k.iter()) {
        let d = act_blk.d * blk.d.to_f32();
        let mut block_sum = 0i32;
        for half in 0..2 {
            let ql_offset = half * 64;
            let qh_offset = half * 32;
            let sc_offset = half * 8;
            let y_offset = half * 128;

            for l in 0..32 {
                let is = l / 16;
                let qh_byte = blk.qh[qh_offset + l];
                let ql0 = blk.ql[ql_offset + l];
                let ql32 = blk.ql[ql_offset + l + 32];

                let q1 = ((ql0 & 0x0F) | ((qh_byte & 3) << 4)) as i32 - 32;
                let q2 = ((ql32 & 0x0F) | (((qh_byte >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i32 - 32;

                let sc0 = blk.scales[sc_offset + is] as i32;
                let sc2 = blk.scales[sc_offset + is + 2] as i32;
                let sc4 = blk.scales[sc_offset + is + 4] as i32;
                let sc6 = blk.scales[sc_offset + is + 6] as i32;

                block_sum += sc0 * q1 * (act_blk.qs[y_offset + l] as i32)
                    + sc2 * q2 * (act_blk.qs[y_offset + l + 32] as i32)
                    + sc4 * q3 * (act_blk.qs[y_offset + l + 64] as i32)
                    + sc6 * q4 * (act_blk.qs[y_offset + l + 96] as i32);
            }
        }
        sum += d * (block_sum as f32);
    }
    sum
}


pub fn dot_q4_k_q8_k_row_portable(
    row_weights: &[crate::gguf_quants::BlockQ4_K],
    act_q8_k: &[crate::int_quant::BlockQ8_K],
) -> f32 {
    let mut sum = 0.0f32;
    for (blk, act_blk) in row_weights.iter().zip(act_q8_k.iter()) {
        let d = act_blk.d * blk.d.to_f32();
        let dmin = -act_blk.d * blk.dmin.to_f32();
        let mut is = 0;
        let mut block_sum = 0i32;
        let mut min_sum = 0i32;

        for chunk in 0..4 {
            let (sc1, m1) = crate::gguf_quants::get_scale_min_k4(is, &blk.scales);
            let (sc2, m2) = crate::gguf_quants::get_scale_min_k4(is + 1, &blk.scales);
            let q_base = chunk * 32;
            let act_base = chunk * 64;

            let mut q_sum1 = 0i32;
            let mut q_sum2 = 0i32;
            for l in 0..32 {
                let byte = blk.qs[q_base + l];
                let q1 = (byte & 0x0F) as i32;
                let q2 = (byte >> 4) as i32;
                let a1 = act_blk.qs[act_base + l] as i32;
                let a2 = act_blk.qs[act_base + 32 + l] as i32;
                q_sum1 += q1 * a1;
                q_sum2 += q2 * a2;
            }
            block_sum += (sc1 as i32) * q_sum1 + (sc2 as i32) * q_sum2;

            let sum_a1 = (act_blk.bsums[chunk * 4] as i32) + (act_blk.bsums[chunk * 4 + 1] as i32);
            let sum_a2 =
                (act_blk.bsums[chunk * 4 + 2] as i32) + (act_blk.bsums[chunk * 4 + 3] as i32);
            min_sum += (m1 as i32) * sum_a1 + (m2 as i32) * sum_a2;

            is += 2;
        }
        sum += d * (block_sum as f32) + dmin * (min_sum as f32);
    }
    sum
}


pub fn dot_pq2_0(qs: &[u8; 32], act: &[f32; 128], scale: f32) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;

    for (byte_idx, &b) in qs.iter().enumerate() {
        let base = byte_idx * 4;
        let c0 = ((b & 0x03) as i32) - 1;
        let c1 = (((b >> 2) & 0x03) as i32) - 1;
        let c2 = (((b >> 4) & 0x03) as i32) - 1;
        let c3 = (((b >> 6) & 0x03) as i32) - 1;

        acc0 += (c0 as f32) * act[base];
        acc1 += (c1 as f32) * act[base + 1];
        acc2 += (c2 as f32) * act[base + 2];
        acc3 += (c3 as f32) * act[base + 3];
    }

    (acc0 + acc1 + acc2 + acc3) * scale
}

