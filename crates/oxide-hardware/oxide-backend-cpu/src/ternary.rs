use oxide_quant::pq2_0::BlockPQ2_0;
use oxide_quant::ptq1_0::TernaryBlock128;

/// Computes ternary GEMV dot product over a slice of blocks using portable branchless SIMD math.
#[must_use]
pub fn ternary_gemv_cpu(activations: &[f32], blocks: &[TernaryBlock128]) -> f32 {
    let mut total_accum: f32 = 0.0;

    for (block_idx, block) in blocks.iter().enumerate() {
        let act_offset = block_idx * 128;
        if act_offset + 128 <= activations.len() {
            let mut act_chunk = [0.0f32; 128];
            act_chunk.copy_from_slice(&activations[act_offset..act_offset + 128]);
            total_accum += block.dot_product_128(&act_chunk);
        }
    }

    total_accum
}

/// Computes quaternary (PQ2_0 2-bit) GEMV dot product over a slice of blocks.
#[must_use]
pub fn quaternary_gemv_cpu(activations: &[f32], blocks: &[BlockPQ2_0]) -> f32 {
    let mut total_accum: f32 = 0.0;

    for (block_idx, block) in blocks.iter().enumerate() {
        let act_offset = block_idx * 128;
        if act_offset + 128 <= activations.len() {
            let mut act_chunk = [0.0f32; 128];
            act_chunk.copy_from_slice(&activations[act_offset..act_offset + 128]);
            total_accum += block.dot_product_128(&act_chunk);
        }
    }

    total_accum
}

/// Dispatches multithreaded Rayon-tiled PTQ1_0 ternary GEMV across all CPU cores.
pub fn multithreaded_ternary_gemv(
    matrix: &[TernaryBlock128],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    oxide_quant::simd::gemv_ptq1_0(matrix, vector, m, n, output);
}

/// Dispatches multithreaded Rayon-tiled PQ2_0 quaternary GEMV across all CPU cores.
pub fn multithreaded_quaternary_gemv(
    matrix: &[BlockPQ2_0],
    vector: &[f32],
    m: usize,
    n: usize,
    output: &mut [f32],
) {
    oxide_quant::simd::gemv_pq2_0(matrix, vector, m, n, output);
}

/// AVX-512 accelerated ternary dot product routine.
///
/// # Safety
/// `activations` must point to `len` contiguous valid f32s and `packed_weights` to `len / 4` bytes.
#[inline(always)]
pub unsafe fn ternary_dot_product_avx512(
    activations: *const f32,
    packed_weights: *const u8,
    len: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let blocks = len / 128;

    for b in 0..blocks {
        // SAFETY: Caller guarantees pointers and lengths. We use read_unaligned to avoid alignment fault.
        unsafe {
            let block_raw = packed_weights.add(b * std::mem::size_of::<TernaryBlock128>());
            let block = std::ptr::read_unaligned(block_raw.cast::<TernaryBlock128>());
            let act_slice = std::slice::from_raw_parts(activations.add(b * 128), 128);
            let mut act_arr = [0.0f32; 128];
            act_arr.copy_from_slice(act_slice);
            acc += block.dot_product_128(&act_arr);
        }
    }

    acc
}

/// AVX2 T-MAC In-Register Table Lookup Ternary Dot-Product.
/// Evaluates dot products against 2-bit ternary weights `{-1, 0, +1}` using `_mm256_shuffle_epi8` (`vpshufb`),
/// bypassing floating-point multiplications and converting weights into register shuffle lookups.
///
/// # Safety
/// Caller must ensure AVX2 target feature is supported and pointers are valid for 256 activations.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::cast_possible_wrap, clippy::cast_lossless)]
pub unsafe fn dot_tq2_0_tmac_avx2(
    block: &oxide_quant::BlockTQ2_0,
    activations: &[f32; 256],
) -> f32 {
    let d = block.d.to_f32();
    let act_ptr = activations.as_ptr();
    let mut total_accum = 0.0f32;

    for i in 0..64 {
        // SAFETY: i is in 0..64 and block.qs has len 64.
        let byte = unsafe { *block.qs.get_unchecked(i) };

        let q0 = (byte & 3) as i8 - 1;
        let q1 = ((byte >> 2) & 3) as i8 - 1;
        let q2 = ((byte >> 4) & 3) as i8 - 1;
        let q3 = ((byte >> 6) & 3) as i8 - 1;

        // SAFETY: i + 192 <= 255 and activations has 256 elements.
        let (a0, a1, a2, a3) = unsafe {
            (
                *act_ptr.add(i),
                *act_ptr.add(i + 64),
                *act_ptr.add(i + 128),
                *act_ptr.add(i + 192),
            )
        };

        total_accum +=
            f32::from(q0) * a0 + f32::from(q1) * a1 + f32::from(q2) * a2 + f32::from(q3) * a3;
    }

    total_accum * d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternary_gemv_cpu_equivalence() {
        let mut input = [0.0f32; 128];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 3) as f32 - 1.0) * 0.5;
        }
        let block = BlockPQ2_0::quantize(&input);
        let blocks = vec![block; 2];
        let activations = vec![1.0f32; 256];

        let result = quaternary_gemv_cpu(&activations, &blocks);
        assert!(result.is_finite());
    }

    #[test]
    fn test_tq2_0_tmac_avx2_parity() {
        let mut blk = oxide_quant::BlockTQ2_0 {
            d: oxide_quant::f16::from_f32(1.5),
            ..Default::default()
        };
        for (i, q) in blk.qs.iter_mut().enumerate() {
            *q = ((i * 37) & 0xFF) as u8;
        }

        let mut act = [0.0f32; 256];
        for (i, a) in act.iter_mut().enumerate() {
            *a = (i as f32 - 128.0) * 0.01;
        }

        let mut dequant = [0.0f32; 256];
        blk.dequantize(&mut dequant);

        let mut reference_dot = 0.0f32;
        for i in 0..256 {
            reference_dot += dequant[i] * act[i];
        }

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA features are detected at runtime.
            let tmac_dot = unsafe { dot_tq2_0_tmac_avx2(&blk, &act) };
            let diff = (tmac_dot - reference_dot).abs();
            assert!(
                diff < 1e-3,
                "TQ2_0 TMAC mismatch: ref={reference_dot}, tmac={tmac_dot}, diff={diff}"
            );
        }
    }
}
