use oxide_quant::ptq1_0::TernaryBlock128;

/// Computes ternary GEMV dot product over a slice of blocks using portable branchless SIMD math.
#[must_use]
pub fn ternary_gemv_cpu(
    activations: &[f32],
    blocks: &[TernaryBlock128],
) -> f32 {
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

/// AVX-512 accelerated ternary dot product stub.
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
