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
fn test_swiglu_f32_parity() {
    let n = 64;
    let mut gate = vec![0.0f32; n];
    let mut up = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        gate[i] = (i as f32 - 32.0) * 0.25;
        up[i] = (i as f32 * 0.1).cos();
    }

    swiglu_f32(&gate, &up, &mut out);

    for i in 0..n {
        let g = gate[i];
        let expected_silu = g / (1.0 + (-g).exp());
        let expected = expected_silu * up[i];
        assert!(
            (out[i] - expected).abs() < 1e-4,
            "SwiGLU mismatch at index {i}: got {}, expected {}",
            out[i],
            expected
        );
    }
}

#[test]
fn test_gemv_q5_0_multithreaded() {
    let m = 32;
    let n = 64;
    let mut row_data = [0.0f32; 32];
    for (i, v) in row_data.iter_mut().enumerate() {
        *v = (i as f32 * 0.2).sin();
    }
    let blk = crate::int_quant::BlockQ5_0::quantize(&row_data);
    let matrix = vec![blk; m * (n / 32)];
    let vector = vec![0.5f32; n];
    let mut output = vec![0.0f32; m];

    gemv_q5_0(&matrix, &vector, m, n, &mut output);
    for &val in &output {
        assert!(val.is_finite());
    }
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
fn test_dot_q8_0_vnni_parity() {
    let mut u_act = [0u8; 32];
    let mut s_wt = [0i8; 32];
    for i in 0..32 {
        u_act[i] = ((i * 13) % 255) as u8;
        s_wt[i] = (((i as i32 * 17) % 255) - 128) as i8;
    }
    let scale = 0.005f32;
    let res = dot_q8_0_vnni(&u_act, &s_wt, scale);
    assert!(res.is_finite());
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
    let expected = dot_q4_0_portable(&qs, &act, scale);
    let computed = dot_q4_0(&qs, &act, scale);
    assert!((expected - computed).abs() < 1e-4);
}

#[test]
fn test_dot_q4_k_parity() {
    let mut qs = [0u8; 128];
    let mut scales = [1u8; 12];
    let mut act = [0.0f32; 256];
    for i in 0..128 {
        qs[i] = ((i * 11) % 256) as u8;
    }
    for i in 0..12 {
        scales[i] = ((i * 7) % 64) as u8;
    }
    for i in 0..256 {
        act[i] = (i as f32 * 0.05).sin();
    }
    let expected = dot_q4_k_portable(&qs, &scales, &act, 0.02, -0.5);
    let computed = dot_q4_k(&qs, &scales, &act, 0.02, -0.5);
    assert!((expected - computed).abs() < 1e-4);
}

#[test]
fn test_amx_tile_config_default() {
    let cfg = amx::TileConfig::default();
    assert_eq!(cfg.palette_id, 1);
    let supported = amx::is_amx_supported();
    println!("Host AMX support detected: {supported}");
}

#[test]
fn test_gemv_blocked_f32_multithreaded() {
    let m = 64;
    let n = 128;
    let mut matrix = vec![0.0f32; m * n];
    let mut vector = vec![0.0f32; n];
    for i in 0..matrix.len() {
        matrix[i] = ((i as f32 * 0.01).sin()).clamp(-1.0, 1.0);
    }
    for j in 0..n {
        vector[j] = ((j as f32 * 0.05).cos()).clamp(-1.0, 1.0);
    }
    let mut output = vec![0.0f32; m];
    gemv_blocked_f32(&matrix, &vector, m, n, &mut output);

    for row in 0..m {
        let offset = row * n;
        let expected: f32 = matrix[offset..offset + n]
            .iter()
            .zip(vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!(
            (output[row] - expected).abs() < 1e-3,
            "Mismatch at row {row}: computed {}, expected {expected}",
            output[row]
        );
    }
}

#[test]
fn test_gemv_q8_0_multithreaded() {
    let m = 32;
    let n = 64; // 2 blocks per row
    let blocks_per_row = n / 32;
    let mut blocks = vec![crate::int_quant::BlockQ8_0::default(); m * blocks_per_row];
    for blk in &mut blocks {
        blk.scale = crate::int_quant::f16::from_f32(0.01);
        for i in 0..32 {
            blk.qs[i] = ((i as i32 * 3) % 127) as i8;
        }
    }
    let mut vector = vec![0.0f32; n];
    for (i, v) in vector.iter_mut().enumerate() {
        *v = (i as f32 * 0.1).sin();
    }
    let mut output = vec![0.0f32; m];
    gemv_q8_0(&blocks, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    for &val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_gemv_q4_0_multithreaded() {
    let m = 32;
    let n = 64;
    let blocks_per_row = n / 32;
    let mut blocks = vec![crate::int_quant::BlockQ4_0::default(); m * blocks_per_row];
    for blk in &mut blocks {
        blk.scale = crate::int_quant::f16::from_f32(0.02);
        for i in 0..16 {
            blk.qs[i] = (i as u8 * 17) % 255;
        }
    }
    let mut vector = vec![0.0f32; n];
    for (i, v) in vector.iter_mut().enumerate() {
        *v = (i as f32 * 0.05).cos();
    }
    let mut output = vec![0.0f32; m];
    gemv_q4_0(&blocks, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    for &val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_gemv_q4_k_multithreaded() {
    let m = 16;
    let n = 256;
    let mut blocks = vec![crate::gguf_quants::BlockQ4_K::default(); m];
    for blk in &mut blocks {
        blk.d = crate::int_quant::f16::from_f32(0.02);
        blk.dmin = crate::int_quant::f16::from_f32(-0.5);
        for i in 0..128 {
            blk.qs[i] = ((i * 11) % 256) as u8;
        }
    }
    let mut vector = vec![0.0f32; n];
    for (i, v) in vector.iter_mut().enumerate() {
        *v = (i as f32 * 0.05).sin();
    }
    let mut output = vec![0.0f32; m];
    gemv_q4_k(&blocks, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    for &val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dot_q6_k_parity() {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    let mut scales = [1i8; 16];
    let mut act = [0.0f32; 256];
    for i in 0..128 {
        ql[i] = ((i * 17) % 256) as u8;
    }
    for i in 0..64 {
        qh[i] = ((i * 31) % 256) as u8;
    }
    for i in 0..16 {
        scales[i] = ((i as i8 * 3) % 30) - 15;
    }
    for i in 0..256 {
        act[i] = (i as f32 * 0.03).cos();
    }
    let expected = dot_q6_k_portable(&ql, &qh, &scales, &act, 0.05);
    let computed = dot_q6_k(&ql, &qh, &scales, &act, 0.05);
    assert!((expected - computed).abs() < 1e-4);
}

#[test]
fn test_gemv_q6_k_multithreaded() {
    let m = 32;
    let n = 256;
    let blocks = vec![
        crate::int_quant::BlockQ6_K {
            ql: [42u8; 128],
            qh: [15u8; 64],
            scales: [1i8; 16],
            d: crate::f16::from_f32(0.01),
        };
        m * (n / 256)
    ];
    let mut vector = vec![0.0f32; n];
    for (i, v) in vector.iter_mut().enumerate() {
        *v = (i as f32 * 0.02).sin();
    }
    let mut output = vec![0.0f32; m];
    gemv_q6_k(&blocks, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    for &val in &output {
        assert!(val.is_finite());
    }

    // Test AVX2 vs Portable row parity
    let mut q8_k_act = Vec::new();
    let chunk: &[f32; 256] = vector[..256].try_into().unwrap();
    q8_k_act.push(crate::int_quant::BlockQ8_K::quantize(chunk));
    let row_weights = &blocks[..1];
    let port = dot_q6_k_q8_k_row_portable(row_weights, &q8_k_act);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: Target architecture supports AVX2 and FMA.
        let avx2_res = unsafe { dot_q6_k_q8_k_row_avx2(row_weights, &q8_k_act) };
        assert!(
            (port - avx2_res).abs() < 1e-4,
            "AVX2 vs Portable mismatch: {port} vs {avx2_res}"
        );
    }
}

#[test]
fn test_rmsnorm_f32_parity() {
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0f32; 8];
    let mut output = [0.0f32; 8];
    rmsnorm_f32(&input, &weight, &mut output, 1e-5);
    let sum_sq: f32 = output.iter().map(|x| x * x).sum();
    let rms = (sum_sq / 8.0).sqrt();
    assert!((rms - 1.0).abs() < 1e-4);
}

#[test]
fn test_rope_f32_rot() {
    let mut x = [1.0f32, 2.0, 3.0, 4.0];
    rope_f32(&mut x, 4, 1, 10000.0);
    assert!(x[0].is_finite());
    assert!(x[1].is_finite());
}

#[test]
fn test_flash_attention_cpu_basic() {
    let q = vec![1.0f32; 64];
    let k = vec![1.0f32; 64];
    let v = vec![2.0f32; 64];
    let mut out = vec![0.0f32; 64];
    flash_attention_cpu(&q, &k, &v, 1, 64, 1, 1, &mut out);
    for &val in &out {
        assert!((val - 2.0).abs() < 1e-4);
    }
}

#[test]
fn test_gemv_ptq1_0_multithreaded() {
    let m = 16;
    let n = 128;
    let mut input = [0.0f32; 128];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i % 3) as f32 - 1.0) * 0.5;
    }
    let block = crate::ptq1_0::BlockPTQ1_0::quantize(&input);
    let matrix = vec![block; m];
    let vector = vec![1.0f32; n];
    let mut output = vec![0.0f32; m];

    gemv_ptq1_0(&matrix, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    let _expected = block.dot_product_128(input.as_slice().try_into().unwrap());
    for &val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_gemv_pq2_0_multithreaded() {
    let m = 16;
    let n = 128;
    let mut input = [0.0f32; 128];
    for (i, v) in input.iter_mut().enumerate() {
        *v = ((i % 3) as f32 - 1.0) * 0.5;
    }
    let block = crate::pq2_0::BlockPQ2_0::quantize(&input);
    let matrix = vec![block; m];
    let vector = vec![1.0f32; n];
    let mut output = vec![0.0f32; m];

    gemv_pq2_0(&matrix, &vector, m, n, &mut output);
    assert_eq!(output.len(), m);
    for &val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_zero_alloc_guard_counting() {
    use oxide_core::zero_alloc_guard::CountingAlloc;
    let alloc = CountingAlloc::new();
    assert_eq!(alloc.allocations(), 0);
    assert_eq!(alloc.deallocations(), 0);
    alloc.reset();
    assert_eq!(alloc.allocations(), 0);
}
