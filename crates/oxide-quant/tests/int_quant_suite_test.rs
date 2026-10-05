use oxide_quant::int_quant::{
    f16, BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ6_K, BlockQ8_0,
};

#[test]
fn test_f16_float_conversion() {
    let original = 3.14159f32;
    let half = f16::from_f32(original);
    let recovered = half.to_f32();
    assert!((recovered - original).abs() < 0.01);

    assert_eq!(f16::from_f32(0.0).to_f32(), 0.0);
    assert_eq!(f16::from_f32(-1.0).to_f32(), -1.0);
}

#[test]
fn test_block_q4_0_quant_dequant_dot() {
    let mut input = [0.0f32; 32];
    for (i, val) in input.iter_mut().enumerate() {
        *val = (i as f32 - 16.0) * 0.25;
    }

    let block = BlockQ4_0::quantize(&input);
    let mut output = [0.0f32; 32];
    block.dequantize(&mut output);

    for i in 0..32 {
        assert!((output[i] - input[i]).abs() < 0.35);
    }

    let dot = block.dot_product(&input);
    let expected_dot: f32 = input.iter().map(|&x| x * x).sum();
    assert!((dot - expected_dot).abs() / expected_dot < 0.15);
}

#[test]
fn test_block_q4_1_quant_dequant() {
    let mut input = [0.0f32; 32];
    for (i, val) in input.iter_mut().enumerate() {
        *val = i as f32 * 0.5 + 2.0;
    }

    let block = BlockQ4_1::quantize(&input);
    let mut output = [0.0f32; 32];
    block.dequantize(&mut output);

    for i in 0..32 {
        assert!((output[i] - input[i]).abs() < 0.6);
    }
}

#[test]
fn test_block_q5_0_quant_dequant() {
    let mut input = [0.0f32; 32];
    for (i, val) in input.iter_mut().enumerate() {
        *val = (i as f32 - 16.0) * 0.15;
    }

    let block = BlockQ5_0::quantize(&input);
    let mut output = [0.0f32; 32];
    block.dequantize(&mut output);

    for i in 0..32 {
        assert!((output[i] - input[i]).abs() < 0.15);
    }
}

#[test]
fn test_block_q8_0_quant_dequant_dot() {
    let mut input = [0.0f32; 32];
    for (i, val) in input.iter_mut().enumerate() {
        *val = (i as f32 - 16.0) * 0.5;
    }

    let block = BlockQ8_0::quantize(&input);
    let mut output = [0.0f32; 32];
    block.dequantize(&mut output);

    for i in 0..32 {
        assert!((output[i] - input[i]).abs() < 0.05); // Q8_0 has high fidelity
    }

    let dot = block.dot_product(&input);
    let expected_dot: f32 = input.iter().map(|&x| x * x).sum();
    assert!((dot - expected_dot).abs() / expected_dot < 0.02);
}

#[test]
fn test_k_quant_superblocks_q2k_q3k_q6k() {
    let mut input = [0.0f32; 256];
    for (i, val) in input.iter_mut().enumerate() {
        *val = ((i as f32) * 0.05).sin() * 2.0;
    }

    // 1. Q2_K (2-bit super-block)
    let q2k = BlockQ2_K::quantize(&input);
    let mut out_q2k = [0.0f32; 256];
    q2k.dequantize(&mut out_q2k);
    assert_eq!(q2k.qs.len(), 64);

    // 2. Q3_K (3-bit super-block)
    let q3k = BlockQ3_K::quantize(&input);
    let mut out_q3k = [0.0f32; 256];
    q3k.dequantize(&mut out_q3k);
    assert_eq!(q3k.qs.len(), 64);
    assert_eq!(q3k.hmask.len(), 32);

    // 3. Q6_K (6-bit super-block)
    let q6k = BlockQ6_K::quantize(&input);
    let mut out_q6k = [0.0f32; 256];
    q6k.dequantize(&mut out_q6k);
    assert_eq!(q6k.ql.len(), 128);
    assert_eq!(q6k.qh.len(), 64);

    // Verify Q6_K high fidelity reconstruction
    for i in 0..256 {
        assert!((out_q6k[i] - input[i]).abs() < 0.2);
    }
}
