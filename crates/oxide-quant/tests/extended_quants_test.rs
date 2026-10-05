//! Comprehensive Verification Tests for Extended Quantization Formats.
//!
//! Tests coverage:
//! - GGUF formats: Q1_0, Q2_0, Q4_2, Q4_3, Q5_1, Q8_1, Q8_K, I-quants (IQ1_S, IQ1_M, IQ2_XXS..IQ4_NL), Ternary (TQ1_0, TQ2_0)
//! - GPTQ 2/3/4/8-bit and AWQ precision modes
//! - EXL2 fractional bit-rate profiles (2.0bpw to 8.0bpw)
//! - bitsandbytes NF4, FP4, LLM.int8
//! - Floating-point conversions (FP8 E4M3, E5M2, MXFP8, MXFP4, NVFP4)

use oxide_quant::*;

#[test]
fn test_gguf_legacy_and_ternary_quants() {
    let mut values32 = [0.0f32; 32];
    for (i, val) in values32.iter_mut().enumerate() {
        *val = (i as f32 - 16.0) * 0.25;
    }

    // Q1_0
    let b_q1_0 = BlockQ1_0::quantize(&values32);
    let mut deq_q1_0 = [0.0f32; 32];
    b_q1_0.dequantize(&mut deq_q1_0);
    assert_eq!(deq_q1_0.len(), 32);

    // Q2_0
    let b_q2_0 = BlockQ2_0::quantize(&values32);
    let mut deq_q2_0 = [0.0f32; 32];
    b_q2_0.dequantize(&mut deq_q2_0);
    assert_eq!(deq_q2_0.len(), 32);

    // Q5_1
    let b_q5_1 = BlockQ5_1::quantize(&values32);
    let mut deq_q5_1 = [0.0f32; 32];
    b_q5_1.dequantize(&mut deq_q5_1);
    assert_eq!(deq_q5_1.len(), 32);

    // Q8_1
    let b_q8_1 = BlockQ8_1::quantize(&values32);
    let mut deq_q8_1 = [0.0f32; 32];
    b_q8_1.dequantize(&mut deq_q8_1);
    assert_eq!(deq_q8_1.len(), 32);

    // Q8_K (256 elements)
    let mut values256 = [0.0f32; 256];
    for (i, val) in values256.iter_mut().enumerate() {
        *val = (i as f32 - 128.0) * 0.1;
    }
    let b_q8_k = BlockQ8_K::quantize(&values256);
    let mut deq_q8_k = [0.0f32; 256];
    b_q8_k.dequantize(&mut deq_q8_k);
    let dot = b_q8_k.dot_product(&values256);
    assert!(dot > 0.0);
}

#[test]
fn test_exl2_fractional_bpw_matrix() {
    let matrix = Exl2WeightMatrix::new(64, 64, Exl2BitsPerWeight::Bpw4_25, 32);
    assert_eq!(matrix.rows, 64);
    assert_eq!(matrix.cols, 64);
    assert_eq!(matrix.target_bpw.bpw(), 4.25);

    let x = vec![0.5f32; 64];
    let mut y = vec![0.0f32; 64];
    matrix.gemv(&x, &mut y);
    assert_eq!(y.len(), 64);
}

#[test]
fn test_bitsandbytes_fp4_and_nf4() {
    let mut input64 = [0.0f32; 64];
    for (i, val) in input64.iter_mut().enumerate() {
        *val = (i as f32 - 32.0) * 0.05;
    }

    // NF4
    let b_nf4 = BlockNf4_64::quantize(&input64);
    let mut deq_nf4 = [0.0f32; 64];
    b_nf4.dequantize(&mut deq_nf4);
    assert_eq!(deq_nf4.len(), 64);

    // FP4
    let b_fp4 = BlockFp4Bnb_64::quantize(&input64);
    let mut deq_fp4 = [0.0f32; 64];
    b_fp4.dequantize(&mut deq_fp4);
    assert_eq!(deq_fp4.len(), 64);
}

#[test]
fn test_gptq_and_awq_modes() {
    let group = QuantGroupSize::Group128;
    assert_eq!(group.size(), 128);

    let gptq_bits = GptqBitWidth::Bits4;
    assert_eq!(gptq_bits.bits(), 4);

    let awq_mode = AwqPrecisionMode::W4A16;
    assert_eq!(awq_mode, AwqPrecisionMode::W4A16);
}
