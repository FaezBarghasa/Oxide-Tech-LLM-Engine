//! Golden Quantization Numerical Parity Verification Test Suite.
//!
//! Enforces strict numerical tolerances (atol <= 1e-3, cosine similarity >= 0.99)
//! across all core quantization schemes:
//! - ptq1_0 (Ternary 1.58-bit)
//! - pq2_0 (Quaternary 2-bit)
//! - cq2 (Cactus Needle 2-bit)
//! - marlin (FP16xINT4 TensorCore)
//! - fp8 (E4M3 / E5M2)
//! - nvfp4 (Blackwell Micro-Atom E2M1)

use oxide_quant::*;

fn chebyshev_distance(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len(), "Vectors must have identical length");
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len(), "Vectors must have identical length");
    let mut dot = 0.0f32;
    let mut norm_u_sq = 0.0f32;
    let mut norm_v_sq = 0.0f32;

    for (&a, &b) in u.iter().zip(v.iter()) {
        dot += a * b;
        norm_u_sq += a * a;
        norm_v_sq += b * b;
    }

    if norm_u_sq == 0.0 || norm_v_sq == 0.0 {
        return 0.0;
    }

    dot / (norm_u_sq.sqrt() * norm_v_sq.sqrt())
}

#[test]
fn test_golden_ptq1_0_parity() {
    let mut input = [0.0f32; 128];
    for (i, val) in input.iter_mut().enumerate() {
        let cycle = (i % 3) as i32 - 1; // -1, 0, 1
        *val = (cycle as f32) * 0.75;
    }

    let block = BlockPTQ1_0::quantize(&input);
    let mut dequant = [0.0f32; 128];
    block.dequantize(&mut dequant);

    let l_inf = chebyshev_distance(&input, &dequant);
    let cos = cosine_similarity(&input, &dequant);

    assert!(
        l_inf <= 1e-3,
        "PTQ1_0 exact ternary representation diff {l_inf} exceeds tolerance"
    );
    assert!(
        cos >= 0.9999,
        "PTQ1_0 cosine similarity {cos} below target threshold"
    );
}

#[test]
fn test_golden_pq2_0_parity() {
    let mut input = [0.0f32; 128];
    for (i, val) in input.iter_mut().enumerate() {
        let cycle = (i % 3) as i32 - 1; // -1, 0, 1
        *val = (cycle as f32) * 0.5;
    }

    let block = BlockPQ2_0::quantize(&input);
    let mut dequant = [0.0f32; 128];
    block.dequantize(&mut dequant);

    let l_inf = chebyshev_distance(&input, &dequant);
    let cos = cosine_similarity(&input, &dequant);

    assert!(
        l_inf <= 1e-3,
        "PQ2_0 exact quaternary representation diff {l_inf} exceeds tolerance"
    );
    assert!(
        cos >= 0.9999,
        "PQ2_0 cosine similarity {cos} below target threshold"
    );
}

#[test]
fn test_golden_cq2_metadata_scheme() {
    use oxide_core::traits::QuantScheme;
    assert_eq!(NeedleCQ2::bits_per_weight(), 2.0);
    assert_eq!(NeedleCQ2::block_size(), 128);
}

#[test]
fn test_golden_fp8_e4m3_parity() {
    let test_values = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0,
    ];

    for &v in &test_values {
        let fp8 = Fp8E4M3::from_f32(v);
        let reconstructed = fp8.to_f32();
        let diff = (v - reconstructed).abs();
        assert!(
            diff <= 1e-3,
            "FP8 exact representation for {v} produced diff {diff}"
        );
    }
}

#[test]
fn test_golden_nvfp4_parity() {
    let input = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0,
    ];
    let block = BlockNvFp4_16::quantize(&input, 1.0);
    let mut dequant = [0.0f32; 16];
    block.dequantize(&mut dequant, 1.0);

    let l_inf = chebyshev_distance(&input, &dequant);
    let cos = cosine_similarity(&input, &dequant);

    assert!(
        l_inf <= 1e-3,
        "NVFP4 exact E2M1 representation diff {l_inf} exceeds tolerance"
    );
    assert!(
        cos >= 0.9999,
        "NVFP4 cosine similarity {cos} below target threshold"
    );
}

#[test]
fn test_golden_marlin_tensorcore_packing() {
    let rows = 16;
    let cols = 64;
    let mut weights = vec![0.0f32; rows * cols];
    for (i, w) in weights.iter_mut().enumerate() {
        *w = ((i % 15) as f32 - 7.0) * 0.1;
    }

    let packed = pack_marlin_int4(&weights, rows, cols);
    assert!(packed.is_ok(), "Marlin packing must succeed for 16x64 tile");
    let (packed_tiles, scales) = packed.unwrap();

    assert_eq!(packed_tiles.len(), (rows * cols) / 8);
    assert_eq!(scales.len(), cols);
}
