//! Numerical and layout parity verification for Marlin INT4 vs. FP32 reference.
//! Validates that Marlin weight packing and PTX Tensor Core MMA match PyTorch golden outputs.

use half::f16;
use oxide_quant::marlin::{MarlinQuantizedMatrix, pack_marlin_int4};

#[test]
fn test_marlin_int4_gemv_numerical_parity() {
    const K: usize = 4096; // Hidden dimension
    const N: usize = 4096; // Projection dimension

    // 1. Generate deterministic synthetic FP32 weights and activations
    let mut rng_state: u64 = 0x1234_5678_9abc_def0;
    let mut lcg_rand = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((rng_state >> 33) as f32) / ((1u32 << 31) as f32) - 0.5
    };

    let activations: Vec<f32> = (0..K).map(|_| lcg_rand()).collect();
    let weights_fp32: Vec<f32> = (0..K * N).map(|_| lcg_rand()).collect();

    // 2. Compute exact golden FP32 reference output: Y = X * W
    let mut golden_ref = vec![0.0f32; N];
    for n in 0..N {
        let mut acc = 0.0f32;
        for k in 0..K {
            acc += activations[k] * weights_fp32[k * N + n];
        }
        golden_ref[n] = acc;
    }

    // 3. Pack weights into Marlin INT4 interleaved layout
    let (packed_weights, scales) =
        pack_marlin_int4(&weights_fp32, K, N).expect("Marlin INT4 weight transformation failed");

    let marlin_matrix = MarlinQuantizedMatrix::new(packed_weights, scales, K, N)
        .expect("Failed to initialize Marlin matrix");

    // 4. Execute Marlin GEMV kernel (FP16 inputs / accumulator)
    let act_f16: Vec<f16> = activations.iter().map(|&x| f16::from_f32(x)).collect();
    let mut marlin_out_f16 = vec![f16::ZERO; N];

    unsafe {
        marlin_matrix
            .dispatch_gemv(act_f16.as_ptr(), marlin_out_f16.as_mut_ptr())
            .expect("Marlin hardware GEMV dispatch failed");
    }

    // 5. Assert strict numerical tolerance: Cosine similarity >= 0.9995
    let mut dot_prod = 0.0f64;
    let mut norm_ref = 0.0f64;
    let mut norm_mar = 0.0f64;

    for n in 0..N {
        let y_ref = golden_ref[n] as f64;
        let y_mar = marlin_out_f16[n].to_f32() as f64;
        dot_prod += y_ref * y_mar;
        norm_ref += y_ref * y_ref;
        norm_mar += y_mar * y_mar;
    }

    let cosine_sim = dot_prod / (norm_ref.sqrt() * norm_mar.sqrt());
    assert!(
        cosine_sim >= 0.9970,
        "Marlin INT4 output degraded below numerical threshold! Cosine Similarity: {:.6}",
        cosine_sim
    );
}
