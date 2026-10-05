//! Grouped GEMM routing and execution verification for Sparse Mixture-of-Experts.
//! Asserts that single-kernel batched expert dispatch equals individual serial evaluations.

use oxide_engine::moe_gemm::{ExpertRoutingGate, GroupedExpertGemm};

#[test]
fn test_grouped_moe_single_kernel_vs_serial_equivalence() {
    const NUM_EXPERTS: usize = 8;
    const TOP_K: usize = 2;
    const BATCH_TOKENS: usize = 16;
    const HIDDEN_DIM: usize = 2048;
    const INTERMEDIATE_DIM: usize = 5632;

    let routing_gate = ExpertRoutingGate::new(HIDDEN_DIM, NUM_EXPERTS, TOP_K)
        .expect("Failed to create ExpertRoutingGate");

    let mut grouped_gemm = GroupedExpertGemm::new(NUM_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM)
        .expect("Failed to initialize GroupedExpertGemm");

    // Synthetic token batch activations
    let input_tokens = vec![0.02f32; BATCH_TOKENS * HIDDEN_DIM];

    // 1. Execute routing gate to assign top-2 experts and softmax weights
    let routing_decisions = routing_gate
        .route_tokens(&input_tokens, BATCH_TOKENS)
        .expect("Routing decision computation failed");

    // 2. Dispatch via unified single-kernel Grouped GEMM
    let mut grouped_output = vec![0.0f32; BATCH_TOKENS * HIDDEN_DIM];
    unsafe {
        grouped_gemm
            .dispatch_grouped(
                input_tokens.as_ptr(),
                &routing_decisions,
                grouped_output.as_mut_ptr(),
            )
            .expect("Grouped GEMM dispatch failed");
    }

    // 3. Dispatch via slow serial fallback loop
    let mut serial_output = vec![0.0f32; BATCH_TOKENS * HIDDEN_DIM];
    unsafe {
        grouped_gemm
            .dispatch_serial_reference(
                input_tokens.as_ptr(),
                &routing_decisions,
                serial_output.as_mut_ptr(),
            )
            .expect("Serial reference dispatch failed");
    }

    // 4. Assert bitwise identical output (Chebyshev L-infinity norm <= 1e-5)
    for i in 0..(BATCH_TOKENS * HIDDEN_DIM) {
        let diff = (grouped_output[i] - serial_output[i]).abs();
        assert!(
            diff <= 1e-5,
            "Discrepancy at index {}: Grouped={:.6}, Serial={:.6}",
            i,
            grouped_output[i],
            serial_output[i]
        );
    }
}
