use oxide_quant::ptq1_0::{TernaryBlock128, f16_to_f32};

#[test]
fn test_f16_conversion() {
    // 1.0 in FP16 = 0x3C00
    assert!((f16_to_f32(0x3C00) - 1.0f32).abs() < 1e-6);
    // 0.0 in FP16 = 0x0000
    assert_eq!(f16_to_f32(0x0000), 0.0f32);
    // -1.0 in FP16 = 0xBC00
    assert!((f16_to_f32(0xBC00) - (-1.0f32)).abs() < 1e-6);
}

#[test]
fn test_ternary_block_unpack_and_dot_product() {
    // Construct a block with scale = 1.0 (0x3C00)
    // and packed_weights: first byte is 0b10_01_00_01 (weights: +1, 0, +1, -1)
    let mut packed = [0u8; 32];
    packed[0] = 0b10_01_00_01; // weight 0 = +1, weight 1 = 0, weight 2 = +1, weight 3 = -1

    let block = TernaryBlock128 {
        scale_fp16: 0x3C00,
        packed_weights: packed,
    };

    assert_eq!(block.unpack_sign(0), 1.0);
    assert_eq!(block.unpack_sign(1), 0.0);
    assert_eq!(block.unpack_sign(2), 1.0);
    assert_eq!(block.unpack_sign(3), -1.0);

    let mut activations = [0.0f32; 128];
    activations[0] = 2.0;
    activations[1] = 5.0;
    activations[2] = 3.0;
    activations[3] = 4.0;

    // Expected dot product: (2.0 * 1.0) + (5.0 * 0.0) + (3.0 * 1.0) + (4.0 * -1.0) = 2 + 0 + 3 - 4 = 1.0
    let dot = block.dot_product_128(&activations);
    assert!((dot - 1.0).abs() < 1e-6);
}
