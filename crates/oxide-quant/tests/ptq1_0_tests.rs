use oxide_quant::pq2_0::BlockPQ2_0;
use oxide_quant::ptq1_0::{BlockPTQ1_0, f16_to_f32};

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
fn test_ptq1_0_quant_and_dot_product() {
    let mut input = [0.0f32; 128];
    input[0] = 1.0;
    input[1] = 0.0;
    input[2] = -1.0;
    input[3] = 1.0;

    let block = BlockPTQ1_0::quantize(&input);
    assert_eq!(block.unpack_sign(0), 1.0);
    assert_eq!(block.unpack_sign(1), 0.0);
    assert_eq!(block.unpack_sign(2), -1.0);
    assert_eq!(block.unpack_sign(3), 1.0);

    let mut activations = [0.0f32; 128];
    activations[0] = 2.0;
    activations[1] = 5.0;
    activations[2] = 3.0;
    activations[3] = 4.0;

    // Expected dot product: (2.0 * 1.0) + (5.0 * 0.0) + (3.0 * -1.0) + (4.0 * 1.0) = 2 + 0 - 3 + 4 = 3.0
    let dot = block.dot_product_128(&activations);
    assert!((dot - 3.0).abs() < 1e-4);
}

#[test]
fn test_pq2_0_quant_and_dot_product() {
    let mut input = [0.0f32; 128];
    input[0] = -1.0;
    input[1] = 0.0;
    input[2] = 1.0;

    let block = BlockPQ2_0::quantize(&input);
    assert_eq!(block.coef(0), -1);
    assert_eq!(block.coef(1), 0);
    assert_eq!(block.coef(2), 1);

    let mut activations = [0.0f32; 128];
    activations[0] = 2.0;
    activations[1] = 5.0;
    activations[2] = 3.0;
    activations[3] = 4.0;

    // Dot product: 2.0 * (-1.0) + 5.0 * 0.0 + 3.0 * 1.0 + 4.0 * 2.0 = -2 + 0 + 3 + 8 = 9.0 * scale (2.0)
    // Here amax = 2.0, so d = 2.0. The quantized coefs normalized: input[0]*id = -0.5 -> round + 1 = 1 -> coef 0!
    // Let's compute actual dot product using block methods:
    let dot = block.dot_product_128(&activations);
    let mut dequant = [0.0f32; 128];
    block.dequantize(&mut dequant);
    let expected: f32 = activations
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| a * b)
        .sum();
    assert!((dot - expected).abs() < 1e-4);
}
