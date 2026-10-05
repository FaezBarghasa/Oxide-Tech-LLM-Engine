use oxide_alloc::{
    ContextShiftManager, KvCacheDumpContainer, KvQuantizationPrecision, PromptCacheRegistry,
    QuantizedKvBlock,
};

#[test]
fn test_on_the_fly_kv_quantization_and_dequantization() {
    let head_dim = 128;
    let token_count = 32;
    let total_elements = token_count * head_dim;

    let mut k_original = vec![0.0f32; total_elements];
    let mut v_original = vec![0.0f32; total_elements];
    for i in 0..total_elements {
        k_original[i] = ((i as f32) * 0.05).cos() * 1.5;
        v_original[i] = ((i as f32) * 0.05).sin() * 1.5;
    }

    // 1. Float16 Quantized Block
    let f16_block = QuantizedKvBlock::quantize_from_f32(
        1,
        token_count,
        head_dim,
        &k_original,
        &v_original,
        KvQuantizationPrecision::Float16,
    );
    let mut k_dequant = vec![0.0f32; total_elements];
    let mut v_dequant = vec![0.0f32; total_elements];
    f16_block.dequantize_into(&mut k_dequant, &mut v_dequant);
    for i in 0..total_elements {
        assert!((k_dequant[i] - k_original[i]).abs() < 0.01);
        assert!((v_dequant[i] - v_original[i]).abs() < 0.01);
    }

    // 2. Q8_0 Quantized Block
    let q8_block = QuantizedKvBlock::quantize_from_f32(
        2,
        token_count,
        head_dim,
        &k_original,
        &v_original,
        KvQuantizationPrecision::Quant8_0,
    );
    q8_block.dequantize_into(&mut k_dequant, &mut v_dequant);
    for i in 0..total_elements {
        assert!((k_dequant[i] - k_original[i]).abs() < 0.05);
    }

    // 3. Q4_0 Quantized Block
    let q4_block = QuantizedKvBlock::quantize_from_f32(
        3,
        token_count,
        head_dim,
        &k_original,
        &v_original,
        KvQuantizationPrecision::Quant4_0,
    );
    q4_block.dequantize_into(&mut k_dequant, &mut v_dequant);
    assert_eq!(q4_block.quantized_k_data.len(), total_elements / 2);
}

#[test]
fn test_context_shift_manager() {
    let mut manager = ContextShiftManager::new(4096, 512, 1024);
    let mut token_ids: Vec<u32> = (0..5000).collect();

    assert!(manager.requires_shift(token_ids.len()));
    let shifted = manager.shift_context(&mut token_ids).unwrap();

    assert_eq!(token_ids.len(), 5000 - shifted);
    assert_eq!(token_ids[0..512], (0..512).collect::<Vec<u32>>()); // Prompt prefix preserved!
}

#[test]
fn test_prompt_cache_registry() {
    let mut registry = PromptCacheRegistry::new(10);
    let prompt1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let kv_block_ids = vec![101, 102];

    registry.register_prompt(&prompt1, kv_block_ids.clone());

    // Exact match lookup
    let lookup = registry.lookup_prompt(&prompt1);
    assert_eq!(lookup, Some(kv_block_ids));

    // Non-matching prompt
    assert_eq!(registry.lookup_prompt(&[9, 9, 9]), None);
}

#[test]
fn test_kv_cache_dumping_and_reloading() {
    let container = KvCacheDumpContainer {
        model_name: "llama-3-8b".to_string(),
        total_layers: 32,
        tokens_cached: 1024,
        head_dim: 128,
        precision: KvQuantizationPrecision::Float16,
        blocks: vec![QuantizedKvBlock::quantize_from_f32(
            1,
            32,
            128,
            &vec![0.5; 32 * 128],
            &vec![0.5; 32 * 128],
            KvQuantizationPrecision::Float16,
        )],
    };

    let serialized = container.dump_to_binary().unwrap();
    let reloaded = KvCacheDumpContainer::reload_from_binary(&serialized).unwrap();

    assert_eq!(reloaded.model_name, "llama-3-8b");
    assert_eq!(reloaded.tokens_cached, 1024);
    assert_eq!(reloaded.blocks.len(), 1);
}
