use oxide_models::formats::{GgufHeader, ModelFileFormat, Nvfp4Block, SafeTensorsHeader};
use oxide_models::moe::{MoELayer, MoERouterConfig};
use oxide_models::registry::{
    ModelArchitectureType, ModelFamily, ModelModality, ModelSpecification, QuantizationClass,
};
use oxide_models::specialized::{
    AgenticDeciderEngine, EmbeddingEngine, RoboticsVlaEngine, TimeSeriesEngine,
};

#[test]
fn test_model_registry_catalog_and_lookup() {
    let qwen = ModelSpecification::lookup("qwen-3.8").expect("Qwen 3.8 found");
    assert_eq!(qwen.family, ModelFamily::Qwen);
    assert_eq!(qwen.modality, ModelModality::TextOnly);
    assert_eq!(qwen.total_parameters_billion, 3.8);

    let deepseek_v4 = ModelSpecification::lookup("deepseek-v4").expect("DeepSeek V4 found");
    assert_eq!(deepseek_v4.family, ModelFamily::DeepSeek);
    assert_eq!(
        deepseek_v4.architecture,
        ModelArchitectureType::MultiHeadLatentAttentionMla
    );
    assert_eq!(deepseek_v4.total_parameters_billion, 236.0);
    assert_eq!(deepseek_v4.active_parameters_billion, 21.0);
    assert_eq!(
        deepseek_v4.default_quantization,
        QuantizationClass::Nvfp4Blackwell
    );

    let deepseek_r1 = ModelSpecification::lookup("deepseek-r1").expect("DeepSeek R1 found");
    assert_eq!(deepseek_r1.modality, ModelModality::CodeReasoning);
    assert_eq!(deepseek_r1.active_parameters_billion, 37.0);

    let kimi = ModelSpecification::lookup("kimi-k3").expect("Kimi K3 found");
    assert_eq!(kimi.max_context_tokens, 1048576); // 1M context

    let omni = ModelSpecification::lookup("nemotron-3-nano-omni").expect("Nemotron Omni found");
    assert_eq!(omni.modality, ModelModality::OmniAnyToAny);
    assert_eq!(omni.total_parameters_billion, 30.0);
    assert_eq!(omni.active_parameters_billion, 3.0);

    let robotics = ModelSpecification::lookup("isaac-gr00t").expect("Isaac GR00T found");
    assert_eq!(robotics.modality, ModelModality::RoboticsActionVla);

    let kairos = ModelSpecification::lookup("kairos-3.0").expect("Kairos 3.0 found");
    assert_eq!(
        kairos.architecture,
        ModelArchitectureType::HybridLinearAttention
    );
}

#[test]
fn test_moe_routing_and_forward() {
    let config = MoERouterConfig {
        num_routed_experts: 16,
        num_shared_experts: 2,
        top_k: 4,
        routed_scaling_factor: 1.0,
        hidden_dim: 128,
        intermediate_dim: 256,
        use_nvfp4: true,
    };

    let moe = MoELayer::new(config);
    let hidden = vec![0.5f32; 128];
    let routing = moe.route_token(&hidden).expect("Routing succeeded");

    assert_eq!(routing.selected_expert_indices.len(), 4);
    assert_eq!(routing.routing_weights.len(), 4);

    let sum_weights: f32 = routing.routing_weights.iter().sum();
    assert!((sum_weights - 1.0).abs() < 1e-4);

    let mut out = vec![0.0f32; 128];
    moe.forward(&hidden, &mut out).expect("Forward succeeded");
    assert!(out.iter().all(|&v| v > 0.0));
}

#[test]
fn test_nvfp4_quantization_and_dequantization() {
    let original = vec![
        0.0f32, 1.0, -1.0, 2.5, -2.5, 4.0, -4.0, 5.5, -0.5, 0.5, 1.5, -1.5, 3.0, -3.0, 6.0, -6.0,
    ];

    let block = Nvfp4Block::quantize(&original);
    assert_eq!(block.packed_nibbles.len(), 8); // 16 values packed into 8 bytes (2x reduction)

    let mut dequantized = vec![0.0f32; 16];
    block.dequantize(&mut dequantized);

    for (orig, deq) in original.iter().zip(dequantized.iter()) {
        assert!((orig - deq).abs() < 1.5, "Mismatch for {orig}: got {deq}");
    }
}

#[test]
fn test_safetensors_and_gguf_parsing() {
    // 1. SafeTensors Header parse test
    let json_header = r#"{"__metadata__":{"format":"pt"},"weight_0":{"dtype":"F32","shape":[64,128],"data_offsets":[0,32768]}}"#;
    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&(json_header.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(json_header.as_bytes());

    let (st_hdr, offset) =
        SafeTensorsHeader::parse_from_bytes(&st_bytes).expect("Parsed SafeTensors");
    assert_eq!(offset, 8 + json_header.len());
    assert_eq!(st_hdr.metadata.get("format").unwrap(), "pt");
    let w0 = st_hdr.tensors.get("weight_0").expect("Found weight_0");
    assert_eq!(w0.shape, vec![64, 128]);
    assert_eq!(w0.data_offsets, (0, 32768));

    // 2. GGUF Header parse test
    let mut gguf_bytes = Vec::new();
    gguf_bytes.extend_from_slice(b"GGUF");
    gguf_bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
    gguf_bytes.extend_from_slice(&128u64.to_le_bytes()); // 128 tensors
    gguf_bytes.extend_from_slice(&45u64.to_le_bytes()); // 45 KV entries

    let gguf_hdr = GgufHeader::parse(&gguf_bytes).expect("Parsed GGUF");
    assert_eq!(gguf_hdr.version, 3);
    assert_eq!(gguf_hdr.tensor_count, 128);
    assert_eq!(gguf_hdr.metadata_kv_count, 45);
}

#[test]
fn test_specialized_engines() {
    // 1. Robotics VLA
    let vla = RoboticsVlaEngine::new(256, 4, true);
    let visual_tokens = vec![0.3f32; 256];
    let actions = vla
        .predict_action_chunk(&visual_tokens)
        .expect("Predicted actions");
    assert_eq!(actions.len(), 4);
    assert_eq!(actions[0].arm_joint_positions.len(), 7);

    // 2. Embedding Engine
    let emb_engine = EmbeddingEngine::new(64, 512);
    let token_states = vec![0.1f32; 4 * 64];
    let emb = emb_engine
        .compute_dense_embedding(&token_states, 4)
        .expect("Embedding generated");
    assert_eq!(emb.len(), 64);
    let norm_sq: f32 = emb.iter().map(|&x| x * x).sum();
    assert!((norm_sq - 1.0).abs() < 1e-3); // L2 normalized

    // 3. Time Series Engine
    let ts_engine = TimeSeriesEngine::new(8, 4);
    let history = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let forecast = ts_engine.forecast(&history).expect("Forecasted");
    assert_eq!(forecast.len(), 4);
    assert!(forecast[0] > 8.0);

    // 4. Agentic Decider
    let categories = vec![
        "tool_call".to_string(),
        "final_answer".to_string(),
        "clarify".to_string(),
    ];
    let decider = AgenticDeciderEngine::new(categories);
    let logits = vec![5.0, 2.0, 1.0];
    let (decision, conf) = decider.decide(&logits).expect("Decided");
    assert_eq!(decision, "tool_call");
    assert!(conf > 0.8);
}
