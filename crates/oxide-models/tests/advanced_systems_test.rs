use oxide_models::{
    ChatMessage, ChatTemplateFormat, ChatTemplateParser, DecisionMakingModelEngine,
    FlashAttentionConfig, FlashAttentionEngine, LoraConfig, LoraHotSwapRegistry, LoraLayerWeights,
    MultiModalProjector, ProjectorConfig, ProjectorType, RopeConfig, RopeScalingEngine,
    RopeScalingType,
};
use std::collections::HashMap;

#[test]
fn test_rope_scaling_variants() {
    // 1. YaRN scaling
    let yarn_cfg = RopeConfig {
        head_dim: 128,
        base_theta: 10000.0,
        scaling: RopeScalingType::Yarn {
            factor: 4.0,
            original_max_position: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        },
        max_position_embeddings: 32768,
    };
    let yarn_engine = RopeScalingEngine::new(yarn_cfg);
    let mut vec = vec![1.0; 128];
    yarn_engine.apply_rotary_in_place(&mut vec, 16384);
    assert_ne!(vec[0], 1.0);

    // 2. Llama 3 RoPE scaling
    let llama3_cfg = RopeConfig {
        head_dim: 128,
        base_theta: 500000.0,
        scaling: RopeScalingType::Llama3 {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8192,
        },
        max_position_embeddings: 131072,
    };
    let llama3_engine = RopeScalingEngine::new(llama3_cfg);
    let mut vec2 = vec![1.0; 128];
    llama3_engine.apply_rotary_in_place(&mut vec2, 32768);
    assert_ne!(vec2[0], 1.0);
}

#[test]
fn test_flash_attention_engine() {
    let config = FlashAttentionConfig::new(1, 1, 64, true);
    let engine = FlashAttentionEngine::new(config);

    let seq_len = 4;
    let head_dim = 64;
    let total = seq_len * head_dim;

    let q = vec![0.1f32; total];
    let k = vec![0.1f32; total];
    let v = vec![0.5f32; total];
    let mut out = vec![0.0f32; total];

    engine.forward_head(&q, &k, &v, seq_len, seq_len, &mut out);

    for &val in &out {
        assert!((val - 0.5).abs() < 0.01);
    }
}

#[test]
fn test_lora_and_qlora_hot_swapping() {
    let mut registry = LoraHotSwapRegistry::new();
    let config = LoraConfig::new("code-assistant-lora", 8, 16.0);

    let mut layers = HashMap::new();
    let layer_weights = LoraLayerWeights::new(64, 64, 8, 16.0);
    layers.insert("layers.0.q_proj".to_string(), layer_weights);

    registry.register_adapter(config, layers);
    assert_eq!(registry.active_adapter(), Some("code-assistant-lora"));

    let input = vec![1.0; 64];
    let mut output = vec![0.0; 64];
    let lora = registry.get_active_layer("layers.0.q_proj").unwrap();
    lora.forward_delta(&input, &mut output);

    // Output delta calculated without error
    assert_eq!(output.len(), 64);
}

#[test]
fn test_multimodal_projector() {
    let config = ProjectorConfig {
        projector_type: ProjectorType::MlpGelu,
        input_dim: 32,
        intermediate_dim: 64,
        output_dim: 128,
    };
    let projector = MultiModalProjector::new(config);

    let features = vec![0.5f32; 32 * 2]; // 2 image tokens
    let mut output = vec![0.0f32; 128 * 2];

    projector.project_features(&features, 2, &mut output);
    assert_eq!(output.len(), 256);
}

#[test]
fn test_chat_template_parser() {
    let parser = ChatTemplateParser::new(ChatTemplateFormat::ChatMl);
    let msgs = vec![
        ChatMessage::system("You are an AI assistant."),
        ChatMessage::user("Explain zero-copy Rust."),
    ];

    let rendered = parser.render_chat(&msgs);
    assert!(rendered.contains("<|im_start|>system\nYou are an AI assistant.<|im_end|>"));
    assert!(rendered.contains("<|im_start|>user\nExplain zero-copy Rust.<|im_end|>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_decision_making_models_jev_laya_clef() {
    let decider = DecisionMakingModelEngine::new(5, 0.1);

    // 1. JEV evaluation
    let state = vec![1.0, 2.0, 3.0];
    let actions = vec![vec![0.5, 0.5, 0.5], vec![2.0, 1.0, 0.0]];
    let jev_res = decider.evaluate_jev(&state, &actions).unwrap();
    assert_eq!(jev_res.optimal_action_index, 1);

    // 2. LAYA action generation
    let goal = vec![1.0, 1.0, 1.0];
    let laya_res = decider.generate_laya_action(&goal, &state).unwrap();
    assert_eq!(laya_res.continuous_deltas.len(), 3);
    assert!(laya_res.policy_confidence > 0.0);

    // 3. CLEF causal reasoning
    let vars = vec!["InterestRate".to_string(), "Inflation".to_string()];
    let evidence = vec![0.05, 0.02];
    let clef_res = decider.infer_causal_clef(&vars, &evidence, "InterestRate").unwrap();
    assert_eq!(clef_res.causal_graph_nodes.len(), 2);
    assert!(clef_res.interventions[0].contains("InterestRate"));
}
