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

    // 5. Kronos Trading Foundation Engine
    let kronos_spec = ModelSpecification::lookup("shiyu-coder/kronos").expect("Found Kronos");
    assert_eq!(kronos_spec.family, ModelFamily::KronosTrading);
    assert_eq!(
        kronos_spec.modality,
        ModelModality::FinancialTradingTimeSeries
    );

    let kronos_engine = oxide_models::KronosTradingEngine::new(2048, 5, 3, 1.0);
    let mut bars = Vec::new();
    for i in 0..10 {
        bars.push(oxide_models::FinancialMarketBar {
            timestamp_epoch_ms: 1_700_000_000_000 + i * 60_000,
            open: 100.0 + (i as f32) * 0.5,
            high: 101.0 + (i as f32) * 0.5,
            low: 99.8 + (i as f32) * 0.5,
            close: 100.8 + (i as f32) * 0.5,
            volume: 50_000.0,
            vwap: 100.5 + (i as f32) * 0.5,
            bid_ask_spread_bps: 1.2,
            order_flow_imbalance: 0.65, // Strong positive order book delta
        });
    }

    let trade_forecast = kronos_engine
        .evaluate_market_bars(&bars)
        .expect("Kronos evaluation succeeded");
    assert_eq!(trade_forecast.signal, oxide_models::TradingSignal::Buy);
    assert!(trade_forecast.expected_return_bps > 25.0);
    assert_eq!(trade_forecast.multi_horizon_returns.len(), 3);
}

#[test]
fn test_microsoft_research_suites() {
    // 1. Model Registry lookups for all requested Microsoft Research models
    let ms_models = [
        (
            "microsoft/muzic",
            oxide_models::registry::ModelFamily::MuzicMuseCoco,
            oxide_models::registry::ModelModality::SymbolicMusicMidi,
        ),
        (
            "microsoft/visual-chatgpt",
            oxide_models::registry::ModelFamily::VisualChatGpt,
            oxide_models::registry::ModelModality::VisionLanguage,
        ),
        (
            "microsoft/nuwa",
            oxide_models::registry::ModelFamily::NuwaVisualGen,
            oxide_models::registry::ModelModality::InfiniteVisualSynthesis,
        ),
        (
            "microsoft/data-formulator",
            oxide_models::registry::ModelFamily::DataFormulator,
            oxide_models::registry::ModelModality::DataAnalyticsVisualizer,
        ),
        (
            "microsoft/qlib",
            oxide_models::registry::ModelFamily::QlibQuant,
            oxide_models::registry::ModelModality::QuantitativeAlphaModel,
        ),
        (
            "microsoft/finance-benchmark",
            oxide_models::registry::ModelFamily::FinanceBenchmarkAgent,
            oxide_models::registry::ModelModality::DecisionAgentic,
        ),
        (
            "microsoft/rhobotics",
            oxide_models::registry::ModelFamily::RhoboticsVla,
            oxide_models::registry::ModelModality::BimanualRoboticsVla,
        ),
        (
            "microsoft/physical-ai-toolchain",
            oxide_models::registry::ModelFamily::PhysicalAiToolchain,
            oxide_models::registry::ModelModality::RoboticsActionVla,
        ),
        (
            "microsoft/scene-aware-robot-bt-planner",
            oxide_models::registry::ModelFamily::RobotBehaviorTreePlanner,
            oxide_models::registry::ModelModality::BehaviorTreePlanner,
        ),
        (
            "microsoft/dayhoff",
            oxide_models::registry::ModelFamily::DayhoffBio,
            oxide_models::registry::ModelModality::ProteinGenomicsAtlas,
        ),
        (
            "microsoft/vermeer",
            oxide_models::registry::ModelFamily::VermeerMicroscopy,
            oxide_models::registry::ModelModality::MicroscopyImageGeneration,
        ),
        (
            "microsoft/healthcareai-examples",
            oxide_models::registry::ModelFamily::HealthcareAiMedImage,
            oxide_models::registry::ModelModality::MedicalImagingDiagnostics,
        ),
        (
            "microsoft/aurora",
            oxide_models::registry::ModelFamily::AuroraAtmospheric,
            oxide_models::registry::ModelModality::AtmosphericEarthSystem,
        ),
        (
            "microsoft/farmvibes-ai",
            oxide_models::registry::ModelFamily::FarmVibesGeospatial,
            oxide_models::registry::ModelModality::GeospatialAgriculture,
        ),
        (
            "microsoft/planetary-explorer",
            oxide_models::registry::ModelFamily::PlanetaryExplorer,
            oxide_models::registry::ModelModality::DataAnalyticsVisualizer,
        ),
        (
            "microsoft/orbitalbrain",
            oxide_models::registry::ModelFamily::OrbitalBrainSpace,
            oxide_models::registry::ModelModality::AtmosphericEarthSystem,
        ),
        (
            "microsoft/a11y-llm-eval",
            oxide_models::registry::ModelFamily::A11yLlmEval,
            oxide_models::registry::ModelModality::CodeReasoning,
        ),
        (
            "microsoft/haste",
            oxide_models::registry::ModelFamily::HasteDisasterSat,
            oxide_models::registry::ModelModality::SatelliteDisasterAssessment,
        ),
        (
            "microsoft/biodiversity",
            oxide_models::registry::ModelFamily::BiodiversityWildlife,
            oxide_models::registry::ModelModality::WildlifeConservationDetection,
        ),
        (
            "microsoft/ai4g-flood",
            oxide_models::registry::ModelFamily::Ai4gFloodSar,
            oxide_models::registry::ModelModality::SatelliteDisasterAssessment,
        ),
        (
            "microsoft/contractor",
            oxide_models::registry::ModelFamily::ContractorLegalAgent,
            oxide_models::registry::ModelModality::LegalContractAudit,
        ),
        (
            "microsoft/wham",
            oxide_models::registry::ModelFamily::WhamGameplay,
            oxide_models::registry::ModelModality::GameplayWorldAction,
        ),
    ];

    for (name, family, modality) in ms_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 2. Symbolic Music Engine (microsoft/muzic - MusicBERT / MuseCoco)
    let music_engine = oxide_models::SymbolicMusicEngine::new(512, 120);
    let prompt_tokens = vec![42, 108, 64, 32];
    let midi_events = music_engine
        .compose_from_prompt(&prompt_tokens)
        .expect("Generated MIDI sequence");
    assert_eq!(midi_events.len(), 4);
    assert_eq!(midi_events[0].pitch_midi, 66);

    // 3. Bimanual Robotics Engine (microsoft/rhobotics)
    let rhobotics_engine = oxide_models::BimanualRoboticsEngine::new(256, 4);
    let visual_latent = vec![0.5f32; 256];
    let bimanual_actions = rhobotics_engine
        .predict_bimanual_trajectory(&visual_latent)
        .expect("Predicted bimanual action chunk");
    assert_eq!(bimanual_actions.len(), 4);
    assert_eq!(bimanual_actions[0].left_arm_joints.len(), 7);
    assert_eq!(bimanual_actions[0].right_arm_joints.len(), 7);

    // 4. Atmospheric Aurora 3D Simulation Engine (microsoft/aurora)
    let aurora_engine = oxide_models::AtmosphericAuroraEngine::new(10, 20, 4);
    let atmospheric_state = vec![285.0f32; 10 * 20 * 4];
    let forecast_grid = aurora_engine
        .forecast_atmosphere(&atmospheric_state, 24)
        .expect("Atmospheric forecasting succeeded");
    assert_eq!(forecast_grid.temperature_kelvin.len(), 10 * 20 * 4);
    assert_eq!(forecast_grid.surface_pressure_pa.len(), 10 * 20);

    // 5. Gameplay WHAM Engine (microsoft/wham)
    let wham_engine = oxide_models::GameplayWhamEngine::new(128, 60);
    let screen_tokens = vec![0.4f32; 128];
    let predicted_input = wham_engine
        .step_gameplay_action(&screen_tokens)
        .expect("Gameplay action prediction succeeded");
    assert!(predicted_input.left_stick_x.is_finite());
    assert!(predicted_input.left_stick_y.is_finite());

    // 6. Satellite Disaster & Earth Observation Engine (microsoft/haste / ai4g-flood)
    let sat_engine = oxide_models::SatelliteEarthEngine::new(128, 0.5);
    let sar_pixels = vec![0.8f32; 128 * 128];
    let detections = sat_engine
        .detect_features(&sar_pixels, "flood")
        .expect("Satellite feature detection succeeded");
    assert_eq!(detections.len(), 1);
    assert!(detections[0].area_sq_km > 0.0);
}
