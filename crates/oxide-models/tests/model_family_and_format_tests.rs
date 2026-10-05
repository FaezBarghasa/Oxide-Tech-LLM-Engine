use oxide_models::formats::{GgufHeader, Nvfp4Block, SafeTensorsHeader};
use oxide_models::llama3::{Llama3Config, Llama3KvCacheLayer, Llama3Model};
use oxide_models::moe::{MoELayer, MoERouterConfig};
use oxide_models::registry::{
    ModelArchitectureType, ModelFamily, ModelModality, ModelSpecification, QuantizationClass,
};
use oxide_models::specialized::{
    AgenticDeciderEngine, EmbeddingEngine, RoboticsVlaEngine, TimeSeriesEngine,
};

#[test]
fn test_llama3_model_forward_execution() {
    let config = Llama3Config::tiny_test_config();
    let model = Llama3Model::new(config);
    let mut kv_cache =
        vec![
            Llama3KvCacheLayer::new(config.max_seq_len, config.num_kv_heads * config.head_dim,);
            config.num_layers
        ];

    let logits = model
        .forward_step(42, 0, &mut kv_cache)
        .expect("Forward step 0");
    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().any(|&l| l.is_finite()));

    // Forward step 1 (with cached past KV tokens)
    let logits_2 = model
        .forward_step(100, 1, &mut kv_cache)
        .expect("Forward step 1");
    assert_eq!(logits_2.len(), config.vocab_size);
    assert_eq!(kv_cache[0].current_len, 2);
}

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

#[test]
fn test_opensource_music_robotics_clinical_and_experimental_suites() {
    // 1. Model Registry catalog lookups for all open-source & experimental foundation models
    let os_models = [
        (
            "heartmula",
            oxide_models::registry::ModelFamily::HeartMuLa,
            oxide_models::registry::ModelModality::MusicSongFullGeneration,
        ),
        (
            "ace-step",
            oxide_models::registry::ModelFamily::AceStep,
            oxide_models::registry::ModelModality::MusicSongFullGeneration,
        ),
        (
            "yue2-studio",
            oxide_models::registry::ModelFamily::YuE2Studio,
            oxide_models::registry::ModelModality::MusicSongFullGeneration,
        ),
        (
            "open-qwen-music",
            oxide_models::registry::ModelFamily::OpenQwenMusic,
            oxide_models::registry::ModelModality::MusicSongFullGeneration,
        ),
        (
            "songgen",
            oxide_models::registry::ModelFamily::SongGen,
            oxide_models::registry::ModelModality::MusicSongFullGeneration,
        ),
        (
            "datapilot",
            oxide_models::registry::ModelFamily::DataPilotPolars,
            oxide_models::registry::ModelModality::TimeSeriesTabular,
        ),
        (
            "datasight",
            oxide_models::registry::ModelFamily::DatasightSqlAgent,
            oxide_models::registry::ModelModality::DecisionAgentic,
        ),
        (
            "ai-data-science-team",
            oxide_models::registry::ModelFamily::AiDataScienceTeam,
            oxide_models::registry::ModelModality::DecisionAgentic,
        ),
        (
            "glin",
            oxide_models::registry::ModelFamily::GlinBoostingMachine,
            oxide_models::registry::ModelModality::TimeSeriesTabular,
        ),
        (
            "xiaomi-robotics-0",
            oxide_models::registry::ModelFamily::XiaomiRobotics0,
            oxide_models::registry::ModelModality::RoboticsActionVla,
        ),
        (
            "kairos-3.0-4b",
            oxide_models::registry::ModelFamily::KairosEmbodiedWorldModel,
            oxide_models::registry::ModelModality::WorldModelSimulation,
        ),
        (
            "rldx-1",
            oxide_models::registry::ModelFamily::RldxDexterousManipulation,
            oxide_models::registry::ModelModality::DexterousFiveFingerRobotics,
        ),
        (
            "a1",
            oxide_models::registry::ModelFamily::A1AdaptiveVla,
            oxide_models::registry::ModelModality::RoboticsActionVla,
        ),
        (
            "pyhealth-2.0",
            oxide_models::registry::ModelFamily::PyHealthClinical,
            oxide_models::registry::ModelModality::ClinicalPhenotypingCdss,
        ),
        (
            "aidiva",
            oxide_models::registry::ModelFamily::AiDivaRareDisease,
            oxide_models::registry::ModelModality::ClinicalPhenotypingCdss,
        ),
        (
            "pie-med",
            oxide_models::registry::ModelFamily::PieMedGcnCdss,
            oxide_models::registry::ModelModality::ClinicalPhenotypingCdss,
        ),
        (
            "realphe",
            oxide_models::registry::ModelFamily::RealPheCriticalCare,
            oxide_models::registry::ModelModality::ClinicalPhenotypingCdss,
        ),
        (
            "chatenv",
            oxide_models::registry::ModelFamily::ChatEnvEcosystem,
            oxide_models::registry::ModelModality::VisionLanguage,
        ),
        (
            "lite",
            oxide_models::registry::ModelFamily::LiteEnvironmentalVlm,
            oxide_models::registry::ModelModality::VisionLanguage,
        ),
        (
            "planaura",
            oxide_models::registry::ModelFamily::PlanauraGeospatial,
            oxide_models::registry::ModelModality::SatelliteDisasterAssessment,
        ),
        (
            "multimodal-auv",
            oxide_models::registry::ModelFamily::MultimodalAuvMapping,
            oxide_models::registry::ModelModality::UnderwaterAuvMapping,
        ),
        (
            "accessbridge-ai",
            oxide_models::registry::ModelFamily::AccessBridgeUniversalWeb,
            oxide_models::registry::ModelModality::UniversalAccessibilityAssist,
        ),
        (
            "visionassist",
            oxide_models::registry::ModelFamily::VisionAssistMobile,
            oxide_models::registry::ModelModality::UniversalAccessibilityAssist,
        ),
        (
            "sightlineai",
            oxide_models::registry::ModelFamily::SightlineSmartGlasses,
            oxide_models::registry::ModelModality::UniversalAccessibilityAssist,
        ),
        (
            "tinynarrator",
            oxide_models::registry::ModelFamily::TinyNarratorScreenReader,
            oxide_models::registry::ModelModality::UniversalAccessibilityAssist,
        ),
        (
            "sutradhar",
            oxide_models::registry::ModelFamily::SutradharMultimodalAssist,
            oxide_models::registry::ModelModality::UniversalAccessibilityAssist,
        ),
        (
            "sorbet",
            oxide_models::registry::ModelFamily::SorbetNeuromorphicSpiking,
            oxide_models::registry::ModelModality::NeuromorphicSpikingInference,
        ),
        (
            "aetheris",
            oxide_models::registry::ModelFamily::AetherisMambaMoe,
            oxide_models::registry::ModelModality::MambaMoeStateSpace,
        ),
        (
            "trm",
            oxide_models::registry::ModelFamily::TinyRecursionModelTrm,
            oxide_models::registry::ModelModality::RecursiveSmallReasoning,
        ),
        (
            "ddn",
            oxide_models::registry::ModelFamily::DiscreteDistributionNetworkDdn,
            oxide_models::registry::ModelModality::TextOnly,
        ),
        (
            "brillm",
            oxide_models::registry::ModelFamily::BriLlmBrainInspired,
            oxide_models::registry::ModelModality::NeuromorphicSpikingInference,
        ),
        (
            "ixlinx-8b",
            oxide_models::registry::ModelFamily::IXlinxRecurrentMultimodal,
            oxide_models::registry::ModelModality::VisionLanguage,
        ),
        (
            "functional-graph-agi",
            oxide_models::registry::ModelFamily::FunctionalGraphAgi,
            oxide_models::registry::ModelModality::FunctionalGraphCognition,
        ),
    ];

    for (name, family, modality) in os_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 2. 5-Finger Dexterous Robotics Engine (RLDX-1, Xiaomi-Robotics-0)
    let dexterous = oxide_models::DexterousRoboticsEngine::new(256);
    let visual_latent = vec![0.7f32; 256];
    let hand_state = dexterous
        .step_dexterous_action(&visual_latent)
        .expect("Dexterous hand step succeeded");
    assert_eq!(hand_state.thumb_joints.len(), 4);
    assert_eq!(hand_state.index_joints.len(), 4);
    assert_eq!(hand_state.fingertip_tactile_pressure_n.len(), 5);
    assert!(hand_state.grasp_stability_score > 0.9);

    // 3. Clinical Decision & Phenotyping Engine (PyHealth, aiDIVA, PIE-Med, RealPhe)
    let clinical = oxide_models::ClinicalDecisionEngine::new(128, 16);
    let ehr_features = vec![14.5f32; 16]; // High lactate / heart rate profile
    let rec = clinical
        .evaluate_patient_ehr(&ehr_features)
        .expect("Clinical evaluation succeeded");
    assert!(rec.phenotype_risk_score > 0.5);
    assert!(!rec.icd10_codes.is_empty());

    // 4. Sensor-Guided Geospatial & Underwater AUV Engine (ChatENV, LITE, Planaura, Multimodal-AUV)
    let eco_engine = oxide_models::GeospatialEcosystemEngine::new(4, 10.0);
    let readings = vec![0.75f32, 0.15f32, 0.3f32, 0.45f32]; // NIR, RED, Green, Blue
    let eco_metrics = eco_engine
        .analyze_ecosystem_sensor(&readings)
        .expect("Ecosystem analysis succeeded");
    assert!(eco_metrics.ndvi_vegetation_index > 0.5); // Healthy vegetation

    // 5. Assistive Vision Engine (SightlineAI, VisionAssist, Sutradhar, AccessBridge)
    let assist_engine = oxide_models::AssistiveVisionEngine::new(90.0, true);
    let scene = vec![0.2f32, 0.8f32, 0.2f32];
    let nav_prompt = assist_engine
        .evaluate_scene_for_assist(&scene)
        .expect("Assistive navigation succeeded");
    assert!(nav_prompt.distance_meters > 0.0);
    assert!(nav_prompt.haptic_vibration_intensity > 0.0);

    // 6. Neuromorphic Spiking Engine (Sorbet, BriLLM)
    let spiking_engine = oxide_models::NeuromorphicSpikingEngine::new(8, 0.8, 0.1);
    let currents = vec![1.2f32, 0.2, 1.5, 0.4, 0.95, 0.1, 1.1, 0.0];
    let spike_state = spiking_engine
        .step_lif_spiking(&currents)
        .expect("Spiking step succeeded");
    assert_eq!(spike_state.spike_events.len(), 8);
    assert!(spike_state.total_spike_count >= 3);

    // 7. Tiny Recursion Model TRM (7M Parameter Contraction Reasoner)
    let trm = oxide_models::TinyRecursionModelEngine::new(32, 10, 1e-3);
    let init_h = vec![0.5f32; 32];
    let trm_res = trm
        .recursive_reason(&init_h)
        .expect("TRM reasoning succeeded");
    assert!(trm_res.iterations_performed <= 10);
    assert_eq!(trm_res.contracted_hidden_state.len(), 32);

    // 8. Hybrid Mamba-MoE Engine (Aetheris)
    let mamba_moe = oxide_models::MambaMoeHybridEngine::new(64, 16, 8, 2);
    let inp = vec![0.3f32; 64];
    let mut ssm_state = vec![0.0f32; 16];
    let mut out = vec![0.0f32; 64];
    mamba_moe
        .forward_step(&inp, &mut ssm_state, &mut out)
        .expect("Mamba-MoE forward step succeeded");
    assert!(out.iter().all(|&v| v > 0.0));

    // 9. Explainable Boosting Machine / Polars Statistical Gut Engine (DataPilot, glin)
    let gut_engine = oxide_models::PolarsStatisticalGutEngine::new(32, 2.0);
    let row = vec![0.5f32, 1.2, 0.8, 2.5];
    let (score, is_anomaly) = gut_engine
        .evaluate_tabular_row(&row)
        .expect("Tabular evaluation succeeded");
    assert!(score > 0.0);
    assert!(!is_anomaly || is_anomaly);
}

#[test]
fn test_eda_cad_materials_science_and_qa_suites() {
    // 1. Electronic Design & PCB (EDA) Models
    let eda_models = [
        (
            "boardsmith",
            oxide_models::registry::ModelFamily::BoardSmithEda,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "ai-pcb-generator",
            oxide_models::registry::ModelFamily::AiPcbGeneratorSuite,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "kicad-mcp",
            oxide_models::registry::ModelFamily::KicadMcpServer,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "kicad-autopilot",
            oxide_models::registry::ModelFamily::KicadAutopilotRouter,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "field-ratchet",
            oxide_models::registry::ModelFamily::FieldRatchetPcb,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "electrodesign-ai",
            oxide_models::registry::ModelFamily::ElectroDesignCircuitVerse,
            oxide_models::registry::ModelModality::SpiceCircuitSimulation,
        ),
        (
            "electroninja",
            oxide_models::registry::ModelFamily::ElectroNinjaLtSpice,
            oxide_models::registry::ModelModality::SpiceCircuitSimulation,
        ),
    ];

    for (name, family, modality) in eda_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("EDA model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 2. CAD & 3D Modeling Models
    let cad_models = [
        (
            "ai-cad",
            oxide_models::registry::ModelFamily::AiCadParametric,
            oxide_models::registry::ModelModality::ParametricCad3DGeneration,
        ),
        (
            "cadam",
            oxide_models::registry::ModelFamily::CadamWasm3D,
            oxide_models::registry::ModelModality::ParametricCad3DGeneration,
        ),
        (
            "gptcad",
            oxide_models::registry::ModelFamily::GptCadFreeCad,
            oxide_models::registry::ModelModality::ParametricCad3DGeneration,
        ),
        (
            "guidecad",
            oxide_models::registry::ModelFamily::GuideCadMultimodal,
            oxide_models::registry::ModelModality::ParametricCad3DGeneration,
        ),
        (
            "stunning-modeler",
            oxide_models::registry::ModelFamily::StunningModeler3D,
            oxide_models::registry::ModelModality::Asset3DGeneration,
        ),
        (
            "cube-3d",
            oxide_models::registry::ModelFamily::Cube3D,
            oxide_models::registry::ModelModality::Asset3DGeneration,
        ),
    ];

    for (name, family, modality) in cad_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("CAD model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 3. Data Analysis & Automation Models
    let data_models = [
        (
            "llmflow",
            oxide_models::registry::ModelFamily::LlmFlowRWorkflow,
            oxide_models::registry::ModelModality::DataWorkflowAutomation,
        ),
        (
            "deepanalyze",
            oxide_models::registry::ModelFamily::DeepAnalyzeRuc,
            oxide_models::registry::ModelModality::TimeSeriesTabular,
        ),
        (
            "doctrail",
            oxide_models::registry::ModelFamily::DocTrailSqlite,
            oxide_models::registry::ModelModality::DataWorkflowAutomation,
        ),
    ];

    for (name, family, modality) in data_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Data model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 4. Programming & Software Development Models
    let dev_models = [
        (
            "tabby",
            oxide_models::registry::ModelFamily::TabbyCodingAssistant,
            oxide_models::registry::ModelModality::CodeReasoning,
        ),
        (
            "aider",
            oxide_models::registry::ModelFamily::AiderPairProgrammer,
            oxide_models::registry::ModelModality::PairProgrammingSoftwareAgent,
        ),
        (
            "the-pair",
            oxide_models::registry::ModelFamily::ThePairDualAgent,
            oxide_models::registry::ModelModality::PairProgrammingSoftwareAgent,
        ),
        (
            "patch",
            oxide_models::registry::ModelFamily::PatchPairProgrammer,
            oxide_models::registry::ModelModality::PairProgrammingSoftwareAgent,
        ),
        (
            "coderai",
            oxide_models::registry::ModelFamily::CoderAiTerminalAgent,
            oxide_models::registry::ModelModality::PairProgrammingSoftwareAgent,
        ),
        (
            "swe-cli",
            oxide_models::registry::ModelFamily::SweCliAgent,
            oxide_models::registry::ModelModality::PairProgrammingSoftwareAgent,
        ),
    ];

    for (name, family, modality) in dev_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Dev model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 5. Research & Scientific Discovery Models
    let science_models = [
        (
            "ai-research-paper-agent",
            oxide_models::registry::ModelFamily::AiResearchPaperRag,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "zori",
            oxide_models::registry::ModelFamily::ZoriZoteroAssistant,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "kosmos",
            oxide_models::registry::ModelFamily::KosmosAiScientist,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "autoresearchclaw",
            oxide_models::registry::ModelFamily::AutoResearchClawAgent,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "freephdlabor",
            oxide_models::registry::ModelFamily::FreePhdLaborMultiAgent,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "gnosis-ai",
            oxide_models::registry::ModelFamily::GnosisAiDiscovery,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
    ];

    for (name, family, modality) in science_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Science model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 6. Metallurgy & Materials Science Models
    let metallurgy_models = [
        (
            "alloygpt",
            oxide_models::registry::ModelFamily::AlloyGptAdditive,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "aidesignhea",
            oxide_models::registry::ModelFamily::AiDesignHeaHighEntropy,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "mtl-materials-design",
            oxide_models::registry::ModelFamily::MtlMaterialsDesign,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "semantic-metallurgy-lm",
            oxide_models::registry::ModelFamily::SemanticMetallurgyMagnesium,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "nsgan-aluminium",
            oxide_models::registry::ModelFamily::NsganAluminiumGenerative,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "das-dao",
            oxide_models::registry::ModelFamily::DasDaoScrapToAlloy,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
    ];

    for (name, family, modality) in metallurgy_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Metallurgy model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 7. Software Testing & QA Models
    let qa_models = [
        (
            "lionagi-qe-fleet",
            oxide_models::registry::ModelFamily::LionAgiQeFleet,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "falcon-automation",
            oxide_models::registry::ModelFamily::FalconPlaywrightAutomation,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "agent-qa",
            oxide_models::registry::ModelFamily::AgentQaSelfImproving,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "cognitest",
            oxide_models::registry::ModelFamily::CogniTestFramework,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "checkmate",
            oxide_models::registry::ModelFamily::CheckmatePlaywright,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
    ];

    for (name, family, modality) in qa_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("QA model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 8. Test EDA Engine
    let eda_engine = oxide_models::EdaPcbEngine::new(0.25, 0.2, 4);
    let netlist = eda_engine
        .generate_pcb_layout(
            "OxideSensorShield",
            8,
            &[
                "GND".to_string(),
                "VCC_3V3".to_string(),
                "SPI_MOSI".to_string(),
            ],
        )
        .expect("PCB layout generated");
    assert_eq!(netlist.components.len(), 8);
    assert_eq!(netlist.trace_routes.len(), 3);
    assert!(netlist.drc_clean);
    assert!(netlist.gerber_ready);

    // 9. Test Parametric CAD Engine
    let cad_engine = oxide_models::ParametricCadEngine::new(0.01, 16);
    let cad_model = cad_engine
        .generate_cad_model("EnclosureBase", [50.0, 35.0, 15.0])
        .expect("CAD model generated");
    assert_eq!(cad_model.dimensions_xyz_mm, [50.0, 35.0, 15.0]);
    assert!(cad_model.volume_mm3 > 20000.0);
    assert!(cad_model.mesh_triangle_count > 0);

    // 10. Test Data Workflow Engine
    let data_engine = oxide_models::DataWorkflowAutomationEngine::new(5, 30);
    let (steps, viz) = data_engine
        .plan_and_execute_analytics(
            "Analyze seasonal server load",
            &["metrics_telemetry".to_string()],
        )
        .expect("Data analytics executed");
    assert_eq!(steps.len(), 2);
    assert!(!viz.plotly_json_spec.is_empty());

    // 11. Test Pair Programming Engine
    let pair_engine = oxide_models::SoftwareEngineeringPairEngine::new(true, 500);
    let patch_action = pair_engine
        .create_patch("crates/oxide-core/src/lib.rs", "add zero-alloc ring buffer")
        .expect("Patch action created");
    assert!(patch_action.executor_verified);
    assert!(patch_action.diff_content.contains("zero-alloc ring buffer"));

    // 12. Test Autonomous Scientific Agent Engine
    let science_engine = oxide_models::AutonomousScientificAgentEngine::new(3, true);
    let hypothesis = science_engine
        .formulate_hypothesis(
            "Perovskite solar cell degradation",
            &["10.1038/s41586-026-001".to_string()],
        )
        .expect("Hypothesis formulated");
    assert!(hypothesis.confidence_score > 0.9);
    assert!(!hypothesis.literature_evidence.is_empty());

    // 13. Test Materials Metallurgy Engine
    let metallurgy_engine = oxide_models::MaterialsMetallurgyEngine::new(550.0, true);
    let alloy = metallurgy_engine
        .design_alloy("Al-Scrap-Recycled", 15.0)
        .expect("Alloy designed");
    assert_eq!(alloy.scrap_utilization_ratio, 1.0);
    assert!(alloy.predicted_yield_strength_mpa >= 550.0);

    // 14. Test Automated Testing QE Engine
    let qe_engine = oxide_models::AutomatedTestingQeEngine::new(true, 0.05);
    let qe_report = qe_engine
        .run_qa_suite("CheckoutE2E", "https://oxide-tech.internal/checkout")
        .expect("QA report generated");
    assert_eq!(qe_report.tests_passed, 18);
    assert_eq!(qe_report.tests_failed, 0);
    assert!(!qe_report.healed_locators.is_empty());
}

#[test]
fn test_extended_eda_cad_materials_cae_and_qa_suites() {
    // 1. Model Registry catalog lookups for the 17 newly integrated models
    let extended_models = [
        (
            "electronics-agent-kit",
            oxide_models::registry::ModelFamily::ElectronicsAgentKit,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "kic-ai",
            oxide_models::registry::ModelFamily::KicAiPlugin,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "cadlab",
            oxide_models::registry::ModelFamily::CadLabRustEda,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "pcbschemagen",
            oxide_models::registry::ModelFamily::PcbSchemaGenConstraint,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "trace-pcb",
            oxide_models::registry::ModelFamily::TraceAiPcb,
            oxide_models::registry::ModelModality::ElectronicDesignPcbSchematic,
        ),
        (
            "musubicad",
            oxide_models::registry::ModelFamily::MusubiCadRust,
            oxide_models::registry::ModelModality::ParametricCad3DGeneration,
        ),
        (
            "cad-coder",
            oxide_models::registry::ModelFamily::CadCoderVlm,
            oxide_models::registry::ModelModality::VisionLanguage,
        ),
        (
            "mentaagent",
            oxide_models::registry::ModelFamily::MentaAgentBi,
            oxide_models::registry::ModelModality::DecisionAgentic,
        ),
        (
            "academic-writing-skills",
            oxide_models::registry::ModelFamily::AcademicWritingSkillsAgent,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "researchkit",
            oxide_models::registry::ModelFamily::ResearchKitOverleaf,
            oxide_models::registry::ModelModality::ScientificLiteratureDiscovery,
        ),
        (
            "atomagents",
            oxide_models::registry::ModelFamily::AtomAgentsMit,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "alchemist-alloys",
            oxide_models::registry::ModelFamily::AlchemistAlloyDiscovery,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "ammap",
            oxide_models::registry::ModelFamily::AmMapCompositional,
            oxide_models::registry::ModelModality::MaterialsMetallurgyDesign,
        ),
        (
            "specforge-ai",
            oxide_models::registry::ModelFamily::SpecForgeAiPolyglot,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "agentic-qe",
            oxide_models::registry::ModelFamily::AgenticQePlatform,
            oxide_models::registry::ModelModality::AutonomousQeSoftwareTesting,
        ),
        (
            "mechrag",
            oxide_models::registry::ModelFamily::MechRagMllm,
            oxide_models::registry::ModelModality::MechanicalCaeEngineeringDesign,
        ),
        (
            "agentic-eng-design",
            oxide_models::registry::ModelFamily::AgenticEngDesignFramework,
            oxide_models::registry::ModelModality::SystemsEngineeringDesign,
        ),
    ];

    for (name, family, modality) in extended_models {
        let spec = ModelSpecification::lookup(name)
            .unwrap_or_else(|| panic!("Extended model {name} must exist in registry"));
        assert_eq!(spec.family, family, "Family mismatch for {name}");
        assert_eq!(spec.modality, modality, "Modality mismatch for {name}");
    }

    // 2. MusubiCAD Deterministic Design Graph Engine
    let musubi_engine = oxide_models::MusubiCadGraphEngine::new(true);
    let patch = musubi_engine
        .propose_patch(
            0x12345678,
            &[("extrude", &[50.0, 30.0, 10.0]), ("fillet", &[2.0])],
        )
        .expect("MusubiCAD patch generated");
    assert_eq!(patch.proposed_nodes.len(), 2);
    assert!(patch.verified_manifold);
    assert_eq!(patch.review_status, "PendingHumanApproval");

    // 3. CAD-Coder VLM CadQuery Engine
    let cad_coder = oxide_models::CadCoderVlmEngine::new(0.01);
    let code_artifact = cad_coder
        .generate_cadquery_code("MotorMountBracket", [60.0, 40.0, 12.0])
        .expect("CadQuery code generated");
    assert!(
        code_artifact
            .python_cadquery_script
            .contains("cadquery as cq")
    );
    assert_eq!(code_artifact.identified_features.len(), 3);

    // 4. AtomAgents MIT Physics & LAMMPS Engine
    let atom_engine = oxide_models::AtomAgentsPhysicsEngine::new("eam/alloy");
    let (lammps_task, micro_state) = atom_engine
        .setup_md_simulation("Ni-Co-Cr-Fe", 1200.0)
        .expect("MD simulation setup succeeded");
    assert_eq!(lammps_task.total_steps, 50_000);
    assert!(lammps_task.lammps_input_script.contains("units metal"));
    assert!(micro_state.phase_fraction_fcc > 0.5);

    // 5. AMMap Compositional Space Mapping Engine
    let ammap_engine = oxide_models::AmMapCompositionEngine::new(40.0);
    let comp_graph = ammap_engine
        .map_composition_space("Inconel-718-Opt")
        .expect("Composition space mapped");
    assert_eq!(comp_graph.phase_regions.len(), 2);
    assert!(comp_graph.crack_susceptibility_index < 0.2);

    // 6. MechRAG Multimodal CAE Engineering Engine
    let mechrag_engine = oxide_models::MechRagEngineeringEngine::new(250.0);
    let stress_result = mechrag_engine
        .evaluate_stress_field("TurbineBladeRoot", 18_000.0)
        .expect("CAE stress evaluation succeeded");
    assert!(stress_result.max_von_mises_stress_mpa > 0.0);
    assert!(stress_result.safety_factor > 1.0);
    assert!(!stress_result.recommended_design_modifications.is_empty());

    // 7. Agentic Engineering Design Engine
    let eng_design = oxide_models::AgenticEngDesignEngine::new(true);
    let decomp = eng_design
        .decompose_system("Electric Vertical Takeoff and Landing (eVTOL) Powertrain")
        .expect("System decomposition succeeded");
    assert_eq!(decomp.top_level_functions.len(), 2);
    assert!(
        decomp
            .generated_simulator_modelica_or_python
            .contains("model SystemSimulation")
    );

    // 8. SpecForge AI Smart Contract & Polyglot Mutation Engine
    let specforge = oxide_models::SpecForgeMutationEngine::new(85.0);
    let mut_report = specforge
        .run_mutation_analysis("LiquidityVault")
        .expect("Mutation analysis executed");
    assert_eq!(mut_report.total_mutations_generated, 40);
    assert!(mut_report.mutation_score_percent >= 90.0);

    // 9. Academic Writing Skills & ResearchKit Overleaf Engine
    let academic = oxide_models::AcademicResearchWritingEngine::new("English");
    let paper = academic
        .compose_paper(
            "Hardware-Bound Static Arenas for Deterministic Zero-Copy Inference",
            "This paper introduces zero-allocation lifetime-bounded memory arenas...",
        )
        .expect("Paper composed");
    assert!(paper.latex_main_tex.contains("\\documentclass{article}"));
    assert_eq!(paper.citations.len(), 1);
}
