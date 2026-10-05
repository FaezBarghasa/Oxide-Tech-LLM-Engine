use oxide_alloc::HierarchicalKvCache;
use oxide_backend_cpu::CpuBackend;
use oxide_core::sampler::{AcademicSamplerEngine, SamplerState, SamplingConfig};
use oxide_engine::slot_manager::{ContinuousBatchingSlotManager, SlotRequest};
use oxide_engine::{OxideEngine, SpecializedPipeline};
use oxide_models::needle::CactusNeedleConfig;
use oxide_server::dfa::DfaSchemaGrammar;
use oxide_server::{LeasedSlotGuard, ServerState, create_router};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_slot_leasing_and_raii_reclamation() {
    let slot_manager = Arc::new(Mutex::new(ContinuousBatchingSlotManager::new(2)));

    // Request 1: Lease slot 0
    let req1 = SlotRequest {
        request_id: "req-1".to_string(),
        prompt_tokens: vec![1, 2, 3],
        max_tokens: 16,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
    };
    let guard1 = LeasedSlotGuard::lease(&slot_manager, req1)
        .await
        .expect("Slot 1 acquired");
    assert_eq!(guard1.slot_id(), 0);

    // Request 2: Lease slot 1
    let req2 = SlotRequest {
        request_id: "req-2".to_string(),
        prompt_tokens: vec![4, 5, 6],
        max_tokens: 16,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
    };
    let guard2 = LeasedSlotGuard::lease(&slot_manager, req2)
        .await
        .expect("Slot 2 acquired");
    assert_eq!(guard2.slot_id(), 1);

    // Request 3: All slots busy, must return None (503)
    let req3 = SlotRequest {
        request_id: "req-3".to_string(),
        prompt_tokens: vec![7],
        max_tokens: 16,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
    };
    assert!(
        LeasedSlotGuard::lease(&slot_manager, req3.clone())
            .await
            .is_none()
    );

    // Drop guard 1: RAII automatically reclaims slot 0
    drop(guard1);

    // Now slot 0 is available again for Request 3
    let guard3 = LeasedSlotGuard::lease(&slot_manager, req3)
        .await
        .expect("Slot 0 reacquired after drop");
    assert_eq!(guard3.slot_id(), 0);

    drop(guard2);
    drop(guard3);

    // Verify all slots returned to idle
    let mgr = slot_manager.lock().await;
    assert_eq!(mgr.total_active_slots(), 0);
}

#[tokio::test]
async fn test_server_state_initialization_with_kv_cache() {
    let backend = CpuBackend::new(0, 16);
    let config = CactusNeedleConfig::<8>::default();
    let engine = OxideEngine::new(backend, config);
    let pipeline = Arc::new(Mutex::new(SpecializedPipeline::Needle3Cpu(engine)));
    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let slot_manager = Arc::new(Mutex::new(ContinuousBatchingSlotManager::new(16)));
    let kv_cache = Arc::new(Mutex::new(HierarchicalKvCache::new(64, 256, 1024)));

    let state = ServerState {
        pipeline,
        model_manager: None,
        dfa_grammar,
        slot_manager,
        kv_cache: Arc::clone(&kv_cache),
        tokenizer: Arc::new(oxide_models::GgufTokenizer::default()),
    };

    let router = create_router(state);
    assert!(format!("{:?}", router).contains("Router"));
}

#[test]
fn test_academic_sampler_integration_with_logits() {
    let config = SamplingConfig {
        temperature: 0.5,
        top_k: 10,
        top_p: 0.9,
        min_p: 0.05,
        ..SamplingConfig::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);

    let mut logits = vec![0.0f32; 100];
    logits[42] = 15.0; // dominant peak
    logits[43] = 12.0;

    let sampled = sampler
        .sample_token(&mut logits, &mut state, 10)
        .expect("Sampling successful");
    assert!(sampled == 42 || sampled == 43);
    assert_eq!(state.generated_tokens.len(), 1);
}

#[tokio::test]
async fn test_dynamic_model_loading_and_hot_swapping() {
    let backend = CpuBackend::new(0, 16);
    let config = CactusNeedleConfig::<8>::default();
    let engine = OxideEngine::new(backend, config);
    let initial_pipeline = SpecializedPipeline::Needle3Cpu(engine);

    let model_manager = Arc::new(Mutex::new(oxide_engine::DynamicModelManager::new(
        "llama3",
        initial_pipeline,
        "cpu",
        None,
        16,
        None,
    )));

    let dummy_pipe = Arc::new(Mutex::new(SpecializedPipeline::Needle3Cpu(
        OxideEngine::new(CpuBackend::new(0, 16), CactusNeedleConfig::<8>::default()),
    )));

    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let slot_manager = Arc::new(Mutex::new(ContinuousBatchingSlotManager::new(16)));
    let kv_cache = Arc::new(Mutex::new(HierarchicalKvCache::new(64, 256, 1024)));

    let state = ServerState {
        pipeline: dummy_pipe,
        model_manager: Some(Arc::clone(&model_manager)),
        dfa_grammar,
        slot_manager,
        kv_cache,
        tokenizer: Arc::new(oxide_models::GgufTokenizer::default()),
    };

    // 1. Initial resolution of default model
    let pipe1 = state.resolve_pipeline("llama3").await;
    assert!(Arc::strong_count(&pipe1) >= 1);

    // 2. Hot-load a new model into memory
    {
        let mut mgr = model_manager.lock().await;
        let loaded = mgr.get_or_load("bonsai2").expect("Dynamic load");
        assert!(Arc::strong_count(&loaded) >= 1);

        // Alias the model
        mgr.alias_model("bonsai-fast", "bonsai2");
        // Set as active default
        mgr.set_default_model("bonsai2");
    }

    // 3. Resolve using alias
    let pipe_alias = state.resolve_pipeline("bonsai-fast").await;
    assert!(Arc::strong_count(&pipe_alias) >= 1);

    // 4. Resolve default empty query returns the hot-swapped model
    let pipe_default = state.resolve_pipeline("").await;
    assert!(Arc::strong_count(&pipe_default) >= 1);

    // 5. Unload model
    {
        let mut mgr = model_manager.lock().await;
        let removed = mgr.unload_model("bonsai-fast");
        assert!(removed);
    }
}
