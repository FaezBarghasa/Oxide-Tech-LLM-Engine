use oxide_engine::{
    AcceleratorKind, ContinuousBatchingSlotManager, SlotRequest, SlotState, SpeculativeConfig,
    SpeculativeDecoderEngine, TensorSplitDistributionEngine, TensorSplitMode,
};

#[test]
fn test_speculative_decoding_verification() {
    let config = SpeculativeConfig {
        max_draft_tokens: 3,
        acceptance_threshold: 0.8,
        ..Default::default()
    };
    let engine = SpeculativeDecoderEngine::new(config);

    let draft_tokens = vec![101, 102, 103];
    let draft_probs = vec![0.9, 0.8, 0.4];
    let target_probs = vec![0.95, 0.85, 0.1]; // Token 103 rejected
    let target_recovery = vec![101, 102, 999];

    let result =
        engine.verify_draft_tokens(&draft_tokens, &draft_probs, &target_probs, &target_recovery);

    assert_eq!(result.accepted_tokens, vec![101, 102]);
    assert_eq!(result.num_accepted, 2);
    assert_eq!(result.recovery_token, Some(999));
}

#[test]
fn test_heterogeneous_tensor_splitting_and_numa() {
    let devices = vec![
        AcceleratorKind::CudaNvidia { device_id: 0 },
        AcceleratorKind::MetalAppleSilicon { device_id: 1 },
        AcceleratorKind::NumaCpuNode { numa_node_id: 0 },
    ];
    let splitter = TensorSplitDistributionEngine::new(devices, TensorSplitMode::ColumnParallel);

    let total_elements = 10000;
    let weights = vec![0.6, 0.3, 0.1]; // 60% GPU, 30% Metal, 10% CPU
    let slices = splitter.compute_tensor_slices(total_elements, &weights);

    assert_eq!(slices.len(), 3);
    assert_eq!(slices[0].element_count, 6000);
    assert_eq!(slices[1].element_count, 3000);
    assert_eq!(slices[2].element_count, 1000);

    let mut agg = vec![0.0; 4];
    let partials = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 0.5, 0.5, 0.5]];
    splitter.all_reduce_sum(&partials, &mut agg);
    assert_eq!(agg, vec![1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_continuous_batching_slot_manager() {
    let mut manager = ContinuousBatchingSlotManager::new(4);
    assert_eq!(manager.total_active_slots(), 0);

    let req = SlotRequest {
        request_id: "req-1".to_string(),
        prompt_tokens: vec![1, 2, 3, 4],
        max_tokens: 10,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
    };

    let slot_id = manager.submit_request(req).unwrap();
    assert_eq!(slot_id, 0);
    assert_eq!(manager.total_active_slots(), 1);

    let slot = manager.get_slot_mut(slot_id).unwrap();
    assert_eq!(slot.state, SlotState::Prefill);
    slot.append_token(42, false);
    assert_eq!(slot.state, SlotState::Decode);

    manager.release_slot(slot_id);
    assert_eq!(manager.total_active_slots(), 0);
}

#[test]
fn test_specialized_pipeline_llama3_dense_execution() {
    use oxide_core::worker::StepCommand;
    use oxide_engine::SpecializedPipeline;
    use oxide_models::llama3::{Llama3Config, Llama3KvCacheLayer, Llama3Model};

    let cfg = Llama3Config::tiny_test_config();
    let model = Llama3Model::new(cfg);
    let kv_cache = (0..cfg.num_layers)
        .map(|_| Llama3KvCacheLayer::default())
        .collect();

    let mut pipeline = SpecializedPipeline::Llama3Dense {
        model,
        kv_cache,
        seq_positions: std::collections::HashMap::new(),
    };

    let cmd = StepCommand::new(101, 42, 0, false);
    let completion = pipeline.step(&cmd).unwrap();
    assert_eq!(completion.sequence_id, 101);
    assert!(completion.sampled_token < cfg.vocab_size as u32);
}

#[test]
fn test_specialized_pipeline_llama3_cpu_monomorphized() {
    use oxide_backend_cpu::CpuBackend;
    use oxide_core::worker::StepCommand;
    use oxide_engine::{OxideEngine, SpecializedPipeline};
    use oxide_models::llama3::Llama3Config;

    let backend = CpuBackend::new(0, 4);
    let config = Llama3Config::tiny_test_config();
    let engine = OxideEngine::new(backend, config);
    let mut pipeline = SpecializedPipeline::Llama3Cpu(engine);

    let cmd = StepCommand::new(102, 10, 0, false);
    let completion = pipeline.step(&cmd).unwrap();
    assert_eq!(completion.sequence_id, 102);
    assert!(completion.sampled_token > 0);
}
