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
fn test_cuda_multi_gpu_16_array_tensor_splitting() {
    // Array of 16 CUDA GPUs
    let devices: Vec<AcceleratorKind> = (0..16)
        .map(|id| AcceleratorKind::CudaNvidia { device_id: id })
        .collect();
    assert_eq!(devices.len(), 16);

    let splitter = TensorSplitDistributionEngine::new(devices, TensorSplitMode::RowParallel);

    // Distribute a large weight matrix (e.g. 16,384,000 elements) evenly across 16 CUDA GPUs
    let total_elements = 16_384_000;
    let weights = vec![1.0f32; 16]; // Equal weighting across 16 GPUs
    let slices = splitter.compute_tensor_slices(total_elements, &weights);

    assert_eq!(slices.len(), 16);
    let expected_slice_size = 16_384_000 / 16;
    for (i, slice) in slices.iter().enumerate() {
        assert_eq!(
            slice.accelerator,
            AcceleratorKind::CudaNvidia {
                device_id: i as u32
            }
        );
        assert_eq!(slice.split_mode, TensorSplitMode::RowParallel);
        assert_eq!(slice.slice_index, i);
        assert_eq!(slice.total_slices, 16);
        assert_eq!(slice.start_offset, i * expected_slice_size);
        assert_eq!(slice.element_count, expected_slice_size);
    }

    // Distributed AllReduce Sum across 16 CUDA devices
    let mut partials = Vec::with_capacity(16);
    for rank in 0..16 {
        partials.push(vec![rank as f32 + 1.0; 8]);
    }
    let mut reduced = vec![0.0f32; 8];
    splitter.all_reduce_sum(&partials, &mut reduced);
    // Sum of 1..=16 is 136.0
    for val in reduced {
        assert_eq!(val, 136.0);
    }
}

#[test]
fn test_all_hardware_heterogeneous_topologies() {
    use oxide_engine::hybrid::{DeviceRole, HybridDeviceTopology};

    // 1. CPU + NVIDIA GPUs
    let top_cpu_nv = HybridDeviceTopology::cpu_nvidia_partition(32, 2, 4);
    assert_eq!(top_cpu_nv.partitions.len(), 3); // 2 NVIDIA + 1 CPU
    assert_eq!(top_cpu_nv.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_cpu_nv.partitions[1].device, DeviceRole::NvidiaGpu(1));
    assert_eq!(top_cpu_nv.partitions[2].device, DeviceRole::Cpu);

    // 2. CPU + AMD GPUs
    let top_cpu_amd = HybridDeviceTopology::cpu_amd_partition(32, 2, 4);
    assert_eq!(top_cpu_amd.partitions.len(), 3); // 2 AMD + 1 CPU
    assert_eq!(top_cpu_amd.partitions[0].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_cpu_amd.partitions[2].device, DeviceRole::Cpu);

    // 3. CPU + Intel GPUs
    let top_cpu_intel = HybridDeviceTopology::cpu_intel_partition(32, 2, 4);
    assert_eq!(top_cpu_intel.partitions.len(), 3); // 2 Intel + 1 CPU
    assert_eq!(top_cpu_intel.partitions[0].device, DeviceRole::IntelGpu(0));
    assert_eq!(top_cpu_intel.partitions[2].device, DeviceRole::Cpu);

    // 4. CPU + Google TPU
    let top_cpu_tpu = HybridDeviceTopology::cpu_tpu_partition(32, 4, 4);
    assert_eq!(top_cpu_tpu.partitions.len(), 5); // 4 TPU + 1 CPU
    assert_eq!(top_cpu_tpu.partitions[0].device, DeviceRole::Tpu(0));
    assert_eq!(top_cpu_tpu.partitions[4].device, DeviceRole::Cpu);

    // 5. CPU + NPU
    let top_cpu_npu = HybridDeviceTopology::cpu_npu_partition(32, 0.65);
    assert_eq!(top_cpu_npu.partitions.len(), 2);
    assert_eq!(top_cpu_npu.partitions[0].device, DeviceRole::Npu);
    assert_eq!(top_cpu_npu.partitions[1].device, DeviceRole::Cpu);

    // 6. CPU + NVIDIA + AMD + Intel GPUs (Triple-Vendor Array)
    let top_triple_gpu = HybridDeviceTopology::cpu_nvidia_amd_intel_partition(32, 1, 1, 1, 5);
    assert_eq!(top_triple_gpu.partitions.len(), 4); // 1 NV + 1 AMD + 1 Intel + 1 CPU
    assert_eq!(
        top_triple_gpu.partitions[0].device,
        DeviceRole::NvidiaGpu(0)
    );
    assert_eq!(top_triple_gpu.partitions[1].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_triple_gpu.partitions[2].device, DeviceRole::IntelGpu(0));
    assert_eq!(top_triple_gpu.partitions[3].device, DeviceRole::Cpu);

    // 7. CPU + AMD + Intel GPUs
    let top_amd_intel = HybridDeviceTopology::cpu_amd_intel_partition(32, 1, 1, 4);
    assert_eq!(top_amd_intel.partitions.len(), 3);
    assert_eq!(top_amd_intel.partitions[0].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_amd_intel.partitions[1].device, DeviceRole::IntelGpu(0));
    assert_eq!(top_amd_intel.partitions[2].device, DeviceRole::Cpu);

    // 8. CPU + NVIDIA + Intel GPUs
    let top_nv_intel = HybridDeviceTopology::cpu_nvidia_intel_partition(32, 1, 1, 4);
    assert_eq!(top_nv_intel.partitions.len(), 3);
    assert_eq!(top_nv_intel.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_nv_intel.partitions[1].device, DeviceRole::IntelGpu(0));
    assert_eq!(top_nv_intel.partitions[2].device, DeviceRole::Cpu);

    // 9. CPU + NVIDIA + AMD GPUs
    let top_nv_amd = HybridDeviceTopology::cpu_nvidia_amd_partition(32, 1, 1, 4);
    assert_eq!(top_nv_amd.partitions.len(), 3);
    assert_eq!(top_nv_amd.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_nv_amd.partitions[1].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_nv_amd.partitions[2].device, DeviceRole::Cpu);

    // 10. CPU + iGPU + NPU (AMD APU Tri-Compute)
    let top_apu = HybridDeviceTopology::cpu_igpu_npu_partition(32);
    assert_eq!(top_apu.partitions.len(), 3);
    assert_eq!(top_apu.partitions[0].device, DeviceRole::Cpu);
    assert_eq!(top_apu.partitions[1].device, DeviceRole::Igpu);
    assert_eq!(top_apu.partitions[2].device, DeviceRole::Npu);

    // 11. CPU + iGPU + TPU
    let top_igpu_tpu = HybridDeviceTopology::cpu_igpu_tpu_partition(32, 2);
    assert_eq!(top_igpu_tpu.partitions.len(), 4); // 2 TPU + 1 iGPU + 1 CPU
    assert_eq!(top_igpu_tpu.partitions[0].device, DeviceRole::Tpu(0));
    assert_eq!(top_igpu_tpu.partitions[2].device, DeviceRole::Igpu);
    assert_eq!(top_igpu_tpu.partitions[3].device, DeviceRole::Cpu);

    // 12. CPU + iGPU + NPU + NVIDIA GPUs
    let top_apu_nv = HybridDeviceTopology::cpu_igpu_npu_nvidia_partition(32, 1);
    assert_eq!(top_apu_nv.partitions.len(), 4); // NVIDIA + NPU + iGPU + CPU
    assert_eq!(top_apu_nv.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_apu_nv.partitions[1].device, DeviceRole::Npu);
    assert_eq!(top_apu_nv.partitions[2].device, DeviceRole::Igpu);
    assert_eq!(top_apu_nv.partitions[3].device, DeviceRole::Cpu);

    // 13. CPU + iGPU + NPU + AMD GPUs
    let top_apu_amd = HybridDeviceTopology::cpu_igpu_npu_amd_partition(32, 1);
    assert_eq!(top_apu_amd.partitions.len(), 4); // AMD + NPU + iGPU + CPU
    assert_eq!(top_apu_amd.partitions[0].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_apu_amd.partitions[1].device, DeviceRole::Npu);
    assert_eq!(top_apu_amd.partitions[2].device, DeviceRole::Igpu);
    assert_eq!(top_apu_amd.partitions[3].device, DeviceRole::Cpu);

    // 14. ARM CPU + Integrated NPU + External NPU HAT (e.g. Raspberry Pi 5 + Hailo-8)
    let top_arm_hat = HybridDeviceTopology::arm_npu_external_hat_partition(32);
    assert_eq!(top_arm_hat.partitions.len(), 3);
    assert_eq!(top_arm_hat.partitions[0].device, DeviceRole::ExternalNpuHat);
    assert_eq!(
        top_arm_hat.partitions[1].device,
        DeviceRole::ArmIntegratedNpu
    );
    assert_eq!(top_arm_hat.partitions[2].device, DeviceRole::Cpu);

    // 15. ARM CPU + Integrated NPU (no external HAT)
    let top_arm_npu = HybridDeviceTopology::arm_integrated_npu_partition(32);
    assert_eq!(top_arm_npu.partitions.len(), 2);
    assert_eq!(
        top_arm_npu.partitions[0].device,
        DeviceRole::ArmIntegratedNpu
    );
    assert_eq!(top_arm_npu.partitions[1].device, DeviceRole::Cpu);

    // 16. AMD EPYC Server CPU (8 to 128 cores per socket, pure CPU cluster)
    let top_epyc = HybridDeviceTopology::epyc_server_standalone_partition(64, 4);
    assert_eq!(top_epyc.partitions.len(), 4); // 4 sockets
    assert_eq!(top_epyc.partitions[0].device, DeviceRole::EpycServer(0));
    assert_eq!(top_epyc.partitions[3].device, DeviceRole::EpycServer(3));
    assert_eq!(top_epyc.partitions[3].end_layer, 64);

    // 17. AMD EPYC Server CPU + Multi-GPU (e.g. Dual EPYC + 4 NVIDIA GPUs)
    let gpus = vec![
        DeviceRole::NvidiaGpu(0),
        DeviceRole::NvidiaGpu(1),
        DeviceRole::NvidiaGpu(2),
        DeviceRole::NvidiaGpu(3),
    ];
    let top_epyc_gpu = HybridDeviceTopology::epyc_server_gpu_partition(80, 2, &gpus);
    assert_eq!(top_epyc_gpu.partitions.len(), 6); // 4 GPUs + 2 EPYC sockets
    assert_eq!(top_epyc_gpu.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_epyc_gpu.partitions[3].device, DeviceRole::NvidiaGpu(3));
    assert_eq!(top_epyc_gpu.partitions[4].device, DeviceRole::EpycServer(0));
    assert_eq!(top_epyc_gpu.partitions[5].device, DeviceRole::EpycServer(1));
    assert_eq!(top_epyc_gpu.partitions[5].end_layer, 80);

    // 18. Apple Silicon UMA (CPU + Metal GPU + Apple Neural Engine ANE)
    let top_apple = HybridDeviceTopology::apple_silicon_uma_partition(32);
    assert_eq!(top_apple.partitions.len(), 3);
    assert_eq!(top_apple.partitions[0].device, DeviceRole::Igpu);
    assert_eq!(top_apple.partitions[1].device, DeviceRole::ArmIntegratedNpu);
    assert_eq!(top_apple.partitions[2].device, DeviceRole::Cpu);

    // 19. Qualcomm Snapdragon (CPU + Adreno GPU + Hexagon HTP NPU)
    let top_snap = HybridDeviceTopology::qualcomm_snapdragon_partition(32);
    assert_eq!(top_snap.partitions.len(), 3);
    assert_eq!(top_snap.partitions[0].device, DeviceRole::ArmIntegratedNpu);
    assert_eq!(top_snap.partitions[1].device, DeviceRole::Igpu);
    assert_eq!(top_snap.partitions[2].device, DeviceRole::Cpu);

    // 20. Intel Core Ultra (CPU + Xe iGPU + Intel NPU + Arc dGPU)
    let top_intel_ultra = HybridDeviceTopology::intel_core_ultra_partition(32, 1);
    assert_eq!(top_intel_ultra.partitions.len(), 4);
    assert_eq!(
        top_intel_ultra.partitions[0].device,
        DeviceRole::IntelGpu(0)
    );
    assert_eq!(top_intel_ultra.partitions[1].device, DeviceRole::Npu);
    assert_eq!(top_intel_ultra.partitions[2].device, DeviceRole::Igpu);
    assert_eq!(top_intel_ultra.partitions[3].device, DeviceRole::Cpu);

    // 21. Raspberry Pi 5 + Hailo AI HAT+
    let top_rpi_hailo = HybridDeviceTopology::raspberry_pi_hailo_partition(32);
    assert_eq!(top_rpi_hailo.partitions.len(), 2);
    assert_eq!(
        top_rpi_hailo.partitions[0].device,
        DeviceRole::ExternalNpuHat
    );
    assert_eq!(top_rpi_hailo.partitions[1].device, DeviceRole::Cpu);

    // 22. Distributed Multi-Node Cluster
    let cluster_nodes = [
        DeviceRole::NvidiaGpu(0),
        DeviceRole::AmdGpu(0),
        DeviceRole::Tpu(0),
        DeviceRole::EpycServer(0),
    ];
    let top_cluster = HybridDeviceTopology::distributed_cluster_partition(64, &cluster_nodes);
    assert_eq!(top_cluster.partitions.len(), 4);
    assert_eq!(top_cluster.partitions[0].device, DeviceRole::NvidiaGpu(0));
    assert_eq!(top_cluster.partitions[1].device, DeviceRole::AmdGpu(0));
    assert_eq!(top_cluster.partitions[2].device, DeviceRole::Tpu(0));
    assert_eq!(top_cluster.partitions[3].device, DeviceRole::EpycServer(0));
    assert_eq!(top_cluster.partitions[3].end_layer, 64);

    // Verify contiguity and complete coverage for all topologies
    let all_topologies = [
        &top_cpu_nv,
        &top_cpu_amd,
        &top_cpu_intel,
        &top_cpu_tpu,
        &top_cpu_npu,
        &top_triple_gpu,
        &top_amd_intel,
        &top_nv_intel,
        &top_nv_amd,
        &top_apu,
        &top_igpu_tpu,
        &top_apu_nv,
        &top_apu_amd,
        &top_arm_hat,
        &top_arm_npu,
        &top_epyc,
        &top_epyc_gpu,
        &top_apple,
        &top_snap,
        &top_intel_ultra,
        &top_rpi_hailo,
        &top_cluster,
    ];

    for top in all_topologies {
        assert_eq!(top.partitions[0].start_layer, 0);
        assert_eq!(top.partitions.last().unwrap().end_layer, top.total_layers);
        for i in 0..top.partitions.len() - 1 {
            assert_eq!(
                top.partitions[i].end_layer,
                top.partitions[i + 1].start_layer
            );
        }

        // Test forward step execution on each topology
        let mut pipeline =
            oxide_engine::hybrid::HybridMultiDevicePipeline::new((*top).clone(), 128);
        let cmd = oxide_core::worker::StepCommand::new(505, 12, 0, false);
        let comp = pipeline.step_hybrid(&cmd).unwrap();
        assert_eq!(comp.sequence_id, 505);
        assert_eq!(comp.sampled_token, 13);
    }
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

    let scratch = model.create_scratch();
    let mut pipeline = SpecializedPipeline::Llama3Dense {
        model,
        kv_cache,
        seq_positions: std::collections::HashMap::new(),
        scratch: Box::new(scratch),
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
