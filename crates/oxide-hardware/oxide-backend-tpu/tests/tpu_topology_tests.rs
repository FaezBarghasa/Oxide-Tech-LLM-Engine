use oxide_backend_tpu::{TpuBackend, TpuExecutionPlan, TpuIciCommunicator};
use oxide_core::hardware::{
    GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology, TensorCoreGeneration,
};

#[test]
fn test_google_tpu_v6e_trillium() {
    let tpu = GpuDeviceProfile::from_known_device_name("Google TPU v6e Trillium")
        .expect("TPU v6e profile");
    assert_eq!(tpu.architecture, GpuArchitecture::GoogleTpuV6eTrillium);
    assert_eq!(tpu.vram_capacity_bytes, 32 * 1024 * 1024 * 1024);
    assert_eq!(tpu.memory_tech, MemoryTechnology::Hbm3);
    assert_eq!(
        tpu.tensor_core_gen,
        TensorCoreGeneration::GoogleTpuMxuV6Trillium
    );
    assert_eq!(tpu.compute_capability.to_string(), "tpu_v6e_trillium");
    assert!(tpu.supports_fp8);
    assert!(tpu.supports_tma);
    assert_eq!(tpu.nvlink_bandwidth_gbps, 3200.0); // 3.2 Tbps ICI

    let plan = TpuExecutionPlan::for_profile(&tpu);
    assert_eq!(plan.mxu_tile_dim, 128);
    assert_eq!(plan.mxu_count_per_core, 8);
    assert!(plan.use_sparsecore);
    assert!(plan.use_fp8_mxu);
    assert!(plan.use_bf16_mxu);
    assert!(plan.use_int8_systolic);
}

#[test]
fn test_google_tpu_v5p_and_v5e() {
    let v5p = GpuDeviceProfile::from_known_device_name("TPU v5p").expect("TPU v5p profile");
    assert_eq!(v5p.architecture, GpuArchitecture::GoogleTpuV5p);
    assert_eq!(v5p.vram_capacity_bytes, 95 * 1024 * 1024 * 1024);
    assert_eq!(v5p.memory_tech, MemoryTechnology::Hbm2e);
    assert_eq!(v5p.form_factor, HardwareFormFactor::DatacenterTpuPod3dTorus);
    assert_eq!(v5p.nvlink_bandwidth_gbps, 4800.0); // 4.8 Tbps 3D Torus

    let plan_v5p = TpuExecutionPlan::for_profile(&v5p);
    assert_eq!(plan_v5p.mxu_count_per_core, 4);
    assert!(plan_v5p.use_sparsecore);
    assert!(plan_v5p.use_fp8_mxu);

    let v5e =
        GpuDeviceProfile::from_known_device_name("TPU v5e ViperLite").expect("TPU v5e profile");
    assert_eq!(v5e.architecture, GpuArchitecture::GoogleTpuV5e);
    assert_eq!(v5e.vram_capacity_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(v5e.memory_tech, MemoryTechnology::Hbm2);
    assert_eq!(v5e.form_factor, HardwareFormFactor::DatacenterTpuPod2dTorus);
    assert_eq!(v5e.nvlink_bandwidth_gbps, 1600.0);
}

#[test]
fn test_google_tpu_v4_and_v3() {
    let v4 = GpuDeviceProfile::from_known_device_name("TPU v4").expect("TPU v4 profile");
    assert_eq!(v4.architecture, GpuArchitecture::GoogleTpuV4);
    assert_eq!(v4.vram_capacity_bytes, 32 * 1024 * 1024 * 1024);
    assert_eq!(v4.tensor_core_gen, TensorCoreGeneration::GoogleTpuMxuV4);
    assert_eq!(v4.form_factor, HardwareFormFactor::DatacenterTpuPod3dTorus);

    let plan_v4 = TpuExecutionPlan::for_profile(&v4);
    assert_eq!(plan_v4.mxu_count_per_core, 4);
    assert!(plan_v4.use_sparsecore);
    assert!(!plan_v4.use_fp8_mxu);

    let v3 = GpuDeviceProfile::from_known_device_name("TPU v3 32GB").expect("TPU v3 profile");
    assert_eq!(v3.architecture, GpuArchitecture::GoogleTpuV3);
    assert_eq!(v3.vram_capacity_bytes, 32 * 1024 * 1024 * 1024);
    assert_eq!(v3.tensor_core_gen, TensorCoreGeneration::GoogleTpuMxuV3);
}

#[test]
fn test_google_tpu_v2_and_edge_coral() {
    let v2 = GpuDeviceProfile::from_known_device_name("TPU v2").expect("TPU v2 profile");
    assert_eq!(v2.architecture, GpuArchitecture::GoogleTpuV2);
    assert_eq!(v2.vram_capacity_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(v2.memory_tech, MemoryTechnology::Hbm);

    let coral =
        GpuDeviceProfile::from_known_device_name("Coral Edge TPU").expect("Coral Edge TPU profile");
    assert_eq!(coral.architecture, GpuArchitecture::GoogleEdgeTpu);
    assert_eq!(coral.vram_capacity_bytes, 8 * 1024 * 1024);
    assert_eq!(coral.memory_tech, MemoryTechnology::SramOnChip);
    assert_eq!(
        coral.tensor_core_gen,
        TensorCoreGeneration::GoogleEdgeTpuInt8Engine
    );

    let plan_coral = TpuExecutionPlan::for_profile(&coral);
    assert_eq!(plan_coral.mxu_tile_dim, 64);
    assert!(plan_coral.use_int8_systolic);
    assert!(!plan_coral.use_bf16_mxu);
}

#[test]
fn test_tpu_backend_dispatch_and_ici_comm() {
    use oxide_core::traits::HardwareBackend;
    use oxide_core::worker::StepCommand;

    let mut backend = TpuBackend::new_with_profile(0, 16, Some("TPU v6e"));
    assert_eq!(
        backend.profile().architecture,
        GpuArchitecture::GoogleTpuV6eTrillium
    );

    let cmd = StepCommand::new(1, 100, 3, false);
    match backend.dispatch_step_kernel(&cmd) {
        Ok(event) => {
            assert!(backend.query_event_completed(event));
            let token = backend.read_sampled_token_host(3);
            assert_eq!(token, 101);
        }
        Err(oxide_core::error::EngineError::DeviceNotFound) => {
            // Clean typed error when physical Google Cloud TPU is not present
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }

    let ici = TpuIciCommunicator::new(0, 256, 3);
    assert_eq!(ici.rank(), 0);
    assert_eq!(ici.world_size(), 256);
    assert_eq!(ici.torus_dimension(), 3);
}
