use oxide_backend_metal::MetalBackend;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;

#[test]
fn test_apple_silicon_m4_max_profile() {
    let profile = GpuDeviceProfile::from_known_device_name("Apple M4 Max").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::AppleSiliconM4);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::APPLE_GPU_FAMILY_10_M4
    );
    assert_eq!(profile.sm_count, 40);
    assert_eq!(profile.vram_capacity_bytes, 128 * 1024 * 1024 * 1024);
    assert!(profile.supports_fp8);
    assert_eq!(profile.form_factor, HardwareFormFactor::UnifiedAppleSiliconMac);
    assert_eq!(
        profile.tensor_core_gen,
        TensorCoreGeneration::AppleSimdgroupMatrixM4
    );
}

#[test]
fn test_apple_silicon_m2_ultra_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("Apple M2 Ultra 192GB").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::AppleSiliconM2);
    assert_eq!(profile.sm_count, 76);
    assert_eq!(profile.vram_capacity_bytes, 192 * 1024 * 1024 * 1024);
    assert_eq!(profile.memory_bandwidth_gbps, 800.0);
    assert!(profile.supports_nvlink); // UltraFusion
}

#[test]
fn test_metal_backend_dispatch_and_sync() {
    let mut backend = MetalBackend::new_with_profile(0, 32, Some("Apple M4 Max"));
    assert_eq!(
        backend.profile().architecture,
        GpuArchitecture::AppleSiliconM4
    );
    assert!(backend.execution_plan().simdgroup_matrix_enabled);
    assert!(backend.execution_plan().dynamic_caching_enabled);
    assert!(backend.execution_plan().zero_copy_unified_memory);

    let cmd = StepCommand::new(100, 42, 3, false);

    let event = backend
        .dispatch_step_kernel(&cmd)
        .expect("Kernel dispatched");
    assert!(backend.query_event_completed(event));
    let token = backend.read_sampled_token_host(3);
    assert_eq!(token, 43);
    assert!(backend.synchronize().is_ok());
}
