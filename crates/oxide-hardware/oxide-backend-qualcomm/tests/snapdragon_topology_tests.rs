use oxide_backend_qualcomm::QualcommBackend;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;

#[test]
fn test_snapdragon_x_elite_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("Snapdragon X Elite").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::QualcommHexagonNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::QUALCOMM_HEXAGON_V75_X_ELITE
    );
    assert_eq!(profile.vram_capacity_bytes, 64 * 1024 * 1024 * 1024);
    assert_eq!(profile.memory_bandwidth_gbps, 135.0);
    assert!(profile.supports_fp8);
    assert_eq!(
        profile.form_factor,
        HardwareFormFactor::UnifiedSnapdragonSoc
    );
    assert_eq!(
        profile.tensor_core_gen,
        TensorCoreGeneration::QualcommHexagonTensorProcessor
    );
}

#[test]
fn test_snapdragon_8_elite_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("Snapdragon 8 Elite").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::QualcommHexagonNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::QUALCOMM_HEXAGON_V79_8_ELITE
    );
    assert_eq!(profile.memory_bandwidth_gbps, 106.5);
}

#[test]
fn test_snapdragon_backend_dispatch_and_sync() {
    let mut backend = QualcommBackend::new_with_profile(0, 16, Some("Snapdragon X Elite"));
    assert_eq!(
        backend.profile().architecture,
        GpuArchitecture::QualcommHexagonNpu
    );
    assert_eq!(backend.execution_plan().peak_npu_tops, 45.0);
    assert!(backend.execution_plan().zero_copy_ion_shared_memory);

    let cmd = StepCommand::new(1, 100, 1, false);

    match backend.dispatch_step_kernel(&cmd) {
        Ok(event) => {
            assert!(backend.query_event_completed(event));
            let token = backend.read_sampled_token_host(1);
            assert_eq!(token, 101);
            assert!(backend.synchronize().is_ok());
        }
        Err(oxide_core::error::EngineError::DeviceNotFound { .. }) => {
            // Clean typed error when physical Qualcomm Hexagon NPU is not present
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
