use oxide_backend_rknn::RknnBackend;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;

#[test]
fn test_orange_pi_6_plus_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("Orange Pi 6 Plus").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::RockchipRknnNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::ROCKCHIP_RKNN_RK3588
    );
    assert_eq!(profile.sm_count, 3);
    assert_eq!(profile.vram_capacity_bytes, 32 * 1024 * 1024 * 1024);
    assert_eq!(
        profile.form_factor,
        HardwareFormFactor::SingleBoardComputerAiHat
    );
    assert_eq!(
        profile.tensor_core_gen,
        TensorCoreGeneration::RockchipRknnNpuCore
    );
}

#[test]
fn test_rockchip_rk3576_profile() {
    let profile = GpuDeviceProfile::from_known_device_name("RK3576").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::RockchipRknnNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::ROCKCHIP_RKNN_RK3576
    );
    assert_eq!(profile.sm_count, 2);
}

#[test]
fn test_rknn_backend_dispatch_and_sync() {
    let mut backend = RknnBackend::new_with_profile(0, 8, Some("Orange Pi 6 Plus"));
    assert_eq!(
        backend.profile().architecture,
        GpuArchitecture::RockchipRknnNpu
    );
    assert_eq!(backend.execution_plan().peak_npu_tops, 6.0);
    assert!(backend.execution_plan().dma_buf_zero_copy);

    let cmd = StepCommand::new(5, 77, 0, false);

    let event = backend
        .dispatch_step_kernel(&cmd)
        .expect("Kernel dispatched");
    assert!(backend.query_event_completed(event));
    let token = backend.read_sampled_token_host(0);
    assert_eq!(token, 78);
    assert!(backend.synchronize().is_ok());
}
