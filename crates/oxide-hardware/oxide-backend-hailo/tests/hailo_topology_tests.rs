use oxide_backend_hailo::HailoBackend;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;

#[test]
fn test_rpi5_ai_hat_plus_profile() {
    let profile = GpuDeviceProfile::from_known_device_name("RPi5 with AI HAT+ 26 TOPS")
        .expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::HailoNpu);
    assert_eq!(profile.compute_capability, ComputeCapability::HAILO_8_26TOPS);
    assert_eq!(
        profile.form_factor,
        HardwareFormFactor::SingleBoardComputerAiHat
    );
    assert_eq!(
        profile.tensor_core_gen,
        TensorCoreGeneration::Hailo8SystolicNpuEngine
    );
}

#[test]
fn test_rpi5_ai_hat_plus_2_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("ai-hat-plus-2").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::HailoNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::HAILO_10_40TOPS
    );
    assert_eq!(
        profile.tensor_core_gen,
        TensorCoreGeneration::Hailo10GenAiEngine
    );
    assert_eq!(profile.vram_capacity_bytes, 16 * 1024 * 1024 * 1024);
}

#[test]
fn test_external_edge_tpu_profile() {
    let profile =
        GpuDeviceProfile::from_known_device_name("Coral Edge TPU USB").expect("Profile found");
    assert_eq!(profile.architecture, GpuArchitecture::ExternalEdgeNpu);
    assert_eq!(
        profile.compute_capability,
        ComputeCapability::EXTERNAL_EDGE_ACCELERATOR
    );
    assert_eq!(
        profile.form_factor,
        HardwareFormFactor::ExternalUsbAccelerator
    );
}

#[test]
fn test_hailo_backend_dispatch_and_sync() {
    let mut backend =
        HailoBackend::new_with_profile(0, 8, Some("RPi5 with AI HAT+ 26 TOPS (Hailo-8)"));
    assert_eq!(backend.profile().architecture, GpuArchitecture::HailoNpu);
    assert_eq!(backend.execution_plan().peak_npu_tops, 26.0);
    assert!(backend.execution_plan().structural_sparsity_enabled);

    let cmd = StepCommand {
        slot_idx: 2,
        seq_id: 20,
        input_token: 250,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
    };

    let event = backend
        .dispatch_step_kernel(&cmd)
        .expect("Kernel dispatched");
    assert!(backend.query_event_completed(event));
    let token = backend.read_sampled_token_host(2);
    assert_eq!(token, 251);
    assert!(backend.synchronize().is_ok());
}
