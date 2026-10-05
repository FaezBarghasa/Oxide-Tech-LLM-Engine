use oxide_backend_rocm::{RocmBackend, RocmExecutionPlan};
use oxide_core::hardware::{
    GpuArchitecture, GpuDeviceProfile, MemoryTechnology, TensorCoreGeneration,
};

#[test]
fn test_amd_instinct_mi300x_and_mi325x() {
    let mi300x =
        GpuDeviceProfile::from_known_device_name("AMD Instinct MI300X").expect("MI300X profile");
    assert_eq!(mi300x.architecture, GpuArchitecture::Cdna3);
    assert_eq!(mi300x.vram_capacity_bytes, 192 * 1024 * 1024 * 1024);
    assert_eq!(mi300x.memory_tech, MemoryTechnology::Hbm3);
    assert_eq!(mi300x.tensor_core_gen, TensorCoreGeneration::AmdMfmaCdna3);

    let plan_300 = RocmExecutionPlan::for_profile(&mi300x);
    assert_eq!(plan_300.wavefront_size, 64);
    assert!(plan_300.use_mfma_cdna);
    assert!(!plan_300.use_wmma_rdna);
    assert!(plan_300.use_fp8_mfma);
    assert!(!plan_300.use_fp4_mfma);

    let mi325x = GpuDeviceProfile::from_known_device_name("MI325X").expect("MI325X profile");
    assert_eq!(mi325x.architecture, GpuArchitecture::Cdna3);
    assert_eq!(mi325x.vram_capacity_bytes, 256 * 1024 * 1024 * 1024);
    assert_eq!(mi325x.memory_tech, MemoryTechnology::Hbm3e);
}

#[test]
fn test_amd_instinct_mi350x_and_mi355x_cdna4() {
    let mi350x = GpuDeviceProfile::from_known_device_name("MI350X").expect("MI350X profile");
    assert_eq!(mi350x.architecture, GpuArchitecture::Cdna4);
    assert_eq!(mi350x.vram_capacity_bytes, 288 * 1024 * 1024 * 1024);
    assert_eq!(mi350x.memory_tech, MemoryTechnology::Hbm3e);
    assert_eq!(mi350x.tensor_core_gen, TensorCoreGeneration::AmdMfmaCdna4);

    let plan_350 = RocmExecutionPlan::for_profile(&mi350x);
    assert_eq!(plan_350.wavefront_size, 64);
    assert!(plan_350.use_mfma_cdna);
    assert!(plan_350.use_fp8_mfma);
    assert!(plan_350.use_fp4_mfma); // CDNA 4 FP4 MFMA

    let mi355x = GpuDeviceProfile::from_known_device_name("MI355X").expect("MI355X profile");
    assert_eq!(mi355x.architecture, GpuArchitecture::Cdna4);
}

#[test]
fn test_amd_commercial_radeon_rx_and_pro() {
    let rx7900 =
        GpuDeviceProfile::from_known_device_name("Radeon RX 7900 XTX").expect("RX 7900 XTX");
    assert_eq!(rx7900.architecture, GpuArchitecture::Rdna3);
    assert_eq!(rx7900.vram_capacity_bytes, 24 * 1024 * 1024 * 1024);
    assert_eq!(rx7900.tensor_core_gen, TensorCoreGeneration::AmdRdnaWmma);

    let plan_rx = RocmExecutionPlan::for_profile(&rx7900);
    assert_eq!(plan_rx.wavefront_size, 32); // Wave32 for RDNA
    assert!(!plan_rx.use_mfma_cdna);
    assert!(plan_rx.use_wmma_rdna);

    let pro_w7900 =
        GpuDeviceProfile::from_known_device_name("Radeon PRO W7900").expect("PRO W7900");
    assert_eq!(pro_w7900.vram_capacity_bytes, 48 * 1024 * 1024 * 1024);
}

#[test]
fn test_amd_apu_with_npu_strix_point_and_hawk_point() {
    let strix =
        GpuDeviceProfile::from_known_device_name("Ryzen AI 9 HX 370").expect("Strix Point APU");
    assert_eq!(strix.architecture, GpuArchitecture::Rdna3_5);
    assert_eq!(strix.memory_tech, MemoryTechnology::UnifiedLpddr5X);
    assert_eq!(
        strix.tensor_core_gen,
        TensorCoreGeneration::AmdXdnaNpuEngine
    );

    let plan_strix = RocmExecutionPlan::for_profile(&strix);
    assert_eq!(plan_strix.wavefront_size, 32);
    assert!(plan_strix.use_wmma_rdna);
    assert!(plan_strix.use_xdna_npu_tile); // XDNA 2 NPU tile offload
    assert!(plan_strix.use_fp4_mfma); // XDNA 2 FP4 block quantization

    let hawk = GpuDeviceProfile::from_known_device_name("Ryzen 7 8840HS").expect("Hawk Point APU");
    assert_eq!(hawk.tensor_core_gen, TensorCoreGeneration::AmdXdnaNpuEngine);
}

#[test]
fn test_rocm_backend_instantiation_and_kernel_dispatch() {
    use oxide_core::traits::HardwareBackend;
    use oxide_core::worker::StepCommand;

    let mut backend = RocmBackend::new_with_profile(0, 16, Some("MI300X"));
    assert_eq!(backend.profile().architecture, GpuArchitecture::Cdna3);

    let cmd = StepCommand::new(1, 42, 2, false);

    let event = backend.dispatch_step_kernel(&cmd).expect("dispatch");
    assert!(backend.query_event_completed(event));
    let token = backend.read_sampled_token_host(2);
    assert_eq!(token, 43);
}
