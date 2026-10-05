use oxide_backend_intel::{IntelBackend, IntelExecutionPlan, LevelZeroCommunicator};
use oxide_core::hardware::{
    GpuArchitecture, GpuDeviceProfile, MemoryTechnology,
    TensorCoreGeneration,
};

#[test]
fn test_intel_arc_b_series_battlemage() {
    let b580 = GpuDeviceProfile::from_known_device_name("Intel Arc B580").expect("Arc B580");
    assert_eq!(b580.architecture, GpuArchitecture::IntelXe2Battlemage);
    assert_eq!(b580.vram_capacity_bytes, 12 * 1024 * 1024 * 1024);
    assert_eq!(b580.memory_bandwidth_gbps, 456.0);
    assert_eq!(b580.sm_count, 20); // 20 Xe2-cores
    assert_eq!(
        b580.tensor_core_gen,
        TensorCoreGeneration::IntelXmxGen2Battlemage
    );
    assert!(b580.supports_fp8); // Gen2 XMX FP8 support

    let plan_b580 = IntelExecutionPlan::for_profile(&b580);
    assert!(plan_b580.use_xmx_matrix_engine);
    assert!(plan_b580.use_fp8_xmx);
    assert!(!plan_b580.use_amx_tile_engine);

    let pro_b60 = GpuDeviceProfile::from_known_device_name("Intel Arc Pro B60").expect("Arc Pro B60");
    assert_eq!(pro_b60.vram_capacity_bytes, 24 * 1024 * 1024 * 1024);
    assert_eq!(pro_b60.architecture, GpuArchitecture::IntelXe2Battlemage);
}

#[test]
fn test_intel_arc_alchemist_and_ponte_vecchio() {
    let a770 = GpuDeviceProfile::from_known_device_name("Intel Arc A770").expect("Arc A770");
    assert_eq!(a770.architecture, GpuArchitecture::IntelXe1Alchemist);
    assert_eq!(a770.vram_capacity_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(a770.sm_count, 32);

    let pvc = GpuDeviceProfile::from_known_device_name("Intel Data Center GPU Max 1550")
        .expect("Ponte Vecchio Max 1550");
    assert_eq!(pvc.architecture, GpuArchitecture::IntelXeHpcPonteVecchio);
    assert_eq!(pvc.vram_capacity_bytes, 128 * 1024 * 1024 * 1024);
    assert_eq!(pvc.memory_tech, MemoryTechnology::Hbm2e);
    assert_eq!(pvc.memory_bandwidth_gbps, 3276.0);
}

#[test]
fn test_intel_xeon_6_granite_rapids_and_sierra_forest() {
    let gnr = GpuDeviceProfile::from_known_device_name("Intel Xeon 6 6980P")
        .expect("Xeon 6980P Granite Rapids");
    assert_eq!(gnr.architecture, GpuArchitecture::IntelXeonGraniteRapids);
    assert_eq!(gnr.sm_count, 128); // 128 Redwood Cove P-Cores
    assert_eq!(gnr.memory_tech, MemoryTechnology::McrDdr5);
    assert_eq!(
        gnr.tensor_core_gen,
        TensorCoreGeneration::IntelAmxTileEngine
    );
    assert_eq!(gnr.memory_bandwidth_gbps, 1536.0); // 1.5 TB/s MCR DDR5

    let plan_gnr = IntelExecutionPlan::for_profile(&gnr);
    assert!(plan_gnr.use_amx_tile_engine);
    assert!(!plan_gnr.use_xmx_matrix_engine);
    assert!(plan_gnr.use_fp8_xmx);

    let srf = GpuDeviceProfile::from_known_device_name("Intel Xeon 6 6780E")
        .expect("Xeon 6780E Sierra Forest");
    assert_eq!(srf.architecture, GpuArchitecture::IntelXeonSierraForest);
    assert_eq!(srf.sm_count, 288); // 288 Crestmont E-Cores
    assert_eq!(srf.tensor_core_gen, TensorCoreGeneration::IntelAvxVnni);
}

#[test]
fn test_intel_xeon_max_with_hbm() {
    let max = GpuDeviceProfile::from_known_device_name("Intel Xeon Max 9480")
        .expect("Xeon Max 9480 with HBM2e");
    assert_eq!(
        max.architecture,
        GpuArchitecture::IntelXeonEmeraldSapphireRapids
    );
    assert_eq!(max.vram_capacity_bytes, 64 * 1024 * 1024 * 1024);
    assert_eq!(max.memory_tech, MemoryTechnology::Hbm2eOnPackageCpu);
    assert_eq!(max.memory_bandwidth_gbps, 1024.0); // 1 TB/s on-package HBM2e
    assert_eq!(
        max.tensor_core_gen,
        TensorCoreGeneration::IntelAmxTileEngine
    );
}

#[test]
fn test_intel_backend_dispatch_and_level_zero_comm() {
    use oxide_core::traits::HardwareBackend;
    use oxide_core::worker::StepCommand;

    let mut backend = IntelBackend::new_with_profile(0, 16, Some("Intel Arc B580"));
    assert_eq!(
        backend.profile().architecture,
        GpuArchitecture::IntelXe2Battlemage
    );

    let cmd = StepCommand::new(1, 200, 4, false);
    let event = backend.dispatch_step_kernel(&cmd).expect("dispatch");
    assert!(backend.query_event_completed(event));
    let token = backend.read_sampled_token_host(4);
    assert_eq!(token, 201);

    let lz = LevelZeroCommunicator::new(0, 8);
    assert_eq!(lz.rank(), 0);
    assert_eq!(lz.world_size(), 8);
}
