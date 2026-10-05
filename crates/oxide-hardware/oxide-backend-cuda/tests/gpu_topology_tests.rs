use oxide_backend_cuda::CudaBackend;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};

#[test]
fn test_commercial_rtx_series_topology() {
    // 1. RTX 2000 Series (Turing)
    let rtx2080 = GpuDeviceProfile::from_known_device_name("NVIDIA GeForce RTX 2080 Ti").unwrap();
    assert_eq!(rtx2080.architecture, GpuArchitecture::Turing);
    assert_eq!(rtx2080.compute_capability, ComputeCapability::SM_75_TURING);
    assert_eq!(rtx2080.tensor_core_gen, TensorCoreGeneration::Gen2Turing);
    assert_eq!(rtx2080.memory_tech, MemoryTechnology::Gddr6);

    // 2. RTX 3000 Series (Ampere)
    let rtx3090 = GpuDeviceProfile::from_known_device_name("NVIDIA GeForce RTX 3090").unwrap();
    assert_eq!(rtx3090.architecture, GpuArchitecture::Ampere);
    assert_eq!(
        rtx3090.compute_capability,
        ComputeCapability::SM_86_AMPERE_CLIENT
    );
    assert_eq!(rtx3090.tensor_core_gen, TensorCoreGeneration::Gen3Ampere);
    assert_eq!(rtx3090.memory_tech, MemoryTechnology::Gddr6X);
    assert!(rtx3090.supports_async_copy);

    // 3. RTX 4000 Series (Ada Lovelace)
    let rtx4090 = GpuDeviceProfile::from_known_device_name("NVIDIA GeForce RTX 4090").unwrap();
    assert_eq!(rtx4090.architecture, GpuArchitecture::Ada);
    assert_eq!(rtx4090.compute_capability, ComputeCapability::SM_89_ADA);
    assert_eq!(rtx4090.tensor_core_gen, TensorCoreGeneration::Gen4HopperAda);
    assert!(rtx4090.supports_fp8);

    // 4. RTX 5000 Series (Blackwell)
    let rtx5090 = GpuDeviceProfile::from_known_device_name("NVIDIA GeForce RTX 5090").unwrap();
    assert_eq!(rtx5090.architecture, GpuArchitecture::Blackwell);
    assert_eq!(
        rtx5090.compute_capability,
        ComputeCapability::SM_120_BLACKWELL_CLIENT
    );
    assert_eq!(rtx5090.tensor_core_gen, TensorCoreGeneration::Gen5Blackwell);
    assert_eq!(rtx5090.memory_tech, MemoryTechnology::Gddr7);
    assert!(rtx5090.supports_nvfp4);
    assert!(rtx5090.supports_tma);
}

#[test]
fn test_enterprise_datacenter_gpus() {
    // A100 80GB SXM
    let a100 = GpuDeviceProfile::from_known_device_name("NVIDIA A100-SXM4-80GB").unwrap();
    assert_eq!(a100.architecture, GpuArchitecture::Ampere);
    assert_eq!(a100.compute_capability, ComputeCapability::SM_80_AMPERE_DC);
    assert_eq!(a100.memory_tech, MemoryTechnology::Hbm2e);
    assert!(a100.supports_nvlink);

    // H100 SXM5
    let h100 = GpuDeviceProfile::from_known_device_name("NVIDIA H100 80GB HBM3").unwrap();
    assert_eq!(h100.architecture, GpuArchitecture::Hopper);
    assert_eq!(h100.compute_capability, ComputeCapability::SM_90_HOPPER);
    assert_eq!(h100.memory_tech, MemoryTechnology::Hbm3);
    assert!(h100.supports_tma);
    assert!(h100.supports_fp8);

    // H200 141GB HBM3e
    let h200 = GpuDeviceProfile::from_known_device_name("NVIDIA H200 141GB").unwrap();
    assert_eq!(h200.architecture, GpuArchitecture::Hopper);
    assert_eq!(h200.memory_tech, MemoryTechnology::Hbm3e);
    assert_eq!(h200.vram_capacity_bytes, 141 * 1024 * 1024 * 1024);

    // B200 SXM 192GB & GB200 Superchip
    let b200 = GpuDeviceProfile::from_known_device_name("NVIDIA B200 SXM 192GB").unwrap();
    assert_eq!(b200.architecture, GpuArchitecture::Blackwell);
    assert_eq!(b200.compute_capability, ComputeCapability::SM_100_BLACKWELL);
    assert_eq!(b200.tensor_core_gen, TensorCoreGeneration::Gen5Blackwell);
    assert_eq!(b200.memory_tech, MemoryTechnology::Hbm3e);
    assert!(b200.supports_nvfp4);
    assert!(b200.supports_tma);

    let gb200 = GpuDeviceProfile::from_known_device_name("NVIDIA GB200 NVL72").unwrap();
    assert_eq!(
        gb200.form_factor,
        HardwareFormFactor::SuperchipGraceBlackwell
    );

    // RTX 6000 Ada Generation / PRO 6000
    let rtx6000_ada =
        GpuDeviceProfile::from_known_device_name("NVIDIA RTX 6000 Ada Generation").unwrap();
    assert_eq!(rtx6000_ada.architecture, GpuArchitecture::Ada);
    assert_eq!(rtx6000_ada.vram_capacity_bytes, 48 * 1024 * 1024 * 1024);
    assert!(rtx6000_ada.supports_fp8);
}

#[test]
fn test_edge_and_specialized_nodes() {
    // Jetson AGX Orin 64GB
    let orin = GpuDeviceProfile::from_known_device_name("NVIDIA Jetson AGX Orin 64GB").unwrap();
    assert_eq!(orin.architecture, GpuArchitecture::Orin);
    assert_eq!(orin.compute_capability, ComputeCapability::SM_87_ORIN);
    assert_eq!(orin.form_factor, HardwareFormFactor::EdgeEmbedded);
    assert_eq!(orin.memory_tech, MemoryTechnology::UnifiedLpddr5X);

    // DGX Spark AI Node
    let spark = GpuDeviceProfile::from_known_device_name("NVIDIA DGX Spark Station").unwrap();
    assert_eq!(spark.form_factor, HardwareFormFactor::DgxStationSpark);
    assert_eq!(spark.vram_capacity_bytes, 128 * 1024 * 1024 * 1024);
}

#[test]
fn test_cuda_backend_autonomic_plan_dispatch() {
    let backend_b200 = CudaBackend::new_with_profile(0, 32, Some("NVIDIA B200 SXM"));
    assert!(backend_b200.execution_plan().use_nvfp4_microscaling);
    assert!(backend_b200.execution_plan().use_tma_async);
    assert_eq!(backend_b200.execution_plan().threadblock_size, 256);

    let backend_4090 = CudaBackend::new_with_profile(0, 32, Some("NVIDIA GeForce RTX 4090"));
    assert!(backend_4090.execution_plan().use_fp8_tensor_cores);
    assert!(!backend_4090.execution_plan().use_nvfp4_microscaling);
    assert_eq!(backend_4090.execution_plan().threadblock_size, 128);

    let backend_2080 = CudaBackend::new_with_profile(0, 32, Some("NVIDIA GeForce RTX 2080 Ti"));
    assert!(!backend_2080.execution_plan().use_fp8_tensor_cores);
    assert_eq!(backend_2080.execution_plan().threadblock_size, 64);
}
