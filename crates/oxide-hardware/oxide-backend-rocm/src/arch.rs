#![allow(clippy::struct_excessive_bools)]

use oxide_core::hardware::{GpuArchitecture, GpuDeviceProfile};

/// Autonomic Kernel Execution Plan for AMD CDNA Instinct, RDNA, and Ryzen AI APU+NPU hardware targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RocmExecutionPlan {
    pub wavefront_size: u32,
    pub workgroup_size: u32,
    pub lds_shared_memory_bytes: usize,
    pub use_mfma_cdna: bool,
    pub use_wmma_rdna: bool,
    pub use_xdna_npu_tile: bool,
    pub use_fp8_mfma: bool,
    pub use_fp4_mfma: bool,
    pub unroll_factor: u32,
}

impl RocmExecutionPlan {
    /// Determines the optimal execution plan for an AMD GPU/APU profile.
    #[must_use]
    pub const fn for_profile(profile: &GpuDeviceProfile) -> Self {
        match profile.architecture {
            GpuArchitecture::Cdna4 => Self {
                wavefront_size: 64, // Wave64 CDNA
                workgroup_size: 256,
                lds_shared_memory_bytes: 65_536, // 64 KB LDS
                use_mfma_cdna: true,
                use_wmma_rdna: false,
                use_xdna_npu_tile: false,
                use_fp8_mfma: true,
                use_fp4_mfma: true, // CDNA 4 MI350X/MI355X FP4 Matrix Engine
                unroll_factor: 8,
            },
            GpuArchitecture::Cdna3 => Self {
                wavefront_size: 64,
                workgroup_size: 256,
                lds_shared_memory_bytes: 65_536,
                use_mfma_cdna: true,
                use_wmma_rdna: false,
                use_xdna_npu_tile: false,
                use_fp8_mfma: true, // CDNA 3 MI300X/MI325X FP8 Matrix Engine
                use_fp4_mfma: false,
                unroll_factor: 8,
            },
            GpuArchitecture::Cdna1 | GpuArchitecture::Cdna2 => Self {
                wavefront_size: 64,
                workgroup_size: 256,
                lds_shared_memory_bytes: 65_536,
                use_mfma_cdna: true,
                use_wmma_rdna: false,
                use_xdna_npu_tile: false,
                use_fp8_mfma: false,
                use_fp4_mfma: false,
                unroll_factor: 4,
            },
            GpuArchitecture::Rdna3 | GpuArchitecture::Rdna4 => Self {
                wavefront_size: 32, // Wave32 RDNA Matrix Cores
                workgroup_size: 128,
                lds_shared_memory_bytes: 65_536,
                use_mfma_cdna: false,
                use_wmma_rdna: true,
                use_xdna_npu_tile: false,
                use_fp8_mfma: true,
                use_fp4_mfma: false,
                unroll_factor: 4,
            },
            GpuArchitecture::Rdna3_5 | GpuArchitecture::XdnaNpu => Self {
                wavefront_size: 32,
                workgroup_size: 128,
                lds_shared_memory_bytes: 65_536,
                use_mfma_cdna: false,
                use_wmma_rdna: true,
                use_xdna_npu_tile: true, // Offload dense spatial GEMM to XDNA NPU
                use_fp8_mfma: true,
                use_fp4_mfma: true, // XDNA 2 block-quantized FP4
                unroll_factor: 4,
            },
            _ => Self {
                wavefront_size: 32,
                workgroup_size: 64,
                lds_shared_memory_bytes: 32_768,
                use_mfma_cdna: false,
                use_wmma_rdna: false,
                use_xdna_npu_tile: false,
                use_fp8_mfma: false,
                use_fp4_mfma: false,
                unroll_factor: 2,
            },
        }
    }
}
