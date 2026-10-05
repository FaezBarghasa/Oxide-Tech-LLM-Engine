#![allow(clippy::struct_excessive_bools)]

use oxide_core::hardware::{GpuArchitecture, GpuDeviceProfile};

/// Autonomic Kernel Configuration tuned for a specific hardware target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelExecutionPlan {
    pub threadblock_size: u32,
    pub shared_memory_bytes: usize,
    pub warps_per_block: u32,
    pub use_tma_async: bool,
    pub use_fp8_tensor_cores: bool,
    pub use_nvfp4_microscaling: bool,
    pub unroll_factor: u32,
}

impl KernelExecutionPlan {
    /// Determines the optimal execution plan for a GPU device profile.
    #[must_use]
    pub const fn for_profile(profile: &GpuDeviceProfile) -> Self {
        match profile.architecture {
            GpuArchitecture::Blackwell => Self {
                threadblock_size: 256,
                shared_memory_bytes: 98_304, // 96 KB
                warps_per_block: 8,
                use_tma_async: true,
                use_fp8_tensor_cores: true,
                use_nvfp4_microscaling: true,
                unroll_factor: 8,
            },
            GpuArchitecture::Hopper => Self {
                threadblock_size: 256,
                shared_memory_bytes: 65_536, // 64 KB
                warps_per_block: 8,
                use_tma_async: true,
                use_fp8_tensor_cores: true,
                use_nvfp4_microscaling: false,
                unroll_factor: 8,
            },
            GpuArchitecture::Ada => Self {
                threadblock_size: 128,
                shared_memory_bytes: 49_152, // 48 KB
                warps_per_block: 4,
                use_tma_async: false,
                use_fp8_tensor_cores: true,
                use_nvfp4_microscaling: false,
                unroll_factor: 4,
            },
            GpuArchitecture::Ampere | GpuArchitecture::Orin => Self {
                threadblock_size: 128,
                shared_memory_bytes: 49_152, // 48 KB
                warps_per_block: 4,
                use_tma_async: false,
                use_fp8_tensor_cores: false,
                use_nvfp4_microscaling: false,
                unroll_factor: 4,
            },
            _ => Self {
                threadblock_size: 64,
                shared_memory_bytes: 32_768, // 32 KB
                warps_per_block: 2,
                use_tma_async: false,
                use_fp8_tensor_cores: false,
                use_nvfp4_microscaling: false,
                unroll_factor: 2,
            },
        }
    }
}
