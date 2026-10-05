#![allow(clippy::struct_excessive_bools)]

use oxide_core::hardware::{GpuArchitecture, GpuDeviceProfile};

/// Autonomic TPU Matrix Multiply Unit (MXU) and Vector Processing Unit (VPU) Execution Plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpuExecutionPlan {
    pub mxu_tile_dim: u32,            // 128 for v2-v6, 64 for Coral
    pub mxu_count_per_core: u32,      // 1, 2, 4, 8
    pub vpu_vector_width: u32,        // 128 / 256 vector lane width
    pub vmem_scratchpad_bytes: usize, // Vector SRAM memory buffer
    pub use_sparsecore: bool,         // Hardware SparseCore acceleration (v4/v5/v6)
    pub use_fp8_mxu: bool,            // Native FP8 Matrix Engine (v5e/v5p/v6e)
    pub use_bf16_mxu: bool,           // Native BF16 Systolic Matrix Engine
    pub use_int8_systolic: bool,      // High-throughput INT8 matrix multiplication
    pub unroll_factor: u32,
}

impl TpuExecutionPlan {
    /// Constructs optimal TPU hardware dispatch parameters for a given profile.
    #[must_use]
    pub const fn for_profile(profile: &GpuDeviceProfile) -> Self {
        match profile.architecture {
            GpuArchitecture::GoogleTpuV6eTrillium => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 8,
                vpu_vector_width: 256,
                vmem_scratchpad_bytes: 67_108_864, // 64 MB Vector SRAM
                use_sparsecore: true,
                use_fp8_mxu: true,
                use_bf16_mxu: true,
                use_int8_systolic: true,
                unroll_factor: 8,
            },
            GpuArchitecture::GoogleTpuV5p => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 4,
                vpu_vector_width: 256,
                vmem_scratchpad_bytes: 67_108_864,
                use_sparsecore: true,
                use_fp8_mxu: true,
                use_bf16_mxu: true,
                use_int8_systolic: true,
                unroll_factor: 8,
            },
            GpuArchitecture::GoogleTpuV5e => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 1,
                vpu_vector_width: 128,
                vmem_scratchpad_bytes: 33_554_432, // 32 MB
                use_sparsecore: true,
                use_fp8_mxu: true,
                use_bf16_mxu: true,
                use_int8_systolic: true,
                unroll_factor: 4,
            },
            GpuArchitecture::GoogleTpuV4 => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 4,
                vpu_vector_width: 128,
                vmem_scratchpad_bytes: 33_554_432,
                use_sparsecore: true,
                use_fp8_mxu: false,
                use_bf16_mxu: true,
                use_int8_systolic: true,
                unroll_factor: 4,
            },
            GpuArchitecture::GoogleTpuV3 => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 2,
                vpu_vector_width: 128,
                vmem_scratchpad_bytes: 16_777_216, // 16 MB
                use_sparsecore: false,
                use_fp8_mxu: false,
                use_bf16_mxu: true,
                use_int8_systolic: false,
                unroll_factor: 4,
            },
            GpuArchitecture::GoogleEdgeTpu => Self {
                mxu_tile_dim: 64,
                mxu_count_per_core: 1,
                vpu_vector_width: 64,
                vmem_scratchpad_bytes: 8_388_608, // 8 MB on-chip SRAM
                use_sparsecore: false,
                use_fp8_mxu: false,
                use_bf16_mxu: false,
                use_int8_systolic: true,
                unroll_factor: 2,
            },
            _ => Self {
                mxu_tile_dim: 128,
                mxu_count_per_core: 1,
                vpu_vector_width: 128,
                vmem_scratchpad_bytes: 16_777_216,
                use_sparsecore: false,
                use_fp8_mxu: false,
                use_bf16_mxu: true,
                use_int8_systolic: false,
                unroll_factor: 2,
            },
        }
    }
}
