#![allow(clippy::struct_excessive_bools)]

use oxide_core::hardware::{GpuArchitecture, GpuDeviceProfile};

/// Autonomic Kernel Execution Plan for Intel Arc GPUs (XMX) and Intel Xeon Scalable CPUs (AMX/AVX).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntelExecutionPlan {
    pub simd_width: u32,               // SIMD16 / SIMD32 for Xe; 512-bit vector for AVX-512/AMX
    pub workgroup_threads: u32,        // 128 / 256
    pub slm_shared_memory_bytes: usize,// Shared Local Memory buffer (64KB - 128KB)
    pub use_xmx_matrix_engine: bool,   // Hardware XMX systolic matrix accelerator
    pub use_amx_tile_engine: bool,     // Hardware AMX-TMUL 1KB matrix tiles (TMM0-TMM7)
    pub use_fp8_xmx: bool,             // Native FP8 (E4M3/E5M2) support on Gen2 XMX (Battlemage)
    pub use_bf16: bool,                // BF16 matrix multiply
    pub use_int8_systolic: bool,       // INT8 dot-product / DPAS / VNNI
    pub unroll_factor: u32,
}

impl IntelExecutionPlan {
    /// Determines the optimal execution plan for an Intel Arc GPU or Xeon CPU profile.
    #[must_use]
    pub const fn for_profile(profile: &GpuDeviceProfile) -> Self {
        match profile.architecture {
            GpuArchitecture::IntelXe2Battlemage => Self {
                simd_width: 16,
                workgroup_threads: 256,
                slm_shared_memory_bytes: 65_536, // 64 KB SLM per Xe-core
                use_xmx_matrix_engine: true,
                use_amx_tile_engine: false,
                use_fp8_xmx: true, // Battlemage Gen2 XMX native FP8
                use_bf16: true,
                use_int8_systolic: true,
                unroll_factor: 8,
            },
            GpuArchitecture::IntelXeHpcPonteVecchio => Self {
                simd_width: 32,
                workgroup_threads: 256,
                slm_shared_memory_bytes: 131_072, // 128 KB SLM
                use_xmx_matrix_engine: true,
                use_amx_tile_engine: false,
                use_fp8_xmx: true,
                use_bf16: true,
                use_int8_systolic: true,
                unroll_factor: 8,
            },
            GpuArchitecture::IntelXe1Alchemist => Self {
                simd_width: 16,
                workgroup_threads: 128,
                slm_shared_memory_bytes: 65_536,
                use_xmx_matrix_engine: true,
                use_amx_tile_engine: false,
                use_fp8_xmx: false,
                use_bf16: true,
                use_int8_systolic: true,
                unroll_factor: 4,
            },
            GpuArchitecture::IntelXeonGraniteRapids => Self {
                simd_width: 512, // 512-bit vector / AMX 1KB matrix tiles
                workgroup_threads: 256,
                slm_shared_memory_bytes: 131_072,
                use_xmx_matrix_engine: false,
                use_amx_tile_engine: true, // AMX-TMUL FP16/BF16/INT8
                use_fp8_xmx: true,
                use_bf16: true,
                use_int8_systolic: true,
                unroll_factor: 8,
            },
            GpuArchitecture::IntelXeonEmeraldSapphireRapids => Self {
                simd_width: 512,
                workgroup_threads: 128,
                slm_shared_memory_bytes: 65_536,
                use_xmx_matrix_engine: false,
                use_amx_tile_engine: true, // AMX-TMUL BF16/INT8
                use_fp8_xmx: false,
                use_bf16: true,
                use_int8_systolic: true,
                unroll_factor: 4,
            },
            GpuArchitecture::IntelXeonSierraForest => Self {
                simd_width: 256, // AVX-VNNI INT8
                workgroup_threads: 128,
                slm_shared_memory_bytes: 65_536,
                use_xmx_matrix_engine: false,
                use_amx_tile_engine: false,
                use_fp8_xmx: false,
                use_bf16: false,
                use_int8_systolic: true, // AVX-VNNI
                unroll_factor: 4,
            },
            _ => Self {
                simd_width: 16,
                workgroup_threads: 128,
                slm_shared_memory_bytes: 65_536,
                use_xmx_matrix_engine: false,
                use_amx_tile_engine: false,
                use_fp8_xmx: false,
                use_bf16: true,
                use_int8_systolic: false,
                unroll_factor: 2,
            },
        }
    }
}
