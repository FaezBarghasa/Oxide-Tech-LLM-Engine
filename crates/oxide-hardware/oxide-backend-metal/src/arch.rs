#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use oxide_core::hardware::{GpuArchitecture, GpuDeviceProfile, TensorCoreGeneration};

/// Microarchitectural Execution Plan for Apple Silicon Metal & MLX unified engines.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct MetalExecutionPlan {
    pub simdgroup_matrix_enabled: bool,
    pub dynamic_caching_enabled: bool,
    pub native_fp8_enabled: bool,
    pub native_bf16_enabled: bool,
    pub ane_co_execution_enabled: bool,
    pub threadgroup_mem_bytes: usize,
    pub threads_per_simdgroup: u32,
    pub simdgroups_per_threadgroup: u32,
    pub max_concurrent_command_encoders: u32,
    pub zero_copy_unified_memory: bool,
}

impl MetalExecutionPlan {
    /// Derives optimal Metal / MLX execution plan from target Apple Silicon hardware profile.
    #[must_use]
    pub fn derive(profile: &GpuDeviceProfile) -> Self {
        match profile.architecture {
            GpuArchitecture::AppleSiliconM4 => Self {
                simdgroup_matrix_enabled: true,
                dynamic_caching_enabled: true,
                native_fp8_enabled: true,
                native_bf16_enabled: true,
                ane_co_execution_enabled: true,
                threadgroup_mem_bytes: 32 * 1024,
                threads_per_simdgroup: 32,
                simdgroups_per_threadgroup: 8, // 256 threads per threadgroup
                max_concurrent_command_encoders: 16,
                zero_copy_unified_memory: true,
            },
            GpuArchitecture::AppleSiliconM3 => Self {
                simdgroup_matrix_enabled: true,
                dynamic_caching_enabled: true,
                native_fp8_enabled: false,
                native_bf16_enabled: true,
                ane_co_execution_enabled: true,
                threadgroup_mem_bytes: 32 * 1024,
                threads_per_simdgroup: 32,
                simdgroups_per_threadgroup: 8,
                max_concurrent_command_encoders: 16,
                zero_copy_unified_memory: true,
            },
            GpuArchitecture::AppleSiliconM2 => Self {
                simdgroup_matrix_enabled: true,
                dynamic_caching_enabled: false,
                native_fp8_enabled: false,
                native_bf16_enabled: true,
                ane_co_execution_enabled: true,
                threadgroup_mem_bytes: 32 * 1024,
                threads_per_simdgroup: 32,
                simdgroups_per_threadgroup: 8,
                max_concurrent_command_encoders: 8,
                zero_copy_unified_memory: true,
            },
            GpuArchitecture::AppleSiliconM1 => Self {
                simdgroup_matrix_enabled: true,
                dynamic_caching_enabled: false,
                native_fp8_enabled: false,
                native_bf16_enabled: false, // M1 uses FP16/INT8 SIMD-group matrix
                ane_co_execution_enabled: true,
                threadgroup_mem_bytes: 32 * 1024,
                threads_per_simdgroup: 32,
                simdgroups_per_threadgroup: 4, // 128 threads per threadgroup
                max_concurrent_command_encoders: 8,
                zero_copy_unified_memory: true,
            },
            _ => Self {
                simdgroup_matrix_enabled: matches!(
                    profile.tensor_core_gen,
                    TensorCoreGeneration::AppleSimdgroupMatrixM1
                        | TensorCoreGeneration::AppleSimdgroupMatrixM2
                        | TensorCoreGeneration::AppleSimdgroupMatrixM3
                        | TensorCoreGeneration::AppleSimdgroupMatrixM4
                ),
                dynamic_caching_enabled: false,
                native_fp8_enabled: profile.supports_fp8,
                native_bf16_enabled: true,
                ane_co_execution_enabled: false,
                threadgroup_mem_bytes: 16 * 1024,
                threads_per_simdgroup: 32,
                simdgroups_per_threadgroup: 4,
                max_concurrent_command_encoders: 4,
                zero_copy_unified_memory: true,
            },
        }
    }
}
