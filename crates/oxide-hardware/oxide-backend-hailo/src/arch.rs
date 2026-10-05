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

use oxide_core::hardware::GpuDeviceProfile;

/// Execution plan for Hailo-8 / Hailo-10 / External NPU / External TPU accelerators.
#[derive(Debug, Clone, PartialEq)]
pub struct HailoExecutionPlan {
    pub pcie_gen: u32,
    pub native_int8_enabled: bool,
    pub native_int4_enabled: bool,
    pub native_fp8_enabled: bool,
    pub structural_sparsity_enabled: bool,
    pub peak_npu_tops: f32,
    pub dataflow_pipeline_stages: u32,
}

impl HailoExecutionPlan {
    #[must_use]
    pub fn derive(profile: &GpuDeviceProfile) -> Self {
        let is_hailo10 = profile.name.to_lowercase().contains("hailo-10")
            || profile.name.to_lowercase().contains("ai-hat-plus-2");

        let is_8l = profile.name.to_lowercase().contains("8l")
            || profile.name.to_lowercase().contains("13");

        if is_hailo10 {
            Self {
                pcie_gen: 3,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp8_enabled: true,
                structural_sparsity_enabled: true,
                peak_npu_tops: 40.0,
                dataflow_pipeline_stages: 16,
            }
        } else if is_8l {
            Self {
                pcie_gen: 2,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp8_enabled: false,
                structural_sparsity_enabled: true,
                peak_npu_tops: 13.0,
                dataflow_pipeline_stages: 8,
            }
        } else {
            Self {
                pcie_gen: 2,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp8_enabled: false,
                structural_sparsity_enabled: true,
                peak_npu_tops: 26.0,
                dataflow_pipeline_stages: 8,
            }
        }
    }
}
