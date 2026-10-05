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

/// Execution plan for Qualcomm Snapdragon Hexagon NPU & HTP (Hexagon Tensor Processor).
#[derive(Debug, Clone, PartialEq)]
pub struct QualcommExecutionPlan {
    pub htp_vector_lanes: u32,
    pub native_fp16_enabled: bool,
    pub native_int8_enabled: bool,
    pub native_int4_enabled: bool,
    pub native_fp8_enabled: bool,
    pub zero_copy_ion_shared_memory: bool,
    pub peak_npu_tops: f32,
    pub hvx_thread_count: u32,
    pub qnn_graph_dispatch_threads: u32,
}

impl QualcommExecutionPlan {
    #[must_use]
    pub fn derive(profile: &GpuDeviceProfile) -> Self {
        let is_x_series = profile.name.to_lowercase().contains("snapdragon x")
            || profile.name.to_lowercase().contains("x1e")
            || profile.name.to_lowercase().contains("x1p")
            || profile.name.to_lowercase().contains("x2");

        let is_8_elite = profile.name.to_lowercase().contains("8 elite");

        if is_x_series || is_8_elite {
            Self {
                htp_vector_lanes: 4096, // 4096-bit Hexagon Vector eXtensions
                native_fp16_enabled: true,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp8_enabled: true,
                zero_copy_ion_shared_memory: true,
                peak_npu_tops: if is_x_series { 45.0 } else { 45.0 },
                hvx_thread_count: 8,
                qnn_graph_dispatch_threads: 4,
            }
        } else {
            Self {
                htp_vector_lanes: 2048,
                native_fp16_enabled: true,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp8_enabled: false,
                zero_copy_ion_shared_memory: true,
                peak_npu_tops: 20.0,
                hvx_thread_count: 4,
                qnn_graph_dispatch_threads: 2,
            }
        }
    }
}
