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

/// Execution plan for Rockchip RKNN NPU (Orange Pi 6 Plus / Orange Pi 5 / RK3588 / RK3576).
#[derive(Debug, Clone, PartialEq)]
pub struct RknnExecutionPlan {
    pub npu_core_count: u32,
    pub native_int8_enabled: bool,
    pub native_int4_enabled: bool,
    pub native_fp16_enabled: bool,
    pub native_bf16_enabled: bool,
    pub dma_buf_zero_copy: bool,
    pub peak_npu_tops: f32,
    pub rknn_batch_size: u32,
}

impl RknnExecutionPlan {
    #[must_use]
    pub fn derive(profile: &GpuDeviceProfile) -> Self {
        let is_rk3588 = profile.name.to_lowercase().contains("rk3588")
            || profile.name.to_lowercase().contains("orange pi 6")
            || profile.name.to_lowercase().contains("orange pi 5");

        if is_rk3588 {
            Self {
                npu_core_count: 3, // Tri-core NPU
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp16_enabled: true,
                native_bf16_enabled: true,
                dma_buf_zero_copy: true,
                peak_npu_tops: 6.0, // 6 TOPS INT8
                rknn_batch_size: 1,
            }
        } else {
            Self {
                npu_core_count: 2,
                native_int8_enabled: true,
                native_int4_enabled: true,
                native_fp16_enabled: true,
                native_bf16_enabled: true,
                dma_buf_zero_copy: true,
                peak_npu_tops: 6.0,
                rknn_batch_size: 1,
            }
        }
    }
}
