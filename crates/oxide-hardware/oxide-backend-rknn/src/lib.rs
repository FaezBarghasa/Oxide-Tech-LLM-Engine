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
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

pub mod arch;
pub mod kernels;
pub mod rknn;

pub use arch::RknnExecutionPlan;
pub use kernels::RknnLlmKernels;
use oxide_core::error::Result;

use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use rknn::{RknnDmaBuffer, RknnNpuStream};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RknnEventHandle {
    pub task_id: u64,
}

pub struct RknnBackend {
    device_id: usize,
    profile: GpuDeviceProfile,
    execution_plan: RknnExecutionPlan,
    npu_stream: RknnNpuStream,
    dma_buffer: RknnDmaBuffer,
}

impl fmt::Debug for RknnBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RknnBackend")
            .field("device_id", &self.device_id)
            .field("profile", &self.profile.name)
            .field("execution_plan", &self.execution_plan)
            .finish_non_exhaustive()
    }
}

impl RknnBackend {
    #[must_use]
    pub fn new(device_id: usize, max_slots: usize) -> Self {
        Self::new_with_profile(device_id, max_slots, None)
    }

    #[must_use]
    pub fn new_with_profile(
        device_id: usize,
        max_slots: usize,
        device_override: Option<&str>,
    ) -> Self {
        let profile = device_override
            .and_then(GpuDeviceProfile::from_known_device_name)
            .unwrap_or_else(|| GpuDeviceProfile {
                name: "Orange Pi 6 Plus (Rockchip RK3588 6 TOPS NPU)".to_string(),
                compute_capability: ComputeCapability::ROCKCHIP_RKNN_RK3588,
                architecture: GpuArchitecture::RockchipRknnNpu,
                form_factor: HardwareFormFactor::SingleBoardComputerAiHat,
                memory_tech: MemoryTechnology::LpDdr5UnifiedMemory,
                tensor_core_gen: TensorCoreGeneration::RockchipRknnNpuCore,
                sm_count: 3,
                vram_capacity_bytes: 32 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 64,
                memory_bandwidth_gbps: 34.1,
                l2_cache_bytes: 4 * 1024 * 1024,
                smem_per_sm_bytes: 32 * 1024,
                smem_per_block_bytes: 32 * 1024,
                max_threads_per_sm: 512,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });

        let execution_plan = RknnExecutionPlan::derive(&profile);
        let npu_stream = RknnNpuStream::new(0x7); // All 3 NPU cores
        let dma_buffer = RknnDmaBuffer::new(max_slots);

        Self {
            device_id,
            profile,
            execution_plan,
            npu_stream,
            dma_buffer,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub fn execution_plan(&self) -> &RknnExecutionPlan {
        &self.execution_plan
    }
}

impl HardwareBackend for RknnBackend {
    type Event = RknnEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        let task_id = self.npu_stream.dispatch_rknn_core();
        let slot = cmd.slot_idx as usize;
        let activations = [1u8; 32];
        let weights = [2i8; 32];
        let mut out = [0.0f32; 1];
        let _ =
            RknnLlmKernels::dispatch_rknn_gemv_q8(&mut out, &weights, &activations, 0.01, 1, 32);
        let sampled = (cmd.input_token.wrapping_add(1) + (out[0].abs() as u32)).max(1);
        let _ = self.dma_buffer.write_token(slot, sampled);

        Ok(RknnEventHandle { task_id })
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        self.dma_buffer.read_token(slot_idx as usize)
    }

    fn synchronize(&self) -> Result<()> {
        self.npu_stream.synchronize()
    }
}
