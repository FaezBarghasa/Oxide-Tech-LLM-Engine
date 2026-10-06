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
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop
)]

pub mod arch;
pub mod hailort;
pub mod kernels;

pub use arch::HailoExecutionPlan;
pub use hailort::{HailoVStreamBuffer, HailoVirtualStream};
pub use kernels::HailoLlmKernels;
use oxide_core::error::Result;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HailoEventHandle {
    pub vstream_event_id: u64,
}

pub struct HailoBackend {
    device_id: usize,
    profile: GpuDeviceProfile,
    execution_plan: HailoExecutionPlan,
    vstream: HailoVirtualStream,
    buffer: HailoVStreamBuffer,
}

impl fmt::Debug for HailoBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HailoBackend")
            .field("device_id", &self.device_id)
            .field("profile", &self.profile.name)
            .field("execution_plan", &self.execution_plan)
            .finish_non_exhaustive()
    }
}

impl HailoBackend {
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
                name: "Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8)".to_string(),
                compute_capability: ComputeCapability::HAILO_8_26TOPS,
                architecture: GpuArchitecture::HailoNpu,
                form_factor: HardwareFormFactor::SingleBoardComputerAiHat,
                memory_tech: MemoryTechnology::LpDdr4x,
                tensor_core_gen: TensorCoreGeneration::Hailo8SystolicNpuEngine,
                sm_count: 4,
                vram_capacity_bytes: 8 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 32,
                memory_bandwidth_gbps: 17.0,
                l2_cache_bytes: 4 * 1024 * 1024,
                smem_per_sm_bytes: 32 * 1024,
                smem_per_block_bytes: 32 * 1024,
                max_threads_per_sm: 256,
                supports_tma: false,
                supports_fp8: false,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });

        let execution_plan = HailoExecutionPlan::derive(&profile);
        let vstream = HailoVirtualStream::new(device_id);
        let buffer = HailoVStreamBuffer::new(max_slots);

        Self {
            device_id,
            profile,
            execution_plan,
            vstream,
            buffer,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub fn execution_plan(&self) -> &HailoExecutionPlan {
        &self.execution_plan
    }
}

impl HardwareBackend for HailoBackend {
    type Event = HailoEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        let event_id = self.vstream.dispatch_vstream();
        let slot = cmd.slot_idx as usize;
        let _ =
            HailoLlmKernels::dispatch_hailo_step_decode(cmd.input_token, slot, &mut self.buffer)?;

        Ok(HailoEventHandle {
            vstream_event_id: event_id,
        })
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        self.buffer.read_token(slot_idx as usize)
    }

    fn synchronize(&self) -> Result<()> {
        self.vstream.synchronize()
    }
}
