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

pub mod arch;
pub mod mlx;

use arch::MetalExecutionPlan;
use mlx::{MetalCommandStream, MetalUnifiedBuffer};
use oxide_core::error::Result;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalEventHandle {
    pub command_buffer_id: u64,
}

pub struct MetalBackend {
    device_id: usize,
    profile: GpuDeviceProfile,
    execution_plan: MetalExecutionPlan,
    command_stream: MetalCommandStream,
    unified_buffer: MetalUnifiedBuffer,
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_id", &self.device_id)
            .field("profile", &self.profile.name)
            .field("execution_plan", &self.execution_plan)
            .finish_non_exhaustive()
    }
}

impl MetalBackend {
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
                name: "Apple M4 Max (40-core GPU)".to_string(),
                compute_capability: ComputeCapability::APPLE_GPU_FAMILY_10_M4,
                architecture: GpuArchitecture::AppleSiliconM4,
                form_factor: HardwareFormFactor::UnifiedAppleSiliconMac,
                memory_tech: MemoryTechnology::LpDdr5xUnifiedMemory,
                tensor_core_gen: TensorCoreGeneration::AppleSimdgroupMatrixM4,
                sm_count: 40,
                vram_capacity_bytes: 128 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbps: 546.0,
                l2_cache_bytes: 48 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 32 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });

        let execution_plan = MetalExecutionPlan::derive(&profile);
        let command_stream = MetalCommandStream::new(device_id);
        let unified_buffer = MetalUnifiedBuffer::new_shared(max_slots);

        Self {
            device_id,
            profile,
            execution_plan,
            command_stream,
            unified_buffer,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub fn execution_plan(&self) -> &MetalExecutionPlan {
        &self.execution_plan
    }
}

impl HardwareBackend for MetalBackend {
    type Event = MetalEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        let cmd_id = self.command_stream.dispatch_simdgroup_encode();
        let slot = cmd.slot_idx as usize;

        let sampled = cmd.input_token.wrapping_add(1);
        let _ = self.unified_buffer.write_token(slot, sampled);

        Ok(MetalEventHandle {
            command_buffer_id: cmd_id,
        })
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        self.unified_buffer.read_token(slot_idx as usize)
    }

    fn synchronize(&self) -> Result<()> {
        self.command_stream.synchronize()
    }
}
