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
pub mod qnn;

use arch::QualcommExecutionPlan;
use oxide_core::error::Result;
use oxide_core::hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use qnn::{QnnHtpStream, QnnSharedBuffer};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualcommEventHandle {
    pub htp_fence_id: u64,
}

pub struct QualcommBackend {
    device_id: usize,
    profile: GpuDeviceProfile,
    execution_plan: QualcommExecutionPlan,
    htp_stream: QnnHtpStream,
    shared_buffer: QnnSharedBuffer,
}

impl fmt::Debug for QualcommBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QualcommBackend")
            .field("device_id", &self.device_id)
            .field("profile", &self.profile.name)
            .field("execution_plan", &self.execution_plan)
            .finish_non_exhaustive()
    }
}

impl QualcommBackend {
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
                name: "Snapdragon X Elite (45 TOPS Hexagon NPU)".to_string(),
                compute_capability: ComputeCapability::QUALCOMM_HEXAGON_V75_X_ELITE,
                architecture: GpuArchitecture::QualcommHexagonNpu,
                form_factor: HardwareFormFactor::UnifiedSnapdragonSoc,
                memory_tech: MemoryTechnology::LpDdr5xUnifiedMemory,
                tensor_core_gen: TensorCoreGeneration::QualcommHexagonTensorProcessor,
                sm_count: 6,
                vram_capacity_bytes: 64 * 1024 * 1024 * 1024,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbps: 135.0,
                l2_cache_bytes: 42 * 1024 * 1024,
                smem_per_sm_bytes: 64 * 1024,
                smem_per_block_bytes: 64 * 1024,
                max_threads_per_sm: 1024,
                supports_tma: false,
                supports_fp8: true,
                supports_nvfp4: false,
                supports_async_copy: true,
                supports_nvlink: false,
                nvlink_bandwidth_gbps: 0.0,
            });

        let execution_plan = QualcommExecutionPlan::derive(&profile);
        let htp_stream = QnnHtpStream::new(device_id);
        let shared_buffer = QnnSharedBuffer::new(max_slots);

        Self {
            device_id,
            profile,
            execution_plan,
            htp_stream,
            shared_buffer,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub fn execution_plan(&self) -> &QualcommExecutionPlan {
        &self.execution_plan
    }
}

impl HardwareBackend for QualcommBackend {
    type Event = QualcommEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        let fence_id = self.htp_stream.dispatch_htp_graph();
        let slot = cmd.slot_idx as usize;

        let sampled = cmd.input_token.wrapping_add(1);
        let _ = self.shared_buffer.write_token(slot, sampled);

        Ok(QualcommEventHandle {
            htp_fence_id: fence_id,
        })
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        self.shared_buffer.read_token(slot_idx as usize)
    }

    fn synchronize(&self) -> Result<()> {
        self.htp_stream.synchronize()
    }
}
