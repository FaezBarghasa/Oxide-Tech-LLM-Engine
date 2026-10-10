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
pub mod driver;
pub mod kernels;
pub mod level_zero;

pub use arch::IntelExecutionPlan;
pub use driver::{LevelZeroDeviceBuffer, is_level_zero_available};
pub use kernels::IntelLlmKernels;
pub use level_zero::LevelZeroCommunicator;

use oxide_core::error::Result;
use oxide_core::hardware::GpuDeviceProfile;
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntelEventHandle {
    pub event_id: u64,
}

pub struct IntelBackend {
    device_id: usize,
    current_event_id: u64,
    host_token_buffer: Vec<u32>,
    profile: GpuDeviceProfile,
    execution_plan: IntelExecutionPlan,
    #[allow(dead_code)]
    d_activations: Option<LevelZeroDeviceBuffer>,
}

impl fmt::Debug for IntelBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntelBackend")
            .field("device_id", &self.device_id)
            .field("device_name", &self.profile.name)
            .field("architecture", &self.profile.architecture)
            .field("compute_capability", &self.profile.compute_capability)
            .field(
                "vram_or_mem_gb",
                &(self.profile.vram_capacity_bytes / (1024 * 1024 * 1024)),
            )
            .field("execution_plan", &self.execution_plan)
            .field("event_counter", &self.current_event_id)
            .field("host_token_buffer_len", &self.host_token_buffer.len())
            .field("level_zero_active", &is_level_zero_available())
            .finish_non_exhaustive()
    }
}

impl IntelBackend {
    /// Creates an Intel backend for a specified device/core and known Intel Arc GPU or Xeon CPU profile.
    #[must_use]
    pub fn new_with_profile(
        device_id: usize,
        max_slots: usize,
        custom_device_name: Option<&str>,
    ) -> Self {
        let name = custom_device_name.unwrap_or("Intel Arc B580");
        let profile = GpuDeviceProfile::from_known_device_name(name)
            .unwrap_or_else(|| GpuDeviceProfile::from_known_device_name("b580").unwrap());
        let execution_plan = IntelExecutionPlan::for_profile(&profile);

        Self {
            device_id,
            current_event_id: 0,
            host_token_buffer: vec![0; max_slots],
            profile,
            execution_plan,
            d_activations: None,
        }
    }

    #[must_use]
    pub fn new(device_id: usize, max_slots: usize) -> Self {
        Self::new_with_profile(device_id, max_slots, None)
    }

    #[must_use]
    pub const fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub const fn execution_plan(&self) -> &IntelExecutionPlan {
        &self.execution_plan
    }

    #[must_use]
    pub const fn device_id(&self) -> usize {
        self.device_id
    }
}

impl HardwareBackend for IntelBackend {
    type Event = IntelEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        if !driver::is_level_zero_available() {
            return Err(oxide_core::error::EngineError::DeviceNotFound);
        }

        self.current_event_id += 1;
        let event = IntelEventHandle {
            event_id: self.current_event_id,
        };

        let slot = cmd.slot_idx as usize;
        IntelLlmKernels::dispatch_intel_step_decode(
            cmd.input_token,
            slot,
            &mut self.host_token_buffer,
        )?;

        Ok(event)
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        let slot = slot_idx as usize;
        if slot < self.host_token_buffer.len() {
            self.host_token_buffer[slot]
        } else {
            0
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
