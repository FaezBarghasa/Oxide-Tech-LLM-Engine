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
pub mod driver;
pub mod kernels;
pub mod rccl;

pub use arch::RocmExecutionPlan;
pub use driver::{HipDeviceBuffer, HipStream, is_hip_available};
pub use kernels::RocmLlmKernels;

use oxide_core::error::Result;
use oxide_core::hardware::GpuDeviceProfile;
use oxide_core::traits::{BackendMaturity, HardwareBackend, MaturityReporting};
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipEventHandle {
    pub event_id: u64,
}

impl MaturityReporting for RocmBackend {
    fn maturity() -> BackendMaturity {
        BackendMaturity::Tier2HardwareBringup
    }

    fn maturity_instance(&self) -> BackendMaturity {
        Self::maturity()
    }
}

pub struct RocmBackend {
    device_id: usize,
    current_event_id: u64,
    host_token_buffer: Vec<u32>,
    profile: GpuDeviceProfile,
    execution_plan: RocmExecutionPlan,
    hip_stream: Option<HipStream>,
    #[allow(dead_code)]
    d_activations: Option<HipDeviceBuffer>,
    #[allow(dead_code)]
    d_norm_out: Option<HipDeviceBuffer>,
    #[allow(dead_code)]
    d_weights: Option<HipDeviceBuffer>,
}

impl fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RocmBackend")
            .field("device_id", &self.device_id)
            .field("gpu_name", &self.profile.name)
            .field("architecture", &self.profile.architecture)
            .field("compute_capability", &self.profile.compute_capability)
            .field(
                "vram_gb",
                &(self.profile.vram_capacity_bytes / (1024 * 1024 * 1024)),
            )
            .field("execution_plan", &self.execution_plan)
            .field("event_counter", &self.current_event_id)
            .field("host_token_buffer_len", &self.host_token_buffer.len())
            .field("hip_active", &self.hip_stream.is_some())
            .finish_non_exhaustive()
    }
}

impl RocmBackend {
    /// Creates a ROCm backend for a specified device and known AMD GPU/APU name (or auto-probed).
    #[must_use]
    pub fn new_with_profile(
        device_id: usize,
        max_slots: usize,
        custom_gpu_name: Option<&str>,
    ) -> Self {
        let name = custom_gpu_name.unwrap_or("AMD Instinct MI300X");
        let profile = GpuDeviceProfile::from_known_device_name(name).unwrap_or_default();
        let execution_plan = RocmExecutionPlan::for_profile(&profile);

        let hip_stream = HipStream::new().ok();
        let (d_activations, d_norm_out, d_weights) = if hip_stream.is_some() {
            let act = HipDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            let norm = HipDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            let mut w = HipDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            if let (Some(w_buf), Some(st)) = (&mut w, &hip_stream) {
                let init_w = vec![1.0f32; 128];
                let _ = w_buf.copy_from_host_async(&init_w, st.raw());
                let _ = st.synchronize();
            }
            (act, norm, w)
        } else {
            (None, None, None)
        };

        Self {
            device_id,
            current_event_id: 0,
            host_token_buffer: vec![0; max_slots],
            profile,
            execution_plan,
            hip_stream,
            d_activations,
            d_norm_out,
            d_weights,
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
    pub const fn execution_plan(&self) -> &RocmExecutionPlan {
        &self.execution_plan
    }

    #[must_use]
    pub const fn device_id(&self) -> usize {
        self.device_id
    }
}

impl HardwareBackend for RocmBackend {
    type Event = HipEventHandle;

    fn dispatch_step_kernel(&mut self, _cmd: &StepCommand) -> Result<Self::Event> {
        if !driver::is_hip_available() {
            return Err(oxide_core::error::EngineError::DeviceNotFound);
        }

        // Zero mock/synthetic kernels per AGENTS.md invariant.
        // Full hardware HIP execution pipeline is loaded in Phase 3.
        Err(oxide_core::error::EngineError::BackendError(
            "HIP hardware execution pipeline requires loaded model weights and compiled ROCm kernels"
                .to_string(),
        ))
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
        if let Some(stream) = &self.hip_stream {
            let _ = stream.synchronize();
        }
        Ok(())
    }
}
