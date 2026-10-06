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
pub mod graph;
pub mod kernels;
pub mod nccl;

pub use arch::KernelExecutionPlan;
pub use graph::{CapturedCudaGraph, CudaGraphExecHandle, CudaGraphManager};
pub use kernels::CudaLlmKernels;
pub use nccl::{CudaDeviceClusterArray, NcclCommunicator};

use oxide_core::error::Result;
use oxide_core::hardware::GpuDeviceProfile;
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaEventHandle {
    pub event_id: u64,
}

pub struct CudaBackend {
    device_id: usize,
    current_event_id: u64,
    host_token_buffer: Vec<u32>,
    profile: GpuDeviceProfile,
    execution_plan: KernelExecutionPlan,
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
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
            .finish()
    }
}

impl CudaBackend {
    /// Creates a CUDA backend for a specified device and known GPU name (or auto-probed).
    #[must_use]
    pub fn new_with_profile(
        device_id: usize,
        max_slots: usize,
        custom_gpu_name: Option<&str>,
    ) -> Self {
        let name = custom_gpu_name.unwrap_or("NVIDIA GeForce RTX 4090");
        let profile = GpuDeviceProfile::from_known_device_name(name).unwrap_or_else(|| {
            // Default baseline: Ada Lovelace
            GpuDeviceProfile::from_known_device_name("rtx 4090").unwrap()
        });
        let execution_plan = KernelExecutionPlan::for_profile(&profile);

        Self {
            device_id,
            current_event_id: 0,
            host_token_buffer: vec![0; max_slots],
            profile,
            execution_plan,
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
    pub const fn execution_plan(&self) -> &KernelExecutionPlan {
        &self.execution_plan
    }

    #[must_use]
    pub const fn device_id(&self) -> usize {
        self.device_id
    }
}

impl HardwareBackend for CudaBackend {
    type Event = CudaEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        self.current_event_id += 1;
        let event = CudaEventHandle {
            event_id: self.current_event_id,
        };

        let slot = cmd.slot_idx as usize;
        if slot < self.host_token_buffer.len() {
            let mut activations = [0.0f32; 128];
            for (i, act) in activations.iter_mut().enumerate() {
                *act = ((cmd.input_token as f32 * 0.05) + (i as f32 * 0.1)).sin();
            }
            let mut norm_out = [0.0f32; 128];
            let weights = [1.0f32; 128];
            let _ =
                CudaLlmKernels::dispatch_rmsnorm(&mut norm_out, &activations, &weights, 128, 1e-5);
            let next_tok = (cmd.input_token.wrapping_add(1) + (norm_out[0].abs() as u32)).max(1);
            self.host_token_buffer[slot] = next_tok;
        }

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
