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
pub mod graph;
pub mod kernels;
pub mod nccl;

pub use arch::KernelExecutionPlan;
pub use driver::{CudaDeviceBuffer, CudaStream};
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
    stream: Option<CudaStream>,
    d_activations: Option<CudaDeviceBuffer>,
    d_norm_out: Option<CudaDeviceBuffer>,
    d_weights: Option<CudaDeviceBuffer>,
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
            .field("has_real_stream", &self.stream.is_some())
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

        // Attempt real physical device initialization
        let stream = CudaStream::new().ok();
        let (d_activations, d_norm_out, d_weights) = if stream.is_some() {
            let _ = unsafe { driver::cudaSetDevice(device_id as i32) };
            let act = CudaDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            let norm = CudaDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            let mut w = CudaDeviceBuffer::allocate(128 * std::mem::size_of::<f32>()).ok();
            if let (Some(w_buf), Some(st)) = (&mut w, &stream) {
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
            stream,
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
                let hash = (cmd.input_token.wrapping_mul(2_654_435_761)).wrapping_add(i as u32);
                let sign = if (hash & 1) == 0 { 1.0f32 } else { -1.0f32 };
                let mag = ((hash >> 1) % 1000) as f32 / 1000.0f32;
                *act = sign * mag * 0.1;
            }

            let mut norm_out = [0.0f32; 128];

            // If real CUDA device stream and memory buffers are active, execute on physical GPU
            if let (Some(stream), Some(d_act), Some(d_norm), Some(d_w)) = (
                &self.stream,
                &mut self.d_activations,
                &mut self.d_norm_out,
                &self.d_weights,
            ) {
                let raw_stream = stream.raw();
                let _ = d_act.copy_from_host_async(&activations, raw_stream);

                // SAFETY: Calling C-ABI launch_cuda_rmsnorm with valid device pointers and stream.
                unsafe {
                    driver::launch_cuda_rmsnorm(
                        d_norm.as_typed_ptr::<f32>(),
                        d_act.as_typed_ptr::<f32>(),
                        d_w.as_typed_ptr::<f32>(),
                        1,
                        128,
                        1e-5,
                        raw_stream,
                    );
                }

                let _ = d_norm.copy_to_host_async(&mut norm_out, raw_stream);
                let _ = stream.synchronize();
            } else {
                let weights = [1.0f32; 128];
                let _ = CudaLlmKernels::dispatch_rmsnorm(
                    &mut norm_out,
                    &activations,
                    &weights,
                    128,
                    1e-5,
                );
            }

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
        if let Some(stream) = &self.stream {
            let _ = stream.synchronize();
        }
        Ok(())
    }
}
