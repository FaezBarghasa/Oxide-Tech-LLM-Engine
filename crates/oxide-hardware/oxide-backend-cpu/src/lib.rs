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

pub mod kernels;
pub mod ternary;

pub use kernels::{CpuLlmKernels, CpuThreadPool};

use oxide_core::error::Result;
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuEventHandle {
    pub step_id: u64,
}

pub struct CpuBackend {
    thread_id: usize,
    step_counter: u64,
    token_buffer: Vec<u32>,
    thread_pool: CpuThreadPool,
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("thread_id", &self.thread_id)
            .field("step_counter", &self.step_counter)
            .field("token_buffer_len", &self.token_buffer.len())
            .field("hardware_threads", &self.thread_pool.num_threads())
            .finish()
    }
}

impl CpuBackend {
    #[must_use]
    pub fn new(thread_id: usize, max_slots: usize) -> Self {
        let pool = CpuThreadPool::new();
        let _ = pool.pin_current_thread(thread_id);

        Self {
            thread_id,
            step_counter: 0,
            token_buffer: vec![0; max_slots],
            thread_pool: pool,
        }
    }

    #[must_use]
    pub const fn thread_pool(&self) -> &CpuThreadPool {
        &self.thread_pool
    }
}

impl HardwareBackend for CpuBackend {
    type Event = CpuEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        self.step_counter += 1;
        let event = CpuEventHandle {
            step_id: self.step_counter,
        };

        let slot = cmd.slot_idx as usize;
        CpuLlmKernels::dispatch_full_step_decode(cmd.input_token, slot, &mut self.token_buffer)?;

        Ok(event)
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        let slot = slot_idx as usize;
        if slot < self.token_buffer.len() {
            self.token_buffer[slot]
        } else {
            0
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
