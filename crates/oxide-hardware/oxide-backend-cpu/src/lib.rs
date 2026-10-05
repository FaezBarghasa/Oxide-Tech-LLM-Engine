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

pub mod ternary;

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
}

impl fmt::Debug for CpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBackend")
            .field("thread_id", &self.thread_id)
            .field("step_counter", &self.step_counter)
            .field("token_buffer_len", &self.token_buffer.len())
            .finish()
    }
}

impl CpuBackend {
    #[must_use]
    pub fn new(thread_id: usize, max_slots: usize) -> Self {
        Self {
            thread_id,
            step_counter: 0,
            token_buffer: vec![0; max_slots],
        }
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
        if slot < self.token_buffer.len() {
            let mut activations = [0.0f32; 128];
            for (i, act) in activations.iter_mut().enumerate() {
                *act = ((cmd.input_token as f32 * 0.05) + (i as f32 * 0.1)).sin();
            }
            let dummy_blocks = [oxide_quant::ptq1_0::TernaryBlock128 {
                scale_fp16: 0x3c00, // 1.0 in FP16
                packed_weights: [0x55; 32],
            }];
            let dot = ternary::ternary_gemv_cpu(&activations, &dummy_blocks);
            let next_tok = (cmd.input_token.wrapping_add(1) + (dot.abs() as u32)).max(1);
            self.token_buffer[slot] = next_tok;
        }

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
