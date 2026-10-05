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

use oxide_core::error::Result;
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalEventHandle {
    pub command_buffer_id: u64,
}

pub struct MetalBackend {
    device_id: usize,
    command_counter: u64,
    unified_token_buffer: Vec<u32>,
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_id", &self.device_id)
            .field("command_counter", &self.command_counter)
            .field("unified_token_buffer_len", &self.unified_token_buffer.len())
            .finish()
    }
}

impl MetalBackend {
    #[must_use]
    pub fn new(device_id: usize, max_slots: usize) -> Self {
        Self {
            device_id,
            command_counter: 0,
            unified_token_buffer: vec![0; max_slots],
        }
    }
}

impl HardwareBackend for MetalBackend {
    type Event = MetalEventHandle;

    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        self.command_counter += 1;
        let event = MetalEventHandle {
            command_buffer_id: self.command_counter,
        };

        let slot = cmd.slot_idx as usize;
        if slot < self.unified_token_buffer.len() {
            self.unified_token_buffer[slot] = cmd.input_token.wrapping_add(1);
        }

        Ok(event)
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        let slot = slot_idx as usize;
        if slot < self.unified_token_buffer.len() {
            self.unified_token_buffer[slot]
        } else {
            0
        }
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
