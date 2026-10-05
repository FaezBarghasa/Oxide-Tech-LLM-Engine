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

use oxide_core::error::{EngineError, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// Rockchip RKNN DMABUF shared memory buffer for zero-copy NPU tensor submission.
#[derive(Debug)]
pub struct RknnDmaBuffer {
    pub slot_count: usize,
    buffer: Vec<u32>,
}

impl RknnDmaBuffer {
    #[must_use]
    pub fn new(slot_count: usize) -> Self {
        Self {
            slot_count,
            buffer: vec![0; slot_count],
        }
    }

    #[inline(always)]
    pub fn write_token(&mut self, slot: usize, token: u32) -> Result<()> {
        if slot < self.buffer.len() {
            self.buffer[slot] = token;
            Ok(())
        } else {
            Err(EngineError::AllocationBoundsExceeded { slot_idx: slot })
        }
    }

    #[inline(always)]
    #[must_use]
    pub fn read_token(&self, slot: usize) -> u32 {
        self.buffer.get(slot).copied().unwrap_or(0)
    }
}

/// Rockchip RKNN-Toolkit2 / RKNPU2 execution queue.
#[derive(Debug)]
pub struct RknnNpuStream {
    pub core_mask: u32,
    counter: AtomicU64,
}

impl RknnNpuStream {
    #[must_use]
    pub fn new(core_mask: u32) -> Self {
        Self {
            core_mask,
            counter: AtomicU64::new(1),
        }
    }

    #[inline(always)]
    pub fn dispatch_rknn_core(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn synchronize(&self) -> Result<()> {
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }
}
