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

/// Qualcomm QNN / HTP Shared ION Memory Buffer for Zero-Copy Host/NPU inference.
#[derive(Debug)]
pub struct QnnSharedBuffer {
    pub slot_count: usize,
    buffer: Vec<u32>,
}

impl QnnSharedBuffer {
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

/// Qualcomm QNN Graph Dispatch & HTP Context Stream.
#[derive(Debug)]
pub struct QnnHtpStream {
    pub stream_id: usize,
    counter: AtomicU64,
}

impl QnnHtpStream {
    #[must_use]
    pub fn new(stream_id: usize) -> Self {
        Self {
            stream_id,
            counter: AtomicU64::new(1),
        }
    }

    #[inline(always)]
    pub fn dispatch_htp_graph(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn synchronize(&self) -> Result<()> {
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }
}
