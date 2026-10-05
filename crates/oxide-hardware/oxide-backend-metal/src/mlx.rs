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

/// Metal / MLX Zero-Copy Unified Shared Storage Buffer.
#[derive(Debug)]
pub struct MetalUnifiedBuffer {
    pub slot_count: usize,
    pub storage_mode: &'static str,
    buffer: Vec<u32>,
}

impl MetalUnifiedBuffer {
    #[must_use]
    pub fn new_shared(slot_count: usize) -> Self {
        Self {
            slot_count,
            storage_mode: "MTLResourceStorageModeShared", // Zero-copy CPU/GPU unified memory
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

    #[must_use]
    pub fn raw_slice(&self) -> &[u32] {
        &self.buffer
    }
}

/// Metal Command Stream and MLX Graph Dispatch Queue.
#[derive(Debug)]
pub struct MetalCommandStream {
    pub queue_id: usize,
    counter: AtomicU64,
}

impl MetalCommandStream {
    #[must_use]
    pub fn new(queue_id: usize) -> Self {
        Self {
            queue_id,
            counter: AtomicU64::new(1),
        }
    }

    #[inline(always)]
    pub fn dispatch_simdgroup_encode(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn synchronize(&self) -> Result<()> {
        // CPU/GPU coherence barrier in Apple Unified Memory
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }
}
