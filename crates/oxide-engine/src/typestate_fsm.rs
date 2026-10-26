//! TokenSpeed-inspired Compile-Time Typestate FSM Control Plane.
//!
//! Replaces runtime state checks and lock contention with strict Rust typestates:
//! - Guarantees impossible state transitions are caught at compile time.
//! - Enforces safe resource management: KV slots cannot be double-freed,
//!   and unallocated requests cannot be dispatched for decoding.
//! - Achieves zero mutex contention in the hot continuous scheduling loop.

use oxide_core::error::{EngineError, Result};
use oxide_core::sampler::SamplingConfig;
use std::collections::VecDeque;

// --- Typestate Markers ---

/// Request has been created but not yet assigned memory resources or slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pending;

/// Request has memory pages / slots allocated in the KV arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocated {
    pub slot_id: usize,
}

/// Request is executing prompt prefill across all input tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prefilling {
    pub slot_id: usize,
    pub prefilled_tokens: usize,
}

/// Request has completed prefill and is in active autoregressive decode generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoding {
    pub slot_id: usize,
    pub step_count: usize,
}

/// Request generation completed successfully (EOS, stop token, or max tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Completed {
    pub finish_reason: FinishReason,
}

/// Request was cancelled early by client disconnect or timeout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

/// Completion termination reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    StopToken(u32),
    MaxTokensReached,
    LengthExceeded,
}

// --- Typestate Request Container ---

/// Strongly-typed Request carrying state marker `S`.
#[derive(Debug, Clone, PartialEq)]
pub struct Request<S> {
    pub request_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingConfig,
    pub state: S,
}

impl Request<Pending> {
    /// Creates a new pending request.
    #[must_use]
    pub fn new(
        request_id: u64,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingConfig,
    ) -> Self {
        Self {
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            sampling,
            state: Pending,
        }
    }

    /// Transitions to `Allocated` state once a physical slot has been reserved.
    #[must_use]
    pub fn allocate(self, slot_id: usize) -> Request<Allocated> {
        Request {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            state: Allocated { slot_id },
        }
    }

    /// Cancels request before allocation.
    #[must_use]
    pub fn cancel(self) -> Request<Cancelled> {
        Request {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            state: Cancelled,
        }
    }
}

impl Request<Allocated> {
    /// Transitions to `Prefilling` state to begin chunked or full prompt processing.
    #[must_use]
    pub fn start_prefill(self) -> Request<Prefilling> {
        Request {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            state: Prefilling {
                slot_id: self.state.slot_id,
                prefilled_tokens: 0,
            },
        }
    }

    /// Releases slot if cancelled during allocation.
    #[must_use]
    pub fn cancel(self) -> (Request<Cancelled>, usize) {
        let slot = self.state.slot_id;
        (
            Request {
                request_id: self.request_id,
                prompt_tokens: self.prompt_tokens,
                generated_tokens: self.generated_tokens,
                max_tokens: self.max_tokens,
                sampling: self.sampling,
                state: Cancelled,
            },
            slot,
        )
    }
}

impl Request<Prefilling> {
    /// Advances prefill progress.
    pub fn advance_prefill(&mut self, chunk_size: usize) {
        self.state.prefilled_tokens += chunk_size;
    }

    /// Checks if all prompt tokens have been prefilled.
    #[must_use]
    pub fn is_prefill_complete(&self) -> bool {
        self.state.prefilled_tokens >= self.prompt_tokens.len()
    }

    /// Transitions to `Decoding` state once prefill is fully completed.
    pub fn finish_prefill(self) -> Result<Request<Decoding>> {
        if !self.is_prefill_complete() {
            return Err(EngineError::InvalidStateTransition {
                sequence_id: self.request_id,
            });
        }
        Ok(Request {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            state: Decoding {
                slot_id: self.state.slot_id,
                step_count: 0,
            },
        })
    }
}

/// Outcome of a single decode step transition.
#[derive(Debug)]
pub enum DecodeStepResult {
    /// Still decoding; token generated and step advanced.
    Continue(Request<Decoding>, u32),
    /// Finished decoding; returns completed request and allocated slot ID for reclamation.
    Done(Request<Completed>, usize),
}

impl Request<Decoding> {
    /// Steps autoregressive generation with newly sampled token ID.
    #[must_use]
    pub fn step(mut self, token_id: u32, eos_token: u32) -> DecodeStepResult {
        self.generated_tokens.push(token_id);
        let current_step = self.state.step_count + 1;
        let slot_id = self.state.slot_id;

        // Check completion conditions
        if token_id == eos_token {
            let completed = Request {
                request_id: self.request_id,
                prompt_tokens: self.prompt_tokens,
                generated_tokens: self.generated_tokens,
                max_tokens: self.max_tokens,
                sampling: self.sampling,
                state: Completed {
                    finish_reason: FinishReason::StopToken(token_id),
                },
            };
            return DecodeStepResult::Done(completed, slot_id);
        }

        if self.generated_tokens.len() >= self.max_tokens {
            let completed = Request {
                request_id: self.request_id,
                prompt_tokens: self.prompt_tokens,
                generated_tokens: self.generated_tokens,
                max_tokens: self.max_tokens,
                sampling: self.sampling,
                state: Completed {
                    finish_reason: FinishReason::MaxTokensReached,
                },
            };
            return DecodeStepResult::Done(completed, slot_id);
        }

        // Continue decoding
        let next_state = Request {
            request_id: self.request_id,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            sampling: self.sampling,
            state: Decoding {
                slot_id,
                step_count: current_step,
            },
        };
        DecodeStepResult::Continue(next_state, token_id)
    }
}

/// Typestate-enforced Request Scheduler maintaining partitioned queues.
#[derive(Debug, Default)]
pub struct TypestateScheduler {
    pub pending: VecDeque<Request<Pending>>,
    pub prefilling: VecDeque<Request<Prefilling>>,
    pub decoding: Vec<Request<Decoding>>,
    pub completed: Vec<Request<Completed>>,
}

impl TypestateScheduler {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueues a new pending request.
    pub fn submit(&mut self, request: Request<Pending>) {
        self.pending.push_back(request);
    }

    /// Schedules next pending request for allocation when a free slot is available.
    pub fn schedule_next_allocation(&mut self, slot_id: usize) -> Option<Request<Prefilling>> {
        let req = self.pending.pop_front()?;
        let allocated = req.allocate(slot_id);
        let prefilling = allocated.start_prefill();
        Some(prefilling)
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.prefilling.len() + self.decoding.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typestate_fsm_lifecycle() {
        let mut scheduler = TypestateScheduler::new();

        let req = Request::new(1001, vec![1, 2, 3], 2, SamplingConfig::default());
        scheduler.submit(req);

        // Allocate slot 0
        let mut prefill_req = scheduler.schedule_next_allocation(0).expect("scheduled");
        assert_eq!(prefill_req.state.slot_id, 0);
        assert!(!prefill_req.is_prefill_complete());

        // Advance prefill
        prefill_req.advance_prefill(3);
        assert!(prefill_req.is_prefill_complete());

        // Transition to decoding
        let decode_req = prefill_req.finish_prefill().expect("prefill complete");
        assert_eq!(decode_req.state.step_count, 0);

        // Step 1: generate token 42 (continue)
        let step1 = decode_req.step(42, 999);
        let decode_req = match step1 {
            DecodeStepResult::Continue(req, tok) => {
                assert_eq!(tok, 42);
                req
            }
            DecodeStepResult::Done(_, _) => panic!("Should not be done"),
        };

        // Step 2: generate token 999 (EOS -> Done)
        let step2 = decode_req.step(999, 999);
        match step2 {
            DecodeStepResult::Done(completed, slot) => {
                assert_eq!(slot, 0);
                assert_eq!(completed.generated_tokens, vec![42, 999]);
                assert_eq!(
                    completed.state.finish_reason,
                    FinishReason::StopToken(999)
                );
            }
            DecodeStepResult::Continue(_, _) => panic!("Should be done on EOS"),
        }
    }
}
