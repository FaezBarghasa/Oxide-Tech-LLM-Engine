use std::fmt;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unallocated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocated;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prefilling;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Terminal;

/// Lifetime-parametric, typestate-governed inference request.
/// Guarantees that forward decode passes cannot execute without physical static allocation.
pub struct SequenceRequest<'arena, State> {
    pub sequence_id: u64,
    pub prompt_tokens: &'arena [u32],
    pub slot_idx: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub state_marker: PhantomData<State>,
}

impl<State> fmt::Debug for SequenceRequest<'_, State> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SequenceRequest")
            .field("sequence_id", &self.sequence_id)
            .field("prompt_len", &self.prompt_tokens.len())
            .field("slot_idx", &self.slot_idx)
            .field("generated_tokens", &self.generated_tokens)
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

impl<'arena> SequenceRequest<'arena, Unallocated> {
    #[must_use]
    pub const fn new(sequence_id: u64, prompt_tokens: &'arena [u32], max_tokens: usize) -> Self {
        Self {
            sequence_id,
            prompt_tokens,
            slot_idx: usize::MAX,
            generated_tokens: 0,
            max_tokens,
            state_marker: PhantomData,
        }
    }

    #[must_use]
    pub fn bind_slot(self, slot_idx: usize) -> SequenceRequest<'arena, Allocated> {
        SequenceRequest {
            sequence_id: self.sequence_id,
            prompt_tokens: self.prompt_tokens,
            slot_idx,
            generated_tokens: 0,
            max_tokens: self.max_tokens,
            state_marker: PhantomData,
        }
    }
}

impl<'arena> SequenceRequest<'arena, Allocated> {
    #[must_use]
    pub fn start_prefill(self) -> SequenceRequest<'arena, Prefilling> {
        SequenceRequest {
            sequence_id: self.sequence_id,
            prompt_tokens: self.prompt_tokens,
            slot_idx: self.slot_idx,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            state_marker: PhantomData,
        }
    }
}

impl<'arena> SequenceRequest<'arena, Prefilling> {
    #[must_use]
    pub fn finish_prefill(self) -> SequenceRequest<'arena, Decoding> {
        SequenceRequest {
            sequence_id: self.sequence_id,
            prompt_tokens: self.prompt_tokens,
            slot_idx: self.slot_idx,
            generated_tokens: 0,
            max_tokens: self.max_tokens,
            state_marker: PhantomData,
        }
    }
}

impl<'arena> SequenceRequest<'arena, Decoding> {
    #[must_use]
    pub fn record_token(mut self) -> Self {
        self.generated_tokens += 1;
        self
    }

    #[must_use]
    pub fn terminate(self) -> SequenceRequest<'arena, Terminal> {
        SequenceRequest {
            sequence_id: self.sequence_id,
            prompt_tokens: self.prompt_tokens,
            slot_idx: self.slot_idx,
            generated_tokens: self.generated_tokens,
            max_tokens: self.max_tokens,
            state_marker: PhantomData,
        }
    }
}
