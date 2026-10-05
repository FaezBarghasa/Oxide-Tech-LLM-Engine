#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Execution State of an Individual Inference Slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlotState {
    /// Slot is idle and available for new prompt assignment.
    Idle,
    /// Slot is actively processing prompt prefill chunks.
    Prefill,
    /// Slot is in token-by-token autoregressive decode loop.
    Decode,
    /// Slot has finished generation and awaits client drain.
    Finished,
}

/// Request Metadata bound to a specific Slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
}

/// Dynamic Continuous Batching Slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceSlot {
    pub slot_id: usize,
    pub state: SlotState,
    pub request: Option<SlotRequest>,
    pub generated_tokens: Vec<u32>,
    pub current_context_pos: usize,
    pub allocated_kv_blocks: usize,
}

impl InferenceSlot {
    #[must_use]
    pub fn new(slot_id: usize) -> Self {
        Self {
            slot_id,
            state: SlotState::Idle,
            request: None,
            generated_tokens: Vec::new(),
            current_context_pos: 0,
            allocated_kv_blocks: 0,
        }
    }

    pub fn assign_request(&mut self, request: SlotRequest) {
        let prompt_len = request.prompt_tokens.len();
        self.state = SlotState::Prefill;
        self.request = Some(request);
        self.generated_tokens.clear();
        self.current_context_pos = prompt_len;
        self.allocated_kv_blocks = (prompt_len + 31) / 32;
    }

    pub fn append_token(&mut self, token: u32, is_eos: bool) {
        self.generated_tokens.push(token);
        self.current_context_pos += 1;
        self.allocated_kv_blocks = (self.current_context_pos + 31) / 32;

        let reached_max = self
            .request
            .as_ref()
            .is_some_and(|r| self.generated_tokens.len() >= r.max_tokens);

        if is_eos || reached_max {
            self.state = SlotState::Finished;
        } else {
            self.state = SlotState::Decode;
        }
    }

    pub fn reset(&mut self) {
        self.state = SlotState::Idle;
        self.request = None;
        self.generated_tokens.clear();
        self.current_context_pos = 0;
        self.allocated_kv_blocks = 0;
    }
}

/// Continuous Batching & Slot Scheduler.
#[derive(Debug, Clone)]
pub struct ContinuousBatchingSlotManager {
    pub max_slots: usize,
    slots: Vec<InferenceSlot>,
    request_to_slot: HashMap<String, usize>,
}

impl ContinuousBatchingSlotManager {
    #[must_use]
    pub fn new(max_slots: usize) -> Self {
        let mut slots = Vec::with_capacity(max_slots);
        for i in 0..max_slots {
            slots.push(InferenceSlot::new(i));
        }
        Self {
            max_slots,
            slots,
            request_to_slot: HashMap::new(),
        }
    }

    /// Finds and reserves an idle slot for an incoming request.
    pub fn submit_request(&mut self, request: SlotRequest) -> Option<usize> {
        let idle_slot_id = self.slots.iter().position(|s| s.state == SlotState::Idle)?;
        let req_id = request.request_id.clone();
        self.slots[idle_slot_id].assign_request(request);
        self.request_to_slot.insert(req_id, idle_slot_id);
        Some(idle_slot_id)
    }

    /// Gathers all active slots ready for the current forward step (prefill or decode).
    #[must_use]
    pub fn get_active_batch_slots(&self) -> Vec<&InferenceSlot> {
        self.slots
            .iter()
            .filter(|s| s.state == SlotState::Prefill || s.state == SlotState::Decode)
            .collect()
    }

    /// Releases a slot upon request completion or client disconnect.
    pub fn release_slot(&mut self, slot_id: usize) {
        if let Some(slot) = self.slots.get_mut(slot_id) {
            if let Some(req) = &slot.request {
                self.request_to_slot.remove(&req.request_id);
            }
            slot.reset();
        }
    }

    #[must_use]
    pub fn get_slot(&self, slot_id: usize) -> Option<&InferenceSlot> {
        self.slots.get(slot_id)
    }

    #[must_use]
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Option<&mut InferenceSlot> {
        self.slots.get_mut(slot_id)
    }

    #[must_use]
    pub fn total_active_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state != SlotState::Idle)
            .count()
    }
}
