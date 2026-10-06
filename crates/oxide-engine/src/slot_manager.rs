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
    pub prefill_progress: usize,
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
            prefill_progress: 0,
        }
    }

    pub fn assign_request(&mut self, request: SlotRequest) {
        let prompt_len = request.prompt_tokens.len();
        self.state = SlotState::Prefill;
        self.request = Some(request);
        self.generated_tokens.clear();
        self.current_context_pos = 0;
        self.prefill_progress = 0;
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
        self.prefill_progress = 0;
    }
}

/// Granular scheduling unit in a chunked prefill/decode forward iteration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChunkedBatchItem {
    /// Ingest a segment of prompt tokens (bounded by chunk_size) to populate KV cache.
    PrefillChunk {
        slot_id: usize,
        token_chunk: Vec<u32>,
        start_pos: usize,
        is_last_chunk: bool,
    },
    /// Generate a single token in autoregressive decoding.
    DecodeStep {
        slot_id: usize,
        token: u32,
        pos: usize,
    },
}

/// Continuous Batching & Chunked Prefill Slot Scheduler.
#[derive(Debug, Clone)]
pub struct ContinuousBatchingSlotManager {
    pub max_slots: usize,
    pub chunk_size: usize,
    slots: Vec<InferenceSlot>,
    request_to_slot: HashMap<String, usize>,
}

impl ContinuousBatchingSlotManager {
    #[must_use]
    pub fn new(max_slots: usize) -> Self {
        Self::new_with_chunk_size(max_slots, 512)
    }

    #[must_use]
    pub fn new_with_chunk_size(max_slots: usize, chunk_size: usize) -> Self {
        let mut slots = Vec::with_capacity(max_slots);
        for i in 0..max_slots {
            slots.push(InferenceSlot::new(i));
        }
        Self {
            max_slots,
            chunk_size: chunk_size.max(32),
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

    /// Schedules an iteration-level batch interleaving prefill chunks and decode steps.
    /// Prevents long prompts from starving running decode streams.
    pub fn schedule_chunked_step(&mut self) -> Vec<ChunkedBatchItem> {
        let mut batch = Vec::new();

        for slot in &mut self.slots {
            match slot.state {
                SlotState::Prefill => {
                    if let Some(req) = &slot.request {
                        let total = req.prompt_tokens.len();
                        let start = slot.prefill_progress;
                        let end = (start + self.chunk_size).min(total);
                        let is_last = end >= total;

                        if start < total {
                            let chunk = req.prompt_tokens[start..end].to_vec();
                            slot.current_context_pos = end;
                            slot.prefill_progress = end;

                            if is_last {
                                slot.state = SlotState::Decode;
                            }

                            batch.push(ChunkedBatchItem::PrefillChunk {
                                slot_id: slot.slot_id,
                                token_chunk: chunk,
                                start_pos: start,
                                is_last_chunk: is_last,
                            });
                        }
                    }
                }
                SlotState::Decode => {
                    let last_token = slot
                        .generated_tokens
                        .last()
                        .copied()
                        .or_else(|| {
                            slot.request
                                .as_ref()
                                .and_then(|r| r.prompt_tokens.last().copied())
                        })
                        .unwrap_or(1);

                    batch.push(ChunkedBatchItem::DecodeStep {
                        slot_id: slot.slot_id,
                        token: last_token,
                        pos: slot.current_context_pos,
                    });
                }
                SlotState::Idle | SlotState::Finished => {}
            }
        }

        batch
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_prefill_and_decode_scheduling() {
        let mut manager = ContinuousBatchingSlotManager::new_with_chunk_size(4, 64);

        // Request 1: 150 prompt tokens (will split into 64 + 64 + 22)
        let prompt_1: Vec<u32> = (1..=150).collect();
        let req_1 = SlotRequest {
            request_id: "req-1".to_string(),
            prompt_tokens: prompt_1,
            max_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
            stream: false,
        };
        let slot_1 = manager.submit_request(req_1).unwrap();
        assert_eq!(slot_1, 0);

        // Step 1: Chunk 0..64
        let batch_1 = manager.schedule_chunked_step();
        assert_eq!(batch_1.len(), 1);
        match &batch_1[0] {
            ChunkedBatchItem::PrefillChunk {
                slot_id,
                token_chunk,
                start_pos,
                is_last_chunk,
            } => {
                assert_eq!(*slot_id, 0);
                assert_eq!(token_chunk.len(), 64);
                assert_eq!(*start_pos, 0);
                assert!(!is_last_chunk);
            }
            _ => panic!("Expected PrefillChunk"),
        }

        // Step 2: Chunk 64..128
        let batch_2 = manager.schedule_chunked_step();
        assert_eq!(batch_2.len(), 1);
        match &batch_2[0] {
            ChunkedBatchItem::PrefillChunk {
                start_pos,
                token_chunk,
                is_last_chunk,
                ..
            } => {
                assert_eq!(*start_pos, 64);
                assert_eq!(token_chunk.len(), 64);
                assert!(!is_last_chunk);
            }
            _ => panic!("Expected PrefillChunk"),
        }

        // Step 3: Chunk 128..150 (final prefill chunk)
        let batch_3 = manager.schedule_chunked_step();
        assert_eq!(batch_3.len(), 1);
        match &batch_3[0] {
            ChunkedBatchItem::PrefillChunk {
                start_pos,
                token_chunk,
                is_last_chunk,
                ..
            } => {
                assert_eq!(*start_pos, 128);
                assert_eq!(token_chunk.len(), 22);
                assert!(is_last_chunk);
            }
            _ => panic!("Expected PrefillChunk"),
        }

        // Step 4: Now transitioned to DecodeStep!
        let batch_4 = manager.schedule_chunked_step();
        assert_eq!(batch_4.len(), 1);
        match &batch_4[0] {
            ChunkedBatchItem::DecodeStep {
                slot_id,
                token,
                pos,
            } => {
                assert_eq!(*slot_id, 0);
                assert_eq!(*token, 150);
                assert_eq!(*pos, 150);
            }
            _ => panic!("Expected DecodeStep"),
        }
    }
}
