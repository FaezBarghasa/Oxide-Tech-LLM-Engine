//! High-Performance Paged KV-Cache Block Arena for Zero-Allocation LLM Serving.
//!
//! Provides vLLM / SGLang parity paged memory management:
//! - Static physical block allocation eliminating runtime heap reallocations.
//! - Lock-free O(1) allocation and retirement via `crossbeam_queue::SegQueue`.
//! - Per-sequence logical block tables mapping contiguous tokens to physical blocks.

use crossbeam_queue::SegQueue;
use std::sync::atomic::{AtomicU32, Ordering};

/// Tokens stored per physical block (standard page size).
pub const TOKENS_PER_PAGE: usize = 16;

/// A single pre-allocated physical KV cache block storing key/value states for up to `TOKENS_PER_PAGE` tokens.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct PhysicalPageBlock {
    pub block_id: u32,
    pub ref_count: AtomicU32,
    /// Key cache flattened: `[TOKENS_PER_PAGE, num_kv_heads * head_dim]`
    pub k: Vec<f32>,
    /// Value cache flattened: `[TOKENS_PER_PAGE, num_kv_heads * head_dim]`
    pub v: Vec<f32>,
}

impl PhysicalPageBlock {
    #[must_use]
    pub fn new(block_id: u32, head_dim_total: usize) -> Self {
        let size = TOKENS_PER_PAGE * head_dim_total;
        Self {
            block_id,
            ref_count: AtomicU32::new(1),
            k: vec![0.0; size],
            v: vec![0.0; size],
        }
    }

    #[inline(always)]
    pub fn retain(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn release(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// Global Paged KV-Cache Arena maintaining pre-allocated physical page blocks across all transformer layers.
#[derive(Debug)]
pub struct PagedKvArena {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub head_dim_total: usize,
    pub total_blocks: usize,
    /// Physical block storage organized as `[layer_idx][block_id]`
    blocks: Vec<Vec<PhysicalPageBlock>>,
    /// Lock-free queue of free physical block IDs
    free_blocks: SegQueue<u32>,
}

impl PagedKvArena {
    #[must_use]
    pub fn new(
        total_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let head_dim_total = num_kv_heads * head_dim;
        let mut layers = Vec::with_capacity(num_layers);
        let free_blocks = SegQueue::new();

        for _ in 0..num_layers {
            let mut layer_blocks = Vec::with_capacity(total_blocks);
            for b in 0..total_blocks {
                layer_blocks.push(PhysicalPageBlock::new(b as u32, head_dim_total));
            }
            layers.push(layer_blocks);
        }

        for b in 0..total_blocks {
            free_blocks.push(b as u32);
        }

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            head_dim_total,
            total_blocks,
            blocks: layers,
            free_blocks,
        }
    }

    /// Allocates an unused physical block ID with O(1) lock-free pop.
    #[must_use]
    pub fn allocate_block(&self) -> Option<u32> {
        let block_id = self.free_blocks.pop()?;
        for layer in &self.blocks {
            layer[block_id as usize]
                .ref_count
                .store(1, Ordering::Relaxed);
        }
        Some(block_id)
    }

    /// Releases a block ID back to the free list when reference count drops to zero.
    pub fn release_block(&self, block_id: u32) {
        let idx = block_id as usize;
        if idx >= self.total_blocks {
            return;
        }
        let should_free = self.blocks[0][idx].release();
        if should_free {
            for layer in self.blocks.iter().skip(1) {
                let _ = layer[idx].release();
            }
            self.free_blocks.push(block_id);
        }
    }

    /// Write a single token's KV projection into a physical block at a given token offset (0..TOKENS_PER_PAGE).
    pub fn write_token_kv(
        &mut self,
        layer_idx: usize,
        block_id: u32,
        token_offset_in_block: usize,
        k_token: &[f32],
        v_token: &[f32],
    ) {
        if layer_idx >= self.num_layers || block_id as usize >= self.total_blocks {
            return;
        }
        let block = &mut self.blocks[layer_idx][block_id as usize];
        let offset = token_offset_in_block * self.head_dim_total;

        let k_len = k_token.len().min(self.head_dim_total);
        let v_len = v_token.len().min(self.head_dim_total);

        block.k[offset..offset + k_len].copy_from_slice(&k_token[..k_len]);
        block.v[offset..offset + v_len].copy_from_slice(&v_token[..v_len]);
    }

    /// Reads a contiguous slice of a physical block's key/value data.
    #[must_use]
    pub fn get_block_data(&self, layer_idx: usize, block_id: u32) -> (&[f32], &[f32]) {
        let block = &self.blocks[layer_idx][block_id as usize];
        (&block.k, &block.v)
    }

    #[must_use]
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }
}

/// Default initial attention sink tokens preserved permanently (StreamingLLM).
pub const DEFAULT_SINK_TOKENS: usize = 4;

/// Default recent local sliding window tokens preserved permanently (StreamingLLM / H2O).
pub const DEFAULT_LOCAL_ROLLING_TOKENS: usize = 256;

/// Logical Block Table mapping a sequence's token stream to physical page blocks
/// with StreamingLLM Attention Sinks & H2O (Heavy-Hitter Oracle) pruning support.
#[derive(Debug, Clone)]
pub struct SequenceBlockTable {
    pub block_ids: Vec<u32>,
    pub num_tokens: usize,
    /// Cumulative attention score mass per logical block
    pub block_scores: Vec<f32>,
    /// Number of attention sink tokens locked at the start of the sequence
    pub sink_tokens: usize,
    /// Number of tokens locked at the rolling recent tail of the sequence
    pub local_rolling_tokens: usize,
}

impl Default for SequenceBlockTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SequenceBlockTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
            block_scores: Vec::new(),
            sink_tokens: DEFAULT_SINK_TOKENS,
            local_rolling_tokens: DEFAULT_LOCAL_ROLLING_TOKENS,
        }
    }

    /// Initializes a block table with custom StreamingLLM sink and rolling window parameters.
    #[must_use]
    pub fn with_streaming(sink_tokens: usize, local_rolling_tokens: usize) -> Self {
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
            block_scores: Vec::new(),
            sink_tokens,
            local_rolling_tokens,
        }
    }

    /// Records accumulated attention score mass for a given logical block.
    pub fn record_block_attention(&mut self, block_idx: usize, score: f32) {
        if block_idx < self.block_scores.len() {
            self.block_scores[block_idx] += score;
        }
    }

    /// Appends a new token position, allocating a new physical block from the arena if boundary crossed.
    pub fn advance_token(&mut self, arena: &PagedKvArena) -> Result<(u32, usize), &'static str> {
        let block_idx = self.num_tokens / TOKENS_PER_PAGE;
        let token_offset = self.num_tokens % TOKENS_PER_PAGE;

        if block_idx >= self.block_ids.len() {
            let new_block = arena.allocate_block().ok_or("Paged KV arena exhausted")?;
            self.block_ids.push(new_block);
            self.block_scores.push(0.0);
        }

        let assigned_block = self.block_ids[block_idx];
        self.num_tokens += 1;
        Ok((assigned_block, token_offset))
    }

    /// Prunes logical blocks down to `max_blocks_budget` using H2O (Heavy-Hitter Oracle) + StreamingLLM:
    /// - Locks initial attention sink blocks (`sink_tokens`)
    /// - Locks recent sliding-window blocks (`local_rolling_tokens`)
    /// - Evicts middle blocks with lowest cumulative attention scores back to the `arena`
    /// - Returns the list of evicted physical block IDs.
    pub fn prune_heavy_hitters(
        &mut self,
        arena: &PagedKvArena,
        max_blocks_budget: usize,
    ) -> Vec<u32> {
        let total_blocks = self.block_ids.len();
        if total_blocks <= max_blocks_budget {
            return Vec::new();
        }

        let sink_blocks = (self.sink_tokens.saturating_add(TOKENS_PER_PAGE - 1)) / TOKENS_PER_PAGE;
        let local_blocks =
            (self.local_rolling_tokens.saturating_add(TOKENS_PER_PAGE - 1)) / TOKENS_PER_PAGE;

        if sink_blocks + local_blocks >= total_blocks {
            // Cannot prune without violating sink or local rolling invariant
            return Vec::new();
        }

        let mid_end = total_blocks - local_blocks;
        let candidate_indices: Vec<usize> = (sink_blocks..mid_end).collect();

        let allowed_mid_blocks = max_blocks_budget.saturating_sub(sink_blocks + local_blocks);
        if candidate_indices.len() <= allowed_mid_blocks {
            return Vec::new();
        }

        // Sort candidate indices descending by cumulative attention score (heavy hitters first)
        let mut scored_candidates: Vec<(usize, f32)> = candidate_indices
            .iter()
            .map(|&idx| {
                let score = self.block_scores.get(idx).copied().unwrap_or(0.0);
                (idx, score)
            })
            .collect();

        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut retained_indices_set = std::collections::HashSet::new();
        for &(idx, _) in scored_candidates.iter().take(allowed_mid_blocks) {
            retained_indices_set.insert(idx);
        }

        let mut evicted_blocks = Vec::new();
        let mut new_block_ids = Vec::with_capacity(max_blocks_budget);
        let mut new_block_scores = Vec::with_capacity(max_blocks_budget);

        // 1. Keep all sink blocks
        for i in 0..sink_blocks {
            new_block_ids.push(self.block_ids[i]);
            new_block_scores.push(self.block_scores.get(i).copied().unwrap_or(0.0));
        }

        // 2. Keep heavy-hitter middle blocks in chronological order
        for i in sink_blocks..mid_end {
            if retained_indices_set.contains(&i) {
                new_block_ids.push(self.block_ids[i]);
                new_block_scores.push(self.block_scores.get(i).copied().unwrap_or(0.0));
            } else {
                let evicted_id = self.block_ids[i];
                arena.release_block(evicted_id);
                evicted_blocks.push(evicted_id);
            }
        }

        // 3. Keep all local rolling blocks
        for i in mid_end..total_blocks {
            new_block_ids.push(self.block_ids[i]);
            new_block_scores.push(self.block_scores.get(i).copied().unwrap_or(0.0));
        }

        self.block_ids = new_block_ids;
        self.block_scores = new_block_scores;
        self.num_tokens = self.block_ids.len() * TOKENS_PER_PAGE;

        evicted_blocks
    }

    /// Releases all held blocks back to the arena upon sequence completion.
    pub fn release_all(&mut self, arena: &PagedKvArena) {
        for &block_id in &self.block_ids {
            arena.release_block(block_id);
        }
        self.block_ids.clear();
        self.block_scores.clear();
        self.num_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2o_streaming_llm_pruning() {
        let total_arena_blocks = 64;
        let arena = PagedKvArena::new(total_arena_blocks, 1, 1, 64);
        assert_eq!(arena.free_block_count(), 64);

        // Create table with 4 sink tokens (1 block) and 32 local rolling tokens (2 blocks)
        let mut table = SequenceBlockTable::with_streaming(4, 32);

        // Advance 160 tokens -> 10 blocks (indices 0..10)
        for _ in 0..160 {
            table.advance_token(&arena).expect("advance should succeed");
        }
        assert_eq!(table.block_ids.len(), 10);
        assert_eq!(arena.free_block_count(), 64 - 10);

        // Assign attention scores to middle blocks (1..8):
        // Make block 3 and block 6 heavy hitters
        table.record_block_attention(3, 100.0);
        table.record_block_attention(6, 85.0);
        table.record_block_attention(1, 1.0);
        table.record_block_attention(2, 0.5);
        table.record_block_attention(4, 2.0);
        table.record_block_attention(5, 0.1);
        table.record_block_attention(7, 3.0);

        // Prune down to budget of 5 blocks:
        // - 1 sink block (block 0)
        // - 2 local rolling blocks (blocks 8, 9)
        // - 2 middle heavy hitters (should pick block 3 and block 6)
        let orig_sink_id = table.block_ids[0];
        let orig_heavy_3 = table.block_ids[3];
        let orig_heavy_6 = table.block_ids[6];
        let orig_roll_8 = table.block_ids[8];
        let orig_roll_9 = table.block_ids[9];

        let evicted = table.prune_heavy_hitters(&arena, 5);
        assert_eq!(evicted.len(), 5); // 10 - 5 = 5 blocks evicted
        assert_eq!(table.block_ids.len(), 5);
        assert_eq!(arena.free_block_count(), 64 - 5);

        // Verify preserved blocks:
        assert_eq!(table.block_ids[0], orig_sink_id, "Sink block preserved");
        assert_eq!(table.block_ids[1], orig_heavy_3, "Heavy hitter 3 preserved");
        assert_eq!(table.block_ids[2], orig_heavy_6, "Heavy hitter 6 preserved");
        assert_eq!(table.block_ids[3], orig_roll_8, "Rolling block 8 preserved");
        assert_eq!(table.block_ids[4], orig_roll_9, "Rolling block 9 preserved");

        // Release remaining
        table.release_all(&arena);
        assert_eq!(arena.free_block_count(), 64);
    }
}
