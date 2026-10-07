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
            layer[block_id as usize].ref_count.store(1, Ordering::Relaxed);
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

/// Logical Block Table mapping a sequence's token stream to physical page blocks.
#[derive(Debug, Clone, Default)]
pub struct SequenceBlockTable {
    pub block_ids: Vec<u32>,
    pub num_tokens: usize,
}

impl SequenceBlockTable {
    #[must_use]
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            num_tokens: 0,
        }
    }

    /// Appends a new token position, allocating a new physical block from the arena if boundary crossed.
    pub fn advance_token(&mut self, arena: &PagedKvArena) -> Result<(u32, usize), &'static str> {
        let block_idx = self.num_tokens / TOKENS_PER_PAGE;
        let token_offset = self.num_tokens % TOKENS_PER_PAGE;

        if block_idx >= self.block_ids.len() {
            let new_block = arena.allocate_block().ok_or("Paged KV arena exhausted")?;
            self.block_ids.push(new_block);
        }

        let assigned_block = self.block_ids[block_idx];
        self.num_tokens += 1;
        Ok((assigned_block, token_offset))
    }

    /// Releases all held blocks back to the arena upon sequence completion.
    pub fn release_all(&mut self, arena: &PagedKvArena) {
        for &block_id in &self.block_ids {
            arena.release_block(block_id);
        }
        self.block_ids.clear();
        self.num_tokens = 0;
    }
}
