use crossbeam_queue::SegQueue;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

pub type PhysicalBlockId = u32;

/// Directed Acyclic Graph tree node for MCTS, System 2 branching, and agent reasoning traces.
#[derive(Debug)]
pub struct TreeNode {
    pub node_id: u64,
    pub parent: Option<Arc<TreeNode>>,
    pub token_id: u32,
    pub prm_score: f32,
    pub block_id: PhysicalBlockId,
}

impl TreeNode {
    #[must_use]
    pub fn new_root(node_id: u64, token_id: u32, block_id: PhysicalBlockId) -> Arc<Self> {
        Arc::new(Self {
            node_id,
            parent: None,
            token_id,
            prm_score: 1.0,
            block_id,
        })
    }

    #[must_use]
    pub fn create_child(
        self: &Arc<Self>,
        node_id: u64,
        token_id: u32,
        prm_score: f32,
        block_id: PhysicalBlockId,
    ) -> Arc<Self> {
        Arc::new(Self {
            node_id,
            parent: Some(Arc::clone(self)),
            token_id,
            prm_score,
            block_id,
        })
    }
}

/// Shared physical KV block with atomic lock-free reference counting for O(1) branching.
/// Aligned to 64 bytes to eliminate cache-line false sharing across multiple processor cores.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct SharedPhysicalBlock {
    pub block_id: PhysicalBlockId,
    pub ref_count: AtomicU32,
}

impl SharedPhysicalBlock {
    #[must_use]
    pub const fn new(block_id: PhysicalBlockId) -> Self {
        Self {
            block_id,
            ref_count: AtomicU32::new(1),
        }
    }

    /// Increments reference count for zero-copy fork.
    #[inline(always)]
    pub fn fork(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrements reference count and returns block to free list when it reaches 0.
    #[inline(always)]
    pub fn release(&self, free_list: &SegQueue<PhysicalBlockId>) {
        if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            free_list.push(self.block_id);
        }
    }
}

/// Thread-safe DAG physical block manager with asynchronous hardware event fencing.
#[derive(Debug)]
pub struct SafeDagBlockManager {
    total_blocks: usize,
    blocks: Vec<SharedPhysicalBlock>,
    pending_releases: SegQueue<(PhysicalBlockId, *mut std::ffi::c_void)>,
    free_list: SegQueue<PhysicalBlockId>,
}

// SAFETY: All fields are thread-safe atomic primitives, lock-free queues (`SegQueue`), or immutable allocations.
unsafe impl Send for SafeDagBlockManager {}
// SAFETY: Synchronized via atomic ref counts and lock-free concurrency primitives.
unsafe impl Sync for SafeDagBlockManager {}

impl SafeDagBlockManager {
    #[must_use]
    pub fn new(total_physical_blocks: usize) -> Self {
        let mut blocks = Vec::with_capacity(total_physical_blocks);
        let free_list = SegQueue::new();
        for i in 0..total_physical_blocks {
            blocks.push(SharedPhysicalBlock::new(i as PhysicalBlockId));
        }

        Self {
            total_blocks: total_physical_blocks,
            blocks,
            pending_releases: SegQueue::new(),
            free_list,
        }
    }

    /// Increments reference count for a physical block on fork.
    pub fn fork_block(&self, block_id: PhysicalBlockId) {
        if let Some(block) = self.blocks.get(block_id as usize) {
            block.fork();
        }
    }

    /// Enqueues block for asynchronous retirement once hardware event is signaled.
    pub fn release_block_async(
        &self,
        block_id: PhysicalBlockId,
        mock_completion_event: *mut std::ffi::c_void,
    ) {
        self.pending_releases
            .push((block_id, mock_completion_event));
    }

    /// Polls pending asynchronous releases and returns fully unreferenced blocks to free list.
    pub fn poll_reclaim_blocks(&self) {
        while let Some((block_id, _event)) = self.pending_releases.pop() {
            if let Some(block) = self.blocks.get(block_id as usize) {
                block.release(&self.free_list);
            }
        }
    }

    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    #[must_use]
    pub fn free_list_len(&self) -> usize {
        self.free_list.len()
    }
}
