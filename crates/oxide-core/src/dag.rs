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
#[derive(Debug)]
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
