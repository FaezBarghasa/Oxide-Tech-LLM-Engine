# Directed Acyclic Graph (DAG) Tree-Native Reasoning & In-Flight Pruning

## Overview

Complex reasoning (System 2 thinking, Monte Carlo Tree Search (MCTS), autonomous agent planning) requires generating and evaluating multiple hypothesis branches in parallel. Standard linear runtimes duplicate prompt KV caches for every branch, causing severe VRAM exhaustion.

`Oxide-Tech-LLM-Engine` transforms the execution pipeline into a **native Directed Acyclic Graph (DAG) tree processor** (`crates/oxide-core/src/dag.rs`), scaling to $100,000+$ active reasoning branches with zero-copy thought forks.

---

## 1. Lock-Free Copy-on-Write (CoW) Block Hierarchy

KV blocks in tree search are shared across ancestral paths. When a sequence forks into multiple candidate branches, the shared prompt prefix blocks are referenced by all children without copying device VRAM.

In `crates/oxide-core/src/dag.rs`:

```rust
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub type PhysicalBlockId = u32;

#[derive(Debug)]
pub struct TreeNode {
    pub node_id: u64,
    pub parent: Option<Arc<TreeNode>>,
    pub token_id: u32,
    pub prm_score: f32,
    pub block_id: PhysicalBlockId,
}

#[derive(Debug)]
pub struct SharedPhysicalBlock {
    pub block_id: PhysicalBlockId,
    pub ref_count: AtomicU32,
}

impl SharedPhysicalBlock {
    #[inline(always)]
    pub fn fork(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn release(&self, free_list: &crossbeam::queue::SegQueue<PhysicalBlockId>) {
        if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            free_list.push(self.block_id);
        }
    }
}
```

- **Branch Forking**: $O(1)$ atomic increment of the parent block reference count.
- **Branch Recycling**: When a branch is rejected, decrementing the reference count automatically returns unreferenced physical blocks to the lock-free free list without stopping the main decode loop.

---

## 2. Topological DAG Attention Mask Synthesis

To evaluate multiple branches in a single GPU forward pass, arbitrary tree topologies are mapped into a unified packed attention mask:

$$\text{AttentionMask}[i, j] = \begin{cases} 1 & \text{if token } j \in \text{Ancestors}(i) \\ 0 & \text{otherwise} \end{cases}$$

All active leaf nodes across disparate branches are scheduled in a single batched forward step. Shared prompt prefixes are computed once, saturating tensor cores without redundant computation.

---

## 3. Asynchronous In-Flight Process Reward Model (PRM)

During reasoning generation, step boundaries (e.g., `"\n\n"`, `"Step N:"`, tool call markers) trigger lightweight co-scheduled verifier heads:
1. The PRM computes step confidence scores $\sigma \in [0.0, 1.0]$.
2. If $\sigma < \tau$ (confidence threshold), the branch is flagged for early pruning.
3. The worker actor decrements the reference counts of the branch's unique KV blocks, releasing VRAM back to the pool mid-flight.
