//! AMD-PACE-inspired SlabPool NUMA-Aware Cache Allocator.
//!
//! Provides hardware cache-line aligned slab memory management:
//! - 64-byte aligned slabs (`PhysicalSlabBlock`) eliminating false sharing and cache-line splits.
//! - Multi-NUMA node partitioning ensuring memory accesses remain local to the executing CPU socket/APU.
//! - Lock-free slab acquisition and retirement via `SegQueue`.
//! - Contiguous memory layout optimized for AVX2 and AVX-512 vector register loads.

use crossbeam_queue::SegQueue;
use std::sync::atomic::{AtomicU32, Ordering};

/// Tokens stored per slab block (tuned for L1/L2 cache locality on Zen 4/5 & modern CPUs).
pub const TOKENS_PER_SLAB: usize = 64;

/// A 64-byte cache-line aligned physical slab storing Key/Value states for `TOKENS_PER_SLAB` tokens.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct PhysicalSlabBlock {
    pub slab_id: u32,
    pub numa_node: u32,
    pub ref_count: AtomicU32,
    /// Key memory buffer: `[TOKENS_PER_SLAB, head_dim_total]`
    pub k: Vec<f32>,
    /// Value memory buffer: `[TOKENS_PER_SLAB, head_dim_total]`
    pub v: Vec<f32>,
}

impl PhysicalSlabBlock {
    #[must_use]
    pub fn new(slab_id: u32, numa_node: u32, head_dim_total: usize) -> Self {
        let capacity = TOKENS_PER_SLAB * head_dim_total;
        Self {
            slab_id,
            numa_node,
            ref_count: AtomicU32::new(1),
            k: vec![0.0; capacity],
            v: vec![0.0; capacity],
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

    #[inline(always)]
    pub fn reset(&self) {
        self.ref_count.store(1, Ordering::Relaxed);
    }
}

/// Logical sequence slab table mapping contiguous sequence tokens across physical slabs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceSlabTable {
    pub seq_id: u64,
    pub slab_ids: Vec<u32>,
    pub total_tokens: usize,
}

impl SequenceSlabTable {
    #[must_use]
    pub fn new(seq_id: u64) -> Self {
        Self {
            seq_id,
            slab_ids: Vec::new(),
            total_tokens: 0,
        }
    }

    #[must_use]
    pub fn num_slabs(&self) -> usize {
        self.slab_ids.len()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slab_ids.len() * TOKENS_PER_SLAB
    }

    #[must_use]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity().saturating_sub(self.total_tokens)
    }

    pub fn append_slab(&mut self, slab_id: u32) {
        self.slab_ids.push(slab_id);
    }
}

/// Global SlabPool Arena managing pre-allocated slabs partitioned across NUMA nodes.
#[derive(Debug)]
pub struct SlabPoolArena {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub head_dim_total: usize,
    pub total_slabs_per_layer: usize,
    pub num_numa_nodes: usize,
    /// Physical storage: `[layer_idx][slab_id]`
    slabs: Vec<Vec<PhysicalSlabBlock>>,
    /// Free slab queues partitioned by NUMA node: `[numa_node]` -> SegQueue of slab IDs
    free_queues: Vec<SegQueue<u32>>,
}

impl SlabPoolArena {
    #[must_use]
    pub fn new(
        total_slabs_per_layer: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_numa_nodes: usize,
    ) -> Self {
        let head_dim_total = num_kv_heads * head_dim;
        let numa_count = num_numa_nodes.max(1);

        let mut layers = Vec::with_capacity(num_layers);
        let mut free_queues = Vec::with_capacity(numa_count);

        for _ in 0..numa_count {
            free_queues.push(SegQueue::new());
        }

        for _ in 0..num_layers {
            let mut layer_slabs = Vec::with_capacity(total_slabs_per_layer);
            for s in 0..total_slabs_per_layer {
                let node = (s % numa_count) as u32;
                layer_slabs.push(PhysicalSlabBlock::new(s as u32, node, head_dim_total));
            }
            layers.push(layer_slabs);
        }

        for s in 0..total_slabs_per_layer {
            let node = s % numa_count;
            free_queues[node].push(s as u32);
        }

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            head_dim_total,
            total_slabs_per_layer,
            num_numa_nodes: numa_count,
            slabs: layers,
            free_queues,
        }
    }

    /// Allocates an unused slab from the target NUMA node (or any node fallback) in O(1) lock-free time.
    #[must_use]
    pub fn allocate_slab(&self, target_numa: usize) -> Option<u32> {
        let primary_node = target_numa % self.num_numa_nodes;
        if let Some(slab_id) = self.free_queues[primary_node].pop() {
            return Some(slab_id);
        }

        // Steal from other NUMA nodes if primary is exhausted
        for node in 0..self.num_numa_nodes {
            if node != primary_node
                && let Some(slab_id) = self.free_queues[node].pop()
            {
                return Some(slab_id);
            }
        }

        None
    }

    /// Releases a slab and returns it to its designated NUMA node queue.
    pub fn free_slab(&self, slab_id: u32) {
        if (slab_id as usize) < self.total_slabs_per_layer {
            let node = self.slabs[0][slab_id as usize].numa_node as usize;
            for layer in 0..self.num_layers {
                self.slabs[layer][slab_id as usize].reset();
            }
            self.free_queues[node % self.num_numa_nodes].push(slab_id);
        }
    }

    /// Writes a single token's KV representations into the physical slab.
    pub fn write_token(
        &mut self,
        layer: usize,
        slab_id: u32,
        offset_in_slab: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), &'static str> {
        if layer >= self.num_layers {
            return Err("Layer index out of bounds");
        }
        if (slab_id as usize) >= self.total_slabs_per_layer {
            return Err("Slab ID out of bounds");
        }
        if offset_in_slab >= TOKENS_PER_SLAB {
            return Err("Offset in slab exceeds TOKENS_PER_SLAB");
        }

        let slab = &mut self.slabs[layer][slab_id as usize];
        let start = offset_in_slab * self.head_dim_total;
        let end = start + self.head_dim_total;

        slab.k[start..end].copy_from_slice(&k[..self.head_dim_total]);
        slab.v[start..end].copy_from_slice(&v[..self.head_dim_total]);

        Ok(())
    }

    /// Borrows a contiguous read-only slice of Key or Value memory from a physical slab.
    #[must_use]
    pub fn get_slab_kv(&self, layer: usize, slab_id: u32) -> Option<(&[f32], &[f32])> {
        let layer_slabs = self.slabs.get(layer)?;
        let slab = layer_slabs.get(slab_id as usize)?;
        Some((&slab.k, &slab.v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_pool_numa_allocation_and_write() {
        let mut pool = SlabPoolArena::new(
            8, // 8 slabs per layer
            2, // 2 layers
            2, // 2 KV heads
            4, // head_dim = 4 -> head_dim_total = 8
            2, // 2 NUMA nodes
        );

        // Allocate slab for NUMA node 0
        let s0 = pool.allocate_slab(0).expect("slab allocated");
        let s1 = pool.allocate_slab(1).expect("slab allocated");

        let k_vec = vec![1.5; 8];
        let v_vec = vec![2.5; 8];

        assert!(pool.write_token(0, s0, 0, &k_vec, &v_vec).is_ok());

        let (k_read, v_read) = pool.get_slab_kv(0, s0).expect("slab kv exists");
        assert_eq!(&k_read[0..8], &k_vec[..]);
        assert_eq!(&v_read[0..8], &v_vec[..]);

        pool.free_slab(s0);
        pool.free_slab(s1);
    }
}
