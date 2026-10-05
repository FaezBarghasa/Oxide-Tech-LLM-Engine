#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of token slots per KV cache block.
pub const TOKENS_PER_KV_BLOCK: usize = 16;

/// Storage tier location in the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTierLocation {
    Tier1DeviceVram,
    Tier2HostRam,
    Tier3ExternalStorage,
}

/// Metadata descriptor for a single KV cache block with prefix-hash tracking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBlockDescriptor {
    pub block_id: u32,
    pub prefix_hash: u64,
    pub token_count: u16,
    pub layer_idx: u16,
    pub head_dim: u16,
    pub location: CacheTierLocation,
    pub access_counter: u64,
}

/// Serialized payload for distributed KV cache block transfer and network reuse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedKvBlockPayload {
    pub descriptor: KvBlockDescriptor,
    pub tokens: [u32; TOKENS_PER_KV_BLOCK],
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub checksum: u32,
}

impl DistributedKvBlockPayload {
    #[must_use]
    pub fn compute_checksum(&self) -> u32 {
        let mut sum: u32 = (self.descriptor.prefix_hash & 0xFFFF_FFFF) as u32;
        for t in self.tokens {
            sum = sum.wrapping_add(t);
        }
        for v in &self.k_data {
            sum = sum.wrapping_add(v.to_bits());
        }
        for v in &self.v_data {
            sum = sum.wrapping_add(v.to_bits());
        }
        sum
    }

    #[must_use]
    pub fn verify_integrity(&self) -> bool {
        self.compute_checksum() == self.checksum
    }
}

/// 3-Tier Hierarchical KV Caching Manager:
/// - Tier 1: Device VRAM / SRAM / LPDDR5x (Fastest, hot decoding)
/// - Tier 2: Host RAM Pinned Arena (Warm paging & staging)
/// - Tier 3: External Storage / NVMe Flash (Cold long-term prefix bank)
#[derive(Debug)]
pub struct HierarchicalKvCache {
    tier1_device_capacity_blocks: usize,
    tier2_host_capacity_blocks: usize,
    tier3_storage_capacity_blocks: usize,

    device_blocks: HashMap<u32, Vec<f32>>,
    host_blocks: HashMap<u32, Vec<f32>>,
    storage_blocks: HashMap<u32, Vec<f32>>,

    prefix_index: HashMap<u64, u32>, // prefix_hash -> block_id
    block_registry: HashMap<u32, KvBlockDescriptor>,
    next_block_id: u32,
    access_clock: AtomicU64,
}

impl HierarchicalKvCache {
    #[must_use]
    pub fn new(
        tier1_device_blocks: usize,
        tier2_host_blocks: usize,
        tier3_storage_blocks: usize,
    ) -> Self {
        Self {
            tier1_device_capacity_blocks: tier1_device_blocks,
            tier2_host_capacity_blocks: tier2_host_blocks,
            tier3_storage_capacity_blocks: tier3_storage_blocks,
            device_blocks: HashMap::with_capacity(tier1_device_blocks),
            host_blocks: HashMap::with_capacity(tier2_host_blocks),
            storage_blocks: HashMap::with_capacity(tier3_storage_blocks),
            prefix_index: HashMap::new(),
            block_registry: HashMap::new(),
            next_block_id: 1,
            access_clock: AtomicU64::new(1),
        }
    }

    /// Computes deterministic prefix hash for a sequence of tokens.
    #[must_use]
    pub fn compute_prefix_hash(tokens: &[u32], parent_hash: u64) -> u64 {
        let mut hash = parent_hash ^ 0xcbf2_9ce4_8422_2325;
        for &tok in tokens {
            hash ^= u64::from(tok);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash
    }

    /// Queries the hierarchical cache for an existing reusable prefix block.
    /// If found in Tier 2 or Tier 3, automatically promotes the block to Tier 1.
    pub fn find_or_promote_prefix(
        &mut self,
        prefix_hash: u64,
    ) -> Option<(&KvBlockDescriptor, &[f32])> {
        let block_id = *self.prefix_index.get(&prefix_hash)?;
        let access = self.access_clock.fetch_add(1, Ordering::Relaxed);

        let location = self.block_registry.get(&block_id)?.location;

        match location {
            CacheTierLocation::Tier1DeviceVram => {}
            CacheTierLocation::Tier2HostRam => {
                if let Some(data) = self.host_blocks.remove(&block_id) {
                    self.ensure_device_capacity();
                    self.device_blocks.insert(block_id, data);
                    if let Some(d) = self.block_registry.get_mut(&block_id) {
                        d.location = CacheTierLocation::Tier1DeviceVram;
                    }
                }
            }
            CacheTierLocation::Tier3ExternalStorage => {
                if let Some(data) = self.storage_blocks.remove(&block_id) {
                    self.ensure_device_capacity();
                    self.device_blocks.insert(block_id, data);
                    if let Some(d) = self.block_registry.get_mut(&block_id) {
                        d.location = CacheTierLocation::Tier1DeviceVram;
                    }
                }
            }
        }

        if let Some(desc) = self.block_registry.get_mut(&block_id) {
            desc.access_counter = access;
        }

        let data_ptr = self.device_blocks.get(&block_id)?;
        let desc_ref = self.block_registry.get(&block_id)?;
        Some((desc_ref, data_ptr.as_slice()))
    }

    /// Allocates or inserts a new KV cache block into Tier 1 (Device VRAM).
    pub fn allocate_block(
        &mut self,
        prefix_hash: u64,
        layer_idx: u16,
        head_dim: u16,
        data: Vec<f32>,
    ) -> Result<u32> {
        self.ensure_device_capacity();

        let block_id = self.next_block_id;
        self.next_block_id += 1;
        let access = self.access_clock.fetch_add(1, Ordering::Relaxed);

        let descriptor = KvBlockDescriptor {
            block_id,
            prefix_hash,
            token_count: TOKENS_PER_KV_BLOCK as u16,
            layer_idx,
            head_dim,
            location: CacheTierLocation::Tier1DeviceVram,
            access_counter: access,
        };

        self.device_blocks.insert(block_id, data);
        self.block_registry.insert(block_id, descriptor);
        self.prefix_index.insert(prefix_hash, block_id);

        Ok(block_id)
    }

    /// Evicts cold blocks down the hierarchy (Tier 1 $\rightarrow$ Tier 2 $\rightarrow$ Tier 3).
    fn ensure_device_capacity(&mut self) {
        if self.device_blocks.len() >= self.tier1_device_capacity_blocks {
            if let Some((&victim_id, _)) = self
                .block_registry
                .iter()
                .filter(|(_, d)| d.location == CacheTierLocation::Tier1DeviceVram)
                .min_by_key(|(_, d)| d.access_counter)
            {
                if let Some(data) = self.device_blocks.remove(&victim_id) {
                    self.ensure_host_capacity();
                    self.host_blocks.insert(victim_id, data);
                    if let Some(desc) = self.block_registry.get_mut(&victim_id) {
                        desc.location = CacheTierLocation::Tier2HostRam;
                    }
                }
            }
        }
    }

    fn ensure_host_capacity(&mut self) {
        if self.host_blocks.len() >= self.tier2_host_capacity_blocks {
            if let Some((&victim_id, _)) = self
                .block_registry
                .iter()
                .filter(|(_, d)| d.location == CacheTierLocation::Tier2HostRam)
                .min_by_key(|(_, d)| d.access_counter)
            {
                if let Some(data) = self.host_blocks.remove(&victim_id) {
                    if self.storage_blocks.len() < self.tier3_storage_capacity_blocks {
                        self.storage_blocks.insert(victim_id, data);
                        if let Some(desc) = self.block_registry.get_mut(&victim_id) {
                            desc.location = CacheTierLocation::Tier3ExternalStorage;
                        }
                    } else if let Some(desc) = self.block_registry.remove(&victim_id) {
                        self.prefix_index.remove(&desc.prefix_hash);
                    }
                }
            }
        }
    }

    /// Exports a cache block into a distributed network payload for remote cluster nodes.
    pub fn export_distributed_block(
        &self,
        block_id: u32,
        tokens: [u32; TOKENS_PER_KV_BLOCK],
    ) -> Result<DistributedKvBlockPayload> {
        let descriptor = self.block_registry.get(&block_id).cloned().ok_or_else(|| {
            EngineError::BackendError(format!("Block {block_id} not found for export"))
        })?;

        let raw_data = match descriptor.location {
            CacheTierLocation::Tier1DeviceVram => self.device_blocks.get(&block_id),
            CacheTierLocation::Tier2HostRam => self.host_blocks.get(&block_id),
            CacheTierLocation::Tier3ExternalStorage => self.storage_blocks.get(&block_id),
        }
        .ok_or_else(|| EngineError::BackendError(format!("Block memory {block_id} missing")))?;

        let half = raw_data.len() / 2;
        let k_data = raw_data[..half].to_vec();
        let v_data = raw_data[half..].to_vec();

        let mut payload = DistributedKvBlockPayload {
            descriptor,
            tokens,
            k_data,
            v_data,
            checksum: 0,
        };
        payload.checksum = payload.compute_checksum();

        Ok(payload)
    }

    /// Imports a distributed cache block from a remote node with integrity verification.
    pub fn import_distributed_block(&mut self, payload: DistributedKvBlockPayload) -> Result<u32> {
        if !payload.verify_integrity() {
            return Err(EngineError::BackendError(
                "Distributed KV payload checksum mismatch".into(),
            ));
        }

        let mut combined_data = Vec::with_capacity(payload.k_data.len() + payload.v_data.len());
        combined_data.extend_from_slice(&payload.k_data);
        combined_data.extend_from_slice(&payload.v_data);

        let prefix_hash = payload.descriptor.prefix_hash;
        let layer_idx = payload.descriptor.layer_idx;
        let head_dim = payload.descriptor.head_dim;

        self.allocate_block(prefix_hash, layer_idx, head_dim, combined_data)
    }

    #[must_use]
    pub fn tier1_count(&self) -> usize {
        self.device_blocks.len()
    }

    #[must_use]
    pub fn tier2_count(&self) -> usize {
        self.host_blocks.len()
    }

    #[must_use]
    pub fn tier3_count(&self) -> usize {
        self.storage_blocks.len()
    }
}
