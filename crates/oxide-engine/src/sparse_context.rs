//! SparseEngine-inspired 128k+ Long-Context Hierarchical Block-Sparse Engine.
//!
//! Enables linear compute and memory scaling up to 128,000+ context tokens:
//! - Partitions KV cache sequences into fixed-size coarse blocks (e.g. 64 tokens/block).
//! - Protects initial prompt sink blocks (attention sinks) and local recency blocks.
//! - Computes query-to-block importance estimators to dynamically route to top-K blocks.
//! - Enforces method-specific prefill scheduling (`AllChunked` vs `LongBs1FullShortBatch`).

use serde::{Deserialize, Serialize};

/// Tokens per block for coarse-grained block-sparse attention.
pub const TOKENS_PER_SPARSE_BLOCK: usize = 64;

/// Registry-owned prefill scheduling policy for sparse context workloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparsePrefillPolicy {
    /// Standard chunked prefill for sequences below 8,192 tokens.
    AllChunked { chunk_size: usize },
    /// Dedicated batch_size=1 full-prefill pass for ultra-long context (>8,192 up to 128,000+ tokens)
    /// to construct the block centroid index before concurrent decoding.
    LongBs1FullShortBatch { full_prefill_threshold: usize },
}

impl Default for SparsePrefillPolicy {
    fn default() -> Self {
        Self::LongBs1FullShortBatch {
            full_prefill_threshold: 8192,
        }
    }
}

/// Configuration parameters for block-sparse attention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockSparseConfig {
    pub max_context_tokens: usize,
    pub top_k_blocks: usize,
    pub sink_blocks: usize,
    pub recent_blocks: usize,
    pub prefill_policy: SparsePrefillPolicy,
}

impl Default for BlockSparseConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 131_072, // 128K context
            top_k_blocks: 16,            // Evaluate top 16 blocks (1,024 active tokens)
            sink_blocks: 2,              // 128 initial sink tokens
            recent_blocks: 4,            // 256 sliding-window recency tokens
            prefill_policy: SparsePrefillPolicy::default(),
        }
    }
}

/// Summary centroid representation of a 64-token KV block.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockCentroid {
    pub block_id: usize,
    pub start_pos: usize,
    pub num_tokens: usize,
    /// Average key representation for fast query-to-block coarse relevance scoring
    pub centroid_key: Vec<f32>,
}

/// Sequence-level sparse block index tracking all blocks and routing attention.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceSparseBlockIndex {
    pub seq_id: u64,
    pub total_tokens: usize,
    pub blocks: Vec<SparseBlockCentroid>,
    pub config: BlockSparseConfig,
}

impl SequenceSparseBlockIndex {
    #[must_use]
    pub fn new(seq_id: u64, config: BlockSparseConfig) -> Self {
        Self {
            seq_id,
            total_tokens: 0,
            blocks: Vec::new(),
            config,
        }
    }

    /// Registers a new token into the block index, creating or updating block centroids.
    pub fn register_token(&mut self, k: &[f32]) {
        let current_pos = self.total_tokens;
        let block_idx = current_pos / TOKENS_PER_SPARSE_BLOCK;

        if block_idx >= self.blocks.len() {
            self.blocks.push(SparseBlockCentroid {
                block_id: block_idx,
                start_pos: current_pos,
                num_tokens: 1,
                centroid_key: k.to_vec(),
            });
        } else {
            let block = &mut self.blocks[block_idx];
            block.num_tokens += 1;
            // Running average update
            let count = block.num_tokens as f32;
            for (c, &val) in block.centroid_key.iter_mut().zip(k.iter()) {
                *c = (*c * (count - 1.0) + val) / count;
            }
        }

        self.total_tokens += 1;
    }

    /// Selects the top-K active blocks to evaluate for the current query vector.
    ///
    /// Guarantees that initial sink blocks and recent sliding-window blocks are always selected.
    #[must_use]
    pub fn select_active_blocks(&self, query: &[f32]) -> Vec<usize> {
        let total_blocks = self.blocks.len();
        if total_blocks <= self.config.top_k_blocks {
            return (0..total_blocks).collect();
        }

        let sink_end = self.config.sink_blocks.min(total_blocks);
        let recent_start = total_blocks.saturating_sub(self.config.recent_blocks);

        let mut selected = Vec::with_capacity(self.config.top_k_blocks);

        // 1. Add mandatory sink blocks
        for b in 0..sink_end {
            selected.push(b);
        }

        // 2. Add mandatory recent blocks
        for b in recent_start..total_blocks {
            if !selected.contains(&b) {
                selected.push(b);
            }
        }

        // 3. Score middle candidate blocks
        let mut middle_candidates = Vec::new();
        for (b_idx, block) in self.blocks.iter().enumerate() {
            if b_idx >= sink_end && b_idx < recent_start {
                let mut dot = 0.0f32;
                for (q, c) in query.iter().zip(block.centroid_key.iter()) {
                    dot += q * c;
                }
                middle_candidates.push((b_idx, dot));
            }
        }

        // Sort descending by score
        middle_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let available_slots = self.config.top_k_blocks.saturating_sub(selected.len());
        for (b, _) in middle_candidates.into_iter().take(available_slots) {
            selected.push(b);
        }

        selected.sort_unstable();
        selected
    }

    /// Determines the optimal prefill execution strategy based on prompt length.
    #[must_use]
    pub fn determine_prefill_strategy(&self, prompt_len: usize) -> &'static str {
        match self.config.prefill_policy {
            SparsePrefillPolicy::AllChunked { .. } => "AllChunked",
            SparsePrefillPolicy::LongBs1FullShortBatch {
                full_prefill_threshold,
            } => {
                if prompt_len >= full_prefill_threshold {
                    "LongBs1Full" // Dedicated full prefill for 128K context
                } else {
                    "ChunkedBatch"
                }
            }
        }
    }
}

/// Global Hierarchical Block-Sparse Context Manager.
#[derive(Debug, Default)]
pub struct SparseContextManager {
    pub config: BlockSparseConfig,
}

impl SparseContextManager {
    #[must_use]
    pub fn new(config: BlockSparseConfig) -> Self {
        Self { config }
    }

    /// Creates a new sequence index for tracking long context.
    #[must_use]
    pub fn create_sequence_index(&self, seq_id: u64) -> SequenceSparseBlockIndex {
        SequenceSparseBlockIndex::new(seq_id, self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sparse_routing_and_policy() {
        let config = BlockSparseConfig {
            max_context_tokens: 131_072,
            top_k_blocks: 4,
            sink_blocks: 1,   // block 0 protected
            recent_blocks: 1, // last block protected
            prefill_policy: SparsePrefillPolicy::LongBs1FullShortBatch {
                full_prefill_threshold: 8192,
            },
        };

        let mut index = SequenceSparseBlockIndex::new(101, config);

        // Populate 10 blocks (640 tokens)
        for i in 0..640 {
            let k = if (64..128).contains(&i) {
                vec![10.0f32; 4] // Block 1 has high similarity
            } else {
                vec![0.1f32; 4]
            };
            index.register_token(&k);
        }

        assert_eq!(index.blocks.len(), 10);

        let query = vec![1.0f32; 4];
        let active_blocks = index.select_active_blocks(&query);

        // Must select top_k=4 blocks
        assert_eq!(active_blocks.len(), 4);
        // Block 0 (sink) and Block 9 (recent) MUST be included
        assert!(active_blocks.contains(&0));
        assert!(active_blocks.contains(&9));
        // Block 1 (high score) MUST be included
        assert!(active_blocks.contains(&1));

        // Prefill policy verification
        assert_eq!(index.determine_prefill_strategy(1000), "ChunkedBatch");
        assert_eq!(index.determine_prefill_strategy(16384), "LongBs1Full");
    }
}
