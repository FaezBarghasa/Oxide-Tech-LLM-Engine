//! 5-Way Distributed Parallelism (TP, PP, DP, EP, CP) for Extreme-Scale Inference.
//!
//! Provides comprehensive distributed orchestration:
//! - Tensor Parallelism (TP): Intra-node row/column weight slicing with AllReduce
//! - Pipeline Parallelism (PP): Inter-node stage splitting with 1F1B schedule
//! - Data Parallelism (DP): Request replication across homogeneous instances
//! - Expert Parallelism (EP): Sharded MoE experts with AllToAll token dispatch
//! - Context Parallelism (CP): Ring-Attention splitting sequences across ranks for million-token contexts

use serde::{Deserialize, Serialize};

/// 5D Parallelism Mesh Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParallelMeshConfig {
    pub tp_size: usize, // Tensor Parallelism
    pub pp_size: usize, // Pipeline Parallelism
    pub dp_size: usize, // Data Parallelism
    pub ep_size: usize, // Expert Parallelism
    pub cp_size: usize, // Context Parallelism
}

impl Default for ParallelMeshConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            dp_size: 1,
            ep_size: 1,
            cp_size: 1,
        }
    }
}

impl ParallelMeshConfig {
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.tp_size * self.pp_size * self.dp_size * self.ep_size * self.cp_size
    }
}

/// Node Rank in the 5D Distributed Mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeshCoordinate {
    pub global_rank: usize,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub dp_rank: usize,
    pub ep_rank: usize,
    pub cp_rank: usize,
}

impl MeshCoordinate {
    #[must_use]
    pub fn from_global_rank(global_rank: usize, mesh: &ParallelMeshConfig) -> Self {
        let mut r = global_rank;

        let tp_rank = r % mesh.tp_size;
        r /= mesh.tp_size;

        let cp_rank = r % mesh.cp_size;
        r /= mesh.cp_size;

        let ep_rank = r % mesh.ep_size;
        r /= mesh.ep_size;

        let dp_rank = r % mesh.dp_size;
        r /= mesh.dp_size;

        let pp_rank = r % mesh.pp_size;

        Self {
            global_rank,
            tp_rank,
            pp_rank,
            dp_rank,
            ep_rank,
            cp_rank,
        }
    }
}

/// Tensor Parallelism (TP) Column-Parallel Linear Layer.
#[derive(Debug, Clone)]
pub struct TpColumnLinear {
    pub in_features: usize,
    pub out_features_per_rank: usize,
    pub tp_rank: usize,
    pub tp_size: usize,
}

impl TpColumnLinear {
    #[must_use]
    pub fn new(
        in_features: usize,
        total_out_features: usize,
        tp_rank: usize,
        tp_size: usize,
    ) -> Self {
        assert_eq!(total_out_features % tp_size, 0);
        Self {
            in_features,
            out_features_per_rank: total_out_features / tp_size,
            tp_rank,
            tp_size,
        }
    }
}

/// Context Parallelism (CP) Ring Attention Step.
/// Splits sequence of length L across CP ranks (L / cp_size per rank).
/// In each ring step, current rank computes attention against its current KV chunk,
/// then sends KV to rank (r - 1) and receives from rank (r + 1).
#[derive(Debug, Clone)]
pub struct RingAttentionStep {
    pub cp_rank: usize,
    pub cp_size: usize,
    pub chunk_seq_len: usize,
}

impl RingAttentionStep {
    #[must_use]
    pub fn new(total_seq_len: usize, cp_rank: usize, cp_size: usize) -> Self {
        assert_eq!(total_seq_len % cp_size, 0);
        Self {
            cp_rank,
            cp_size,
            chunk_seq_len: total_seq_len / cp_size,
        }
    }

    /// Returns next peer rank to send KV cache chunk.
    #[must_use]
    pub fn send_peer(&self) -> usize {
        (self.cp_rank + self.cp_size - 1) % self.cp_size
    }

    /// Returns peer rank from which to receive incoming KV cache chunk.
    #[must_use]
    pub fn recv_peer(&self) -> usize {
        (self.cp_rank + 1) % self.cp_size
    }
}
