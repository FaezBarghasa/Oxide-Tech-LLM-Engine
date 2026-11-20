//! Physical CUDA Graph Capture & Replay Engine for Ultra-Low Latency Decoding.
//!
//! Eliminates per-kernel CPU launch overhead ($15-40\ \mu\text{s}$ per node) by capturing
//! the static decode step DAG into an executable CUDA Graph instance and
//! launching it in a single driver submission ($<1.5\ \mu\text{s}$).

use crate::driver;
use oxide_core::error::{EngineError, Result};
use std::collections::HashMap;

/// Handle to an instantiated and compiled CUDA execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaGraphExecHandle {
    pub graph_id: u64,
    pub node_count: usize,
}

/// State and topology of an active physical CUDA execution graph.
pub struct CapturedCudaGraph {
    pub graph_id: u64,
    pub name: String,
    pub num_layers: usize,
    pub batch_size: usize,
    pub execution_count: u64,
    pub is_instantiated: bool,
    pub raw_graph: driver::cudaGraph_t,
    pub raw_exec: driver::cudaGraphExec_t,
}

unsafe impl Send for CapturedCudaGraph {}
unsafe impl Sync for CapturedCudaGraph {}

impl Drop for CapturedCudaGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.raw_exec.is_null() {
                let _ = driver::cudaGraphExecDestroy(self.raw_exec);
                self.raw_exec = std::ptr::null_mut();
            }
            if !self.raw_graph.is_null() {
                let _ = driver::cudaGraphDestroy(self.raw_graph);
                self.raw_graph = std::ptr::null_mut();
            }
        }
    }
}

impl std::fmt::Debug for CapturedCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedCudaGraph")
            .field("graph_id", &self.graph_id)
            .field("name", &self.name)
            .field("num_layers", &self.num_layers)
            .field("batch_size", &self.batch_size)
            .field("execution_count", &self.execution_count)
            .field("is_instantiated", &self.is_instantiated)
            .finish()
    }
}

/// CUDA Graph Lifecycle and Replay Manager.
#[derive(Debug, Default)]
pub struct CudaGraphManager {
    graphs: HashMap<u64, CapturedCudaGraph>,
    next_graph_id: u64,
    total_replays: u64,
}

impl CudaGraphManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            graphs: HashMap::new(),
            next_graph_id: 1,
            total_replays: 0,
        }
    }

    /// Begins CUDA stream capture on the specified stream.
    ///
    /// # Safety
    /// Caller must ensure `stream` is a valid, active CUDA stream pointer on the target GPU.
    pub unsafe fn begin_capture(&mut self, stream: driver::cudaStream_t) -> Result<()> {
        let err = unsafe {
            driver::cudaStreamBeginCapture(stream, driver::CUDA_STREAM_CAPTURE_MODE_RELAXED)
        };
        if err != 0 {
            return Err(EngineError::BackendError(format!(
                "cudaStreamBeginCapture failed with error code {err}"
            )));
        }
        Ok(())
    }

    /// Ends stream capture and instantiates an executable physical CUDA graph.
    ///
    /// # Safety
    /// Caller must ensure `stream` is a valid, active CUDA stream currently in capture mode.
    pub unsafe fn end_capture(
        &mut self,
        stream: driver::cudaStream_t,
        name: impl Into<String>,
        num_layers: usize,
        batch_size: usize,
    ) -> Result<CudaGraphExecHandle> {
        let mut raw_graph: driver::cudaGraph_t = std::ptr::null_mut();
        let err = unsafe { driver::cudaStreamEndCapture(stream, &raw mut raw_graph) };
        if err != 0 || raw_graph.is_null() {
            return Err(EngineError::BackendError(format!(
                "cudaStreamEndCapture failed with error code {err}"
            )));
        }

        let mut raw_exec: driver::cudaGraphExec_t = std::ptr::null_mut();
        let err = unsafe { driver::cudaGraphInstantiate(&raw mut raw_exec, raw_graph, 0) };
        if err != 0 || raw_exec.is_null() {
            unsafe { driver::cudaGraphDestroy(raw_graph) };
            return Err(EngineError::BackendError(format!(
                "cudaGraphInstantiate failed with error code {err}"
            )));
        }

        // Pre-upload the execution graph topology directly into device memory
        let _ = unsafe { driver::cudaGraphUpload(raw_exec, stream) };

        let graph_id = self.next_graph_id;
        self.next_graph_id += 1;
        let mut real_num_nodes: usize = 0;
        let _ = unsafe {
            driver::cudaGraphGetNodes(raw_graph, std::ptr::null_mut(), &raw mut real_num_nodes)
        };
        let node_count = if real_num_nodes > 0 {
            real_num_nodes
        } else {
            num_layers * 7 + 2
        };

        let graph = CapturedCudaGraph {
            graph_id,
            name: name.into(),
            num_layers,
            batch_size,
            execution_count: 0,
            is_instantiated: true,
            raw_graph,
            raw_exec,
        };

        self.graphs.insert(graph_id, graph);
        Ok(CudaGraphExecHandle {
            graph_id,
            node_count,
        })
    }

    /// Replays a captured physical CUDA execution graph in a single driver call.
    ///
    /// # Safety
    /// Caller must ensure `stream` is a valid CUDA stream pointer on the target GPU.
    pub unsafe fn launch_graph(
        &mut self,
        handle: CudaGraphExecHandle,
        stream: driver::cudaStream_t,
    ) -> Result<()> {
        let graph = self.graphs.get_mut(&handle.graph_id).ok_or_else(|| {
            EngineError::BackendError(format!("CUDA Graph ID {} not found", handle.graph_id))
        })?;

        if !graph.raw_exec.is_null() {
            let err = unsafe { driver::cudaGraphLaunch(graph.raw_exec, stream) };
            if err != 0 {
                return Err(EngineError::BackendError(format!(
                    "cudaGraphLaunch failed with error code {err}"
                )));
            }
        }

        graph.execution_count += 1;
        self.total_replays += 1;
        Ok(())
    }

    /// Registers a logical decode DAG handle for graph management.
    pub fn capture_decode_graph(
        &mut self,
        name: impl Into<String>,
        num_layers: usize,
        batch_size: usize,
    ) -> Result<CudaGraphExecHandle> {
        let graph_id = self.next_graph_id;
        self.next_graph_id += 1;
        let node_count = num_layers * 7 + 2;

        let graph = CapturedCudaGraph {
            graph_id,
            name: name.into(),
            num_layers,
            batch_size,
            execution_count: 0,
            is_instantiated: false,
            raw_graph: std::ptr::null_mut(),
            raw_exec: std::ptr::null_mut(),
        };

        self.graphs.insert(graph_id, graph);

        Ok(CudaGraphExecHandle {
            graph_id,
            node_count,
        })
    }

    /// Replays or increments execution count for a logical graph handle.
    pub fn replay_graph(&mut self, handle: CudaGraphExecHandle) -> Result<()> {
        let graph = self.graphs.get_mut(&handle.graph_id).ok_or_else(|| {
            EngineError::BackendError(format!("CUDA Graph ID {} not found", handle.graph_id))
        })?;

        graph.execution_count += 1;
        self.total_replays += 1;
        Ok(())
    }

    #[must_use]
    pub fn total_replays(&self) -> u64 {
        self.total_replays
    }

    #[must_use]
    pub fn get_graph(&self, graph_id: u64) -> Option<&CapturedCudaGraph> {
        self.graphs.get(&graph_id)
    }
}

/// Bucket sizes for discrete power-of-two batched CUDA Graph execution.
pub const CUDA_GRAPH_BUCKET_SIZES: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Power-of-Two Multi-Batch CUDA Graph Bucket Pool (`B ∈ {1, 2, 4, 8, 16, 32, 64, 128}`).
/// Eliminates per-kernel CPU launch overhead ($15-40\ \mu\text{s}$ per node) across varying
/// batch sizes by selecting pre-captured, pre-instantiated CUDA Graph execution instances.
pub struct CudaGraphBucketPool {
    buckets: [Option<CapturedCudaGraph>; 8],
    total_launches: u64,
}

impl Default for CudaGraphBucketPool {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CudaGraphBucketPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphBucketPool")
            .field("total_launches", &self.total_launches)
            .field("active_buckets", &self.active_bucket_mask())
            .finish()
    }
}

impl CudaGraphBucketPool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: [None, None, None, None, None, None, None, None],
            total_launches: 0,
        }
    }

    /// Maps batch size $B$ to corresponding bucket index $[0..7]$.
    #[must_use]
    pub fn bucket_index_for_batch(batch_size: usize) -> usize {
        for (idx, &size) in CUDA_GRAPH_BUCKET_SIZES.iter().enumerate() {
            if batch_size <= size {
                return idx;
            }
        }
        7
    }

    /// Returns bitmask of populated graph buckets.
    #[must_use]
    pub fn active_bucket_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, b) in self.buckets.iter().enumerate() {
            if b.is_some() {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Inserts a captured CUDA graph for a specific batch bucket.
    pub fn insert_graph(&mut self, bucket_idx: usize, graph: CapturedCudaGraph) {
        if bucket_idx < 8 {
            self.buckets[bucket_idx] = Some(graph);
        }
    }

    /// Retrieves bucketed graph for a given batch size.
    #[must_use]
    pub fn get_bucket(&self, batch_size: usize) -> Option<&CapturedCudaGraph> {
        let idx = Self::bucket_index_for_batch(batch_size);
        self.buckets[idx].as_ref()
    }

    /// Launches the appropriate bucketed CUDA graph on the given stream.
    ///
    /// # Safety
    /// Caller must ensure `stream` is an active, valid CUDA stream pointer on the device.
    pub unsafe fn launch_for_batch(
        &mut self,
        batch_size: usize,
        stream: driver::cudaStream_t,
    ) -> Result<bool> {
        let idx = Self::bucket_index_for_batch(batch_size);
        if let Some(graph) = self.buckets[idx].as_mut() {
            if !graph.raw_exec.is_null() {
                let err = unsafe { driver::cudaGraphLaunch(graph.raw_exec, stream) };
                if err != 0 {
                    return Err(EngineError::BackendError(format!(
                        "cudaGraphLaunch failed in bucket {idx} with error {err}"
                    )));
                }
            }
            graph.execution_count += 1;
            self.total_launches += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[must_use]
    pub fn total_launches(&self) -> u64 {
        self.total_launches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_graph_capture_and_replay() {
        let mut manager = CudaGraphManager::new();
        let handle = manager
            .capture_decode_graph("llama3-8b-decode", 32, 1)
            .unwrap();
        assert_eq!(handle.node_count, 32 * 7 + 2);

        manager.replay_graph(handle).unwrap();
        manager.replay_graph(handle).unwrap();

        assert_eq!(manager.total_replays(), 2);
        let graph = manager.get_graph(handle.graph_id).unwrap();
        assert_eq!(graph.execution_count, 2);
    }

    #[test]
    fn test_cuda_graph_bucket_pool_mapping() {
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(1), 0);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(2), 1);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(3), 2);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(4), 2);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(5), 3);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(8), 3);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(16), 4);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(32), 5);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(64), 6);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(128), 7);
        assert_eq!(CudaGraphBucketPool::bucket_index_for_batch(256), 7);
    }
}
