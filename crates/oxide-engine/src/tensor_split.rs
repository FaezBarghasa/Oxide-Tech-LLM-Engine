#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Accelerator Device Backend Classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorKind {
    CudaNvidia { device_id: u32 },
    RocmExtAmd { device_id: u32 },
    MetalAppleSilicon { device_id: u32 },
    IntelNpuVpu { device_id: u32 },
    GoogleTpuV4V5 { core_id: u32 },
    NumaCpuNode { numa_node_id: u32 },
}

/// Tensor Splitting Strategy for Multi-Device and Heterogeneous Compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorSplitMode {
    /// Column-parallel split (e.g. Q, K, V projections).
    ColumnParallel,
    /// Row-parallel split (e.g. Output projections with AllReduce).
    RowParallel,
    /// Pipeline-parallel partition across entire layer groups.
    PipelineParallel,
    /// Expert-parallel partition across MoE experts.
    ExpertParallel,
}

/// Description of a Tensor Slice assigned to a specific accelerator or NUMA node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSliceDescriptor {
    pub accelerator: AcceleratorKind,
    pub split_mode: TensorSplitMode,
    pub slice_index: usize,
    pub total_slices: usize,
    pub start_offset: usize,
    pub element_count: usize,
    pub memory_bandwidth_gbps: f32,
    pub compute_weight_fraction: f32, // Proportional load assigned based on device TFLOPS
}

/// Multi-Device Heterogeneous Tensor Splitting and NUMA Distribution Engine.
#[derive(Debug, Clone)]
pub struct TensorSplitDistributionEngine {
    pub devices: Vec<AcceleratorKind>,
    pub split_mode: TensorSplitMode,
}

impl TensorSplitDistributionEngine {
    #[must_use]
    pub fn new(devices: Vec<AcceleratorKind>, split_mode: TensorSplitMode) -> Self {
        Self {
            devices,
            split_mode,
        }
    }

    /// Automatically partitions a weight tensor across heterogeneous accelerators according to relative capability weights.
    #[must_use]
    pub fn compute_tensor_slices(
        &self,
        total_elements: usize,
        device_weights: &[f32],
    ) -> Vec<TensorSliceDescriptor> {
        let num_devs = self.devices.len();
        if num_devs == 0 {
            return Vec::new();
        }

        let weights: Vec<f32> = if device_weights.len() == num_devs {
            let total_w: f32 = device_weights.iter().sum();
            if total_w > 0.0 {
                device_weights.iter().map(|&w| w / total_w).collect()
            } else {
                vec![1.0 / num_devs as f32; num_devs]
            }
        } else {
            vec![1.0 / num_devs as f32; num_devs]
        };

        let mut slices = Vec::with_capacity(num_devs);
        let mut current_offset = 0;

        for (idx, &accel) in self.devices.iter().enumerate() {
            let frac = weights[idx];
            let is_last = idx == num_devs - 1;
            let count = if is_last {
                total_elements.saturating_sub(current_offset)
            } else {
                ((total_elements as f32 * frac).round() as usize)
                    .min(total_elements.saturating_sub(current_offset))
            };

            let bw = match accel {
                AcceleratorKind::CudaNvidia { .. } => 1008.0,
                AcceleratorKind::RocmExtAmd { .. } => 800.0,
                AcceleratorKind::MetalAppleSilicon { .. } => 400.0,
                AcceleratorKind::IntelNpuVpu { .. } => 128.0,
                AcceleratorKind::GoogleTpuV4V5 { .. } => 1200.0,
                AcceleratorKind::NumaCpuNode { .. } => 64.0,
            };

            slices.push(TensorSliceDescriptor {
                accelerator: accel,
                split_mode: self.split_mode,
                slice_index: idx,
                total_slices: num_devs,
                start_offset: current_offset,
                element_count: count,
                memory_bandwidth_gbps: bw,
                compute_weight_fraction: frac,
            });

            current_offset += count;
        }

        slices
    }

    /// Performs an All-Reduce / Sum Aggregation across partial device outputs.
    pub fn all_reduce_sum(&self, partial_outputs: &[Vec<f32>], aggregated_output: &mut [f32]) {
        if partial_outputs.is_empty() {
            return;
        }
        aggregated_output.fill(0.0);

        for partial in partial_outputs {
            for (idx, &val) in partial.iter().enumerate().take(aggregated_output.len()) {
                aggregated_output[idx] += val;
            }
        }
    }
}
