#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::cast_ptr_alignment
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::undocumented_unsafe_blocks,
    clippy::collapsible_if,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::missing_fields_in_debug,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::manual_slice_size_calculation,
    clippy::ptr_as_ptr,
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::manual_memcpy
)]

pub mod arch;
pub mod cuda_oxide;
pub mod driver;
pub mod energy;
pub mod gpu_unlock;
pub mod graph;
pub mod kernels;
pub mod megakernel;
pub mod nccl;

pub use arch::KernelExecutionPlan;
pub use driver::{CudaDeviceBuffer, CudaStream};
pub use energy::{DynamicNvml, EnergyMeasurement, EnergyMethod, EnergySampler};
pub use gpu_unlock::{
    EffectiveSpeedReadout, GpuThroughputProjection, GpuUnlockCapability, HardwareGpuUnlockEngine,
    LiveMmioRegisters, MmioAccessStatus, PciBarRegion, PhysicalPciGpu, UnlockResult,
    clock_stabilizer, nv_rm, registers as gpu_unlock_registers,
};
pub use graph::{CapturedCudaGraph, CudaGraphExecHandle, CudaGraphManager};
pub use kernels::CudaLlmKernels;
pub use megakernel::{
    CooperativeMegakernel, CooperativeMegakernelConfig, LuceDeltaNetWeights, LuceFullAttnWeights,
    LuceLayerWeights, LuceRecurrentBuffers, MegakernelLayerType, MegakernelStats,
};
pub use nccl::{CudaDeviceClusterArray, NcclCommunicator};

use oxide_core::error::{EngineError, Result};
use oxide_core::hardware::GpuDeviceProfile;
use oxide_core::traits::HardwareBackend;
use oxide_core::worker::StepCommand;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaEventHandle {
    pub event_id: u64,
}

pub struct CudaLayerWeights {
    pub d_wq: CudaDeviceBuffer,
    pub d_wk: CudaDeviceBuffer,
    pub d_wv: CudaDeviceBuffer,
    pub d_wo: CudaDeviceBuffer,
    pub d_w1: CudaDeviceBuffer,
    pub d_w3: CudaDeviceBuffer,
    pub d_w2: CudaDeviceBuffer,
    pub d_attn_norm: CudaDeviceBuffer,
    pub d_ffn_norm: CudaDeviceBuffer,
    pub d_attn_q_norm: Option<CudaDeviceBuffer>,
    pub d_attn_k_norm: Option<CudaDeviceBuffer>,
    pub d_post_attn_norm: Option<CudaDeviceBuffer>,
    pub d_post_ffn_norm: Option<CudaDeviceBuffer>,
    pub d_q_bias: Option<CudaDeviceBuffer>,
    pub d_k_bias: Option<CudaDeviceBuffer>,
    pub d_v_bias: Option<CudaDeviceBuffer>,
    pub q_m: usize,
    pub q_k: usize,
    pub kv_m: usize,
    pub kv_k: usize,
    pub inter_m: usize,
    pub inter_k: usize,
    pub o_m: usize,
    pub o_k: usize,
    pub layer_output_scale: Option<f32>,
    pub q_quant: u32,
    pub k_quant: u32,
    pub v_quant: u32,
    pub o_quant: u32,
    pub gate_quant: u32,
    pub up_quant: u32,
    pub down_quant: u32,
    pub d_inp_gate: Option<CudaDeviceBuffer>,
    pub d_proj: Option<CudaDeviceBuffer>,
    pub d_post_norm: Option<CudaDeviceBuffer>,
    pub inp_gate_quant: u32,
    pub proj_quant: u32,
}

impl fmt::Debug for CudaLayerWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaLayerWeights")
            .field("q_m", &self.q_m)
            .field("q_k", &self.q_k)
            .field("inter_m", &self.inter_m)
            .finish()
    }
}

pub type EmbeddingLookupFn = Box<dyn Fn(usize, &mut [f32]) + Send + Sync>;

#[allow(clippy::struct_excessive_bools)]
pub struct CudaBackend {
    pub device_id: usize,
    pub current_event_id: u64,
    pub host_token_buffer: Vec<u32>,
    pub profile: GpuDeviceProfile,
    pub execution_plan: KernelExecutionPlan,
    pub stream: Option<CudaStream>,
    pub layers: Vec<CudaLayerWeights>,
    pub d_output_norm: Option<CudaDeviceBuffer>,
    pub d_lm_head: Option<CudaDeviceBuffer>,
    pub d_token_embd: Option<CudaDeviceBuffer>,
    pub d_hidden: Option<CudaDeviceBuffer>,
    pub d_scratch_norm: Option<CudaDeviceBuffer>,
    pub d_scratch_q: Option<CudaDeviceBuffer>,
    pub d_scratch_k: Option<CudaDeviceBuffer>,
    pub d_scratch_v: Option<CudaDeviceBuffer>,
    pub d_scratch_attn_out: Option<CudaDeviceBuffer>,
    pub d_scratch_act: Option<CudaDeviceBuffer>,
    pub d_scratch_gate: Option<CudaDeviceBuffer>,
    pub d_scratch_up: Option<CudaDeviceBuffer>,
    pub d_scratch_dequant_w: Option<CudaDeviceBuffer>,
    pub d_scratch_act_f16: Option<CudaDeviceBuffer>,
    pub d_logits: Option<CudaDeviceBuffer>,
    pub d_sampled_token: Option<CudaDeviceBuffer>,
    pub kv_k: Vec<CudaDeviceBuffer>,
    pub kv_v: Vec<CudaDeviceBuffer>,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_pos: usize,
    pub slot_positions: Vec<usize>,
    pub embedding_lookup: Option<EmbeddingLookupFn>,
    pub mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
    pub lm_head_quant_type: u32,
    pub token_embd_quant_type: u32,
    pub rope_theta: f32,
    pub rope_scaling_factor: f32,
    pub rope_scaling_orig_ctx: usize,
    pub is_gemma: bool,
    pub final_logit_softcapping: Option<f32>,
    pub sampler: oxide_core::sampler::AcademicSamplerEngine,
    pub sampler_state: oxide_core::sampler::SamplerState,
    pub kv_capacity: usize,
    pub rms_norm_eps: f32,
    pub is_qwen: bool,
    pub sliding_window: Option<usize>,
    pub d_scratch_q8_1_a: Option<CudaDeviceBuffer>,
    pub d_scratch_q8_1_b: Option<CudaDeviceBuffer>,
    pub d_per_layer_emb: Option<CudaDeviceBuffer>,
    pub d_scratch_gate256: Option<CudaDeviceBuffer>,
    pub d_scratch_ple256: Option<CudaDeviceBuffer>,
    pub host_ple_buffer: Vec<f32>,
    pub host_emb_buffer: Vec<f32>,
    pub host_logits_buffer: Vec<f32>,
    pub per_layer_embedding_lookup: Option<EmbeddingLookupFn>,
    pub d_seq_pos: Option<CudaDeviceBuffer>,
    pub d_input_token: Option<CudaDeviceBuffer>,
    pub d_batch_positions: Option<CudaDeviceBuffer>,
    pub d_batch_slots: Option<CudaDeviceBuffer>,
    pub d_batch_tokens: Option<CudaDeviceBuffer>,
    pub max_slots: usize,
    pub graph_manager: CudaGraphManager,
    pub decode_graph_handle: Option<CudaGraphExecHandle>,
    pub batch_decode_graphs: std::collections::HashMap<usize, CudaGraphExecHandle>,
    pub cuda_graph_enabled: bool,
    pub last_sampled_token: Option<u32>,
    pub last_gpu_seq_pos: Option<usize>,
    pub d_split_k_partial_out: Option<CudaDeviceBuffer>,
    pub d_split_k_partial_lse: Option<CudaDeviceBuffer>,
    pub d_block_table: Option<CudaDeviceBuffer>,
    pub d_block_scores: Option<CudaDeviceBuffer>,
    pub max_blocks_per_seq: usize,
    pub num_splits: usize,
    pub ffn_sparsity_threshold: f32,
    pub enable_2_4_sparsity: bool,
    pub enable_h2o_kv_pruning: bool,
    pub h2o_max_blocks_budget: usize,
    pub pinned_host_token_ptr: usize,
    pub current_step_counter: u32,
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        if self.pinned_host_token_ptr != 0 {
            unsafe {
                let _ = driver::cudaFreeHost(self.pinned_host_token_ptr as *mut std::ffi::c_void);
            }
            self.pinned_host_token_ptr = 0;
        }
    }
}

impl fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_id", &self.device_id)
            .field("gpu_name", &self.profile.name)
            .field("architecture", &self.profile.architecture)
            .field("compute_capability", &self.profile.compute_capability)
            .field(
                "vram_gb",
                &(self.profile.vram_capacity_bytes / (1024 * 1024 * 1024)),
            )
            .field("execution_plan", &self.execution_plan)
            .field("event_counter", &self.current_event_id)
            .field("host_token_buffer_len", &self.host_token_buffer.len())
            .field("has_real_stream", &self.stream.is_some())
            .field("allocated_layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
    /// Creates a CUDA backend for a specified device and known GPU name (or auto-probed).
    #[must_use]
    pub fn new_with_profile(
        device_id: usize,
        max_slots: usize,
        custom_gpu_name: Option<&str>,
    ) -> Self {
        let name = custom_gpu_name.unwrap_or("NVIDIA GeForce RTX 4060 Laptop GPU");
        let profile = GpuDeviceProfile::from_known_device_name(name).unwrap_or_default();
        let execution_plan = KernelExecutionPlan::for_profile(&profile);

        // Attempt real physical device initialization with zero-latency CPU busy-spin
        #[allow(clippy::cast_possible_wrap)]
        unsafe {
            let _ = driver::cudaSetDeviceFlags(driver::CUDA_DEVICE_SCHEDULE_SPIN);
            let _ = driver::cudaSetDevice(device_id as i32);
        }
        let stream = CudaStream::new().ok();
        if stream.is_some() {
            // SAFETY: Configures physical device cache preferences (L1 preference, 0 shared memory carveout)
            unsafe { driver::init_cuda_kernel_attributes() };
        }

        let mut pinned_host_token: *mut u32 = std::ptr::null_mut();
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = driver::cudaHostAlloc(&raw mut ptr, 256, 0x03);
            if err == 0 && !ptr.is_null() {
                pinned_host_token = ptr as *mut u32;
                std::ptr::write_bytes(pinned_host_token, 0, 64);
            }
        }

        Self {
            device_id,
            current_event_id: 0,
            host_token_buffer: vec![0; max_slots],
            profile,
            execution_plan,
            stream,
            layers: Vec::new(),
            d_output_norm: None,
            d_lm_head: None,
            d_token_embd: None,
            d_hidden: None,
            d_scratch_norm: None,
            d_scratch_q: None,
            d_scratch_k: None,
            d_scratch_v: None,
            d_scratch_attn_out: None,
            d_scratch_act: None,
            d_scratch_gate: None,
            d_scratch_up: None,
            d_scratch_dequant_w: None,
            d_scratch_act_f16: None,
            d_logits: None,
            d_sampled_token: None,
            kv_k: Vec::new(),
            kv_v: Vec::new(),
            hidden_dim: 0,
            vocab_size: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            seq_pos: 0,
            slot_positions: vec![0; max_slots.max(64)],
            embedding_lookup: None,
            mmap_handle: None,
            lm_head_quant_type: 1,
            token_embd_quant_type: 1,
            rope_theta: 10_000.0,
            rope_scaling_factor: 1.0,
            rope_scaling_orig_ctx: 0,
            is_gemma: false,
            final_logit_softcapping: None,
            sampler: {
                let sampling_config = oxide_core::sampler::SamplingConfig {
                    temperature: 0.7,
                    top_p: 0.95,
                    min_p: 0.05,
                    repeat_penalty: 1.1,
                    repeat_last_n: 64,
                    dry_multiplier: 0.8,
                    ..Default::default()
                };
                oxide_core::sampler::AcademicSamplerEngine::new(sampling_config)
            },
            sampler_state: {
                let sampling_config = oxide_core::sampler::SamplingConfig {
                    temperature: 0.7,
                    top_p: 0.95,
                    min_p: 0.05,
                    repeat_penalty: 1.1,
                    repeat_last_n: 64,
                    dry_multiplier: 0.8,
                    ..Default::default()
                };
                oxide_core::sampler::SamplerState::from_config(&sampling_config)
            },
            kv_capacity: 8192,
            rms_norm_eps: 1e-5,
            is_qwen: false,
            sliding_window: None,
            d_scratch_q8_1_a: None,
            d_scratch_q8_1_b: None,
            d_per_layer_emb: None,
            d_scratch_gate256: None,
            d_scratch_ple256: None,
            host_ple_buffer: Vec::new(),
            host_emb_buffer: Vec::new(),
            host_logits_buffer: Vec::new(),
            per_layer_embedding_lookup: None,
            d_seq_pos: None,
            d_input_token: None,
            d_batch_positions: None,
            d_batch_slots: None,
            d_batch_tokens: None,
            max_slots,
            graph_manager: CudaGraphManager::new(),
            decode_graph_handle: None,
            batch_decode_graphs: std::collections::HashMap::new(),
            cuda_graph_enabled: std::env::var("OXIDE_CUDA_GRAPH")
                .map_or(true, |v| v != "0" && v.to_lowercase() != "false"),
            last_sampled_token: None,
            last_gpu_seq_pos: None,
            d_split_k_partial_out: None,
            d_split_k_partial_lse: None,
            d_block_table: None,
            d_block_scores: None,
            max_blocks_per_seq: 512,
            num_splits: 1,
            ffn_sparsity_threshold: std::env::var("OXIDE_SWIGLU_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0f32),
            enable_2_4_sparsity: std::env::var("OXIDE_SPARSE_2_4")
                .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            enable_h2o_kv_pruning: std::env::var("OXIDE_H2O_PRUNING")
                .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            h2o_max_blocks_budget: std::env::var("OXIDE_H2O_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128),
            pinned_host_token_ptr: pinned_host_token as usize,
            current_step_counter: 0,
        }
    }

    #[inline(always)]
    pub fn pinned_host_token(&self) -> *mut u32 {
        self.pinned_host_token_ptr as *mut u32
    }

    #[allow(clippy::too_many_arguments)]
    pub fn configure_hyperparameters(
        &mut self,
        rms_norm_eps: f32,
        is_qwen: bool,
        sliding_window: Option<usize>,
        kv_capacity: usize,
        rope_theta: f32,
        rope_scaling_factor: f32,
        rope_scaling_orig_ctx: usize,
        is_gemma: bool,
        final_logit_softcapping: Option<f32>,
    ) {
        if rms_norm_eps > 0.0 {
            self.rms_norm_eps = rms_norm_eps;
        }
        self.is_qwen = is_qwen;
        self.sliding_window = sliding_window;
        if kv_capacity > 0 {
            self.kv_capacity = kv_capacity;
        }
        if rope_theta > 0.0 {
            self.rope_theta = rope_theta;
        }
        self.rope_scaling_factor = rope_scaling_factor;
        self.rope_scaling_orig_ctx = rope_scaling_orig_ctx;
        self.is_gemma = is_gemma;
        self.final_logit_softcapping = final_logit_softcapping;
    }

    /// Sets or updates the block mapping table for paged RadixAttention execution.
    pub fn update_block_table(
        &mut self,
        block_table: &[i32],
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("CUDA stream not available".to_string()))?;
        let s = stream.raw();
        self.max_blocks_per_seq = max_blocks_per_seq;
        let req_bytes = block_table.len() * std::mem::size_of::<i32>();
        if self
            .d_block_table
            .as_ref()
            .map_or(0, driver::CudaDeviceBuffer::size_bytes)
            < req_bytes
        {
            self.d_block_table =
                Some(CudaDeviceBuffer::allocate(req_bytes).map_err(EngineError::BackendError)?);
            self.d_block_scores =
                Some(CudaDeviceBuffer::allocate(req_bytes).map_err(EngineError::BackendError)?);
            if let Some(buf) = &mut self.d_block_scores {
                unsafe {
                    let _ = driver::cudaMemsetAsync(buf.as_raw_ptr(), 0, req_bytes, s);
                }
            }
        }
        if let Some(buf) = &mut self.d_block_table {
            buf.copy_from_host_async(block_table, s)
                .map_err(EngineError::BackendError)?;
        }
        Ok(())
    }

    /// Reads cumulative attention score mass for all blocks from the physical GPU.
    pub fn read_block_scores(&mut self, total_elements: usize) -> Result<Vec<f32>> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("CUDA stream not available".to_string()))?;
        let s = stream.raw();
        if let Some(buf) = &self.d_block_scores {
            let mut host_scores = vec![0.0f32; total_elements];
            buf.copy_to_host_async(&mut host_scores, s)
                .map_err(EngineError::BackendError)?;
            stream.synchronize().map_err(EngineError::BackendError)?;
            Ok(host_scores)
        } else {
            Ok(vec![0.0f32; total_elements])
        }
    }

    /// Prunes physical KV cache for a given slot using H2O (Heavy-Hitter Oracle) + StreamingLLM:
    /// - Keeps the first `sink_tokens` (default 4)
    /// - Keeps the last `local_rolling_tokens` (default 256)
    /// - Selects the top heavy-hitter tokens in the middle based on accumulated attention scores
    /// - Compacts physical GPU KV cache memory in-place via D2D cudaMemcpyAsync
    /// - Returns the new compacted sequence position
    pub fn prune_slot_kv_h2o(
        &mut self,
        slot_idx: usize,
        max_budget_tokens: usize,
        sink_tokens: usize,
        local_rolling_tokens: usize,
    ) -> Result<usize> {
        let cur_pos = if slot_idx < self.slot_positions.len() {
            self.slot_positions[slot_idx]
        } else {
            self.seq_pos
        };

        if cur_pos <= max_budget_tokens || sink_tokens + local_rolling_tokens >= cur_pos {
            return Ok(cur_pos);
        }

        let allowed_mid = max_budget_tokens.saturating_sub(sink_tokens + local_rolling_tokens);
        let mid_start = sink_tokens;
        let mid_end = cur_pos - local_rolling_tokens;

        // Retrieve accumulated block scores if available
        let host_scores =
            self.read_block_scores(self.max_blocks_per_seq * self.max_slots.max(1))?;
        let slot_score_offset = slot_idx * self.max_blocks_per_seq;

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("CUDA stream not available".to_string()))?;
        let s = stream.raw();

        // Score middle tokens (each block = 16 tokens)
        let mut mid_token_scores: Vec<(usize, f32)> = (mid_start..mid_end)
            .map(|tok_idx| {
                let blk_idx = tok_idx / 16;
                let score = host_scores
                    .get(slot_score_offset + blk_idx)
                    .copied()
                    .unwrap_or(0.0f32);
                (tok_idx, score)
            })
            .collect();

        // Sort descending by score to pick heavy-hitters
        mid_token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut retained_tokens: Vec<usize> = mid_token_scores
            .iter()
            .take(allowed_mid)
            .map(|&(tok_idx, _)| tok_idx)
            .collect();

        // Sort back in chronological order to preserve causal ordering
        retained_tokens.sort_unstable();

        let cur_head_dim = self.head_dim;
        let cur_num_kv_heads = self.num_kv_heads;
        let kv_dim = cur_num_kv_heads * cur_head_dim;
        let tok_bytes = kv_dim * std::mem::size_of::<u16>();
        let slot_stride = (self.kv_capacity / self.max_slots.max(1)).max(64);
        let slot_base_bytes = slot_idx * slot_stride * tok_bytes;

        // Compact retained middle tokens immediately following sink tokens
        let mut write_pos = sink_tokens;
        for &tok_pos in &retained_tokens {
            if tok_pos != write_pos {
                for layer_idx in 0..self.layers.len() {
                    if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                        let k_buf = &mut self.kv_k[layer_idx];
                        let v_buf = &mut self.kv_v[layer_idx];
                        unsafe {
                            let k_src = (k_buf.as_raw_ptr() as *const u8)
                                .add(slot_base_bytes + tok_pos * tok_bytes);
                            let k_dst = (k_buf.as_raw_ptr() as *mut u8)
                                .add(slot_base_bytes + write_pos * tok_bytes);
                            let _ = driver::cudaMemcpyAsync(
                                k_dst as *mut std::ffi::c_void,
                                k_src as *const std::ffi::c_void,
                                tok_bytes,
                                driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                s,
                            );

                            let v_src = (v_buf.as_raw_ptr() as *const u8)
                                .add(slot_base_bytes + tok_pos * tok_bytes);
                            let v_dst = (v_buf.as_raw_ptr() as *mut u8)
                                .add(slot_base_bytes + write_pos * tok_bytes);
                            let _ = driver::cudaMemcpyAsync(
                                v_dst as *mut std::ffi::c_void,
                                v_src as *const std::ffi::c_void,
                                tok_bytes,
                                driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                s,
                            );
                        }
                    }
                }
            }
            write_pos += 1;
        }

        // Compact rolling window tokens
        let roll_bytes = local_rolling_tokens * tok_bytes;
        let roll_src_offset = slot_base_bytes + mid_end * tok_bytes;
        let roll_dst_offset = slot_base_bytes + write_pos * tok_bytes;

        if roll_src_offset != roll_dst_offset {
            for layer_idx in 0..self.layers.len() {
                if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                    let k_buf = &mut self.kv_k[layer_idx];
                    let v_buf = &mut self.kv_v[layer_idx];
                    unsafe {
                        let k_src = (k_buf.as_raw_ptr() as *const u8).add(roll_src_offset);
                        let k_dst = (k_buf.as_raw_ptr() as *mut u8).add(roll_dst_offset);
                        let _ = driver::cudaMemcpyAsync(
                            k_dst as *mut std::ffi::c_void,
                            k_src as *const std::ffi::c_void,
                            roll_bytes,
                            driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                            s,
                        );

                        let v_src = (v_buf.as_raw_ptr() as *const u8).add(roll_src_offset);
                        let v_dst = (v_buf.as_raw_ptr() as *mut u8).add(roll_dst_offset);
                        let _ = driver::cudaMemcpyAsync(
                            v_dst as *mut std::ffi::c_void,
                            v_src as *const std::ffi::c_void,
                            roll_bytes,
                            driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                            s,
                        );
                    }
                }
            }
        }
        write_pos += local_rolling_tokens;

        stream.synchronize().map_err(EngineError::BackendError)?;

        if slot_idx < self.slot_positions.len() {
            self.slot_positions[slot_idx] = write_pos;
        }
        if slot_idx == 0 {
            self.seq_pos = write_pos;
        }

        Ok(write_pos)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn configure_sampling(
        &mut self,
        temp: f32,
        top_p: f32,
        top_k: usize,
        min_p: f32,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        greedy: bool,
    ) {
        let mut sampling_config = oxide_core::sampler::SamplingConfig::default();
        if greedy || temp <= 0.0 {
            sampling_config.temperature = 0.0;
        } else {
            sampling_config.temperature = temp;
            sampling_config.top_p = top_p;
            sampling_config.top_k = top_k;
            sampling_config.min_p = min_p;
            sampling_config.repeat_penalty = repeat_penalty;
            sampling_config.repeat_last_n = repeat_last_n;
        }
        sampling_config.seed = Some(seed);
        self.sampler = oxide_core::sampler::AcademicSamplerEngine::new(sampling_config.clone());
        self.sampler_state = oxide_core::sampler::SamplerState::from_config(&sampling_config);
    }

    #[inline(always)]
    unsafe fn dispatch_mmvq(
        d_y: *mut f32,
        d_w: &CudaDeviceBuffer,
        d_x: *const f32,
        d_x_q8_1: *const std::ffi::c_void,
        m: usize,
        k: usize,
        quant_type: u32,
        batch_size: usize,
        stream: driver::cudaStream_t,
    ) {
        if d_w.size_bytes() == 0
            || m == 0
            || k == 0
            || d_w.as_raw_ptr().is_null()
            || batch_size == 0
        {
            return;
        }
        unsafe {
            if quant_type == 1 {
                driver::launch_cuda_mmvq_q4_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 2 {
                driver::launch_cuda_mmvq_q6_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 0 || quant_type == 3 {
                driver::launch_cuda_mmvq_q8_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 9 {
                driver::launch_cuda_mmvq_q5_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else {
                for b in 0..batch_size {
                    let y_ptr = d_y.add(b * m);
                    let x_ptr = d_x.add(b * k);
                    let x_q8_ptr = if d_x_q8_1.is_null() {
                        d_x_q8_1
                    } else {
                        (d_x_q8_1 as *const u8).add(b * (k / 32) * 36) as *const std::ffi::c_void
                    };
                    Self::dispatch_gemv(y_ptr, d_w, x_ptr, x_q8_ptr, m, k, quant_type, stream);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    unsafe fn dispatch_mmvq_add(
        d_y: *mut f32,
        d_w: &CudaDeviceBuffer,
        d_x: *const f32,
        d_x_q8_1: *const std::ffi::c_void,
        m: usize,
        k: usize,
        quant_type: u32,
        batch_size: usize,
        stream: driver::cudaStream_t,
        d_scratch: &mut CudaDeviceBuffer,
    ) {
        if d_w.size_bytes() == 0
            || m == 0
            || k == 0
            || d_w.as_raw_ptr().is_null()
            || batch_size == 0
        {
            return;
        }
        unsafe {
            if quant_type == 1 {
                driver::launch_cuda_mmvq_q4_k_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 2 {
                driver::launch_cuda_mmvq_q6_k_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 0 || quant_type == 3 {
                driver::launch_cuda_mmvq_q8_0_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else if quant_type == 9 {
                driver::launch_cuda_mmvq_q5_0_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    batch_size as i32,
                    stream,
                );
            } else {
                Self::dispatch_mmvq(
                    d_scratch.as_typed_ptr::<f32>(),
                    d_w,
                    d_x,
                    d_x_q8_1,
                    m,
                    k,
                    quant_type,
                    batch_size,
                    stream,
                );
                driver::launch_cuda_vector_add(
                    d_y,
                    d_scratch.as_typed_ptr::<f32>(),
                    (batch_size * m) as i32,
                    stream,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn dispatch_gemv(
        d_y: *mut f32,
        d_w: &CudaDeviceBuffer,
        d_x: *const f32,
        d_x_q8_1: *const std::ffi::c_void,
        m: usize,
        k: usize,
        quant_type: u32,
        stream: driver::cudaStream_t,
    ) {
        if d_w.size_bytes() == 0 || m == 0 || k == 0 || d_w.as_raw_ptr().is_null() {
            return;
        }
        unsafe {
            if quant_type == 1 {
                driver::launch_cuda_gemv_q4_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 2 {
                driver::launch_cuda_gemv_q6_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 4 {
                driver::launch_cuda_gemv_ptq1_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 5 {
                driver::launch_cuda_gemv_q3_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 6 {
                driver::launch_cuda_gemv_q2_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 7 {
                driver::launch_cuda_gemv_q5_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 8 {
                driver::launch_cuda_gemv_q4_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 9 {
                if !d_x_q8_1.is_null() {
                    driver::launch_cuda_mmvq_q5_0(
                        d_y,
                        d_w.as_raw_ptr(),
                        d_x_q8_1,
                        m as i32,
                        k as i32,
                        1,
                        stream,
                    );
                } else {
                    driver::launch_cuda_gemv_q5_0(
                        d_y,
                        d_w.as_raw_ptr(),
                        d_x,
                        m as i32,
                        k as i32,
                        stream,
                    );
                }
            } else if quant_type == 10 {
                driver::launch_cuda_gemv_tq2_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 11 {
                driver::launch_cuda_gemv_q4_1(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 12 {
                driver::launch_cuda_gemv_q5_1(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 13 {
                driver::launch_cuda_gemv_iq4_nl(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else {
                driver::launch_cuda_gemv_q8_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn dispatch_gemv_add(
        d_y: *mut f32,
        d_w: &CudaDeviceBuffer,
        d_x: *const f32,
        d_x_q8_1: *const std::ffi::c_void,
        m: usize,
        k: usize,
        quant_type: u32,
        stream: driver::cudaStream_t,
        d_scratch: &mut CudaDeviceBuffer,
    ) {
        if d_w.size_bytes() == 0 || m == 0 || k == 0 || d_w.as_raw_ptr().is_null() {
            return;
        }
        unsafe {
            if quant_type == 1 {
                driver::launch_cuda_gemv_q4_k_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 2 {
                driver::launch_cuda_gemv_q6_k_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 3 || quant_type > 13 {
                driver::launch_cuda_gemv_q8_0_fused_add(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x_q8_1,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 9 {
                if !d_x_q8_1.is_null() {
                    driver::launch_cuda_mmvq_q5_0_fused_add(
                        d_y,
                        d_w.as_raw_ptr(),
                        d_x_q8_1,
                        m as i32,
                        k as i32,
                        1,
                        stream,
                    );
                } else {
                    Self::dispatch_gemv(
                        d_scratch.as_typed_ptr::<f32>(),
                        d_w,
                        d_x,
                        d_x_q8_1,
                        m,
                        k,
                        quant_type,
                        stream,
                    );
                    driver::launch_cuda_vector_add(
                        d_y,
                        d_scratch.as_typed_ptr::<f32>(),
                        m as i32,
                        stream,
                    );
                }
            } else {
                Self::dispatch_gemv(
                    d_scratch.as_typed_ptr::<f32>(),
                    d_w,
                    d_x,
                    d_x_q8_1,
                    m,
                    k,
                    quant_type,
                    stream,
                );
                driver::launch_cuda_vector_add(
                    d_y,
                    d_scratch.as_typed_ptr::<f32>(),
                    m as i32,
                    stream,
                );
            }
        }
    }

    /// Retains strong reference to memory-mapped model file to prevent premature unmapping.
    pub fn set_mmap_handle(&mut self, mmap: std::sync::Arc<memmap2::Mmap>) {
        self.mmap_handle = Some(mmap);
    }

    /// Configures model architecture options such as Gemma-style offset norms and logit softcapping.
    pub fn set_model_architecture_options(
        &mut self,
        is_gemma: bool,
        final_logit_softcapping: Option<f32>,
    ) {
        self.is_gemma = is_gemma;
        self.final_logit_softcapping = final_logit_softcapping;
    }

    /// Configures a host-side token embedding row dequantization callback.
    pub fn set_embedding_lookup<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut [f32]) + Send + Sync + 'static,
    {
        self.embedding_lookup = Some(Box::new(f));
    }

    /// Configures a host-side token per-layer embedding (PLE) row dequantization callback.
    pub fn set_per_layer_embedding_lookup<F>(&mut self, ple_len: usize, f: F)
    where
        F: Fn(usize, &mut [f32]) + Send + Sync + 'static,
    {
        self.per_layer_embedding_lookup = Some(Box::new(f));
        let buf_bytes = ple_len * std::mem::size_of::<f32>();
        if self
            .d_per_layer_emb
            .as_ref()
            .is_none_or(|b| b.size_bytes() < buf_bytes)
        {
            if let Ok(buf) = CudaDeviceBuffer::allocate(buf_bytes) {
                self.d_per_layer_emb = Some(buf);
            }
        }
        self.host_ple_buffer = vec![0.0f32; ple_len];
    }

    #[must_use]
    pub fn new(device_id: usize, max_slots: usize) -> Self {
        Self::new_with_profile(device_id, max_slots, None)
    }

    #[must_use]
    pub const fn profile(&self) -> &GpuDeviceProfile {
        &self.profile
    }

    #[must_use]
    pub const fn execution_plan(&self) -> &KernelExecutionPlan {
        &self.execution_plan
    }

    #[must_use]
    pub const fn device_id(&self) -> usize {
        self.device_id
    }

    /// Allocates model scratch space and global projections in physical VRAM.
    pub fn configure_model(
        &mut self,
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output_norm: &[f32],
        lm_head_bytes: &[u8],
        token_embd_bytes: &[u8],
        lm_head_quant_type: u32,
    ) -> Result<()> {
        let stream = self.stream.as_ref().ok_or_else(|| {
            oxide_core::error::EngineError::BackendError("No active CUDA stream".to_string())
        })?;
        let s = stream.raw();

        self.hidden_dim = hidden_dim;
        self.vocab_size = vocab_size;
        self.num_heads = num_heads;
        self.num_kv_heads = num_kv_heads;
        self.head_dim = head_dim;
        self.lm_head_quant_type = lm_head_quant_type;

        tracing::info!(
            "CUDA configure_model: hidden={}, vocab={}, heads={}/{}, head_dim={}, lm_head_quant={}, lm_head_bytes={}",
            hidden_dim,
            vocab_size,
            num_heads,
            num_kv_heads,
            head_dim,
            lm_head_quant_type,
            lm_head_bytes.len()
        );

        let max_batch = 512usize;
        let max_logits_batch = 64usize;
        let mut d_hidden =
            CudaDeviceBuffer::allocate(max_batch * hidden_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_hidden
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_hidden = Some(d_hidden);

        let mut d_scratch_norm =
            CudaDeviceBuffer::allocate(max_batch * hidden_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_scratch_norm
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_scratch_norm = Some(d_scratch_norm);

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let max_dim = q_dim.max(kv_dim).max(hidden_dim);

        self.d_scratch_q = Some(
            CudaDeviceBuffer::allocate(max_batch * q_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_k = Some(
            CudaDeviceBuffer::allocate(max_batch * kv_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_v = Some(
            CudaDeviceBuffer::allocate(max_batch * kv_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_attn_out = Some(
            CudaDeviceBuffer::allocate(max_batch * q_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_act = Some(
            CudaDeviceBuffer::allocate(max_batch * max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_gate = Some(
            CudaDeviceBuffer::allocate(max_batch * max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_up = Some(
            CudaDeviceBuffer::allocate(max_batch * max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        let max_inter_m = hidden_dim * 4;
        let dequant_w_bytes = (hidden_dim * max_inter_m * 2).max(104_857_600);
        self.d_scratch_dequant_w =
            Some(CudaDeviceBuffer::allocate(dequant_w_bytes).map_err(EngineError::BackendError)?);
        let act_f16_bytes = max_batch * max_inter_m * 2;
        self.d_scratch_act_f16 =
            Some(CudaDeviceBuffer::allocate(act_f16_bytes).map_err(EngineError::BackendError)?);
        self.d_logits = Some(
            CudaDeviceBuffer::allocate(max_logits_batch * vocab_size * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_sampled_token = Some(
            CudaDeviceBuffer::allocate(max_logits_batch * 8).map_err(EngineError::BackendError)?,
        );
        self.d_seq_pos = Some(
            CudaDeviceBuffer::allocate(std::mem::size_of::<i32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_input_token = Some(
            CudaDeviceBuffer::allocate(std::mem::size_of::<i32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_batch_positions = Some(
            CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_batch_slots = Some(
            CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_batch_tokens = Some(
            CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_gate256 = Some(
            CudaDeviceBuffer::allocate(max_batch * 256 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_ple256 = Some(
            CudaDeviceBuffer::allocate(max_batch * 256 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );

        let q8_1_bytes = max_batch * ((max_dim * 4).max(65536) / 32 + 1) * 36;
        let mut d_q8_1_a =
            CudaDeviceBuffer::allocate(q8_1_bytes).map_err(EngineError::BackendError)?;
        d_q8_1_a
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_scratch_q8_1_a = Some(d_q8_1_a);
        let mut d_q8_1_b =
            CudaDeviceBuffer::allocate(q8_1_bytes).map_err(EngineError::BackendError)?;
        d_q8_1_b
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_scratch_q8_1_b = Some(d_q8_1_b);

        let mut d_out_norm =
            CudaDeviceBuffer::allocate(output_norm.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_out_norm
            .copy_from_host_async(output_norm, s)
            .map_err(EngineError::BackendError)?;
        self.d_output_norm = Some(d_out_norm);

        if !lm_head_bytes.is_empty() {
            let mut d_lm = CudaDeviceBuffer::allocate(lm_head_bytes.len())
                .map_err(EngineError::BackendError)?;
            d_lm.copy_from_host_async(lm_head_bytes, s)
                .map_err(EngineError::BackendError)?;
            self.d_lm_head = Some(d_lm);
        }

        if !token_embd_bytes.is_empty() {
            let q4_k_expected = vocab_size * (hidden_dim / 256) * 144;
            let q5_0_expected = vocab_size * (hidden_dim / 32) * 22;
            let q8_0_expected = vocab_size * (hidden_dim / 32) * 34;
            let fp32_expected = vocab_size * hidden_dim * std::mem::size_of::<f32>();
            if token_embd_bytes.len() == q4_k_expected && hidden_dim.is_multiple_of(256) {
                let mut d_emb = CudaDeviceBuffer::allocate(token_embd_bytes.len())
                    .map_err(EngineError::BackendError)?;
                d_emb
                    .copy_from_host_async(token_embd_bytes, s)
                    .map_err(EngineError::BackendError)?;
                self.d_token_embd = Some(d_emb);
                self.token_embd_quant_type = 1;
            } else if token_embd_bytes.len() == q5_0_expected && hidden_dim.is_multiple_of(32) {
                let mut d_emb = CudaDeviceBuffer::allocate(token_embd_bytes.len())
                    .map_err(EngineError::BackendError)?;
                d_emb
                    .copy_from_host_async(token_embd_bytes, s)
                    .map_err(EngineError::BackendError)?;
                self.d_token_embd = Some(d_emb);
                self.token_embd_quant_type = 2;
            } else if token_embd_bytes.len() == q8_0_expected && hidden_dim.is_multiple_of(32) {
                let mut d_emb = CudaDeviceBuffer::allocate(token_embd_bytes.len())
                    .map_err(EngineError::BackendError)?;
                d_emb
                    .copy_from_host_async(token_embd_bytes, s)
                    .map_err(EngineError::BackendError)?;
                self.d_token_embd = Some(d_emb);
                self.token_embd_quant_type = 3;
            } else if token_embd_bytes.len() == fp32_expected {
                let mut d_emb = CudaDeviceBuffer::allocate(token_embd_bytes.len())
                    .map_err(EngineError::BackendError)?;
                d_emb
                    .copy_from_host_async(token_embd_bytes, s)
                    .map_err(EngineError::BackendError)?;
                self.d_token_embd = Some(d_emb);
                self.token_embd_quant_type = 0;
            } else {
                // Non-supported quantized embedding where no GPU dequant kernel is present.
                // Leave d_token_embd as None so that embedding_lookup executes on host with zero-copy mmap.
                self.d_token_embd = None;
                self.token_embd_quant_type = 0;
            }
        }

        self.host_logits_buffer.resize(vocab_size, 0.0f32);
        stream.synchronize().map_err(EngineError::BackendError)?;
        Ok(())
    }

    /// Allocates and offloads a single transformer layer to GPU VRAM.
    #[allow(clippy::too_many_arguments)]
    pub fn add_layer_weights(
        &mut self,
        wq: &[u8],
        wk: &[u8],
        wv: &[u8],
        wo: &[u8],
        w1: &[u8],
        w3: &[u8],
        w2: &[u8],
        attn_norm: &[f32],
        ffn_norm: &[f32],
        attn_q_norm: &[f32],
        attn_k_norm: &[f32],
        post_attn_norm: &[f32],
        post_ffn_norm: &[f32],
        q_bias: &[f32],
        k_bias: &[f32],
        v_bias: &[f32],
        q_m: usize,
        q_k: usize,
        kv_m: usize,
        kv_k: usize,
        inter_m: usize,
        inter_k: usize,
        o_m: usize,
        o_k: usize,
        layer_output_scale: Option<f32>,
        quant_types: [u32; 7],
    ) -> Result<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("No active CUDA stream".to_string()))?;
        let s = stream.raw();
        tracing::info!(
            "CUDA add_layer_weights layer {}: wq={}, wk={}, wv={}, wo={}, w1={}, w3={}, w2={}, q_m={}, q_k={}, inter_m={}, quant_types={:?}",
            self.layers.len(),
            wq.len(),
            wk.len(),
            wv.len(),
            wo.len(),
            w1.len(),
            w3.len(),
            w2.len(),
            q_m,
            q_k,
            inter_m,
            quant_types,
        );

        let mut d_wq = CudaDeviceBuffer::allocate(wq.len()).map_err(EngineError::BackendError)?;
        d_wq.copy_from_host_async(wq, s)
            .map_err(EngineError::BackendError)?;

        let mut d_wk = CudaDeviceBuffer::allocate(wk.len()).map_err(EngineError::BackendError)?;
        d_wk.copy_from_host_async(wk, s)
            .map_err(EngineError::BackendError)?;

        let mut d_wv = CudaDeviceBuffer::allocate(wv.len()).map_err(EngineError::BackendError)?;
        d_wv.copy_from_host_async(wv, s)
            .map_err(EngineError::BackendError)?;

        let mut d_wo = CudaDeviceBuffer::allocate(wo.len()).map_err(EngineError::BackendError)?;
        d_wo.copy_from_host_async(wo, s)
            .map_err(EngineError::BackendError)?;

        let mut d_w1 = CudaDeviceBuffer::allocate(w1.len()).map_err(EngineError::BackendError)?;
        d_w1.copy_from_host_async(w1, s)
            .map_err(EngineError::BackendError)?;

        let mut d_w3 = CudaDeviceBuffer::allocate(w3.len()).map_err(EngineError::BackendError)?;
        d_w3.copy_from_host_async(w3, s)
            .map_err(EngineError::BackendError)?;

        let mut d_w2 = CudaDeviceBuffer::allocate(w2.len()).map_err(EngineError::BackendError)?;
        d_w2.copy_from_host_async(w2, s)
            .map_err(EngineError::BackendError)?;

        let mut d_attn_norm =
            CudaDeviceBuffer::allocate(attn_norm.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_attn_norm
            .copy_from_host_async(attn_norm, s)
            .map_err(EngineError::BackendError)?;

        let mut d_ffn_norm =
            CudaDeviceBuffer::allocate(ffn_norm.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_ffn_norm
            .copy_from_host_async(ffn_norm, s)
            .map_err(EngineError::BackendError)?;

        let d_attn_q_norm = if attn_q_norm.is_empty() {
            None
        } else {
            let mut buf =
                CudaDeviceBuffer::allocate(attn_q_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(attn_q_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_attn_k_norm = if attn_k_norm.is_empty() {
            None
        } else {
            let mut buf =
                CudaDeviceBuffer::allocate(attn_k_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(attn_k_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_post_attn_norm = if post_attn_norm.is_empty() {
            None
        } else {
            let mut buf =
                CudaDeviceBuffer::allocate(post_attn_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(post_attn_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_post_ffn_norm = if post_ffn_norm.is_empty() {
            None
        } else {
            let mut buf =
                CudaDeviceBuffer::allocate(post_ffn_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(post_ffn_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_q_bias = if q_bias.is_empty() {
            None
        } else {
            let mut buf = CudaDeviceBuffer::allocate(q_bias.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(q_bias, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_k_bias = if k_bias.is_empty() {
            None
        } else {
            let mut buf = CudaDeviceBuffer::allocate(k_bias.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(k_bias, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let d_v_bias = if v_bias.is_empty() {
            None
        } else {
            let mut buf = CudaDeviceBuffer::allocate(v_bias.len() * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(v_bias, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        };

        let kv_cap = self.kv_capacity.max(64);
        let layer_kv_dim = if kv_m > 0 {
            kv_m
        } else {
            self.num_kv_heads.max(1) * self.head_dim.max(64)
        };
        // FP16 KV Cache: 2 bytes per element (half precision)
        let kv_bytes = kv_cap * layer_kv_dim * 2;
        let mut kv_k_buf =
            CudaDeviceBuffer::allocate(kv_bytes.max(1024)).map_err(EngineError::BackendError)?;
        kv_k_buf
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        let mut kv_v_buf =
            CudaDeviceBuffer::allocate(kv_bytes.max(1024)).map_err(EngineError::BackendError)?;
        kv_v_buf
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;

        self.kv_k.push(kv_k_buf);
        self.kv_v.push(kv_v_buf);

        self.layers.push(CudaLayerWeights {
            d_wq,
            d_wk,
            d_wv,
            d_wo,
            d_w1,
            d_w3,
            d_w2,
            d_attn_norm,
            d_ffn_norm,
            d_attn_q_norm,
            d_attn_k_norm,
            d_post_attn_norm,
            d_post_ffn_norm,
            d_q_bias,
            d_k_bias,
            d_v_bias,
            q_m,
            q_k,
            kv_m,
            kv_k,
            inter_m,
            inter_k,
            o_m,
            o_k,
            layer_output_scale,
            q_quant: quant_types[0],
            k_quant: quant_types[1],
            v_quant: quant_types[2],
            o_quant: quant_types[3],
            gate_quant: quant_types[4],
            up_quant: quant_types[5],
            down_quant: quant_types[6],
            d_inp_gate: None,
            d_proj: None,
            d_post_norm: None,
            inp_gate_quant: 0,
            proj_quant: 0,
        });

        stream.synchronize().map_err(EngineError::BackendError)?;
        Ok(())
    }

    /// Configures Per-Layer Embedding (PLE) gating, projection, and normalization weights for a layer.
    pub fn add_layer_ple_weights(
        &mut self,
        layer_idx: usize,
        inp_gate: Option<(&[u8], u32)>,
        proj: Option<(&[u8], u32)>,
        post_norm: &[f32],
    ) -> Result<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("No active CUDA stream".to_string()))?;
        let s = stream.raw();

        if let Some(layer) = self.layers.get_mut(layer_idx) {
            if let Some((bytes, qtype)) = inp_gate {
                let mut d_inp =
                    CudaDeviceBuffer::allocate(bytes.len()).map_err(EngineError::BackendError)?;
                d_inp
                    .copy_from_host_async(bytes, s)
                    .map_err(EngineError::BackendError)?;
                layer.d_inp_gate = Some(d_inp);
                layer.inp_gate_quant = qtype;
            }
            if let Some((bytes, qtype)) = proj {
                let mut d_p =
                    CudaDeviceBuffer::allocate(bytes.len()).map_err(EngineError::BackendError)?;
                d_p.copy_from_host_async(bytes, s)
                    .map_err(EngineError::BackendError)?;
                layer.d_proj = Some(d_p);
                layer.proj_quant = qtype;
            }
            if !post_norm.is_empty() {
                let mut d_pn =
                    CudaDeviceBuffer::allocate(post_norm.len() * std::mem::size_of::<f32>())
                        .map_err(EngineError::BackendError)?;
                d_pn.copy_from_host_async(post_norm, s)
                    .map_err(EngineError::BackendError)?;
                layer.d_post_norm = Some(d_pn);
            }
        }
        stream.synchronize().map_err(EngineError::BackendError)?;
        Ok(())
    }

    /// Evaluates candidate draft tokens using Multi-Column MMVQ in a single physical pass over weights.
    pub fn dispatch_speculative_verification(
        &mut self,
        cur_token: u32,
        draft_tokens: &[u32],
    ) -> Result<Vec<u32>> {
        if draft_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let stream = self.stream.as_ref().ok_or_else(|| {
            EngineError::BackendError("Physical CUDA stream not available on host dGPU".to_string())
        })?;
        let s = stream.raw();
        let t_start_spec = std::time::Instant::now();

        let max_draft = 5usize;
        let num_draft = draft_tokens.len().min(max_draft);
        let mut candidate_tokens = Vec::with_capacity(1 + num_draft);
        candidate_tokens.push(cur_token);
        candidate_tokens.extend_from_slice(&draft_tokens[..num_draft]);

        let batch_size = candidate_tokens.len();
        let slot = 0usize;
        let slot_stride = (self.kv_capacity / self.max_slots.max(1)).max(64);
        let pos = self.seq_pos;
        let kv_cap = self.kv_capacity.max(64);

        if pos + batch_size >= kv_cap {
            return Err(EngineError::BackendError(format!(
                "Sequence length {} exceeds allocated KV cache capacity {}",
                pos + batch_size,
                kv_cap
            )));
        }

        let h = self.hidden_dim;
        if self.layers.is_empty() || h == 0 {
            return Err(EngineError::BackendError(
                "Physical CUDA transformer layers not loaded into GPU VRAM for execution"
                    .to_string(),
            ));
        }

        // 1. Initial embedding lookup for all B candidate tokens on GPU
        if let Some(lookup) = &self.embedding_lookup {
            let total_floats = batch_size * h;
            if self.host_emb_buffer.len() < total_floats {
                self.host_emb_buffer.resize(total_floats, 0.0f32);
            }
            for (i, &tok) in candidate_tokens.iter().enumerate() {
                let tok_idx = (tok as usize) % self.vocab_size.max(1);
                lookup(tok_idx, &mut self.host_emb_buffer[i * h..(i + 1) * h]);
            }
            if let Some(d_hid) = &mut self.d_hidden {
                unsafe {
                    let _ = driver::cudaMemcpyAsync(
                        d_hid.as_raw_ptr(),
                        self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                        total_floats * std::mem::size_of::<f32>(),
                        driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                        s,
                    );
                }
            }
        } else if let Some(d_emb) = &self.d_token_embd {
            let emb_scale = if self.is_gemma {
                (h as f32).sqrt()
            } else {
                1.0f32
            };
            for (i, &tok) in candidate_tokens.iter().enumerate() {
                let tok_idx = (tok as usize) % self.vocab_size.max(1);
                let dst_offset = i * h;
                if let Some(d_hid) = &self.d_hidden {
                    unsafe {
                        let dst_ptr = d_hid.as_typed_ptr::<f32>().add(dst_offset);
                        if self.token_embd_quant_type > 0 {
                            driver::launch_cuda_dequantize_token_embd(
                                dst_ptr,
                                d_emb.as_raw_ptr(),
                                tok_idx as i32,
                                h as i32,
                                emb_scale,
                                self.token_embd_quant_type as i32,
                                s,
                            );
                        } else {
                            let src_offset = tok_idx * h * std::mem::size_of::<f32>();
                            if src_offset + h * 4 <= d_emb.size_bytes() {
                                let src_ptr = (d_emb.as_raw_ptr() as *const u8).add(src_offset);
                                let _ = driver::cudaMemcpyAsync(
                                    dst_ptr as *mut std::ffi::c_void,
                                    src_ptr as *const std::ffi::c_void,
                                    h * std::mem::size_of::<f32>(),
                                    driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                    s,
                                );
                            }
                        }
                    }
                }
            }
        }

        let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
        let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };

        // 2. Multi-layer forward pass on physical GPU with Multi-Column MMVQ
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let tail_batch = batch_size;
            let tail_tok_offset = 0usize;
            if let (Some(d_hid), Some(d_norm)) = (&self.d_hidden, &mut self.d_scratch_norm) {
                let has_attn =
                    layer.d_wq.size_bytes() > 0 && layer.d_wo.size_bytes() > 0 && layer.q_m > 0;
                if has_attn {
                    if layer.d_attn_norm.size_bytes() > 0 {
                        unsafe {
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                d_hid.as_typed_ptr::<f32>(),
                                layer.d_attn_norm.as_typed_ptr::<f32>(),
                                batch_size as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    let cur_head_dim = if let Some(ref d_q_norm) = layer.d_attn_q_norm {
                        let nf = d_q_norm.size_bytes() / std::mem::size_of::<f32>();
                        if nf > 0 { nf } else { self.head_dim }
                    } else {
                        self.head_dim
                    };
                    let cur_num_heads = if layer.o_k > 0 && cur_head_dim > 0 {
                        layer.o_k / cur_head_dim
                    } else if cur_head_dim > 0 && layer.q_m > 0 {
                        layer.q_m / cur_head_dim
                    } else {
                        self.num_heads
                    };
                    let cur_num_kv_heads = if cur_head_dim > 0 && layer.kv_m > 0 {
                        layer.kv_m / cur_head_dim
                    } else {
                        self.num_kv_heads
                    };
                    let kv_dim = cur_num_kv_heads * cur_head_dim;
                    let cur_slot_offset_bytes =
                        slot * slot_stride * kv_dim * std::mem::size_of::<u16>();

                    let v_cache_ptr = if layer_idx < self.kv_v.len() {
                        unsafe {
                            (self.kv_v[layer_idx].as_raw_ptr() as *mut u8)
                                .add(cur_slot_offset_bytes)
                                as *mut std::ffi::c_void
                        }
                    } else {
                        std::ptr::null_mut()
                    };
                    let k_cache_ptr = if layer_idx < self.kv_k.len() {
                        unsafe {
                            (self.kv_k[layer_idx].as_raw_ptr() as *mut u8)
                                .add(cur_slot_offset_bytes)
                                as *mut std::ffi::c_void
                        }
                    } else {
                        std::ptr::null_mut()
                    };

                    let mut v_written_to_cache = false;
                    let qkv_grouped =
                        if let (Some(d_q), Some(d_k), Some(d_v), Some(d_w_f16), Some(d_x_f16)) = (
                            &mut self.d_scratch_q,
                            &mut self.d_scratch_k,
                            &mut self.d_scratch_v,
                            &mut self.d_scratch_dequant_w,
                            &mut self.d_scratch_act_f16,
                        ) {
                            let req_w_bytes = layer.q_m.max(layer.kv_m) * layer.q_k * 2;
                            let req_x_bytes = batch_size * layer.q_k * 2;
                            if false && batch_size >= 8
                                && layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                unsafe {
                                    let num_x = (batch_size * layer.q_k) as i32;
                                    driver::launch_cuda_convert_f32_to_f16(
                                        d_x_f16.as_raw_ptr(),
                                        d_norm.as_typed_ptr::<f32>(),
                                        num_x,
                                        s,
                                    );
                                    // Q projection (Q4_K)
                                    let num_q_w = (layer.q_m * layer.q_k) as i32;
                                    driver::launch_cuda_dequantize_q4_k_f16(
                                        d_w_f16.as_raw_ptr(),
                                        layer.d_wq.as_raw_ptr(),
                                        num_q_w,
                                        s,
                                    );
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.q_m as i32,
                                        layer.q_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    // K projection (Q4_K)
                                    let num_k_w = (layer.kv_m * layer.kv_k) as i32;
                                    driver::launch_cuda_dequantize_q4_k_f16(
                                        d_w_f16.as_raw_ptr(),
                                        layer.d_wk.as_raw_ptr(),
                                        num_k_w,
                                        s,
                                    );
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_k.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.kv_m as i32,
                                        layer.kv_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    // V projection (Q4_K or Q6_K)
                                    let num_v_w = (layer.kv_m * layer.kv_k) as i32;
                                    if layer.v_quant == 1 {
                                        driver::launch_cuda_dequantize_q4_k_f16(
                                            d_w_f16.as_raw_ptr(),
                                            layer.d_wv.as_raw_ptr(),
                                            num_v_w,
                                            s,
                                        );
                                    } else {
                                        driver::launch_cuda_dequantize_q6_k_f16(
                                            d_w_f16.as_raw_ptr(),
                                            layer.d_wv.as_raw_ptr(),
                                            num_v_w,
                                            s,
                                        );
                                    }
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_v.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.kv_m as i32,
                                        layer.kv_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    if !v_cache_ptr.is_null() {
                                        let v_dst = (v_cache_ptr as *mut u8).add(pos * kv_dim * 2)
                                            as *mut std::ffi::c_void;
                                        driver::launch_cuda_convert_f32_to_f16(
                                            v_dst,
                                            d_v.as_typed_ptr::<f32>(),
                                            (batch_size * kv_dim) as i32,
                                            s,
                                        );
                                        v_written_to_cache = true;
                                    }
                                }
                                true
                            } else if layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                                && layer.q_k == layer.kv_k
                                && !d_q8_1_ptr.is_null()
                            {
                                let y_v_ptr = d_v.as_typed_ptr::<f32>();
                                unsafe {
                                    driver::launch_cuda_mmvq_qkv_mixed(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_k.as_typed_ptr::<f32>(),
                                        y_v_ptr,
                                        v_cache_ptr,
                                        pos as i32,
                                        layer.d_wq.as_raw_ptr(),
                                        layer.d_wk.as_raw_ptr(),
                                        layer.d_wv.as_raw_ptr(),
                                        d_q8_1_ptr,
                                        layer.q_m as i32,
                                        layer.kv_m as i32,
                                        layer.q_k as i32,
                                        layer.v_quant as i32,
                                        batch_size as i32,
                                        s,
                                    );
                                }
                                v_written_to_cache = !v_cache_ptr.is_null();
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                    if !qkv_grouped {
                        if let Some(d_q) = &mut self.d_scratch_q {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_q.as_typed_ptr::<f32>(),
                                    &layer.d_wq,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.q_m,
                                    layer.q_k,
                                    layer.q_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }

                        if let Some(d_k) = &mut self.d_scratch_k {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_k.as_typed_ptr::<f32>(),
                                    &layer.d_wk,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.k_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }

                        if let Some(d_v) = &mut self.d_scratch_v {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_v.as_typed_ptr::<f32>(),
                                    &layer.d_wv,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.v_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }
                    }

                    // Add Q, K, V biases if present (e.g. Qwen2 / Qwen2.5)
                    if let Some(ref d_bias) = layer.d_q_bias {
                        if let Some(d_q) = &mut self.d_scratch_q {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.q_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }
                    if let Some(ref d_bias) = layer.d_k_bias {
                        if let Some(d_k) = &mut self.d_scratch_k {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_k.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }
                    if let Some(ref d_bias) = layer.d_v_bias {
                        if let Some(d_v) = &mut self.d_scratch_v {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_v.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }

                    if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                        if let (Some(d_q), Some(d_k), Some(d_v)) = (&self.d_scratch_q, &self.d_scratch_k, &self.d_scratch_v) {
                            let mut h_q = [0.0f32; 4];
                            let mut h_k = [0.0f32; 4];
                            let mut h_v = [0.0f32; 4];
                            let _ = d_q.copy_to_host_async(&mut h_q, s);
                            let _ = d_k.copy_to_host_async(&mut h_k, s);
                            let _ = d_v.copy_to_host_async(&mut h_v, s);
                            let _ = stream.synchronize();
                            eprintln!("[PREFILL L0] q_after_bias: {:?}", h_q);
                            eprintln!("[PREFILL L0] k_after_bias: {:?}", h_k);
                            eprintln!("[PREFILL L0] v_after_bias: {:?}", h_v);
                        }
                    }

                    let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                        10_000.0f32
                    } else {
                        self.rope_theta
                    };

                    let q_norm_ptr = layer
                        .d_attn_q_norm
                        .as_ref()
                        .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                        .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());
                    let k_norm_ptr = layer
                        .d_attn_k_norm
                        .as_ref()
                        .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                        .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());

                    if let (Some(d_q), Some(d_k)) = (&mut self.d_scratch_q, &mut self.d_scratch_k) {
                        unsafe {
                            driver::launch_cuda_fused_qknorm_rope_batch(
                                d_q.as_typed_ptr::<f32>(),
                                d_k.as_typed_ptr::<f32>(),
                                q_norm_ptr,
                                k_norm_ptr,
                                k_cache_ptr,
                                pos as i32,
                                batch_size as i32,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                1e-6,
                                layer_rope_theta,
                                1, // is_fp16
                                self.rope_scaling_factor,
                                self.rope_scaling_orig_ctx as i32,
                                s,
                            );
                        }
                    }

                    if !v_written_to_cache {
                        if let Some(d_v) = &self.d_scratch_v {
                            if !v_cache_ptr.is_null() {
                                unsafe {
                                    let v_dst_ptr = (v_cache_ptr as *mut u8).add(pos * kv_dim * 2)
                                        as *mut std::ffi::c_void;
                                    driver::launch_cuda_convert_f32_to_f16(
                                        v_dst_ptr,
                                        d_v.as_typed_ptr::<f32>(),
                                        (batch_size * kv_dim) as i32,
                                        s,
                                    );
                                }
                            }
                        }
                    }

                    if let (Some(d_q), Some(d_out)) =
                        (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                    {
                        if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                            let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                            let q_dim = cur_num_heads * cur_head_dim;
                            unsafe {
                                let q_ptr = d_q.as_typed_ptr::<f32>().add(tail_tok_offset * q_dim);
                                driver::launch_cuda_flash_causal_prefill(
                                    d_out.as_typed_ptr::<f32>(),
                                    q_ptr,
                                    k_cache_ptr,
                                    v_cache_ptr,
                                    (pos + tail_tok_offset) as i32,
                                    tail_batch as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    slot_stride as i32,
                                    sm_scale,
                                    1, // is_fp16
                                    s,
                                );
                                if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                                    let mut h_attn = [0.0f32; 4];
                                    let _ = d_out.copy_to_host_async(&mut h_attn, s);
                                    let _ = stream.synchronize();
                                    eprintln!("[PREFILL L0] attn_out: {:?}", h_attn);
                                }
                            }
                        }
                    }

                    if let (Some(d_out), Some(d_w_f16), Some(d_x_f16), Some(d_act)) = (
                        &self.d_scratch_attn_out,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                        &mut self.d_scratch_act,
                    ) {
                        let o_m = if layer.o_m > 0 { layer.o_m } else { h };
                        let o_k = if layer.o_k > 0 {
                            layer.o_k
                        } else {
                            cur_num_heads * cur_head_dim
                        };
                        unsafe {
                            let hid_dst_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            let req_w_bytes = o_m * o_k * 2;
                            let req_x_bytes = tail_batch * o_k * 2;
                            if layer.o_quant == 1
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (o_m * o_k) as i32;
                                let num_x_elements = (tail_batch * o_k) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_out.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_wo.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    hid_dst_ptr,
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    o_m as i32,
                                    o_k as i32,
                                    tail_batch as i32,
                                    1,
                                    s,
                                );
                            } else {
                                if !d_q8_1_ptr.is_null() {
                                    driver::launch_cuda_quantize_q8_1(
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        (tail_batch * o_k) as i32,
                                        s,
                                    );
                                }
                                Self::dispatch_mmvq_add(
                                    hid_dst_ptr,
                                    &layer.d_wo,
                                    d_out.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    o_m,
                                    o_k,
                                    layer.o_quant,
                                    tail_batch,
                                    s,
                                    d_act,
                                );
                            }
                        }
                    }

                    if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                        let mut h_hid = [0.0f32; 4];
                        let _ = d_hid.copy_to_host_async(&mut h_hid, s);
                        let _ = stream.synchronize();
                        eprintln!("[PREFILL L0] hid_after_attn: {:?}", h_hid);
                    }
                }

                let has_ffn =
                    layer.d_w1.size_bytes() > 0 && layer.d_w2.size_bytes() > 0 && layer.inter_m > 0;
                if has_ffn {
                    if layer.d_ffn_norm.size_bytes() > 0 {
                        unsafe {
                            let hid_src_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                hid_src_ptr,
                                layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                tail_batch as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    if let (Some(d_gate), Some(d_up), Some(d_w_f16), Some(d_x_f16)) = (
                        &mut self.d_scratch_gate,
                        &mut self.d_scratch_up,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                    ) {
                        unsafe {
                            let req_w_bytes = layer.inter_m * h * 2;
                            let req_x_bytes = tail_batch * h * 2;
                            if layer.gate_quant == 1
                                && layer.up_quant == 1
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (layer.inter_m * h) as i32;
                                let num_x_elements = (tail_batch * h) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_norm.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                // Gate
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w1.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    0,
                                    s,
                                );
                                // Up
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    d_up.as_typed_ptr::<f32>(),
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    0,
                                    s,
                                );
                            } else if layer.gate_quant == 1
                                && layer.up_quant == 1
                                && !d_q8_1_ptr.is_null()
                            {
                                driver::launch_cuda_mmvq_gate_up_q4_k(
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    layer.d_w1.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    s,
                                );
                            } else {
                                Self::dispatch_mmvq(
                                    d_gate.as_typed_ptr::<f32>(),
                                    &layer.d_w1,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.gate_quant,
                                    tail_batch,
                                    s,
                                );
                                Self::dispatch_mmvq(
                                    d_up.as_typed_ptr::<f32>(),
                                    &layer.d_w3,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.up_quant,
                                    tail_batch,
                                    s,
                                );
                            }
                        }

                        if !d_q8_1_ptr.is_null() {
                            unsafe {
                                driver::launch_cuda_fused_swiglu_q8_1_sparse(
                                    d_q8_1_ptr,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    (tail_batch * layer.inter_m) as i32,
                                    self.ffn_sparsity_threshold,
                                    i32::from(self.enable_2_4_sparsity),
                                    s,
                                );
                            }
                        }
                    }

                    if let (Some(d_gate), Some(d_w_f16), Some(d_x_f16), Some(d_act)) = (
                        &self.d_scratch_gate,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                        &mut self.d_scratch_act,
                    ) {
                        unsafe {
                            let hid_dst_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            let req_w_bytes = h * layer.inter_m * 2;
                            let req_x_bytes = tail_batch * layer.inter_m * 2;
                            if layer.down_quant == 2
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (h * layer.inter_m) as i32;
                                let num_x_elements = (tail_batch * layer.inter_m) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                driver::launch_cuda_dequantize_q6_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w2.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    hid_dst_ptr,
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    h as i32,
                                    layer.inter_m as i32,
                                    tail_batch as i32,
                                    1,
                                    s,
                                );
                            } else {
                                Self::dispatch_mmvq_add(
                                    hid_dst_ptr,
                                    &layer.d_w2,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    h,
                                    layer.inter_m,
                                    layer.down_quant,
                                    tail_batch,
                                    s,
                                    d_act,
                                );
                            }
                        }
                    }
                }
            }
        }

        // 3. Final Output Norm
        if let (Some(d_hid), Some(d_norm), Some(d_out_norm)) = (
            &self.d_hidden,
            &mut self.d_scratch_norm,
            &self.d_output_norm,
        ) {
            if d_out_norm.size_bytes() > 0 {
                unsafe {
                    driver::launch_cuda_fused_rmsnorm_q8_1(
                        d_norm.as_typed_ptr::<f32>(),
                        d_q8_1_ptr,
                        d_hid.as_typed_ptr::<f32>(),
                        d_out_norm.as_typed_ptr::<f32>(),
                        batch_size as i32,
                        h as i32,
                        self.rms_norm_eps,
                        weight_offset,
                        s,
                    );
                }
            }
        }

        // 4. LM Head Logit Projection
        if let (Some(d_norm), Some(d_lm), Some(d_logits)) =
            (&self.d_scratch_norm, &self.d_lm_head, &mut self.d_logits)
        {
            unsafe {
                Self::dispatch_mmvq(
                    d_logits.as_typed_ptr::<f32>(),
                    d_lm,
                    d_norm.as_typed_ptr::<f32>(),
                    d_q8_1_ptr,
                    self.vocab_size,
                    h,
                    self.lm_head_quant_type,
                    batch_size,
                    s,
                );
            }

            if let Some(cap) = self.final_logit_softcapping {
                if cap > 0.0 {
                    unsafe {
                        driver::launch_cuda_logit_softcap(
                            d_logits.as_typed_ptr::<f32>(),
                            (batch_size * self.vocab_size) as i32,
                            cap,
                            s,
                        );
                    }
                }
            }

            // 5. On-device Argmax for all B tokens
            if let Some(ref mut d_sampled) = self.d_sampled_token {
                let d_tok_ptr = d_sampled.as_raw_ptr() as *mut u32;
                let d_val_ptr = unsafe { d_tok_ptr.add(batch_size) as *mut f32 };
                unsafe {
                    driver::launch_cuda_argmax(
                        d_logits.as_typed_ptr::<f32>(),
                        self.vocab_size as i32,
                        d_tok_ptr,
                        d_val_ptr,
                        batch_size as i32,
                        s,
                    );
                }
                let mut sampled_res = vec![0u32; batch_size * 2];
                let _ = d_sampled.copy_to_host_async(&mut sampled_res, s);
                stream.synchronize().map_err(EngineError::BackendError)?;

                let mut accepted = Vec::new();
                let mut cur_seq_pos = pos;

                for b in 0..num_draft {
                    let pred_for_b = sampled_res[b];
                    let expected = draft_tokens[b];
                    if pred_for_b == expected {
                        accepted.push(expected);
                        cur_seq_pos += 1;
                    } else {
                        accepted.push(pred_for_b);
                        cur_seq_pos += 1;
                        self.seq_pos = cur_seq_pos;
                        tracing::info!(
                            "Speculative verify: batch={} accepted={}/{} in {:?}",
                            batch_size,
                            accepted.len(),
                            num_draft,
                            t_start_spec.elapsed()
                        );
                        return Ok(accepted);
                    }
                }

                let bonus_token = sampled_res[num_draft];
                accepted.push(bonus_token);
                cur_seq_pos += 1;
                self.seq_pos = cur_seq_pos;
                tracing::info!(
                    "Speculative verify: batch={} accepted={}/{} in {:?}",
                    batch_size,
                    accepted.len(),
                    num_draft,
                    t_start_spec.elapsed()
                );
                Ok(accepted)
            } else {
                Err(EngineError::BackendError(
                    "d_sampled_token buffer not allocated".to_string(),
                ))
            }
        } else {
            Err(EngineError::BackendError(
                "LM head buffers not initialized".to_string(),
            ))
        }
    }

    /// Retrieves the most recent normalized hidden state vector [hidden_dim] from the physical GPU.
    pub fn get_last_hidden_state(&mut self) -> Result<Vec<f32>> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("CUDA stream not available".to_string()))?;
        let s = stream.raw();
        if let Some(d_norm) = &self.d_scratch_norm {
            let mut host_hidden = vec![0.0f32; self.hidden_dim];
            d_norm
                .copy_to_host_async(&mut host_hidden, s)
                .map_err(EngineError::BackendError)?;
            stream.synchronize().map_err(EngineError::BackendError)?;
            Ok(host_hidden)
        } else {
            Err(EngineError::BackendError(
                "Hidden buffer not allocated".to_string(),
            ))
        }
    }

    /// High-throughput chunked prefill pass on physical GPU evaluating up to 24 prompt tokens simultaneously.
    /// Streams layer weights ONCE for all tokens in the chunk and skips the 510MB LM Head on intermediate chunks.
    pub fn dispatch_prefill_chunk(
        &mut self,
        chunk_tokens: &[u32],
        is_final: bool,
    ) -> Result<Option<u32>> {
        self.dispatch_prefill_chunk_slot(chunk_tokens, is_final, 0)
    }

    /// Slot-aware chunked prefill pass writing into dedicated KV cache partition for multi-user continuous batching.
    pub fn dispatch_prefill_chunk_slot(
        &mut self,
        chunk_tokens: &[u32],
        is_final: bool,
        slot: usize,
    ) -> Result<Option<u32>> {
        self.decode_graph_handle = None;
        self.last_sampled_token = None;
        self.last_gpu_seq_pos = None;
        self.current_step_counter = 0;
        if self.pinned_host_token_ptr != 0 {
            unsafe {
                std::ptr::write_bytes(self.pinned_host_token(), 0, 64);
            }
        }

        if chunk_tokens.is_empty() {
            return Ok(None);
        }

        let stream = self.stream.as_ref().ok_or_else(|| {
            EngineError::BackendError("Physical CUDA stream not available on host dGPU".to_string())
        })?;
        let s = stream.raw();

        let batch_size = chunk_tokens.len();
        if slot >= self.slot_positions.len() {
            self.slot_positions.resize(slot + 1, 0);
        }
        let pos = self.slot_positions[slot];
        let slot_stride = (self.kv_capacity / self.max_slots.max(1)).max(64);

        if pos + batch_size >= slot_stride {
            return Err(EngineError::BackendError(format!(
                "Sequence length {} exceeds allocated slot KV cache capacity {}",
                pos + batch_size,
                slot_stride
            )));
        }

        let h = self.hidden_dim;
        if self.layers.is_empty() || h == 0 {
            return Err(EngineError::BackendError(
                "Physical CUDA transformer layers not loaded into GPU VRAM for execution"
                    .to_string(),
            ));
        }

        let t_prefill_start = std::time::Instant::now();

        // 1. Initial embedding lookup for all B candidate tokens on GPU
        if let Some(d_emb) = &self.d_token_embd {
            let emb_scale = if self.is_gemma {
                (h as f32).sqrt()
            } else {
                1.0f32
            };
            for (i, &tok) in chunk_tokens.iter().enumerate() {
                let tok_idx = (tok as usize) % self.vocab_size.max(1);
                let dst_offset = i * h;
                if let Some(d_hid) = &self.d_hidden {
                    unsafe {
                        let dst_ptr = d_hid.as_typed_ptr::<f32>().add(dst_offset);
                        if self.token_embd_quant_type > 0 {
                            driver::launch_cuda_dequantize_token_embd(
                                dst_ptr,
                                d_emb.as_raw_ptr(),
                                tok_idx as i32,
                                h as i32,
                                emb_scale,
                                self.token_embd_quant_type as i32,
                                s,
                            );
                        } else {
                            let src_offset = tok_idx * h * std::mem::size_of::<f32>();
                            if src_offset + h * 4 <= d_emb.size_bytes() {
                                let src_ptr = (d_emb.as_raw_ptr() as *const u8).add(src_offset);
                                let _ = driver::cudaMemcpyAsync(
                                    dst_ptr as *mut std::ffi::c_void,
                                    src_ptr as *const std::ffi::c_void,
                                    h * std::mem::size_of::<f32>(),
                                    driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                    s,
                                );
                            }
                        }
                    }
                }
            }
        } else if let Some(lookup) = &self.embedding_lookup {
            let total_floats = batch_size * h;
            if self.host_emb_buffer.len() < total_floats {
                self.host_emb_buffer.resize(total_floats, 0.0f32);
            }
            for (i, &tok) in chunk_tokens.iter().enumerate() {
                let tok_idx = (tok as usize) % self.vocab_size.max(1);
                lookup(tok_idx, &mut self.host_emb_buffer[i * h..(i + 1) * h]);
            }
            if let Some(d_hid) = &mut self.d_hidden {
                unsafe {
                    let _ = driver::cudaMemcpyAsync(
                        d_hid.as_raw_ptr(),
                        self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                        total_floats * std::mem::size_of::<f32>(),
                        driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                        s,
                    );
                }
            }
        }

        let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
        let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };

        let num_layers = self.layers.len();
        let t_before_layers = std::time::Instant::now();
        // 2. Multi-layer forward pass on physical GPU with Multi-Column MMVQ
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let is_last_final_layer = is_final && (layer_idx + 1 == num_layers) && batch_size > 1;
            let tail_batch = if is_last_final_layer { 1 } else { batch_size };
            let tail_tok_offset = if is_last_final_layer {
                batch_size - 1
            } else {
                0
            };
            if let (Some(d_hid), Some(d_norm)) = (&self.d_hidden, &mut self.d_scratch_norm) {
                let has_attn =
                    layer.d_wq.size_bytes() > 0 && layer.d_wo.size_bytes() > 0 && layer.q_m > 0;
                if has_attn {
                    if layer.d_attn_norm.size_bytes() > 0 {
                        unsafe {
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                d_hid.as_typed_ptr::<f32>(),
                                layer.d_attn_norm.as_typed_ptr::<f32>(),
                                batch_size as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    let cur_head_dim = if let Some(ref d_q_norm) = layer.d_attn_q_norm {
                        let nf = d_q_norm.size_bytes() / std::mem::size_of::<f32>();
                        if nf > 0 { nf } else { self.head_dim }
                    } else {
                        self.head_dim
                    };
                    let cur_num_heads = if layer.o_k > 0 && cur_head_dim > 0 {
                        layer.o_k / cur_head_dim
                    } else if cur_head_dim > 0 && layer.q_m > 0 {
                        layer.q_m / cur_head_dim
                    } else {
                        self.num_heads
                    };
                    let cur_num_kv_heads = if cur_head_dim > 0 && layer.kv_m > 0 {
                        layer.kv_m / cur_head_dim
                    } else {
                        self.num_kv_heads
                    };
                    let kv_dim = cur_num_kv_heads * cur_head_dim;
                    let cur_slot_offset_bytes =
                        slot * slot_stride * kv_dim * std::mem::size_of::<u16>();

                    let v_cache_ptr = if layer_idx < self.kv_v.len() {
                        unsafe {
                            (self.kv_v[layer_idx].as_raw_ptr() as *mut u8)
                                .add(cur_slot_offset_bytes)
                                as *mut std::ffi::c_void
                        }
                    } else {
                        std::ptr::null_mut()
                    };
                    let k_cache_ptr = if layer_idx < self.kv_k.len() {
                        unsafe {
                            (self.kv_k[layer_idx].as_raw_ptr() as *mut u8)
                                .add(cur_slot_offset_bytes)
                                as *mut std::ffi::c_void
                        }
                    } else {
                        std::ptr::null_mut()
                    };

                    let mut v_written_to_cache = false;
                    let qkv_grouped =
                        if let (Some(d_q), Some(d_k), Some(d_v), Some(d_w_f16), Some(d_x_f16)) = (
                            &mut self.d_scratch_q,
                            &mut self.d_scratch_k,
                            &mut self.d_scratch_v,
                            &mut self.d_scratch_dequant_w,
                            &mut self.d_scratch_act_f16,
                        ) {
                            let req_w_bytes = layer.q_m.max(layer.kv_m) * layer.q_k * 2;
                            let req_x_bytes = batch_size * layer.q_k * 2;
                            if false && batch_size >= 8
                                && layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                unsafe {
                                    let num_x = (batch_size * layer.q_k) as i32;
                                    driver::launch_cuda_convert_f32_to_f16(
                                        d_x_f16.as_raw_ptr(),
                                        d_norm.as_typed_ptr::<f32>(),
                                        num_x,
                                        s,
                                    );
                                    // Q projection (Q4_K)
                                    let num_q_w = (layer.q_m * layer.q_k) as i32;
                                    driver::launch_cuda_dequantize_q4_k_f16(
                                        d_w_f16.as_raw_ptr(),
                                        layer.d_wq.as_raw_ptr(),
                                        num_q_w,
                                        s,
                                    );
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.q_m as i32,
                                        layer.q_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    // K projection (Q4_K)
                                    let num_k_w = (layer.kv_m * layer.kv_k) as i32;
                                    driver::launch_cuda_dequantize_q4_k_f16(
                                        d_w_f16.as_raw_ptr(),
                                        layer.d_wk.as_raw_ptr(),
                                        num_k_w,
                                        s,
                                    );
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_k.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.kv_m as i32,
                                        layer.kv_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    // V projection (Q4_K or Q6_K)
                                    let num_v_w = (layer.kv_m * layer.kv_k) as i32;
                                    if layer.v_quant == 1 {
                                        driver::launch_cuda_dequantize_q4_k_f16(
                                            d_w_f16.as_raw_ptr(),
                                            layer.d_wv.as_raw_ptr(),
                                            num_v_w,
                                            s,
                                        );
                                    } else {
                                        driver::launch_cuda_dequantize_q6_k_f16(
                                            d_w_f16.as_raw_ptr(),
                                            layer.d_wv.as_raw_ptr(),
                                            num_v_w,
                                            s,
                                        );
                                    }
                                    driver::launch_cuda_cublas_gemm_fp16_f32(
                                        d_v.as_typed_ptr::<f32>(),
                                        d_w_f16.as_raw_ptr(),
                                        d_x_f16.as_raw_ptr(),
                                        layer.kv_m as i32,
                                        layer.kv_k as i32,
                                        batch_size as i32,
                                        0,
                                        s,
                                    );
                                    if !v_cache_ptr.is_null() {
                                        let v_dst = (v_cache_ptr as *mut u8).add(pos * kv_dim * 2)
                                            as *mut std::ffi::c_void;
                                        driver::launch_cuda_convert_f32_to_f16(
                                            v_dst,
                                            d_v.as_typed_ptr::<f32>(),
                                            (batch_size * kv_dim) as i32,
                                            s,
                                        );
                                        v_written_to_cache = true;
                                    }
                                }
                                true
                            } else if layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                                && layer.q_k == layer.kv_k
                                && !d_q8_1_ptr.is_null()
                            {
                                let y_v_ptr = d_v.as_typed_ptr::<f32>();
                                unsafe {
                                    driver::launch_cuda_mmvq_qkv_mixed(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_k.as_typed_ptr::<f32>(),
                                        y_v_ptr,
                                        v_cache_ptr,
                                        pos as i32,
                                        layer.d_wq.as_raw_ptr(),
                                        layer.d_wk.as_raw_ptr(),
                                        layer.d_wv.as_raw_ptr(),
                                        d_q8_1_ptr,
                                        layer.q_m as i32,
                                        layer.kv_m as i32,
                                        layer.q_k as i32,
                                        layer.v_quant as i32,
                                        batch_size as i32,
                                        s,
                                    );
                                }
                                v_written_to_cache = !v_cache_ptr.is_null();
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                    if !qkv_grouped {
                        if let Some(d_q) = &mut self.d_scratch_q {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_q.as_typed_ptr::<f32>(),
                                    &layer.d_wq,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.q_m,
                                    layer.q_k,
                                    layer.q_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }

                        if let Some(d_k) = &mut self.d_scratch_k {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_k.as_typed_ptr::<f32>(),
                                    &layer.d_wk,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.k_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }

                        if let Some(d_v) = &mut self.d_scratch_v {
                            unsafe {
                                Self::dispatch_mmvq(
                                    d_v.as_typed_ptr::<f32>(),
                                    &layer.d_wv,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.v_quant,
                                    batch_size,
                                    s,
                                );
                            }
                        }
                    }

                    // Add Q, K, V biases if present (e.g. Qwen2 / Qwen2.5)
                    if let Some(ref d_bias) = layer.d_q_bias {
                        if let Some(d_q) = &mut self.d_scratch_q {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.q_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }
                    if let Some(ref d_bias) = layer.d_k_bias {
                        if let Some(d_k) = &mut self.d_scratch_k {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_k.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }
                    if let Some(ref d_bias) = layer.d_v_bias {
                        if let Some(d_v) = &mut self.d_scratch_v {
                            unsafe {
                                driver::launch_cuda_vector_bias_add_batch(
                                    d_v.as_typed_ptr::<f32>(),
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    batch_size as i32,
                                    s,
                                );
                            }
                        }
                    }

                    if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                        if let (Some(d_q), Some(d_k), Some(d_v)) = (&self.d_scratch_q, &self.d_scratch_k, &self.d_scratch_v) {
                            let mut h_q = [0.0f32; 4];
                            let mut h_k = [0.0f32; 4];
                            let mut h_v = [0.0f32; 4];
                            let _ = d_q.copy_to_host_async(&mut h_q, s);
                            let _ = d_k.copy_to_host_async(&mut h_k, s);
                            let _ = d_v.copy_to_host_async(&mut h_v, s);
                            let _ = stream.synchronize();
                            eprintln!("[PREFILL L0] q_after_bias: {:?}", h_q);
                            eprintln!("[PREFILL L0] k_after_bias: {:?}", h_k);
                            eprintln!("[PREFILL L0] v_after_bias: {:?}", h_v);
                        }
                    }

                    let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                        10_000.0f32
                    } else {
                        self.rope_theta
                    };

                    let q_norm_ptr = layer
                        .d_attn_q_norm
                        .as_ref()
                        .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                        .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());
                    let k_norm_ptr = layer
                        .d_attn_k_norm
                        .as_ref()
                        .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                        .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());

                    if let (Some(d_q), Some(d_k)) = (&mut self.d_scratch_q, &mut self.d_scratch_k) {
                        unsafe {
                            driver::launch_cuda_fused_qknorm_rope_batch(
                                d_q.as_typed_ptr::<f32>(),
                                d_k.as_typed_ptr::<f32>(),
                                q_norm_ptr,
                                k_norm_ptr,
                                k_cache_ptr,
                                pos as i32,
                                batch_size as i32,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                1e-6,
                                layer_rope_theta,
                                1, // is_fp16
                                self.rope_scaling_factor,
                                self.rope_scaling_orig_ctx as i32,
                                s,
                            );
                        }
                    }

                    if !v_written_to_cache {
                        if let Some(d_v) = &self.d_scratch_v {
                            if !v_cache_ptr.is_null() {
                                unsafe {
                                    let v_dst_ptr = (v_cache_ptr as *mut u8).add(pos * kv_dim * 2)
                                        as *mut std::ffi::c_void;
                                    driver::launch_cuda_convert_f32_to_f16(
                                        v_dst_ptr,
                                        d_v.as_typed_ptr::<f32>(),
                                        (batch_size * kv_dim) as i32,
                                        s,
                                    );
                                }
                            }
                        }
                    }

                    if let (Some(d_q), Some(d_out)) =
                        (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                    {
                        if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                            let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                            let q_dim = cur_num_heads * cur_head_dim;
                            unsafe {
                                let q_ptr = d_q.as_typed_ptr::<f32>().add(tail_tok_offset * q_dim);
                                driver::launch_cuda_flash_causal_prefill(
                                    d_out.as_typed_ptr::<f32>(),
                                    q_ptr,
                                    k_cache_ptr,
                                    v_cache_ptr,
                                    (pos + tail_tok_offset) as i32,
                                    tail_batch as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    slot_stride as i32,
                                    sm_scale,
                                    1, // is_fp16
                                    s,
                                );
                                if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                                    let mut h_attn = [0.0f32; 4];
                                    let _ = d_out.copy_to_host_async(&mut h_attn, s);
                                    let _ = stream.synchronize();
                                    eprintln!("[PREFILL L0] attn_out: {:?}", h_attn);
                                }
                            }
                        }
                    }

                    if let (Some(d_out), Some(d_w_f16), Some(d_x_f16), Some(d_act)) = (
                        &self.d_scratch_attn_out,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                        &mut self.d_scratch_act,
                    ) {
                        let o_m = if layer.o_m > 0 { layer.o_m } else { h };
                        let o_k = if layer.o_k > 0 {
                            layer.o_k
                        } else {
                            cur_num_heads * cur_head_dim
                        };
                        unsafe {
                            let hid_dst_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            let req_w_bytes = o_m * o_k * 2;
                            let req_x_bytes = tail_batch * o_k * 2;
                            if layer.o_quant == 1
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (o_m * o_k) as i32;
                                let num_x_elements = (tail_batch * o_k) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_out.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_wo.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    hid_dst_ptr,
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    o_m as i32,
                                    o_k as i32,
                                    tail_batch as i32,
                                    1,
                                    s,
                                );
                            } else {
                                if !d_q8_1_ptr.is_null() {
                                    driver::launch_cuda_quantize_q8_1(
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        (tail_batch * o_k) as i32,
                                        s,
                                    );
                                }
                                Self::dispatch_mmvq_add(
                                    hid_dst_ptr,
                                    &layer.d_wo,
                                    d_out.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    o_m,
                                    o_k,
                                    layer.o_quant,
                                    tail_batch,
                                    s,
                                    d_act,
                                );
                            }
                        }
                    }

                    if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                        let mut h_hid = [0.0f32; 4];
                        let _ = d_hid.copy_to_host_async(&mut h_hid, s);
                        let _ = stream.synchronize();
                        eprintln!("[PREFILL L0] hid_after_attn: {:?}", h_hid);
                    }
                }

                let has_ffn =
                    layer.d_w1.size_bytes() > 0 && layer.d_w2.size_bytes() > 0 && layer.inter_m > 0;
                if has_ffn {
                    if layer.d_ffn_norm.size_bytes() > 0 {
                        unsafe {
                            let hid_src_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                hid_src_ptr,
                                layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                tail_batch as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    if let (Some(d_gate), Some(d_up), Some(d_w_f16), Some(d_x_f16)) = (
                        &mut self.d_scratch_gate,
                        &mut self.d_scratch_up,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                    ) {
                        unsafe {
                            let req_w_bytes = layer.inter_m * h * 2;
                            let req_x_bytes = tail_batch * h * 2;
                            if layer.gate_quant == 1
                                && layer.up_quant == 1
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (layer.inter_m * h) as i32;
                                let num_x_elements = (tail_batch * h) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_norm.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                // Gate
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w1.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    0,
                                    s,
                                );
                                // Up
                                driver::launch_cuda_dequantize_q4_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    d_up.as_typed_ptr::<f32>(),
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    0,
                                    s,
                                );
                            } else if layer.gate_quant == 1
                                && layer.up_quant == 1
                                && !d_q8_1_ptr.is_null()
                            {
                                driver::launch_cuda_mmvq_gate_up_q4_k(
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    layer.d_w1.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.inter_m as i32,
                                    h as i32,
                                    tail_batch as i32,
                                    s,
                                );
                            } else {
                                Self::dispatch_mmvq(
                                    d_gate.as_typed_ptr::<f32>(),
                                    &layer.d_w1,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.gate_quant,
                                    tail_batch,
                                    s,
                                );
                                Self::dispatch_mmvq(
                                    d_up.as_typed_ptr::<f32>(),
                                    &layer.d_w3,
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.up_quant,
                                    tail_batch,
                                    s,
                                );
                            }
                        }

                        if !d_q8_1_ptr.is_null() {
                            unsafe {
                                driver::launch_cuda_fused_swiglu_q8_1_sparse(
                                    d_q8_1_ptr,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    (tail_batch * layer.inter_m) as i32,
                                    self.ffn_sparsity_threshold,
                                    i32::from(self.enable_2_4_sparsity),
                                    s,
                                );
                            }
                        }
                    }

                    if let (Some(d_gate), Some(d_w_f16), Some(d_x_f16), Some(d_act)) = (
                        &self.d_scratch_gate,
                        &mut self.d_scratch_dequant_w,
                        &mut self.d_scratch_act_f16,
                        &mut self.d_scratch_act,
                    ) {
                        unsafe {
                            let hid_dst_ptr = d_hid.as_typed_ptr::<f32>().add(tail_tok_offset * h);
                            let req_w_bytes = h * layer.inter_m * 2;
                            let req_x_bytes = tail_batch * layer.inter_m * 2;
                            if layer.down_quant == 2
                                && false && tail_batch >= 8
                                && d_w_f16.size_bytes() >= req_w_bytes
                                && d_x_f16.size_bytes() >= req_x_bytes
                            {
                                let num_w_elements = (h * layer.inter_m) as i32;
                                let num_x_elements = (tail_batch * layer.inter_m) as i32;
                                driver::launch_cuda_convert_f32_to_f16(
                                    d_x_f16.as_raw_ptr(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    num_x_elements,
                                    s,
                                );
                                driver::launch_cuda_dequantize_q6_k_f16(
                                    d_w_f16.as_raw_ptr(),
                                    layer.d_w2.as_raw_ptr(),
                                    num_w_elements,
                                    s,
                                );
                                driver::launch_cuda_cublas_gemm_fp16_f32(
                                    hid_dst_ptr,
                                    d_w_f16.as_raw_ptr(),
                                    d_x_f16.as_raw_ptr(),
                                    h as i32,
                                    layer.inter_m as i32,
                                    tail_batch as i32,
                                    1,
                                    s,
                                );
                            } else {
                                Self::dispatch_mmvq_add(
                                    hid_dst_ptr,
                                    &layer.d_w2,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    h,
                                    layer.inter_m,
                                    layer.down_quant,
                                    tail_batch,
                                    s,
                                    d_act,
                                );
                            }
                        }
                    }
                }
            }
        }

        let _ = stream.synchronize();
        let t_layers = t_before_layers.elapsed();
        if slot >= self.slot_positions.len() {
            self.slot_positions.resize(slot + 1, 0);
        }
        self.slot_positions[slot] += batch_size;
        if slot == 0 {
            self.seq_pos = self.slot_positions[0];
        }

        if !is_final {
            return Ok(None);
        }

        let t_before_lm = std::time::Instant::now();
        // 3. Final Output Norm ONLY for the last token in the final prefill chunk
        let last_b = batch_size - 1;
        if let (Some(d_hid), Some(d_norm), Some(d_out_norm)) = (
            &self.d_hidden,
            &mut self.d_scratch_norm,
            &self.d_output_norm,
        ) {
            if d_out_norm.size_bytes() > 0 {
                let last_hid_ptr = unsafe { d_hid.as_typed_ptr::<f32>().add(last_b * h) };
                unsafe {
                    driver::launch_cuda_fused_rmsnorm_q8_1(
                        d_norm.as_typed_ptr::<f32>(),
                        d_q8_1_ptr,
                        last_hid_ptr,
                        d_out_norm.as_typed_ptr::<f32>(),
                        1,
                        h as i32,
                        self.rms_norm_eps,
                        weight_offset,
                        s,
                    );
                }
            }
        }

        // 4. LM Head Logit Projection ONLY for the last token
        if let (Some(d_norm), Some(d_lm), Some(d_logits)) =
            (&self.d_scratch_norm, &self.d_lm_head, &mut self.d_logits)
        {
            unsafe {
                Self::dispatch_mmvq(
                    d_logits.as_typed_ptr::<f32>(),
                    d_lm,
                    d_norm.as_typed_ptr::<f32>(),
                    d_q8_1_ptr,
                    self.vocab_size,
                    h,
                    self.lm_head_quant_type,
                    1,
                    s,
                );
            }

            if let Some(cap) = self.final_logit_softcapping {
                if cap > 0.0 {
                    unsafe {
                        driver::launch_cuda_logit_softcap(
                            d_logits.as_typed_ptr::<f32>(),
                            self.vocab_size as i32,
                            cap,
                            s,
                        );
                    }
                }
            }

            // 5. Sampling for the last prompt token
            let selected_token = if self.sampler.config.temperature <= 0.0 {
                if let Some(ref mut d_sampled) = self.d_sampled_token {
                    let d_tok_ptr = d_sampled.as_raw_ptr() as *mut u32;
                    let d_val_ptr = unsafe { d_tok_ptr.add(1) as *mut f32 };
                    unsafe {
                        driver::launch_cuda_argmax(
                            d_logits.as_typed_ptr::<f32>(),
                            self.vocab_size as i32,
                            d_tok_ptr,
                            d_val_ptr,
                            1,
                            s,
                        );
                    }
                    let mut sampled_res = [0u32, 0u32];
                    let _ = d_sampled.copy_to_host_async(&mut sampled_res, s);
                    stream.synchronize().map_err(EngineError::BackendError)?;
                    sampled_res[0]
                } else {
                    return Err(EngineError::BackendError(
                        "d_sampled_token buffer not allocated".to_string(),
                    ));
                }
            } else {
                if self.host_logits_buffer.len() != self.vocab_size {
                    self.host_logits_buffer.resize(self.vocab_size, 0.0f32);
                }
                let host_logits = &mut self.host_logits_buffer;
                let _ = d_logits.copy_to_host_async(host_logits, s);
                stream.synchronize().map_err(EngineError::BackendError)?;

                let mut best_tok = 1u32;
                let mut max_val = f32::NEG_INFINITY;
                for (idx, &v) in host_logits.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        best_tok = idx as u32;
                    }
                }

                self.sampler
                    .sample_token(host_logits, &mut self.sampler_state, 198)
                    .unwrap_or(best_tok)
            };

            let t_lm = t_before_lm.elapsed();
            tracing::info!(
                "CUDA prefill chunk (tokens={}, is_final={}) finished in {:?} (layers: {:?}, lm+sync: {:?})",
                batch_size,
                is_final,
                t_prefill_start.elapsed(),
                t_layers,
                t_lm
            );

            Ok(Some(selected_token))
        } else {
            Err(EngineError::BackendError(
                "LM head buffers not initialized".to_string(),
            ))
        }
    }

    /// High-throughput continuous batch decode step on physical GPU evaluating up to 64 active sequence streams simultaneously.
    /// Model weights across all transformer layers are streamed through the GDDR6 bus EXACTLY ONCE per step.
    pub fn dispatch_batch_decode_step(
        &mut self,
        batch_tokens: &[u32],
        batch_positions: &[usize],
        slot_indices: &[usize],
    ) -> Result<Vec<u32>> {
        let b = batch_tokens.len();
        if b == 0 {
            return Ok(Vec::new());
        }
        if b > 64 {
            return Err(EngineError::BackendError(format!(
                "Batch size {b} exceeds max continuous batching slots (64)"
            )));
        }

        let stream = self.stream.as_ref().ok_or_else(|| {
            EngineError::BackendError("Physical CUDA stream not available on host dGPU".to_string())
        })?;
        let s = stream.raw();

        let h = self.hidden_dim;
        let num_layers = self.layers.len();
        if num_layers == 0 || h == 0 {
            return Err(EngineError::BackendError(
                "Physical CUDA transformer layers not loaded into GPU VRAM for execution"
                    .to_string(),
            ));
        }

        let bucket = match b {
            1 => 1,
            2 => 2,
            3..=4 => 4,
            5..=8 => 8,
            9..=16 => 16,
            17..=32 => 32,
            _ => 64,
        };

        let mut padded_tokens = batch_tokens.to_vec();
        let mut padded_positions = batch_positions.to_vec();
        let mut padded_slots = slot_indices.to_vec();
        while padded_tokens.len() < bucket {
            padded_tokens.push(0);
            padded_positions.push(0);
            padded_slots.push(0);
        }

        let pos_i32: Vec<i32> = padded_positions.iter().map(|&p| p as i32).collect();
        let slots_i32: Vec<i32> = padded_slots.iter().map(|&slot| slot as i32).collect();
        let tok_i32: Vec<i32> = padded_tokens.iter().map(|&t| t as i32).collect();

        if self.d_batch_positions.is_none() {
            self.d_batch_positions = Some(
                CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                    .map_err(EngineError::BackendError)?,
            );
        }
        if self.d_batch_slots.is_none() {
            self.d_batch_slots = Some(
                CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                    .map_err(EngineError::BackendError)?,
            );
        }
        if self.d_batch_tokens.is_none() {
            self.d_batch_tokens = Some(
                CudaDeviceBuffer::allocate(64 * std::mem::size_of::<i32>())
                    .map_err(EngineError::BackendError)?,
            );
        }

        let d_pos_buf = self.d_batch_positions.as_mut().unwrap();
        let d_slots_buf = self.d_batch_slots.as_mut().unwrap();
        let d_tok_buf = self.d_batch_tokens.as_mut().unwrap();

        d_pos_buf
            .copy_from_host_async(&pos_i32, s)
            .map_err(EngineError::BackendError)?;
        d_slots_buf
            .copy_from_host_async(&slots_i32, s)
            .map_err(EngineError::BackendError)?;
        d_tok_buf
            .copy_from_host_async(&tok_i32, s)
            .map_err(EngineError::BackendError)?;

        let max_pos = padded_positions.iter().copied().max().unwrap_or(0);
        let num_splits = if max_pos >= 256 { 4usize } else { 1usize };
        self.num_splits = num_splits;

        if num_splits > 1 {
            let cur_head_dim = if self.head_dim > 0 {
                self.head_dim
            } else {
                128
            };
            let cur_num_heads = if self.num_heads > 0 {
                self.num_heads
            } else {
                32
            };
            let req_partial_out_bytes =
                bucket * cur_num_heads * cur_head_dim * num_splits * std::mem::size_of::<f32>();
            if self
                .d_split_k_partial_out
                .as_ref()
                .map_or(0, driver::CudaDeviceBuffer::size_bytes)
                < req_partial_out_bytes
            {
                self.d_split_k_partial_out = Some(
                    CudaDeviceBuffer::allocate(req_partial_out_bytes)
                        .map_err(EngineError::BackendError)?,
                );
            }
            let req_partial_lse_bytes =
                bucket * cur_num_heads * num_splits * std::mem::size_of::<f32>() * 2;
            if self
                .d_split_k_partial_lse
                .as_ref()
                .map_or(0, driver::CudaDeviceBuffer::size_bytes)
                < req_partial_lse_bytes
            {
                self.d_split_k_partial_lse = Some(
                    CudaDeviceBuffer::allocate(req_partial_lse_bytes)
                        .map_err(EngineError::BackendError)?,
                );
            }
        }

        let can_capture = self.cuda_graph_enabled
            && self.token_embd_quant_type == 1
            && self.d_token_embd.is_some()
            && self.d_per_layer_emb.is_none();

        if can_capture {
            if let Some(&handle) = self.batch_decode_graphs.get(&bucket) {
                unsafe {
                    self.graph_manager.launch_graph(handle, s)?;
                }
                if let Some(d_sampled) = &mut self.d_sampled_token {
                    let d_sampled_ptr = d_sampled.as_raw_ptr() as *mut u32;
                    let mut host_tokens = vec![0u32; b];
                    unsafe {
                        let _ = driver::cudaMemcpyAsync(
                            host_tokens.as_mut_ptr() as *mut std::ffi::c_void,
                            d_sampled_ptr as *const std::ffi::c_void,
                            b * std::mem::size_of::<u32>(),
                            driver::CUDA_MEMCPY_DEVICE_TO_HOST,
                            s,
                        );
                    }
                    stream.synchronize().map_err(EngineError::BackendError)?;
                    for i in 0..b {
                        let slot = slot_indices[i];
                        if slot < self.host_token_buffer.len() {
                            self.host_token_buffer[slot] = host_tokens[i];
                        }
                    }
                    return Ok(host_tokens);
                }
            }
        }

        let is_capturing = if can_capture && !self.batch_decode_graphs.contains_key(&bucket) {
            unsafe { self.graph_manager.begin_capture(s).is_ok() }
        } else {
            false
        };

        let d_pos_ptr = d_pos_buf.as_typed_ptr::<i32>();
        let d_slots_ptr = d_slots_buf.as_typed_ptr::<i32>();

        let emb_scale = if self.is_gemma {
            (h as f32).sqrt()
        } else {
            1.0f32
        };

        if let (Some(d_emb), Some(d_hid)) = (&self.d_token_embd, &mut self.d_hidden) {
            if self.token_embd_quant_type > 0 {
                for i in 0..bucket {
                    let d_row = unsafe { d_hid.as_typed_ptr::<f32>().add(i * h) };
                    let in_tok_ptr = unsafe { d_tok_buf.as_typed_ptr::<i32>().add(i) };
                    unsafe {
                        driver::launch_cuda_dequantize_token_embd_ptr(
                            d_row,
                            d_emb.as_raw_ptr(),
                            in_tok_ptr,
                            h as i32,
                            emb_scale,
                            self.token_embd_quant_type as i32,
                            s,
                        );
                    }
                }
            } else if let Some(ref lookup) = self.embedding_lookup {
                let total_floats = bucket * h;
                if self.host_emb_buffer.len() < total_floats {
                    self.host_emb_buffer.resize(total_floats, 0.0);
                }
                for (i, &tok) in padded_tokens.iter().enumerate() {
                    let tok_idx = (tok as usize) % self.vocab_size.max(1);
                    lookup(tok_idx, &mut self.host_emb_buffer[i * h..(i + 1) * h]);
                }
                if (emb_scale - 1.0f32).abs() > 1e-6 {
                    for x in &mut self.host_emb_buffer[..total_floats] {
                        *x *= emb_scale;
                    }
                }
                unsafe {
                    let _ = driver::cudaMemcpyAsync(
                        d_hid.as_raw_ptr(),
                        self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                        total_floats * std::mem::size_of::<f32>(),
                        driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                        s,
                    );
                }
            }
        } else if let Some(ref lookup) = self.embedding_lookup {
            let total_floats = bucket * h;
            if self.host_emb_buffer.len() < total_floats {
                self.host_emb_buffer.resize(total_floats, 0.0);
            }
            for (i, &tok) in padded_tokens.iter().enumerate() {
                let tok_idx = (tok as usize) % self.vocab_size.max(1);
                lookup(tok_idx, &mut self.host_emb_buffer[i * h..(i + 1) * h]);
            }
            if let Some(d_hid) = &mut self.d_hidden {
                unsafe {
                    let _ = driver::cudaMemcpyAsync(
                        d_hid.as_raw_ptr(),
                        self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                        total_floats * std::mem::size_of::<f32>(),
                        driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                        s,
                    );
                }
            }
        }

        let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };
        let d_q8_1_swiglu_ptr = if let Some(ref buf) = self.d_scratch_q8_1_b {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };

        let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
        let slot_stride = (self.kv_capacity / self.max_slots.max(1)).max(64) as i32;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let (Some(d_hid), Some(d_norm)) = (&self.d_hidden, &mut self.d_scratch_norm) {
                let has_attn =
                    layer.d_wq.size_bytes() > 0 && layer.d_wo.size_bytes() > 0 && layer.q_m > 0;
                if has_attn {
                    if layer.d_attn_norm.size_bytes() > 0 {
                        unsafe {
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                d_hid.as_typed_ptr::<f32>(),
                                layer.d_attn_norm.as_typed_ptr::<f32>(),
                                bucket as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    let cur_head_dim = if let Some(ref d_q_norm) = layer.d_attn_q_norm {
                        let nf = d_q_norm.size_bytes() / std::mem::size_of::<f32>();
                        if nf > 0 { nf } else { self.head_dim }
                    } else {
                        self.head_dim
                    };
                    let cur_num_heads = if layer.o_k > 0 && cur_head_dim > 0 {
                        layer.o_k / cur_head_dim
                    } else if cur_head_dim > 0 && layer.q_m > 0 {
                        layer.q_m / cur_head_dim
                    } else {
                        self.num_heads
                    };
                    let cur_num_kv_heads = if cur_head_dim > 0 && layer.kv_m > 0 {
                        layer.kv_m / cur_head_dim
                    } else {
                        self.num_kv_heads
                    };
                    let kv_dim = cur_num_kv_heads * cur_head_dim;

                    let v_cache_ptr = if layer_idx < self.kv_v.len() {
                        self.kv_v[layer_idx].as_raw_ptr()
                    } else {
                        std::ptr::null_mut()
                    };
                    let k_cache_ptr = if layer_idx < self.kv_k.len() {
                        self.kv_k[layer_idx].as_raw_ptr()
                    } else {
                        std::ptr::null_mut()
                    };

                    if let (Some(d_q), Some(d_k), Some(d_v)) = (
                        &mut self.d_scratch_q,
                        &mut self.d_scratch_k,
                        &mut self.d_scratch_v,
                    ) {
                        let y_v_ptr = d_v.as_typed_ptr::<f32>();
                        unsafe {
                            driver::launch_cuda_mmvq_qkv_mixed(
                                d_q.as_typed_ptr::<f32>(),
                                d_k.as_typed_ptr::<f32>(),
                                y_v_ptr,
                                std::ptr::null_mut(),
                                0,
                                layer.d_wq.as_raw_ptr(),
                                layer.d_wk.as_raw_ptr(),
                                layer.d_wv.as_raw_ptr(),
                                d_q8_1_ptr,
                                layer.q_m as i32,
                                layer.kv_m as i32,
                                layer.q_k as i32,
                                layer.v_quant as i32,
                                bucket as i32,
                                s,
                            );
                        }

                        if let Some(ref d_qb) = layer.d_q_bias {
                            for i in 0..bucket {
                                let row_ptr =
                                    unsafe { d_q.as_typed_ptr::<f32>().add(i * layer.q_m) };
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        row_ptr,
                                        d_qb.as_typed_ptr::<f32>(),
                                        layer.q_m as i32,
                                        s,
                                    );
                                }
                            }
                        }
                        if let Some(ref d_kb) = layer.d_k_bias {
                            for i in 0..bucket {
                                let row_ptr =
                                    unsafe { d_k.as_typed_ptr::<f32>().add(i * layer.kv_m) };
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        row_ptr,
                                        d_kb.as_typed_ptr::<f32>(),
                                        layer.kv_m as i32,
                                        s,
                                    );
                                }
                            }
                        }
                        if let Some(ref d_vb) = layer.d_v_bias {
                            for i in 0..bucket {
                                let row_ptr =
                                    unsafe { d_v.as_typed_ptr::<f32>().add(i * layer.kv_m) };
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        row_ptr,
                                        d_vb.as_typed_ptr::<f32>(),
                                        layer.kv_m as i32,
                                        s,
                                    );
                                }
                            }
                        }

                        if let (Some(d_q_norm), Some(d_k_norm)) =
                            (&layer.d_attn_q_norm, &layer.d_attn_k_norm)
                        {
                            for i in 0..bucket {
                                let q_row = unsafe { d_q.as_typed_ptr::<f32>().add(i * layer.q_m) };
                                let k_row =
                                    unsafe { d_k.as_typed_ptr::<f32>().add(i * layer.kv_m) };
                                unsafe {
                                    driver::launch_cuda_rmsnorm(
                                        q_row,
                                        q_row,
                                        d_q_norm.as_typed_ptr::<f32>(),
                                        cur_num_heads as i32,
                                        cur_head_dim as i32,
                                        self.rms_norm_eps,
                                        0.0f32,
                                        s,
                                    );
                                    driver::launch_cuda_rmsnorm(
                                        k_row,
                                        k_row,
                                        d_k_norm.as_typed_ptr::<f32>(),
                                        cur_num_kv_heads as i32,
                                        cur_head_dim as i32,
                                        self.rms_norm_eps,
                                        0.0f32,
                                        s,
                                    );
                                }
                            }
                        }

                        let block_tbl_ptr = self
                            .d_block_table
                            .as_ref()
                            .map_or(std::ptr::null(), |b| b.as_typed_ptr::<i32>());
                        let max_blocks = self.max_blocks_per_seq as i32;

                        unsafe {
                            driver::launch_cuda_rope_paged_continuous_batch(
                                d_q.as_typed_ptr::<f32>(),
                                d_k.as_typed_ptr::<f32>(),
                                k_cache_ptr,
                                d_pos_ptr,
                                d_slots_ptr,
                                block_tbl_ptr,
                                max_blocks,
                                bucket as i32,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                slot_stride,
                                self.rope_theta,
                                1,
                                self.rope_scaling_factor,
                                self.rope_scaling_orig_ctx as i32,
                                s,
                            );
                        }

                        if !v_cache_ptr.is_null() {
                            unsafe {
                                driver::launch_cuda_write_v_cache_paged_continuous_batch(
                                    v_cache_ptr,
                                    d_v.as_typed_ptr::<f32>(),
                                    d_pos_ptr,
                                    d_slots_ptr,
                                    block_tbl_ptr,
                                    max_blocks,
                                    bucket as i32,
                                    kv_dim as i32,
                                    slot_stride,
                                    1,
                                    s,
                                );
                            }
                        }

                        if let Some(d_out) = &mut self.d_scratch_attn_out {
                            let sm_scale = 1.0f32 / (cur_head_dim as f32).sqrt();
                            let part_out_ptr = if num_splits > 1 {
                                self.d_split_k_partial_out
                                    .as_mut()
                                    .map_or(std::ptr::null_mut(), |b| b.as_typed_ptr::<f32>())
                            } else {
                                std::ptr::null_mut()
                            };
                            let part_lse_ptr = if num_splits > 1 {
                                self.d_split_k_partial_lse
                                    .as_mut()
                                    .map_or(std::ptr::null_mut(), |b| b.as_raw_ptr())
                            } else {
                                std::ptr::null_mut()
                            };
                            let block_scores_ptr = self
                                .d_block_scores
                                .as_mut()
                                .map_or(std::ptr::null_mut(), |b| b.as_typed_ptr::<f32>());
                            unsafe {
                                driver::launch_cuda_split_k_flash_decode_h2o(
                                    d_out.as_typed_ptr::<f32>(),
                                    part_out_ptr,
                                    part_lse_ptr,
                                    d_q.as_typed_ptr::<f32>(),
                                    k_cache_ptr,
                                    v_cache_ptr,
                                    d_pos_ptr,
                                    d_slots_ptr,
                                    block_tbl_ptr,
                                    max_blocks,
                                    bucket as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    slot_stride,
                                    sm_scale,
                                    1,
                                    num_splits as i32,
                                    block_scores_ptr,
                                    s,
                                );
                            }
                        }
                    }

                    if let Some(d_out) = &self.d_scratch_attn_out {
                        unsafe {
                            driver::launch_cuda_quantize_q8_1(
                                d_out.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                (bucket * layer.o_k) as i32,
                                s,
                            );
                            if layer.o_quant == 2 {
                                driver::launch_cuda_mmvq_q6_k_fused_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_wo.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.o_m as i32,
                                    layer.o_k as i32,
                                    bucket as i32,
                                    s,
                                );
                            } else {
                                driver::launch_cuda_mmvq_q4_k_fused_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_wo.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.o_m as i32,
                                    layer.o_k as i32,
                                    bucket as i32,
                                    s,
                                );
                            }
                        }
                    }
                }

                let has_ffn =
                    layer.d_w1.size_bytes() > 0 && layer.d_w2.size_bytes() > 0 && layer.inter_m > 0;
                if has_ffn {
                    if layer.d_ffn_norm.size_bytes() > 0 {
                        unsafe {
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                d_hid.as_typed_ptr::<f32>(),
                                layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                bucket as i32,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    if let (Some(d_gate), Some(d_up)) =
                        (&mut self.d_scratch_gate, &mut self.d_scratch_up)
                    {
                        unsafe {
                            driver::launch_cuda_mmvq_gate_up_q4_k(
                                d_gate.as_typed_ptr::<f32>(),
                                d_up.as_typed_ptr::<f32>(),
                                layer.d_w1.as_raw_ptr(),
                                layer.d_w3.as_raw_ptr(),
                                d_q8_1_ptr,
                                layer.inter_m as i32,
                                h as i32,
                                bucket as i32,
                                s,
                            );
                        }

                        let act_n = bucket * layer.inter_m;
                        if let Some(d_act) = &mut self.d_scratch_act {
                            unsafe {
                                driver::launch_cuda_fused_swiglu_q8_1_sparse(
                                    d_q8_1_swiglu_ptr,
                                    d_act.as_typed_ptr::<f32>(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    act_n as i32,
                                    self.ffn_sparsity_threshold,
                                    i32::from(self.enable_2_4_sparsity),
                                    s,
                                );
                                if layer.down_quant == 2 {
                                    driver::launch_cuda_mmvq_q6_k_fused_add(
                                        d_hid.as_typed_ptr::<f32>(),
                                        layer.d_w2.as_raw_ptr(),
                                        d_q8_1_swiglu_ptr,
                                        h as i32,
                                        layer.inter_m as i32,
                                        bucket as i32,
                                        s,
                                    );
                                } else {
                                    driver::launch_cuda_mmvq_q4_k_fused_add(
                                        d_hid.as_typed_ptr::<f32>(),
                                        layer.d_w2.as_raw_ptr(),
                                        d_q8_1_swiglu_ptr,
                                        h as i32,
                                        layer.inter_m as i32,
                                        bucket as i32,
                                        s,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if let (Some(d_out_norm), Some(d_hid), Some(d_norm)) = (
            &self.d_output_norm,
            &self.d_hidden,
            &mut self.d_scratch_norm,
        ) {
            unsafe {
                driver::launch_cuda_fused_rmsnorm_q8_1(
                    d_norm.as_typed_ptr::<f32>(),
                    d_q8_1_ptr,
                    d_hid.as_typed_ptr::<f32>(),
                    d_out_norm.as_typed_ptr::<f32>(),
                    bucket as i32,
                    h as i32,
                    self.rms_norm_eps,
                    weight_offset,
                    s,
                );
            }
        }

        if let (Some(d_lm), Some(d_logits)) = (&self.d_lm_head, &mut self.d_logits) {
            unsafe {
                if self.lm_head_quant_type == 1 {
                    driver::launch_cuda_mmvq_q4_k(
                        d_logits.as_typed_ptr::<f32>(),
                        d_lm.as_raw_ptr(),
                        d_q8_1_ptr,
                        self.vocab_size as i32,
                        h as i32,
                        bucket as i32,
                        s,
                    );
                } else if self.lm_head_quant_type == 0 || self.lm_head_quant_type == 3 {
                    driver::launch_cuda_mmvq_q8_0(
                        d_logits.as_typed_ptr::<f32>(),
                        d_lm.as_raw_ptr(),
                        d_q8_1_ptr,
                        self.vocab_size as i32,
                        h as i32,
                        bucket as i32,
                        s,
                    );
                } else if self.lm_head_quant_type == 2 {
                    driver::launch_cuda_mmvq_q6_k(
                        d_logits.as_typed_ptr::<f32>(),
                        d_lm.as_raw_ptr(),
                        d_q8_1_ptr,
                        self.vocab_size as i32,
                        h as i32,
                        bucket as i32,
                        s,
                    );
                }
            }

            if let Some(d_sampled) = &mut self.d_sampled_token {
                let d_sampled_ptr = d_sampled.as_raw_ptr() as *mut u32;
                let d_val_ptr = unsafe { d_sampled_ptr.add(bucket) as *mut f32 };
                unsafe {
                    driver::launch_cuda_argmax(
                        d_logits.as_typed_ptr::<f32>(),
                        self.vocab_size as i32,
                        d_sampled_ptr,
                        d_val_ptr,
                        bucket as i32,
                        s,
                    );
                }

                if is_capturing {
                    let end_res = unsafe {
                        self.graph_manager.end_capture(
                            s,
                            "oxide_batch_decode_graph",
                            self.layers.len(),
                            bucket,
                        )
                    };
                    if let Ok(handle) = end_res {
                        self.batch_decode_graphs.insert(bucket, handle);
                        unsafe {
                            self.graph_manager.launch_graph(handle, s)?;
                        }
                    }
                }

                let mut host_tokens = vec![0u32; b];
                unsafe {
                    let _ = driver::cudaMemcpyAsync(
                        host_tokens.as_mut_ptr() as *mut std::ffi::c_void,
                        d_sampled_ptr as *const std::ffi::c_void,
                        b * std::mem::size_of::<u32>(),
                        driver::CUDA_MEMCPY_DEVICE_TO_HOST,
                        s,
                    );
                }
                stream.synchronize().map_err(EngineError::BackendError)?;

                for i in 0..b {
                    let slot = slot_indices[i];
                    if slot < self.host_token_buffer.len() {
                        self.host_token_buffer[slot] = host_tokens[i];
                    }
                }
                return Ok(host_tokens);
            }
        }

        Err(EngineError::BackendError(
            "LM head buffers not initialized for batch decode".to_string(),
        ))
    }

    /// Pre-warms GPU clock state (P8 -> P0 boost) and JIT-initializes CUDA kernel modules
    /// during model initialization so the first user prefill pass runs at full hardware speed.
    pub fn warmup_gpu(&mut self) {
        if !self.layers.is_empty() && self.hidden_dim > 0 {
            let prev_temp = self.sampler.config.temperature;
            self.sampler.config.temperature = 0.0;
            let _ = self.dispatch_speculative_verification(0, &[0; 5]);
            self.seq_pos = 0;
            let _ = self.dispatch_prefill_chunk(&[0; 19], true);
            self.seq_pos = 0;
            self.slot_positions.fill(0);
            self.sampler.config.temperature = prev_temp;
        }
    }
}

impl HardwareBackend for CudaBackend {
    type Event = CudaEventHandle;

    #[allow(clippy::unnecessary_unwrap)]
    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event> {
        self.current_event_id += 1;
        let event = CudaEventHandle {
            event_id: self.current_event_id,
        };

        let stream = self.stream.as_ref().ok_or_else(|| {
            oxide_core::error::EngineError::BackendError(
                "Physical CUDA stream not available on host dGPU".to_string(),
            )
        })?;
        let s = stream.raw();
        let slot = cmd.slot_idx as usize;

        // --- PHYSICAL CUDA GRAPH DECODE REPLAY ENGINE ---
        if !cmd.is_prefill_chunk
            && self.cuda_graph_enabled
            && self.d_token_embd.is_some()
            && self.d_per_layer_emb.is_none()
            && !self.layers.is_empty()
            && self.hidden_dim > 0
        {
            if self.d_seq_pos.is_none() {
                self.d_seq_pos = Some(
                    CudaDeviceBuffer::allocate(std::mem::size_of::<i32>())
                        .map_err(EngineError::BackendError)?,
                );
            }
            if self.d_input_token.is_none() {
                self.d_input_token = Some(
                    CudaDeviceBuffer::allocate(std::mem::size_of::<i32>())
                        .map_err(EngineError::BackendError)?,
                );
            }

            let d_sp = self.d_seq_pos.as_mut().unwrap();
            let d_in_tok = self.d_input_token.as_mut().unwrap();

            if let Some(handle) = self.decode_graph_handle {
                let gpu_pos_match = self.last_gpu_seq_pos == Some(self.seq_pos);
                let token_match = self.last_sampled_token == Some(cmd.input_token);
                if !gpu_pos_match {
                    let _ = d_sp.copy_from_host_async(&[self.seq_pos as i32], s);
                }
                if !token_match {
                    let _ = d_in_tok.copy_from_host_async(&[cmd.input_token as i32], s);
                }

                let t_graph_start = std::time::Instant::now();
                unsafe {
                    self.graph_manager.launch_graph(handle, s)?;
                }

                if self.pinned_host_token_ptr != 0 {
                    let expected_step = self.current_step_counter + 1;
                    let h_tok_ptr = self.pinned_host_token();
                    let t_spin_start = std::time::Instant::now();
                    while unsafe { std::ptr::read_volatile(h_tok_ptr.add(1)) } < expected_step {
                        std::hint::spin_loop();
                        if t_spin_start.elapsed().as_millis() > 2000 {
                            stream.synchronize().map_err(EngineError::BackendError)?;
                            break;
                        }
                    }
                    self.current_step_counter = expected_step;
                    let graph_dt = t_graph_start.elapsed();
                    let best_tok = unsafe { std::ptr::read_volatile(h_tok_ptr) };
                    if self.seq_pos.is_multiple_of(16) {
                        tracing::info!(
                            "CUDA decode graph step dt: {:.3}ms ({:.2} tok/s)",
                            graph_dt.as_secs_f64() * 1000.0,
                            1.0 / graph_dt.as_secs_f64()
                        );
                    }
                    self.last_sampled_token = Some(best_tok);
                    self.seq_pos += 1;
                    self.last_gpu_seq_pos = Some(self.seq_pos);
                    if slot < self.slot_positions.len() {
                        self.slot_positions[slot] = self.seq_pos;
                    }
                    if slot < self.host_token_buffer.len() {
                        self.host_token_buffer[slot] = best_tok;
                    }
                    return Ok(event);
                } else if let Some(ref d_sampled) = self.d_sampled_token {
                    let mut sampled_res = [0u32, 0u32];
                    let _ = d_sampled.copy_to_host_async(&mut sampled_res, s);
                    stream.synchronize().map_err(EngineError::BackendError)?;
                    let graph_dt = t_graph_start.elapsed();
                    let best_tok = sampled_res[0];
                    if self.seq_pos.is_multiple_of(16) {
                        tracing::info!(
                            "CUDA decode graph step dt: {:.3}ms ({:.2} tok/s)",
                            graph_dt.as_secs_f64() * 1000.0,
                            1.0 / graph_dt.as_secs_f64()
                        );
                    }
                    self.last_sampled_token = Some(best_tok);
                    self.seq_pos += 1;
                    self.last_gpu_seq_pos = Some(self.seq_pos);
                    if slot < self.slot_positions.len() {
                        self.slot_positions[slot] = self.seq_pos;
                    }
                    if slot < self.host_token_buffer.len() {
                        self.host_token_buffer[slot] = best_tok;
                    }
                    return Ok(event);
                }
            } else {
                let pos = self.seq_pos;
                let _ = d_sp.copy_from_host_async(&[pos as i32], s);
                let _ = d_in_tok.copy_from_host_async(&[cmd.input_token as i32], s);
                let _ = stream.synchronize();

                let capture_res = unsafe { self.graph_manager.begin_capture(s) };
                if capture_res.is_ok() {
                    let h = self.hidden_dim;
                    let emb_scale = if self.is_gemma {
                        (h as f32).sqrt()
                    } else {
                        1.0f32
                    };
                    let d_sp_ptr = self.d_seq_pos.as_ref().unwrap().as_typed_ptr::<i32>();
                    let d_in_tok_ptr = self.d_input_token.as_ref().unwrap().as_typed_ptr::<i32>();
                    let d_emb_ptr = self.d_token_embd.as_ref().unwrap().as_raw_ptr();
                    let d_hid_ptr = self.d_hidden.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_norm_ptr = self.d_scratch_norm.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_q_ptr = self.d_scratch_q.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_k_ptr = self.d_scratch_k.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_v_ptr = self.d_scratch_v.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_out_ptr = self
                        .d_scratch_attn_out
                        .as_ref()
                        .unwrap()
                        .as_typed_ptr::<f32>();
                    let d_gate_ptr = self.d_scratch_gate.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_up_ptr = self.d_scratch_up.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_logits_ptr = self.d_logits.as_ref().unwrap().as_typed_ptr::<f32>();
                    let d_sampled_ptr =
                        self.d_sampled_token.as_ref().unwrap().as_raw_ptr() as *mut u32;
                    let d_val_ptr = unsafe { d_sampled_ptr.add(1) as *mut f32 };
                    let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
                        buf.as_raw_ptr()
                    } else {
                        std::ptr::null_mut()
                    };
                    let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };

                    // 1. Dequantize token embedding directly in GPU registers
                    unsafe {
                        driver::launch_cuda_dequantize_token_embd_ptr(
                            d_hid_ptr,
                            d_emb_ptr,
                            d_in_tok_ptr,
                            h as i32,
                            emb_scale,
                            self.token_embd_quant_type as i32,
                            s,
                        );
                    }

                    // 2. Multi-layer forward execution
                    for (layer_idx, layer) in self.layers.iter().enumerate() {
                        // Attention RMSNorm + Q8_1
                        if layer.d_attn_norm.size_bytes() > 0 {
                            let norm_out_ptr = if layer.q_quant == 1 {
                                std::ptr::null_mut()
                            } else {
                                d_norm_ptr
                            };
                            unsafe {
                                driver::launch_cuda_fused_rmsnorm_q8_1(
                                    norm_out_ptr,
                                    d_q8_1_ptr,
                                    d_hid_ptr,
                                    layer.d_attn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    self.rms_norm_eps,
                                    weight_offset,
                                    s,
                                );
                            }
                        }

                        let cur_head_dim = if let Some(ref d_q_norm) = layer.d_attn_q_norm {
                            let nf = d_q_norm.size_bytes() / std::mem::size_of::<f32>();
                            if nf > 0 { nf } else { self.head_dim }
                        } else {
                            self.head_dim
                        };
                        let cur_num_heads = if layer.o_k > 0 && cur_head_dim > 0 {
                            layer.o_k / cur_head_dim
                        } else if cur_head_dim > 0 && layer.q_m > 0 {
                            layer.q_m / cur_head_dim
                        } else {
                            self.num_heads
                        };
                        let cur_num_kv_heads = if cur_head_dim > 0 && layer.kv_m > 0 {
                            layer.kv_m / cur_head_dim
                        } else {
                            self.num_kv_heads
                        };

                        let v_cache_ptr = if layer_idx < self.kv_v.len() {
                            self.kv_v[layer_idx].as_raw_ptr()
                        } else {
                            std::ptr::null_mut()
                        };
                        let k_cache_ptr = if layer_idx < self.kv_k.len() {
                            self.kv_k[layer_idx].as_raw_ptr()
                        } else {
                            std::ptr::null_mut()
                        };

                        // Grouped QKV GEMV writing V-cache directly
                        unsafe {
                            if layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                            {
                                driver::launch_cuda_mmvq_qkv_mixed_ptr(
                                    d_q_ptr,
                                    d_k_ptr,
                                    d_v_ptr,
                                    v_cache_ptr,
                                    d_sp_ptr,
                                    layer.d_wq.as_raw_ptr(),
                                    layer.d_wk.as_raw_ptr(),
                                    layer.d_wv.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.q_m as i32,
                                    layer.kv_m as i32,
                                    layer.q_k as i32,
                                    layer.v_quant as i32,
                                    s,
                                );
                            } else {
                                Self::dispatch_gemv(
                                    d_q_ptr,
                                    &layer.d_wq,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    layer.q_m,
                                    layer.q_k,
                                    layer.q_quant,
                                    s,
                                );
                                Self::dispatch_gemv(
                                    d_k_ptr,
                                    &layer.d_wk,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.k_quant,
                                    s,
                                );
                                Self::dispatch_gemv(
                                    d_v_ptr,
                                    &layer.d_wv,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.v_quant,
                                    s,
                                );
                            }

                            // Add Q, K, V biases for single-token decode if present (e.g. Qwen2 / Qwen2.5)
                            if let Some(ref d_bias) = layer.d_q_bias {
                                driver::launch_cuda_vector_add(
                                    d_q_ptr,
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.q_m as i32,
                                    s,
                                );
                            }
                            if let Some(ref d_bias) = layer.d_k_bias {
                                driver::launch_cuda_vector_add(
                                    d_k_ptr,
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    s,
                                );
                            }
                            if let Some(ref d_bias) = layer.d_v_bias {
                                driver::launch_cuda_vector_add(
                                    d_v_ptr,
                                    d_bias.as_typed_ptr::<f32>(),
                                    layer.kv_m as i32,
                                    s,
                                );
                            }
                        }

                        // Write computed V to FP16 V-cache if not done in grouped GEMV
                        if !v_cache_ptr.is_null() && !(layer.q_quant == 1 && layer.k_quant == 1 && (layer.v_quant == 1 || layer.v_quant == 2) && layer.d_q_bias.is_none() && layer.d_v_bias.is_none()) {
                            unsafe {
                                driver::launch_cuda_write_v_cache_ptr(
                                    v_cache_ptr,
                                    d_v_ptr,
                                    d_sp_ptr,
                                    layer.kv_m as i32,
                                    s,
                                );
                            }
                        }

                        // RoPE writing K-cache directly
                        let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                            10_000.0f32
                        } else {
                            self.rope_theta
                        };
                        let q_norm_ptr = layer
                            .d_attn_q_norm
                            .as_ref()
                            .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                            .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());
                        let k_norm_ptr = layer
                            .d_attn_k_norm
                            .as_ref()
                            .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                            .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());

                        unsafe {
                            driver::launch_cuda_fused_qknorm_rope_batch_ptr(
                                d_q_ptr,
                                d_k_ptr,
                                q_norm_ptr,
                                k_norm_ptr,
                                k_cache_ptr,
                                d_sp_ptr,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                1e-6,
                                layer_rope_theta,
                                1,
                                self.rope_scaling_factor,
                                self.rope_scaling_orig_ctx as i32,
                                s,
                            );
                        }

                        // Flash Decode + Direct Q8_1 Output
                        let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                        unsafe {
                            driver::launch_cuda_flash_decode_ptr(
                                d_out_ptr,
                                d_q8_1_ptr,
                                d_q_ptr,
                                k_cache_ptr,
                                v_cache_ptr,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                d_sp_ptr,
                                0,
                                sm_scale,
                                1,
                                s,
                            );
                        }

                        // Output Projection + Fused Residual 1
                        let o_m = if layer.o_m > 0 { layer.o_m } else { h };
                        let o_k = if layer.o_k > 0 {
                            layer.o_k
                        } else {
                            cur_num_heads * cur_head_dim
                        };
                        unsafe {
                            if layer.o_quant == 1 {
                                driver::launch_cuda_gemv_q4_k_fused_add(
                                    d_hid_ptr,
                                    layer.d_wo.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    o_m as i32,
                                    o_k as i32,
                                    s,
                                );
                            } else if layer.o_quant == 2 {
                                driver::launch_cuda_gemv_q6_k_fused_add(
                                    d_hid_ptr,
                                    layer.d_wo.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    o_m as i32,
                                    o_k as i32,
                                    s,
                                );
                            } else {
                                driver::launch_cuda_quantize_q8_1(
                                    d_out_ptr, d_q8_1_ptr, o_k as i32, s,
                                );
                                Self::dispatch_gemv(
                                    d_out_ptr,
                                    &layer.d_wo,
                                    d_out_ptr,
                                    d_q8_1_ptr,
                                    o_m,
                                    o_k,
                                    layer.o_quant,
                                    s,
                                );
                                driver::launch_cuda_vector_add(d_hid_ptr, d_out_ptr, o_m as i32, s);
                            }
                        }

                        // FFN RMSNorm + Q8_1
                        if layer.d_ffn_norm.size_bytes() > 0 {
                            let norm_ffn_out_ptr = if layer.gate_quant == 1 {
                                std::ptr::null_mut()
                            } else {
                                d_norm_ptr
                            };
                            unsafe {
                                driver::launch_cuda_fused_rmsnorm_q8_1(
                                    norm_ffn_out_ptr,
                                    d_q8_1_ptr,
                                    d_hid_ptr,
                                    layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    self.rms_norm_eps,
                                    weight_offset,
                                    s,
                                );
                            }
                        }

                        // FFN Gate + Up GEMV
                        unsafe {
                            let (pass_up_ptr, pass_swiglu_up) =
                                if !self.is_gemma
                                    && ((layer.gate_quant == 1 && layer.up_quant == 1)
                                        || (layer.gate_quant == 9 && layer.up_quant == 9))
                                {
                                    (std::ptr::null_mut(), std::ptr::null())
                                } else {
                                    (d_up_ptr, d_up_ptr.cast_const())
                                };

                            if layer.gate_quant == 1 && layer.up_quant == 1 {
                                driver::launch_cuda_gemv_gate_up_q4_k(
                                    d_gate_ptr,
                                    pass_up_ptr,
                                    layer.d_w1.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.inter_m as i32,
                                    h as i32,
                                    s,
                                );
                            } else if layer.gate_quant == 9 && layer.up_quant == 9 && !d_q8_1_ptr.is_null() {
                                driver::launch_cuda_gemv_gate_up_q5_0(
                                    d_gate_ptr,
                                    pass_up_ptr,
                                    layer.d_w1.as_raw_ptr(),
                                    layer.d_w3.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    layer.inter_m as i32,
                                    h as i32,
                                    s,
                                );
                            } else {
                                Self::dispatch_gemv(
                                    d_gate_ptr,
                                    &layer.d_w1,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.gate_quant,
                                    s,
                                );
                                Self::dispatch_gemv(
                                    d_up_ptr,
                                    &layer.d_w3,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    layer.inter_m,
                                    h,
                                    layer.up_quant,
                                    s,
                                );
                            }
                            let out_f32_ptr = if layer.down_quant == 1 || layer.down_quant == 2 {
                                std::ptr::null_mut()
                            } else {
                                d_gate_ptr
                            };
                            driver::launch_cuda_fused_swiglu_q8_1_sparse(
                                d_q8_1_ptr,
                                out_f32_ptr,
                                d_gate_ptr,
                                pass_swiglu_up,
                                layer.inter_m as i32,
                                self.ffn_sparsity_threshold,
                                i32::from(self.enable_2_4_sparsity),
                                s,
                            );
                            if layer.down_quant == 1 {
                                driver::launch_cuda_gemv_q4_k_fused_add(
                                    d_hid_ptr,
                                    layer.d_w2.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    h as i32,
                                    layer.inter_m as i32,
                                    s,
                                );
                            } else if layer.down_quant == 2 {
                                driver::launch_cuda_gemv_q6_k_fused_add(
                                    d_hid_ptr,
                                    layer.d_w2.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    h as i32,
                                    layer.inter_m as i32,
                                    s,
                                );
                            } else {
                                Self::dispatch_gemv(
                                    d_gate_ptr,
                                    &layer.d_w2,
                                    d_gate_ptr,
                                    d_q8_1_ptr,
                                    h,
                                    layer.inter_m,
                                    layer.down_quant,
                                    s,
                                );
                                driver::launch_cuda_vector_add(d_hid_ptr, d_gate_ptr, h as i32, s);
                            }
                        }
                    }

                    // 3. Final Output Norm
                    if let Some(d_out_norm) = &self.d_output_norm {
                        let norm_lm_out_ptr =
                            if self.lm_head_quant_type == 1 || self.lm_head_quant_type == 2 {
                                std::ptr::null_mut()
                            } else {
                                d_norm_ptr
                            };
                        unsafe {
                            driver::launch_cuda_fused_rmsnorm_q8_1(
                                norm_lm_out_ptr,
                                d_q8_1_ptr,
                                d_hid_ptr,
                                d_out_norm.as_typed_ptr::<f32>(),
                                1,
                                h as i32,
                                self.rms_norm_eps,
                                weight_offset,
                                s,
                            );
                        }
                    }

                    // 4. LM Head Projection
                    if let Some(d_lm) = &self.d_lm_head {
                        unsafe {
                            if self.lm_head_quant_type == 1 {
                                driver::launch_cuda_gemv_q4_k(
                                    d_logits_ptr,
                                    d_lm.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    self.vocab_size as i32,
                                    h as i32,
                                    s,
                                );
                            } else if self.lm_head_quant_type == 2 {
                                driver::launch_cuda_gemv_q6_k(
                                    d_logits_ptr,
                                    d_lm.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    self.vocab_size as i32,
                                    h as i32,
                                    s,
                                );
                            } else if self.lm_head_quant_type == 0 || self.lm_head_quant_type == 3 {
                                driver::launch_cuda_mmvq_q8_0(
                                    d_logits_ptr,
                                    d_lm.as_raw_ptr(),
                                    d_q8_1_ptr,
                                    self.vocab_size as i32,
                                    h as i32,
                                    1,
                                    s,
                                );
                            } else {
                                Self::dispatch_gemv(
                                    d_logits_ptr,
                                    d_lm,
                                    d_norm_ptr,
                                    d_q8_1_ptr,
                                    self.vocab_size,
                                    h,
                                    self.lm_head_quant_type,
                                    s,
                                );
                            }
                        }
                    }

                    // 5. Native GPU Sampling (Temperature / Gumbel-Max / ArgMax)
                    let temp = self.sampler.config.temperature;
                    let top_p = self.sampler.config.top_p;
                    let seed = self.sampler.config.seed.unwrap_or(42);
                    unsafe {
                        driver::launch_cuda_sample_native(
                            d_logits_ptr,
                            self.vocab_size as i32,
                            d_sampled_ptr,
                            d_val_ptr,
                            temp,
                            top_p,
                            seed,
                            d_sp_ptr,
                            s,
                        );
                        // 6. Advance decode step on device
                        driver::launch_cuda_advance_decode_step(
                            d_sp_ptr,
                            d_sampled_ptr,
                            d_in_tok_ptr,
                            self.pinned_host_token(),
                            s,
                        );
                    }

                    let end_res = unsafe {
                        self.graph_manager.end_capture(
                            s,
                            "oxide_decode_graph",
                            self.layers.len(),
                            1,
                        )
                    };

                    match end_res {
                        Ok(handle) => {
                            tracing::info!(
                                "CUDA decode graph captured successfully (id={}, nodes={})",
                                handle.graph_id,
                                handle.node_count
                            );
                            self.decode_graph_handle = Some(handle);
                            let t_graph_start = std::time::Instant::now();
                            unsafe {
                                self.graph_manager.launch_graph(handle, s)?;
                            }
                            if self.pinned_host_token_ptr != 0 {
                                let expected_step = self.current_step_counter + 1;
                                let h_tok_ptr = self.pinned_host_token();
                                let t_spin_start = std::time::Instant::now();
                                while unsafe { std::ptr::read_volatile(h_tok_ptr.add(1)) }
                                    < expected_step
                                {
                                    std::hint::spin_loop();
                                    if t_spin_start.elapsed().as_millis() > 2000 {
                                        stream.synchronize().map_err(EngineError::BackendError)?;
                                        break;
                                    }
                                }
                                self.current_step_counter = expected_step;
                                let graph_dt = t_graph_start.elapsed();
                                let best_tok = unsafe { std::ptr::read_volatile(h_tok_ptr) };
                                if self.seq_pos.is_multiple_of(16) {
                                    tracing::info!(
                                        "CUDA decode graph step dt: {:.3}ms ({:.2} tok/s)",
                                        graph_dt.as_secs_f64() * 1000.0,
                                        1.0 / graph_dt.as_secs_f64()
                                    );
                                }
                                self.last_sampled_token = Some(best_tok);
                                self.seq_pos += 1;
                                self.last_gpu_seq_pos = Some(self.seq_pos);
                                if slot < self.slot_positions.len() {
                                    self.slot_positions[slot] = self.seq_pos;
                                }
                                if slot < self.host_token_buffer.len() {
                                    self.host_token_buffer[slot] = best_tok;
                                }
                                return Ok(event);
                            } else if let Some(ref d_sampled) = self.d_sampled_token {
                                let mut sampled_res = [0u32, 0u32];
                                let _ = d_sampled.copy_to_host_async(&mut sampled_res, s);
                                stream.synchronize().map_err(EngineError::BackendError)?;
                                let best_tok = sampled_res[0];
                                self.last_sampled_token = Some(best_tok);
                                self.seq_pos += 1;
                                self.last_gpu_seq_pos = Some(self.seq_pos);
                                if slot < self.slot_positions.len() {
                                    self.slot_positions[slot] = self.seq_pos;
                                }
                                if slot < self.host_token_buffer.len() {
                                    self.host_token_buffer[slot] = best_tok;
                                }
                                return Ok(event);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "CUDA graph capture failed: {}, falling back to uncaptured stream dispatch",
                                e
                            );
                            self.cuda_graph_enabled = false;
                        }
                    }
                } else {
                    tracing::warn!(
                        "cudaStreamBeginCapture failed, falling back to uncaptured stream dispatch"
                    );
                    self.cuda_graph_enabled = false;
                }
            }
        }

        if !self.layers.is_empty() && self.hidden_dim > 0 {
            let h = self.hidden_dim;
            let pos = self.seq_pos;
            self.seq_pos += 1;

            // 1. Initial embedding lookup on GPU (or dequantized from host)
            if let Some(d_emb) = &self.d_token_embd {
                let tok_idx = (cmd.input_token as usize) % self.vocab_size.max(1);
                let emb_scale = if self.is_gemma {
                    (h as f32).sqrt()
                } else {
                    1.0f32
                };
                if let Some(d_hid) = &mut self.d_hidden {
                    if self.token_embd_quant_type > 0 {
                        unsafe {
                            driver::launch_cuda_dequantize_token_embd(
                                d_hid.as_typed_ptr::<f32>(),
                                d_emb.as_raw_ptr(),
                                tok_idx as i32,
                                h as i32,
                                emb_scale,
                                self.token_embd_quant_type as i32,
                                s,
                            );
                        }
                    } else {
                        let offset = tok_idx * h * std::mem::size_of::<f32>();
                        if offset + h * 4 <= d_emb.size_bytes() {
                            let _ = unsafe {
                                driver::cudaMemcpyAsync(
                                    d_hid.as_raw_ptr(),
                                    (d_emb.as_raw_ptr() as *const u8).add(offset)
                                        as *const std::ffi::c_void,
                                    h * std::mem::size_of::<f32>(),
                                    driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                    s,
                                )
                            };
                        }
                    }
                }
            } else if let Some(lookup) = &self.embedding_lookup {
                let tok_idx = (cmd.input_token as usize) % self.vocab_size.max(1);
                if self.host_emb_buffer.len() != h {
                    self.host_emb_buffer.resize(h, 0.0f32);
                }
                lookup(tok_idx, &mut self.host_emb_buffer);
                if let Some(d_hid) = &mut self.d_hidden {
                    let _ = d_hid.copy_from_host_async(&self.host_emb_buffer, s);
                }
            }

            let tok_idx = (cmd.input_token as usize) % self.vocab_size.max(1);
            if let (Some(d_ple), Some(ple_fn)) =
                (&mut self.d_per_layer_emb, &self.per_layer_embedding_lookup)
            {
                ple_fn(tok_idx, &mut self.host_ple_buffer);
                let _ = unsafe {
                    driver::cudaMemcpyAsync(
                        d_ple.as_raw_ptr(),
                        self.host_ple_buffer.as_ptr() as *const std::ffi::c_void,
                        self.host_ple_buffer.len() * std::mem::size_of::<f32>(),
                        driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                        s,
                    )
                };
            }

            let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
            let check_cuda = |name: &str, l_idx: usize| -> Result<()> {
                #[cfg(debug_assertions)]
                {
                    let err = unsafe { driver::cudaGetLastError() };
                    if err != 0 {
                        tracing::error!("CUDA error at step {} (layer {}): {}", name, l_idx, err);
                        return Err(EngineError::BackendError(format!(
                            "CUDA error at {name} (layer {l_idx}): {err}"
                        )));
                    }
                }
                let _ = (name, l_idx);
                Ok(())
            };

            let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
                buf.as_raw_ptr()
            } else {
                std::ptr::null_mut()
            };

            // 2. Multi-layer forward pass on physical GPU
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if let (Some(d_hid), Some(d_norm)) = (&self.d_hidden, &mut self.d_scratch_norm) {
                    // 0. PLE (Per-Layer Embeddings) Block (Gemma 4)
                    if let (Some(d_inp_gate), Some(d_proj), Some(d_ple_all)) =
                        (&layer.d_inp_gate, &layer.d_proj, &self.d_per_layer_emb)
                    {
                        if let (Some(d_gate256), Some(d_ple256), Some(d_act)) = (
                            &mut self.d_scratch_gate256,
                            &mut self.d_scratch_ple256,
                            &mut self.d_scratch_act,
                        ) {
                            let ple_offset = layer_idx * 256;
                            let ple_ptr =
                                unsafe { (d_ple_all.as_raw_ptr() as *const f32).add(ple_offset) };

                            // 1. inp_gate GEMV: [256, h] * hidden -> gate256
                            if !d_q8_1_ptr.is_null() {
                                unsafe {
                                    driver::launch_cuda_quantize_q8_1(
                                        d_hid.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        h as i32,
                                        s,
                                    );
                                }
                            }
                            unsafe {
                                Self::dispatch_gemv(
                                    d_gate256.as_typed_ptr::<f32>(),
                                    d_inp_gate,
                                    d_hid.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    256,
                                    h,
                                    layer.inp_gate_quant,
                                    s,
                                );
                            }
                            check_cuda("ple_inp_gate", layer_idx)?;

                            // 2. Elementwise GeLU(gate256) * ple256 -> ple256
                            unsafe {
                                driver::launch_cuda_gelu(
                                    d_ple256.as_typed_ptr::<f32>(),
                                    d_gate256.as_typed_ptr::<f32>(),
                                    ple_ptr,
                                    256,
                                    s,
                                );
                            }
                            check_cuda("ple_gelu", layer_idx)?;

                            // 3. proj GEMV: [h, 256] * ple256 -> act (dim h)
                            if !d_q8_1_ptr.is_null() {
                                unsafe {
                                    driver::launch_cuda_quantize_q8_1(
                                        d_ple256.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        256,
                                        s,
                                    );
                                }
                            }
                            unsafe {
                                Self::dispatch_gemv(
                                    d_act.as_typed_ptr::<f32>(),
                                    d_proj,
                                    d_ple256.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    h,
                                    256,
                                    layer.proj_quant,
                                    s,
                                );
                            }
                            check_cuda("ple_proj", layer_idx)?;

                            // 4. Residual add to hidden
                            unsafe {
                                driver::launch_cuda_vector_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    d_act.as_typed_ptr::<f32>(),
                                    h as i32,
                                    s,
                                );
                            }
                            check_cuda("ple_residual", layer_idx)?;

                            // 5. Post-PLE RMSNorm (post_per_layer_input_norm)
                            if let Some(d_post_norm) = &layer.d_post_norm {
                                if d_post_norm.size_bytes() > 0 {
                                    unsafe {
                                        driver::launch_cuda_rmsnorm(
                                            d_hid.as_typed_ptr::<f32>(),
                                            d_hid.as_typed_ptr::<f32>(),
                                            d_post_norm.as_typed_ptr::<f32>(),
                                            1,
                                            h as i32,
                                            self.rms_norm_eps,
                                            weight_offset,
                                            s,
                                        );
                                    }
                                    check_cuda("ple_post_norm", layer_idx)?;
                                }
                            }
                        }
                    }

                    // Attention Block
                    let has_attn =
                        layer.d_wq.size_bytes() > 0 && layer.d_wo.size_bytes() > 0 && layer.q_m > 0;
                    if has_attn {
                        // Attention RMSNorm + Q8_1 quantization fused
                        if layer.d_attn_norm.size_bytes() > 0 {
                            unsafe {
                                driver::launch_cuda_fused_rmsnorm_q8_1(
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_attn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    self.rms_norm_eps,
                                    weight_offset,
                                    s,
                                );
                            }
                            check_cuda("attn_fused_rmsnorm", layer_idx)?;
                            if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                                let mut h_norm = [0.0f32; 4];
                                let _ = d_norm.copy_to_host_async(&mut h_norm, s);
                                let _ = stream.synchronize();
                                eprintln!("[CUDA L0] attn_norm: {:?}", h_norm);
                            }
                        }

                        let cur_head_dim = if let Some(ref d_q_norm) = layer.d_attn_q_norm {
                            let nf = d_q_norm.size_bytes() / std::mem::size_of::<f32>();
                            if nf > 0 { nf } else { self.head_dim }
                        } else {
                            self.head_dim
                        };
                        let cur_num_heads = if layer.o_k > 0 && cur_head_dim > 0 {
                            layer.o_k / cur_head_dim
                        } else if cur_head_dim > 0 && layer.q_m > 0 {
                            layer.q_m / cur_head_dim
                        } else {
                            self.num_heads
                        };
                        let cur_num_kv_heads = if cur_head_dim > 0 && layer.kv_m > 0 {
                            layer.kv_m / cur_head_dim
                        } else {
                            self.num_kv_heads
                        };

                        let kv_cap = self.kv_capacity.max(64);
                        if pos >= kv_cap {
                            return Err(EngineError::BackendError(format!(
                                "Sequence length {pos} exceeds allocated KV cache capacity {kv_cap}"
                            )));
                        }
                        let kv_dim = cur_num_kv_heads * cur_head_dim;
                        let v_cache_ptr = if layer_idx < self.kv_v.len() {
                            self.kv_v[layer_idx].as_raw_ptr()
                        } else {
                            std::ptr::null_mut()
                        };
                        let k_cache_ptr = if layer_idx < self.kv_k.len() {
                            self.kv_k[layer_idx].as_raw_ptr()
                        } else {
                            std::ptr::null_mut()
                        };
                        let v_dst_ptr = if v_cache_ptr.is_null() {
                            std::ptr::null_mut()
                        } else {
                            unsafe {
                                (v_cache_ptr as *mut u8).add(pos * kv_dim * 2)
                                    as *mut std::ffi::c_void
                            }
                        };

                        let (q_done, k_done, v_done) = if let (Some(d_q), Some(d_k), Some(d_v)) = (
                            &mut self.d_scratch_q,
                            &mut self.d_scratch_k,
                            &mut self.d_scratch_v,
                        ) {
                            if layer.q_quant == 1
                                && layer.k_quant == 1
                                && (layer.v_quant == 1 || layer.v_quant == 2)
                                && layer.q_k == layer.kv_k
                                && !d_q8_1_ptr.is_null()
                            {
                                unsafe {
                                    driver::launch_cuda_mmvq_qkv_mixed(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_k.as_typed_ptr::<f32>(),
                                        d_v.as_typed_ptr::<f32>(),
                                        v_cache_ptr,
                                        pos as i32,
                                        layer.d_wq.as_raw_ptr(),
                                        layer.d_wk.as_raw_ptr(),
                                        layer.d_wv.as_raw_ptr(),
                                        d_q8_1_ptr,
                                        layer.q_m as i32,
                                        layer.kv_m as i32,
                                        layer.q_k as i32,
                                        layer.v_quant as i32,
                                        1,
                                        s,
                                    );
                                }
                                check_cuda("qkv_grouped_proj", layer_idx)?;
                                (true, true, !v_cache_ptr.is_null())
                            } else {
                                (false, false, false)
                            }
                        } else {
                            (false, false, false)
                        };

                        // Q Projection fallback
                        if !q_done {
                            if let Some(d_q) = &mut self.d_scratch_q {
                                unsafe {
                                    Self::dispatch_gemv(
                                        d_q.as_typed_ptr::<f32>(),
                                        &layer.d_wq,
                                        d_norm.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        layer.q_m,
                                        layer.q_k,
                                        layer.q_quant,
                                        s,
                                    );
                                }
                                check_cuda("q_proj", layer_idx)?;
                            }
                        }

                        // K Projection fallback
                        if !k_done {
                            if let Some(d_k) = &mut self.d_scratch_k {
                                unsafe {
                                    Self::dispatch_gemv(
                                        d_k.as_typed_ptr::<f32>(),
                                        &layer.d_wk,
                                        d_norm.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        layer.kv_m,
                                        layer.kv_k,
                                        layer.k_quant,
                                        s,
                                    );
                                }
                                check_cuda("k_proj", layer_idx)?;
                            }
                        }

                        // V Projection: computes GEMV into scratch buffer, then converts to FP16 in KV cache
                        if !v_done {
                            if let Some(d_v) = &mut self.d_scratch_v {
                                unsafe {
                                    Self::dispatch_gemv(
                                        d_v.as_typed_ptr::<f32>(),
                                        &layer.d_wv,
                                        d_norm.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        layer.kv_m,
                                        layer.kv_k,
                                        layer.v_quant,
                                        s,
                                    );
                                }
                                check_cuda("v_proj", layer_idx)?;
                            }
                        }

                        // Add Q, K, V biases if present (e.g. Qwen2 / Qwen2.5)
                        if let Some(ref d_bias) = layer.d_q_bias {
                            if let Some(d_q) = &mut self.d_scratch_q {
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_bias.as_typed_ptr::<f32>(),
                                        layer.q_m as i32,
                                        s,
                                    );
                                }
                            }
                        }
                        if let Some(ref d_bias) = layer.d_k_bias {
                            if let Some(d_k) = &mut self.d_scratch_k {
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        d_k.as_typed_ptr::<f32>(),
                                        d_bias.as_typed_ptr::<f32>(),
                                        layer.kv_m as i32,
                                        s,
                                    );
                                }
                            }
                        }
                        if let Some(ref d_bias) = layer.d_v_bias {
                            if let Some(d_v) = &mut self.d_scratch_v {
                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        d_v.as_typed_ptr::<f32>(),
                                        d_bias.as_typed_ptr::<f32>(),
                                        layer.kv_m as i32,
                                        s,
                                    );
                                }
                            }
                        }

                        if !v_done {
                            if let Some(d_v) = &self.d_scratch_v {
                                if !v_dst_ptr.is_null() {
                                    unsafe {
                                        driver::launch_cuda_convert_f32_to_f16(
                                            v_dst_ptr,
                                            d_v.as_typed_ptr::<f32>(),
                                            kv_dim as i32,
                                            s,
                                        );
                                    }
                                }
                            }
                        }

                        if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                            if let (Some(d_q), Some(d_k), Some(d_v)) = (&self.d_scratch_q, &self.d_scratch_k, &self.d_scratch_v) {
                                let mut h_q = [0.0f32; 4];
                                let mut h_k = [0.0f32; 4];
                                let mut h_v = [0.0f32; 4];
                                let _ = d_q.copy_to_host_async(&mut h_q, s);
                                let _ = d_k.copy_to_host_async(&mut h_k, s);
                                let _ = d_v.copy_to_host_async(&mut h_v, s);
                                let _ = stream.synchronize();
                                eprintln!("[CUDA L0] q_after_bias: {:?}", h_q);
                                eprintln!("[CUDA L0] k_after_bias: {:?}", h_k);
                                eprintln!("[CUDA L0] v_after_bias: {:?}", h_v);
                            }
                        }

                        // Fused QK-Norm + Rotary Position Embedding (RoPE) on Q and K + direct FP16 K-cache write
                        let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                            10_000.0f32
                        } else {
                            self.rope_theta
                        };
                        let q_norm_ptr = layer
                            .d_attn_q_norm
                            .as_ref()
                            .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                            .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());
                        let k_norm_ptr = layer
                            .d_attn_k_norm
                            .as_ref()
                            .filter(|b| b.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>())
                            .map_or(std::ptr::null(), |b| b.as_typed_ptr::<f32>());

                        if let (Some(d_q), Some(d_k)) =
                            (&mut self.d_scratch_q, &mut self.d_scratch_k)
                        {
                            unsafe {
                                driver::launch_cuda_fused_qknorm_rope_batch(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_k.as_typed_ptr::<f32>(),
                                    q_norm_ptr,
                                    k_norm_ptr,
                                    k_cache_ptr,
                                    pos as i32,
                                    1,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    1e-6,
                                    layer_rope_theta,
                                    1, // is_fp16
                                    self.rope_scaling_factor,
                                    self.rope_scaling_orig_ctx as i32,
                                    s,
                                );
                            }
                            check_cuda("fused_qknorm_rope", layer_idx)?;
                        }

                        // Attention Flash Decode (with GQA and Sliding Window Attention)
                        if let (Some(d_q), Some(d_out)) =
                            (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                        {
                            if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                                let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                                let start_token = if let Some(w) = self.sliding_window {
                                    if self.is_gemma && cur_head_dim <= 256 {
                                        (pos + 1).saturating_sub(w) as i32
                                    } else {
                                        0
                                    }
                                } else {
                                    0
                                };
                                let num_tokens = (pos + 1).min(kv_cap) as i32;
                                unsafe {
                                    driver::launch_cuda_flash_decode(
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        d_q.as_typed_ptr::<f32>(),
                                        self.kv_k[layer_idx].as_raw_ptr(),
                                        self.kv_v[layer_idx].as_raw_ptr(),
                                        cur_num_heads as i32,
                                        cur_num_kv_heads as i32,
                                        cur_head_dim as i32,
                                        num_tokens,
                                        start_token,
                                        sm_scale,
                                        1, // is_fp16
                                        s,
                                    );
                                }
                                check_cuda("flash_decode", layer_idx)?;
                                if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                                    let mut h_attn = [0.0f32; 4];
                                    let _ = d_out.copy_to_host_async(&mut h_attn, s);
                                    let _ = stream.synchronize();
                                    eprintln!("[CUDA L0] attn_out: {:?}", h_attn);
                                }
                            }
                        }

                        // Output Projection: wo x attn_out -> d_hid (Fused Residual 1)
                        let has_post_attn_norm = layer
                            .d_post_attn_norm
                            .as_ref()
                            .is_some_and(|p| p.size_bytes() > 0);

                        if has_post_attn_norm {
                            if let (Some(d_out), Some(d_act)) =
                                (&self.d_scratch_attn_out, &mut self.d_scratch_act)
                            {
                                let o_m = if layer.o_m > 0 { layer.o_m } else { h };
                                let o_k = if layer.o_k > 0 {
                                    layer.o_k
                                } else {
                                    cur_num_heads * cur_head_dim
                                };
                                if !d_q8_1_ptr.is_null() {
                                    unsafe {
                                        driver::launch_cuda_quantize_q8_1(
                                            d_out.as_typed_ptr::<f32>(),
                                            d_q8_1_ptr,
                                            o_k as i32,
                                            s,
                                        );
                                    }
                                }
                                unsafe {
                                    Self::dispatch_gemv(
                                        d_act.as_typed_ptr::<f32>(),
                                        &layer.d_wo,
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        o_m,
                                        o_k,
                                        layer.o_quant,
                                        s,
                                    );
                                }
                                check_cuda("wo_proj", layer_idx)?;

                                if let Some(d_post_attn) = &layer.d_post_attn_norm {
                                    unsafe {
                                        driver::launch_cuda_rmsnorm(
                                            d_act.as_typed_ptr::<f32>(),
                                            d_act.as_typed_ptr::<f32>(),
                                            d_post_attn.as_typed_ptr::<f32>(),
                                            1,
                                            h as i32,
                                            self.rms_norm_eps,
                                            weight_offset,
                                            s,
                                        );
                                    }
                                    check_cuda("post_attn_norm", layer_idx)?;
                                }

                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        d_hid.as_typed_ptr::<f32>(),
                                        d_act.as_typed_ptr::<f32>(),
                                        h as i32,
                                        s,
                                    );
                                }
                                check_cuda("residual_1", layer_idx)?;
                            }
                        } else if let (Some(d_out), Some(d_act)) =
                            (&self.d_scratch_attn_out, &mut self.d_scratch_act)
                        {
                            let o_m = if layer.o_m > 0 { layer.o_m } else { h };
                            let o_k = if layer.o_k > 0 {
                                layer.o_k
                            } else {
                                cur_num_heads * cur_head_dim
                            };
                            if !d_q8_1_ptr.is_null() {
                                unsafe {
                                    driver::launch_cuda_quantize_q8_1(
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        o_k as i32,
                                        s,
                                    );
                                }
                            }
                            unsafe {
                                Self::dispatch_gemv_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    &layer.d_wo,
                                    d_out.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    o_m,
                                    o_k,
                                    layer.o_quant,
                                    s,
                                    d_act,
                                );
                            }
                            check_cuda("wo_proj_fused_residual", layer_idx)?;
                        }

                        if layer_idx == 0 && std::env::var("OXIDE_DEBUG_LAYER").is_ok() {
                            let mut h_hid = [0.0f32; 4];
                            let _ = d_hid.copy_to_host_async(&mut h_hid, s);
                            let _ = stream.synchronize();
                            eprintln!("[CUDA L0] hid_after_attn: {:?}", h_hid);
                        }
                    }

                    // FFN Block
                    let has_ffn = layer.d_w1.size_bytes() > 0
                        && layer.d_w2.size_bytes() > 0
                        && layer.inter_m > 0;
                    if has_ffn {
                        // FFN RMSNorm + Q8_1 quantization fused
                        if layer.d_ffn_norm.size_bytes() > 0 {
                            unsafe {
                                driver::launch_cuda_fused_rmsnorm_q8_1(
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    self.rms_norm_eps,
                                    weight_offset,
                                    s,
                                );
                            }
                            check_cuda("ffn_fused_rmsnorm", layer_idx)?;
                        }

                        // FFN Gate and Up projections
                        if let (Some(d_gate), Some(d_up)) =
                            (&mut self.d_scratch_gate, &mut self.d_scratch_up)
                        {
                            unsafe {
                                let (pass_up_ptr, pass_swiglu_up) = if !self.is_gemma
                                    && ((layer.gate_quant == 1 && layer.up_quant == 1)
                                        || (layer.gate_quant == 9 && layer.up_quant == 9))
                                {
                                    (std::ptr::null_mut(), std::ptr::null())
                                } else {
                                    (
                                        d_up.as_typed_ptr::<f32>(),
                                        d_up.as_typed_ptr::<f32>().cast_const(),
                                    )
                                };

                                if layer.gate_quant == 1
                                    && layer.up_quant == 1
                                    && !d_q8_1_ptr.is_null()
                                {
                                    driver::launch_cuda_gemv_gate_up_q4_k(
                                        d_gate.as_typed_ptr::<f32>(),
                                        pass_up_ptr,
                                        layer.d_w1.as_raw_ptr(),
                                        layer.d_w3.as_raw_ptr(),
                                        d_q8_1_ptr,
                                        layer.inter_m as i32,
                                        h as i32,
                                        s,
                                    );
                                } else if layer.gate_quant == 9
                                    && layer.up_quant == 9
                                    && !d_q8_1_ptr.is_null()
                                {
                                    driver::launch_cuda_gemv_gate_up_q5_0(
                                        d_gate.as_typed_ptr::<f32>(),
                                        pass_up_ptr,
                                        layer.d_w1.as_raw_ptr(),
                                        layer.d_w3.as_raw_ptr(),
                                        d_q8_1_ptr,
                                        layer.inter_m as i32,
                                        h as i32,
                                        s,
                                    );
                                } else {
                                    Self::dispatch_gemv(
                                        d_gate.as_typed_ptr::<f32>(),
                                        &layer.d_w1,
                                        d_norm.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        layer.inter_m,
                                        h,
                                        layer.gate_quant,
                                        s,
                                    );
                                    Self::dispatch_gemv(
                                        d_up.as_typed_ptr::<f32>(),
                                        &layer.d_w3,
                                        d_norm.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        layer.inter_m,
                                        h,
                                        layer.up_quant,
                                        s,
                                    );
                                }
                                check_cuda("ffn_gate_up", layer_idx)?;
                                if self.is_gemma {
                                    driver::launch_cuda_gelu(
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_up.as_typed_ptr::<f32>(),
                                        layer.inter_m as i32,
                                        s,
                                    );
                                    if !d_q8_1_ptr.is_null() {
                                        driver::launch_cuda_quantize_q8_1(
                                            d_gate.as_typed_ptr::<f32>(),
                                            d_q8_1_ptr,
                                            layer.inter_m as i32,
                                            s,
                                        );
                                    }
                                } else if !d_q8_1_ptr.is_null() {
                                    // Fused SwiGLU directly writes quantized Q8_1 blocks for down-projection
                                    let out_f32_ptr =
                                        if layer.down_quant == 1 || layer.down_quant == 2 {
                                            std::ptr::null_mut()
                                        } else {
                                            d_gate.as_typed_ptr::<f32>()
                                        };
                                    driver::launch_cuda_fused_swiglu_q8_1_sparse(
                                        d_q8_1_ptr,
                                        out_f32_ptr,
                                        d_gate.as_typed_ptr::<f32>(),
                                        pass_swiglu_up,
                                        layer.inter_m as i32,
                                        self.ffn_sparsity_threshold,
                                        i32::from(self.enable_2_4_sparsity),
                                        s,
                                    );
                                } else {
                                    driver::launch_cuda_swiglu(
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_up.as_typed_ptr::<f32>(),
                                        layer.inter_m as i32,
                                        s,
                                    );
                                }
                                check_cuda("ffn_act", layer_idx)?;
                            }
                        }

                        // FFN Down Projection: w2 x gate -> d_hid (Fused Residual 2)
                        let has_post_ffn_norm = layer
                            .d_post_ffn_norm
                            .as_ref()
                            .is_some_and(|p| p.size_bytes() > 0);

                        if has_post_ffn_norm {
                            if let (Some(d_gate), Some(d_act)) =
                                (&self.d_scratch_gate, &mut self.d_scratch_act)
                            {
                                unsafe {
                                    Self::dispatch_gemv(
                                        d_act.as_typed_ptr::<f32>(),
                                        &layer.d_w2,
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_q8_1_ptr,
                                        h,
                                        layer.inter_m,
                                        layer.down_quant,
                                        s,
                                    );
                                }
                                check_cuda("ffn_down", layer_idx)?;

                                if let Some(d_post_ffn) = &layer.d_post_ffn_norm {
                                    unsafe {
                                        driver::launch_cuda_rmsnorm(
                                            d_act.as_typed_ptr::<f32>(),
                                            d_act.as_typed_ptr::<f32>(),
                                            d_post_ffn.as_typed_ptr::<f32>(),
                                            1,
                                            h as i32,
                                            self.rms_norm_eps,
                                            weight_offset,
                                            s,
                                        );
                                    }
                                    check_cuda("post_ffn_norm", layer_idx)?;
                                }

                                unsafe {
                                    driver::launch_cuda_vector_add(
                                        d_hid.as_typed_ptr::<f32>(),
                                        d_act.as_typed_ptr::<f32>(),
                                        h as i32,
                                        s,
                                    );
                                }
                                check_cuda("residual_2", layer_idx)?;
                            }
                        } else if let (Some(d_gate), Some(d_act)) =
                            (&self.d_scratch_gate, &mut self.d_scratch_act)
                        {
                            unsafe {
                                Self::dispatch_gemv_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    &layer.d_w2,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_q8_1_ptr,
                                    h,
                                    layer.inter_m,
                                    layer.down_quant,
                                    s,
                                    d_act,
                                );
                            }
                            check_cuda("ffn_down_fused_residual", layer_idx)?;
                        }

                        // Layer scalar output scale (Gemma 4 / Nemotron)
                        if let Some(scale) = layer.layer_output_scale {
                            unsafe {
                                driver::launch_cuda_vector_scale(
                                    d_hid.as_typed_ptr::<f32>(),
                                    scale,
                                    h as i32,
                                    s,
                                );
                            }
                            check_cuda("layer_scale", layer_idx)?;
                        }
                    }
                }
            }

            // 3. Final Output Norm + Q8_1 quantization fused
            if let (Some(d_hid), Some(d_norm), Some(d_out_norm)) = (
                &self.d_hidden,
                &mut self.d_scratch_norm,
                &self.d_output_norm,
            ) {
                if d_out_norm.size_bytes() > 0 {
                    unsafe {
                        driver::launch_cuda_fused_rmsnorm_q8_1(
                            d_norm.as_typed_ptr::<f32>(),
                            d_q8_1_ptr,
                            d_hid.as_typed_ptr::<f32>(),
                            d_out_norm.as_typed_ptr::<f32>(),
                            1,
                            h as i32,
                            self.rms_norm_eps,
                            weight_offset,
                            s,
                        );
                    }
                }
            }

            // 4. LM Head Logit Projection (full vocabulary)
            if cmd.is_prefill_chunk {
                if slot < self.host_token_buffer.len() {
                    self.host_token_buffer[slot] = cmd.input_token;
                }
                return Ok(event);
            }

            if let (Some(d_norm), Some(d_lm), Some(d_logits)) =
                (&self.d_scratch_norm, &self.d_lm_head, &mut self.d_logits)
            {
                unsafe {
                    Self::dispatch_gemv(
                        d_logits.as_typed_ptr::<f32>(),
                        d_lm,
                        d_norm.as_typed_ptr::<f32>(),
                        d_q8_1_ptr,
                        self.vocab_size,
                        h,
                        self.lm_head_quant_type,
                        s,
                    );
                }
                let err = unsafe { driver::cudaGetLastError() };
                if err != 0 {
                    tracing::error!("CUDA error after LM Head launch: {}", err);
                }

                if let Some(cap) = self.final_logit_softcapping {
                    if cap > 0.0 {
                        unsafe {
                            driver::launch_cuda_logit_softcap(
                                d_logits.as_typed_ptr::<f32>(),
                                self.vocab_size as i32,
                                cap,
                                s,
                            );
                        }
                    }
                }

                // On-device argmax when greedy, avoiding 1MB host-device memory transfers per token
                let selected_token = if self.sampler.config.temperature <= 0.0 {
                    if let Some(ref mut d_sampled) = self.d_sampled_token {
                        let d_tok_ptr = d_sampled.as_raw_ptr() as *mut u32;
                        let d_val_ptr = unsafe { d_tok_ptr.add(1) as *mut f32 };
                        unsafe {
                            driver::launch_cuda_argmax(
                                d_logits.as_typed_ptr::<f32>(),
                                self.vocab_size as i32,
                                d_tok_ptr,
                                d_val_ptr,
                                1,
                                s,
                            );
                        }
                        let mut sampled_res = [0u32, 0u32];
                        let _ = d_sampled.copy_to_host_async(&mut sampled_res, s);
                        if let Err(e) = stream.synchronize() {
                            tracing::error!("stream.synchronize failed after argmax: {}", e);
                        }
                        let best_tok = sampled_res[0];
                        let max_val = f32::from_bits(sampled_res[1]);
                        tracing::debug!(
                            "CUDA on-device argmax pos {}: sampled_tok={}, max_val={}",
                            pos,
                            best_tok,
                            max_val
                        );
                        Some(best_tok)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let selected_token = if let Some(tok) = selected_token {
                    tok
                } else {
                    if self.host_logits_buffer.len() != self.vocab_size {
                        self.host_logits_buffer.resize(self.vocab_size, 0.0f32);
                    }
                    let host_logits = &mut self.host_logits_buffer;
                    let _ = d_logits.copy_to_host_async(host_logits, s);
                    if let Err(e) = stream.synchronize() {
                        tracing::error!("stream.synchronize failed after LM Head: {}", e);
                    }

                    let mut best_tok = 1u32;
                    let mut max_val = f32::NEG_INFINITY;
                    for (idx, &v) in host_logits.iter().enumerate() {
                        if v > max_val {
                            max_val = v;
                            best_tok = idx as u32;
                        }
                    }

                    // Sample token using AcademicSamplerEngine (temperature, top-p, min-p, DRY repetition penalty)
                    let tok = self
                        .sampler
                        .sample_token(host_logits, &mut self.sampler_state, 198)
                        .unwrap_or(best_tok);

                    tracing::debug!(
                        "CUDA host sampled pos {}: greedy_top={}, sampled_tok={}, max_val={}",
                        pos,
                        best_tok,
                        tok,
                        max_val
                    );
                    tok
                };

                if slot < self.host_token_buffer.len() {
                    self.host_token_buffer[slot] = selected_token;
                }
                self.seq_pos += 1;
                if slot < self.slot_positions.len() {
                    self.slot_positions[slot] = self.seq_pos;
                }
            }
        } else {
            return Err(oxide_core::error::EngineError::BackendError(
                "Physical CUDA transformer layers not loaded into GPU VRAM for execution"
                    .to_string(),
            ));
        }

        Ok(event)
    }

    fn query_event_completed(&self, _event: Self::Event) -> bool {
        true
    }

    fn read_sampled_token_host(&self, slot_idx: u16) -> u32 {
        let slot = slot_idx as usize;
        if slot < self.host_token_buffer.len() {
            self.host_token_buffer[slot]
        } else {
            0
        }
    }

    fn synchronize(&self) -> Result<()> {
        if let Some(stream) = &self.stream {
            let _ = stream.synchronize();
        }
        Ok(())
    }
}
