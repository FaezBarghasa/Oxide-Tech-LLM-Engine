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
    LiveMmioRegisters, MmioAccessStatus, PciBarRegion, PhysicalPciGpu, UnlockResult, nv_rm,
    registers as gpu_unlock_registers,
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
    pub embedding_lookup: Option<EmbeddingLookupFn>,
    pub mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
    pub lm_head_quant_type: u32,
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

        // Attempt real physical device initialization
        let stream = CudaStream::new().ok();
        if stream.is_some() {
            #[allow(clippy::cast_possible_wrap)]
            // SAFETY: device_id is a valid integer device index passed to cudaSetDevice.
            let _ = unsafe { driver::cudaSetDevice(device_id as i32) };
            // SAFETY: Configures physical device cache preferences (L1 preference, 0 shared memory carveout)
            unsafe { driver::init_cuda_kernel_attributes() };
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
            embedding_lookup: None,
            mmap_handle: None,
            lm_head_quant_type: 1,
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
        }
    }

    pub fn configure_hyperparameters(
        &mut self,
        rms_norm_eps: f32,
        is_qwen: bool,
        sliding_window: Option<usize>,
        kv_capacity: usize,
    ) {
        if rms_norm_eps > 0.0 {
            self.rms_norm_eps = rms_norm_eps;
        }
        self.is_qwen = is_qwen;
        self.sliding_window = sliding_window;
        if kv_capacity > 0 {
            self.kv_capacity = kv_capacity;
        }
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
                driver::launch_cuda_gemv_q5_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
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
        let max_logits_batch = 8usize;
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
        self.d_logits = Some(
            CudaDeviceBuffer::allocate(max_logits_batch * vocab_size * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_sampled_token = Some(
            CudaDeviceBuffer::allocate(max_logits_batch * 8).map_err(EngineError::BackendError)?,
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
        self.d_scratch_q8_1_a =
            Some(CudaDeviceBuffer::allocate(q8_1_bytes).map_err(EngineError::BackendError)?);
        self.d_scratch_q8_1_b =
            Some(CudaDeviceBuffer::allocate(q8_1_bytes).map_err(EngineError::BackendError)?);

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
            let mut d_emb = CudaDeviceBuffer::allocate(token_embd_bytes.len())
                .map_err(EngineError::BackendError)?;
            d_emb
                .copy_from_host_async(token_embd_bytes, s)
                .map_err(EngineError::BackendError)?;
            self.d_token_embd = Some(d_emb);
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

        let max_draft = 5usize;
        let num_draft = draft_tokens.len().min(max_draft);
        let mut candidate_tokens = Vec::with_capacity(1 + num_draft);
        candidate_tokens.push(cur_token);
        candidate_tokens.extend_from_slice(&draft_tokens[..num_draft]);

        let batch_size = candidate_tokens.len();
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
        for (i, &tok) in candidate_tokens.iter().enumerate() {
            let tok_idx = (tok as usize) % self.vocab_size.max(1);
            if let Some(lookup) = &self.embedding_lookup {
                if self.host_emb_buffer.len() != h {
                    self.host_emb_buffer.resize(h, 0.0f32);
                }
                lookup(tok_idx, &mut self.host_emb_buffer);
                if let Some(d_hid) = &mut self.d_hidden {
                    unsafe {
                        let dst_ptr =
                            (d_hid.as_raw_ptr() as *mut u8).add(i * h * std::mem::size_of::<f32>());
                        let _ = driver::cudaMemcpyAsync(
                            dst_ptr as *mut std::ffi::c_void,
                            self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                            h * std::mem::size_of::<f32>(),
                            driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                            s,
                        );
                    }
                }
            } else if let Some(d_emb) = &self.d_token_embd {
                let src_offset = tok_idx * h * std::mem::size_of::<f32>();
                let dst_offset = i * h * std::mem::size_of::<f32>();
                if src_offset + h * 4 <= d_emb.size_bytes() {
                    if let Some(d_hid) = &self.d_hidden {
                        unsafe {
                            let dst_ptr = (d_hid.as_raw_ptr() as *mut u8).add(dst_offset);
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

        let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
        let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };

        // 2. Multi-layer forward pass on physical GPU with Multi-Column MMVQ
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
                    let q_dim = cur_num_heads * cur_head_dim;

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

                    if let (Some(d_q), Some(d_q_norm)) =
                        (&mut self.d_scratch_q, &layer.d_attn_q_norm)
                    {
                        if d_q_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                            for b in 0..batch_size {
                                unsafe {
                                    driver::launch_cuda_qk_norm(
                                        d_q.as_typed_ptr::<f32>().add(b * q_dim),
                                        d_q_norm.as_typed_ptr::<f32>(),
                                        cur_num_heads as i32,
                                        cur_head_dim as i32,
                                        1e-6,
                                        s,
                                    );
                                }
                            }
                        }
                    }
                    if let (Some(d_k), Some(d_k_norm)) =
                        (&mut self.d_scratch_k, &layer.d_attn_k_norm)
                    {
                        if d_k_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                            for b in 0..batch_size {
                                unsafe {
                                    driver::launch_cuda_qk_norm(
                                        d_k.as_typed_ptr::<f32>().add(b * kv_dim),
                                        d_k_norm.as_typed_ptr::<f32>(),
                                        cur_num_kv_heads as i32,
                                        cur_head_dim as i32,
                                        1e-6,
                                        s,
                                    );
                                }
                            }
                        }
                    }

                    let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                        10_000.0f32
                    } else {
                        self.rope_theta
                    };

                    for b in 0..batch_size {
                        let token_pos = pos + b;
                        let k_dst_ptr = if layer_idx < self.kv_k.len() {
                            unsafe {
                                (self.kv_k[layer_idx].as_raw_ptr() as *mut u8)
                                    .add(token_pos * kv_dim * 2)
                                    as *mut std::ffi::c_void
                            }
                        } else {
                            std::ptr::null_mut()
                        };
                        let v_dst_ptr = if layer_idx < self.kv_v.len() {
                            unsafe {
                                (self.kv_v[layer_idx].as_raw_ptr() as *mut u8)
                                    .add(token_pos * kv_dim * 2)
                                    as *mut std::ffi::c_void
                            }
                        } else {
                            std::ptr::null_mut()
                        };

                        if let (Some(d_q), Some(d_k)) =
                            (&mut self.d_scratch_q, &mut self.d_scratch_k)
                        {
                            unsafe {
                                driver::launch_cuda_rope_single(
                                    d_q.as_typed_ptr::<f32>().add(b * q_dim),
                                    d_k.as_typed_ptr::<f32>().add(b * kv_dim),
                                    k_dst_ptr,
                                    token_pos as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    layer_rope_theta,
                                    1, // is_fp16
                                    self.rope_scaling_factor,
                                    self.rope_scaling_orig_ctx as i32,
                                    s,
                                );
                            }
                        }

                        if let Some(d_v) = &self.d_scratch_v {
                            if !v_dst_ptr.is_null() {
                                unsafe {
                                    driver::launch_cuda_convert_f32_to_f16(
                                        v_dst_ptr,
                                        d_v.as_typed_ptr::<f32>().add(b * kv_dim),
                                        kv_dim as i32,
                                        s,
                                    );
                                }
                            }
                        }

                        if let (Some(d_q), Some(d_out)) =
                            (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                        {
                            if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                                let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                                let num_tokens = (token_pos + 1).min(kv_cap) as i32;
                                unsafe {
                                    driver::launch_cuda_flash_decode(
                                        d_out.as_typed_ptr::<f32>().add(b * q_dim),
                                        d_q.as_typed_ptr::<f32>().add(b * q_dim),
                                        self.kv_k[layer_idx].as_raw_ptr(),
                                        self.kv_v[layer_idx].as_raw_ptr(),
                                        cur_num_heads as i32,
                                        cur_num_kv_heads as i32,
                                        cur_head_dim as i32,
                                        num_tokens,
                                        0,
                                        sm_scale,
                                        1, // is_fp16
                                        s,
                                    );
                                }
                            }
                        }
                    }

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
                            for b in 0..batch_size {
                                let out_ptr = unsafe { d_out.as_typed_ptr::<f32>().add(b * o_k) };
                                let q8_ptr = unsafe {
                                    (d_q8_1_ptr as *mut u8).add(b * (o_k / 32) * 36)
                                        as *mut std::ffi::c_void
                                };
                                unsafe {
                                    driver::launch_cuda_quantize_q8_1(
                                        out_ptr, q8_ptr, o_k as i32, s,
                                    );
                                }
                            }
                        }
                        unsafe {
                            Self::dispatch_mmvq(
                                d_act.as_typed_ptr::<f32>(),
                                &layer.d_wo,
                                d_out.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                o_m,
                                o_k,
                                layer.o_quant,
                                batch_size,
                                s,
                            );
                        }
                    }

                    if let Some(d_act) = &self.d_scratch_act {
                        for b in 0..batch_size {
                            unsafe {
                                driver::launch_cuda_vector_add(
                                    d_hid.as_typed_ptr::<f32>().add(b * h),
                                    d_act.as_typed_ptr::<f32>().add(b * h),
                                    h as i32,
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
                                batch_size as i32,
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
                            Self::dispatch_mmvq(
                                d_gate.as_typed_ptr::<f32>(),
                                &layer.d_w1,
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                layer.inter_m,
                                h,
                                layer.gate_quant,
                                batch_size,
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
                                batch_size,
                                s,
                            );
                        }

                        if !d_q8_1_ptr.is_null() {
                            unsafe {
                                driver::launch_cuda_fused_swiglu_q8_1(
                                    d_q8_1_ptr,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    (batch_size * layer.inter_m) as i32,
                                    s,
                                );
                            }
                        }
                    }

                    if let (Some(d_gate), Some(d_act)) =
                        (&self.d_scratch_gate, &mut self.d_scratch_act)
                    {
                        unsafe {
                            Self::dispatch_mmvq(
                                d_act.as_typed_ptr::<f32>(),
                                &layer.d_w2,
                                d_gate.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                h,
                                layer.inter_m,
                                layer.down_quant,
                                batch_size,
                                s,
                            );
                        }
                    }

                    if let Some(d_act) = &self.d_scratch_act {
                        for b in 0..batch_size {
                            unsafe {
                                driver::launch_cuda_vector_add(
                                    d_hid.as_typed_ptr::<f32>().add(b * h),
                                    d_act.as_typed_ptr::<f32>().add(b * h),
                                    h as i32,
                                    s,
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
                        return Ok(accepted);
                    }
                }

                let bonus_token = sampled_res[num_draft];
                accepted.push(bonus_token);
                cur_seq_pos += 1;
                self.seq_pos = cur_seq_pos;
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

    /// High-throughput chunked prefill pass on physical GPU evaluating up to 4 prompt tokens simultaneously.
    /// Streams layer weights ONCE for all tokens in the chunk and skips the 510MB LM Head on intermediate chunks.
    pub fn dispatch_prefill_chunk(
        &mut self,
        chunk_tokens: &[u32],
        is_final: bool,
    ) -> Result<Option<u32>> {
        if chunk_tokens.is_empty() {
            return Ok(None);
        }

        let stream = self.stream.as_ref().ok_or_else(|| {
            EngineError::BackendError("Physical CUDA stream not available on host dGPU".to_string())
        })?;
        let s = stream.raw();

        let batch_size = chunk_tokens.len();
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
        for (i, &tok) in chunk_tokens.iter().enumerate() {
            let tok_idx = (tok as usize) % self.vocab_size.max(1);
            if let Some(lookup) = &self.embedding_lookup {
                if self.host_emb_buffer.len() != h {
                    self.host_emb_buffer.resize(h, 0.0f32);
                }
                lookup(tok_idx, &mut self.host_emb_buffer);
                if let Some(d_hid) = &mut self.d_hidden {
                    unsafe {
                        let dst_ptr =
                            (d_hid.as_raw_ptr() as *mut u8).add(i * h * std::mem::size_of::<f32>());
                        let _ = driver::cudaMemcpyAsync(
                            dst_ptr as *mut std::ffi::c_void,
                            self.host_emb_buffer.as_ptr() as *const std::ffi::c_void,
                            h * std::mem::size_of::<f32>(),
                            driver::CUDA_MEMCPY_HOST_TO_DEVICE,
                            s,
                        );
                    }
                }
            } else if let Some(d_emb) = &self.d_token_embd {
                let src_offset = tok_idx * h * std::mem::size_of::<f32>();
                let dst_offset = i * h * std::mem::size_of::<f32>();
                if src_offset + h * 4 <= d_emb.size_bytes() {
                    if let Some(d_hid) = &self.d_hidden {
                        unsafe {
                            let dst_ptr = (d_hid.as_raw_ptr() as *mut u8).add(dst_offset);
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

        let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };
        let d_q8_1_ptr = if let Some(ref buf) = self.d_scratch_q8_1_a {
            buf.as_raw_ptr()
        } else {
            std::ptr::null_mut()
        };

        // 2. Multi-layer forward pass on physical GPU with Multi-Column MMVQ
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

                    if let (Some(d_q), Some(d_q_norm)) =
                        (&mut self.d_scratch_q, &layer.d_attn_q_norm)
                    {
                        if d_q_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                            unsafe {
                                driver::launch_cuda_qk_norm(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_q_norm.as_typed_ptr::<f32>(),
                                    (cur_num_heads * batch_size) as i32,
                                    cur_head_dim as i32,
                                    1e-6,
                                    s,
                                );
                            }
                        }
                    }
                    if let (Some(d_k), Some(d_k_norm)) =
                        (&mut self.d_scratch_k, &layer.d_attn_k_norm)
                    {
                        if d_k_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                            unsafe {
                                driver::launch_cuda_qk_norm(
                                    d_k.as_typed_ptr::<f32>(),
                                    d_k_norm.as_typed_ptr::<f32>(),
                                    (cur_num_kv_heads * batch_size) as i32,
                                    cur_head_dim as i32,
                                    1e-6,
                                    s,
                                );
                            }
                        }
                    }

                    let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                        10_000.0f32
                    } else {
                        self.rope_theta
                    };

                    let k_cache_ptr = if layer_idx < self.kv_k.len() {
                        self.kv_k[layer_idx].as_raw_ptr()
                    } else {
                        std::ptr::null_mut()
                    };

                    if let (Some(d_q), Some(d_k)) = (&mut self.d_scratch_q, &mut self.d_scratch_k) {
                        unsafe {
                            driver::launch_cuda_rope_batch(
                                d_q.as_typed_ptr::<f32>(),
                                d_k.as_typed_ptr::<f32>(),
                                k_cache_ptr,
                                pos as i32,
                                batch_size as i32,
                                cur_num_heads as i32,
                                cur_num_kv_heads as i32,
                                cur_head_dim as i32,
                                layer_rope_theta,
                                1, // is_fp16
                                self.rope_scaling_factor,
                                self.rope_scaling_orig_ctx as i32,
                                s,
                            );
                        }
                    }

                    if let Some(d_v) = &self.d_scratch_v {
                        if layer_idx < self.kv_v.len() {
                            unsafe {
                                let v_dst_ptr = (self.kv_v[layer_idx].as_raw_ptr() as *mut u8)
                                    .add(pos * kv_dim * 2)
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

                    if let (Some(d_q), Some(d_out)) =
                        (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                    {
                        if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                            let sm_scale = 1.0 / (cur_head_dim.max(1) as f32).sqrt();
                            unsafe {
                                driver::launch_cuda_flash_causal_prefill(
                                    d_out.as_typed_ptr::<f32>(),
                                    d_q.as_typed_ptr::<f32>(),
                                    self.kv_k[layer_idx].as_raw_ptr(),
                                    self.kv_v[layer_idx].as_raw_ptr(),
                                    pos as i32,
                                    batch_size as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    kv_cap as i32,
                                    sm_scale,
                                    1, // is_fp16
                                    s,
                                );
                            }
                        }
                    }

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
                                    (batch_size * o_k) as i32,
                                    s,
                                );
                            }
                        }
                        unsafe {
                            Self::dispatch_mmvq(
                                d_act.as_typed_ptr::<f32>(),
                                &layer.d_wo,
                                d_out.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                o_m,
                                o_k,
                                layer.o_quant,
                                batch_size,
                                s,
                            );
                        }
                    }

                    if let Some(d_act) = &self.d_scratch_act {
                        unsafe {
                            driver::launch_cuda_vector_add(
                                d_hid.as_typed_ptr::<f32>(),
                                d_act.as_typed_ptr::<f32>(),
                                (batch_size * h) as i32,
                                s,
                            );
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
                                batch_size as i32,
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
                            Self::dispatch_mmvq(
                                d_gate.as_typed_ptr::<f32>(),
                                &layer.d_w1,
                                d_norm.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                layer.inter_m,
                                h,
                                layer.gate_quant,
                                batch_size,
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
                                batch_size,
                                s,
                            );
                        }

                        if !d_q8_1_ptr.is_null() {
                            unsafe {
                                driver::launch_cuda_fused_swiglu_q8_1(
                                    d_q8_1_ptr,
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_gate.as_typed_ptr::<f32>(),
                                    d_up.as_typed_ptr::<f32>(),
                                    (batch_size * layer.inter_m) as i32,
                                    s,
                                );
                            }
                        }
                    }

                    if let (Some(d_gate), Some(d_act)) =
                        (&self.d_scratch_gate, &mut self.d_scratch_act)
                    {
                        unsafe {
                            Self::dispatch_mmvq(
                                d_act.as_typed_ptr::<f32>(),
                                &layer.d_w2,
                                d_gate.as_typed_ptr::<f32>(),
                                d_q8_1_ptr,
                                h,
                                layer.inter_m,
                                layer.down_quant,
                                batch_size,
                                s,
                            );
                        }
                    }

                    if let Some(d_act) = &self.d_scratch_act {
                        unsafe {
                            driver::launch_cuda_vector_add(
                                d_hid.as_typed_ptr::<f32>(),
                                d_act.as_typed_ptr::<f32>(),
                                (batch_size * h) as i32,
                                s,
                            );
                        }
                    }
                }
            }
        }

        self.seq_pos += batch_size;

        if !is_final {
            return Ok(None);
        }

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

            Ok(Some(selected_token))
        } else {
            Err(EngineError::BackendError(
                "LM head buffers not initialized".to_string(),
            ))
        }
    }
}

impl HardwareBackend for CudaBackend {
    type Event = CudaEventHandle;

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

        if !self.layers.is_empty() && self.hidden_dim > 0 {
            let h = self.hidden_dim;
            let pos = self.seq_pos;
            self.seq_pos += 1;

            // 1. Initial embedding lookup on GPU (or dequantized from host)
            if let Some(lookup) = &self.embedding_lookup {
                let tok_idx = (cmd.input_token as usize) % self.vocab_size.max(1);
                if self.host_emb_buffer.len() != h {
                    self.host_emb_buffer.resize(h, 0.0f32);
                }
                lookup(tok_idx, &mut self.host_emb_buffer);
                if let Some(d_hid) = &mut self.d_hidden {
                    let _ = d_hid.copy_from_host_async(&self.host_emb_buffer, s);
                }
            } else if let Some(d_emb) = &self.d_token_embd {
                let tok_idx = (cmd.input_token as usize) % self.vocab_size.max(1);
                let offset = tok_idx * h * std::mem::size_of::<f32>();
                if offset + h * 4 <= d_emb.size_bytes() {
                    if let Some(d_hid) = &self.d_hidden {
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
                        let (k_dst_ptr, v_dst_ptr) =
                            if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                                unsafe {
                                    (
                                        (self.kv_k[layer_idx].as_raw_ptr() as *mut u8)
                                            .add(pos * kv_dim * 2)
                                            as *mut std::ffi::c_void,
                                        (self.kv_v[layer_idx].as_raw_ptr() as *mut u8)
                                            .add(pos * kv_dim * 2)
                                            as *mut std::ffi::c_void,
                                    )
                                }
                            } else {
                                (std::ptr::null_mut(), std::ptr::null_mut())
                            };

                        // Q Projection
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

                        // K Projection
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

                        // V Projection: computes GEMV into scratch buffer, then converts to FP16 in KV cache
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
                                if !v_dst_ptr.is_null() {
                                    driver::launch_cuda_convert_f32_to_f16(
                                        v_dst_ptr,
                                        d_v.as_typed_ptr::<f32>(),
                                        kv_dim as i32,
                                        s,
                                    );
                                }
                            }
                            check_cuda("v_proj", layer_idx)?;
                        }

                        // QK-Norm (applied per-head before RoPE if present)
                        if let (Some(d_q), Some(d_q_norm)) =
                            (&mut self.d_scratch_q, &layer.d_attn_q_norm)
                        {
                            if d_q_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                                unsafe {
                                    driver::launch_cuda_qk_norm(
                                        d_q.as_typed_ptr::<f32>(),
                                        d_q_norm.as_typed_ptr::<f32>(),
                                        cur_num_heads as i32,
                                        cur_head_dim as i32,
                                        1e-6,
                                        s,
                                    );
                                }
                                check_cuda("q_norm", layer_idx)?;
                            }
                        }
                        if let (Some(d_k), Some(d_k_norm)) =
                            (&mut self.d_scratch_k, &layer.d_attn_k_norm)
                        {
                            if d_k_norm.size_bytes() >= cur_head_dim * std::mem::size_of::<f32>() {
                                unsafe {
                                    driver::launch_cuda_qk_norm(
                                        d_k.as_typed_ptr::<f32>(),
                                        d_k_norm.as_typed_ptr::<f32>(),
                                        cur_num_kv_heads as i32,
                                        cur_head_dim as i32,
                                        1e-6,
                                        s,
                                    );
                                }
                                check_cuda("k_norm", layer_idx)?;
                            }
                        }

                        // Rotary Position Embedding (RoPE) on Q and K: directly emits rotated K to FP16 KV cache
                        let layer_rope_theta = if self.is_gemma && cur_head_dim <= 256 {
                            10_000.0f32
                        } else {
                            self.rope_theta
                        };
                        if let (Some(d_q), Some(d_k)) =
                            (&mut self.d_scratch_q, &mut self.d_scratch_k)
                        {
                            unsafe {
                                driver::launch_cuda_rope_single(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_k.as_typed_ptr::<f32>(),
                                    k_dst_ptr,
                                    pos as i32,
                                    cur_num_heads as i32,
                                    cur_num_kv_heads as i32,
                                    cur_head_dim as i32,
                                    layer_rope_theta,
                                    1, // is_fp16
                                    self.rope_scaling_factor,
                                    self.rope_scaling_orig_ctx as i32,
                                    s,
                                );
                            }
                            check_cuda("rope", layer_idx)?;
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
                                    driver::launch_cuda_fused_swiglu_q8_1(
                                        d_q8_1_ptr,
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_gate.as_typed_ptr::<f32>(),
                                        d_up.as_typed_ptr::<f32>(),
                                        layer.inter_m as i32,
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
