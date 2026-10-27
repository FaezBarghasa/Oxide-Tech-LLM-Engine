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
    pub q_quant: u32,
    pub k_quant: u32,
    pub v_quant: u32,
    pub o_quant: u32,
    pub gate_quant: u32,
    pub up_quant: u32,
    pub down_quant: u32,
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

pub struct CudaBackend {
    device_id: usize,
    current_event_id: u64,
    host_token_buffer: Vec<u32>,
    profile: GpuDeviceProfile,
    execution_plan: KernelExecutionPlan,
    stream: Option<CudaStream>,
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
    pub kv_k: Vec<CudaDeviceBuffer>,
    pub kv_v: Vec<CudaDeviceBuffer>,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub seq_pos: usize,
    pub embedding_lookup: Option<Box<dyn Fn(usize, &mut [f32]) + Send + Sync>>,
    pub mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
    pub lm_head_quant_type: u32,
    pub rope_theta: f32,
    pub is_gemma: bool,
    pub final_logit_softcapping: Option<f32>,
    pub sampler: oxide_core::sampler::AcademicSamplerEngine,
    pub sampler_state: oxide_core::sampler::SamplerState,
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
            is_gemma: false,
            final_logit_softcapping: None,
            sampler: {
                let mut sampling_config = oxide_core::sampler::SamplingConfig::default();
                sampling_config.temperature = 0.7;
                sampling_config.top_p = 0.95;
                sampling_config.min_p = 0.05;
                sampling_config.dry_multiplier = 0.8;
                oxide_core::sampler::AcademicSamplerEngine::new(sampling_config)
            },
            sampler_state: {
                let mut sampling_config = oxide_core::sampler::SamplingConfig::default();
                sampling_config.temperature = 0.7;
                sampling_config.top_p = 0.95;
                sampling_config.min_p = 0.05;
                sampling_config.dry_multiplier = 0.8;
                oxide_core::sampler::SamplerState::from_config(&sampling_config)
            },
        }
    }

    #[inline(always)]
    unsafe fn dispatch_gemv(
        d_y: *mut f32,
        d_w: &CudaDeviceBuffer,
        d_x: *const f32,
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
                    d_x,
                    m as i32,
                    k as i32,
                    stream,
                );
            } else if quant_type == 2 {
                driver::launch_cuda_gemv_q6_k(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
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
            } else {
                driver::launch_cuda_gemv_q8_0(
                    d_y,
                    d_w.as_raw_ptr(),
                    d_x,
                    m as i32,
                    k as i32,
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

        let mut d_hidden = CudaDeviceBuffer::allocate(hidden_dim * std::mem::size_of::<f32>())
            .map_err(EngineError::BackendError)?;
        d_hidden
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_hidden = Some(d_hidden);

        let mut d_scratch_norm =
            CudaDeviceBuffer::allocate(hidden_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?;
        d_scratch_norm
            .memset_async(0, s)
            .map_err(EngineError::BackendError)?;
        self.d_scratch_norm = Some(d_scratch_norm);

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let max_dim = q_dim.max(kv_dim).max(hidden_dim);

        self.d_scratch_q = Some(
            CudaDeviceBuffer::allocate(q_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_k = Some(
            CudaDeviceBuffer::allocate(kv_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_v = Some(
            CudaDeviceBuffer::allocate(kv_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_attn_out = Some(
            CudaDeviceBuffer::allocate(q_dim * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_act = Some(
            CudaDeviceBuffer::allocate(max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_gate = Some(
            CudaDeviceBuffer::allocate(max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_scratch_up = Some(
            CudaDeviceBuffer::allocate(max_dim * 4 * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );
        self.d_logits = Some(
            CudaDeviceBuffer::allocate(vocab_size * std::mem::size_of::<f32>())
                .map_err(EngineError::BackendError)?,
        );

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
        quant_types: [u32; 7],
    ) -> Result<()> {
        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| EngineError::BackendError("No active CUDA stream".to_string()))?;
        let s = stream.raw();

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

        let d_attn_q_norm = if !attn_q_norm.is_empty() {
            let mut buf =
                CudaDeviceBuffer::allocate(attn_q_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(attn_q_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        } else {
            None
        };

        let d_attn_k_norm = if !attn_k_norm.is_empty() {
            let mut buf =
                CudaDeviceBuffer::allocate(attn_k_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(attn_k_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        } else {
            None
        };

        let d_post_attn_norm = if !post_attn_norm.is_empty() {
            let mut buf =
                CudaDeviceBuffer::allocate(post_attn_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(post_attn_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        } else {
            None
        };

        let d_post_ffn_norm = if !post_ffn_norm.is_empty() {
            let mut buf =
                CudaDeviceBuffer::allocate(post_ffn_norm.len() * std::mem::size_of::<f32>())
                    .map_err(EngineError::BackendError)?;
            buf.copy_from_host_async(post_ffn_norm, s)
                .map_err(EngineError::BackendError)?;
            Some(buf)
        } else {
            None
        };

        let kv_bytes =
            2048 * self.num_kv_heads.max(1) * self.head_dim.max(64) * std::mem::size_of::<f32>();
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
            q_quant: quant_types[0],
            k_quant: quant_types[1],
            v_quant: quant_types[2],
            o_quant: quant_types[3],
            gate_quant: quant_types[4],
            up_quant: quant_types[5],
            down_quant: quant_types[6],
        });

        stream.synchronize().map_err(EngineError::BackendError)?;
        Ok(())
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
                let mut host_emb = vec![0.0f32; h];
                lookup(tok_idx, &mut host_emb);
                tracing::info!(
                    "CUDA pos {}: tok_idx={}, host_emb[0..5]={:?}",
                    pos,
                    tok_idx,
                    &host_emb[0..5.min(host_emb.len())]
                );
                if let Some(d_hid) = &mut self.d_hidden {
                    let _ = d_hid.copy_from_host_async(&host_emb, s);
                    let _ = stream.synchronize();
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

            let weight_offset = if self.is_gemma { 1.0f32 } else { 0.0f32 };

            // 2. Multi-layer forward pass on physical GPU
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if let (Some(d_hid), Some(d_norm)) = (&self.d_hidden, &mut self.d_scratch_norm) {
                    // Attention Block
                    let has_attn = layer.d_wq.size_bytes() > 0
                        && layer.d_wo.size_bytes() > 0
                        && layer.q_m > 0;
                    if has_attn {
                        // Attention RMSNorm
                        if layer.d_attn_norm.size_bytes() > 0 {
                            unsafe {
                                driver::launch_cuda_rmsnorm(
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_attn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    1e-5,
                                    weight_offset,
                                    s,
                                );
                            }
                        }

                        // Q Projection
                        if let Some(d_q) = &mut self.d_scratch_q {
                            unsafe {
                                Self::dispatch_gemv(
                                    d_q.as_typed_ptr::<f32>(),
                                    &layer.d_wq,
                                    d_norm.as_typed_ptr::<f32>(),
                                    layer.q_m,
                                    layer.q_k,
                                    layer.q_quant,
                                    s,
                                );
                            }
                        }

                        // K Projection
                        if let Some(d_k) = &mut self.d_scratch_k {
                            unsafe {
                                Self::dispatch_gemv(
                                    d_k.as_typed_ptr::<f32>(),
                                    &layer.d_wk,
                                    d_norm.as_typed_ptr::<f32>(),
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.k_quant,
                                    s,
                                );
                            }
                        }

                        // V Projection
                        if let Some(d_v) = &mut self.d_scratch_v {
                            unsafe {
                                Self::dispatch_gemv(
                                    d_v.as_typed_ptr::<f32>(),
                                    &layer.d_wv,
                                    d_norm.as_typed_ptr::<f32>(),
                                    layer.kv_m,
                                    layer.kv_k,
                                    layer.v_quant,
                                    s,
                                );
                            }
                        }

                        // QK-Norm (applied per-head before RoPE if present)
                        if let (Some(d_q), Some(d_q_norm)) =
                            (&mut self.d_scratch_q, &layer.d_attn_q_norm)
                        {
                            unsafe {
                                driver::launch_cuda_qk_norm(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_q_norm.as_typed_ptr::<f32>(),
                                    self.num_heads as i32,
                                    self.head_dim as i32,
                                    1e-6,
                                    s,
                                );
                            }
                        }
                        if let (Some(d_k), Some(d_k_norm)) =
                            (&mut self.d_scratch_k, &layer.d_attn_k_norm)
                        {
                            unsafe {
                                driver::launch_cuda_qk_norm(
                                    d_k.as_typed_ptr::<f32>(),
                                    d_k_norm.as_typed_ptr::<f32>(),
                                    self.num_kv_heads as i32,
                                    self.head_dim as i32,
                                    1e-6,
                                    s,
                                );
                            }
                        }

                        // Rotary Position Embedding (RoPE) on Q and K
                        if let (Some(d_q), Some(d_k)) = (&mut self.d_scratch_q, &mut self.d_scratch_k) {
                            unsafe {
                                driver::launch_cuda_rope_single(
                                    d_q.as_typed_ptr::<f32>(),
                                    d_k.as_typed_ptr::<f32>(),
                                    pos as i32,
                                    self.num_heads as i32,
                                    self.num_kv_heads as i32,
                                    self.head_dim as i32,
                                    self.rope_theta,
                                    s,
                                );
                            }
                        }

                        // KV Cache update: copy current token's K and V into KV cache ring buffer at `pos`
                        let kv_dim = self.num_kv_heads * self.head_dim;
                        let kv_bytes = kv_dim * std::mem::size_of::<f32>();
                        let kv_offset = (pos % 2048) * kv_bytes;
                        if let (Some(d_k), Some(d_v)) = (&self.d_scratch_k, &self.d_scratch_v) {
                            if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                                let k_dst = unsafe {
                                    (self.kv_k[layer_idx].as_raw_ptr() as *mut u8).add(kv_offset)
                                        as *mut std::ffi::c_void
                                };
                                let v_dst = unsafe {
                                    (self.kv_v[layer_idx].as_raw_ptr() as *mut u8).add(kv_offset)
                                        as *mut std::ffi::c_void
                                };
                                let _ = unsafe {
                                    driver::cudaMemcpyAsync(
                                        k_dst,
                                        d_k.as_raw_ptr(),
                                        kv_bytes,
                                        driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                        s,
                                    )
                                };
                                let _ = unsafe {
                                    driver::cudaMemcpyAsync(
                                        v_dst,
                                        d_v.as_raw_ptr(),
                                        kv_bytes,
                                        driver::CUDA_MEMCPY_DEVICE_TO_DEVICE,
                                        s,
                                    )
                                };
                            }
                        }

                        // Attention Flash Decode (with GQA)
                        if let (Some(d_q), Some(d_out)) =
                            (&self.d_scratch_q, &mut self.d_scratch_attn_out)
                        {
                            if layer_idx < self.kv_k.len() && layer_idx < self.kv_v.len() {
                                let sm_scale = 1.0 / (self.head_dim.max(1) as f32).sqrt();
                                unsafe {
                                    driver::launch_cuda_flash_decode(
                                        d_out.as_typed_ptr::<f32>(),
                                        d_q.as_typed_ptr::<f32>(),
                                        self.kv_k[layer_idx].as_typed_ptr::<f32>(),
                                        self.kv_v[layer_idx].as_typed_ptr::<f32>(),
                                        self.num_heads as i32,
                                        self.num_kv_heads as i32,
                                        self.head_dim as i32,
                                        (pos + 1).min(2048) as i32,
                                        sm_scale,
                                        s,
                                    );
                                }
                            }
                        }

                        // Output Projection: wo x attn_out -> d_act
                        if let (Some(d_out), Some(d_act)) =
                            (&self.d_scratch_attn_out, &mut self.d_scratch_act)
                        {
                            unsafe {
                                Self::dispatch_gemv(
                                    d_act.as_typed_ptr::<f32>(),
                                    &layer.d_wo,
                                    d_out.as_typed_ptr::<f32>(),
                                    h,
                                    layer.q_m,
                                    layer.o_quant,
                                    s,
                                );
                            }
                        }

                        // Post-attention norm (Gemma 2 / Gemma 4 / Nemotron)
                        if let (Some(d_post_attn), Some(d_act)) =
                            (&layer.d_post_attn_norm, &mut self.d_scratch_act)
                        {
                            if d_post_attn.size_bytes() > 0 {
                                unsafe {
                                    driver::launch_cuda_rmsnorm(
                                        d_act.as_typed_ptr::<f32>(),
                                        d_act.as_typed_ptr::<f32>(),
                                        d_post_attn.as_typed_ptr::<f32>(),
                                        1,
                                        h as i32,
                                        1e-5,
                                        weight_offset,
                                        s,
                                    );
                                }
                            }
                        }

                        // Residual 1: hidden += attn_out
                        if let Some(d_act) = &self.d_scratch_act {
                            unsafe {
                                driver::launch_cuda_vector_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    d_act.as_typed_ptr::<f32>(),
                                    h as i32,
                                    s,
                                );
                            }
                        }
                    }

                    // FFN Block
                    let has_ffn = layer.d_w1.size_bytes() > 0
                        && layer.d_w2.size_bytes() > 0
                        && layer.inter_m > 0;
                    if has_ffn {
                        // FFN RMSNorm
                        if layer.d_ffn_norm.size_bytes() > 0 {
                            unsafe {
                                driver::launch_cuda_rmsnorm(
                                    d_norm.as_typed_ptr::<f32>(),
                                    d_hid.as_typed_ptr::<f32>(),
                                    layer.d_ffn_norm.as_typed_ptr::<f32>(),
                                    1,
                                    h as i32,
                                    1e-5,
                                    weight_offset,
                                    s,
                                );
                            }
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
                                    layer.inter_m,
                                    h,
                                    layer.gate_quant,
                                    s,
                                );
                                Self::dispatch_gemv(
                                    d_up.as_typed_ptr::<f32>(),
                                    &layer.d_w3,
                                    d_norm.as_typed_ptr::<f32>(),
                                    layer.inter_m,
                                    h,
                                    layer.up_quant,
                                    s,
                                );
                                if self.is_gemma {
                                    driver::launch_cuda_gelu(
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
                            }
                        }

                        // FFN Down Projection: w2 x gate -> d_act
                        if let (Some(d_gate), Some(d_act)) =
                            (&self.d_scratch_gate, &mut self.d_scratch_act)
                        {
                            unsafe {
                                Self::dispatch_gemv(
                                    d_act.as_typed_ptr::<f32>(),
                                    &layer.d_w2,
                                    d_gate.as_typed_ptr::<f32>(),
                                    h,
                                    layer.inter_m,
                                    layer.down_quant,
                                    s,
                                );
                            }
                        }

                        // Post-FFN norm (Gemma 2 / Gemma 4 / Nemotron)
                        if let (Some(d_post_ffn), Some(d_act)) =
                            (&layer.d_post_ffn_norm, &mut self.d_scratch_act)
                        {
                            if d_post_ffn.size_bytes() > 0 {
                                unsafe {
                                    driver::launch_cuda_rmsnorm(
                                        d_act.as_typed_ptr::<f32>(),
                                        d_act.as_typed_ptr::<f32>(),
                                        d_post_ffn.as_typed_ptr::<f32>(),
                                        1,
                                        h as i32,
                                        1e-5,
                                        weight_offset,
                                        s,
                                    );
                                }
                            }
                        }

                        // Residual 2: hidden += ffn_out
                        if let Some(d_act) = &self.d_scratch_act {
                            unsafe {
                                driver::launch_cuda_vector_add(
                                    d_hid.as_typed_ptr::<f32>(),
                                    d_act.as_typed_ptr::<f32>(),
                                    h as i32,
                                    s,
                                );
                            }
                        }
                    }

                    // Residual 2: hidden += ffn_out
                    if let Some(d_act) = &self.d_scratch_act {
                        unsafe {
                            driver::launch_cuda_vector_add(
                                d_hid.as_typed_ptr::<f32>(),
                                d_act.as_typed_ptr::<f32>(),
                                h as i32,
                                s,
                            );
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
                        driver::launch_cuda_rmsnorm(
                            d_norm.as_typed_ptr::<f32>(),
                            d_hid.as_typed_ptr::<f32>(),
                            d_out_norm.as_typed_ptr::<f32>(),
                            1,
                            h as i32,
                            1e-5,
                            weight_offset,
                            s,
                        );
                    }
                }
            }

            // 4. LM Head Logit Projection (full vocabulary)
            if let (Some(d_norm), Some(d_lm), Some(d_logits)) =
                (&self.d_scratch_norm, &self.d_lm_head, &mut self.d_logits)
            {
                let mut norm_check = vec![0.0f32; 5];
                let _ = d_norm.copy_to_host_async(&mut norm_check, s);
                let _ = stream.synchronize();
                tracing::info!("CUDA pos {}: norm_before_lmhead[0..5]={:?}", pos, norm_check);

                unsafe {
                    Self::dispatch_gemv(
                        d_logits.as_typed_ptr::<f32>(),
                        d_lm,
                        d_norm.as_typed_ptr::<f32>(),
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

                let mut host_logits = vec![0.0f32; self.vocab_size];
                let _ = d_logits.copy_to_host_async(&mut host_logits, s);
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
                let sampled_tok = self
                    .sampler
                    .sample_token(&mut host_logits, &mut self.sampler_state, 198)
                    .unwrap_or(best_tok);

                let selected_token = sampled_tok;

                if self.seq_pos <= 5 || self.seq_pos % 25 == 0 {
                    tracing::info!(
                        "CUDA pos {}: greedy_top={}, sampled_tok={}, max_val={}, logits[0..5]={:?}",
                        pos,
                        best_tok,
                        selected_token,
                        max_val,
                        &host_logits[0..5.min(host_logits.len())]
                    );
                }
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
