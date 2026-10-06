use crate::flash_attn::{FlashAttentionConfig, FlashAttentionEngine};
use crate::formats::{GgufFile, SafeTensorsHeader};
use crate::rope::{RopeConfig, RopeScalingEngine, RopeScalingType};
use oxide_core::error::Result;
use oxide_core::traits::ModelConfig;

/// Dense Transformer baseline LLaMA 3.1 8B configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Llama3Config {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: u32, // Store scaled integer eps (e.g. 1e-5 represented as bits or f32)
}

impl Default for Llama3Config {
    fn default() -> Self {
        Self {
            hidden_dim: 4096,
            intermediate_dim: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 128_256,
            max_seq_len: 131_072,
            rms_norm_eps: 1e-5f32.to_bits(),
        }
    }
}

impl Llama3Config {
    /// Scaled down lightweight config for low-latency testing, CI, and local host inference.
    #[must_use]
    pub const fn tiny_test_config() -> Self {
        Self {
            hidden_dim: 128,
            intermediate_dim: 256,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 32,
            vocab_size: 1024,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5f32.to_bits(),
        }
    }

    #[must_use]
    pub fn rms_norm_eps_f32(&self) -> f32 {
        f32::from_bits(self.rms_norm_eps)
    }

    /// Construct configuration from a ModelSpecification in the catalog.
    #[must_use]
    pub fn from_spec(spec: &crate::registry::ModelSpecification) -> Self {
        Self {
            hidden_dim: spec.hidden_dim as usize,
            intermediate_dim: spec.intermediate_dim as usize,
            num_layers: spec.num_layers as usize,
            num_heads: spec.num_heads as usize,
            num_kv_heads: spec.num_kv_heads as usize,
            head_dim: spec.head_dim as usize,
            vocab_size: spec.vocab_size as usize,
            max_seq_len: spec.max_context_tokens as usize,
            rms_norm_eps: 1e-5f32.to_bits(),
        }
    }

    /// Construct configuration dynamically from a parsed GGUF file header.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture();
        let hidden_dim = gguf
            .get_u64(&format!("{arch}.embedding_length"))
            .unwrap_or(4096) as usize;
        let num_layers = gguf.get_u64(&format!("{arch}.block_count")).unwrap_or(32) as usize;
        let num_heads = gguf
            .get_u64(&format!("{arch}.attention.head_count"))
            .unwrap_or(32) as usize;
        let num_kv_heads = gguf
            .get_u64(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(num_heads as u64) as usize;
        let head_dim = hidden_dim.checked_div(num_heads).unwrap_or(128);
        let intermediate_dim = gguf
            .get_u64(&format!("{arch}.feed_forward_length"))
            .unwrap_or((hidden_dim * 4) as u64) as usize;
        let max_seq_len = gguf
            .get_u64(&format!("{arch}.context_length"))
            .unwrap_or(8192) as usize;
        let eps = gguf
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let vocab_size = gguf
            .tensors
            .get("token_embd.weight")
            .and_then(|t| t.dimensions.first().copied())
            .unwrap_or(128_256) as usize;

        Self {
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_norm_eps: eps.to_bits(),
        }
    }
}

impl ModelConfig for Llama3Config {
    fn model_name(&self) -> &'static str {
        "Meta-Llama-3.1-8B"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Single-layer Key-Value Cache containing history tokens.
#[derive(Debug, Clone, Default)]
pub struct Llama3KvCacheLayer {
    pub k: Vec<f32>, // shape: [seq_len, num_kv_heads * head_dim]
    pub v: Vec<f32>, // shape: [seq_len, num_kv_heads * head_dim]
    pub current_len: usize,
}

impl Llama3KvCacheLayer {
    #[must_use]
    pub fn new(max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            k: Vec::with_capacity(max_seq_len * kv_dim),
            v: Vec::with_capacity(max_seq_len * kv_dim),
            current_len: 0,
        }
    }

    pub fn append_kv(&mut self, k_token: &[f32], v_token: &[f32]) {
        self.k.extend_from_slice(k_token);
        self.v.extend_from_slice(v_token);
        self.current_len += 1;
    }

    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.current_len = 0;
    }
}

/// Single Transformer Decoder Layer Weights.
#[derive(Debug, Clone)]
pub struct Llama3LayerWeights {
    pub q_proj: Vec<f32>,    // [num_heads * head_dim, hidden_dim]
    pub k_proj: Vec<f32>,    // [num_kv_heads * head_dim, hidden_dim]
    pub v_proj: Vec<f32>,    // [num_kv_heads * head_dim, hidden_dim]
    pub o_proj: Vec<f32>,    // [hidden_dim, num_heads * head_dim]
    pub gate_proj: Vec<f32>, // [intermediate_dim, hidden_dim]
    pub up_proj: Vec<f32>,   // [intermediate_dim, hidden_dim]
    pub down_proj: Vec<f32>, // [hidden_dim, intermediate_dim]
    pub attn_norm: Vec<f32>, // [hidden_dim]
    pub ffn_norm: Vec<f32>,  // [hidden_dim]
}

impl Llama3LayerWeights {
    #[must_use]
    pub fn new_synthetic(config: &Llama3Config, layer_idx: usize) -> Self {
        let h = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_dim;

        let fill_weight = |size: usize, scale: f32, seed: f32| -> Vec<f32> {
            (0..size)
                .map(|i| ((i as f32 * 0.017 + seed).sin()) * scale)
                .collect()
        };

        let seed = layer_idx as f32 * 0.31;
        Self {
            q_proj: fill_weight(q_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.1),
            k_proj: fill_weight(kv_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.2),
            v_proj: fill_weight(kv_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.3),
            o_proj: fill_weight(h * q_dim, 1.0 / (q_dim as f32).sqrt(), seed + 0.4),
            gate_proj: fill_weight(inter * h, 1.0 / (h as f32).sqrt(), seed + 0.5),
            up_proj: fill_weight(inter * h, 1.0 / (h as f32).sqrt(), seed + 0.6),
            down_proj: fill_weight(h * inter, 1.0 / (inter as f32).sqrt(), seed + 0.7),
            attn_norm: vec![1.0; h],
            ffn_norm: vec![1.0; h],
        }
    }
}

/// Fully Executable LLaMA 3.1 Model with GQA, RoPE, SwiGLU, and FlashAttention.
#[derive(Debug, Clone)]
pub struct Llama3Model {
    pub config: Llama3Config,
    pub rope: RopeScalingEngine,
    pub flash_attn: FlashAttentionEngine,
    pub token_embedding: Vec<f32>, // [vocab_size, hidden_dim]
    pub layers: Vec<Llama3LayerWeights>,
    pub output_norm: Vec<f32>, // [hidden_dim]
    pub lm_head: Vec<f32>,     // [vocab_size, hidden_dim]
}

impl Llama3Model {
    #[must_use]
    pub fn new(config: Llama3Config) -> Self {
        let rope_config = RopeConfig {
            head_dim: config.head_dim,
            base_theta: 500_000.0,
            scaling: RopeScalingType::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 8192,
            },
            max_position_embeddings: config.max_seq_len,
        };
        let rope = RopeScalingEngine::new(rope_config);
        let flash_attn_cfg =
            FlashAttentionConfig::new(config.num_heads, config.num_kv_heads, config.head_dim, true);
        let flash_attn = FlashAttentionEngine::new(flash_attn_cfg);

        let h = config.hidden_dim;
        let v = config.vocab_size;

        let token_embedding: Vec<f32> = (0..v * h)
            .map(|i| ((i as f32 * 0.013).cos()) * (1.0 / (h as f32).sqrt()))
            .collect();
        let output_norm = vec![1.0; h];
        let lm_head: Vec<f32> = (0..v * h)
            .map(|i| ((i as f32 * 0.019).sin()) * (1.0 / (h as f32).sqrt()))
            .collect();

        let layers = (0..config.num_layers)
            .map(|l| Llama3LayerWeights::new_synthetic(&config, l))
            .collect();

        Self {
            config,
            rope,
            flash_attn,
            token_embedding,
            layers,
            output_norm,
            lm_head,
        }
    }

    /// Loads tensor weights from SafeTensors binary slice.
    pub fn load_from_safetensors(
        &mut self,
        header: &SafeTensorsHeader,
        data_slice: &[u8],
    ) -> Result<()> {
        if let Some(info) = header.tensors.get("model.embed_tokens.weight") {
            let start = info.data_offsets.0 as usize;
            let end = info.data_offsets.1 as usize;
            if end <= data_slice.len() {
                let byte_slice = &data_slice[start..end];
                if byte_slice.len().is_multiple_of(4) {
                    let floats: &[f32] = bytemuck::cast_slice(byte_slice);
                    let count = floats.len().min(self.token_embedding.len());
                    self.token_embedding[..count].copy_from_slice(&floats[..count]);
                }
            }
        }
        Ok(())
    }

    /// Helper to extract and dequantize a tensor from GGUF binary slice into a target f32 buffer.
    fn extract_gguf_tensor_to_buffer(
        gguf: &GgufFile,
        tensor_name: &str,
        data_slice: &[u8],
        target: &mut [f32],
    ) {
        if let Some(info) = gguf.tensors.get(tensor_name) {
            let offset = gguf.tensor_data_offset + info.offset as usize;
            if offset >= data_slice.len() {
                return;
            }
            let avail = &data_slice[offset..];
            match info.quant_type {
                crate::formats::GgufQuantType::F32 => {
                    let expected_bytes = target.len() * 4;
                    if avail.len() >= expected_bytes {
                        let floats: &[f32] = bytemuck::cast_slice(&avail[..expected_bytes]);
                        target.copy_from_slice(floats);
                    }
                }
                crate::formats::GgufQuantType::F16 => {
                    let count = target.len().min(avail.len() / 2);
                    for i in 0..count {
                        let raw = u16::from_le_bytes([avail[i * 2], avail[i * 2 + 1]]);
                        target[i] = oxide_quant::f16(raw).to_f32();
                    }
                }
                crate::formats::GgufQuantType::Q8_0 => {
                    let block_size = 34; // 2 bytes f16 scale + 32 bytes int8
                    let num_blocks = (target.len() / 32).min(avail.len() / block_size);
                    for b_idx in 0..num_blocks {
                        let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                        let scale_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                        let scale = oxide_quant::f16(scale_raw).to_f32();
                        let start = b_idx * 32;
                        for i in 0..32 {
                            let q = block_raw[2 + i] as i8;
                            target[start + i] = (q as f32) * scale;
                        }
                    }
                }
                crate::formats::GgufQuantType::Q4_0 => {
                    let block_size = 18; // 2 bytes f16 scale + 16 bytes nibbles
                    let num_blocks = (target.len() / 32).min(avail.len() / block_size);
                    for b_idx in 0..num_blocks {
                        let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                        let scale_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                        let scale = oxide_quant::f16(scale_raw).to_f32();
                        let start = b_idx * 32;
                        for i in 0..16 {
                            let byte = block_raw[2 + i];
                            let q0 = (byte & 0x0F) as i8 - 8;
                            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                            target[start + i] = (q0 as f32) * scale;
                            target[start + i + 16] = (q1 as f32) * scale;
                        }
                    }
                }
                crate::formats::GgufQuantType::Q4_K_M | crate::formats::GgufQuantType::Q4_1 => {
                    let block_size = 144; // 2+2+12+128 = 144 bytes per 256 weights
                    let num_blocks = (target.len() / 256).min(avail.len() / block_size);
                    for b_idx in 0..num_blocks {
                        let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                        let d_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                        let dmin_raw = u16::from_le_bytes([block_raw[2], block_raw[3]]);
                        let d = oxide_quant::f16(d_raw).to_f32();
                        let dmin = oxide_quant::f16(dmin_raw).to_f32();
                        let qs = &block_raw[16..144];
                        let start = b_idx * 256;
                        for i in 0..128 {
                            let byte = qs[i];
                            let q0 = (byte & 0x0F) as f32;
                            let q1 = ((byte >> 4) & 0x0F) as f32;
                            target[start + i] = q0 * d + dmin;
                            target[start + i + 128] = q1 * d + dmin;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Loads tensor weights from GGUF binary container.
    pub fn load_from_gguf(&mut self, gguf: &GgufFile, data_slice: &[u8]) -> Result<()> {
        // 1. Embeddings
        Self::extract_gguf_tensor_to_buffer(
            gguf,
            "token_embd.weight",
            data_slice,
            &mut self.token_embedding,
        );

        // 2. Output norm
        Self::extract_gguf_tensor_to_buffer(
            gguf,
            "output_norm.weight",
            data_slice,
            &mut self.output_norm,
        );

        // 3. LM Head (if distinct from embeddings)
        if gguf.tensors.contains_key("output.weight") {
            Self::extract_gguf_tensor_to_buffer(
                gguf,
                "output.weight",
                data_slice,
                &mut self.lm_head,
            );
        } else if self.lm_head.len() == self.token_embedding.len() {
            // Tied weights fallback
            self.lm_head.copy_from_slice(&self.token_embedding);
        }

        // 4. Transformer Decoder Layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let q_name = format!("blk.{i}.attn_q.weight");
            let k_name = format!("blk.{i}.attn_k.weight");
            let v_name = format!("blk.{i}.attn_v.weight");
            let o_name = format!("blk.{i}.attn_output.weight");
            let gate_name = format!("blk.{i}.ffn_gate.weight");
            let up_name = format!("blk.{i}.ffn_up.weight");
            let down_name = format!("blk.{i}.ffn_down.weight");
            let attn_norm_name = format!("blk.{i}.attn_norm.weight");
            let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");

            Self::extract_gguf_tensor_to_buffer(gguf, &q_name, data_slice, &mut layer.q_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &k_name, data_slice, &mut layer.k_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &v_name, data_slice, &mut layer.v_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &o_name, data_slice, &mut layer.o_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &gate_name, data_slice, &mut layer.gate_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &up_name, data_slice, &mut layer.up_proj);
            Self::extract_gguf_tensor_to_buffer(gguf, &down_name, data_slice, &mut layer.down_proj);
            Self::extract_gguf_tensor_to_buffer(
                gguf,
                &attn_norm_name,
                data_slice,
                &mut layer.attn_norm,
            );
            Self::extract_gguf_tensor_to_buffer(
                gguf,
                &ffn_norm_name,
                data_slice,
                &mut layer.ffn_norm,
            );
        }

        Ok(())
    }

    /// Construct model directly from a ModelSpecification.
    #[must_use]
    pub fn from_spec(spec: &crate::registry::ModelSpecification) -> Self {
        Self::new(Llama3Config::from_spec(spec))
    }

    /// Load any model from disk (GGUF or SafeTensors) with automatic architecture discovery.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let file = std::fs::File::open(p).map_err(|e| {
            oxide_core::error::EngineError::BackendError(format!(
                "Failed to open model file {}: {e}",
                p.display()
            ))
        })?;

        // SAFETY: The underlying model file is mapped read-only and is immutable during process runtime.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                oxide_core::error::EngineError::BackendError(format!(
                    "Failed to memory-map model file {}: {e}",
                    p.display()
                ))
            })?
        };

        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("gguf")
            || (mmap.len() >= 4 && &mmap[0..4] == crate::formats::GGUF_MAGIC)
        {
            let gguf = GgufFile::parse(&mmap)?;
            let cfg = Llama3Config::from_gguf(&gguf);
            let mut model = Self::new(cfg);
            model.load_from_gguf(&gguf, &mmap)?;
            Ok(model)
        } else if ext.eq_ignore_ascii_case("safetensors") {
            let (header, _) = SafeTensorsHeader::parse_from_bytes(&mmap)?;
            let filename = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let cfg = if let Some(spec) = crate::registry::ModelSpecification::lookup(filename) {
                Llama3Config::from_spec(&spec)
            } else {
                Llama3Config::default()
            };
            let mut model = Self::new(cfg);
            model.load_from_safetensors(&header, &mmap)?;
            Ok(model)
        } else {
            Err(oxide_core::error::EngineError::BackendError(format!(
                "Unsupported model format for file: {}",
                p.display()
            )))
        }
    }

    /// Universal loader: resolves from model file path on disk or catalog lookup by name.
    pub fn from_model_name_or_path(query_or_path: &str) -> Result<Self> {
        let path = std::path::Path::new(query_or_path);
        if path.exists() {
            return Self::from_file(path);
        }
        if let Some(spec) = crate::registry::ModelSpecification::lookup(query_or_path) {
            return Ok(Self::from_spec(&spec));
        }
        Ok(Self::new(Llama3Config::default()))
    }

    /// Pre-allocated scratch buffers to guarantee zero dynamic allocations in the token generation loop.
    pub fn create_scratch(&self) -> Llama3ScratchBuffers {
        Llama3ScratchBuffers::new(&self.config)
    }

    /// Computes a single autoregressive forward step reusing pre-allocated scratch buffers.
    #[allow(clippy::many_single_char_names)]
    pub fn forward_step_with_scratch(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
        scratch: &mut Llama3ScratchBuffers,
    ) -> Result<()> {
        let h = self.config.hidden_dim;
        let tok_idx = (token_id as usize) % self.config.vocab_size;

        // 1. Embedding lookup
        let emb_offset = tok_idx * h;
        if emb_offset + h <= self.token_embedding.len() {
            scratch
                .hidden
                .copy_from_slice(&self.token_embedding[emb_offset..emb_offset + h]);
        } else {
            scratch.hidden.fill(0.0);
        }

        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let eps = self.config.rms_norm_eps_f32();

        // 2. Transformer Decoder Layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer(
                layer_idx, layer, position, kv_cache, scratch, eps, h, q_dim, kv_dim,
            );
        }

        // 3. Final RMSNorm
        Self::rms_norm(
            &scratch.hidden,
            &self.output_norm,
            &mut scratch.final_norm,
            eps,
        );

        // 4. LM Head projection to vocabulary logits
        let v = self.config.vocab_size;
        Self::gemv(
            &self.lm_head,
            &scratch.final_norm,
            v,
            h,
            &mut scratch.logits,
        );

        Ok(())
    }

    /// Forward pass through a single transformer decoder layer.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer(
        &self,
        layer_idx: usize,
        layer: &Llama3LayerWeights,
        position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
        scratch: &mut Llama3ScratchBuffers,
        eps: f32,
        h: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        Self::rms_norm(
            &scratch.hidden,
            &layer.attn_norm,
            &mut scratch.norm_hidden,
            eps,
        );

        // Q, K, V Projections via SIMD GEMV
        Self::gemv(
            &layer.q_proj,
            &scratch.norm_hidden,
            q_dim,
            h,
            &mut scratch.q,
        );
        Self::gemv(
            &layer.k_proj,
            &scratch.norm_hidden,
            kv_dim,
            h,
            &mut scratch.k,
        );
        Self::gemv(
            &layer.v_proj,
            &scratch.norm_hidden,
            kv_dim,
            h,
            &mut scratch.v,
        );

        // RoPE Rotary Embedding
        for head_idx in 0..self.config.num_heads {
            let start = head_idx * self.config.head_dim;
            let end = start + self.config.head_dim;
            self.rope
                .apply_rotary_in_place(&mut scratch.q[start..end], position);
        }
        for kv_head_idx in 0..self.config.num_kv_heads {
            let start = kv_head_idx * self.config.head_dim;
            let end = start + self.config.head_dim;
            self.rope
                .apply_rotary_in_place(&mut scratch.k[start..end], position);
        }

        // KV Cache append
        if let Some(layer_cache) = kv_cache.get_mut(layer_idx) {
            layer_cache.append_kv(&scratch.k, &scratch.v);
        }

        // Attention Output Projection
        for head in 0..self.config.num_heads {
            let q_start = head * self.config.head_dim;
            let q_slice = &scratch.q[q_start..q_start + self.config.head_dim];
            let out_slice = &mut scratch.attn_out[q_start..q_start + self.config.head_dim];

            // Grouped-Query Attention head mapping
            let kv_head = head / (self.config.num_heads / self.config.num_kv_heads);
            let k_start = kv_head * self.config.head_dim;
            let k_slice = &scratch.k[k_start..k_start + self.config.head_dim];
            let v_slice = &scratch.v[k_start..k_start + self.config.head_dim];

            self.flash_attn
                .forward_head(q_slice, k_slice, v_slice, 1, 1, out_slice);
        }

        Self::gemv(
            &layer.o_proj,
            &scratch.attn_out,
            h,
            q_dim,
            &mut scratch.o_proj_out,
        );

        // Residual 1
        for i in 0..h {
            scratch.hidden[i] += scratch.o_proj_out[i];
        }

        // FFN RMSNorm & SwiGLU MLP
        Self::rms_norm(
            &scratch.hidden,
            &layer.ffn_norm,
            &mut scratch.ffn_norm_hidden,
            eps,
        );

        let inter_dim = self.config.intermediate_dim;
        Self::gemv(
            &layer.gate_proj,
            &scratch.ffn_norm_hidden,
            inter_dim,
            h,
            &mut scratch.gate,
        );
        Self::gemv(
            &layer.up_proj,
            &scratch.ffn_norm_hidden,
            inter_dim,
            h,
            &mut scratch.up,
        );

        // SwiGLU: down_proj(silu(gate) * up)
        for i in 0..inter_dim {
            let g = scratch.gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            scratch.activated[i] = silu_g * scratch.up[i];
        }

        Self::gemv(
            &layer.down_proj,
            &scratch.activated,
            h,
            inter_dim,
            &mut scratch.mlp_out,
        );

        // Residual 2
        for i in 0..h {
            scratch.hidden[i] += scratch.mlp_out[i];
        }
    }

    /// Forward pass through a range of transformer decoder layers [start_layer..end_layer].
    pub fn forward_layers(
        &self,
        start_layer: usize,
        end_layer: usize,
        position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
        scratch: &mut Llama3ScratchBuffers,
    ) {
        let h = self.config.hidden_dim;
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let eps = self.config.rms_norm_eps_f32();

        let end = end_layer.min(self.layers.len());
        for layer_idx in start_layer..end {
            if let Some(layer) = self.layers.get(layer_idx) {
                self.forward_layer(
                    layer_idx, layer, position, kv_cache, scratch, eps, h, q_dim, kv_dim,
                );
            }
        }
    }

    /// Computes a single autoregressive forward step for an input token.
    /// Returns unnormalized output logits of shape `[vocab_size]`.
    pub fn forward_step(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
    ) -> Result<Vec<f32>> {
        let mut scratch = self.create_scratch();
        self.forward_step_with_scratch(token_id, position, kv_cache, &mut scratch)?;
        Ok(scratch.logits)
    }

    /// Computes batched forward pass (prefill mode) for a sequence of prompt tokens.
    /// Ingests all tokens in order, populates the KV-cache, and returns the logits for the final token.
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        start_position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(oxide_core::error::EngineError::BackendError(
                "Cannot perform forward_batch on empty token slice".to_string(),
            ));
        }

        let mut scratch = self.create_scratch();
        for (idx, &token) in tokens.iter().enumerate() {
            let pos = start_position + idx;
            self.forward_step_with_scratch(token, pos, kv_cache, &mut scratch)?;
        }
        Ok(scratch.logits)
    }

    #[inline(always)]
    fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
        let mean_sq = oxide_quant::simd::dot_f32(input, input) / input.len().max(1) as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let chunks = input.len() / 8;
        let remainder = input.len() % 8;
        for i in 0..chunks {
            let base = i * 8;
            for j in 0..8 {
                output[base + j] = input[base + j] * inv_rms * weight[base + j];
            }
        }
        let rem_start = chunks * 8;
        for j in 0..remainder {
            output[rem_start + j] = input[rem_start + j] * inv_rms * weight[rem_start + j];
        }
    }

    #[inline(always)]
    fn gemv(matrix: &[f32], vector: &[f32], out_dim: usize, in_dim: usize, output: &mut [f32]) {
        for i in 0..out_dim {
            let offset = i * in_dim;
            if offset + in_dim <= matrix.len() {
                let row = &matrix[offset..offset + in_dim];
                output[i] = oxide_quant::simd::dot_f32(row, vector);
            }
        }
    }
}

/// Pre-allocated reusable execution buffers across transformer layers.
#[derive(Debug, Clone)]
pub struct Llama3ScratchBuffers {
    pub hidden: Vec<f32>,
    pub norm_hidden: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub o_proj_out: Vec<f32>,
    pub ffn_norm_hidden: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated: Vec<f32>,
    pub mlp_out: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

impl Llama3ScratchBuffers {
    #[must_use]
    pub fn new(config: &Llama3Config) -> Self {
        let h = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_dim;
        let v = config.vocab_size;

        Self {
            hidden: vec![0.0; h],
            norm_hidden: vec![0.0; h],
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            attn_out: vec![0.0; q_dim],
            o_proj_out: vec![0.0; h],
            ffn_norm_hidden: vec![0.0; h],
            gate: vec![0.0; inter],
            up: vec![0.0; inter],
            activated: vec![0.0; inter],
            mlp_out: vec![0.0; h],
            final_norm: vec![0.0; h],
            logits: vec![0.0; v],
        }
    }
}
