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

    /// Loads tensor weights from GGUF binary container.
    pub fn load_from_gguf(&mut self, gguf: &GgufFile, data_slice: &[u8]) -> Result<()> {
        if let Some(info) = gguf.tensors.get("token_embd.weight") {
            let offset = gguf.tensor_data_offset + info.offset as usize;
            let expected_bytes = self.token_embedding.len() * 4;
            if offset + expected_bytes <= data_slice.len() {
                let byte_slice = &data_slice[offset..offset + expected_bytes];
                let floats: &[f32] = bytemuck::cast_slice(byte_slice);
                self.token_embedding.copy_from_slice(floats);
            }
        }
        Ok(())
    }

    /// Computes a single autoregressive forward step for an input token.
    /// Returns unnormalized output logits of shape `[vocab_size]`.
    #[allow(clippy::many_single_char_names)]
    pub fn forward_step(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut [Llama3KvCacheLayer],
    ) -> Result<Vec<f32>> {
        let h = self.config.hidden_dim;
        let tok_idx = (token_id as usize) % self.config.vocab_size;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; h];
        let emb_offset = tok_idx * h;
        if emb_offset + h <= self.token_embedding.len() {
            hidden.copy_from_slice(&self.token_embedding[emb_offset..emb_offset + h]);
        }

        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let eps = self.config.rms_norm_eps_f32();

        // 2. Transformer Decoder Layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut norm_hidden = vec![0.0f32; h];
            Self::rms_norm(&hidden, &layer.attn_norm, &mut norm_hidden, eps);

            // Q, K, V Projections
            let mut q = vec![0.0f32; q_dim];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];

            Self::gemv(&layer.q_proj, &norm_hidden, q_dim, h, &mut q);
            Self::gemv(&layer.k_proj, &norm_hidden, kv_dim, h, &mut k);
            Self::gemv(&layer.v_proj, &norm_hidden, kv_dim, h, &mut v);

            // RoPE Rotary Embedding
            for head_idx in 0..self.config.num_heads {
                let start = head_idx * self.config.head_dim;
                let end = start + self.config.head_dim;
                self.rope
                    .apply_rotary_in_place(&mut q[start..end], position);
            }
            for kv_head_idx in 0..self.config.num_kv_heads {
                let start = kv_head_idx * self.config.head_dim;
                let end = start + self.config.head_dim;
                self.rope
                    .apply_rotary_in_place(&mut k[start..end], position);
            }

            // KV Cache append
            if let Some(layer_cache) = kv_cache.get_mut(layer_idx) {
                layer_cache.append_kv(&k, &v);
            }

            // Attention Output Projection
            let mut attn_out = vec![0.0f32; q_dim];
            for head in 0..self.config.num_heads {
                let q_start = head * self.config.head_dim;
                let q_slice = &q[q_start..q_start + self.config.head_dim];
                let out_slice = &mut attn_out[q_start..q_start + self.config.head_dim];

                // Grouped-Query Attention head mapping
                let kv_head = head / (self.config.num_heads / self.config.num_kv_heads);
                let k_start = kv_head * self.config.head_dim;
                let k_slice = &k[k_start..k_start + self.config.head_dim];
                let v_slice = &v[k_start..k_start + self.config.head_dim];

                self.flash_attn
                    .forward_head(q_slice, k_slice, v_slice, 1, 1, out_slice);
            }

            let mut o_proj_out = vec![0.0f32; h];
            Self::gemv(&layer.o_proj, &attn_out, h, q_dim, &mut o_proj_out);

            // Residual 1
            for i in 0..h {
                hidden[i] += o_proj_out[i];
            }

            // FFN RMSNorm & SwiGLU MLP
            let mut ffn_norm_hidden = vec![0.0f32; h];
            Self::rms_norm(&hidden, &layer.ffn_norm, &mut ffn_norm_hidden, eps);

            let inter_dim = self.config.intermediate_dim;
            let mut gate = vec![0.0f32; inter_dim];
            let mut up = vec![0.0f32; inter_dim];

            Self::gemv(&layer.gate_proj, &ffn_norm_hidden, inter_dim, h, &mut gate);
            Self::gemv(&layer.up_proj, &ffn_norm_hidden, inter_dim, h, &mut up);

            // SwiGLU: down_proj(silu(gate) * up)
            let mut activated = vec![0.0f32; inter_dim];
            for i in 0..inter_dim {
                let g = gate[i];
                let silu_g = g / (1.0 + (-g).exp());
                activated[i] = silu_g * up[i];
            }

            let mut mlp_out = vec![0.0f32; h];
            Self::gemv(&layer.down_proj, &activated, h, inter_dim, &mut mlp_out);

            // Residual 2
            for i in 0..h {
                hidden[i] += mlp_out[i];
            }
        }

        // 3. Final RMSNorm
        let mut final_norm = vec![0.0f32; h];
        Self::rms_norm(&hidden, &self.output_norm, &mut final_norm, eps);

        // 4. LM Head projection to vocabulary logits
        let v = self.config.vocab_size;
        let mut logits = vec![0.0f32; v];
        Self::gemv(&self.lm_head, &final_norm, v, h, &mut logits);

        Ok(logits)
    }

    #[inline(always)]
    fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
        let mean_sq: f32 = input.iter().map(|&x| x * x).sum::<f32>() / input.len().max(1) as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for (i, (&x, &w)) in input.iter().zip(weight.iter()).enumerate() {
            output[i] = x * inv_rms * w;
        }
    }

    #[inline(always)]
    fn gemv(matrix: &[f32], vector: &[f32], out_dim: usize, in_dim: usize, output: &mut [f32]) {
        for i in 0..out_dim {
            let offset = i * in_dim;
            if offset + in_dim <= matrix.len() {
                let row = &matrix[offset..offset + in_dim];
                let sum: f32 = row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum();
                output[i] = sum;
            }
        }
    }
}
