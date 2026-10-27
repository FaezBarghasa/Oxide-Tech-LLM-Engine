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
    pub head_dim_full: Option<usize>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: u32, // Store scaled integer eps (e.g. 1e-5 represented as bits or f32)
    pub eos_token_id: Option<u32>,
    pub rope_theta: u32,
    pub rope_theta_swa: Option<u32>,
    pub sliding_window: Option<usize>,
    pub is_gemma: bool,
    pub is_qwen: bool,
    pub final_logit_softcapping: Option<u32>,
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
            head_dim_full: None,
            vocab_size: 128_256,
            max_seq_len: 131_072,
            rms_norm_eps: 1e-5f32.to_bits(),
            eos_token_id: Some(128_001),
            rope_theta: 500_000.0f32.to_bits(),
            rope_theta_swa: None,
            sliding_window: None,
            is_gemma: false,
            is_qwen: false,
            final_logit_softcapping: None,
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
            head_dim_full: None,
            vocab_size: 1024,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5f32.to_bits(),
            eos_token_id: Some(2),
            rope_theta: 10_000.0f32.to_bits(),
            rope_theta_swa: None,
            sliding_window: None,
            is_gemma: false,
            is_qwen: false,
            final_logit_softcapping: None,
        }
    }

    #[must_use]
    pub fn max_head_dim(&self) -> usize {
        self.head_dim_full
            .unwrap_or(self.head_dim)
            .max(self.head_dim)
    }

    #[must_use]
    pub fn rms_norm_eps_f32(&self) -> f32 {
        f32::from_bits(self.rms_norm_eps)
    }

    #[must_use]
    pub fn rope_theta_f32(&self) -> f32 {
        f32::from_bits(self.rope_theta)
    }

    #[must_use]
    pub fn rope_theta_swa_f32(&self) -> Option<f32> {
        self.rope_theta_swa.map(f32::from_bits)
    }

    #[must_use]
    pub fn final_logit_softcapping_f32(&self) -> Option<f32> {
        self.final_logit_softcapping.map(f32::from_bits)
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
            head_dim_full: None,
            vocab_size: spec.vocab_size as usize,
            max_seq_len: spec.max_context_tokens as usize,
            rms_norm_eps: 1e-5f32.to_bits(),
            eos_token_id: None,
            rope_theta: 500_000.0f32.to_bits(),
            rope_theta_swa: None,
            sliding_window: None,
            is_gemma: false,
            is_qwen: false,
            final_logit_softcapping: None,
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

        // Determine accurate head_dim from tensor shapes
        let head_dim = if let Some(t) = gguf.tensors.get("blk.0.attn_q.weight") {
            if t.dimensions.len() >= 2 && num_heads > 0 {
                (t.dimensions[1] as usize) / num_heads
            } else {
                hidden_dim.checked_div(num_heads).unwrap_or(128)
            }
        } else if let Some(t) = gguf.tensors.get("blk.3.attn_q.weight") {
            if t.dimensions.len() >= 2 && num_heads > 0 {
                (t.dimensions[1] as usize) / num_heads
            } else {
                hidden_dim.checked_div(num_heads).unwrap_or(128)
            }
        } else if let Some(t) = gguf.tensors.get("blk.0.attn_qkv.weight") {
            let total_heads = num_heads + 2 * num_kv_heads;
            if t.dimensions.len() >= 2 && total_heads > 0 {
                (t.dimensions[1] as usize) / total_heads
            } else {
                hidden_dim.checked_div(num_heads).unwrap_or(128)
            }
        } else {
            hidden_dim.checked_div(num_heads).unwrap_or(128)
        };

        // Determine accurate intermediate_dim from ffn tensor shapes
        let intermediate_dim = if let Some(t) = gguf
            .tensors
            .get("blk.0.ffn_gate.weight")
            .or_else(|| gguf.tensors.get("blk.0.ffn_up.weight"))
        {
            if t.dimensions.len() >= 2 {
                t.dimensions[1] as usize
            } else {
                gguf.get_u64(&format!("{arch}.feed_forward_length"))
                    .unwrap_or((hidden_dim * 4) as u64) as usize
            }
        } else {
            gguf.get_u64(&format!("{arch}.feed_forward_length"))
                .unwrap_or((hidden_dim * 4) as u64) as usize
        };

        let max_seq_len = gguf
            .get_u64(&format!("{arch}.context_length"))
            .unwrap_or(8192) as usize;
        let eps = gguf
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let vocab_size = gguf
            .tensors
            .get("token_embd.weight")
            .and_then(|t| t.dimensions.iter().max().copied())
            .unwrap_or(128_256) as usize;
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id");

        let rope_theta = gguf
            .get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(500_000.0);
        let key_len = gguf
            .get_u64(&format!("{arch}.attention.key_length"))
            .map(|v| v as usize);
        let key_len_swa = gguf
            .get_u64(&format!("{arch}.attention.key_length_swa"))
            .map(|v| v as usize);
        let head_dim_full = key_len;
        let rope_theta_swa = gguf
            .get_f32(&format!("{arch}.rope.freq_base_swa"))
            .map(|f| f.to_bits());
        let sliding_window = gguf
            .get_u64(&format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize);
        let head_dim = key_len.or(key_len_swa).unwrap_or(head_dim);

        let is_gemma = arch.contains("gemma");
        let is_qwen = arch.contains("qwen");
        let final_logit_softcapping = gguf
            .get_f32(&format!("{arch}.final_logit_softcapping"))
            .map(|f| f.to_bits());

        Self {
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            head_dim_full,
            vocab_size,
            max_seq_len,
            rms_norm_eps: eps.to_bits(),
            eos_token_id,
            rope_theta: rope_theta.to_bits(),
            rope_theta_swa,
            sliding_window,
            is_gemma,
            is_qwen,
            final_logit_softcapping,
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

/// Quantized or dense weight matrix with direct SIMD kernel dispatch and zero-copy mmap slicing.
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub enum QuantizedTensor {
    F32 {
        data: std::borrow::Cow<'static, [f32]>,
        m: usize,
        n: usize,
    },
    Q4_K {
        blocks: std::borrow::Cow<'static, [oxide_quant::BlockQ4_K]>,
        m: usize,
        n: usize,
    },
    Q6_K {
        blocks: std::borrow::Cow<'static, [oxide_quant::BlockQ6_K]>,
        m: usize,
        n: usize,
    },
    Q8_0 {
        blocks: std::borrow::Cow<'static, [oxide_quant::BlockQ8_0]>,
        m: usize,
        n: usize,
    },
    PTQ1_0 {
        blocks: std::borrow::Cow<'static, [oxide_quant::BlockPTQ1_0]>,
        m: usize,
        n: usize,
    },
}

impl QuantizedTensor {
    #[must_use]
    pub fn empty() -> Self {
        Self::F32 {
            data: std::borrow::Cow::Borrowed(&[]),
            m: 0,
            n: 0,
        }
    }

    #[must_use]
    pub fn from_f32(data: Vec<f32>, m: usize, n: usize) -> Self {
        Self::F32 {
            data: std::borrow::Cow::Owned(data),
            m,
            n,
        }
    }

    #[must_use]
    pub fn from_f32_slice(data: &'static [f32], m: usize, n: usize) -> Self {
        Self::F32 {
            data: std::borrow::Cow::Borrowed(data),
            m,
            n,
        }
    }

    #[must_use]
    pub fn from_q4_k_slice(blocks: &'static [oxide_quant::BlockQ4_K], m: usize, n: usize) -> Self {
        Self::Q4_K {
            blocks: std::borrow::Cow::Borrowed(blocks),
            m,
            n,
        }
    }

    #[must_use]
    pub fn from_q6_k_slice(blocks: &'static [oxide_quant::BlockQ6_K], m: usize, n: usize) -> Self {
        Self::Q6_K {
            blocks: std::borrow::Cow::Borrowed(blocks),
            m,
            n,
        }
    }

    #[must_use]
    pub fn from_q8_0_slice(blocks: &'static [oxide_quant::BlockQ8_0], m: usize, n: usize) -> Self {
        Self::Q8_0 {
            blocks: std::borrow::Cow::Borrowed(blocks),
            m,
            n,
        }
    }

    #[must_use]
    pub fn from_ptq1_0_slice(
        blocks: &'static [oxide_quant::BlockPTQ1_0],
        m: usize,
        n: usize,
    ) -> Self {
        Self::PTQ1_0 {
            blocks: std::borrow::Cow::Borrowed(blocks),
            m,
            n,
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { m, .. }
            | Self::Q4_K { m, .. }
            | Self::Q6_K { m, .. }
            | Self::Q8_0 { m, .. }
            | Self::PTQ1_0 { m, .. } => *m,
        }
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { n, .. }
            | Self::Q4_K { n, .. }
            | Self::Q6_K { n, .. }
            | Self::Q8_0 { n, .. }
            | Self::PTQ1_0 { n, .. } => *n,
        }
    }

    /// Zero-copy row slicing of contiguous quantized or F32 weight blocks.
    #[must_use]
    pub fn slice_rows(&self, start_row: usize, num_rows: usize) -> Option<Self> {
        let n = self.cols();
        match self {
            Self::F32 { data, .. } => {
                let start = start_row * n;
                let end = start + num_rows * n;
                if end <= data.len() {
                    match data {
                        std::borrow::Cow::Borrowed(s) => Some(Self::F32 {
                            data: std::borrow::Cow::Borrowed(&s[start..end]),
                            m: num_rows,
                            n,
                        }),
                        std::borrow::Cow::Owned(v) => Some(Self::F32 {
                            data: std::borrow::Cow::Owned(v[start..end].to_vec()),
                            m: num_rows,
                            n,
                        }),
                    }
                } else {
                    None
                }
            }
            Self::Q4_K { blocks, .. } => {
                if !n.is_multiple_of(256) {
                    return None;
                }
                let bpr = n / 256;
                let start_b = start_row * bpr;
                let end_b = start_b + num_rows * bpr;
                if end_b <= blocks.len() {
                    match blocks {
                        std::borrow::Cow::Borrowed(s) => Some(Self::Q4_K {
                            blocks: std::borrow::Cow::Borrowed(&s[start_b..end_b]),
                            m: num_rows,
                            n,
                        }),
                        std::borrow::Cow::Owned(v) => Some(Self::Q4_K {
                            blocks: std::borrow::Cow::Owned(v[start_b..end_b].to_vec()),
                            m: num_rows,
                            n,
                        }),
                    }
                } else {
                    None
                }
            }
            Self::Q6_K { blocks, .. } => {
                if !n.is_multiple_of(256) {
                    return None;
                }
                let bpr = n / 256;
                let start_b = start_row * bpr;
                let end_b = start_b + num_rows * bpr;
                if end_b <= blocks.len() {
                    match blocks {
                        std::borrow::Cow::Borrowed(s) => Some(Self::Q6_K {
                            blocks: std::borrow::Cow::Borrowed(&s[start_b..end_b]),
                            m: num_rows,
                            n,
                        }),
                        std::borrow::Cow::Owned(v) => Some(Self::Q6_K {
                            blocks: std::borrow::Cow::Owned(v[start_b..end_b].to_vec()),
                            m: num_rows,
                            n,
                        }),
                    }
                } else {
                    None
                }
            }
            Self::Q8_0 { blocks, .. } => {
                if !n.is_multiple_of(32) {
                    return None;
                }
                let bpr = n / 32;
                let start_b = start_row * bpr;
                let end_b = start_b + num_rows * bpr;
                if end_b <= blocks.len() {
                    match blocks {
                        std::borrow::Cow::Borrowed(s) => Some(Self::Q8_0 {
                            blocks: std::borrow::Cow::Borrowed(&s[start_b..end_b]),
                            m: num_rows,
                            n,
                        }),
                        std::borrow::Cow::Owned(v) => Some(Self::Q8_0 {
                            blocks: std::borrow::Cow::Owned(v[start_b..end_b].to_vec()),
                            m: num_rows,
                            n,
                        }),
                    }
                } else {
                    None
                }
            }
            Self::PTQ1_0 { blocks, .. } => {
                if !n.is_multiple_of(128) {
                    return None;
                }
                let bpr = n / 128;
                let start_b = start_row * bpr;
                let end_b = start_b + num_rows * bpr;
                if end_b <= blocks.len() {
                    match blocks {
                        std::borrow::Cow::Borrowed(s) => Some(Self::PTQ1_0 {
                            blocks: std::borrow::Cow::Borrowed(&s[start_b..end_b]),
                            m: num_rows,
                            n,
                        }),
                        std::borrow::Cow::Owned(v) => Some(Self::PTQ1_0 {
                            blocks: std::borrow::Cow::Owned(v[start_b..end_b].to_vec()),
                            m: num_rows,
                            n,
                        }),
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Extracts a single dequantized embedding row on-the-fly with zero intermediate heap allocations.
    pub fn extract_row(&self, row_idx: usize, out: &mut [f32]) {
        match self {
            Self::F32 { data, n, .. } => {
                let start = row_idx * (*n);
                if start + (*n) <= data.len() && out.len() >= *n {
                    out[..*n].copy_from_slice(&data[start..start + *n]);
                }
            }
            Self::Q8_0 { blocks, n, .. } => {
                let blocks_per_row = *n / 32;
                let start_block = row_idx * blocks_per_row;
                if start_block + blocks_per_row <= blocks.len() && out.len() >= *n {
                    for b in 0..blocks_per_row {
                        let blk = &blocks[start_block + b];
                        let d = blk.scale.to_f32();
                        for i in 0..32 {
                            out[b * 32 + i] = f32::from(blk.qs[i]) * d;
                        }
                    }
                }
            }
            Self::Q4_K { blocks, n, .. } => {
                let blocks_per_row = *n / 256;
                let start_block = row_idx * blocks_per_row;
                if start_block + blocks_per_row <= blocks.len() && out.len() >= *n {
                    let mut chunk = [0.0f32; 256];
                    for b in 0..blocks_per_row {
                        blocks[start_block + b].dequantize(&mut chunk);
                        out[b * 256..(b + 1) * 256].copy_from_slice(&chunk);
                    }
                }
            }
            Self::Q6_K { blocks, n, .. } => {
                let blocks_per_row = *n / 256;
                let start_block = row_idx * blocks_per_row;
                if start_block + blocks_per_row <= blocks.len() && out.len() >= *n {
                    let mut chunk = [0.0f32; 256];
                    for b in 0..blocks_per_row {
                        blocks[start_block + b].dequantize(&mut chunk);
                        out[b * 256..(b + 1) * 256].copy_from_slice(&chunk);
                    }
                }
            }
            Self::PTQ1_0 { blocks, n, .. } => {
                let blocks_per_row = *n / 128;
                let start_block = row_idx * blocks_per_row;
                if start_block + blocks_per_row <= blocks.len() && out.len() >= *n {
                    let mut chunk = [0.0f32; 128];
                    for b in 0..blocks_per_row {
                        blocks[start_block + b].dequantize(&mut chunk);
                        out[b * 128..(b + 1) * 128].copy_from_slice(&chunk);
                    }
                }
            }
        }
    }

    /// High-performance SIMD GEMV vector-matrix multiply dispatching directly to native quantized kernels.
    #[inline(always)]
    pub fn gemv(&self, vector: &[f32], output: &mut [f32]) {
        match self {
            Self::F32 { data, m, n } => {
                oxide_quant::simd::gemv_blocked_f32(data.as_ref(), vector, *m, *n, output);
            }
            Self::Q4_K { blocks, m, n } => {
                oxide_quant::simd::gemv_q4_k(blocks.as_ref(), vector, *m, *n, output);
            }
            Self::Q6_K { blocks, m, n } => {
                oxide_quant::simd::gemv_q6_k(blocks.as_ref(), vector, *m, *n, output);
            }
            Self::Q8_0 { blocks, m, n } => {
                assert!(
                    vector.len() >= *n,
                    "Q8_0 gemv: vector.len() ({}) < n ({}) for m={}",
                    vector.len(),
                    *n,
                    *m
                );
                assert!(
                    output.len() >= *m,
                    "Q8_0 gemv: output.len() ({}) < m ({}) for n={}",
                    output.len(),
                    *m,
                    *n
                );
                oxide_quant::simd::gemv_q8_0(blocks.as_ref(), vector, *m, *n, output);
            }
            Self::PTQ1_0 { blocks, m, n } => {
                assert!(
                    vector.len() >= *n,
                    "PTQ1_0 gemv: vector.len() ({}) < n ({}) for m={}",
                    vector.len(),
                    *n,
                    *m
                );
                assert!(
                    output.len() >= *m,
                    "PTQ1_0 gemv: output.len() ({}) < m ({}) for n={}",
                    output.len(),
                    *m,
                    *n
                );
                oxide_quant::simd::gemv_ptq1_0(blocks.as_ref(), vector, *m, *n, output);
            }
        }
    }

    /// Fused 3-way projection (e.g. Q, K, V) over the same normalized input vector.
    pub fn fused_gemv_3way(
        t1: &Self,
        out1: &mut [f32],
        t2: &Self,
        out2: &mut [f32],
        t3: &Self,
        out3: &mut [f32],
        vector: &[f32],
    ) {
        rayon::join(
            || t1.gemv(vector, out1),
            || rayon::join(|| t2.gemv(vector, out2), || t3.gemv(vector, out3)),
        );
    }

    /// Fused 2-way projection (e.g. Gate and Up) over the same normalized input vector.
    pub fn fused_gemv_2way(
        t1: &Self,
        out1: &mut [f32],
        t2: &Self,
        out2: &mut [f32],
        vector: &[f32],
    ) {
        rayon::join(|| t1.gemv(vector, out1), || t2.gemv(vector, out2));
    }

    #[must_use]
    pub fn as_raw_bytes(&self) -> &[u8] {
        match self {
            Self::F32 { data, .. } => {
                // SAFETY: f32 slice trivially reinterprets as u8 slice without alignment fault.
                unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>(),
                        data.len() * std::mem::size_of::<f32>(),
                    )
                }
            }
            Self::Q4_K { blocks, .. } => {
                // SAFETY: BlockQ4_K slice trivially reinterprets as u8 slice without alignment fault.
                unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr().cast::<u8>(),
                        blocks.len() * std::mem::size_of::<oxide_quant::BlockQ4_K>(),
                    )
                }
            }
            Self::Q6_K { blocks, .. } => {
                // SAFETY: BlockQ6_K slice trivially reinterprets as u8 slice without alignment fault.
                unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr().cast::<u8>(),
                        blocks.len() * std::mem::size_of::<oxide_quant::BlockQ6_K>(),
                    )
                }
            }
            Self::Q8_0 { blocks, .. } => {
                // SAFETY: BlockQ8_0 slice trivially reinterprets as u8 slice without alignment fault.
                unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr().cast::<u8>(),
                        blocks.len() * std::mem::size_of::<oxide_quant::BlockQ8_0>(),
                    )
                }
            }
            Self::PTQ1_0 { blocks, .. } => {
                // SAFETY: BlockPTQ1_0 slice trivially reinterprets as u8 slice without alignment fault.
                unsafe {
                    std::slice::from_raw_parts(
                        blocks.as_ptr().cast::<u8>(),
                        blocks.len() * std::mem::size_of::<oxide_quant::BlockPTQ1_0>(),
                    )
                }
            }
        }
    }

    #[must_use]
    pub fn quant_type(&self) -> u32 {
        match self {
            Self::F32 { .. } => 0,
            Self::Q4_K { .. } => 1,
            Self::Q6_K { .. } => 2,
            Self::Q8_0 { .. } => 3,
            Self::PTQ1_0 { .. } => 4,
        }
    }
}

/// Single Transformer Decoder Layer Weights.
#[derive(Debug, Clone)]
pub struct Llama3LayerWeights {
    pub q_proj: QuantizedTensor,    // [num_heads * head_dim, hidden_dim]
    pub k_proj: QuantizedTensor,    // [num_kv_heads * head_dim, hidden_dim]
    pub v_proj: QuantizedTensor,    // [num_kv_heads * head_dim, hidden_dim]
    pub o_proj: QuantizedTensor,    // [hidden_dim, num_heads * head_dim]
    pub gate_proj: QuantizedTensor, // [intermediate_dim, hidden_dim]
    pub up_proj: QuantizedTensor,   // [intermediate_dim, hidden_dim]
    pub down_proj: QuantizedTensor, // [hidden_dim, intermediate_dim]
    pub attn_norm: Vec<f32>,        // [hidden_dim]
    pub ffn_norm: Vec<f32>,         // [hidden_dim]
    pub attn_q_norm: Vec<f32>,      // [head_dim] (QK-Norm)
    pub attn_k_norm: Vec<f32>,      // [head_dim] (QK-Norm)
    pub post_attn_norm: Vec<f32>,   // [hidden_dim] (Gemma 2 / Gemma 4 / Nemotron)
    pub post_ffn_norm: Vec<f32>,    // [hidden_dim] (Gemma 2 / Gemma 4 / Nemotron)
}

impl Llama3LayerWeights {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            q_proj: QuantizedTensor::empty(),
            k_proj: QuantizedTensor::empty(),
            v_proj: QuantizedTensor::empty(),
            o_proj: QuantizedTensor::empty(),
            gate_proj: QuantizedTensor::empty(),
            up_proj: QuantizedTensor::empty(),
            down_proj: QuantizedTensor::empty(),
            attn_norm: Vec::new(),
            ffn_norm: Vec::new(),
            attn_q_norm: Vec::new(),
            attn_k_norm: Vec::new(),
            post_attn_norm: Vec::new(),
            post_ffn_norm: Vec::new(),
        }
    }

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
            q_proj: QuantizedTensor::from_f32(
                fill_weight(q_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.1),
                q_dim,
                h,
            ),
            k_proj: QuantizedTensor::from_f32(
                fill_weight(kv_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.2),
                kv_dim,
                h,
            ),
            v_proj: QuantizedTensor::from_f32(
                fill_weight(kv_dim * h, 1.0 / (h as f32).sqrt(), seed + 0.3),
                kv_dim,
                h,
            ),
            o_proj: QuantizedTensor::from_f32(
                fill_weight(h * q_dim, 1.0 / (q_dim as f32).sqrt(), seed + 0.4),
                h,
                q_dim,
            ),
            gate_proj: QuantizedTensor::from_f32(
                fill_weight(inter * h, 1.0 / (h as f32).sqrt(), seed + 0.5),
                inter,
                h,
            ),
            up_proj: QuantizedTensor::from_f32(
                fill_weight(inter * h, 1.0 / (h as f32).sqrt(), seed + 0.6),
                inter,
                h,
            ),
            down_proj: QuantizedTensor::from_f32(
                fill_weight(h * inter, 1.0 / (inter as f32).sqrt(), seed + 0.7),
                h,
                inter,
            ),
            attn_norm: vec![1.0; h],
            ffn_norm: vec![1.0; h],
            attn_q_norm: Vec::new(),
            attn_k_norm: Vec::new(),
            post_attn_norm: Vec::new(),
            post_ffn_norm: Vec::new(),
        }
    }
}

/// Fully Executable LLaMA 3.1 Model with GQA, RoPE, SwiGLU, and FlashAttention.
#[derive(Debug, Clone)]
pub struct Llama3Model {
    pub config: Llama3Config,
    pub rope: RopeScalingEngine,
    pub rope_swa: Option<RopeScalingEngine>,
    pub flash_attn: FlashAttentionEngine,
    pub token_embedding: Vec<f32>, // [vocab_size, hidden_dim]
    pub token_embd_tensor: Option<QuantizedTensor>,
    pub layers: Vec<Llama3LayerWeights>,
    pub output_norm: Vec<f32>,    // [hidden_dim]
    pub lm_head: QuantizedTensor, // [vocab_size, hidden_dim]
    pub mmap_handle: Option<std::sync::Arc<memmap2::Mmap>>,
}

impl Llama3Model {
    #[must_use]
    pub fn new(config: Llama3Config) -> Self {
        let scaling = if config.is_gemma || config.is_qwen {
            RopeScalingType::Default
        } else {
            RopeScalingType::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 8192,
            }
        };
        let rope_dim = config.head_dim_full.unwrap_or(config.head_dim);
        let rope_config = RopeConfig {
            head_dim: rope_dim,
            base_theta: config.rope_theta_f32(),
            scaling,
            max_position_embeddings: config.max_seq_len,
        };
        let rope = RopeScalingEngine::new(rope_config);
        let rope_swa = if let Some(swa_theta_bits) = config.rope_theta_swa {
            let swa_theta = f32::from_bits(swa_theta_bits);
            let rope_swa_cfg = RopeConfig {
                head_dim: config.head_dim,
                base_theta: swa_theta,
                scaling: RopeScalingType::Default,
                max_position_embeddings: config.max_seq_len,
            };
            Some(RopeScalingEngine::new(rope_swa_cfg))
        } else {
            None
        };
        let flash_attn_cfg =
            FlashAttentionConfig::new(config.num_heads, config.num_kv_heads, config.head_dim, true);
        let flash_attn = FlashAttentionEngine::new(flash_attn_cfg);

        let h = config.hidden_dim;
        let v = config.vocab_size;

        let token_embedding: Vec<f32> = (0..v * h)
            .map(|i| ((i as f32 * 0.013).cos()) * (1.0 / (h as f32).sqrt()))
            .collect();
        let output_norm = vec![1.0; h];
        let lm_head_data: Vec<f32> = (0..v * h)
            .map(|i| ((i as f32 * 0.019).sin()) * (1.0 / (h as f32).sqrt()))
            .collect();
        let lm_head = QuantizedTensor::from_f32(lm_head_data, v, h);

        let layers = (0..config.num_layers)
            .map(|l| Llama3LayerWeights::new_synthetic(&config, l))
            .collect();

        Self {
            config,
            rope,
            rope_swa,
            flash_attn,
            token_embedding,
            token_embd_tensor: None,
            layers,
            output_norm,
            lm_head,
            mmap_handle: None,
        }
    }

    /// Construct an unpopulated model with pre-allocated layer slots without synthetic tensor allocation.
    #[must_use]
    pub fn new_empty(config: Llama3Config) -> Self {
        let scaling = if config.is_gemma || config.is_qwen {
            RopeScalingType::Default
        } else {
            RopeScalingType::Llama3 {
                factor: 8.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position: 8192,
            }
        };
        let rope_dim = config.head_dim_full.unwrap_or(config.head_dim);
        let rope_config = RopeConfig {
            head_dim: rope_dim,
            base_theta: config.rope_theta_f32(),
            scaling,
            max_position_embeddings: config.max_seq_len,
        };
        let rope = RopeScalingEngine::new(rope_config);
        let rope_swa = if let Some(swa_theta_bits) = config.rope_theta_swa {
            let swa_theta = f32::from_bits(swa_theta_bits);
            let rope_swa_cfg = RopeConfig {
                head_dim: config.head_dim,
                base_theta: swa_theta,
                scaling: RopeScalingType::Default,
                max_position_embeddings: config.max_seq_len,
            };
            Some(RopeScalingEngine::new(rope_swa_cfg))
        } else {
            None
        };
        let flash_attn_cfg =
            FlashAttentionConfig::new(config.num_heads, config.num_kv_heads, config.head_dim, true);
        let flash_attn = FlashAttentionEngine::new(flash_attn_cfg);

        let layers = (0..config.num_layers)
            .map(|_| Llama3LayerWeights::empty())
            .collect();

        Self {
            config,
            rope,
            rope_swa,
            flash_attn,
            token_embedding: Vec::new(),
            token_embd_tensor: None,
            layers,
            output_norm: Vec::new(),
            lm_head: QuantizedTensor::empty(),
            mmap_handle: None,
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
    pub fn extract_gguf_tensor_to_buffer(
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
                    use rayon::prelude::*;
                    let count = target.len().min(avail.len() / 2);
                    target[..count].par_chunks_mut(1024).enumerate().for_each(
                        |(chunk_idx, chunk)| {
                            let base_idx = chunk_idx * 1024;
                            for (j, val) in chunk.iter_mut().enumerate() {
                                let i = base_idx + j;
                                let raw = u16::from_le_bytes([avail[i * 2], avail[i * 2 + 1]]);
                                *val = oxide_quant::f16(raw).to_f32();
                            }
                        },
                    );
                }
                crate::formats::GgufQuantType::Q8_0 => {
                    use rayon::prelude::*;
                    let block_size = 34; // 2 bytes f16 scale + 32 bytes int8
                    let num_blocks = (target.len() / 32).min(avail.len() / block_size);
                    target[..num_blocks * 32]
                        .par_chunks_mut(32 * 64)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 64;
                            let n_b = c_slice.len() / 32;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let scale_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                                let scale = oxide_quant::f16(scale_raw).to_f32();
                                let start = b * 32;
                                for i in 0..32 {
                                    let q = block_raw[2 + i] as i8;
                                    c_slice[start + i] = (q as f32) * scale;
                                }
                            }
                        });
                }
                crate::formats::GgufQuantType::Q4_0 => {
                    use rayon::prelude::*;
                    let block_size = 18; // 2 bytes f16 scale + 16 bytes nibbles
                    let num_blocks = (target.len() / 32).min(avail.len() / block_size);
                    target[..num_blocks * 32]
                        .par_chunks_mut(32 * 64)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 64;
                            let n_b = c_slice.len() / 32;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let scale_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                                let scale = oxide_quant::f16(scale_raw).to_f32();
                                let start = b * 32;
                                for i in 0..16 {
                                    let byte = block_raw[2 + i];
                                    let q0 = (byte & 0x0F) as i8 - 8;
                                    let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                                    c_slice[start + i] = (q0 as f32) * scale;
                                    c_slice[start + i + 16] = (q1 as f32) * scale;
                                }
                            }
                        });
                }
                crate::formats::GgufQuantType::Q4_K_M | crate::formats::GgufQuantType::Q4_1 => {
                    use rayon::prelude::*;
                    let block_size = 144; // 2+2+12+128 = 144 bytes per 256 weights
                    let num_blocks = (target.len() / 256).min(avail.len() / block_size);
                    target[..num_blocks * 256]
                        .par_chunks_mut(256 * 16)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 16;
                            let n_b = c_slice.len() / 256;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let d_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                                let dmin_raw = u16::from_le_bytes([block_raw[2], block_raw[3]]);
                                let mut scales = [0u8; 12];
                                scales.copy_from_slice(&block_raw[4..16]);
                                let mut qs = [0u8; 128];
                                qs.copy_from_slice(&block_raw[16..144]);
                                let blk = oxide_quant::BlockQ4_K {
                                    d: oxide_quant::f16(d_raw),
                                    dmin: oxide_quant::f16(dmin_raw),
                                    scales,
                                    qs,
                                };
                                let start = b * 256;
                                let mut out_arr = [0.0f32; 256];
                                blk.dequantize(&mut out_arr);
                                c_slice[start..start + 256].copy_from_slice(&out_arr);
                            }
                        });
                }
                crate::formats::GgufQuantType::Q6_K => {
                    use rayon::prelude::*;
                    let block_size = 210;
                    let num_blocks = (target.len() / 256).min(avail.len() / block_size);
                    target[..num_blocks * 256]
                        .par_chunks_mut(256 * 16)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 16;
                            let n_b = c_slice.len() / 256;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let mut ql = [0u8; 128];
                                ql.copy_from_slice(&block_raw[0..128]);
                                let mut qh = [0u8; 64];
                                qh.copy_from_slice(&block_raw[128..192]);
                                let mut scales = [0i8; 16];
                                for i in 0..16 {
                                    scales[i] = block_raw[192 + i] as i8;
                                }
                                let d_raw = u16::from_le_bytes([block_raw[208], block_raw[209]]);
                                let blk = oxide_quant::BlockQ6_K {
                                    ql,
                                    qh,
                                    scales,
                                    d: oxide_quant::f16(d_raw),
                                };
                                let start = b * 256;
                                let mut out_arr = [0.0f32; 256];
                                blk.dequantize(&mut out_arr);
                                c_slice[start..start + 256].copy_from_slice(&out_arr);
                            }
                        });
                }
                crate::formats::GgufQuantType::Q5_0 => {
                    use rayon::prelude::*;
                    let block_size = 22; // 2 bytes f16 scale + 4 bytes qh + 16 bytes qs
                    let num_blocks = (target.len() / 32).min(avail.len() / block_size);
                    target[..num_blocks * 32]
                        .par_chunks_mut(32 * 64)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 64;
                            let n_b = c_slice.len() / 32;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let d_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                                let d = oxide_quant::f16(d_raw).to_f32();
                                let qh = u32::from_le_bytes([
                                    block_raw[2],
                                    block_raw[3],
                                    block_raw[4],
                                    block_raw[5],
                                ]);
                                let start = b * 32;
                                for i in 0..16 {
                                    let byte = block_raw[6 + i];
                                    let h0 = ((qh >> i) & 1) as i8;
                                    let h1 = ((qh >> (i + 16)) & 1) as i8;
                                    let q0 = (((byte & 0x0F) as i8) | (h0 << 4)) - 16;
                                    let q1 = ((((byte >> 4) & 0x0F) as i8) | (h1 << 4)) - 16;
                                    c_slice[start + i] = (q0 as f32) * d;
                                    c_slice[start + i + 16] = (q1 as f32) * d;
                                }
                            }
                        });
                }
                crate::formats::GgufQuantType::Q5_K_M => {
                    use rayon::prelude::*;
                    let block_size = 176; // BlockQ5_K: 2 + 2 + 12 + 32 + 128 = 176 bytes per 256 weights
                    let num_blocks = (target.len() / 256).min(avail.len() / block_size);
                    target[..num_blocks * 256]
                        .par_chunks_mut(256 * 16)
                        .enumerate()
                        .for_each(|(c_idx, c_slice)| {
                            let base_b = c_idx * 16;
                            let n_b = c_slice.len() / 256;
                            for b in 0..n_b {
                                let b_idx = base_b + b;
                                let block_raw =
                                    &avail[b_idx * block_size..(b_idx + 1) * block_size];
                                let d_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                                let dmin_raw = u16::from_le_bytes([block_raw[2], block_raw[3]]);
                                let mut scales = [0u8; 12];
                                scales.copy_from_slice(&block_raw[4..16]);
                                let mut qh = [0u8; 32];
                                qh.copy_from_slice(&block_raw[16..48]);
                                let mut qs = [0u8; 128];
                                qs.copy_from_slice(&block_raw[48..176]);
                                let blk = oxide_quant::BlockQ5_K {
                                    d: oxide_quant::f16(d_raw),
                                    dmin: oxide_quant::f16(dmin_raw),
                                    scales,
                                    qh,
                                    qs,
                                };
                                let start = b * 256;
                                let mut out_arr = [0.0f32; 256];
                                blk.dequantize(&mut out_arr);
                                c_slice[start..start + 256].copy_from_slice(&out_arr);
                            }
                        });
                }
                _ => {}
            }
        }
    }

    /// Helper to extract a tensor directly into a `QuantizedTensor` with zero-copy mmap slicing.
    fn extract_gguf_tensor(
        gguf: &GgufFile,
        tensor_name: &str,
        data_slice: &'static [u8],
        default_m: usize,
        default_n: usize,
    ) -> Option<QuantizedTensor> {
        let info = gguf.tensors.get(tensor_name)?;
        let offset = gguf.tensor_data_offset + info.offset as usize;
        if offset >= data_slice.len() {
            return None;
        }
        let avail = &data_slice[offset..];

        // Read real tensor dimensions from GGUF metadata table
        let (m, n) = if info.dimensions.len() >= 2 {
            // In GGML/GGUF dimension layout: dimensions[0] is columns (n), dimensions[1] is rows (m)
            (info.dimensions[1] as usize, info.dimensions[0] as usize)
        } else if info.dimensions.len() == 1 {
            (info.dimensions[0] as usize, 1)
        } else {
            (default_m, default_n)
        };

        match info.quant_type {
            crate::formats::GgufQuantType::Q4_K_M | crate::formats::GgufQuantType::Q4_1 => {
                let block_size = 144; // 2+2+12+128 = 144 bytes per 256 weights
                let num_blocks = (m * n) / 256;
                let needed_bytes = num_blocks * block_size;
                if avail.len() >= needed_bytes {
                    let ptr = avail.as_ptr();
                    if (ptr as usize).is_multiple_of(std::mem::align_of::<oxide_quant::BlockQ4_K>())
                    {
                        // SAFETY: Valid pointer with verified alignment, length, and block size
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                ptr.cast::<oxide_quant::BlockQ4_K>(),
                                num_blocks,
                            )
                        };
                        Some(QuantizedTensor::from_q4_k_slice(blocks, m, n))
                    } else {
                        let mut blocks = Vec::with_capacity(num_blocks);
                        for b_idx in 0..num_blocks {
                            let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                            let d_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                            let dmin_raw = u16::from_le_bytes([block_raw[2], block_raw[3]]);
                            let mut scales = [0u8; 12];
                            scales.copy_from_slice(&block_raw[4..16]);
                            let mut qs = [0u8; 128];
                            qs.copy_from_slice(&block_raw[16..144]);
                            blocks.push(oxide_quant::BlockQ4_K {
                                d: oxide_quant::f16(d_raw),
                                dmin: oxide_quant::f16(dmin_raw),
                                scales,
                                qs,
                            });
                        }
                        Some(QuantizedTensor::Q4_K {
                            blocks: std::borrow::Cow::Owned(blocks),
                            m,
                            n,
                        })
                    }
                } else {
                    None
                }
            }
            crate::formats::GgufQuantType::Q6_K => {
                let block_size = 210; // 128 + 64 + 16 + 2
                let num_blocks = (m * n) / 256;
                let needed_bytes = num_blocks * block_size;
                if avail.len() >= needed_bytes {
                    let ptr = avail.as_ptr();
                    if (ptr as usize).is_multiple_of(std::mem::align_of::<oxide_quant::BlockQ6_K>())
                    {
                        // SAFETY: Valid pointer with verified alignment, length, and block size
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                ptr.cast::<oxide_quant::BlockQ6_K>(),
                                num_blocks,
                            )
                        };
                        Some(QuantizedTensor::from_q6_k_slice(blocks, m, n))
                    } else {
                        let mut blocks = Vec::with_capacity(num_blocks);
                        for b_idx in 0..num_blocks {
                            let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                            let mut ql = [0u8; 128];
                            ql.copy_from_slice(&block_raw[0..128]);
                            let mut qh = [0u8; 64];
                            qh.copy_from_slice(&block_raw[128..192]);
                            let mut scales = [0i8; 16];
                            for i in 0..16 {
                                scales[i] = block_raw[192 + i] as i8;
                            }
                            let d_raw = u16::from_le_bytes([block_raw[208], block_raw[209]]);
                            blocks.push(oxide_quant::BlockQ6_K {
                                ql,
                                qh,
                                scales,
                                d: oxide_quant::f16(d_raw),
                            });
                        }
                        Some(QuantizedTensor::Q6_K {
                            blocks: std::borrow::Cow::Owned(blocks),
                            m,
                            n,
                        })
                    }
                } else {
                    None
                }
            }
            crate::formats::GgufQuantType::Q8_0 => {
                let block_size = 34; // 2 + 32
                let num_blocks = (m * n) / 32;
                let needed_bytes = num_blocks * block_size;
                if avail.len() >= needed_bytes {
                    let ptr = avail.as_ptr();
                    if (ptr as usize).is_multiple_of(std::mem::align_of::<oxide_quant::BlockQ8_0>())
                    {
                        // SAFETY: Valid pointer with verified alignment, length, and block size
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                ptr.cast::<oxide_quant::BlockQ8_0>(),
                                num_blocks,
                            )
                        };
                        Some(QuantizedTensor::from_q8_0_slice(blocks, m, n))
                    } else {
                        let mut blocks = Vec::with_capacity(num_blocks);
                        for b_idx in 0..num_blocks {
                            let block_raw = &avail[b_idx * block_size..(b_idx + 1) * block_size];
                            let scale_raw = u16::from_le_bytes([block_raw[0], block_raw[1]]);
                            let mut qs = [0i8; 32];
                            for i in 0..32 {
                                qs[i] = block_raw[2 + i] as i8;
                            }
                            blocks.push(oxide_quant::BlockQ8_0 {
                                scale: oxide_quant::f16(scale_raw),
                                qs,
                            });
                        }
                        Some(QuantizedTensor::Q8_0 {
                            blocks: std::borrow::Cow::Owned(blocks),
                            m,
                            n,
                        })
                    }
                } else {
                    None
                }
            }
            crate::formats::GgufQuantType::PTQ1_0 => {
                let block_size = 28; // 24 (qs) + 2 (qh) + 2 (d) = 28 bytes per 128 weights
                let num_blocks = (m * n) / 128;
                let needed_bytes = num_blocks * block_size;
                if avail.len() >= needed_bytes {
                    let ptr = avail.as_ptr();
                    if (ptr as usize)
                        .is_multiple_of(std::mem::align_of::<oxide_quant::BlockPTQ1_0>())
                    {
                        // SAFETY: Valid pointer with verified alignment, length, and block size
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                ptr.cast::<oxide_quant::BlockPTQ1_0>(),
                                num_blocks,
                            )
                        };
                        Some(QuantizedTensor::from_ptq1_0_slice(blocks, m, n))
                    } else {
                        let mut blocks = Vec::with_capacity(num_blocks);
                        for b_idx in 0..num_blocks {
                            let block_raw =
                                &avail[b_idx * block_size..(b_idx + 1) * block_size];
                            let mut qs = [0u8; 24];
                            qs.copy_from_slice(&block_raw[0..24]);
                            let mut qh = [0u8; 2];
                            qh.copy_from_slice(&block_raw[24..26]);
                            let d_raw = u16::from_le_bytes([block_raw[26], block_raw[27]]);
                            blocks.push(oxide_quant::BlockPTQ1_0 {
                                qs,
                                qh,
                                d: oxide_quant::f16(d_raw),
                            });
                        }
                        Some(QuantizedTensor::PTQ1_0 {
                            blocks: std::borrow::Cow::Owned(blocks),
                            m,
                            n,
                        })
                    }
                } else {
                    None
                }
            }
            _ => {
                // Fallback to F32 decompression if format is uncompressed or unsupported directly
                let mut data = vec![0.0f32; m * n];
                Self::extract_gguf_tensor_to_buffer(gguf, tensor_name, data_slice, &mut data);
                Some(QuantizedTensor::from_f32(data, m, n))
            }
        }
    }

    /// Loads tensor weights from GGUF binary container with zero-copy slicing.
    pub fn load_from_gguf(&mut self, gguf: &GgufFile, data_slice: &'static [u8]) -> Result<()> {
        use rayon::prelude::*;

        // 1. Output norm
        self.output_norm.resize(self.config.hidden_dim, 1.0);
        Self::extract_gguf_tensor_to_buffer(
            gguf,
            "output_norm.weight",
            data_slice,
            &mut self.output_norm,
        );

        // 1b. Embeddings: Zero-copy slicing if large, or inflate if small
        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;
        if v * h <= 32_000_000 {
            self.token_embedding.resize(v * h, 0.0);
            Self::extract_gguf_tensor_to_buffer(
                gguf,
                "token_embd.weight",
                data_slice,
                &mut self.token_embedding,
            );
            self.token_embd_tensor = None;
        } else {
            self.token_embedding.clear();
            self.token_embd_tensor =
                Self::extract_gguf_tensor(gguf, "token_embd.weight", data_slice, v, h);
        }

        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;

        // 3. LM Head (if distinct from embeddings)
        if let Some(t) = Self::extract_gguf_tensor(gguf, "output.weight", data_slice, v, h) {
            self.lm_head = t;
        } else if let Some(t) =
            Self::extract_gguf_tensor(gguf, "token_embd.weight", data_slice, v, h)
        {
            // Tied weights: token_embd.weight serves as lm_head
            self.lm_head = t;
        } else {
            self.lm_head = QuantizedTensor::from_f32(self.token_embedding.clone(), v, h);
        }
        let q_dim = self.config.num_heads * self.config.head_dim;
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let inter = self.config.intermediate_dim;

        // 4. Transformer Decoder Layers loaded in parallel across all CPU cores
        self.layers
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, layer)| {
                let q_name = format!("blk.{i}.attn_q.weight");
                let k_name = format!("blk.{i}.attn_k.weight");
                let v_name = format!("blk.{i}.attn_v.weight");
                let o_name = format!("blk.{i}.attn_output.weight");
                let gate_name = format!("blk.{i}.ffn_gate.weight");
                let up_name = format!("blk.{i}.ffn_up.weight");
                let down_name = format!("blk.{i}.ffn_down.weight");
                let attn_norm_name = format!("blk.{i}.attn_norm.weight");
                let ffn_norm_name = format!("blk.{i}.ffn_norm.weight");

                if let Some(t) = Self::extract_gguf_tensor(gguf, &q_name, data_slice, q_dim, h) {
                    layer.q_proj = t;
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &k_name, data_slice, kv_dim, h) {
                    layer.k_proj = t;
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &v_name, data_slice, kv_dim, h) {
                    layer.v_proj = t;
                }
                // Handle fused QKV for models where Q, K, V are combined in blk.{i}.attn_qkv.weight
                if layer.q_proj.rows() == 0 {
                    let qkv_name = format!("blk.{i}.attn_qkv.weight");
                    if let Some(qkv) = Self::extract_gguf_tensor(
                        gguf,
                        &qkv_name,
                        data_slice,
                        q_dim + 2 * kv_dim,
                        h,
                    ) {
                        if qkv.rows() == q_dim + 2 * kv_dim {
                            if let Some(q) = qkv.slice_rows(0, q_dim) {
                                layer.q_proj = q;
                            }
                            if let Some(k) = qkv.slice_rows(q_dim, kv_dim) {
                                layer.k_proj = k;
                            }
                            if let Some(v) = qkv.slice_rows(q_dim + kv_dim, kv_dim) {
                                layer.v_proj = v;
                            }
                        }
                    }
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &o_name, data_slice, h, q_dim) {
                    layer.o_proj = t;
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &gate_name, data_slice, inter, h) {
                    layer.gate_proj = t;
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &up_name, data_slice, inter, h) {
                    layer.up_proj = t;
                }
                if let Some(t) = Self::extract_gguf_tensor(gguf, &down_name, data_slice, h, inter) {
                    layer.down_proj = t;
                }

                layer.attn_norm.resize(h, 1.0);
                Self::extract_gguf_tensor_to_buffer(
                    gguf,
                    &attn_norm_name,
                    data_slice,
                    &mut layer.attn_norm,
                );
                layer.ffn_norm.resize(h, 1.0);
                Self::extract_gguf_tensor_to_buffer(
                    gguf,
                    &ffn_norm_name,
                    data_slice,
                    &mut layer.ffn_norm,
                );

                let q_norm_name = format!("blk.{i}.attn_q_norm.weight");
                let k_norm_name = format!("blk.{i}.attn_k_norm.weight");
                let post_attn_norm_name = format!("blk.{i}.post_attention_norm.weight");
                let post_ffw_norm_name = format!("blk.{i}.post_ffw_norm.weight");

                if let Some(t) = gguf.tensors.get(&q_norm_name) {
                    let len = t.dimensions[0] as usize;
                    layer.attn_q_norm.resize(len, 1.0);
                    Self::extract_gguf_tensor_to_buffer(
                        gguf,
                        &q_norm_name,
                        data_slice,
                        &mut layer.attn_q_norm,
                    );
                }
                if let Some(t) = gguf.tensors.get(&k_norm_name) {
                    let len = t.dimensions[0] as usize;
                    layer.attn_k_norm.resize(len, 1.0);
                    Self::extract_gguf_tensor_to_buffer(
                        gguf,
                        &k_norm_name,
                        data_slice,
                        &mut layer.attn_k_norm,
                    );
                }
                if let Some(t) = gguf.tensors.get(&post_attn_norm_name) {
                    let len = t.dimensions[0] as usize;
                    layer.post_attn_norm.resize(len, 1.0);
                    Self::extract_gguf_tensor_to_buffer(
                        gguf,
                        &post_attn_norm_name,
                        data_slice,
                        &mut layer.post_attn_norm,
                    );
                }
                if let Some(t) = gguf.tensors.get(&post_ffw_norm_name) {
                    let len = t.dimensions[0] as usize;
                    layer.post_ffn_norm.resize(len, 1.0);
                    Self::extract_gguf_tensor_to_buffer(
                        gguf,
                        &post_ffw_norm_name,
                        data_slice,
                        &mut layer.post_ffn_norm,
                    );
                }
            });

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

        let mmap = std::sync::Arc::new(mmap);
        // SAFETY: mmap is kept alive by `model.mmap_handle` for the lifetime of the returned model.
        let data_slice: &'static [u8] =
            unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("gguf")
            || (data_slice.len() >= 4 && &data_slice[0..4] == crate::formats::GGUF_MAGIC)
        {
            let gguf = GgufFile::parse(data_slice)?;
            let cfg = Llama3Config::from_gguf(&gguf);
            let mut model = Self::new_empty(cfg);
            model.mmap_handle = Some(mmap);
            model.load_from_gguf(&gguf, data_slice)?;
            Ok(model)
        } else if ext.eq_ignore_ascii_case("safetensors") {
            let (header, _) = SafeTensorsHeader::parse_from_bytes(data_slice)?;
            let filename = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let cfg = if let Some(spec) = crate::registry::ModelSpecification::lookup(filename) {
                Llama3Config::from_spec(&spec)
            } else {
                Llama3Config::default()
            };
            let mut model = Self::new_empty(cfg);
            model.mmap_handle = Some(mmap);
            model.load_from_safetensors(&header, data_slice)?;
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
        let mut scratch = Llama3ScratchBuffers::new(&self.config);
        for layer in &self.layers {
            let q_dim = layer.q_proj.rows().max(layer.o_proj.cols());
            if q_dim > scratch.q.len() {
                scratch.q.resize(q_dim, 0.0);
                scratch.attn_out.resize(q_dim, 0.0);
            }
            let k_dim = layer.k_proj.rows();
            if k_dim > scratch.k.len() {
                scratch.k.resize(k_dim, 0.0);
                scratch.v.resize(k_dim, 0.0);
            }
            let inter = layer
                .gate_proj
                .rows()
                .max(layer.up_proj.rows())
                .max(layer.down_proj.cols());
            if inter > scratch.gate.len() {
                scratch.gate.resize(inter, 0.0);
                scratch.up.resize(inter, 0.0);
                scratch.activated.resize(inter, 0.0);
            }
        }
        scratch
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
        if let Some(ref emb_tensor) = self.token_embd_tensor {
            emb_tensor.extract_row(tok_idx, &mut scratch.hidden[..h]);
        } else {
            let emb_offset = tok_idx * h;
            if emb_offset + h <= self.token_embedding.len() {
                scratch
                    .hidden
                    .copy_from_slice(&self.token_embedding[emb_offset..emb_offset + h]);
            } else {
                scratch.hidden.fill(0.0);
            }
        }

        // Gemma embedding scaling by sqrt(hidden_dim)
        if self.config.is_gemma {
            let scale = (h as f32).sqrt();
            for x in &mut scratch.hidden[..h] {
                *x *= scale;
            }
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
            &scratch.hidden[..h],
            &self.output_norm,
            &mut scratch.final_norm[..h],
            eps,
            self.config.is_gemma,
        );

        // 4. LM Head projection to vocabulary logits
        self.lm_head
            .gemv(&scratch.final_norm[..h], &mut scratch.logits);

        if let Some(cap) = self.config.final_logit_softcapping_f32() {
            if cap > 0.0 {
                for l in &mut scratch.logits {
                    *l = cap * (*l / cap).tanh();
                }
            }
        }

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
        _q_dim: usize,
        _kv_dim: usize,
    ) {
        let cur_q_dim = layer.q_proj.rows();
        let cur_k_dim = layer.k_proj.rows();
        let cur_v_dim = layer.v_proj.rows();

        if cur_q_dim > 0 && cur_k_dim > 0 && cur_v_dim > 0 {
            let cur_heads = self.config.num_heads;
            let cur_kv_heads = self.config.num_kv_heads;
            let cur_head_dim = if cur_heads > 0 {
                cur_q_dim / cur_heads
            } else {
                self.config.head_dim
            };

            let is_swa =
                cur_head_dim <= self.config.head_dim && self.config.sliding_window.is_some();
            let sliding_window = if is_swa {
                self.config.sliding_window
            } else {
                None
            };
            let rope_engine = if cur_head_dim <= self.config.head_dim && self.rope_swa.is_some() {
                self.rope_swa.as_ref().unwrap()
            } else {
                &self.rope
            };

            Self::rms_norm(
                &scratch.hidden[..h],
                &layer.attn_norm,
                &mut scratch.norm_hidden[..h],
                eps,
                self.config.is_gemma,
            );

            // Fused Q, K, V Projections via SIMD GEMV
            QuantizedTensor::fused_gemv_3way(
                &layer.q_proj,
                &mut scratch.q,
                &layer.k_proj,
                &mut scratch.k,
                &layer.v_proj,
                &mut scratch.v,
                &scratch.norm_hidden[..h],
            );

            // QK-Norm (applied per-head before RoPE)
            let hd = cur_head_dim;
            if !layer.attn_q_norm.is_empty() && layer.attn_q_norm.len() >= hd {
                let mut tmp = [0.0f32; 512];
                if hd <= tmp.len() {
                    for head_idx in 0..cur_heads {
                        let start = head_idx * hd;
                        let end = start + hd;
                        if end <= scratch.q.len() {
                            tmp[..hd].copy_from_slice(&scratch.q[start..end]);
                            Self::rms_norm(
                                &tmp[..hd],
                                &layer.attn_q_norm[..hd],
                                &mut scratch.q[start..end],
                                eps,
                                false,
                            );
                        }
                    }
                }
            }
            if !layer.attn_k_norm.is_empty() && layer.attn_k_norm.len() >= hd {
                let mut tmp = [0.0f32; 512];
                if hd <= tmp.len() {
                    for kv_head_idx in 0..cur_kv_heads {
                        let start = kv_head_idx * hd;
                        let end = start + hd;
                        if end <= scratch.k.len() {
                            tmp[..hd].copy_from_slice(&scratch.k[start..end]);
                            Self::rms_norm(
                                &tmp[..hd],
                                &layer.attn_k_norm[..hd],
                                &mut scratch.k[start..end],
                                eps,
                                false,
                            );
                        }
                    }
                }
            }

            // RoPE Rotary Embedding
            for head_idx in 0..cur_heads {
                let start = head_idx * cur_head_dim;
                let end = start + cur_head_dim;
                if end <= scratch.q.len() {
                    rope_engine.apply_rotary_in_place(&mut scratch.q[start..end], position);
                }
            }
            for kv_head_idx in 0..cur_kv_heads {
                let start = kv_head_idx * cur_head_dim;
                let end = start + cur_head_dim;
                if end <= scratch.k.len() {
                    rope_engine.apply_rotary_in_place(&mut scratch.k[start..end], position);
                }
            }

            // KV Cache append
            if let Some(layer_cache) = kv_cache.get_mut(layer_idx) {
                layer_cache.append_kv(&scratch.k[..cur_k_dim], &scratch.v[..cur_v_dim]);
            }

            let context_len = position + 1;
            let (k_buf, v_buf) = if let Some(layer_cache) = kv_cache.get(layer_idx) {
                (&layer_cache.k[..], &layer_cache.v[..])
            } else {
                (&scratch.k[..cur_k_dim], &scratch.v[..cur_v_dim])
            };

            // Attention Output Projection with GQA over full historical KV context
            for head in 0..cur_heads {
                let q_start = head * cur_head_dim;
                let q_slice = &scratch.q[q_start..q_start + cur_head_dim];
                let out_slice = &mut scratch.attn_out[q_start..q_start + cur_head_dim];

                let kv_head = head / (cur_heads / cur_kv_heads.max(1)).max(1);

                self.flash_attn.forward_decode_gqa_dyn(
                    q_slice,
                    k_buf,
                    v_buf,
                    kv_head,
                    cur_kv_heads,
                    cur_head_dim,
                    context_len,
                    sliding_window,
                    out_slice,
                );
            }

            let o_in_dim = layer.o_proj.cols();
            layer
                .o_proj
                .gemv(&scratch.attn_out[..o_in_dim], &mut scratch.o_proj_out[..h]);

            // Post-attention norm (Gemma 2 / Gemma 4 / Nemotron)
            if !layer.post_attn_norm.is_empty() && layer.post_attn_norm.len() >= h {
                let mut tmp = [0.0f32; 4096];
                if h <= tmp.len() {
                    tmp[..h].copy_from_slice(&scratch.o_proj_out[..h]);
                    Self::rms_norm(
                        &tmp[..h],
                        &layer.post_attn_norm[..h],
                        &mut scratch.o_proj_out[..h],
                        eps,
                        self.config.is_gemma,
                    );
                }
            }

            // Residual 1
            for i in 0..h {
                scratch.hidden[i] += scratch.o_proj_out[i];
            }
        }

        // FFN RMSNorm & MLP
        let cur_inter = layer.gate_proj.rows();
        if cur_inter > 0 && layer.up_proj.rows() > 0 && layer.down_proj.rows() > 0 {
            Self::rms_norm(
                &scratch.hidden[..h],
                &layer.ffn_norm,
                &mut scratch.ffn_norm_hidden[..h],
                eps,
                self.config.is_gemma,
            );

            QuantizedTensor::fused_gemv_2way(
                &layer.gate_proj,
                &mut scratch.gate[..cur_inter],
                &layer.up_proj,
                &mut scratch.up[..cur_inter],
                &scratch.ffn_norm_hidden[..h],
            );

            if self.config.is_gemma {
                // Gemma GeLU activation (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                for i in 0..cur_inter {
                    let g = scratch.gate[i];
                    let inner = SQRT_2_OVER_PI * (g + 0.044_715 * g * g * g);
                    let gelu_g = 0.5 * g * (1.0 + inner.tanh());
                    scratch.activated[i] = gelu_g * scratch.up[i];
                }
            } else {
                // SwiGLU: silu(gate) * up
                for i in 0..cur_inter {
                    let g = scratch.gate[i];
                    let silu_g = g / (1.0 + (-g).exp());
                    scratch.activated[i] = silu_g * scratch.up[i];
                }
            }

            let d_cols = layer.down_proj.cols().min(cur_inter);
            layer
                .down_proj
                .gemv(&scratch.activated[..d_cols], &mut scratch.mlp_out[..h]);

            // Post-FFN norm (Gemma 2 / Gemma 4 / Nemotron)
            if !layer.post_ffn_norm.is_empty() && layer.post_ffn_norm.len() >= h {
                let mut tmp = [0.0f32; 4096];
                if h <= tmp.len() {
                    tmp[..h].copy_from_slice(&scratch.mlp_out[..h]);
                    Self::rms_norm(
                        &tmp[..h],
                        &layer.post_ffn_norm[..h],
                        &mut scratch.mlp_out[..h],
                        eps,
                        self.config.is_gemma,
                    );
                }
            }

            // Residual 2
            for i in 0..h {
                scratch.hidden[i] += scratch.mlp_out[i];
            }
        }
    }

    /// Forward pass through a single transformer decoder layer using Paged KV-Cache Arena.
    /// Performs ZERO heap allocations during decode step.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_paged(
        &self,
        layer_idx: usize,
        layer: &Llama3LayerWeights,
        position: usize,
        paged_arena: &mut oxide_alloc::PagedKvArena,
        block_table: &[u32],
        token_offset_in_current_block: usize,
        current_block_id: u32,
        scratch: &mut Llama3ScratchBuffers,
        eps: f32,
        h: usize,
        _q_dim: usize,
        _kv_dim: usize,
    ) {
        Self::rms_norm(
            &scratch.hidden,
            &layer.attn_norm,
            &mut scratch.norm_hidden,
            eps,
            self.config.is_gemma,
        );

        // Fused Q, K, V Projections via SIMD GEMV
        QuantizedTensor::fused_gemv_3way(
            &layer.q_proj,
            &mut scratch.q,
            &layer.k_proj,
            &mut scratch.k,
            &layer.v_proj,
            &mut scratch.v,
            &scratch.norm_hidden,
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

        // Write token into physical page block (ZERO dynamic allocations)
        paged_arena.write_token_kv(
            layer_idx,
            current_block_id,
            token_offset_in_current_block,
            &scratch.k,
            &scratch.v,
        );

        let context_len = position + 1;

        // Attention Output Projection with GQA over Paged KV Cache
        for head in 0..self.config.num_heads {
            let q_start = head * self.config.head_dim;
            let q_slice = &scratch.q[q_start..q_start + self.config.head_dim];
            let out_slice = &mut scratch.attn_out[q_start..q_start + self.config.head_dim];

            let kv_head = head / (self.config.num_heads / self.config.num_kv_heads);

            self.flash_attn.forward_decode_paged_gqa(
                q_slice,
                paged_arena,
                layer_idx,
                block_table,
                kv_head,
                self.config.num_kv_heads,
                context_len,
                out_slice,
            );
        }

        layer
            .o_proj
            .gemv(&scratch.attn_out, &mut scratch.o_proj_out);

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
            self.config.is_gemma,
        );

        let inter_dim = self.config.intermediate_dim;
        QuantizedTensor::fused_gemv_2way(
            &layer.gate_proj,
            &mut scratch.gate,
            &layer.up_proj,
            &mut scratch.up,
            &scratch.ffn_norm_hidden,
        );

        // SwiGLU: down_proj(silu(gate) * up)
        for i in 0..inter_dim {
            let g = scratch.gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            scratch.activated[i] = silu_g * scratch.up[i];
        }

        layer
            .down_proj
            .gemv(&scratch.activated, &mut scratch.mlp_out);

        // Residual 2
        for i in 0..h {
            scratch.hidden[i] += scratch.mlp_out[i];
        }
    }

    /// Autoregressive forward step utilizing zero-allocation Paged KV-Cache Arena.
    pub fn forward_step_paged(
        &self,
        token_id: u32,
        position: usize,
        paged_arena: &mut oxide_alloc::PagedKvArena,
        block_table: &mut oxide_alloc::SequenceBlockTable,
        scratch: &mut Llama3ScratchBuffers,
    ) -> Result<()> {
        let (current_block_id, token_offset_in_block) = block_table
            .advance_token(paged_arena)
            .map_err(|e| oxide_core::error::EngineError::BackendError(e.to_string()))?;

        let h = self.config.hidden_dim;
        let tok_idx = (token_id as usize) % self.config.vocab_size;

        // Embedding lookup
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

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_paged(
                layer_idx,
                layer,
                position,
                paged_arena,
                &block_table.block_ids,
                token_offset_in_block,
                current_block_id,
                scratch,
                eps,
                h,
                q_dim,
                kv_dim,
            );
        }

        Self::rms_norm(
            &scratch.hidden,
            &self.output_norm,
            &mut scratch.final_norm,
            eps,
            self.config.is_gemma,
        );

        // 4. LM Head projection to vocabulary logits
        self.lm_head.gemv(&scratch.final_norm, &mut scratch.logits);

        Ok(())
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
    fn rms_norm(input: &[f32], weight: &[f32], output: &mut [f32], eps: f32, is_gemma: bool) {
        if is_gemma {
            oxide_quant::simd::rmsnorm_f32_offset(input, weight, output, eps);
        } else {
            oxide_quant::simd::rmsnorm_f32(input, weight, output, eps);
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
        let max_hd = config.max_head_dim();
        let q_dim = config.num_heads * max_hd;
        let kv_dim = config.num_kv_heads * max_hd;
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

#[cfg(test)]
mod tests_size {
    use super::*;

    #[test]
    fn test_print_sizes() {
        println!(
            "BlockQ4_K size: {}",
            std::mem::size_of::<oxide_quant::BlockQ4_K>()
        );
        println!(
            "BlockQ6_K size: {}",
            std::mem::size_of::<oxide_quant::BlockQ6_K>()
        );
        println!(
            "QuantizedTensor size: {}",
            std::mem::size_of::<QuantizedTensor>()
        );
    }
}
