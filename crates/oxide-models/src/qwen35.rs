//! Native Qwen 3.5 / Ternary Bonsai 2 Hybrid Architecture Model.
//!
//! Implements the hybrid architecture alternating Gated Delta Net (GDN / Linear Recurrent Attention)
//! with full Multi-Head Attention, utilizing branchless PTQ1_0 ternary arithmetic and In-SMem FWHT.

use crate::formats::GgufFile;
use crate::llama3::{Llama3Model, QuantizedTensor};
use oxide_core::error::{EngineError, Result};
use oxide_quant::simd::dot_f32;

/// Configuration for Qwen 3.5 / Ternary Bonsai 2 models.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35Config {
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_d_state: usize,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self {
            hidden_dim: 5120,
            intermediate_dim: 17408,
            num_layers: 64,
            num_heads: 24,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 248320,
            rms_norm_eps: 1e-6,
            ssm_conv_kernel: 4,
            ssm_d_state: 48,
        }
    }
}

impl Qwen35Config {
    /// Extracts Qwen 3.5 configuration from GGUF metadata.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture();
        let prefix = if gguf.get_u32(&format!("{arch}.block_count")).is_some() {
            arch
        } else {
            "qwen35"
        };

        let hidden_dim = gguf
            .get_u32(&format!("{prefix}.embedding_length"))
            .unwrap_or(5120) as usize;
        let intermediate_dim = gguf
            .get_u32(&format!("{prefix}.feed_forward_length"))
            .unwrap_or(17408) as usize;
        let num_layers = gguf
            .get_u32(&format!("{prefix}.block_count"))
            .unwrap_or(64) as usize;
        let num_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count"))
            .unwrap_or(24) as usize;
        let num_kv_heads = gguf
            .get_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(4) as usize;
        let head_dim = gguf
            .get_u32(&format!("{prefix}.attention.key_length"))
            .unwrap_or(128) as usize;
        let vocab_size = gguf
            .get_u32(&format!("{prefix}.vocab_size"))
            .or_else(|| gguf.get_u32("tokenizer.ggml.tokens"))
            .unwrap_or(248320) as usize;
        let rms_norm_eps = gguf
            .get_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        Self {
            hidden_dim,
            intermediate_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            ssm_conv_kernel: 4,
            ssm_d_state: 48,
        }
    }
}

/// A single transformer layer in Qwen 3.5: either a Recurrent Gated Delta Net or Full Attention.
#[derive(Debug)]
pub struct Qwen35Layer {
    pub is_recurrent: bool,
    pub attn_norm: Vec<f32>,

    // Recurrent linear attention weights (if is_recurrent == true)
    pub wqkv: Option<QuantizedTensor>,
    pub wgate: Option<QuantizedTensor>,
    pub conv1d: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_norm: Vec<f32>,
    pub ssm_out: Option<QuantizedTensor>,

    // Full softmax attention weights (if is_recurrent == false)
    pub q_proj: Option<QuantizedTensor>,
    pub k_proj: Option<QuantizedTensor>,
    pub v_proj: Option<QuantizedTensor>,
    pub o_proj: Option<QuantizedTensor>,

    // Feed-forward network (FFN)
    pub post_attn_norm: Vec<f32>,
    pub ffn_gate: QuantizedTensor,
    pub ffn_up: QuantizedTensor,
    pub ffn_down: QuantizedTensor,
}

/// Full Qwen 3.5 / Ternary Bonsai 2 Hybrid Model.
#[derive(Debug)]
pub struct Qwen35Model {
    pub config: Qwen35Config,
    pub token_embd: QuantizedTensor,
    pub layers: Vec<Qwen35Layer>,
    pub output_norm: Vec<f32>,
    pub output: QuantizedTensor,
}

impl Qwen35Model {
    /// Loads a Qwen 3.5 / Ternary Bonsai 2 model from parsed GGUF metadata and static data buffer.
    pub fn load_from_gguf(gguf: &GgufFile, data_slice: &'static [u8]) -> Result<Self> {
        let config = Qwen35Config::from_gguf(gguf);
        let h = config.hidden_dim;
        let v = config.vocab_size;
        let inter = config.intermediate_dim;

        // Load token embedding
        let token_embd = Llama3Model::extract_gguf_tensor(gguf, "token_embd.weight", data_slice, v, h)
            .ok_or_else(|| EngineError::ModelNotFound("token_embd.weight".to_string()))?;

        // Load output norm
        let mut output_norm = vec![1.0f32; h];
        Llama3Model::extract_gguf_tensor_to_buffer(gguf, "output_norm.weight", data_slice, &mut output_norm);

        // Load output head
        let output = Llama3Model::extract_gguf_tensor(gguf, "output.weight", data_slice, v, h)
            .or_else(|| Llama3Model::extract_gguf_tensor(gguf, "token_embd.weight", data_slice, v, h))
            .ok_or_else(|| EngineError::ModelNotFound("output.weight".to_string()))?;

        let mut layers = Vec::with_capacity(config.num_layers);

        for il in 0..config.num_layers {
            let mut attn_norm = vec![1.0f32; h];
            Llama3Model::extract_gguf_tensor_to_buffer(
                gguf,
                &format!("blk.{il}.attn_norm.weight"),
                data_slice,
                &mut attn_norm,
            );

            let mut post_attn_norm = vec![1.0f32; h];
            Llama3Model::extract_gguf_tensor_to_buffer(
                gguf,
                &format!("blk.{il}.post_attention_norm.weight"),
                data_slice,
                &mut post_attn_norm,
            );

            // Load FFN tensors
            let ffn_gate = Llama3Model::extract_gguf_tensor(
                gguf,
                &format!("blk.{il}.ffn_gate.weight"),
                data_slice,
                inter,
                h,
            ).ok_or_else(|| EngineError::ModelNotFound(format!("blk.{il}.ffn_gate.weight")))?;

            let ffn_up = Llama3Model::extract_gguf_tensor(
                gguf,
                &format!("blk.{il}.ffn_up.weight"),
                data_slice,
                inter,
                h,
            ).ok_or_else(|| EngineError::ModelNotFound(format!("blk.{il}.ffn_up.weight")))?;

            let ffn_down = Llama3Model::extract_gguf_tensor(
                gguf,
                &format!("blk.{il}.ffn_down.weight"),
                data_slice,
                h,
                inter,
            ).ok_or_else(|| EngineError::ModelNotFound(format!("blk.{il}.ffn_down.weight")))?;

            // Check if layer is recurrent
            let is_recurrent = gguf.tensors.contains_key(&format!("blk.{il}.ssm_conv1d.weight"));

            if is_recurrent {
                let wqkv = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_qkv.weight"),
                    data_slice,
                    h,
                    h,
                );
                let wgate = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_gate.weight"),
                    data_slice,
                    h,
                    h,
                );
                let ssm_out = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.ssm_out.weight"),
                    data_slice,
                    h,
                    h,
                );

                let mut conv1d = vec![0.0f32; config.ssm_conv_kernel * h];
                Llama3Model::extract_gguf_tensor_to_buffer(
                    gguf,
                    &format!("blk.{il}.ssm_conv1d.weight"),
                    data_slice,
                    &mut conv1d,
                );

                let mut ssm_dt_bias = vec![0.0f32; config.ssm_d_state];
                Llama3Model::extract_gguf_tensor_to_buffer(
                    gguf,
                    &format!("blk.{il}.ssm_dt.bias"),
                    data_slice,
                    &mut ssm_dt_bias,
                );

                let mut ssm_a = vec![0.0f32; config.ssm_d_state];
                Llama3Model::extract_gguf_tensor_to_buffer(
                    gguf,
                    &format!("blk.{il}.ssm_a"),
                    data_slice,
                    &mut ssm_a,
                );

                let mut ssm_norm = vec![1.0f32; h];
                Llama3Model::extract_gguf_tensor_to_buffer(
                    gguf,
                    &format!("blk.{il}.ssm_norm.weight"),
                    data_slice,
                    &mut ssm_norm,
                );

                layers.push(Qwen35Layer {
                    is_recurrent: true,
                    attn_norm,
                    wqkv,
                    wgate,
                    conv1d,
                    ssm_dt_bias,
                    ssm_a,
                    ssm_norm,
                    ssm_out,
                    q_proj: None,
                    k_proj: None,
                    v_proj: None,
                    o_proj: None,
                    post_attn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                });
            } else {
                let q_proj = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_q.weight"),
                    data_slice,
                    config.num_heads * config.head_dim,
                    h,
                );
                let k_proj = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_k.weight"),
                    data_slice,
                    config.num_kv_heads * config.head_dim,
                    h,
                );
                let v_proj = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_v.weight"),
                    data_slice,
                    config.num_kv_heads * config.head_dim,
                    h,
                );
                let o_proj = Llama3Model::extract_gguf_tensor(
                    gguf,
                    &format!("blk.{il}.attn_output.weight"),
                    data_slice,
                    h,
                    config.num_heads * config.head_dim,
                );

                layers.push(Qwen35Layer {
                    is_recurrent: false,
                    attn_norm,
                    wqkv: None,
                    wgate: None,
                    conv1d: Vec::new(),
                    ssm_dt_bias: Vec::new(),
                    ssm_a: Vec::new(),
                    ssm_norm: Vec::new(),
                    ssm_out: None,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    post_attn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                });
            }
        }

        Ok(Self {
            config,
            token_embd,
            layers,
            output_norm,
            output,
        })
    }

    /// RMSNorm helper.
    #[inline]
    pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
        let n = x.len();
        let sum_sq = dot_f32(x, x);
        let scale = 1.0 / (sum_sq / (n as f32) + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] * scale * weight[i];
        }
    }
}
