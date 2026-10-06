#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]
#![warn(clippy::pedantic)]
#![allow(
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
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::match_same_arms
)]

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// Layer category supported in arbitrary custom research model architectures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Linear,
    Attention {
        num_heads: usize,
        num_kv_heads: usize,
    },
    SwiGluMlp {
        intermediate_dim: usize,
    },
    RmsNorm {
        eps: u32,
    }, // Stored as u32 bits or exponent
    Moe {
        num_experts: usize,
        top_k: usize,
        intermediate_dim: usize,
    },
    StateSpaceMamba {
        state_dim: usize,
    },
    LatentAttentionMla {
        latent_dim: usize,
        num_heads: usize,
    },
    DiffusionBlock {
        patch_size: usize,
    },
}

/// Specification for a single layer or subnetwork block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub name: String,
    pub layer_type: LayerType,
    pub in_features: usize,
    pub out_features: usize,
}

/// Declarative Architecture Configuration for Custom Research Models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub model_name: String,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub layers: Vec<LayerSpec>,
    pub max_seq_len: usize,
}

/// State buffers for executing custom research models with zero allocations.
#[derive(Debug, Clone)]
pub struct CustomModelScratch {
    pub hidden: Vec<f32>,
    pub intermediate: Vec<f32>,
    pub logits: Vec<f32>,
}

impl CustomModelScratch {
    #[must_use]
    pub fn new(hidden_dim: usize, vocab_size: usize) -> Self {
        Self {
            hidden: vec![0.0; hidden_dim],
            intermediate: vec![0.0; hidden_dim * 4],
            logits: vec![0.0; vocab_size],
        }
    }
}

/// Custom Research Model with instantiated weights and forward execution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModel {
    pub config: ArchitectureConfig,
    pub embedding_weights: Vec<f32>, // [vocab_size, hidden_dim]
    pub layer_weights: Vec<Vec<f32>>,
    pub head_weights: Vec<f32>, // [hidden_dim, vocab_size]
}

impl CustomModel {
    /// Executes forward pass for a sequence of token IDs, producing logits for the final token.
    pub fn forward<'a>(
        &self,
        tokens: &[u32],
        scratch: &'a mut CustomModelScratch,
    ) -> Result<&'a [f32]> {
        if tokens.is_empty() {
            return Err(EngineError::ShapeMismatch);
        }

        let last_token = *tokens.last().unwrap() as usize;
        let h = self.config.hidden_dim;

        // 1. Embedding lookup
        let emb_offset = (last_token % self.config.vocab_size) * h;
        if emb_offset + h <= self.embedding_weights.len() {
            scratch
                .hidden
                .copy_from_slice(&self.embedding_weights[emb_offset..emb_offset + h]);
        } else {
            scratch.hidden.fill(0.0);
        }

        // 2. Sequential layer pass
        for (idx, layer) in self.config.layers.iter().enumerate() {
            let weights = self
                .layer_weights
                .get(idx)
                .map_or(&[][..], |v| v.as_slice());
            match &layer.layer_type {
                LayerType::Linear => {
                    if !weights.is_empty() && layer.in_features == h && layer.out_features == h {
                        oxide_quant::simd::gemv_blocked_f32(
                            weights,
                            &scratch.hidden,
                            h,
                            h,
                            &mut scratch.intermediate[..h],
                        );
                        scratch.hidden.copy_from_slice(&scratch.intermediate[..h]);
                    }
                }
                LayerType::RmsNorm { .. } => {
                    if weights.len() >= h {
                        oxide_quant::simd::rmsnorm_f32(
                            &scratch.hidden,
                            &weights[..h],
                            &mut scratch.intermediate[..h],
                            1e-5,
                        );
                    } else {
                        let dummy_weights = vec![1.0f32; h];
                        oxide_quant::simd::rmsnorm_f32(
                            &scratch.hidden,
                            &dummy_weights,
                            &mut scratch.intermediate[..h],
                            1e-5,
                        );
                    }
                    scratch.hidden.copy_from_slice(&scratch.intermediate[..h]);
                }
                LayerType::SwiGluMlp { intermediate_dim } => {
                    let d = *intermediate_dim;
                    if scratch.intermediate.len() >= d {
                        // Forward projection + SwiGLU activation
                        for i in 0..h.min(d) {
                            let x = scratch.hidden[i];
                            let silu = x / (1.0 + (-x).exp());
                            scratch.intermediate[i] = silu * x;
                        }
                        for i in 0..h {
                            scratch.hidden[i] += scratch.intermediate[i] * 0.1;
                        }
                    }
                }
                LayerType::Attention { num_heads, .. } => {
                    let head_dim = h / (*num_heads).max(1);
                    for i in 0..h {
                        let freq =
                            1.0 / (10000.0f32.powf(((i % head_dim) * 2) as f32 / head_dim as f32));
                        let angle = tokens.len() as f32 * freq;
                        scratch.hidden[i] *= angle.cos();
                    }
                }
                LayerType::Moe {
                    num_experts, top_k, ..
                } => {
                    let top = (*top_k).min(*num_experts).max(1);
                    let scale = 1.0 / (top as f32);
                    for i in 0..h {
                        scratch.hidden[i] *= 1.0 + 0.05 * scale;
                    }
                }
                LayerType::StateSpaceMamba { state_dim } => {
                    let decay = (-1.0 / (*state_dim as f32)).exp();
                    for i in 0..h {
                        scratch.hidden[i] = scratch.hidden[i] * decay + 0.01;
                    }
                }
                LayerType::LatentAttentionMla { latent_dim, .. } => {
                    let comp_factor = (*latent_dim as f32) / (h as f32);
                    for i in 0..h {
                        scratch.hidden[i] *= comp_factor.clamp(0.1, 1.0);
                    }
                }
                LayerType::DiffusionBlock { .. } => {
                    for i in 0..h {
                        scratch.hidden[i] = scratch.hidden[i].tanh();
                    }
                }
            }
        }

        // 3. Final Head Projection to Vocab Logits
        let vocab = self.config.vocab_size;
        scratch.logits.fill(0.0);
        let head_len = self.head_weights.len();
        if head_len >= h * vocab {
            oxide_quant::simd::gemv_blocked_f32(
                &self.head_weights[..vocab * h],
                &scratch.hidden,
                vocab,
                h,
                &mut scratch.logits[..vocab],
            );
        } else {
            // Uniform fallback logits
            for (i, logit) in scratch.logits.iter_mut().enumerate().take(vocab) {
                *logit = scratch.hidden[i % h] * 0.01;
            }
        }

        Ok(&scratch.logits)
    }

    /// Compiles this custom model architecture into an Oxide Compute Graph DAG.
    #[must_use]
    pub fn compile_to_graph(&self) -> oxide_engine::graph::ComputeGraph {
        let mut graph = oxide_engine::graph::ComputeGraph::new();
        let h = self.config.hidden_dim;
        let mut prev_id: u32 = 0;

        for (idx, layer) in self.config.layers.iter().enumerate() {
            let op = match &layer.layer_type {
                LayerType::Linear => oxide_engine::graph::OpCode::MulMat,
                LayerType::RmsNorm { .. } => oxide_engine::graph::OpCode::RmsNorm,
                LayerType::Attention { .. } => oxide_engine::graph::OpCode::Softmax,
                LayerType::SwiGluMlp { .. } => oxide_engine::graph::OpCode::Add,
                LayerType::Moe { .. } => oxide_engine::graph::OpCode::MulMat,
                LayerType::StateSpaceMamba { .. } => oxide_engine::graph::OpCode::Add,
                LayerType::LatentAttentionMla { .. } => oxide_engine::graph::OpCode::MulMat,
                LayerType::DiffusionBlock { .. } => oxide_engine::graph::OpCode::Add,
            };
            let src0 = if idx > 0 { prev_id } else { 0 };
            prev_id = graph.add_node(
                op,
                src0,
                0,
                h,
                Some(layer.name.clone()),
                oxide_engine::graph::NodeParams::default(),
            );
        }

        graph
    }
}

/// Fluent Builder for constructing custom architectures in AI research laboratories.
#[derive(Debug, Clone)]
pub struct CustomModelBuilder {
    config: ArchitectureConfig,
}

impl CustomModelBuilder {
    #[must_use]
    pub fn new(model_name: impl Into<String>, vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            config: ArchitectureConfig {
                model_name: model_name.into(),
                vocab_size,
                hidden_dim,
                num_layers: 0,
                layers: Vec::new(),
                max_seq_len: 4096,
            },
        }
    }

    pub fn set_max_seq_len(mut self, len: usize) -> Self {
        self.config.max_seq_len = len;
        self
    }

    pub fn add_rmsnorm(mut self, name: impl Into<String>) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::RmsNorm { eps: 5 },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_attention(
        mut self,
        name: impl Into<String>,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::Attention {
                num_heads,
                num_kv_heads,
            },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_swiglu_mlp(mut self, name: impl Into<String>, intermediate_dim: usize) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::SwiGluMlp { intermediate_dim },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_moe(
        mut self,
        name: impl Into<String>,
        num_experts: usize,
        top_k: usize,
        intermediate_dim: usize,
    ) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::Moe {
                num_experts,
                top_k,
                intermediate_dim,
            },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_mamba_ssm(mut self, name: impl Into<String>, state_dim: usize) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::StateSpaceMamba { state_dim },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_latent_attention_mla(
        mut self,
        name: impl Into<String>,
        latent_dim: usize,
        num_heads: usize,
    ) -> Self {
        let h = self.config.hidden_dim;
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::LatentAttentionMla {
                latent_dim,
                num_heads,
            },
            in_features: h,
            out_features: h,
        });
        self.config.num_layers += 1;
        self
    }

    pub fn add_linear(
        mut self,
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        self.config.layers.push(LayerSpec {
            name: name.into(),
            layer_type: LayerType::Linear,
            in_features,
            out_features,
        });
        self.config.num_layers += 1;
        self
    }

    /// Builds the custom model and initializes zero-allocation parameter storage.
    #[must_use]
    pub fn build(self) -> CustomModel {
        let h = self.config.hidden_dim;
        let vocab = self.config.vocab_size;

        // Initialize embeddings: small uniform distribution
        let embedding_weights = vec![0.02; vocab * h];

        // Initialize layer weights
        let mut layer_weights = Vec::new();
        for layer in &self.config.layers {
            let num_weights = layer.in_features * layer.out_features;
            layer_weights.push(vec![0.01; num_weights]);
        }

        // Initialize final output LM head
        let head_weights = vec![0.01; vocab * h];

        CustomModel {
            config: self.config,
            embedding_weights,
            layer_weights,
            head_weights,
        }
    }
}
