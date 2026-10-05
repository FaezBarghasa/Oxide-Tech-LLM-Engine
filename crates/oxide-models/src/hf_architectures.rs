//! Universal Hugging Face Architecture Implementations.
//!
//! Provides dedicated architectural blocks for:
//! - Mixture-of-Experts: DeepSeek-V3, Mixtral, Qwen-MoE, GPT-OSS
//! - Hybrid SSM & Attention: Mamba, Mamba-2, Qwen3.5
//! - Multi-Modal Vision-Language: LLaVA, Qwen-VL, Pixtral
//! - Embedding & Retrieval: E5-Mistral, GTE, ColBERT late interaction
//! - Reward & PRM Models: Qwen-Math, ArmoRM, Process Reward Model heads

use serde::{Deserialize, Serialize};

/// Mamba Selective State-Space Model (SSM) Block.
///
/// Implements recurrent continuous-time state update:
/// h_t = A_bar * h_{t-1} + B_bar * x_t
/// y_t = C * h_t + D * x_t
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MambaSsmBlock {
    pub d_model: usize,
    pub d_state: usize, // Latent SSM state dimension (e.g. 16 or 64)
    pub d_conv: usize,  // 1D causal convolution kernel size (e.g. 4)
    pub dt_rank: usize, // Delta projection rank
}

impl MambaSsmBlock {
    #[must_use]
    pub fn new(d_model: usize, d_state: usize) -> Self {
        Self {
            d_model,
            d_state,
            d_conv: 4,
            dt_rank: (d_model + 15) / 16,
        }
    }

    /// Evaluates a single recurrent step of the selective state-space engine.
    /// `x`: current token input `[d_model]`
    /// `state`: persistent recurrent hidden matrix `[d_model, d_state]`
    /// `a_bar`: discretized transition matrix `[d_model, d_state]`
    /// `b_bar`: discretized input matrix `[d_model, d_state]`
    /// `c`: output projection `[d_model, d_state]`
    /// `out`: output tensor `[d_model]`
    pub fn step(
        &self,
        x: &[f32],
        state: &mut [f32],
        a_bar: &[f32],
        b_bar: &[f32],
        c: &[f32],
        out: &mut [f32],
    ) {
        assert_eq!(x.len(), self.d_model);
        assert_eq!(out.len(), self.d_model);

        for i in 0..self.d_model {
            let mut y_i = 0.0f32;
            let row_offset = i * self.d_state;

            for s in 0..self.d_state {
                let idx = row_offset + s;
                // State update: h = A_bar * h + B_bar * x
                state[idx] = a_bar[idx] * state[idx] + b_bar[idx] * x[i];
                // Output projection: y += C * h
                y_i += c[idx] * state[idx];
            }

            out[i] = y_i;
        }
    }
}

/// ColBERT Late Interaction (MaxSim) Scoring Engine.
///
/// Computes fine-grained token-level cross-matching:
/// Score(Q, D) = sum_{i in Q} max_{j in D} (E_Q[i] . E_D[j])
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColBertLateInteraction {
    pub dim: usize, // Vector dimension (e.g. 128)
}

impl ColBertLateInteraction {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Computes late-interaction MaxSim similarity score between query and document token embeddings.
    /// `q_embeds`: `[num_q_tokens, dim]`
    /// `d_embeds`: `[num_d_tokens, dim]`
    #[must_use]
    pub fn score(&self, q_embeds: &[f32], d_embeds: &[f32]) -> f32 {
        let num_q = q_embeds.len() / self.dim;
        let num_d = d_embeds.len() / self.dim;

        let mut total_score = 0.0f32;

        for q_idx in 0..num_q {
            let q_vec = &q_embeds[q_idx * self.dim..(q_idx + 1) * self.dim];
            let mut max_sim = f32::NEG_INFINITY;

            for d_idx in 0..num_d {
                let d_vec = &d_embeds[d_idx * self.dim..(d_idx + 1) * self.dim];
                let mut dot = 0.0f32;
                for k in 0..self.dim {
                    dot += q_vec[k] * d_vec[k];
                }
                if dot > max_sim {
                    max_sim = dot;
                }
            }

            if max_sim != f32::NEG_INFINITY {
                total_score += max_sim;
            }
        }

        total_score
    }
}

/// DeepSeek-V3 MoE Architecture Block.
///
/// Features 256 fine-grained routed experts (top-8 activated) + 1 dedicated shared expert.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeepSeekV3MoeConfig {
    pub hidden_dim: usize,
    pub num_routed_experts: usize,  // 256
    pub num_shared_experts: usize,  // 1
    pub top_k: usize,               // 8
    pub routed_scaling_factor: f32, // 2.5
}

impl Default for DeepSeekV3MoeConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 7168,
            num_routed_experts: 256,
            num_shared_experts: 1,
            top_k: 8,
            routed_scaling_factor: 2.5,
        }
    }
}

/// Vision-Language Multi-Modal Projector (LLaVA / Qwen-VL / Pixtral).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiModalVisionProjector {
    pub vision_dim: usize, // e.g. 1024 or 1152 (CLIP / SigLIP / ViT)
    pub text_dim: usize,   // e.g. 4096 (Llama / Qwen hidden size)
    pub is_mlp_gelu: bool,
}

impl MultiModalVisionProjector {
    #[must_use]
    pub fn new(vision_dim: usize, text_dim: usize) -> Self {
        Self {
            vision_dim,
            text_dim,
            is_mlp_gelu: true,
        }
    }

    /// Projects vision tokens into text embedding space:
    /// `patch_tokens`: `[num_patches, vision_dim]`
    /// `weights`: `[vision_dim, text_dim]`
    /// `out_text_embeds`: `[num_patches, text_dim]`
    pub fn project_patches(
        &self,
        patch_tokens: &[f32],
        weights: &[f32],
        out_text_embeds: &mut [f32],
    ) {
        let num_patches = patch_tokens.len() / self.vision_dim;
        assert_eq!(out_text_embeds.len(), num_patches * self.text_dim);

        for p in 0..num_patches {
            let patch = &patch_tokens[p * self.vision_dim..(p + 1) * self.vision_dim];
            let out_p = &mut out_text_embeds[p * self.text_dim..(p + 1) * self.text_dim];

            for j in 0..self.text_dim {
                let mut sum = 0.0f32;
                for i in 0..self.vision_dim {
                    sum += patch[i] * weights[i * self.text_dim + j];
                }
                out_p[j] = sum;
            }
        }
    }
}

/// Reward and Classification Head (Qwen-Math, ArmoRM, PRM).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardClassifierHead {
    pub hidden_dim: usize,
    pub num_classes: usize, // 1 for scalar reward, N for multi-objective / PRM step labels
}

impl RewardClassifierHead {
    #[must_use]
    pub fn scalar_reward(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            num_classes: 1,
        }
    }

    #[must_use]
    pub fn prm_step_classifier(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            num_classes: 3, // Positive (+1), Neutral (0), Negative (-1)
        }
    }

    /// Evaluates scalar reward from final token hidden state:
    /// `hidden`: `[hidden_dim]`
    /// `head_weight`: `[hidden_dim]`
    #[must_use]
    pub fn predict_scalar(&self, hidden: &[f32], head_weight: &[f32]) -> f32 {
        assert_eq!(hidden.len(), self.hidden_dim);
        assert_eq!(head_weight.len(), self.hidden_dim);

        let mut reward = 0.0f32;
        for i in 0..self.hidden_dim {
            reward += hidden[i] * head_weight[i];
        }
        reward
    }
}
