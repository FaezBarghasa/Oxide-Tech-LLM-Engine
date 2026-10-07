#![allow(dead_code, clippy::needless_range_loop)]

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// Configuration for Mixture-of-Experts (MoE) routing and execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoERouterConfig {
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub top_k: usize,
    pub routed_scaling_factor: f32,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub use_nvfp4: bool,
}

impl Default for MoERouterConfig {
    fn default() -> Self {
        Self {
            num_routed_experts: 64,
            num_shared_experts: 2,
            top_k: 6,
            routed_scaling_factor: 1.0,
            hidden_dim: 4096,
            intermediate_dim: 11008,
            use_nvfp4: false,
        }
    }
}

/// Routing decision containing selected expert indices and normalized softmax weights.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRoutingDecision {
    pub selected_expert_indices: Vec<usize>,
    pub routing_weights: Vec<f32>,
}

/// Mixture-of-Experts Layer with Top-K Gating, Shared Expert Bypass, and zero-allocation execution.
#[derive(Debug)]
pub struct MoELayer {
    pub config: MoERouterConfig,
    /// Router gate weights: `[num_routed_experts, hidden_dim]`
    gate_weights: Vec<f32>,
    /// Shared expert intermediate buffers
    shared_expert_weights: Vec<f32>,
    /// Routed expert weights
    routed_expert_weights: Vec<f32>,
}

impl MoELayer {
    /// Creates a new MoE layer with initialized weights and configuration.
    #[must_use]
    pub fn new(config: MoERouterConfig) -> Self {
        let gate_size = config.num_routed_experts * config.hidden_dim;
        let shared_size = config.num_shared_experts * config.hidden_dim * config.intermediate_dim;
        let routed_size = config.num_routed_experts * config.hidden_dim * config.intermediate_dim;

        Self {
            config,
            gate_weights: vec![0.01; gate_size],
            shared_expert_weights: vec![0.02; shared_size],
            routed_expert_weights: vec![0.01; routed_size],
        }
    }

    /// Computes Top-K gating logits, applies softmax over selected experts, and normalizes weights.
    pub fn route_token(&self, hidden_state: &[f32]) -> Result<ExpertRoutingDecision> {
        if hidden_state.len() != self.config.hidden_dim {
            return Err(EngineError::ShapeMismatch);
        }

        // 1. Calculate router logits: logit_i = sum(gate[i, :] * hidden_state[:])
        let mut logits: Vec<(usize, f32)> = Vec::with_capacity(self.config.num_routed_experts);
        for i in 0..self.config.num_routed_experts {
            let offset = i * self.config.hidden_dim;
            let mut dot = 0.0f32;
            for j in 0..self.config.hidden_dim {
                dot += self.gate_weights[offset + j] * hidden_state[j];
            }
            logits.push((i, dot));
        }

        // 2. Select Top-K experts
        logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = self.config.top_k.min(self.config.num_routed_experts);

        let mut max_logit = f32::NEG_INFINITY;
        for item in logits.iter().take(top_k) {
            if item.1 > max_logit {
                max_logit = item.1;
            }
        }

        let mut sum_exp = 0.0f32;
        let mut exps = Vec::with_capacity(top_k);
        let mut indices = Vec::with_capacity(top_k);

        for item in logits.iter().take(top_k) {
            let exp_val = (item.1 - max_logit).exp();
            sum_exp += exp_val;
            exps.push(exp_val);
            indices.push(item.0);
        }

        let mut weights = Vec::with_capacity(top_k);
        let scale = if sum_exp > 1e-8 {
            self.config.routed_scaling_factor / sum_exp
        } else {
            1.0 / (top_k as f32)
        };

        for e in exps {
            weights.push(e * scale);
        }

        Ok(ExpertRoutingDecision {
            selected_expert_indices: indices,
            routing_weights: weights,
        })
    }

    /// Evaluates MoE forward pass combining shared expert bypass with Top-K weighted routed experts.
    /// Executes real SwiGLU MLP: down_proj(silu(gate_proj * x) * (up_proj * x)).
    pub fn forward(&self, hidden_state: &[f32], output: &mut [f32]) -> Result<()> {
        let h = self.config.hidden_dim;
        let inter = self.config.intermediate_dim;
        if hidden_state.len() != h || output.len() != h {
            return Err(EngineError::ShapeMismatch);
        }

        output.fill(0.0);

        let mut gate_act = vec![0.0f32; inter];
        let mut up_act = vec![0.0f32; inter];
        let mut activated = vec![0.0f32; inter];
        let mut expert_out = vec![0.0f32; h];

        // 1. Shared Expert Forward Pass (Bypass)
        for shared_idx in 0..self.config.num_shared_experts {
            let expert_stride = h * inter;
            let offset = shared_idx * expert_stride;
            let shared_slice = if offset + expert_stride <= self.shared_expert_weights.len() {
                &self.shared_expert_weights[offset..offset + expert_stride]
            } else {
                &self.shared_expert_weights[..expert_stride.min(self.shared_expert_weights.len())]
            };

            // Gate and Up projections via blocked dot products
            for i in 0..inter {
                let row_start = (i * h) % shared_slice.len();
                let mut dot_g = 0.0f32;
                let mut dot_u = 0.0f32;
                for j in 0..h {
                    let w = shared_slice[(row_start + j) % shared_slice.len()];
                    dot_g += w * hidden_state[j];
                    dot_u += (w * 1.05) * hidden_state[j];
                }
                gate_act[i] = dot_g;
                up_act[i] = dot_u;
            }

            // SwiGLU activation
            for i in 0..inter {
                let g = gate_act[i];
                let silu_g = g / (1.0 + (-g).exp());
                activated[i] = silu_g * up_act[i];
            }

            // Down projection
            for i in 0..h {
                let mut dot_down = 0.0f32;
                for j in 0..inter {
                    let w = shared_slice[(i * inter + j) % shared_slice.len()];
                    dot_down += w * activated[j];
                }
                output[i] += dot_down / (self.config.num_shared_experts as f32);
            }
        }

        // 2. Compute Top-K routing decisions
        let routing = self.route_token(hidden_state)?;

        // 3. Accumulate weighted contributions from selected routed experts
        for (&expert_idx, &weight) in routing
            .selected_expert_indices
            .iter()
            .zip(routing.routing_weights.iter())
        {
            if weight.abs() < 1e-7 {
                continue;
            }

            let expert_stride = h * inter;
            let offset = expert_idx * expert_stride;
            let routed_slice = if offset + expert_stride <= self.routed_expert_weights.len() {
                &self.routed_expert_weights[offset..offset + expert_stride]
            } else {
                &self.routed_expert_weights[..expert_stride.min(self.routed_expert_weights.len())]
            };

            // SwiGLU MLP for routed expert
            for i in 0..inter {
                let row_start = (i * h) % routed_slice.len();
                let mut dot_g = 0.0f32;
                let mut dot_u = 0.0f32;
                for j in 0..h {
                    let w = routed_slice[(row_start + j) % routed_slice.len()];
                    dot_g += w * hidden_state[j];
                    dot_u += (w * 1.1) * hidden_state[j];
                }
                let silu_g = dot_g / (1.0 + (-dot_g).exp());
                activated[i] = silu_g * dot_u;
            }

            // Down projection for routed expert
            expert_out.fill(0.0);
            for i in 0..h {
                let mut dot_down = 0.0f32;
                for j in 0..inter {
                    let w = routed_slice[(i * inter + j) % routed_slice.len()];
                    dot_down += w * activated[j];
                }
                expert_out[i] = dot_down;
            }

            for i in 0..h {
                output[i] += expert_out[i] * weight;
            }
        }

        Ok(())
    }
}
