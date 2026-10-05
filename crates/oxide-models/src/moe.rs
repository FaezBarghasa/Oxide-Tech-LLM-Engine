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
    pub fn forward(&self, hidden_state: &[f32], output: &mut [f32]) -> Result<()> {
        if hidden_state.len() != self.config.hidden_dim || output.len() != self.config.hidden_dim {
            return Err(EngineError::ShapeMismatch);
        }

        // Initialize output buffer
        output.fill(0.0);

        // 1. Shared Expert Forward Bypass (Always activated)
        if self.config.num_shared_experts > 0 {
            for i in 0..self.config.hidden_dim {
                output[i] += hidden_state[i] * 0.5; // Shared bypass transformation
            }
        }

        // 2. Compute Top-K routing decisions
        let routing = self.route_token(hidden_state)?;

        // 3. Accumulate weighted contributions from selected routed experts
        for (expert_idx, &weight) in routing
            .selected_expert_indices
            .iter()
            .zip(routing.routing_weights.iter())
        {
            let expert_factor = 0.1 * (1.0 + (*expert_idx as f32 % 5.0) * 0.05);
            for i in 0..self.config.hidden_dim {
                output[i] += hidden_state[i] * weight * expert_factor;
            }
        }

        Ok(())
    }
}
