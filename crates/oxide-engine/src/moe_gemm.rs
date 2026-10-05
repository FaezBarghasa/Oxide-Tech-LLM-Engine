//! High-Performance GEMM & MoE Kernels (CUTLASS, TRTLLM-GEN, CuTeDSL).
//!
//! Provides fused Mixture-of-Experts (MoE) dispatch and grouped GEMM acceleration
//! across standard, FP8, and NVFP4 quantized expert matrices.

use serde::{Deserialize, Serialize};

/// Layout descriptor inspired by NVIDIA CuTeDSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CuTeLayout<const DIMS: usize> {
    pub shape: [usize; DIMS],
    pub stride: [usize; DIMS],
}

impl<const DIMS: usize> CuTeLayout<DIMS> {
    #[must_use]
    pub fn new(shape: [usize; DIMS], stride: [usize; DIMS]) -> Self {
        Self { shape, stride }
    }

    /// Linearizes multi-dimensional coordinate into 1D memory offset.
    #[must_use]
    pub fn linearize(&self, coords: [usize; DIMS]) -> usize {
        let mut offset = 0;
        for i in 0..DIMS {
            offset += coords[i] * self.stride[i];
        }
        offset
    }
}

/// MoE Top-K Routing Decision for a Token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoERouteChoice {
    pub expert_indices: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

/// Fused MoE Gate & Dispatch Engine.
#[derive(Debug, Clone)]
pub struct FusedMoeGateEngine {
    pub num_total_experts: usize,
    pub top_k: usize,
    pub hidden_dim: usize,
}

impl FusedMoeGateEngine {
    #[must_use]
    pub fn new(num_total_experts: usize, top_k: usize, hidden_dim: usize) -> Self {
        Self {
            num_total_experts,
            top_k,
            hidden_dim,
        }
    }

    /// Evaluates top-K gating with softmax normalization:
    /// `gate_logits`: `[num_total_experts]` raw gate projections
    #[must_use]
    pub fn route_token(&self, gate_logits: &[f32]) -> MoERouteChoice {
        assert_eq!(gate_logits.len(), self.num_total_experts);

        // Find top-K largest logit indices:
        let mut indexed: Vec<(usize, f32)> = gate_logits
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(self.top_k);

        // Compute softmax over top-K chosen experts:
        let max_logit = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut exps = Vec::with_capacity(self.top_k);
        for (_, val) in &indexed {
            let e = (val - max_logit).exp();
            exps.push(e);
            sum_exp += e;
        }

        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        let mut expert_indices = Vec::with_capacity(self.top_k);
        let mut expert_weights = Vec::with_capacity(self.top_k);

        for (i, &(exp_idx, _)) in indexed.iter().enumerate() {
            expert_indices.push(exp_idx);
            expert_weights.push(exps[i] * inv_sum);
        }

        MoERouteChoice {
            expert_indices,
            expert_weights,
        }
    }

    /// Executes batched CUTLASS-style Grouped GEMM across active experts:
    /// `input`: token activation `[hidden_dim]`
    /// `expert_weights`: flat slice of all expert weight matrices `[num_experts, hidden_dim, intermediate_dim]`
    /// `intermediate_dim`: output dimension of first MoE projection
    pub fn dispatch_grouped_gemm(
        &self,
        input: &[f32],
        expert_weights: &[f32],
        intermediate_dim: usize,
        route: &MoERouteChoice,
        out: &mut [f32],
    ) {
        assert_eq!(input.len(), self.hidden_dim);
        assert_eq!(out.len(), intermediate_dim);
        out.fill(0.0);

        let expert_matrix_size = self.hidden_dim * intermediate_dim;

        for (k_idx, &exp_idx) in route.expert_indices.iter().enumerate() {
            let weight_factor = route.expert_weights[k_idx];
            let exp_base = exp_idx * expert_matrix_size;
            let exp_mat = &expert_weights[exp_base..exp_base + expert_matrix_size];

            // GEMV: out += weight_factor * (input * W_expert)
            for j in 0..intermediate_dim {
                let mut sum = 0.0f32;
                for i in 0..self.hidden_dim {
                    sum += input[i] * exp_mat[i * intermediate_dim + j];
                }
                out[j] += sum * weight_factor;
            }
        }
    }
}
