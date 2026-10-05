#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// LoRA Adapter Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraConfig {
    pub adapter_name: String,
    pub rank_r: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub is_qlora: bool,
}

impl LoraConfig {
    #[must_use]
    pub fn new(adapter_name: impl Into<String>, rank_r: usize, alpha: f32) -> Self {
        Self {
            adapter_name: adapter_name.into(),
            rank_r,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            is_qlora: false,
        }
    }
}

/// Single Weight Matrix LoRA Weights (Low-Rank Decomposition: `A: [r, in_dim]`, `B: [out_dim, r]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoraLayerWeights {
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank_r: usize,
    pub lora_a: Vec<f32>, // Shape: [rank_r, in_dim]
    pub lora_b: Vec<f32>, // Shape: [out_dim, rank_r]
    pub scaling: f32,
}

impl LoraLayerWeights {
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, rank_r: usize, alpha: f32) -> Self {
        let scaling = alpha / (rank_r as f32);
        Self {
            in_dim,
            out_dim,
            rank_r,
            lora_a: vec![0.01; rank_r * in_dim],
            lora_b: vec![0.0; out_dim * rank_r],
            scaling,
        }
    }

    /// Computes delta: `output += (X @ A^T @ B^T) * scaling`.
    /// `input`: `[batch, in_dim]`
    /// `output`: `[batch, out_dim]`
    pub fn forward_delta(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_dim);
        assert_eq!(output.len(), self.out_dim);

        // Temp buffer for intermediate rank activation: [rank_r]
        let mut rank_act = vec![0.0; self.rank_r];

        // 1. rank_act = input @ A^T
        for r in 0..self.rank_r {
            let mut sum = 0.0;
            let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
            for (i, &inp) in input.iter().enumerate() {
                sum += inp * a_row[i];
            }
            rank_act[r] = sum;
        }

        // 2. output += (rank_act @ B^T) * scaling
        for (o, out_val) in output.iter_mut().enumerate().take(self.out_dim) {
            let b_row = &self.lora_b[o * self.rank_r..(o + 1) * self.rank_r];
            let mut sum = 0.0;
            for (r, &ract) in rank_act.iter().enumerate() {
                sum += ract * b_row[r];
            }
            *out_val += sum * self.scaling;
        }
    }

    /// Merges LoRA weights directly into the base model weights matrix `W` (`[out_dim, in_dim]`).
    pub fn merge_into_base(&self, base_weights: &mut [f32]) {
        assert_eq!(base_weights.len(), self.out_dim * self.in_dim);

        for o in 0..self.out_dim {
            let b_row = &self.lora_b[o * self.rank_r..(o + 1) * self.rank_r];
            for i in 0..self.in_dim {
                let mut delta = 0.0;
                for r in 0..self.rank_r {
                    delta += b_row[r] * self.lora_a[r * self.in_dim + i];
                }
                base_weights[o * self.in_dim + i] += delta * self.scaling;
            }
        }
    }
}

/// Dynamic LoRA Hot-Swapping Adapter Registry.
#[derive(Debug, Clone, Default)]
pub struct LoraHotSwapRegistry {
    adapters: HashMap<String, (LoraConfig, HashMap<String, LoraLayerWeights>)>,
    active_adapter_id: Option<String>,
}

impl LoraHotSwapRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers or replaces an adapter in the hot-swap cache without stopping inference.
    pub fn register_adapter(
        &mut self,
        config: LoraConfig,
        layers: HashMap<String, LoraLayerWeights>,
    ) {
        let name = config.adapter_name.clone();
        self.adapters.insert(name.clone(), (config, layers));
        if self.active_adapter_id.is_none() {
            self.active_adapter_id = Some(name);
        }
    }

    /// Hot-swaps the active adapter in O(1) time.
    pub fn set_active_adapter(&mut self, name: &str) -> bool {
        if self.adapters.contains_key(name) {
            self.active_adapter_id = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// Deactivates LoRA adapter (pure base model inference).
    pub fn clear_active_adapter(&mut self) {
        self.active_adapter_id = None;
    }

    /// Retrieves the active layer weights for a target submodule name (e.g. "layers.0.q_proj").
    #[must_use]
    pub fn get_active_layer(&self, module_path: &str) -> Option<&LoraLayerWeights> {
        let active_name = self.active_adapter_id.as_ref()?;
        let (_, layers) = self.adapters.get(active_name)?;
        layers.get(module_path)
    }

    #[must_use]
    pub fn active_adapter(&self) -> Option<&str> {
        self.active_adapter_id.as_deref()
    }

    #[must_use]
    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }
}
