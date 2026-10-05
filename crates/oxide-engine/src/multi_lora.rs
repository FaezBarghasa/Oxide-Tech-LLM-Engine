//! Efficient Multi-LoRA Engine for Dense and MoE Layers.
//!
//! Provides dynamic multi-tenant LoRA routing, allowing concurrent inference slots
//! to apply distinct adapter weights to base model layers and MoE experts without copying base weights.

use std::collections::HashMap;

/// A Low-Rank Adapter (LoRA) for a single linear projection:
/// Delta_W = (B * A) * (alpha / r)
#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapterWeights {
    pub lora_a: Vec<f32>, // shape [r, in_dim]
    pub lora_b: Vec<f32>, // shape [out_dim, r]
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub scaling: f32, // alpha / r
}

impl LoraAdapterWeights {
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32) -> Self {
        Self {
            lora_a: vec![0.01; rank * in_dim],
            lora_b: vec![0.01; out_dim * rank],
            rank,
            in_dim,
            out_dim,
            scaling: alpha / rank as f32,
        }
    }

    /// Evaluates LoRA delta on input: output += (input * A^T) * B^T * scaling
    pub fn apply_delta(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_dim);
        assert_eq!(output.len(), self.out_dim);

        // Step 1: intermediate = input * A^T -> shape [r]
        let mut intermediate = vec![0.0f32; self.rank];
        for (r_idx, item) in intermediate.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let a_row = &self.lora_a[r_idx * self.in_dim..(r_idx + 1) * self.in_dim];
            for (i, &inp) in input.iter().enumerate() {
                sum += inp * a_row[i];
            }
            *item = sum;
        }

        // Step 2: output += intermediate * B^T * scaling -> shape [out_dim]
        for (out_idx, out_elem) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let b_row = &self.lora_b[out_idx * self.rank..(out_idx + 1) * self.rank];
            for (r_idx, &inter) in intermediate.iter().enumerate() {
                sum += inter * b_row[r_idx];
            }
            *out_elem += sum * self.scaling;
        }
    }
}

/// Multi-Tenant Multi-LoRA Manager across Dense and MoE Layers.
#[derive(Debug, Clone, Default)]
pub struct MultiLoraManager {
    /// Mapping: adapter_id -> (layer_name -> LoraAdapterWeights)
    pub adapters: HashMap<String, HashMap<String, LoraAdapterWeights>>,
}

impl MultiLoraManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    /// Registers a new LoRA adapter layer.
    pub fn register_layer(
        &mut self,
        adapter_id: &str,
        layer_name: &str,
        weights: LoraAdapterWeights,
    ) {
        self.adapters
            .entry(adapter_id.to_string())
            .or_default()
            .insert(layer_name.to_string(), weights);
    }

    /// Applies LoRA delta for a specific active tenant slot and layer.
    pub fn apply_if_active(
        &self,
        adapter_id: Option<&str>,
        layer_name: &str,
        input: &[f32],
        output: &mut [f32],
    ) {
        if let Some(id) = adapter_id {
            if let Some(layer_map) = self.adapters.get(id) {
                if let Some(lora) = layer_map.get(layer_name) {
                    lora.apply_delta(input, output);
                }
            }
        }
    }
}
