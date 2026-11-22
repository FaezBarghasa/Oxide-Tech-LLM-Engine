//! Mode B: S-LoRA & Punica Multi-Persona Dynamic Hot-Swapping.
//!
//! Evaluates LoRA delta on-the-fly alongside frozen quantized base weights:
//! y = BaseGEMM(x, W_base) + (alpha / r) * ((x * A^T) * B^T)
//!
//! Enables persona switching in < 1 ms with zero disk I/O and zero re-quantization.

use crate::loader::{UnslothAdapter, UnslothTensorPair};
use std::collections::HashMap;

/// S-LoRA Multi-Persona Dynamic Adapter Manager.
#[derive(Debug, Default)]
pub struct UnslothLoraManager {
    adapters: HashMap<String, UnslothAdapter>,
    active_adapter_id: Option<String>,
}

impl UnslothLoraManager {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new Unsloth adapter in memory.
    pub fn register_adapter(&mut self, adapter: UnslothAdapter) {
        self.adapters.insert(adapter.name.clone(), adapter);
    }

    /// Switches the active persona adapter in < 1 ms.
    pub fn set_active_adapter(&mut self, name: Option<&str>) -> bool {
        if let Some(n) = name {
            if self.adapters.contains_key(n) {
                self.active_adapter_id = Some(n.to_string());
                true
            } else {
                false
            }
        } else {
            self.active_adapter_id = None;
            true
        }
    }

    #[must_use]
    pub fn active_adapter(&self) -> Option<&UnslothAdapter> {
        self.active_adapter_id
            .as_ref()
            .and_then(|id| self.adapters.get(id))
    }

    #[must_use]
    pub fn is_adapter_active(&self) -> bool {
        self.active_adapter_id.is_some()
    }
}

impl UnslothTensorPair {
    /// Evaluates S-LoRA dynamic forward: y = x + (alpha / r) * ((x * A^T) * B^T)
    pub fn apply_lora_delta(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_dim);
        assert_eq!(output.len(), self.out_dim);

        // Step 1: inter = x * A^T in R^rank
        let mut inter = vec![0.0f32; self.rank];
        for (r, inter_val) in inter.iter_mut().enumerate() {
            let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
            let mut sum = 0.0f32;
            for (i, &inp) in input.iter().enumerate() {
                sum += inp * a_row[i];
            }
            *inter_val = sum;
        }

        // Step 2: output += (alpha / r) * inter * B^T
        for (o, out_elem) in output.iter_mut().enumerate() {
            let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            let mut sum = 0.0f32;
            for (r, &b_val) in b_row.iter().enumerate() {
                sum += inter[r] * b_val;
            }
            *out_elem += sum * self.scaling;
        }
    }
}
