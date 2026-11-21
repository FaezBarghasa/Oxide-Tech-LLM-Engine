//! Native Unsloth Safetensors LoRA Ingestion, Dequant-Time Fusion, and S-LoRA Hot-Swapping.
//!
//! Directly loads Unsloth training checkpoints without Python merge scripts,
//! supporting zero-overhead weight fusion and sub-millisecond S-LoRA multi-persona hot-swapping.

use oxide_core::error::EngineError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Unsloth `adapter_config.json` representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnslothAdapterConfig {
    pub r: usize,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: String,
}

impl Default for UnslothAdapterConfig {
    fn default() -> Self {
        Self {
            r: 16,
            lora_alpha: 32.0,
            lora_dropout: 0.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: "none".to_string(),
        }
    }
}

/// A parsed Unsloth LoRA tensor pair: A in R^{r x d_in}, B in R^{d_out x r}.
#[derive(Debug, Clone, PartialEq)]
pub struct UnslothTensorPair {
    pub lora_a: Vec<f32>,
    pub lora_b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub scaling: f32,
}

impl UnslothTensorPair {
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32) -> Self {
        let scaling = alpha / rank as f32;
        Self {
            lora_a: vec![0.01; rank * in_dim],
            lora_b: vec![0.01; out_dim * rank],
            in_dim,
            out_dim,
            rank,
            scaling,
        }
    }

    /// Dequant-time fusion: Delta_W = (alpha / r) * B * A
    pub fn compute_weight_delta(&self, delta_w: &mut [f32]) {
        assert_eq!(delta_w.len(), self.out_dim * self.in_dim);
        delta_w.fill(0.0);

        for o in 0..self.out_dim {
            let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            for (r, &b_elem) in b_row.iter().enumerate() {
                let b_val = b_elem * self.scaling;
                let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
                let out_offset = o * self.in_dim;
                for (i, &a_elem) in a_row.iter().enumerate() {
                    delta_w[out_offset + i] += b_val * a_elem;
                }
            }
        }
    }

    /// Evaluates S-LoRA forward: y = x + (alpha / r) * ((x * A^T) * B^T)
    pub fn apply_lora_delta(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_dim);
        assert_eq!(output.len(), self.out_dim);

        let mut inter = vec![0.0f32; self.rank];
        for (r, inter_val) in inter.iter_mut().enumerate() {
            let a_row = &self.lora_a[r * self.in_dim..(r + 1) * self.in_dim];
            let mut sum = 0.0f32;
            for (i, &inp) in input.iter().enumerate() {
                sum += inp * a_row[i];
            }
            *inter_val = sum;
        }

        for (o, out_elem) in output.iter_mut().enumerate() {
            let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
            let mut sum = 0.0f32;
            for (r, &val) in inter.iter().enumerate() {
                sum += b_row[r] * val;
            }
            *out_elem += sum * self.scaling;
        }
    }
}

/// In-Memory Unsloth Adapter instance containing all target projection weights.
#[derive(Debug, Clone, PartialEq)]
pub struct UnslothAdapter {
    pub name: String,
    pub config: UnslothAdapterConfig,
    pub modules: HashMap<String, UnslothTensorPair>,
}

impl UnslothAdapter {
    #[must_use]
    pub fn new(name: impl Into<String>, config: UnslothAdapterConfig) -> Self {
        Self {
            name: name.into(),
            config,
            modules: HashMap::new(),
        }
    }

    pub fn add_module(&mut self, module_name: impl Into<String>, pair: UnslothTensorPair) {
        self.modules.insert(module_name.into(), pair);
    }

    /// Loads an Unsloth LoRA adapter from raw SafeTensors bytes.
    pub fn from_safetensors(
        name: impl Into<String>,
        bytes: &[u8],
        config: UnslothAdapterConfig,
    ) -> Result<Self, EngineError> {
        let tensors = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| EngineError::BackendError(format!("Failed to parse SafeTensors: {e}")))?;

        let mut adapter = Self::new(name, config);
        let mut a_map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut b_map: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        for (t_name, view) in tensors.tensors() {
            let f32_data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let u8_slice = view.data();
                    let (chunks, _) = u8_slice.as_chunks::<4>();
                    chunks
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk))
                        .collect()
                }
                safetensors::Dtype::F16 => {
                    let u8_slice = view.data();
                    let (chunks, _) = u8_slice.as_chunks::<2>();
                    chunks
                        .iter()
                        .map(|chunk| half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    let u8_slice = view.data();
                    let (chunks, _) = u8_slice.as_chunks::<2>();
                    chunks
                        .iter()
                        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                        .collect()
                }
                _ => continue,
            };

            let shape = view.shape().to_vec();

            if t_name.contains("lora_A") || t_name.contains("lora_a") {
                let key = t_name
                    .replace(".lora_A.weight", "")
                    .replace(".lora_a.weight", "");
                a_map.insert(key, (f32_data, shape));
            } else if t_name.contains("lora_B") || t_name.contains("lora_b") {
                let key = t_name
                    .replace(".lora_B.weight", "")
                    .replace(".lora_b.weight", "");
                b_map.insert(key, (f32_data, shape));
            }
        }

        for (key, (a_data, a_shape)) in a_map {
            if let Some((b_data, b_shape)) = b_map.remove(&key) {
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    let rank = a_shape[0];
                    let in_dim = a_shape[1];
                    let out_dim = b_shape[0];
                    let pair = UnslothTensorPair {
                        lora_a: a_data,
                        lora_b: b_data,
                        in_dim,
                        out_dim,
                        rank,
                        scaling: adapter.config.lora_alpha / rank.max(1) as f32,
                    };
                    adapter.add_module(key, pair);
                }
            }
        }

        Ok(adapter)
    }

    /// Loads an Unsloth LoRA adapter from a file on disk (SafeTensors).
    pub fn from_safetensors_file(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        config: UnslothAdapterConfig,
    ) -> Result<Self, EngineError> {
        let p = path.as_ref();
        if !p.exists() {
            return Err(EngineError::ModelNotFound(format!(
                "SafeTensors file not found at {}",
                p.display()
            )));
        }
        let bytes = std::fs::read(p).map_err(|e| {
            EngineError::BackendError(format!("Read SafeTensors at {}: {e}", p.display()))
        })?;
        Self::from_safetensors(name, &bytes, config)
    }
}

/// S-LoRA Hot-Swap Manager for Multi-Persona inference with < 1ms switching.
#[derive(Debug, Default)]
pub struct UnslothLoraManager {
    adapters: HashMap<String, UnslothAdapter>,
    active_adapter: Option<String>,
}

impl UnslothLoraManager {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_adapter(&mut self, adapter: UnslothAdapter) {
        self.adapters.insert(adapter.name.clone(), adapter);
    }

    /// Switches the active persona adapter with zero disk I/O (< 1ms latency).
    pub fn set_active_adapter(&mut self, name: Option<&str>) -> bool {
        if let Some(n) = name {
            if self.adapters.contains_key(n) {
                self.active_adapter = Some(n.to_string());
                true
            } else {
                false
            }
        } else {
            self.active_adapter = None;
            true
        }
    }

    #[must_use]
    pub fn active_adapter(&self) -> Option<&UnslothAdapter> {
        self.active_adapter
            .as_ref()
            .and_then(|name| self.adapters.get(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsloth_lora_delta_and_fusion() {
        let pair = UnslothTensorPair::new(4, 4, 2, 4.0);
        let input = [1.0, 1.0, 1.0, 1.0];
        let mut output = [0.0; 4];
        pair.apply_lora_delta(&input, &mut output);
        for &y in &output {
            assert!(y > 0.0);
        }

        let mut delta_w = vec![0.0f32; 16];
        pair.compute_weight_delta(&mut delta_w);
        assert_eq!(delta_w.len(), 16);
    }

    #[test]
    fn test_unsloth_lora_hot_swap() {
        let mut mgr = UnslothLoraManager::new();
        let adapter1 = UnslothAdapter::new("mix_master_engineer", UnslothAdapterConfig::default());
        let adapter2 = UnslothAdapter::new("music_theorist", UnslothAdapterConfig::default());

        mgr.register_adapter(adapter1);
        mgr.register_adapter(adapter2);

        assert!(mgr.set_active_adapter(Some("mix_master_engineer")));
        assert_eq!(mgr.active_adapter().unwrap().name, "mix_master_engineer");

        assert!(mgr.set_active_adapter(Some("music_theorist")));
        assert_eq!(mgr.active_adapter().unwrap().name, "music_theorist");

        assert!(mgr.set_active_adapter(None));
        assert!(mgr.active_adapter().is_none());
    }
}
