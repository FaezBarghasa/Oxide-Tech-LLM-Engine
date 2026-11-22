//! Unsloth Safetensors Adapter Loader.
//!
//! Directly loads fine-tuned Unsloth LoRA checkpoints (`adapter_config.json` + `adapter_model.safetensors`)
//! without Python tooling or offline merge scripts.

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Unsloth `adapter_config.json` schema.
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
}

/// In-memory Unsloth LoRA Adapter holding target module tensor pairs.
#[derive(Debug, Clone)]
pub struct UnslothAdapter {
    pub name: String,
    pub config: UnslothAdapterConfig,
    pub layers: HashMap<String, UnslothTensorPair>,
}

impl UnslothAdapter {
    #[must_use]
    pub fn new(name: impl Into<String>, config: UnslothAdapterConfig) -> Self {
        Self {
            name: name.into(),
            config,
            layers: HashMap::new(),
        }
    }

    /// Loads an adapter directly from directory containing `adapter_config.json` and `adapter_model.safetensors`.
    pub fn from_directory(name: impl Into<String>, dir_path: impl AsRef<Path>) -> Result<Self> {
        let dir = dir_path.as_ref();
        let config_path = dir.join("adapter_config.json");
        let safetensors_path = dir.join("adapter_model.safetensors");

        let config = if config_path.exists() {
            let data = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::BackendError(format!("Read adapter_config: {e}")))?;
            serde_json::from_str(&data)
                .map_err(|e| EngineError::BackendError(format!("Parse adapter_config: {e}")))?
        } else {
            UnslothAdapterConfig::default()
        };

        let mut adapter = Self::new(name, config);

        if safetensors_path.exists() {
            let file = std::fs::File::open(&safetensors_path)
                .map_err(|e| EngineError::BackendError(format!("Open safetensors: {e}")))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| EngineError::BackendError(format!("Mmap safetensors: {e}")))?;
            let tensors = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| EngineError::BackendError(format!("Deserialize safetensors: {e}")))?;

            for (tensor_name, view) in tensors.tensors() {
                if tensor_name.ends_with(".lora_A.weight") {
                    let base_name = tensor_name.trim_end_matches(".lora_A.weight");
                    let b_name = format!("{base_name}.lora_B.weight");
                    if let Ok(b_view) = tensors.tensor(&b_name) {
                        let rank = view.shape()[0];
                        let in_dim = view.shape()[1];
                        let out_dim = b_view.shape()[0];
                        let mut pair = UnslothTensorPair::new(
                            in_dim,
                            out_dim,
                            rank,
                            adapter.config.lora_alpha,
                        );
                        let a_bytes = view.data();
                        let b_bytes = b_view.data();
                        if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(a_bytes) {
                            pair.lora_a.copy_from_slice(slice);
                        } else {
                            for (chunk, out) in a_bytes.chunks_exact(4).zip(pair.lora_a.iter_mut()) {
                                *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        }
                        if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(b_bytes) {
                            pair.lora_b.copy_from_slice(slice);
                        } else {
                            for (chunk, out) in b_bytes.chunks_exact(4).zip(pair.lora_b.iter_mut()) {
                                *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        }
                        adapter.layers.insert(base_name.to_string(), pair);
                    }
                }
            }
        }

        Ok(adapter)
    }

    /// Loads an adapter from a standalone safetensors file with explicit config.
    pub fn from_safetensors_file(
        name: impl Into<String>,
        file_path: impl AsRef<Path>,
        config: UnslothAdapterConfig,
    ) -> Result<Self> {
        let path = file_path.as_ref();
        if path.is_dir() {
            return Self::from_directory(name, path);
        }

        let mut adapter = Self::new(name, config);
        let file = std::fs::File::open(path)
            .map_err(|e| EngineError::BackendError(format!("Open safetensors: {e}")))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| EngineError::BackendError(format!("Mmap safetensors: {e}")))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| EngineError::BackendError(format!("Deserialize safetensors: {e}")))?;

        for (tensor_name, view) in tensors.tensors() {
            if tensor_name.ends_with(".lora_A.weight") {
                let base_name = tensor_name.trim_end_matches(".lora_A.weight");
                let b_name = format!("{base_name}.lora_B.weight");
                if let Ok(b_view) = tensors.tensor(&b_name) {
                    let rank = view.shape()[0];
                    let in_dim = view.shape()[1];
                    let out_dim = b_view.shape()[0];
                    let mut pair = UnslothTensorPair::new(
                        in_dim,
                        out_dim,
                        rank,
                        adapter.config.lora_alpha,
                    );
                    let a_bytes = view.data();
                    let b_bytes = b_view.data();
                    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(a_bytes) {
                        pair.lora_a.copy_from_slice(slice);
                    } else {
                        for (chunk, out) in a_bytes.chunks_exact(4).zip(pair.lora_a.iter_mut()) {
                            *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                    }
                    if let Ok(slice) = bytemuck::try_cast_slice::<u8, f32>(b_bytes) {
                        pair.lora_b.copy_from_slice(slice);
                    } else {
                        for (chunk, out) in b_bytes.chunks_exact(4).zip(pair.lora_b.iter_mut()) {
                            *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        }
                    }
                    adapter.layers.insert(base_name.to_string(), pair);
                }
            }
        }

        Ok(adapter)
    }

    /// Serializes adapter tensor pairs to aligned Safetensors binary format.
    pub fn save_safetensors_bytes(&self) -> Result<Vec<u8>> {
        let mut header_entries = Vec::new();
        let mut data_buffer = Vec::new();

        for (name, pair) in &self.layers {
            let a_name = format!("{name}.lora_A.weight");
            let a_bytes: &[u8] = bytemuck::cast_slice(&pair.lora_a);
            let a_start = data_buffer.len();
            data_buffer.extend_from_slice(a_bytes);
            let a_end = data_buffer.len();

            header_entries.push(format!(
                r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
                a_name, pair.rank, pair.in_dim, a_start, a_end
            ));

            let b_name = format!("{name}.lora_B.weight");
            let b_bytes: &[u8] = bytemuck::cast_slice(&pair.lora_b);
            let b_start = data_buffer.len();
            data_buffer.extend_from_slice(b_bytes);
            let b_end = data_buffer.len();

            header_entries.push(format!(
                r#""{}":{{"dtype":"F32","shape":[{},{}],"data_offsets":[{},{}]}}"#,
                b_name, pair.out_dim, pair.rank, b_start, b_end
            ));
        }

        let mut header_json = format!("{{{}}}", header_entries.join(","));
        // Pad header so (8 + header_bytes.len()) % 8 == 0 for SafeTensors data alignment
        let rem = (8 + header_json.len()) % 8;
        if rem != 0 {
            let pad = 8 - rem;
            header_json.extend(std::iter::repeat(' ').take(pad));
        }

        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buffer.len());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&data_buffer);

        Ok(out)
    }
}
