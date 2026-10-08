use crate::formats::{GGUF_MAGIC, GgufFile, GgufQuantType, SafeTensorsHeader};
use oxide_core::error::{EngineError, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Tensor descriptor extracted from model metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub quant_type: GgufQuantType,
    pub offset: u64,
    pub size_bytes: usize,
}

/// Universal architecture metadata extracted at runtime from GGUF / SafeTensors.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub arch: String,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub tensor_info: HashMap<String, TensorInfo>,
}

impl ModelMetadata {
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().to_string();
        let hidden_size = gguf
            .get_u64(&format!("{arch}.embedding_length"))
            .unwrap_or(4096) as u32;
        let num_layers = gguf.get_u64(&format!("{arch}.block_count")).unwrap_or(32) as u32;
        let num_heads = gguf
            .get_u64(&format!("{arch}.attention.head_count"))
            .unwrap_or(32) as u32;
        let num_kv_heads = gguf
            .get_u64(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(u64::from(num_heads)) as u32;
        let head_dim = hidden_size.checked_div(num_heads).unwrap_or(128);
        let intermediate_size = gguf
            .get_u64(&format!("{arch}.feed_forward_length"))
            .unwrap_or(u64::from(hidden_size * 4)) as u32;
        let max_seq_len = gguf
            .get_u64(&format!("{arch}.context_length"))
            .unwrap_or(8192) as u32;
        let rope_freq_base = gguf
            .get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(500_000.0);
        let rms_norm_eps = gguf
            .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let mut tensor_info = HashMap::new();
        let mut vocab_size = 128_256u32;

        for (name, info) in &gguf.tensors {
            let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            if name == "token_embd.weight" && !shape.is_empty() {
                vocab_size = shape.iter().max().copied().unwrap_or(128_256) as u32;
            }
            let element_count: usize = shape.iter().product();
            let size_bytes = match info.quant_type {
                GgufQuantType::F32 => element_count * 4,
                GgufQuantType::F16 | GgufQuantType::Bf16 => element_count * 2,
                GgufQuantType::Q4_0 | GgufQuantType::Q4_1 => element_count / 2,
                GgufQuantType::Q8_0 => element_count,
                _ => element_count,
            };

            tensor_info.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    shape,
                    quant_type: info.quant_type,
                    offset: info.offset,
                    size_bytes,
                },
            );
        }

        Self {
            arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_freq_base,
            rms_norm_eps,
            tensor_info,
        }
    }
}

/// Device placement classification for dynamic layer offloading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePlacement {
    Gpu(u32),
    CpuRam,
}

/// Allocates VRAM and CPU buffers dynamically based on `--n-gpu-layers`.
#[derive(Debug, Clone)]
pub struct WeightAllocator {
    pub placements: HashMap<String, DevicePlacement>,
    pub n_gpu_layers: usize,
}

impl WeightAllocator {
    #[must_use]
    pub fn plan(meta: &ModelMetadata, n_gpu_layers: usize) -> Self {
        let mut placements = HashMap::new();
        for name in meta.tensor_info.keys() {
            let layer_idx = extract_layer_index(name);
            let placement = match layer_idx {
                Some(idx) if idx < n_gpu_layers => DevicePlacement::Gpu(idx as u32),
                Some(_) => DevicePlacement::CpuRam,
                None => {
                    if n_gpu_layers >= meta.num_layers as usize {
                        DevicePlacement::Gpu(0)
                    } else {
                        DevicePlacement::CpuRam
                    }
                }
            };
            placements.insert(name.clone(), placement);
        }
        Self {
            placements,
            n_gpu_layers,
        }
    }
}

fn extract_layer_index(name: &str) -> Option<usize> {
    for part in name.split('.') {
        if let Ok(idx) = part.parse::<usize>() {
            return Some(idx);
        }
    }
    None
}

/// Zero-copy Memory-Mapped Model Loader using OS virtual memory paging.
#[derive(Debug)]
pub struct MmapModel {
    pub metadata: ModelMetadata,
    pub mmap: memmap2::Mmap,
    pub tensor_data_offset: usize,
}

impl MmapModel {
    /// Memory-maps the model file without reading weights into heap RAM.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            EngineError::BackendError(format!("Failed to open model file {}: {e}", p.display()))
        })?;

        // Safety: We do not modify the file while mapped.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|e| {
                EngineError::BackendError(format!("Failed to mmap {}: {e}", p.display()))
            })?
        };

        if mmap.len() >= 4 && &mmap[0..4] == GGUF_MAGIC {
            let gguf = GgufFile::parse(&mmap)?;
            let tensor_data_offset = gguf.tensor_data_offset;
            let metadata = ModelMetadata::from_gguf(&gguf);
            Ok(Self {
                metadata,
                mmap,
                tensor_data_offset,
            })
        } else {
            // SafeTensors format
            let (header, header_size) = SafeTensorsHeader::parse_from_bytes(&mmap)?;
            let tensor_data_offset = 8 + header_size;
            let mut tensor_info = HashMap::new();
            for (name, info) in header.tensors {
                let size_bytes = (info.data_offsets.1 - info.data_offsets.0) as usize;
                tensor_info.insert(
                    name.clone(),
                    TensorInfo {
                        name,
                        shape: info.shape,
                        quant_type: GgufQuantType::F32,
                        offset: info.data_offsets.0,
                        size_bytes,
                    },
                );
            }
            let metadata = ModelMetadata {
                arch: "llama".to_string(),
                vocab_size: 128_256,
                hidden_size: 4096,
                intermediate_size: 14336,
                num_layers: 32,
                num_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                max_seq_len: 8192,
                rope_freq_base: 500_000.0,
                rms_norm_eps: 1e-5,
                tensor_info,
            };
            Ok(Self {
                metadata,
                mmap,
                tensor_data_offset,
            })
        }
    }

    /// Slices raw tensor bytes directly from memory-mapped page buffer.
    #[must_use]
    pub fn get_tensor_bytes(&self, tensor_name: &str) -> Option<&[u8]> {
        let info = self.metadata.tensor_info.get(tensor_name)?;
        let start = self.tensor_data_offset + info.offset as usize;
        let end = start + info.size_bytes;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }
}
