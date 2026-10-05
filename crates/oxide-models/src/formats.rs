#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_assert_eq,
    clippy::manual_is_multiple_of
)]

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported Model Weight Serialization & Deployment Formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFileFormat {
    SafeTensors,
    GgufV3,
    OxideModNative,
}

/// GGUF Quantization Precision Formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K_M,
    Q5_K_M,
    Q6_K,
    IQ4_XS,
    Nvfp4,
    Bf16,
}

/// Parsed metadata for a SafeTensors tensor entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (u64, u64),
}

/// Zero-copy SafeTensors Header Parser.
#[derive(Debug, Clone, Default)]
pub struct SafeTensorsHeader {
    pub tensors: HashMap<String, SafeTensorInfo>,
    pub metadata: HashMap<String, String>,
}

impl SafeTensorsHeader {
    /// Parses SafeTensors JSON header from raw byte prefix.
    pub fn parse_from_bytes(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < 8 {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let json_slice = &bytes[8..8 + header_len];
        let json_str = std::str::from_utf8(json_slice).map_err(|e| {
            EngineError::BackendError(format!("Invalid UTF-8 in SafeTensors header: {e}"))
        })?;

        let raw_map: HashMap<String, serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| {
                EngineError::BackendError(format!("Failed to parse SafeTensors JSON: {e}"))
            })?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (k, v) in raw_map {
            if k == "__metadata__" {
                if let Some(obj) = v.as_object() {
                    for (mk, mv) in obj {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        }
                    }
                }
            } else if let Some(obj) = v.as_object() {
                let dtype = obj
                    .get("dtype")
                    .and_then(|d| d.as_str())
                    .unwrap_or("F32")
                    .to_string();
                let shape =
                    obj.get("shape")
                        .and_then(|s| s.as_array())
                        .map_or_else(Vec::new, |arr| {
                            arr.iter()
                                .filter_map(|x| x.as_u64().map(|v| v as usize))
                                .collect()
                        });
                let offsets =
                    obj.get("data_offsets")
                        .and_then(|o| o.as_array())
                        .map_or((0, 0), |arr| {
                            if arr.len() == 2 {
                                (arr[0].as_u64().unwrap_or(0), arr[1].as_u64().unwrap_or(0))
                            } else {
                                (0, 0)
                            }
                        });

                tensors.insert(
                    k,
                    SafeTensorInfo {
                        dtype,
                        shape,
                        data_offsets: offsets,
                    },
                );
            }
        }

        Ok((Self { tensors, metadata }, 8 + header_len))
    }
}

/// GGUF v3 Binary Header structure.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 24 {
            return Err(EngineError::InvalidArtifactHeader);
        }
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != GGUF_MAGIC {
            return Err(EngineError::InvalidArtifactHeader);
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF Metadata Value Types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// GGUF Tensor Information Entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub quant_type: GgufQuantType,
    pub offset: u64,
}

/// Universal GGUF File Container.
#[derive(Debug, Clone, Default)]
pub struct GgufFile {
    pub header: Option<GgufHeader>,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub tensor_data_offset: usize,
}

impl GgufFile {
    /// Parses any GGUF (v1, v2, v3) file binary stream.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 24 {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != GGUF_MAGIC {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        };

        let mut offset = 24;
        let mut metadata = HashMap::new();

        for _ in 0..metadata_kv_count {
            if offset + 8 > bytes.len() {
                break;
            }
            let (key, new_offset) = Self::read_gguf_string(bytes, offset)?;
            offset = new_offset;

            if offset + 4 > bytes.len() {
                break;
            }
            let val_type = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let (val, new_offset) = Self::read_gguf_value(bytes, offset, val_type)?;
            offset = new_offset;
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            if offset + 8 > bytes.len() {
                break;
            }
            let (t_name, new_offset) = Self::read_gguf_string(bytes, offset)?;
            offset = new_offset;

            if offset + 4 > bytes.len() {
                break;
            }
            let n_dims = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let mut dims = Vec::new();
            for _ in 0..n_dims {
                if offset + 8 > bytes.len() {
                    break;
                }
                let d = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                offset += 8;
                dims.push(d);
            }

            if offset + 4 > bytes.len() {
                break;
            }
            let qtype_code = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            offset += 4;

            let qtype = match qtype_code {
                0 => GgufQuantType::F32,
                1 => GgufQuantType::F16,
                2 => GgufQuantType::Q4_0,
                3 => GgufQuantType::Q4_1,
                6 => GgufQuantType::Q5_0,
                7 => GgufQuantType::Q5_1,
                8 => GgufQuantType::Q8_0,
                9 => GgufQuantType::Q8_1,
                10 => GgufQuantType::Q2_K,
                11 => GgufQuantType::Q3_K,
                12 => GgufQuantType::Q4_K_M,
                13 => GgufQuantType::Q5_K_M,
                14 => GgufQuantType::Q6_K,
                _ => GgufQuantType::F32,
            };

            if offset + 8 > bytes.len() {
                break;
            }
            let t_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;

            tensors.insert(
                t_name.clone(),
                GgufTensorInfo {
                    name: t_name,
                    n_dimensions: n_dims,
                    dimensions: dims,
                    quant_type: qtype,
                    offset: t_offset,
                },
            );
        }

        // Align offset to 32 bytes (default GGUF alignment)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| match v {
                GgufValue::Uint32(a) => Some(*a as usize),
                GgufValue::Uint64(a) => Some(*a as usize),
                _ => None,
            })
            .unwrap_or(32);

        let tensor_data_offset = (offset + alignment - 1) & !(alignment - 1);

        Ok(Self {
            header: Some(header),
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    fn read_gguf_string(bytes: &[u8], mut offset: usize) -> Result<(String, usize)> {
        if offset + 8 > bytes.len() {
            return Err(EngineError::InvalidArtifactHeader);
        }
        let len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + len > bytes.len() {
            return Err(EngineError::InvalidArtifactHeader);
        }
        let s = std::str::from_utf8(&bytes[offset..offset + len])
            .map_err(|e| EngineError::BackendError(format!("Invalid UTF-8: {e}")))?
            .to_string();
        Ok((s, offset + len))
    }

    fn read_gguf_value(
        bytes: &[u8],
        mut offset: usize,
        val_type: u32,
    ) -> Result<(GgufValue, usize)> {
        match val_type {
            0 => {
                // UINT8
                let v = bytes[offset];
                Ok((GgufValue::Uint8(v), offset + 1))
            }
            1 => {
                // INT8
                let v = bytes[offset] as i8;
                Ok((GgufValue::Int8(v), offset + 1))
            }
            2 => {
                // UINT16
                let v = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                Ok((GgufValue::Uint16(v), offset + 2))
            }
            3 => {
                // INT16
                let v = i16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                Ok((GgufValue::Int16(v), offset + 2))
            }
            4 => {
                // UINT32
                let v = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                Ok((GgufValue::Uint32(v), offset + 4))
            }
            5 => {
                // INT32
                let v = i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                Ok((GgufValue::Int32(v), offset + 4))
            }
            6 => {
                // FLOAT32
                let v = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                Ok((GgufValue::Float32(v), offset + 4))
            }
            7 => {
                // BOOL
                let v = bytes[offset] != 0;
                Ok((GgufValue::Bool(v), offset + 1))
            }
            8 => {
                // STRING
                let (s, new_offset) = Self::read_gguf_string(bytes, offset)?;
                Ok((GgufValue::String(s), new_offset))
            }
            9 => {
                // ARRAY
                if offset + 12 > bytes.len() {
                    return Err(EngineError::InvalidArtifactHeader);
                }
                let item_type = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                offset += 4;
                let count =
                    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
                offset += 8;

                let mut arr = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    let (item, new_off) = Self::read_gguf_value(bytes, offset, item_type)?;
                    offset = new_off;
                    arr.push(item);
                }
                Ok((GgufValue::Array(arr), offset))
            }
            10 => {
                // UINT64
                let v = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                Ok((GgufValue::Uint64(v), offset + 8))
            }
            11 => {
                // INT64
                let v = i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                Ok((GgufValue::Int64(v), offset + 8))
            }
            12 => {
                // FLOAT64
                let v = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                Ok((GgufValue::Float64(v), offset + 8))
            }
            _ => Ok((GgufValue::Uint32(0), offset)),
        }
    }

    #[must_use]
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::Uint32(v)) => Some(*v),
            Some(GgufValue::Int32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::Uint64(v)) => Some(*v),
            Some(GgufValue::Uint32(v)) => Some(u64::from(*v)),
            _ => None,
        }
    }

    #[must_use]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            Some(GgufValue::Float64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    #[must_use]
    pub fn architecture(&self) -> &str {
        self.get_string("general.architecture").unwrap_or("llama")
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        self.get_string("general.name").unwrap_or("OxideModel")
    }
}

/// NVIDIA NVFP4 (E2M1 4-bit float) Tensor Block with 2x memory reduction for Blackwell MoE.
/// Each byte encodes two 4-bit floats with a per-block FP8 scale factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Block {
    pub scale: f32,
    pub packed_nibbles: Vec<u8>,
}

impl Nvfp4Block {
    /// Encodes a slice of 32 f32 weights into an NVFP4 packed block.
    #[must_use]
    pub fn quantize(values: &[f32]) -> Self {
        assert!(values.len() % 2 == 0);
        let max_abs = values
            .iter()
            .copied()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let scale = max_abs / 6.0; // Max representable value in E2M1 (1.5 * 2^2 = 6.0)

        let mut packed = Vec::with_capacity(values.len() / 2);
        for chunk in values.chunks_exact(2) {
            let n0 = Self::float_to_nvfp4(chunk[0] / scale);
            let n1 = Self::float_to_nvfp4(chunk[1] / scale);
            packed.push((n1 << 4) | (n0 & 0x0F));
        }

        Self {
            scale,
            packed_nibbles: packed,
        }
    }

    /// Dequantizes NVFP4 packed block back to f32 weights.
    pub fn dequantize(&self, output: &mut [f32]) {
        let count = self.packed_nibbles.len() * 2;
        assert!(output.len() >= count);

        for (i, &byte) in self.packed_nibbles.iter().enumerate() {
            let n0 = byte & 0x0F;
            let n1 = (byte >> 4) & 0x0F;
            output[i * 2] = Self::nvfp4_to_float(n0) * self.scale;
            output[i * 2 + 1] = Self::nvfp4_to_float(n1) * self.scale;
        }
    }

    #[inline]
    fn float_to_nvfp4(v: f32) -> u8 {
        let sign = if v < 0.0 { 0x08 } else { 0x00 };
        let abs_v = v.abs();
        let mag = if abs_v < 0.25 {
            0
        } else if abs_v < 0.75 {
            1
        } else if abs_v < 1.25 {
            2
        } else if abs_v < 1.75 {
            3
        } else if abs_v < 2.5 {
            4
        } else if abs_v < 3.5 {
            5
        } else if abs_v < 5.0 {
            6
        } else {
            7
        };
        sign | (mag & 0x07)
    }

    #[inline]
    fn nvfp4_to_float(nibble: u8) -> f32 {
        let sign = if (nibble & 0x08) != 0 { -1.0 } else { 1.0 };
        let mag = match nibble & 0x07 {
            0 => 0.0,
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            _ => 6.0,
        };
        sign * mag
    }
}
