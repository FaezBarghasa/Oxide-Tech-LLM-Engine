//! Advanced KV Cache Operations:
//! - Context Shifting (Sliding-window context management & rotary offset updates)
//! - Prompt Caching (Prefix-tree hash matching for instant prompt reuse)
//! - On-the-fly Dynamic KV Cache Quantization (FP8, Q8_0, Q4_0, INT4)
//! - KV Cache Dumping & Reloading to disk/persistent storage

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported On-the-fly KV Cache Quantization Precision Modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum KvQuantizationPrecision {
    #[default]
    Float16,
    Float32,
    Fp8E4M3,
    Quant8_0,
    Quant4_0,
    Int4Packed,
}

/// Dynamic Quantized KV Block supporting in-memory compressed keys and values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedKvBlock {
    pub block_id: u32,
    pub token_count: usize,
    pub head_dim: usize,
    pub precision: KvQuantizationPrecision,
    pub k_scale: f32,
    pub v_scale: f32,
    pub quantized_k_data: Vec<u8>,
    pub quantized_v_data: Vec<u8>,
}

impl QuantizedKvBlock {
    /// Compresses float K/V tensors into quantized block on-the-fly.
    #[must_use]
    pub fn quantize_from_f32(
        block_id: u32,
        k: &[f32],
        v: &[f32],
        head_dim: usize,
        precision: KvQuantizationPrecision,
    ) -> Self {
        assert_eq!(k.len(), v.len());
        let token_count = if head_dim > 0 { k.len() / head_dim } else { 0 };

        match precision {
            KvQuantizationPrecision::Float32 => {
                let mut k_bytes = Vec::with_capacity(k.len() * 4);
                let mut v_bytes = Vec::with_capacity(v.len() * 4);
                for &val in k {
                    k_bytes.extend_from_slice(&val.to_le_bytes());
                }
                for &val in v {
                    v_bytes.extend_from_slice(&val.to_le_bytes());
                }
                Self {
                    block_id,
                    token_count,
                    head_dim,
                    precision,
                    k_scale: 1.0,
                    v_scale: 1.0,
                    quantized_k_data: k_bytes,
                    quantized_v_data: v_bytes,
                }
            }
            KvQuantizationPrecision::Quant8_0 | KvQuantizationPrecision::Fp8E4M3 => {
                let max_k = k
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, x| m.max(x.abs()))
                    .max(1e-6);
                let max_v = v
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, x| m.max(x.abs()))
                    .max(1e-6);
                let k_scale = max_k / 127.0;
                let v_scale = max_v / 127.0;

                let k_bytes: Vec<u8> = k
                    .iter()
                    .map(|&x| ((x / k_scale).round().clamp(-128.0, 127.0) as i8) as u8)
                    .collect();
                let v_bytes: Vec<u8> = v
                    .iter()
                    .map(|&x| ((x / v_scale).round().clamp(-128.0, 127.0) as i8) as u8)
                    .collect();

                Self {
                    block_id,
                    token_count,
                    head_dim,
                    precision,
                    k_scale,
                    v_scale,
                    quantized_k_data: k_bytes,
                    quantized_v_data: v_bytes,
                }
            }
            KvQuantizationPrecision::Quant4_0 | KvQuantizationPrecision::Int4Packed => {
                let max_k = k
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, x| m.max(x.abs()))
                    .max(1e-6);
                let max_v = v
                    .iter()
                    .copied()
                    .fold(0.0f32, |m, x| m.max(x.abs()))
                    .max(1e-6);
                let k_scale = max_k / 7.0;
                let v_scale = max_v / 7.0;

                let mut k_bytes = Vec::with_capacity((k.len() + 1) / 2);
                let mut v_bytes = Vec::with_capacity((v.len() + 1) / 2);

                for chunk in k.chunks(2) {
                    let q0 = ((chunk[0] / k_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                    let q1 = if chunk.len() > 1 {
                        ((chunk[1] / k_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
                    } else {
                        8
                    };
                    k_bytes.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }

                for chunk in v.chunks(2) {
                    let q0 = ((chunk[0] / v_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                    let q1 = if chunk.len() > 1 {
                        ((chunk[1] / v_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8
                    } else {
                        8
                    };
                    v_bytes.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                }

                Self {
                    block_id,
                    token_count,
                    head_dim,
                    precision,
                    k_scale,
                    v_scale,
                    quantized_k_data: k_bytes,
                    quantized_v_data: v_bytes,
                }
            }
            KvQuantizationPrecision::Float16 => {
                let mut k_bytes = Vec::with_capacity(k.len() * 2);
                let mut v_bytes = Vec::with_capacity(v.len() * 2);
                for &val in k {
                    let f16_bits = f32_to_f16_bits(val);
                    k_bytes.extend_from_slice(&f16_bits.to_le_bytes());
                }
                for &val in v {
                    let f16_bits = f32_to_f16_bits(val);
                    v_bytes.extend_from_slice(&f16_bits.to_le_bytes());
                }
                Self {
                    block_id,
                    token_count,
                    head_dim,
                    precision,
                    k_scale: 1.0,
                    v_scale: 1.0,
                    quantized_k_data: k_bytes,
                    quantized_v_data: v_bytes,
                }
            }
        }
    }

    /// Dequantizes compressed block back into f32 memory.
    pub fn dequantize_into(&self, k_out: &mut [f32], v_out: &mut [f32]) {
        match self.precision {
            KvQuantizationPrecision::Float32 => {
                for (i, chunk) in self.quantized_k_data.chunks_exact(4).enumerate() {
                    if i < k_out.len() {
                        k_out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
                for (i, chunk) in self.quantized_v_data.chunks_exact(4).enumerate() {
                    if i < v_out.len() {
                        v_out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
                    }
                }
            }
            KvQuantizationPrecision::Quant8_0 | KvQuantizationPrecision::Fp8E4M3 => {
                for (i, &b) in self.quantized_k_data.iter().enumerate() {
                    if i < k_out.len() {
                        k_out[i] = (b as i8 as f32) * self.k_scale;
                    }
                }
                for (i, &b) in self.quantized_v_data.iter().enumerate() {
                    if i < v_out.len() {
                        v_out[i] = (b as i8 as f32) * self.v_scale;
                    }
                }
            }
            KvQuantizationPrecision::Quant4_0 | KvQuantizationPrecision::Int4Packed => {
                for (i, &b) in self.quantized_k_data.iter().enumerate() {
                    let q0 = (b & 0x0F) as i8 - 8;
                    let q1 = ((b >> 4) & 0x0F) as i8 - 8;
                    if i * 2 < k_out.len() {
                        k_out[i * 2] = (q0 as f32) * self.k_scale;
                    }
                    if i * 2 + 1 < k_out.len() {
                        k_out[i * 2 + 1] = (q1 as f32) * self.k_scale;
                    }
                }
                for (i, &b) in self.quantized_v_data.iter().enumerate() {
                    let q0 = (b & 0x0F) as i8 - 8;
                    let q1 = ((b >> 4) & 0x0F) as i8 - 8;
                    if i * 2 < v_out.len() {
                        v_out[i * 2] = (q0 as f32) * self.v_scale;
                    }
                    if i * 2 + 1 < v_out.len() {
                        v_out[i * 2 + 1] = (q1 as f32) * self.v_scale;
                    }
                }
            }
            KvQuantizationPrecision::Float16 => {
                for (i, chunk) in self.quantized_k_data.chunks_exact(2).enumerate() {
                    if i < k_out.len() {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        k_out[i] = f16_bits_to_f32(bits);
                    }
                }
                for (i, chunk) in self.quantized_v_data.chunks_exact(2).enumerate() {
                    if i < v_out.len() {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        v_out[i] = f16_bits_to_f32(bits);
                    }
                }
            }
        }
    }
}

#[inline]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let frac = bits & 0x7F_FFFF;

    if exp == 0 {
        return (sign as u16) << 15;
    }
    if exp == 0xFF {
        return ((sign as u16) << 15) | 0x7C00 | if frac != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp as i32 - 127 + 15;
    if new_exp >= 31 {
        return ((sign as u16) << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        return (sign as u16) << 15;
    }

    let new_frac = (frac >> 13) as u16;
    ((sign as u16) << 15) | ((new_exp as u16) << 10) | new_frac
}

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let frac = bits & 0x03FF;

    if exp == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 31 {
        return if frac == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }

    let new_exp = (exp as u32 + 127 - 15) << 23;
    let new_frac = (frac as u32) << 13;
    let new_sign = (sign as u32) << 31;
    f32::from_bits(new_sign | new_exp | new_frac)
}

/// Context Shifting Engine (Sliding-window context truncation preserving prompt prefix).
#[derive(Debug, Clone)]
pub struct ContextShiftManager {
    pub max_context_length: usize,
    pub preserved_prefix_tokens: usize,
    pub shift_step_size: usize,
}

impl ContextShiftManager {
    #[must_use]
    pub fn new(
        max_context_length: usize,
        preserved_prefix_tokens: usize,
        shift_step_size: usize,
    ) -> Self {
        Self {
            max_context_length,
            preserved_prefix_tokens,
            shift_step_size,
        }
    }

    /// Computes shift offset when active sequence exceeds maximum context length.
    pub fn shift_context_window(&self, current_tokens: &[u32]) -> Result<Vec<u32>> {
        if current_tokens.len() <= self.max_context_length {
            return Ok(current_tokens.to_vec());
        }

        let prefix_len = self.preserved_prefix_tokens.min(current_tokens.len());
        let prefix = &current_tokens[..prefix_len];

        let tokens_to_drop = self
            .shift_step_size
            .max(current_tokens.len() - self.max_context_length);
        let start_recent = (prefix_len + tokens_to_drop).min(current_tokens.len());
        let recent = &current_tokens[start_recent..];

        let mut shifted = Vec::with_capacity(prefix.len() + recent.len());
        shifted.extend_from_slice(prefix);
        shifted.extend_from_slice(recent);
        Ok(shifted)
    }
}

/// Prompt Prefix Cache Registry with Hash Matching.
#[derive(Debug, Clone, Default)]
pub struct PromptCacheRegistry {
    prefix_map: HashMap<u64, Vec<u32>>, // Hash -> Block IDs
}

impl PromptCacheRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            prefix_map: HashMap::new(),
        }
    }

    /// Computes fast 64-bit token sequence prefix hash.
    #[must_use]
    pub fn compute_prefix_hash(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Registers cached block IDs for given prompt prefix.
    pub fn insert_cached_prefix(&mut self, prompt: &[u32], block_ids: &[u32]) {
        let hash = Self::compute_prefix_hash(prompt);
        self.prefix_map.insert(hash, block_ids.to_vec());
    }

    /// Looks up cached block IDs for given prompt prefix.
    #[must_use]
    pub fn lookup_cached_prefix(&self, prompt: &[u32]) -> Option<&[u32]> {
        let hash = Self::compute_prefix_hash(prompt);
        self.prefix_map.get(&hash).map(|v| v.as_slice())
    }
}

/// KV Cache Dumping and Reloading Utility for Session Persistence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCacheDumpContainer {
    pub session_id: String,
    pub model_identifier: String,
    pub total_blocks: usize,
    pub blocks: Vec<QuantizedKvBlock>,
    pub timestamp_epoch_ms: u64,
}

impl KvCacheDumpContainer {
    /// Serializes KV cache blocks into binary payload.
    pub fn dump_to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| EngineError::BackendError(format!("Failed to serialize KV dump: {e}")))
    }

    /// Reloads KV cache blocks from binary payload.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| EngineError::BackendError(format!("Failed to deserialize KV dump: {e}")))
    }
}
