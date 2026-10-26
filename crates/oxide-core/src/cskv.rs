//! CSKV: High-performance flat binary KV-cache state I/O.
//!
//! Provides zero-copy header layout, byte-range mapping, and streaming
//! persistence of KV cache states across dense, hybrid (Attention + GatedDeltaNet),
//! and Multi-Head Latent Attention (MLA) architectures.

use crate::error::{EngineError, Result};
use std::fmt;

/// Magic bytes identifying a CSKV file.
pub const CSKV_MAGIC: &[u8; 4] = b"CSKV";

/// Supported CSKV format versions.
pub const VERSION_DENSE_V1: u32 = 1;
pub const VERSION_HYBRID: u32 = 2;
pub const VERSION_MLA: u32 = 3;
pub const VERSION_DENSE_V4: u32 = 4;
pub const VERSION_DENSE_F16: u32 = 5;
pub const VERSION_HYBRID_F16: u32 = 6;
pub const VERSION_MLA_F16: u32 = 7;

/// Upper limit on layers to prevent catastrophic allocation from corrupt files.
pub const MAX_LAYERS: usize = 1 << 16;

/// Element data type stored in the cache file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvDtype {
    #[default]
    F32,
    F16,
}

impl KvDtype {
    #[inline]
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

/// Identifies the sublayer type for hybrid architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    GatedAttention = 0,
    GatedDeltaNet = 1,
}

impl LayerKind {
    #[inline]
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::GatedAttention),
            1 => Ok(Self::GatedDeltaNet),
            other => Err(EngineError::BackendError(format!(
                "invalid CSKV layer kind: {other}"
            ))),
        }
    }
}

/// Buffer role within a cache layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CskvBufferKind {
    Key,
    Value,
    ConvState,
    Recurrent,
    MlaLatent,
}

/// Specific architectural variant for CSKV cache states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CskvVariant {
    Dense {
        num_kv_heads: usize,
        head_dim: usize,
        space_prefix_next: bool,
    },
    Hybrid {
        attn_num_kv_heads: usize,
        attn_head_dim: usize,
        gdn_conv_state_len: usize,
        gdn_recurrent_len: usize,
        layer_kinds: Vec<LayerKind>,
    },
    Mla {
        qk_dim: usize,
    },
}

/// Parsed CSKV header metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CskvHeader {
    pub version: u32,
    pub num_layers: usize,
    pub seq_len: usize,
    pub dtype: KvDtype,
    pub variant: CskvVariant,
}

/// A slice range inside a CSKV file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CskvFileRange {
    pub layer_idx: usize,
    pub kind: CskvBufferKind,
    pub offset: u64,
    pub num_elements: usize,
    pub dtype: KvDtype,
}

impl CskvFileRange {
    #[inline]
    #[must_use]
    pub const fn byte_length(&self) -> usize {
        self.num_elements * self.dtype.byte_size()
    }
}

/// Precomputed memory-mapped or streaming layout of all buffer ranges in a CSKV file.
#[derive(Debug, Clone)]
pub struct CskvLayout {
    pub header: CskvHeader,
    pub header_bytes_len: usize,
    pub total_file_bytes: usize,
    pub ranges: Vec<CskvFileRange>,
}

impl CskvHeader {
    /// Parses a CSKV header from the start of a byte slice.
    ///
    /// Returns the parsed header and the number of bytes consumed by the header.
    pub fn parse(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < 8 {
            return Err(EngineError::InvalidArtifactHeader);
        }

        if &bytes[0..4] != CSKV_MAGIC {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let (dtype, is_dense, is_hybrid, is_mla) = match version {
            VERSION_DENSE_V1 | VERSION_DENSE_V4 => (KvDtype::F32, true, false, false),
            VERSION_HYBRID => (KvDtype::F32, false, true, false),
            VERSION_MLA => (KvDtype::F32, false, false, true),
            VERSION_DENSE_F16 => (KvDtype::F16, true, false, false),
            VERSION_HYBRID_F16 => (KvDtype::F16, false, true, false),
            VERSION_MLA_F16 => (KvDtype::F16, false, false, true),
            other => {
                return Err(EngineError::BackendError(format!(
                    "unsupported CSKV version: {other}"
                )));
            }
        };

        if is_dense {
            let min_header_len = if version >= VERSION_DENSE_V4 { 25 } else { 24 };
            if bytes.len() < min_header_len {
                return Err(EngineError::InvalidArtifactHeader);
            }
            let num_layers = u32::from_le_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let seq_len = u32::from_le_bytes(
                bytes[12..16]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let num_kv_heads = u32::from_le_bytes(
                bytes[16..20]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let head_dim = u32::from_le_bytes(
                bytes[20..24]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let space_prefix_next = if version >= VERSION_DENSE_V4 {
                bytes[24] != 0
            } else {
                false
            };

            if num_layers == 0 || num_layers > MAX_LAYERS {
                return Err(EngineError::BackendError(format!(
                    "invalid layer count: {num_layers}"
                )));
            }

            Ok((
                Self {
                    version,
                    num_layers,
                    seq_len,
                    dtype,
                    variant: CskvVariant::Dense {
                        num_kv_heads,
                        head_dim,
                        space_prefix_next,
                    },
                },
                min_header_len,
            ))
        } else if is_hybrid {
            if bytes.len() < 32 {
                return Err(EngineError::InvalidArtifactHeader);
            }
            let num_layers = u32::from_le_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let seq_len = u32::from_le_bytes(
                bytes[12..16]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let attn_num_kv_heads = u32::from_le_bytes(
                bytes[16..20]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let attn_head_dim = u32::from_le_bytes(
                bytes[20..24]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let gdn_conv_state_len = u32::from_le_bytes(
                bytes[24..28]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let gdn_recurrent_len = u32::from_le_bytes(
                bytes[28..32]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;

            if num_layers == 0 || num_layers > MAX_LAYERS {
                return Err(EngineError::BackendError(format!(
                    "invalid layer count: {num_layers}"
                )));
            }

            let total_header_len = 32 + num_layers;
            if bytes.len() < total_header_len {
                return Err(EngineError::InvalidArtifactHeader);
            }

            let mut layer_kinds = Vec::with_capacity(num_layers);
            for &b in &bytes[32..32 + num_layers] {
                layer_kinds.push(LayerKind::from_u8(b)?);
            }

            Ok((
                Self {
                    version,
                    num_layers,
                    seq_len,
                    dtype,
                    variant: CskvVariant::Hybrid {
                        attn_num_kv_heads,
                        attn_head_dim,
                        gdn_conv_state_len,
                        gdn_recurrent_len,
                        layer_kinds,
                    },
                },
                total_header_len,
            ))
        } else if is_mla {
            if bytes.len() < 20 {
                return Err(EngineError::InvalidArtifactHeader);
            }
            let num_layers = u32::from_le_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let seq_len = u32::from_le_bytes(
                bytes[12..16]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;
            let qk_dim = u32::from_le_bytes(
                bytes[16..20]
                    .try_into()
                    .map_err(|_| EngineError::InvalidArtifactHeader)?,
            ) as usize;

            if num_layers == 0 || num_layers > MAX_LAYERS {
                return Err(EngineError::BackendError(format!(
                    "invalid layer count: {num_layers}"
                )));
            }

            Ok((
                Self {
                    version,
                    num_layers,
                    seq_len,
                    dtype,
                    variant: CskvVariant::Mla { qk_dim },
                },
                20,
            ))
        } else {
            Err(EngineError::InvalidArtifactHeader)
        }
    }

    /// Serializes this header into binary format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CSKV_MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.num_layers as u32).to_le_bytes());
        out.extend_from_slice(&(self.seq_len as u32).to_le_bytes());

        match &self.variant {
            CskvVariant::Dense {
                num_kv_heads,
                head_dim,
                space_prefix_next,
            } => {
                out.extend_from_slice(&(*num_kv_heads as u32).to_le_bytes());
                out.extend_from_slice(&(*head_dim as u32).to_le_bytes());
                if self.version >= VERSION_DENSE_V4 {
                    out.push(u8::from(*space_prefix_next));
                }
            }
            CskvVariant::Hybrid {
                attn_num_kv_heads,
                attn_head_dim,
                gdn_conv_state_len,
                gdn_recurrent_len,
                layer_kinds,
            } => {
                out.extend_from_slice(&(*attn_num_kv_heads as u32).to_le_bytes());
                out.extend_from_slice(&(*attn_head_dim as u32).to_le_bytes());
                out.extend_from_slice(&(*gdn_conv_state_len as u32).to_le_bytes());
                out.extend_from_slice(&(*gdn_recurrent_len as u32).to_le_bytes());
                for kind in layer_kinds {
                    out.push(*kind as u8);
                }
            }
            CskvVariant::Mla { qk_dim } => {
                out.extend_from_slice(&(*qk_dim as u32).to_le_bytes());
            }
        }
        out
    }

    /// Computes the complete layout of all buffer ranges without allocating memory buffers.
    pub fn compute_layout(&self) -> Result<CskvLayout> {
        let header_bytes = self.to_bytes();
        let header_bytes_len = header_bytes.len();
        let mut offset = header_bytes_len as u64;
        let mut ranges = Vec::new();

        match &self.variant {
            CskvVariant::Dense {
                num_kv_heads,
                head_dim,
                ..
            } => {
                let elems_per_buf = self.seq_len * num_kv_heads * head_dim;
                let buf_bytes = elems_per_buf * self.dtype.byte_size();

                for layer_idx in 0..self.num_layers {
                    ranges.push(CskvFileRange {
                        layer_idx,
                        kind: CskvBufferKind::Key,
                        offset,
                        num_elements: elems_per_buf,
                        dtype: self.dtype,
                    });
                    offset += buf_bytes as u64;

                    ranges.push(CskvFileRange {
                        layer_idx,
                        kind: CskvBufferKind::Value,
                        offset,
                        num_elements: elems_per_buf,
                        dtype: self.dtype,
                    });
                    offset += buf_bytes as u64;
                }
            }
            CskvVariant::Hybrid {
                attn_num_kv_heads,
                attn_head_dim,
                gdn_conv_state_len,
                gdn_recurrent_len,
                layer_kinds,
            } => {
                if layer_kinds.len() != self.num_layers {
                    return Err(EngineError::ShapeMismatch);
                }
                let attn_elems_per_buf = self.seq_len * attn_num_kv_heads * attn_head_dim;
                let attn_buf_bytes = attn_elems_per_buf * self.dtype.byte_size();

                // Note: GatedDeltaNet recurrent states stay f32 even in f16 mode to prevent drift.
                let gdn_dtype = KvDtype::F32;
                let conv_bytes = gdn_conv_state_len * gdn_dtype.byte_size();
                let recurrent_bytes = gdn_recurrent_len * gdn_dtype.byte_size();

                for (layer_idx, &kind) in layer_kinds.iter().enumerate() {
                    match kind {
                        LayerKind::GatedAttention => {
                            ranges.push(CskvFileRange {
                                layer_idx,
                                kind: CskvBufferKind::Key,
                                offset,
                                num_elements: attn_elems_per_buf,
                                dtype: self.dtype,
                            });
                            offset += attn_buf_bytes as u64;

                            ranges.push(CskvFileRange {
                                layer_idx,
                                kind: CskvBufferKind::Value,
                                offset,
                                num_elements: attn_elems_per_buf,
                                dtype: self.dtype,
                            });
                            offset += attn_buf_bytes as u64;
                        }
                        LayerKind::GatedDeltaNet => {
                            ranges.push(CskvFileRange {
                                layer_idx,
                                kind: CskvBufferKind::ConvState,
                                offset,
                                num_elements: *gdn_conv_state_len,
                                dtype: gdn_dtype,
                            });
                            offset += conv_bytes as u64;

                            ranges.push(CskvFileRange {
                                layer_idx,
                                kind: CskvBufferKind::Recurrent,
                                offset,
                                num_elements: *gdn_recurrent_len,
                                dtype: gdn_dtype,
                            });
                            offset += recurrent_bytes as u64;
                        }
                    }
                }
            }
            CskvVariant::Mla { qk_dim } => {
                let elems_per_buf = self.seq_len * qk_dim;
                let buf_bytes = elems_per_buf * self.dtype.byte_size();

                for layer_idx in 0..self.num_layers {
                    ranges.push(CskvFileRange {
                        layer_idx,
                        kind: CskvBufferKind::MlaLatent,
                        offset,
                        num_elements: elems_per_buf,
                        dtype: self.dtype,
                    });
                    offset += buf_bytes as u64;
                }
            }
        }

        Ok(CskvLayout {
            header: self.clone(),
            header_bytes_len,
            total_file_bytes: offset as usize,
            ranges,
        })
    }
}

impl fmt::Display for CskvHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CSKV(v={}, layers={}, seq_len={}, dtype={})",
            self.version,
            self.num_layers,
            self.seq_len,
            self.dtype.as_str()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cskv_dense_v4_roundtrip() {
        let header = CskvHeader {
            version: VERSION_DENSE_V4,
            num_layers: 32,
            seq_len: 128,
            dtype: KvDtype::F32,
            variant: CskvVariant::Dense {
                num_kv_heads: 8,
                head_dim: 128,
                space_prefix_next: true,
            },
        };

        let bytes = header.to_bytes();
        let (parsed, len) = CskvHeader::parse(&bytes).expect("parse header");
        assert_eq!(len, bytes.len());
        assert_eq!(parsed, header);

        let layout = parsed.compute_layout().expect("compute layout");
        assert_eq!(layout.ranges.len(), 32 * 2);
        assert_eq!(layout.ranges[0].num_elements, 128 * 8 * 128);
        assert_eq!(layout.total_file_bytes, len + 32 * 2 * (128 * 8 * 128 * 4));
    }

    #[test]
    fn test_cskv_hybrid_roundtrip() {
        let header = CskvHeader {
            version: VERSION_HYBRID,
            num_layers: 4,
            seq_len: 64,
            dtype: KvDtype::F32,
            variant: CskvVariant::Hybrid {
                attn_num_kv_heads: 4,
                attn_head_dim: 64,
                gdn_conv_state_len: 256,
                gdn_recurrent_len: 1024,
                layer_kinds: vec![
                    LayerKind::GatedAttention,
                    LayerKind::GatedDeltaNet,
                    LayerKind::GatedAttention,
                    LayerKind::GatedDeltaNet,
                ],
            },
        };

        let bytes = header.to_bytes();
        let (parsed, len) = CskvHeader::parse(&bytes).expect("parse header");
        assert_eq!(len, bytes.len());
        assert_eq!(parsed, header);

        let layout = parsed.compute_layout().expect("compute layout");
        assert_eq!(layout.ranges.len(), 8); // 2 attn layers * 2 + 2 gdn layers * 2
    }

    #[test]
    fn test_cskv_mla_f16_roundtrip() {
        let header = CskvHeader {
            version: VERSION_MLA_F16,
            num_layers: 16,
            seq_len: 256,
            dtype: KvDtype::F16,
            variant: CskvVariant::Mla { qk_dim: 576 },
        };

        let bytes = header.to_bytes();
        let (parsed, len) = CskvHeader::parse(&bytes).expect("parse header");
        assert_eq!(len, bytes.len());
        assert_eq!(parsed, header);

        let layout = parsed.compute_layout().expect("compute layout");
        assert_eq!(layout.ranges.len(), 16);
        assert_eq!(layout.ranges[0].byte_length(), 256 * 576 * 2);
    }
}
