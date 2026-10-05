use bytemuck::{Pod, Zeroable};
use oxide_core::error::{EngineError, Result};
use std::fs::File;
use std::path::Path;

/// 64-byte aligned header format for .oxide model weights and metadata artifacts.
/// Total size: 128 bytes with zero padding holes.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OxideModelHeader {
    pub magic: [u8; 8],              // b"OXIDEMOD" [0..8]
    pub format_version: u32,         // [8..12]
    pub architecture_id: u32,        // 1 = Bonsai2, 2 = Needle3, 3 = Llama3 [12..16]
    pub quantization_id: u32,        // 1 = PTQ1_0, 2 = PQ2_0, 3 = CQ2, 4 = NVFP4 [16..20]
    pub hidden_dim: u32,             // [20..24]
    pub intermediate_dim: u32,       // [24..28]
    pub num_layers: u32,             // [28..32]
    pub num_heads: u32,              // [32..36]
    pub num_kv_heads: u32,           // [36..40]
    pub head_dim: u32,               // [40..44]
    pub vocab_size: u32,             // [44..48]
    pub max_seq_len: u32,            // [48..52]
    pub pad0: u32,                   // [52..56] 8-byte alignment padding for u64 fields
    pub tensor_manifest_offset: u64, // [56..64]
    pub tensor_manifest_len: u64,    // [64..72]
    pub reserved_32: [u8; 32],       // [72..104]
    pub reserved_16: [u8; 16],       // [104..120]
    pub reserved_8: [u8; 8],         // [120..128]
}

pub const OXIDE_MAGIC: &[u8; 8] = b"OXIDEMOD";

impl OxideModelHeader {
    /// Ingests and validates an .oxide model artifact via zero-copy memory mapping.
    pub fn open_mmap<P: AsRef<Path>>(path: P) -> Result<(Self, memmap2::Mmap)> {
        let file = File::open(path).map_err(|e| EngineError::BackendError(e.to_string()))?;
        // SAFETY: File is opened read-only and mapped for zero-copy deserialization.
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| EngineError::BackendError(e.to_string()))?
        };

        if mmap.len() < std::mem::size_of::<Self>() {
            return Err(EngineError::InvalidArtifactHeader);
        }

        let header: Self = *bytemuck::from_bytes(&mmap[..std::mem::size_of::<Self>()]);
        if &header.magic != OXIDE_MAGIC {
            return Err(EngineError::InvalidArtifactHeader);
        }

        Ok((header, mmap))
    }
}
