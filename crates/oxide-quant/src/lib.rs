#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_range_loop,
    clippy::unreadable_literal
)]

pub mod bitsandbytes;
pub mod compressed_tensors;
pub mod cq2;
pub mod exl2;
pub mod fp8;
pub mod gguf_quants;
pub mod gptq_awq;
pub mod int_quant;
pub mod int_standard;
pub mod marlin;
pub mod modelopt;
pub mod mxfp;
pub mod nvfp4;
pub mod pq2_0;
pub mod ptq1_0;
pub mod torchao;

// High-level re-exports
pub use bitsandbytes::{BlockFp4Bnb_64, BlockNf4_64, BnbInt8, FP4_TABLE, NF4_TABLE};
pub use compressed_tensors::{
    CompressedQuantType, CompressedTensorsConfig, CompressionFormat, QuantizationStrategy,
    dequantize_compressed_slice,
};
pub use cq2::NeedleCQ2;
pub use exl2::{Exl2BitsPerWeight, Exl2Quant, Exl2WeightMatrix};
pub use fp8::{BlockFp8E4M3, Fp8E4M3, Fp8E5M2};
pub use gguf_quants::{
    BlockIQ1_M, BlockIQ1_S, BlockIQ2_M, BlockIQ2_S, BlockIQ2_XS, BlockIQ2_XXS, BlockIQ3_M,
    BlockIQ3_S, BlockIQ3_XS, BlockIQ3_XXS, BlockIQ4_NL, BlockIQ4_XS, BlockQ2_K_S, BlockQ3_K_L,
    BlockQ3_K_M, BlockQ3_K_S, BlockQ4_K, BlockQ4_K_M, BlockQ4_K_S, BlockQ5_K, BlockQ5_K_M,
    BlockQ5_K_S, BlockTQ1_0, BlockTQ2_0,
};
pub use gptq_awq::{
    AwqPrecisionMode, AwqQuant, AwqWeightMatrix, GptqBitWidth, GptqQuant, GptqWeightMatrix,
    QuantGroupSize,
};
pub use int_quant::{
    BlockQ1_0, BlockQ2_0, BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ4_2, BlockQ4_3,
    BlockQ5_0, BlockQ5_1, BlockQ6_K, BlockQ8_0, BlockQ8_1, BlockQ8_K, f16,
};
pub use int_standard::{Int4Asym, Int4Sym, Int8Asym, Int8Sym};
pub use marlin::{MarlinQuant, MarlinQuantizedMatrix, MarlinWeightMatrix, pack_marlin_int4};
pub use modelopt::{ModelOptAlgorithm, ModelOptConfig, apply_smoothquant_weights};
pub use mxfp::{BlockMxFp4, BlockMxFp6, BlockMxFp8, BlockMxInt8, E8M0Scale};
pub use nvfp4::{BlockNvFp4_16, NvFp4};
pub use pq2_0::PackedQuant2_0;
pub use ptq1_0::{Ternary1_58Bit, TernaryBlock128};
pub use torchao::{TorchAoLinearDescriptor, TorchAoQuantType, dequantize_fp6_e3m2};
