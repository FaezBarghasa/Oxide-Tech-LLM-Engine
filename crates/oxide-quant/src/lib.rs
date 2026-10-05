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

pub mod cq2;
pub mod int_quant;
pub mod nvfp4;
pub mod pq2_0;
pub mod ptq1_0;

pub use cq2::NeedleCQ2;
pub use int_quant::{
    BlockQ2_K, BlockQ3_K, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ6_K, BlockQ8_0, f16,
};
pub use nvfp4::NvFp4;
pub use pq2_0::PackedQuant2_0;
pub use ptq1_0::{Ternary1_58Bit, TernaryBlock128};
