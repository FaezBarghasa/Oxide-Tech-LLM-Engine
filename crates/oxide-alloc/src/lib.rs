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
    clippy::cast_sign_loss
)]

pub mod device_arena;
pub mod hierarchical_kv;
pub mod host_arena;
pub mod transactional;

pub use device_arena::DeviceMemoryArena;
pub use hierarchical_kv::{
    CacheTierLocation, DistributedKvBlockPayload, HierarchicalKvCache, KvBlockDescriptor,
    TOKENS_PER_KV_BLOCK,
};
pub use host_arena::HostPinnedArena;
pub use transactional::TransactionalBlockTable;
