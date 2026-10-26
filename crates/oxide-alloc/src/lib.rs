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
    clippy::too_many_lines,
    clippy::manual_checked_ops,
    clippy::manual_div_ceil,
    clippy::chunks_exact_to_as_chunks,
    clippy::redundant_closure_for_method_calls,
    clippy::unreadable_literal
)]

pub mod compact_kv;
pub mod device_arena;
pub mod hierarchical_kv;
pub mod host_arena;
pub mod io_uring_loader;
pub mod kv_advanced;
pub mod paged_kv;
pub mod radix_cache;
pub mod slab_pool;
pub mod transactional;

pub use compact_kv::{CompactSlotMetadata, DmsCompactKvArena, DmsPolicy, SequenceCompactTable};
pub use device_arena::{DeviceMemoryArena, MemoryBudgetProfile};
pub use hierarchical_kv::{
    CacheTierLocation, DistributedKvBlockPayload, HierarchicalKvCache, KvBlockDescriptor,
    TOKENS_PER_KV_BLOCK,
};
pub use host_arena::HostPinnedArena;
pub use io_uring_loader::{AsyncIoStats, AsyncReadRequest, HighThroughputWeightLoader};
pub use kv_advanced::{
    ContextShiftManager, KvCacheDumpContainer, KvQuantizationPrecision, PromptCacheRegistry,
    QuantizedKvBlock,
};
pub use paged_kv::{PagedKvArena, PhysicalPageBlock, SequenceBlockTable, TOKENS_PER_PAGE};
pub use radix_cache::{RadixNode, RadixPrefixCache};
pub use slab_pool::{PhysicalSlabBlock, SequenceSlabTable, SlabPoolArena, TOKENS_PER_SLAB};
pub use transactional::{KvTransaction, TransactionalBlockTable, TransactionalKvGuard};
