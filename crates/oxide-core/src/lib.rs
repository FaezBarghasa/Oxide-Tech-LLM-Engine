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
    clippy::struct_excessive_bools,
    clippy::too_many_lines
)]

pub mod dag;
pub mod error;
pub mod hardware;
pub mod macros;
pub mod memory;
pub mod traits;
pub mod typestate;
pub mod worker;

pub use dag::{PhysicalBlockId, SharedPhysicalBlock, TreeNode};
pub use error::{EngineError, Result};
pub use hardware::{
    ComputeCapability, GpuArchitecture, GpuDeviceProfile, HardwareFormFactor, MemoryTechnology,
    TensorCoreGeneration,
};
pub use memory::DevicePtr;
pub use traits::{
    HardwareBackend, HardwareSupports, ModelConfig, OpticalComputeFabric, QuantScheme,
};
pub use typestate::{Allocated, Decoding, Prefilling, SequenceRequest, Terminal, Unallocated};
pub use worker::{ExecutionWorker, StepCommand, StepCompletion};
