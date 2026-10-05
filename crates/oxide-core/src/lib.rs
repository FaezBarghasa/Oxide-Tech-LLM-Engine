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
    clippy::doc_markdown
)]

pub mod dag;
pub mod error;
pub mod macros;
pub mod memory;
pub mod traits;
pub mod typestate;
pub mod worker;

pub use dag::{PhysicalBlockId, SharedPhysicalBlock, TreeNode};
pub use error::{EngineError, Result};
pub use memory::DevicePtr;
pub use traits::{HardwareBackend, HardwareSupports, ModelConfig, OpticalComputeFabric, QuantScheme};
pub use typestate::{Allocated, Decoding, Prefilling, SequenceRequest, Terminal, Unallocated};
pub use worker::{ExecutionWorker, StepCommand, StepCompletion};
