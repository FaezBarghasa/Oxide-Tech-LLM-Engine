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
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::manual_div_ceil,
    clippy::cast_lossless
)]

pub mod engine;
pub mod hybrid;
pub mod model_manager;
pub mod pipeline;
pub mod slot_manager;
pub mod speculative;
pub mod tensor_split;

pub use engine::OxideEngine;
pub use hybrid::{DeviceRole, HybridDeviceTopology, HybridMultiDevicePipeline, LayerPartition};
pub use model_manager::DynamicModelManager;
pub use pipeline::SpecializedPipeline;
pub use slot_manager::{ContinuousBatchingSlotManager, InferenceSlot, SlotRequest, SlotState};
pub use speculative::{SpeculativeConfig, SpeculativeDecoderEngine, SpeculativeVerificationResult};
pub use tensor_split::{
    AcceleratorKind, TensorSliceDescriptor, TensorSplitDistributionEngine, TensorSplitMode,
};
