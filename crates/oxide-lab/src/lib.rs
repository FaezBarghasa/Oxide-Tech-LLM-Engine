#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
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
    clippy::cast_precision_loss
)]

//! Oxide-Lab: High-Performance AI Research Laboratory Tools Suite.
//!
//! Provides zero-allocation, bare-metal primitives for:
//! 1. Custom Neural Architectures & Dynamic Model Building (`custom_model`).
//! 2. Zero-Allocation Training, Backpropagation & LoRA/QLoRA Optimization (`training`).
//! 3. Post-Training Quantization (AWQ/PTQ) & Model Compression (`quant_compression`).
//! 4. Real-Time Tensor Debugging, Representation Drift Detection & Telemetry (`realtime_debug`).
//! 5. Universal Multi-Modal Input Ingestion & Generative Media Synthesis (`multimodal_lab`).

pub mod custom_model;
pub mod multimodal_lab;
pub mod quant_compression;
pub mod realtime_debug;
pub mod training;

pub use custom_model::{
    ArchitectureConfig, CustomModel, CustomModelBuilder, CustomModelScratch, LayerSpec, LayerType,
};
pub use multimodal_lab::{MultiModalInput, MultiModalLabEngine, MultiModalOutput};
pub use quant_compression::{
    LabCompressor, LabQuantMethod, LabQuantizer, PrunedMatrix, QuantizedWeightMatrix,
    SvdDecomposedMatrix,
};
pub use realtime_debug::{
    ActivationTelemetry, DriftDetector, DriftSummary, MemoryAndJitterProfile, PerplexityAuditor,
    TensorAnomaly, TensorDebugger,
};
pub use training::{
    AdamWOptimizer, LoraFineTuner, LossComputer, LossType, LrScheduler, TrainingConfig,
};
