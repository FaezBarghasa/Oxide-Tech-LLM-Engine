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
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::match_same_arms,
    clippy::if_not_else,
    clippy::manual_div_ceil,
    clippy::similar_names,
    clippy::cast_ptr_alignment,
    clippy::chunks_exact_to_as_chunks,
    clippy::format_push_string,
    clippy::unreadable_literal
)]

pub mod attention_kernels;
pub mod audio;
pub mod bonsai2;
pub mod chat_template;
pub mod diffusion;
pub mod engram;
pub mod flash_attn;
pub mod formats;
pub mod hf_architectures;
pub mod llama3;
pub mod loader;
pub mod lora;
pub mod manifest;
pub mod moe;
pub mod monarch;
pub mod needle;
pub mod projector;
pub mod registry;
pub mod rope;
pub mod specialized;

pub use attention_kernels::{
    AttentionKernelBackend, FlashInferConfig, FlashInferEngine, FlashMlaEngine,
};
pub use hf_architectures::{
    ColBertLateInteraction, DeepSeekV3MoeConfig, MambaSsmBlock, MultiModalVisionProjector,
    RewardClassifierHead,
};

pub use chat_template::{ChatMessage, ChatRole, ChatTemplateFormat, ChatTemplateParser};
pub use flash_attn::{FlashAttentionConfig, FlashAttentionEngine};
pub use lora::{LoraConfig, LoraHotSwapRegistry, LoraLayerWeights};
pub use projector::{MultiModalProjector, ProjectorConfig, ProjectorType};
pub use rope::{RopeConfig, RopeScalingEngine, RopeScalingType};

pub use audio::{AudioEngineMode, AudioModelConfig, AudioServingEngine};
pub use bonsai2::TernaryBonsai2Config;
pub use diffusion::{DiffusionEngine, DiffusionSchedulerType, DiffusionTransformerConfig};
pub use engram::EngramGatherTable;
pub use formats::{
    GgufFile, GgufHeader, GgufQuantType, GgufTensorInfo, GgufValue, ModelFileFormat, Nvfp4Block,
    SafeTensorInfo, SafeTensorsHeader,
};
pub use llama3::{Llama3Config, Llama3KvCacheLayer, Llama3LayerWeights, Llama3Model};
pub use loader::{DevicePlacement, MmapModel, ModelMetadata, TensorInfo, WeightAllocator};
pub use manifest::OxideModelHeader;
pub use moe::{ExpertRoutingDecision, MoELayer, MoERouterConfig};
pub use monarch::MonarchMlp;
pub use needle::{CactusNeedleConfig, NeedleSubnetwork};
pub use registry::{
    ModelArchitectureType, ModelFamily, ModelModality, ModelSpecification, QuantizationClass,
};
pub use specialized::{
    AcademicResearchWritingEngine, AgenticDeciderEngine, AgenticEngDesignEngine,
    AlloyCompositionDesign, AlloyMicrostructureState, AmMapCompositionEngine,
    AssistiveNavigationPrompt, AssistiveVisionEngine, AtmosphericAuroraEngine,
    AtmosphericForecastGrid, AtomAgentsPhysicsEngine, AutomatedTestingQeEngine,
    AutonomousScientificAgentEngine, BimanualActionChunk, BimanualRoboticsEngine,
    CadCoderVlmEngine, CadFeatureOperation, CadParametricGeometry, CadQueryCodeArtifact,
    CaeStressAnalysisResult, ClefCausalInferenceResult, ClinicalDecisionEngine,
    ClinicalRecommendation, CompositionalDesignGraph, DataInsightVisualization,
    DataWorkflowAutomationEngine, DataWorkflowPipelineStep, DecisionMakingModelEngine,
    DesignGraphNode, DesignGraphPatch, DexterousHandState, DexterousRoboticsEngine, EdaPcbEngine,
    ElementWeightFraction, EmbeddingEngine, FinancialMarketBar, FunctionalRequirementNode,
    GamepadControllerState, GameplayWhamEngine, GeospatialEcosystemEngine,
    GeospatialEcosystemMetrics, KronosTradingEngine, LammpsSimulationTask, MambaMoeHybridEngine,
    MaterialsMetallurgyEngine, MechRagEngineeringEngine, MidiNoteEvent, MusubiCadGraphEngine,
    NeuromorphicSpikingEngine, OverleafProjectDocument, PairProgrammingAction, ParametricCadEngine,
    PcbComponentPlacement, PcbNetlistSpecification, PcbTraceRoute, PolarsStatisticalGutEngine,
    QeTestExecutionReport, RecursiveStepResult, ResearchHypothesisResult, RoboticsVlaActionChunk,
    RoboticsVlaEngine, SatelliteEarthEngine, SatelliteGeoDetection, SelfHealingLocatorAction,
    SmartContractMutationReport, SoftwareEngineeringPairEngine, SpecForgeMutationEngine,
    SpikingNeuronState, SymbolicMusicEngine, SystemFunctionalDecomposition,
    ThermodynamicPhaseRegion, TimeSeriesEngine, TinyRecursionModelEngine, TraceableCitation,
    TradingForecast, TradingSignal,
};
