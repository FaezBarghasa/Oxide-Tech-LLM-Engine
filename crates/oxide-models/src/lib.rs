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
    clippy::cast_precision_loss
)]

pub mod audio;
pub mod bonsai2;
pub mod diffusion;
pub mod engram;
pub mod formats;
pub mod llama3;
pub mod manifest;
pub mod moe;
pub mod monarch;
pub mod needle;
pub mod registry;
pub mod specialized;

pub use audio::{AudioEngineMode, AudioModelConfig, AudioServingEngine};
pub use bonsai2::TernaryBonsai2Config;
pub use diffusion::{DiffusionEngine, DiffusionSchedulerType, DiffusionTransformerConfig};
pub use engram::EngramGatherTable;
pub use formats::{GgufHeader, GgufQuantType, ModelFileFormat, Nvfp4Block, SafeTensorsHeader};
pub use llama3::Llama3Config;
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
    AtmosphericForecastGrid, AutomatedTestingQeEngine, AutonomousScientificAgentEngine,
    BimanualActionChunk, BimanualRoboticsEngine, CadCoderVlmEngine, CadFeatureOperation,
    CadParametricGeometry, CadQueryCodeArtifact, CaeStressAnalysisResult, ClinicalDecisionEngine,
    ClinicalRecommendation, CompositionalDesignGraph, DataInsightVisualization,
    DataWorkflowAutomationEngine, DataWorkflowPipelineStep, DesignGraphNode, DesignGraphPatch,
    DexterousHandState, DexterousRoboticsEngine, EdaPcbEngine, ElementWeightFraction,
    EmbeddingEngine, FinancialMarketBar, FunctionalRequirementNode, GamepadControllerState,
    GameplayWhamEngine, GeospatialEcosystemEngine, GeospatialEcosystemMetrics, KronosTradingEngine,
    LammpsSimulationTask, MambaMoeHybridEngine, MaterialsMetallurgyEngine,
    MechRagEngineeringEngine, MidiNoteEvent, MusubiCadGraphEngine, NeuromorphicSpikingEngine,
    OverleafProjectDocument, PairProgrammingAction, ParametricCadEngine, PcbComponentPlacement,
    PcbNetlistSpecification, PcbTraceRoute, PolarsStatisticalGutEngine, QeTestExecutionReport,
    RecursiveStepResult, ResearchHypothesisResult, RoboticsVlaActionChunk, RoboticsVlaEngine,
    SatelliteEarthEngine, SatelliteGeoDetection, SelfHealingLocatorAction,
    SmartContractMutationReport, SoftwareEngineeringPairEngine, SpecForgeMutationEngine,
    SpikingNeuronState, SymbolicMusicEngine, SystemFunctionalDecomposition,
    ThermodynamicPhaseRegion, TimeSeriesEngine, TinyRecursionModelEngine, TraceableCitation,
    TradingForecast, TradingSignal,
};
