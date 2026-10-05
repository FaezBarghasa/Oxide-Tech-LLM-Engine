# Models, Modalities & Domain Engines Catalog

## Overview

`Oxide-Tech-LLM-Engine` provides an extensive built-in model specification catalog (`crates/oxide-models/src/registry.rs`) and specialized domain execution engines (`crates/oxide-models/src/specialized.rs`).

---

## 1. Model Families & Registry Index

| Model Identifier | Family | Modality | Default Architecture | Default Quantization |
| :--- | :--- | :--- | :--- | :--- |
| `qwen-3.8` | `Qwen` | `TextOnly` | `CausalTransformer` | `Bfloat16` |
| `deepseek-v4` | `DeepSeek` | `TextOnly` | `MultiHeadLatentAttentionMla` | `Nvfp4Blackwell` |
| `deepseek-r1` | `DeepSeek` | `CodeReasoning` | `DeepSeekMoE` | `Nvfp4Blackwell` |
| `kimi-k3` | `Kimi` | `TextOnly` | `HybridLinearAttention` | `Ptq1_0Ternary` |
| `shiyu-coder/kronos` | `KronosTrading` | `FinancialTradingTimeSeries` | `CausalTransformer` | `Float16` |
| `electronics-agent-kit` | `ElectronicsAgentKit` | `ElectronicDesignPcbSchematic` | `AutonomousAgentMesh` | `Float16` |
| `kic-ai` | `KicAiPlugin` | `ElectronicDesignPcbSchematic` | `DecoderOnlyDense` | `Float16` |
| `cadlab` | `CadLabRustEda` | `ElectronicDesignPcbSchematic` | `CausalTransformer` | `Float16` |
| `pcbschemagen` | `PcbSchemaGenConstraint` | `ElectronicDesignPcbSchematic` | `CausalTransformer` | `Float16` |
| `trace-pcb` | `TraceAiPcb` | `ElectronicDesignPcbSchematic` | `CausalTransformer` | `Float16` |
| `musubicad` | `MusubiCadRust` | `ParametricCad3DGeneration` | `DeterministicCadDesignGraph` | `Float16` |
| `cad-coder` | `CadCoderVlm` | `VisionLanguage` | `CadQueryCodeGenerator` | `Float16` |
| `mentaagent` | `MentaAgentBi` | `DecisionAgentic` | `AutonomousAgentMesh` | `Float16` |
| `academic-writing-skills` | `AcademicWritingSkillsAgent` | `ScientificLiteratureDiscovery` | `AutonomousAgentMesh` | `Float16` |
| `researchkit` | `ResearchKitOverleaf` | `ScientificLiteratureDiscovery` | `AutonomousAgentMesh` | `Float16` |
| `atomagents` | `AtomAgentsMit` | `MaterialsMetallurgyDesign` | `PhysicsSimulationMultiAgent` | `Float16` |
| `alchemist-alloys` | `AlchemistAlloyDiscovery` | `MaterialsMetallurgyDesign` | `HierarchicalConditionalDiffusion` | `Float16` |
| `ammap` | `AmMapCompositional` | `MaterialsMetallurgyDesign` | `GraphNeuralNetworkGnn` | `Float16` |
| `specforge-ai` | `SpecForgeAiPolyglot` | `AutonomousQeSoftwareTesting` | `SmartContractMutationEngine` | `Float16` |
| `agentic-qe` | `AgenticQePlatform` | `AutonomousQeSoftwareTesting` | `AutonomousAgentMesh` | `Float16` |
| `mechrag` | `MechRagMllm` | `MechanicalCaeEngineeringDesign` | `MechanicalMultimodalCaeRAG` | `Float16` |
| `agentic-eng-design` | `AgenticEngDesignFramework` | `SystemsEngineeringDesign` | `SystemsEngineeringDecompositionEngine` | `Float16` |

---

## 2. Specialized Execution Engines

### A. MusubiCAD Deterministic Design Graph (`MusubiCadGraphEngine`)
- Manages parametric CAD trees as typed patches on a deterministic DAG.
- Generates reviewable patch structures with topological hash validation and Euler-Poincaré manifold checks.

### B. CAD-Coder VLM CadQuery Engine (`CadCoderVlmEngine`)
- Translates visual feature descriptions of mechanical parts into executable, editable CadQuery Python scripts (`import cadquery as cq`).

### C. AtomAgents Physics & Molecular Dynamics Engine (`AtomAgentsPhysicsEngine`)
- Formulates automated LAMMPS molecular dynamics simulation scripts.
- Evaluates microstructure FCC/BCC phase fractions, stacking fault energy, and bulk modulus.

### D. AMMap Compositional Space Mapping Engine (`AmMapCompositionEngine`)
- Maps multi-component alloy systems into thermodynamic phase graphs with solidus/liquidus windows and crack susceptibility indices.

### E. MechRAG Multimodal CAE Engineering Engine (`MechRagEngineeringEngine`)
- Computes Von Mises stress fields and safety factors from structural loads.
- Recommends geometric revisions (e.g. increasing fillet radius, adding reinforcing ribs).

### F. Agentic Engineering Design Engine (`AgenticEngDesignEngine`)
- Decomposes high-level mission prompts into functional requirement trees and synthesizes Modelica/Python simulators.

### G. SpecForge AI Mutation Testing Engine (`SpecForgeMutationEngine`)
- Executes polyglot mutation operators on smart contracts and Rust modules, calculating mutation scores and reporting surviving mutants.

### H. Academic Writing & Overleaf Engine (`AcademicResearchWritingEngine`)
- Generates evidence-traceable LaTeX research papers with structured BibTeX references and verified DOI citations.

### I. Kronos Financial Trading Engine (`KronosTradingEngine`)
- Ingests multi-horizon OHLCV bars, order book imbalance, and VWAP delta to evaluate buy/sell/neutral trading signals and expected returns in basis points.
