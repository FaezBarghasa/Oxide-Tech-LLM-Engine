# Changelog

All notable changes to the **Oxide-Tech-LLM-Engine** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-05

### Added

#### Foundational Architecture & Memory Typing (Horizons 0 & I)
- **Zero-Cost Device Typing (`DevicePtr<T>`)**: Transparent struct encapsulating device pointers with Zero-Sized Type (ZST) marker proofs in `crates/oxide-core/src/memory.rs`. Dereferencing device VRAM on CPU triggers compile-time type errors.
- **Static Thread Stack Guard Macro**: Procedural macro `#[assert_stack_safe(max_bytes = 4096)]` preventing stack overflow allocations on OS threads.
- **Lifetime-Bounded Memory Arenas (`'arena`)**:
  - `HostPinnedArena`: Direct page-locked host memory mapped to GPU virtual address spaces (`cudaHostAllocMapped | cudaHostAllocWriteCombined`).
  - `DeviceMemoryArena<'arena>`: Pre-allocated activation scratchpads and static KV pools eliminating dynamic allocations on the forward path.
- **Typestate Request Lifecycle**: Enforced compile-time request transitions (`Unallocated` $\to$ `Allocated` $\to$ `Prefilling` $\to$ `Decoding` $\to$ `Terminal`).
- **Wait-Free SPSC Ringbuffers**: 64-byte cacheline-aligned command and completion ringbuffers (`StepCommand`, `StepCompletion`) eliminating false sharing.
- **3-Phase Thermal-Aware Worker (`ExecutionWorker`)**: Micro-spin $\to$ coarse yield $\to$ futex parking via `crossbeam::sync::Parker`, eliminating CPU thermal throttling under idle/fluctuating load.
- **PyTorch Golden Oracle Test Harness**: Reference verification harness with Chebyshev distance ($L_\infty \le 1.5 \times 10^{-3}$) and cosine similarity ($\ge 0.9999$) assertions.

#### Closed-Enum Pipeline Monomorphization & Binary Ingestion (Horizon II)
- **Closed Dispatch Engine (`SpecializedPipeline`)**: Direct monomorphized jump table matching models and backends at initialization without virtual dispatch (`dyn Trait`).
- **Zero-Copy Artifact Format (`.oxide`)**: 64-byte aligned header specification (`OxideModelHeader`) supporting memory-mapped ingestion via `memmap2::MmapOptions`.

#### Microarchitectural Kernels & Quantization (Horizon III)
- **In-Shared-Memory Fast Walsh-Hadamard Transform (FWHT)**: 7-stage unrolled butterfly network executing entirely within warp registers and L1 Shared Memory.
- **Branchless Ternary GEMV (`PTQ1_0` and `PQ2_0`)**: 2-bit packed ternary arithmetic mapping directly to `__dp4a` on CUDA and `vpdpbusd` on AVX-512.
- **Micro-Scaled Quantization (`NVFP4`, `CQ2`)**: Hardware-accelerated E2M1 FP4 and 2-bit codebook quantization support.

#### Non-Euclidean Architectures & In-Kernel Grammar (Horizon IV)
- **Monarch Hadamard Block-Diagonal MLP**: $O(D \sqrt{D})$ parameter reduction via factorized block-diagonal matrix projections.
- **Const-Generic Subnetwork Depth Laddering (`NeedleSubnetwork<L>`)**: Zero-branch compile-time layer depth execution ($2\text{L}, 4\text{L}, 8\text{L}, 16\text{L}, 20\text{L}$).
- **Direct-Mapped Hashed Engram Table**: 70.8M parameter n-gram gather table bypassed through $O(1)$ memory hashing.
- **In-Kernel Byte-DFA Schema Grammar Masking**: Logit processor applying DFA state transition tables $\delta(S, \text{byte})$ to enforce $100\%$ JSON schema validity without host roundtrips.

#### Dual-Topology Hybrid Memory & Speculative Rollbacks (Horizon V)
- **Dual-Topology State Allocator**:
  - Topology 1: Contiguous $O(1)$ linear recurrent state matrix for linear attention layers.
  - Topology 2: Block-paged virtual KV cache pool ($B=16$) for softmax attention layers.
- **Transactional Speculative Rollback Engine (`TransactionalBlockTable`)**: $O(1)$ host metadata truncation upon speculative rejection with zero VRAM zeroing or rewriting.
- **Sarathi Chunked Prefill**: Co-scheduled prefill chunks ($C=512$) spreading TTFT spikes during continuous batch decoding.

#### Directed Acyclic Graph (DAG) Tree-Native Reasoning (Horizon VI)
- **Lock-Free Copy-on-Write Block DAG (`SharedPhysicalBlock`, `TreeNode`)**: Atomic reference counting managing $100,000+$ active reasoning branches with zero-copy thought forks.
- **Topological Tree Attention Masking**: Unified single-pass forward execution evaluating disparate leaf nodes with shared prompt prefix reuse.
- **In-Flight PRM Evaluator**: Asynchronous verifier scoring subtrees at reasoning step boundaries and recycling rejected branches mid-decode.

#### Multi-Silicon Hardware Acceleration Backends (Horizon VII)
- `oxide-backend-cuda`: NVIDIA Blackwell GB200, Hopper H100, Ada RTX 4090, Ampere A100.
- `oxide-backend-rocm`: AMD Instinct MI300X, MI350X/MI355X (CDNA4), Radeon RX 7900 XTX, Ryzen AI Max+ 395 (Strix Halo).
- `oxide-backend-metal`: Apple Silicon M4 Max, M3 Max, M2 Ultra with Unified Memory zero-copy and MSL threadgroup shaders.
- `oxide-backend-tpu`: Google TPU v6e Trillium, v5p, v5e, v4, Edge TPU Coral with ICI interconnect support.
- `oxide-backend-intel`: Intel Arc B580/B570 Battlemage, Xeon 6980P (Granite Rapids / Sierra Forest), Xeon Max with Level-Zero driver bindings.
- `oxide-backend-qualcomm`: Qualcomm Snapdragon X Elite (X1E-84-100), Snapdragon 8 Elite Hexagon NPU (45 TOPS).
- `oxide-backend-rknn`: Rockchip RK3588, RK3576, Orange Pi 6 Plus tri-core NPU with DMA-BUF zero-copy memory.
- `oxide-backend-hailo`: Raspberry Pi 5 with AI HAT+ (13 TOPS) and AI HAT+ 2 (26 TOPS / Hailo-8).
- `oxide-backend-cpu`: Vectorized AVX-512 and ARM Neon CPU backend.

#### Hierarchical 3-Tier KV Cache (Horizon VIII)
- **Hierarchical KV Cache Engine (`HierarchicalKvCache`)**:
  - Tier 1: Device VRAM fast pool.
  - Tier 2: Host pinned RAM pool with asynchronous PCIe DMA prefetching.
  - Tier 3: External NVMe storage pool with Blake3 content-addressed block sharing.

#### Autonomics & Hardware Co-Design (Horizons IX & X)
- **Self-Tuning Autonomic Plan (`AutonomicPlan`)**: Offline hardware profile registry selecting optimal threadblock dimensions, warp configurations, and memory limits.
- **Compile-Time Hardware Capability Proving (`HardwareSupports<B, M>`)**: Type-level compile-time validation for optical fabrics, CXL.mem pools, and neuromorphic spiking processors.

#### Comprehensive Model Catalog & Domain Engines
- **Foundational LLMs**: DeepSeek V4 (MLA), DeepSeek R1, Qwen 3.8, Qwen3-Coder, Llama 4, Kimi K3, GLM-5.3.
- **Multimodal Audio & Video**: Diffusion Transformer (DiT), Audio TTS/ASR streaming engine, Symbolic Music (Muzic, HeartMuLa, ACE-Step, YuE2-Studio, SongGen), Vision-Language (CAD-Coder, Visual-ChatGPT, NUWA).
- **Quantitative Trading & Finance**: `shiyu-coder/Kronos` trading foundation model, `microsoft/qlib`, `microsoft/FinanceBenchmark`.
- **Electronic Design & PCB (EDA)**: `Electronics Agent Kit`, `KiC-AI`, `cadlab`, `PCBSchemaGen`, `Trace`, `EdaPcbEngine`.
- **CAD & 3D Modeling**: `MusubiCAD` (`MusubiCadGraphEngine`), `CAD-Coder` (`CadCoderVlmEngine`), `AI-CAD`, `GPTCAD` (`ParametricCadEngine`).
- **Materials Science & Metallurgy**: `AtomAgents` (`AtomAgentsPhysicsEngine`), `AMMap` (`AmMapCompositionEngine`), `AlloyGPT`, `DAS-DAO`, `ALCHEMIST` (`MaterialsMetallurgyEngine`).
- **Engineering Design & CAE**: `MechRAG` (`MechRagEngineeringEngine`), `agentic-eng-design` (`AgenticEngDesignEngine`).
- **Software QA & Testing**: `SpecForge AI` (`SpecForgeMutationEngine`), `LionAGI QE Fleet`, `agentic-qe`, `Falcon-Automation` (`AutomatedTestingQeEngine`).
- **Academic Research & Writing**: `Academic Writing Skills`, `ResearchKit` (`AcademicResearchWritingEngine`).
