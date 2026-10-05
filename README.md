# Oxide-Tech-LLM-Engine (`oxide-engine`)

[![Rust 2024](https://img.shields.io/badge/rust-2024%20Edition-orange.svg)](https://www.rust-lang.org)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Safety: #![deny(unsafe_op_in_unsafe_fn)]](https://img.shields.io/badge/safety-%23!%5Bdeny(unsafe__op__in__unsafe__fn)%5D-brightgreen.svg)]()
[![Hardware: Multi-Silicon](https://img.shields.io/badge/hardware-CUDA%20%7C%20ROCm%20%7C%20Metal%20%7C%20TPU%20%7C%20Intel%20%7C%20Snapdragon%20%7C%20RKNN%20%7C%20Hailo%20%7C%20CPU-blueviolet.svg)]()

> **Oxide-Tech-LLM-Engine** is a bare-metal, high-throughput, deterministic inference runtime written in pure Rust. It eliminates the abstraction tax, scheduler jitter, and memory fragmentation inherent in generalized runtime frameworks while saturating heterogeneous silicon across NVIDIA GPUs, AMD CDNA/RDNA, Apple Silicon, Google TPUs, Intel Arc/Xeon, Qualcomm Snapdragon, Rockchip RKNN, and Raspberry Pi 5 AI HAT+ 2.

---

## Table of Contents

- [Architectural Charter & Immutable Invariants](#architectural-charter--immutable-invariants)
- [Developmental Horizons Architecture](#developmental-horizons-architecture)
- [System Architecture Topology](#system-architecture-topology)
- [Multi-Silicon Hardware Acceleration Matrix](#multi-silicon-hardware-acceleration-matrix)
- [Hierarchical KV Cache Hierarchy (Tier 1 / Tier 2 / Tier 3)](#hierarchical-kv-cache-hierarchy)
- [Model Families & Modalities Catalog](#model-families--modalities-catalog)
- [Domain-Specialized Execution Engines](#domain-specialized-execution-engines)
- [Quantization & Microarchitectural Kernels](#quantization--microarchitectural-kernels)
- [DAG Tree-Native Reasoning & Speculative Rollbacks](#dag-tree-native-reasoning--speculative-rollbacks)
- [CLI Quickstart & Server API](#cli-quickstart--server-api)
- [Workspace Crate Structure](#workspace-crate-structure)
- [Building & Verification](#building--verification)

---

## Architectural Charter & Immutable Invariants

The runtime is engineered around five strict, non-negotiable systems invariants:

1. **Zero Dynamic Allocation on Forward Paths**: Once execution enters the decode loop, zero calls to system memory allocators (`malloc`, `free`, Rust `alloc::*`, jemalloc) are permitted. All KV allocations, activation scratchpads, temporary buffers, and staging ring buffers reside in pre-allocated, lifetime-bounded static arenas (`'arena`).
2. **Compile-Time Device/Host Type Separation**: Host code never manipulates raw device pointers as untyped integers or raw host pointers. GPU memory is typed via `DevicePtr<T>` and verified through Zero-Sized Type (ZST) marker proofs, making CPU dereferences of device VRAM a compile-time failure.
3. **Monomorphized Hot Loop via Closed Dispatch**: Hot forward passes contain zero virtual dispatch (`dyn Trait` / vtables). Heterogeneous model and hardware execution is achieved via closed dispatch enums (`SpecializedPipeline`) matched once at pipeline initialization and lowered to flat direct jump tables.
4. **Thermal-Aware Adaptive Concurrency**: Worker threads never peg CPU cores at continuous $100\%$ active spin under idle or fluctuating load. Execution actors execute a strict three-phase cycle (micro-spin $\to$ `thread::yield_now()` $\to$ futex parking via `Parker`) to prevent package thermal throttling.
5. **Deterministic Bitwise Correctness**: Every quantized kernel (`PTQ1_0`, `PQ2_0`, `CQ2`, `NVFP4`) and custom layer (In-SMem FWHT, Engram table gather) passes automated bitwise tolerance tests against PyTorch/GGUF reference implementations before serving integration.

---

## Developmental Horizons Architecture

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                    DEVELOPMENTAL TOPOLOGY                                   |
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON 0: Foundational Workspace Substrate, Memory Typing & Golden Oracle                 |
|  └── Precondition: Hermetic pure-Rust workspace, #![deny(unsafe_op_in_unsafe_fn)]           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 0: Zero LTO Warnings & Oracle Match]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON I: Hardware-Bound Memory Safety, Static Arenas & Core-Pinned Actors                |
|  └── Precondition: DevicePtr<T> invariants & stack guard macro verified                     |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 1: Zero Host Data Races & Zero Stack Overflow]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON II: Closed-Enum Pipeline Monomorphization & Zero-Copy Artifact Ingestion          |
|  └── Precondition: SpecializedPipeline jump tables validated with zero indirect calls        |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 2: Branchless Model Ingestion]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON III: Microarchitectural Kernels: In-SMem FWHT & Branchless Ternary GEMV             |
|  └── Precondition: Group-128 FP16 scales & unrolled butterfly networks stabilized           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 3: Layer Latency Saturation & Cosine Similarity]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON IV: Non-Euclidean Architectures: Needle Simple Attention, Monarch MLP & Engrams    |
|  └── Precondition: Subnetwork depth laddering & in-kernel DFA byte masks compiled           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 4: 100% DFA Schema Conformance]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON V: Dual-Topology Hybrid Memory Management & Transactional Rollback                 |
|  └── Precondition: Static recurrent O(1) state split from virtual paged KV pools           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 5: Zero VRAM Fragmentation Across Continuous Load]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VI: Directed Acyclic Graph (DAG) Tree-Native Reasoning & In-Flight Pruning         |
|  └── Precondition: Lock-free atomic reference-counted blocks (crossbeam-epoch)              |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 6: Zero-Copy Thought Branching]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VII: Cross-Silicon Portability: Heterogeneous Silicon & SIMD CPU Engines          |
|  └── Precondition: Unified memory zero-copy and AVX-512 / ARM Neon bitwise parity          |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 7: Cross-Platform Numerical Invariance]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VIII: Multi-Device Distributed Topologies & Collective Fabrics                     |
|  └── Precondition: Hierarchical KV cache transfer & direct in-kernel collective bindings    |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 8: Near-Linear Multi-Device Scaling]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON IX: Profile-Guided Synthesis & Continuous Self-Tuning Autonomics                   |
|  └── Precondition: AutonomicPlan offline device registry & basic-block alignment           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 9: Zero Instruction-Cache Thrashing]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON X: Physical Hardware Co-Design: Optical, In-Memory & Neuromorphic Substrates       |
|  └── Precondition: HardwareSupports<Backend, Model> compile-time marker proving            |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## Multi-Silicon Hardware Acceleration Matrix

`Oxide-Tech-LLM-Engine` provides dedicated backend crates optimized for distinct compute substrates:

| Hardware Family | Backend Crate | Profile Support | Key Acceleration Features |
| :--- | :--- | :--- | :--- |
| **NVIDIA CUDA** | `oxide-backend-cuda` | Blackwell GB200, Hopper H100/H200, Ada RTX 4090, Ampere A100 | In-SMem FWHT butterflies, `__dp4a` ternary dot products, NVFP4 tensor cores, P2P DMA NVLink. |
| **AMD ROCm** | `oxide-backend-rocm` | Instinct MI300X/MI325X, MI350X/MI355X (CDNA4), Radeon RX 7900 XTX, Ryzen AI Max+ 395 (Strix Halo) | Matrix Core MFMA GEMV, unified memory direct zero-copy, AIE2 NPU offloading. |
| **Apple Silicon** | `oxide-backend-metal` | M4 Max, M3 Max, M2 Ultra, M1 Pro | Metal Shading Language (MSL) threadgroup memory, SIMD-group matrix intrinsics, Unified Memory zero-copy. |
| **Google TPU** | `oxide-backend-tpu` | TPU v6e Trillium, v5p, v5e, v4, Edge TPU Coral | ICI inter-chip interconnect AllReduce, Systolic Array Matrix Multiplication Units (MXU). |
| **Intel Arc & Xeon** | `oxide-backend-intel` | Arc B580/B570 (Battlemage), A770, Ponte Vecchio, Xeon 6980P (Granite Rapids / Sierra Forest), Xeon Max (HBM2e) | XMX Matrix Engines, AMX Advanced Matrix Extensions (TMM), Level-Zero command queues. |
| **Qualcomm Snapdragon** | `oxide-backend-qualcomm` | Snapdragon X Elite (X1E-84-100), Snapdragon X Plus (X1P-64-100), Snapdragon 8 Elite (Gen 4) | Hexagon Tensor Processor (HTP) NPU (45 TOPS), FastRPC DMA buffers. |
| **Rockchip RKNN** | `oxide-backend-rknn` | RK3588, RK3588S, RK3576, Orange Pi 6 Plus | Tri-core NPU (6.0 TOPS INT8 / 16-bit FP), zero-copy DMA-BUF memory sharing. |
| **Raspberry Pi & Hailo** | `oxide-backend-hailo` | Raspberry Pi 5 with AI HAT+ (13 TOPS), AI HAT+ 2 (26 TOPS / Hailo-8), M.2 / PCIe NPUs | Hailo-8 Dataflow Architecture, PCIe DMA circular ring buffers. |
| **CPU SIMD** | `oxide-backend-cpu` | x86_64 (AVX-512, AVX2, VNNI), aarch64 (ARM Neon, SVE2) | Branchless `vpdpbusd` ternary GEMV, cacheline-aligned static thread slabs. |

---

## Hierarchical KV Cache Hierarchy

`Oxide-Tech-LLM-Engine` features a **3-Tier Hierarchical KV Caching Engine** ([`oxide-alloc`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-alloc)) that seamlessly scales context across physical memory tiers without CPU stalls:

```
[ Active Stream Request ]
         │
         ▼
┌─────────────────────────────────┐
│   Tier 1: Device VRAM Pool      │  ◄── Sub-microsecond latency (SRAM / HBM3 / GDDR6X)
│   (Contiguous Page Blocks)      │      Direct GPU kernel addressable
└────────────────┬────────────────┘
                 │ (Evict LRU / Prefetch Hit)
                 ▼
┌─────────────────────────────────┐
│   Tier 2: Host Pinned RAM Pool  │  ◄── Sub-millisecond latency (DDR5 / LPDDR5X)
│   (Zero-Copy DMA Mapped)        │      PCIe Gen5 / CXL.mem asynchronous transfer
└────────────────┬────────────────┘
                 │ (Deep Context Archive / Reuse)
                 ▼
┌─────────────────────────────────┐
│   Tier 3: NVMe Storage Pool     │  ◄── Disk I/O bound (O_DIRECT / io_uring)
│   (Persistent Content Addressing)│     Shared distributed cross-instance cache reuse
└─────────────────────────────────┘
```

- **Tier 1 (Device VRAM)**: Ultra-low-latency direct GPU block pool for active decoding tokens.
- **Tier 2 (Host Pinned Memory)**: `cudaHostAllocWriteCombined` pinned memory accessible via asynchronous DMA.
- **Tier 3 (External NVMe Storage)**: Persistent content-addressed block tables with SHA-256 / Blake3 prefix hashing, enabling instant reuse of prompt prefixes across distributed inference nodes.

---

## Model Families & Modalities Catalog

The engine includes native registry definitions and architectures for foundational, multimodal, domain-specialized, and research models ([`oxide-models`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models)):

### 1. Foundational Text & Coding Models
- **DeepSeek V4 & DeepSeek R1**: Multi-Head Latent Attention (MLA) with 236B parameters (21B active) and DeepSeekMoE architecture.
- **Qwen 3.8 & Qwen3-Coder**: Dense and code-specialized causal transformers with SwiGLU activations and RoPE embeddings.
- **Llama 4 & Llama 3.1**: Dense Transformer baseline with Grouped-Query Attention (GQA) and NVFP4 Blackwell quantization.
- **Kimi K3 & GLM-5.3**: 1M+ ultra-long-context models with hybrid linear-attention prefill.

### 2. Multimodal, Audio & Diffusion
- **Diffusion Transformer (DiT)**: Diffusion video and image generation with DDIM, DPMSolver, Euler, and Flow-Matching schedulers.
- **Audio Serving Engine**: Real-time Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) streaming chunks.
- **Symbolic Music (Muzic, HeartMuLa, ACE-Step, YuE2-Studio, SongGen)**: End-to-end song and MIDI generation.
- **Vision-Language Models (CAD-Coder, Visual-ChatGPT, NUWA)**: Image understanding, visual synthesis, and CAD generation.

### 3. Quantitative Finance & Trading
- **shiyu-coder/Kronos**: Quantitative trading foundation model processing multi-horizon OHLCV bars, order book imbalance, and VWAP delta.
- **microsoft/qlib**: AI-oriented quantitative investment platform alpha prediction.
- **microsoft/FinanceBenchmark**: Decision agent benchmark for enterprise financial compliance.

---

## Domain-Specialized Execution Engines

### 🔌 Electronic Design & PCB (EDA)
- **Electronics Agent Kit & KiC-AI**: Agentic electronic schematic generation and KiCad plugin integration.
- **cadlab**: Headless Rust electronics CAD engine for end-to-end netlist and board autorouting.
- **PCBSchemaGen & Trace**: Constraint-guided schematic synthesis and KiCad circuit assistant.
- **EdaPcbEngine**: Built-in netlist router generating valid KiCad `.kicad_pcb` S-expressions and DRC clean routes.

### 🏗️ CAD & 3D Modeling
- **MusubiCAD ([`MusubiCadGraphEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1610))**: AI-native parametric CAD engine operating on deterministic design graphs with human-in-the-loop reviewed patches.
- **CAD-Coder ([`CadCoderVlmEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1670))**: VLM mapping visual part inputs directly to executable CadQuery Python scripts.
- **AI-CAD & GPTCAD ([`ParametricCadEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1151))**: STEP B-Rep and OpenSCAD parametric solid generation.

### 🔩 Materials Science & Metallurgy
- **AtomAgents ([`AtomAgentsPhysicsEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1733))**: Physics-aware molecular dynamics engine generating LAMMPS simulation scripts and evaluating FCC/BCC phase fractions.
- **AMMap ([`AmMapCompositionEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1801))**: Additive manufacturing compositional space mapping with thermodynamic phase region graphs.
- **AlloyGPT & DAS-DAO ([`MaterialsMetallurgyEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1424))**: High-entropy alloy (HEA) and 100% circular recycled scrap alloy composition design.

### ⚙️ Engineering Design & Simulation
- **MechRAG ([`MechRagEngineeringEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1855))**: Multimodal CAE/FEA structural analysis engine evaluating Von Mises stress, safety factors, and geometric fillet/rib modifications.
- **agentic-eng-design ([`AgenticEngDesignEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1919))**: Conceptual systems engineering engine generating functional requirement decompositions and Modelica/Python simulators.

### 🧪 Software Testing & QA
- **SpecForge AI ([`SpecForgeMutationEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1982))**: Polyglot mutation testing engine for smart contracts and systems software.
- **LionAGI QE Fleet & agentic-qe ([`AutomatedTestingQeEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L1537))**: Self-healing locator and Playwright QA automation engine.

### 📚 Academic Research & Writing
- **Academic Writing Skills & ResearchKit ([`AcademicResearchWritingEngine`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-models/src/specialized.rs#L2048))**: Evidence-traceable LaTeX / Overleaf paper composer with structured BibTeX citations and DOI provenance.

---

## Quantization & Microarchitectural Kernels

### 1. In-Shared-Memory Fast Walsh-Hadamard Transform (FWHT)
To eliminate activation outliers without altering ternary weights, inputs are conditioned using an orthogonal Walsh-Hadamard matrix $H_N$. For a block of 128 elements, a 7-stage unrolled butterfly network executes entirely within warp registers and L1 Shared Memory without accessing global device VRAM:
$$\text{Output} = \frac{1}{\sqrt{128}} H_{128} X$$

### 2. Branchless Ternary Arithmetic (`PTQ1_0` and `PQ2_0`)
Ternary weights $\{-1, 0, +1\}$ are encoded in 2-bit packed nibbles ($00_2 \implies 0, 01_2 \implies +1, 10_2 \implies -1$). The decode loop executes branchless unpack arithmetic:
$$\text{sign} = ((W \gg 2i) \ \& \ 1) - ((W \gg 2i) \ \& \ 2)$$
Eliminating warp divergence and mapping directly to SIMD integer instructions (`__dp4a` / `vpdpbusd`).

### 3. In-Kernel Byte-DFA Schema Grammar Masking
JSON schemas and tool-call signatures are compiled into an offline Deterministic Finite Automaton (DFA) transition matrix $\delta(S, \text{byte})$. Prior to token sampling, active vocabulary tokens that violate valid schema transitions have their logits set to $-\infty$, guaranteeing $100\%$ schema conformance without host roundtrips.

---

## DAG Tree-Native Reasoning & Speculative Rollbacks

`Oxide-Tech-LLM-Engine` provides native primitives for tree search (MCTS), chain-of-thought exploration, and speculative decoding:

- **Lock-Free Copy-on-Write (CoW) Blocks ([`SharedPhysicalBlock`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-core/src/dag.rs))**: Spawning exploration branches increments atomic reference counts without copying device VRAM.
- **Topological DAG Attention Masking**: Maps arbitrary branching trees into unified packed attention masks, processing shared prompt prefixes once across all child paths.
- **Transactional Speculative Rollback ([`TransactionalBlockTable`](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/crates/oxide-alloc/src/transactional.rs))**: Speculative rejection truncates only host block metadata; device VRAM is never zeroed or rewritten.

---

## CLI Quickstart & Server API

### 1. CLI Usage (`oxide`)

```bash
# Serve Ternary Bonsai 2 on NVIDIA CUDA with target RTX 4090 profile
oxide --model bonsai2 --backend cuda --gpu "RTX 4090" --serve 127.0.0.1:8080

# Serve Needle 3 on Apple Silicon Metal
oxide --model needle3 --backend metal --gpu "Apple M4 Max" --serve 127.0.0.1:8080

# Serve Llama 3 on Intel Arc Battlemage with Hierarchical KV Cache
oxide --model llama3 --backend intel --gpu "Arc B580" --kv-device-blocks 2048 --kv-host-blocks 16384

# Serve on Raspberry Pi 5 with AI HAT+ 2 (Hailo-8)
oxide --model bonsai2 --backend hailo --gpu "RPi5 with AI HAT+ 2"

# Serve on Rockchip Orange Pi 6 Plus
oxide --model bonsai2 --backend rknn --gpu "Orange Pi 6 Plus"

# Run Quantitative Trading Foundation Model (Kronos)
oxide --model kronos --backend cuda --serve 127.0.0.1:8080
```

### 2. HTTP/2 & Server-Sent Events (SSE) API

#### `POST /v1/chat/completions`
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Explain zero-copy memory hierarchy in Oxide."}],
    "stream": true,
    "temperature": 0.2
  }'
```

#### `GET /health`
```json
{
  "status": "healthy",
  "engine": "oxide-tech-llm-engine",
  "version": "0.1.0",
  "hardware_accelerator": "cuda",
  "active_slots": 64,
  "memory_arena_status": "zero_alloc_ready"
}
```

---

## Workspace Crate Structure

```
Oxide-Tech-LLM-Engine/
├── Cargo.toml                              # Workspace root definition (Rust 2024, resolver = "3")
├── crates/
│   ├── oxide-core/                         # Core traits, DevicePtr<T>, hardware markers, typestates, DAG
│   ├── oxide-alloc/                        # Static HostPinnedArena, DeviceMemoryArena, HierarchicalKvCache
│   ├── oxide-hardware/
│   │   ├── oxide-backend-cuda/             # NVIDIA CUDA backend & custom PTQ1_0 / FWHT kernels
│   │   ├── oxide-backend-rocm/             # AMD ROCm / CDNA4 / Strix Halo backend
│   │   ├── oxide-backend-metal/            # Apple Silicon Metal backend & MSL shaders
│   │   ├── oxide-backend-tpu/              # Google TPU v6e/v5/v4 & Edge Coral backend
│   │   ├── oxide-backend-intel/            # Intel Arc Battlemage & Xeon 6980P Level-Zero backend
│   │   ├── oxide-backend-qualcomm/         # Qualcomm Snapdragon X Elite / 8 Elite HTP backend
│   │   ├── oxide-backend-rknn/             # Rockchip RK3588 / Orange Pi 6 Plus backend
│   │   ├── oxide-backend-hailo/            # Raspberry Pi 5 AI HAT+ 2 / Hailo-8 backend
│   │   └── oxide-backend-cpu/              # AVX-512 & ARM Neon vectorized CPU backend
│   ├── oxide-models/                       # Model specifications, architectures, specialized domain engines
│   ├── oxide-quant/                        # PTQ1_0, PQ2_0, CQ2, NVFP4 quantization routines
│   ├── oxide-server/                       # Axum HTTP/2 server, SSE streaming, DFA grammar masks
│   ├── oxide-engine/                       # Monomorphized SpecializedPipeline coordinator
│   └── oxide-cli/                          # Solitary standalone binary entry point (`oxide`)
├── docs/                                   # Full technical documentation suite
└── tests/                                  # Integration and golden oracle verification harnesses
```

---

## Building & Verification

```bash
# Format check
cargo fmt --check

# Strict zero-warning lint check
cargo clippy --workspace -- -D warnings

# Execute comprehensive integration test suites
cargo test --workspace

# Production release build with fat LTO and binary stripping
cargo build --release --workspace

# Synchronize AST and semantic knowledge graph with oxide-embed
oxide-embed index --device auto
```

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
