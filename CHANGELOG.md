# Changelog

All notable changes to the **Oxide-Tech-LLM-Engine** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-10-05

### Added

#### Dynamic ggml / llama.cpp Execution Architecture & Universal Model Selection
- **Tripartite Architecture**: Pre-compiled kernel library + streaming universal GGUF/SafeTensors mmap loader + dynamic compute graph DAG with kernel fusions (`FusedRmsMulMat`).
- **Zero-Recompilation Guarantee**: Single binary runtime supporting dynamic loading of any model architecture (`llama`, `qwen2`, `deepseek2`, `mistral`, `gemma`, `phi3`) without recompiling.
- **Zero-Allocation `GraphArena` Bump Allocator (`crates/oxide-engine/src/arena.rs`)**: $O(1)$ turnaround memory allocation with zero heap fragmentation during forward passes.
- **Dynamic Model Manager (`crates/oxide-engine/src/model_manager.rs`)**: Automatic hierarchical model discovery across exact paths, `--models-dir`, `./models/`, current working directory, and engine catalog.
- **`llama.cpp` CLI Parity (`crates/oxide-cli/src/lib.rs`)**:
  - Subcommands: `chat`, `server`, `img`.
  - Flags: `-m / --model`, `-p / --prompt`, `-i / --interactive`, `-ngl / --n-gpu-layers`, `--models-dir`, `--serve`.
  - Single-hyphen `-ngl` normalized automatically for script and ecosystem parity.
- **Interactive REPL Model Hot-Swapping**: In-session commands `/model <path_or_name>`, `/models`, `/list`, and `/info` to switch models without restarting the process.
- **OpenAI Dynamic Model Hot-Swapping (`crates/oxide-server/src/lib.rs`)**:
  - Endpoint `POST /v1/models/load` to dynamically load or switch models on the fly.
  - Automatic on-demand model resolution in `POST /v1/chat/completions` and `POST /v1/completions`.

#### Comprehensive Academic Sampling Algorithms Suite (`crates/oxide-core/src/sampler.rs`)
- **Mirostat (v1 & v2)**: Active dynamic entropy regulation maintaining target perplexity $\tau$ via online surprise updates and dynamic temperature scaling.
- **DRY (Don't Repeat Yourself) Sampling**: Exponential multi-token prefix matching repetition penalty scanning recent generation history backwards to break repetitive cycles.
- **XTC (Exclude Top Choices)**: Dynamic truncation mechanism masking out dominant top tokens during high-confidence sampling to explore secondary high-quality reasoning branches.
- **Min-P Sampling**: Probability threshold cutoff relative to top candidate probability ($p_i \ge p_{\max} \cdot p_{\text{base}}$).
- **Tail-Free Sampling (TFS-Z)**: Second-derivative curvature detection cutting off flat distribution tails without fixed $P$ or $K$ limits.
- **Locally Typical Sampling**: Information-density matching minimizing the distance between conditional information content and Shannon entropy $|-\log_2 p_i - H(P)|$.
- **Grammar-Based Sampling (GBNF & In-Kernel DFA)**: Strict bitmask logit masking enforcing valid JSON, SQL, or Python syntax.
- **Logit Bias & Token Bans**: Dynamic positive/negative token weight shifting and hard suppression.
- **Penalize Newline (`--penalize-nl`)**: Fine-grained penalty suppressing runaway newline spamming.

#### Complete Integer Quantization Suite (`crates/oxide-quant/src/int_quant.rs`)
- **2-bit K-Quant (`BlockQ2_K`)**: 256 weights packed into 16 sub-blocks with 2 bits per weight, 16 scale bytes, and super-block scale/min factors ($2.5625\text{ bpw}$).
- **3-bit K-Quant (`BlockQ3_K`)**: 256 weights packed into low 2-bit and high 1-bit arrays with 16 scale factors ($3.4375\text{ bpw}$).
- **4-bit Standard (`BlockQ4_0`, `BlockQ4_1`)**: 32 weights per block in symmetric and affine configurations with direct SIMD integer dot products.
- **5-bit Standard (`BlockQ5_0`)**: 32 weights per block with high 5th-bit array `qh` and low 4-bit array `qs` ($5.5\text{ bpw}$).
- **6-bit K-Quant (`BlockQ6_K`)**: 256 weights split across 128 bytes of low 4 bits, 64 bytes of high 2 bits, and 16 signed 8-bit scales ($6.5625\text{ bpw}$).
- **8-bit Standard (`BlockQ8_0`)**: 32 signed int8 weights with half-precision scale factor for fast integer matrix multiplication.
- **Half-Precision Float (`f16`)**: Custom zero-dependency IEEE 754 half-precision float type with subnormal and infinity handling.

#### Universal GGUF Parser & Loader (`crates/oxide-models/src/formats.rs`)
- **Full GGUF Specification Support**: Parses GGUF v1, v2, and v3 binary headers and key-value metadata dictionaries.
- **All 13 Value Types**: `Uint8`, `Int8`, `Uint16`, `Int16`, `Uint32`, `Int32`, `Float32`, `Bool`, `String`, `Array`, `Uint64`, `Int64`, `Float64`.
- **Tensor Directory & Offsets**: Memory-mapped parsing of tensor names, shapes, quantization types, and data offsets.

#### Heterogeneous Multi-Device Tensor Splitting & NUMA (`crates/oxide-engine/src/tensor_split.rs`, `hybrid.rs`)
- **Heterogeneous Tensor Splitting**: Distributes tensor rows/columns across mixed accelerators (NVIDIA CUDA, AMD ROCm, Apple Metal, Intel NPU, Google TPU v4/v5, and Host CPU NUMA nodes).
- **All-Reduce Ring Aggregation**: Asynchronous inter-device all-reduce sum reduction.
- **CPU+GPU Hybrid Inference**: Automatic VRAM-budgeted layer partitioning to run oversized models across CPU and multiple GPUs.

#### Advanced KV Cache Operations (`crates/oxide-alloc/src/kv_advanced.rs`)
- **Context Shifting**: Sliding window manager preserving system prompt prefixes and recalculating RoPE position offsets.
- **Prompt Caching**: 64-bit FNV prefix hashing and instant prompt prefill reuse.
- **On-The-Fly KV Quantization**: In-flight compression of KV tensors into `Q8_0`, `Q4_0`, `FP8`, or `INT4`.
- **KV Cache Dump & Reload**: Binary zero-copy persistence container for pausing and resuming agent sessions.

#### Flash Attention, RoPE Scaling, LoRA, Projectors & Chat Templates (`crates/oxide-models/`)
- **Flash Attention Engine (`flash_attn.rs`)**: In-SMem tiled online softmax algorithm operating in $O(1)$ intermediate SRAM memory.
- **RoPE Position Scaling (`rope.rs`)**: Support for YaRN, LongRoPE, Llama-3, and Linear RoPE context scaling.
- **LoRA & QLoRA Hot-Swapping (`lora.rs`)**: Dynamic adapter insertion and quantized base weight delta application.
- **Multi-Modal Projectors (`projector.rs`)**: Linear and MLP-GELU cross-modal projection engines.
- **Chat Template Parser (`chat_template.rs`)**: Renders ChatML, Llama-3, DeepSeek, Mistral, and Alpaca chat templates.
- **Decision-Making Models (`specialized.rs`, `registry.rs`)**: Added `JEV` (Joint Estimation of Value), `LAYA` (Latent Action Yielding Agent), and `CLEF` (Causal Latent Evidence Framework).

#### OpenAI API Parity Server & Continuous Batching (`crates/oxide-server/src/lib.rs`, `crates/oxide-engine/src/slot_manager.rs`)
- **OpenAI HTTP Endpoints**:
  - `GET /v1/models`
  - `POST /v1/chat/completions` (Streaming SSE + Non-Streaming)
  - `POST /v1/completions`
  - `POST /v1/embeddings`
  - `GET /health`
- **Continuous Batching Slot Manager**: Non-blocking iteration-level scheduling, dynamic slot state machine, and KV block recycling.

---

## [0.1.0] - 2026-10-05

### Added
- Foundational pure-Rust workspace with `#![deny(unsafe_op_in_unsafe_fn)]` and Rust 2024 edition.
- Zero-cost memory typing with `DevicePtr<T>` and compile-time device/host safety proofs.
- In-SMem Fast Walsh-Hadamard Transform (FWHT) and branchless ternary GEMV (`PTQ1_0`, `PQ2_0`).
- 3-Tier Hierarchical KV cache (`HierarchicalKvCache`) across VRAM, pinned host RAM, and NVMe.
- Hardware backends: CUDA, ROCm, Metal, TPU, Intel, Snapdragon, RKNN, Hailo, and CPU.
- Domain-specialized engines: MusubiCAD, CAD-Coder, AtomAgents, AMMap, MechRAG, SpecForge AI, Academic Writing, and Kronos.
- DAG tree-native reasoning engine with lock-free copy-on-write blocks.
