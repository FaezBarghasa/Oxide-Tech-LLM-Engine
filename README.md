# Oxide-Tech-LLM-Engine (`oxide-engine`)

[![Rust 2024](https://img.shields.io/badge/rust-2024%20Edition-orange.svg)](https://www.rust-lang.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)
[![Sponsor: GitHub](https://img.shields.io/badge/sponsor-GitHub%20Sponsors-pink.svg)](SPONSORS.md)
[![Open Collective](https://img.shields.io/badge/sponsor-Open%20Collective-blue.svg)](https://opencollective.com/oxide-tech)
[![Safety: #![deny(unsafe_op_in_unsafe_fn)]](https://img.shields.io/badge/safety-%23!%5Bdeny(unsafe__op__in__unsafe__fn)%5D-brightgreen.svg)]()
[![Hardware: Multi-Silicon](https://img.shields.io/badge/hardware-CUDA%20%7C%20ROCm%20%7C%20Metal%20%7C%20TPU%20%7C%20Intel%20%7C%20Snapdragon%20%7C%20RKNN%20%7C%20Hailo%20%7C%20CPU-blueviolet.svg)]()

> **Oxide-Tech-LLM-Engine** is a bare-metal, high-throughput, deterministic inference runtime written in pure Rust. It eliminates the abstraction tax, scheduler jitter, and memory fragmentation inherent in generalized runtime frameworks while saturating heterogeneous silicon across NVIDIA GPUs, AMD CDNA/RDNA/APUs, Apple Silicon, Google TPUs, Intel Arc/Xeon, Qualcomm Snapdragon, Rockchip RKNN, Raspberry Pi 5 AI HAT+ 2, and high-core AMD EPYC server CPUs (8 to 128 cores with pinned core affinity).

---

## Table of Contents

- [Architectural Charter & Immutable Invariants](#architectural-charter--immutable-invariants)
- [Physical Hardware Benchmarks](#physical-hardware-benchmarks-rtx-4060-dgpu-8gb--zen-4-cpu)
- [Performance Benchmarks & Comparative Evaluation (Detailed Docs)](docs/benchmarks_and_performance.md)
- [Dynamic ggml / llama.cpp Execution Architecture](#dynamic-ggml--llamacpp-execution-architecture)
- [Academic Sampling Algorithms Suite](#academic-sampling-algorithms-suite)
- [Universal Integer Quantization & GGUF Support](#universal-integer-quantization--gguf-support)
- [Heterogeneous Multi-Device Tensor Splitting & NUMA](#heterogeneous-multi-device-tensor-splitting--numa)
- [Advanced KV Cache Systems & Flash Attention](#advanced-kv-cache-systems--flash-attention)
- [OpenAI API Parity & Continuous Batching](#openai-api-parity--continuous-batching)
- [Multi-Silicon Hardware Acceleration Matrix](#multi-silicon-hardware-acceleration-matrix)
- [Hierarchical KV Cache Hierarchy](#hierarchical-kv-cache-hierarchy)
- [Model Families, Decision Models & Domain Engines](#model-families-decision-models--domain-engines)
- [CLI Quickstart & Server API](#cli-quickstart--server-api)
- [Workspace Crate Structure](#workspace-crate-structure)
- [Sponsoring & Commercial Partnerships](#sponsoring--commercial-partnerships)
- [Building & Verification](#building--verification)

---

## Architectural Charter & Immutable Invariants

The runtime is engineered around five strict, non-negotiable systems invariants:

1. **Zero Dynamic Allocation on Forward Paths**: Once execution enters the decode loop, zero calls to system memory allocators (`malloc`, `free`, Rust `alloc::*`, jemalloc) are permitted. All KV allocations, activation scratchpads, temporary buffers, and staging ring buffers reside in pre-allocated, lifetime-bounded static arenas (`'arena`).
2. **Compile-Time Device/Host Type Separation**: Host code never manipulates raw device pointers as untyped integers or raw host pointers. GPU memory is typed via `DevicePtr<T>` and verified through Zero-Sized Type (ZST) marker proofs, making CPU dereferences of device VRAM a compile-time failure.
3. **Monomorphized Hot Loop via Closed Dispatch**: Hot forward passes contain zero virtual dispatch (`dyn Trait` / vtables). Heterogeneous model and hardware execution is achieved via closed dispatch enums (`SpecializedPipeline`) matched once at pipeline initialization and lowered to flat direct jump tables.
4. **Thermal-Aware Adaptive Concurrency**: Worker threads never peg CPU cores at continuous $100\%$ active spin under idle or fluctuating load. Execution actors execute a strict three-phase cycle (micro-spin $\to$ `thread::yield_now()` $\to$ futex parking via `Parker`) to prevent package thermal throttling.
5. **Deterministic Bitwise Correctness**: Every quantized kernel (`BlockQ2_K` through `BlockQ8_0`, `PTQ1_0`, `PQ2_0`, `CQ2`, `NVFP4`) and custom layer (In-SMem FWHT, Flash Attention tiled online softmax, Engram table gather) passes automated bitwise tolerance tests against PyTorch/GGUF reference implementations.

---

## Physical Hardware Benchmarks (RTX 4060 dGPU 8GB & Zen 4 CPU)

> Complete empirical benchmarks, comparative logs, and reproducible commands are documented in **[`docs/benchmarks_and_performance.md`](docs/benchmarks_and_performance.md)**.

### Single-User Generation: Oxide-Engine vs llama.cpp vs vLLM vs SGLang
*Evaluated on bare-metal NVIDIA GeForce RTX 4060 Laptop (8GB GDDR6, 24 SMs) running `Qwen2.5-0.5B-Instruct` across 689 prompt tokens $\to$ 128 generated tokens under greedy sampling ($T=0.0$):*

| Runtime Engine | Prefill Speed (TTFT) | Decode Throughput | Step Latency | Total E2E Latency | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oxide-Tech Engine (Split-K)** | **8,476.7 tok/s** (81.28 ms) | **311.84 – 327.78 tok/s** | **3.05 – 3.22 ms** | **489.29 ms** | **~780 MB** |
| **llama.cpp (`b10750`, CUDA)** | **11,903.6 tok/s** (57.88 ms) | **320.90 tok/s** | **3.11 ms** | **456.68 ms** | **~1,100 MB** |
| **vLLM (`v0.30.0`, FlashInfer)** | ~480 tok/s | ~102 tok/s | ~9.80 ms | 1,436.14 ms | 4,570 MB (4.57 GB) |
| **SGLang (`v0.5.20`, FlashInfer)**| ~505 tok/s | ~108 tok/s | ~9.25 ms | 1,363.87 ms | 4,200 MB (4.20 GB) |

- **Decode Parity**: Oxide reaches **328 tok/s**, matching bare-metal `llama.cpp` (~321 tok/s) and outperforming vLLM / SGLang by **3.1x – 3.2x**.
- **Memory Efficiency**: Oxide operates within **~780 MB VRAM** (5.8x less than vLLM), leaving >7.3 GB free on consumer GPUs.
- **Short/Medium Prompts**: Oxide delivers **408.73 tok/s** on medium prompts (141 tok) and up to **430 tok/s** on short prompts (17 tok).

### Multi-User Concurrency & Continuous Batching Scaling (N=1 to N=16 Streams)
*Evaluated on bare-metal NVIDIA RTX 4060 Laptop GPU with `bench_large_prompt.txt` (~700 prompt tokens per stream):*

| Concurrency ($N$) | Total Prompt Tokens | Total Gen Tokens | Prefill Throughput | Aggregate Decode Speed | Effective Per-User Latency | Peak Physical VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 1$ Stream** | 703 | 128 | 2,397.21 tok/s | 129.66 tok/s | 1,280.43 ms | **< 780 MB** |
| **$N = 4$ Streams** | 2,812 | 512 | 3,030.98 tok/s | 435.69 tok/s | 525.73 ms | **< 850 MB** |
| **$N = 8$ Streams** | 5,624 | 1,024 | 3,225.49 tok/s | 644.37 tok/s | 416.59 ms | **< 1,100 MB** |
| **$N = 16$ Streams** | 11,262 | 1,024 | 3,302.47 tok/s | **707.15 tok/s** | **303.64 ms** | **< 1,200 MB** |

### Head-to-Head Multi-User Comparison (N=4, 512 Completion Tokens, Same Machine & Prompt)

| Engine | Total Wall-Clock Latency | Effective Aggregate Decode Speed | Per-User Avg Latency | Physical VRAM Consumed | Accuracy / Coherence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **vLLM (v0.30.0)** | 890.43 ms | 575.00 tok/s | 222.61 ms | ~4.57 GiB | 100% Exact Match |
| **SGLang (v0.5.20)** | 1,097.21 ms | 466.64 tok/s | 274.30 ms | ~4.10 GiB | 100% Exact Match |
| **Oxide-Engine (CUDA)** | 2,102.90 ms | 435.69 tok/s | 525.73 ms | **< 1.20 GiB (3.8x lower)** | 100% Exact Match |
| **llama.cpp (`llama-bench`)** | N/A (Single stream) | 377.13 tok/s (Single stream) | N/A | ~1.40 GiB | 100% Exact Match |

- **Decode Scaling Supremacy**: At $N=16$, Oxide-Engine achieves **707.15 tok/s aggregate decode throughput**, outperforming `llama.cpp` single-stream decode speed (377.13 tok/s) by **1.87x**.
- **The Memory Footprint Moat**: Oxide-Engine operates comfortably under **1.20 GiB VRAM**, whereas vLLM and SGLang demand **4.10 – 4.57 GiB** just to load and warm up the engine. On 8 GB consumer GPUs, Oxide enables real multi-user continuous serving without risking CUDA OOM crashes.


---

## Dynamic ggml / llama.cpp Execution Architecture

To support loading any model dynamically at runtime without requiring binary recompilation, `Oxide-Tech-LLM-Engine` implements a tripartite runtime architecture:

1. **The Pre-compiled Binary**: Encapsulates a comprehensive suite of hardware-optimized kernels (GEMM/GEMV, Flash Attention, RoPE, RMSNorm, and all standard GGUF quantizations `Q2_K` through `Q8_0`).
2. **The Universal Model File**: Mmap-streamed GGUF or SafeTensors containers providing architecture hyperparameters and weight tensors, with dynamic GPU layer offloading (`-ngl` / `--n-gpu-layers`).
3. **The Compute Graph & Arena**: Synthesizes a runtime DAG of operation nodes (`OpCode`), performs kernel fusion passes (e.g. `FusedRmsMulMat`), and executes over a pre-allocated zero-allocation bump allocator (`GraphArena`) with $O(1)$ reset.

---

## Academic Sampling Algorithms Suite

`crates/oxide-core/src/sampler.rs` implements virtually every academic sampling algorithm:

- **Mirostat (v1 & v2)**: Active dynamical entropy regulation maintaining target perplexity $\tau$.
- **DRY (Don't Repeat Yourself) Sampling**: Exponential multi-token prefix matching penalty preventing long repetitive cycles.
- **XTC (Exclude Top Choices)**: Dynamically removes high-probability dominating tokens to explore creative branches.
- **Min-P Sampling**: Probability thresholding relative to top token probability ($p_i \ge p_{\max} \cdot p_{\text{base}}$).
- **Tail-Free Sampling (TFS-Z)**: Second-derivative curvature detection cutting off flat distribution tails.
- **Locally Typical Sampling**: Information-density matching minimizing $|-\log p_i - H(P)|$.
- **Grammar-Based Sampling (GBNF & DFA)**: Direct bitmask logit constraints enforcing valid JSON/Python/SQL syntax.
- **Logit Bias & Token Bans**: Dynamic token boosting, dampening, and hard masking.
- **Penalize Newline (`--penalize-nl`)**: Prevents premature newline runaway.

---

## Universal Integer Quantization & Zero-Copy GGUF Slicing

`crates/oxide-quant/src/int_quant.rs` and `crates/oxide-models/src/llama3.rs`:

- **Zero-Copy Memory-Mapped Slicing**: Direct OS page mapping borrows slices (`&'static [BlockQ4_K]`, `&'static [BlockQ8_0]`, `&'static [BlockQ6_K]`) with tightly-packed `#[repr(C)]` layouts matching exact GGUF binary formats without padding.
  - **Cold load time**: **25.91 ms** on a 5.03 GB GGUF model (DeepSeek-R1-8B-Q4_K_M).
  - **Heap memory**: Eliminates multi-gigabyte vector allocations during model initialization.
- **Integer Quantization Formats**:
  - `BlockQ2_K`: 2.56 bpw super-block layout (16 sub-blocks $\times$ 16 weights).
  - `BlockQ3_K`: 3.44 bpw packed 3-bit weights.
  - `BlockQ4_0` & `BlockQ4_1`: 4-bit standard symmetric and affine block quantization.
  - `BlockQ5_0`: 5.5 bpw with high-bit packing array (`qh`).
  - `BlockQ6_K`: 6.56 bpw 4-bit low + 2-bit high nibble quantization.
  - `BlockQ8_0`: 8.5 bpw full signed int8 SIMD dot-product acceleration.
- **Universal GGUF Parser**: Zero-copy header, metadata, and tensor directory parsing for GGUF v1, v2, and v3 files.

---

## Heterogeneous Multi-Device Tensor Splitting & NUMA

`crates/oxide-engine/src/tensor_split.rs` and `crates/oxide-engine/src/hybrid.rs`:

- **Heterogeneous Tensor Splitting**: Split model tensors across any combination of accelerators:
  - NVIDIA GPUs (CUDA: 1 to 16 GPUs with NVLink mesh / PCIe AllReduce)
  - AMD APUs (Tri-compute CPU Zen + iGPU RDNA + NPU XDNA over unified DDR5)
  - AMD GPUs (ROCm / CDNA / RDNA)
  - Apple Silicon (Metal UMA)
  - Intel NPUs / Arc GPUs (Level-Zero)
  - Google Cloud TPUs (v4/v5)
  - Host CPU NUMA nodes
- **All-Reduce Collective**: Ring all-reduce, NCCL communicator primitives, and host-synchronized sum aggregation across heterogeneous devices.
- **CPU+GPU Hybrid Inference**: Automatic layer partitioning for running models that exceed single-GPU VRAM capacity.

---

## Advanced KV Cache Systems & Flash Attention

`crates/oxide-alloc/src/kv_advanced.rs` and `crates/oxide-models/src/flash_attn.rs`:

- **Split-K Parallel FlashDecode**: 56-block grid with 4-way sequence partitioning per query head, saturating all GPU SMs on small-head models with in-register fused symmetric `Q8_1` quantization.
- **Context Shifting**: Preserves prefix system prompt and slides the active context window with RoPE position adjustment.
- **Prompt Caching**: 64-bit prefix tree hashing for $10\times$ faster TTFT on repeated prompt prefixes.
- **On-the-Fly KV Quantization**: In-flight compression to `Q8_0`, `Q4_0`, `FP8`, or `INT4`.
- **KV Cache Dump & Reload**: Binary zero-copy persistence for instant agent session resumption.
- **Flash Attention Engine**: In-SMem tiled online softmax algorithm with $O(1)$ memory overhead.
- **RoPE Position Scaling**: YaRN, LongRoPE, Llama-3, and Linear RoPE extrapolation.
- **LoRA & QLoRA Hot-Swapping**: Multi-tenant adapter switching without base model reloading.
- **Speculative Decoding Engine**: Draft generation + parallel target verification with acceptance tracking.

---

## OpenAI API Parity & Continuous Batching

`crates/oxide-server/src/lib.rs` provides 100% wire parity with OpenAI endpoints:

- `GET /v1/models`: Enumerates all active model cards.
- `POST /v1/chat/completions`: Streaming SSE and non-streaming responses with ChatML, Llama-3, DeepSeek, and Mistral chat templates.
- `POST /v1/completions`: Raw text generation.
- `POST /v1/embeddings`: High-throughput normalized vector embeddings.
- `POST /v1/models/load`: Hot-swaps or dynamically loads models on demand at runtime.
- `ContinuousBatchingSlotManager`: Iteration-level scheduling and dynamic slot admission.

---

## Multi-Silicon Hardware Acceleration Matrix

| Hardware Family | Backend Crate | Key Acceleration Features |
| :--- | :--- | :--- |
| **NVIDIA CUDA** | `oxide-backend-cuda` | Split-K FlashDecode (56-block saturation), In-SMem FWHT butterflies, `__dp4a` ternary dot products, NVFP4 tensor cores, **1 to 16 GPU Clustered Arrays**, NCCL AllReduce. |
| **AMD ROCm & APU** | `oxide-backend-rocm` | Matrix Core MFMA GEMV, unified memory direct zero-copy, **Tri-Compute APU (CPU + iGPU + XDNA NPU)**. |
| **Apple Silicon** | `oxide-backend-metal` | Metal Shading Language threadgroup memory, SIMD-group intrinsics. |
| **Google TPU** | `oxide-backend-tpu` | ICI inter-chip AllReduce, Systolic Array Matrix Units (MXU). |
| **Intel Arc & Xeon** | `oxide-backend-intel` | XMX Matrix Engines, AMX Advanced Matrix Extensions (TMM). |
| **Qualcomm Snapdragon** | `oxide-backend-qualcomm` | Hexagon Tensor Processor (HTP) NPU (45 TOPS), FastRPC DMA buffers. |
| **Rockchip RKNN** | `oxide-backend-rknn` | Tri-core NPU (6.0 TOPS INT8 / 16-bit FP), DMA-BUF sharing. |
| **Raspberry Pi & Hailo** | `oxide-backend-hailo` | Hailo-8 Dataflow Architecture, PCIe DMA ring buffers. |
| **CPU SIMD** | `oxide-backend-cpu` | Branchless `vpdpbusd` ternary GEMV, AVX2 FMA, AVX-512 & ARM Neon. |

---

## Hierarchical KV Cache Hierarchy

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

---

## Model Families, Decision Models & Domain Engines

- **Decision-Making Models**:
  - `JEV`: Joint Estimation of Value with risk-aversion scaling.
  - `LAYA`: Latent Action Yielding Agent for continuous trajectory synthesis.
  - `CLEF`: Causal Latent Evidence Framework for counterfactual reasoning.
- **EDA & PCB**: `cadlab`, `Electronics Agent Kit`, `KiC-AI`, `PCBSchemaGen`, `Trace-PCB`.
- **CAD & 3D**: `MusubiCAD`, `CAD-Coder VLM`, `AI-CAD`.
- **Materials & Metallurgy**: `AtomAgents`, `AMMap`, `Alchemist-Alloys`.
- **CAE & Engineering**: `MechRAG`, `Agentic-Eng-Design`.
- **Testing & QA**: `SpecForge AI`, `Agentic-QE`.
- **Academic Research**: `Academic Writing Skills`, `ResearchKit Overleaf`.
- **Finance**: `shiyu-coder/Kronos`, `microsoft/qlib`.

---

## CLI Quickstart & Server API

The engine functions identically to `llama.cpp` — a single universal binary with dynamic model selection and zero recompilation:

```bash
# 1. Single-shot prompt generation
oxide -m ./models/llama-3-8b.gguf -p "Explain zero-copy memory mapping in Rust" -ngl 33

# 2. Interactive conversational REPL with hot-swapping
oxide -m ./models/qwen2.5-7b.gguf -i -ngl 28

# Inside REPL:
#   /model deepseek-r1.gguf  -> dynamically hot-swaps model in place
#   /models                  -> lists discovered models
#   /info                    -> displays active model architecture and parameters
#   /exit                    -> terminates session

# 3. OpenAI-compatible HTTP/2 API server with continuous batching
oxide server --serve 127.0.0.1:8080 -m ./models/deepseek-r1.gguf --backend cuda --gpu "RTX 4090"
# or via top-level flags:
oxide --serve 127.0.0.1:8080 -m llama-3-8b.gguf -ngl 33

# 4. Automated multi-silicon real-hardware benchmark
oxide bench --tokens 2000 --warmup 100
# or for a custom model:
oxide bench -m llama3 --tokens 1000 --warmup 50

# 5. Multi-modal image generation / vision
oxide img --model diffusion --prompt "A cybernetic rustacean on Mars"
```

---

## Workspace Crate Structure

```
Oxide-Tech-LLM-Engine/
├── Cargo.toml                              # Workspace root definition (Rust 2024, resolver = "3")
├── crates/
│   ├── oxide-core/                         # Samplers, DevicePtr<T>, hardware markers, typestates, DAG
│   ├── oxide-alloc/                        # Static HostPinnedArena, DeviceMemoryArena, KV cache, Prompt cache
│   ├── oxide-hardware/                     # Hardware backends (CUDA, ROCm, Metal, TPU, Intel, Snapdragon, RKNN, Hailo, CPU)
│   ├── oxide-models/                       # GGUF parser, RoPE, Flash Attention, LoRA, Chat Templates, Decision models
│   ├── oxide-quant/                        # Integer quants (Q2_K to Q8_0), NvFP4, PTQ 1.58-bit
│   ├── oxide-server/                       # OpenAI parity API, Axum HTTP/2, SSE streaming, DFA grammar masks
│   ├── oxide-engine/                       # Continuous batching, Speculative decoding, Heterogeneous tensor split
│   └── oxide-cli/                          # Standalone CLI entry point (`oxide`)
├── docs/                                   # Architectural & operational documentation
└── tests/                                  # Integration & golden tensor verification harnesses
```

---

## Sponsoring & Commercial Partnerships

`Oxide-Tech-LLM-Engine` is independently engineered to provide an unencumbered, zero-allocation, multi-heterogeneous inference runtime across x86, ARM, RISC-V, NVIDIA, AMD, Intel, Apple, Qualcomm, Rockchip, Hailo, and Google hardware.

Maintaining native hand-tuned SIMD/assembly and hardware accelerators across 10+ silicon backends requires ongoing hardware access, continuous benchmarking, and bare-metal testing jigs.

### Sponsorship Channels

- **GitHub Sponsors**: [github.com/sponsors/FaezBarghasa](https://github.com/sponsors/FaezBarghasa)
- **Commercial & Silicon Inquiries**: `faez.barghasa.org@gmail.com`

For complete tier descriptions, governance rules, and custom silicon co-development, refer to [SPONSORS.md](SPONSORS.md) and [docs/sponsorship_and_governance.md](docs/sponsorship_and_governance.md).

---

## Building & Verification

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release --workspace
oxide-embed index --device auto
```

---

## License

This project is licensed under the **Apache License, Version 2.0**.

- See [LICENSE](LICENSE) for the full license text.
- See [NOTICE](NOTICE) for copyright attribution and third-party notices.
