# Changelog

All notable changes to the **Oxide-Tech-LLM-Engine** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-10-05

### Added

#### Complete Universal Quantization Matrix (`crates/oxide-quant/`)
- **GGUF / llama.cpp Full Precision Suite**:
  - Legacy & Standard Integer: `Q1_0`, `Q2_0`, `Q4_0`, `Q4_1`, `Q4_2`, `Q4_3`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`.
  - K-Quants: `Q2_K`, `Q2_K_S`, `Q3_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_K`, `Q4_K_S`, `Q4_K_M`, `Q5_K`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_K`.
  - I-Quants (Importance Matrix Codebooks): `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, `IQ4_XS`, `IQ4_NL`.
  - Ternary Quantization: `TQ1_0`, `TQ2_0`, `Ptq1_0Ternary`, `BitNet1_58`.
- **GPTQ (Generative Pre-trained Transformer Quantization)**:
  - Multi-bit weight quantization: 2-bit, 3-bit, 4-bit (`GptqW4A16`), and 8-bit column-packed layouts with group scales, zero-points, and activation-order (desc_act) permutation.
- **AWQ (Activation-aware Weight Quantization)**:
  - Precision modes: `W4A16` (INT4 weights, FP16 activations), `W4A8` (INT4 weights, FP8 activations), `W8A16` (INT8 weights, FP16 activations), and native `BF16` with interleaved bank-conflict-free warp layouts `[0, 4, 1, 5, 2, 6, 3, 7]`.
- **EXL2 (ExLlamaV2 Fractional-Bit Formats)**:
  - Fractional-bit quantization engine (`crates/oxide-quant/src/exl2.rs`): `2.0bpw`, `3.0bpw`, `3.5bpw`, `4.0bpw`, `4.25bpw`, `5.0bpw`, `6.0bpw`, `6.5bpw`, and `8.0bpw`.
- **bitsandbytes Suite**:
  - `LLM.int8()` vector-wise outlier decomposition (`BnbInt8`).
  - `NF4` (NormalFloat4) information-theoretically optimal quantile quantization (`BlockNf4_64`, `NF4_TABLE`).
  - `FP4` (Float4 E2M1) for uniformly distributed weights (`BlockFp4Bnb_64`, `FP4_TABLE`).
  - `FP8 E4M3` format compatibility.
- **Floating-Point Precision Formats**:
  - IEEE & AI floats: `FP64`, `FP32`, NVIDIA `TF32`, `BF16`, `FP16`, `FP8 E4M3` (dynamic range 448.0), `FP8 E5M2` (extended dynamic range up to 57344.0), and `FP4 E2M1`.
- **OCP Microscaling (MX) & Hardware Acceleration**:
  - `MXFP8`, `MXFP4`, `MXFP6`, `MXINT8`, and `MXFP4_MOE` using power-of-two `E8M0` scales per 32 elements.
  - Blackwell `NVFP4` dual-scaling layout (`BlockNvFp4_16`).
  - `Marlin` Tensor Core GEMV layouts and `ModelOpt` SmoothQuant calibration.
  - `compressed-tensors` (Neural Magic / vLLM / HuggingFace) and `torchao` sub-byte integers & FP6/FP5.

#### Advanced Attention & GEMM/MoE Kernels
- **FlashInfer Engine (`crates/oxide-models/src/attention_kernels.rs`)**: Paged KV cache attention with ragged batching and variable sequence lengths.
- **FlashMLA Engine**: DeepSeek-V2 / DeepSeek-V3 absorbed latent projection Multi-Head Latent Attention.
- **MoE & Grouped GEMM (`crates/oxide-engine/src/moe_gemm.rs`)**: CuTeLayout multi-dimensional stride descriptors, fused top-K gating, and expert routing.

#### Speculative Decoding & Advanced Decoding Algorithms
- **Multi-Algorithm Speculative Decoding (`crates/oxide-engine/src/speculative.rs`)**:
  - N-gram prompt lookup, Suffix matching, EAGLE tree drafting, and DFlash diffusion speculation blocks.
- **High-Throughput Decoding Engines (`crates/oxide-engine/src/decoding.rs`)**:
  - Beam search with length normalization penalty ($\alpha = 0.6$).
  - Parallel sampling (Best-of-N) with independent trajectory generation.

#### Edge Computer Vision Engine (`crates/oxide-models/src/vision.rs`)
- **Multi-Scale Edge Backbones**: Real-time inference support for MobileNetV4, YOLO-World open-vocabulary object detector, RT-DETR, FastSAM, SigLIP Vision Transformer, ViT (Tiny/Small/Base), and Depth Anything.
- **Zero-Copy Image Preprocessing**: Bilinear/bicubic resizing, per-channel normalization, and fast planar RGB to CHW tensor conversion.
- **Bounding Box Regression & NMS**: Branchless Non-Maximum Suppression (NMS) with fast Intersection-over-Union (IoU) calculation.
- **Vision-Language Feature Extraction**: High-throughput patch token extraction `[num_patches, patch_dim]` aligned with `MultiModalVisionProjector` for VLM/VLA models.
- **Vision Server API**: Endpoint `POST /v1/vision/detect` serving object detections and segmentation coordinates with sub-2ms edge latency.

#### Distributed Inference & Serving Protocols
- **Distributed 5D Parallelism Mesh (`crates/oxide-engine/src/distributed.rs`)**: Tensor (TP), Pipeline (PP), Data (DP), Expert (EP), and Context (CP / Ring Attention) parallelism coordinate routing.
- **Dynamic Multi-LoRA Manager (`crates/oxide-engine/src/multi_lora.rs`)**: Dynamic adapter hot-routing for dense projections and sparse MoE experts.
- **Serving Protocols (`crates/oxide-server/`)**:
  - Anthropic Messages API parity (`POST /v1/messages`) with SSE streaming events (`message_start`, `content_block_start`, `content_block_delta`, `message_delta`, `message_stop`).
  - High-throughput streaming gRPC protocol contracts (`crates/oxide-server/src/grpc.rs`).
  - Structured output generation engine (`crates/oxide-server/src/grammar_engine.rs`) for JSON Schemas, regex, and EBNF grammars.
  - DeepSeek-R1 / QwQ `<think>` reasoning extraction and tool calling parser (`crates/oxide-server/src/reasoning_tools.rs`).
- **Hugging Face Model Architectures (`crates/oxide-models/src/hf_architectures.rs`)**:
  - Decoder-only (Llama, Qwen, Gemma), MoE (Mixtral, DeepSeek-V3), Hybrid SSM (Mamba, Qwen3.5), Multi-modal (LLaVA, Qwen-VL, Pixtral), Embedding/Retrieval (E5, GTE, ColBERT MaxSim), and Reward/PRM classifiers.

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
