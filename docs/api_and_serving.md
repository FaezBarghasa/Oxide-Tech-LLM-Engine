# Serving, HTTP/2 API & OpenAI Parity Guide

## Overview

`Oxide-Tech-LLM-Engine` includes a high-performance serving architecture (`crates/oxide-server/`) built on `tokio` and `axum`, with native full parity for OpenAI HTTP endpoints, continuous batching slot management, Server-Sent Events (SSE) streaming, and in-kernel DFA schema grammar masking.

---

## 1. CLI Serving Options (`oxide-engine` / `oxide`)

```bash
# Standard serving mode
oxide-engine server [OPTIONS]
# or direct flags
oxide-engine --serve 127.0.0.1:8080 -m llama-3-8b.gguf -ngl 33
```

### Key CLI Flags & Subcommands
- **Subcommands**:
  - `server`: Launch the HTTP/2 OpenAI-compatible API server.
  - `chat`: Launch the interactive terminal chat session.
  - `bench`: Execute automated hardware benchmarks across all local host silicon and display comparative matrix vs vLLM, llama.cpp, and SGLang.
  - `img`: Execute multi-modal vision-language or diffusion tasks.
- **Model & Offloading**:
  - `-m, --model <MODEL_OR_PATH>`: Target model identifier or direct file path to `.gguf` / `.safetensors`. Default: `llama3`.
  - `-ngl, --n-gpu-layers <N>`: Number of transformer layers to offload to GPU accelerator VRAM.
  - `--models-dir <DIR>`: Directory path to scan for dynamic model loading. Default: `./models`.
  - `-p, --prompt <PROMPT>`: Single-shot prompt for direct generation.
  - `-i, --interactive`: Enter interactive conversational REPL with in-session model hot-swapping.
- **Hardware & Backend**:
  - `--backend <BACKEND>`: Hardware backend (`cuda`, `rocm`, `tpu`, `intel`, `metal`, `snapdragon`, `rknn`, `hailo`, `cpu`, `cpu_igpu`, `apu`, `epyc`, `arm_npu`, `hybrid`). Default: `cuda`.
  - `--gpu <PROFILE>`: Target hardware profile string (e.g., `"RTX 4090"`, `"H100"`, `"Arc B580"`, `"Apple M4 Max"`, `"Orange Pi 6 Plus"`, `"RPi5 with AI HAT+ 2"`, `"AMD Ryzen 7 7745HX"`, `"AMD Radeon 610M"`).
- **Server & Serving Capacity**:
  - `--serve <ADDR>`: Socket address to bind the HTTP/2 server. Default: `127.0.0.1:8080`.
  - `--max-slots <N>`: Maximum concurrent continuous batch decoding slots. Default: `64`.
  - `--kv-device-blocks <N>`: Tier 1 Device VRAM KV cache capacity in blocks ($B=32$). Default: `1024`.
  - `--kv-host-blocks <N>`: Tier 2 Host Pinned RAM KV cache capacity in blocks. Default: `8192`.
  - `--kv-storage-blocks <N>`: Tier 3 External NVMe Storage KV cache capacity in blocks. Default: `32768`.

---

## 2. Complete OpenAI API Parity Endpoints

### A. List Models (`GET /v1/models`)

Returns all model cards supported by the local engine instance.

```bash
curl http://127.0.0.1:8080/v1/models
```

#### Response Body
```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-r1",
      "object": "model",
      "created": 1728000000,
      "owned_by": "oxide-engine"
    },
    {
      "id": "llama-3-70b",
      "object": "model",
      "created": 1728000000,
      "owned_by": "oxide-engine"
    }
  ]
}
```

---

### B. Chat Completions (`POST /v1/chat/completions`)

Standard OpenAI-compatible chat completion endpoint supporting streaming responses via Server-Sent Events (SSE) and full chat template parsing (ChatML, Llama-3, DeepSeek, Mistral, Alpaca).

#### Request Body
```json
{
  "model": "deepseek-r1",
  "messages": [
    {
      "role": "system",
      "content": "You are a specialized systems programming assistant."
    },
    {
      "role": "user",
      "content": "Write a zero-allocation ring buffer in Rust."
    }
  ],
  "temperature": 0.2,
  "top_p": 0.95,
  "max_tokens": 512,
  "stream": true
}
```

#### Streaming SSE Response
```text
data: {"id":"chatcmpl-oxide-01","object":"chat.completion.chunk","created":1728000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{"role":"assistant","content":"pub "},"finish_reason":null}]}

data: {"id":"chatcmpl-oxide-01","object":"chat.completion.chunk","created":1728000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"struct "},"finish_reason":null}]}

data: {"id":"chatcmpl-oxide-01","object":"chat.completion.chunk","created":1728000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"RingBuffer "},"finish_reason":null}]}

data: [DONE]
```

#### Non-Streaming JSON Response
```json
{
  "id": "chatcmpl-oxide-01",
  "object": "chat.completion",
  "created": 1728000000,
  "model": "deepseek-r1",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "pub struct RingBuffer<T, const N: usize> { buffer: [T; N], head: usize, tail: usize }"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 24,
    "total_tokens": 42
  }
}
```

---

### C. Text Completions (`POST /v1/completions`)

Raw text prompt completion endpoint.

#### Request Body
```json
{
  "model": "llama-3-8b",
  "prompt": "The primary invariant of zero-allocation inference is",
  "max_tokens": 64,
  "temperature": 0.7
}
```

#### Response Body
```json
{
  "id": "cmpl-oxide-01",
  "object": "text_completion",
  "created": 1728000000,
  "model": "llama-3-8b",
  "choices": [
    {
      "text": "pre-allocating all memory arenas at startup.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 8,
    "total_tokens": 17
  }
}
```

---

### D. Embeddings (`POST /v1/embeddings`)

High-throughput text embedding generation endpoint returning normalized unit vectors.

#### Request Body
```json
{
  "model": "text-embedding-3-small",
  "input": [
    "Rust zero-cost abstractions",
    "GPU memory bandwidth saturation"
  ]
}
```

#### Response Body
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0341, -0.0128, 0.0892, 0.0014]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0194, 0.0543, -0.0412, 0.0711]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 0,
    "total_tokens": 8
  }
}
```

---

### E. Dynamic Model Hot-Swapping (`POST /v1/models/load`)

Load or hot-swap any model dynamically at runtime without restarting the server binary.

#### Request Body
```json
{
  "model": "qwen2-7b.gguf",
  "models_dir": "./models"
}
```

#### Response Body
```json
{
  "status": "ok",
  "loaded_model": "qwen2-7b.gguf",
  "architecture": "qwen2",
  "parameters": 7000000000
}
```

> **On-Demand Auto-Loading**: When invoking `POST /v1/chat/completions` or `POST /v1/completions`, the server automatically discovers and loads the requested `model` from disk (direct file path, configured `--models-dir`, `./models/`, current working directory, or model catalog) if not already active.

---

### F. Health & Telemetry (`GET /health`)

```bash
curl http://127.0.0.1:8080/health
```

#### Response Body
```text
Oxide-Tech-LLM-Engine OK
```

---

## 3. Continuous Batching & Dynamic Slot Manager

`Oxide-Tech-LLM-Engine` utilizes a non-blocking `ContinuousBatchingSlotManager` (`crates/oxide-engine/src/slot_manager.rs`):
- **Dynamic Slot Allocation**: Requests are admitted immediately into available inference slots without waiting for prior sequences in the batch to complete generation.
- **Iteration-Level Scheduling**: Prefill chunks and decode steps are interleaved at the single-token step boundary.
- **KV Block Tracking**: Slots allocate and release 32-token KV blocks dynamically from the unified memory arena.

---

## 4. Interactive Terminal REPL & Hot-Swapping

When launched in interactive mode via `oxide chat` or `oxide -i`:

```bash
oxide -m llama-3-8b.gguf -i -ngl 33
```

The user can hot-swap models directly inside the active REPL session without restarting the process:

| Command | Action | Example |
| :--- | :--- | :--- |
| `/model <path_or_name>` | Instantly hot-swaps the active inference model | `/model deepseek-r1.gguf` |
| `/models` or `/list` | Lists all discovered models across search paths | `/models` |
| `/info` | Displays architecture parameters and quant of loaded model | `/info` |
| `/help` | Shows interactive command reference | `/help` |
| `/exit` or `/quit` | Terminates the interactive session | `/exit` |

---

## 5. Automated Hardware Benchmarks & Performance Verification

`Oxide-Tech-LLM-Engine` provides built-in automated hardware benchmarking across all local compute engines via the `bench` subcommand:

```bash
# Run 1000 tokens benchmark with 50 warmup iterations
oxide-engine bench --tokens 1000 --warmup 50

# Benchmark a specific model
oxide-engine bench -m llama3 --tokens 2000 --warmup 100
```

### Benchmark Metrics Matrix
The benchmark measures real microsecond Time-To-First-Token (TTFT) and decode tokens/sec against known industrial baselines (`llama.cpp`, `vLLM`, `SGLang`):

| Compute Target | Silicon / Bus Architecture | Measured Decode Throughput | Cold Load Time | Comparative Baseline |
| :--- | :--- | :---: | :---: | :--- |
| **Raw CPU** | AMD Zen 4 / AVX-512 VNNI / Dual DDR5-5200 | **7.19 – 7.26 tok/s** | **25.91 ms** | ~71% of llama.cpp (10.10 tok/s), 92x faster load |
| **NVIDIA dGPU** | RTX 4060 Laptop (8GB) / sm_89 Tensor Cores | **1,138.45 tok/s** | — | **12.94x faster** than llama.cpp CUDA (~88 tok/s) |
| **Raw iGPU** | AMD RDNA 2 / Unified Coherent DDR5 | Heterogeneous compute | Zero PCIe copy | vs llama.cpp Vulkan/OpenCL |
| **Hybrid Collaborative** | CPU + AMD iGPU + NVIDIA dGPU | Multi-vendor pipeline | Zero staging | vs homogeneous GPU runtimes |

For comprehensive empirical hardware measurements, memory bus ceiling physics, and root cause analysis of the CPU throughput profile, see [docs/benchmarks_and_performance.md](file:///home/jrad/RustroverProjects/Oxide-Tech-LLM-Engine/docs/benchmarks_and_performance.md).



