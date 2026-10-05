# Serving, HTTP/2 API & OpenAI Parity Guide

## Overview

`Oxide-Tech-LLM-Engine` includes a high-performance serving architecture (`crates/oxide-server/`) built on `tokio` and `axum`, with native full parity for OpenAI HTTP endpoints, continuous batching slot management, Server-Sent Events (SSE) streaming, and in-kernel DFA schema grammar masking.

---

## 1. CLI Serving Options (`oxide-engine` / `oxide`)

```bash
oxide-engine [OPTIONS]
```

### Key CLI Flags
- `--model <MODEL>`: Target model identifier (`bonsai2`, `needle3`, `llama3`, `deepseek-r1`, `diffusion`, `audio-tts`, `audio-asr`, `kronos`, `jev`, `laya`, `clef`). Default: `bonsai2`.
- `--backend <BACKEND>`: Hardware backend (`cuda`, `rocm`, `tpu`, `intel`, `metal`, `snapdragon`, `rknn`, `hailo`, `cpu`). Default: `cuda`.
- `--gpu <PROFILE>`: Target hardware profile string (e.g., `"RTX 4090"`, `"H100"`, `"Arc B580"`, `"Apple M4 Max"`, `"Orange Pi 6 Plus"`, `"RPi5 with AI HAT+ 2"`).
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

### E. Health & Telemetry (`GET /health`)

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
