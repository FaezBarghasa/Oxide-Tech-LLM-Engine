# Serving, HTTP/2 API & Deployment Guide

## Overview

`Oxide-Tech-LLM-Engine` includes a high-performance HTTP/2 serving architecture (`crates/oxide-server/`) built on `tokio` and `axum`, with native support for Server-Sent Events (SSE) streaming and in-kernel DFA schema grammar masking.

---

## 1. CLI Serving Options (`oxide-engine` / `oxide`)

```bash
oxide-engine [OPTIONS]
```

### Key CLI Flags
- `--model <MODEL>`: Target model (`bonsai2`, `needle3`, `llama3`, `diffusion`, `audio-tts`, `audio-asr`, `kronos`). Default: `bonsai2`.
- `--backend <BACKEND>`: Hardware backend (`cuda`, `rocm`, `tpu`, `intel`, `metal`, `snapdragon`, `rknn`, `hailo`, `cpu`). Default: `cuda`.
- `--gpu <PROFILE>`: Target hardware profile string (e.g., `"RTX 4090"`, `"H100"`, `"Arc B580"`, `"Apple M4 Max"`, `"Orange Pi 6 Plus"`, `"RPi5 with AI HAT+ 2"`).
- `--serve <ADDR>`: Socket address to bind the HTTP/2 server. Default: `127.0.0.1:8080`.
- `--max-slots <N>`: Maximum concurrent batch decoding sequences. Default: `64`.
- `--kv-device-blocks <N>`: Tier 1 Device VRAM KV cache capacity in blocks ($B=16$). Default: `1024`.
- `--kv-host-blocks <N>`: Tier 2 Host Pinned RAM KV cache capacity in blocks. Default: `8192`.
- `--kv-storage-blocks <N>`: Tier 3 External NVMe Storage KV cache capacity in blocks. Default: `32768`.

---

## 2. API Endpoints

### A. Chat Completions (`POST /v1/chat/completions`)

Standard OpenAI-compatible chat completion endpoint supporting streaming responses via Server-Sent Events (SSE).

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
  "max_tokens": 512,
  "stream": true
}
```

#### Streaming SSE Response
```text
data: {"id":"chatcmpl-oxide-1001","object":"chat.completion.chunk","created":1759665600,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"pub "},"finish_reason":null}]}

data: {"id":"chatcmpl-oxide-1001","object":"chat.completion.chunk","created":1759665600,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"struct "},"finish_reason":null}]}

data: {"id":"chatcmpl-oxide-1001","object":"chat.completion.chunk","created":1759665600,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"RingBuffer "},"finish_reason":null}]}

data: [DONE]
```

---

### B. Health & Telemetry (`GET /health`)

#### Response Body
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

## 3. In-Kernel DFA Schema Enforcement

To enforce guaranteed schema validity for structured JSON output or tool calls, pass a JSON schema in the request. The server compiles the schema into an offline DFA transition matrix, preventing the model from sampling invalid syntax bytes on the GPU.

---

## 4. Production Deployment & Benchmarking

### Benchmarking Throughput
```bash
# Benchmark batch throughput with 64 concurrent streams
oxide --model bonsai2 --backend cuda --gpu "RTX 4090" --max-slots 64
```

### Verification & Health Probes
```bash
# Check server readiness
curl -f http://127.0.0.1:8080/health
```
