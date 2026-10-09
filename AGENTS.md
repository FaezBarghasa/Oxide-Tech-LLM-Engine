# STRICT PRODUCTION INVARIANTS — ZERO SIMULATION / ZERO STUBS

## 1. Absolute Real-World Requirements
- **Zero Mock / Synthetic Kernels**: No synthetic hash-based activations, mock token loops, or 128-element dummy buffers in `CudaBackend`, `CpuBackend`, or any execution pipeline.
- **Zero Unimplemented / Silent Stubs**: Every hardware target exposed in the engine must either execute real physical hardware instructions or cleanly return a typed error indicating hardware absence.
- **Zero Fake Benchmarks**: Benchmarks must load genuine model weights, allocate real tensors on physical hardware, compute actual matrix-vector products, and measure genuine end-to-end wall-clock latency.
- **End-to-End Real Tests**: Integration tests must execute forward decode passes using genuine model files from `~/models/` (e.g. `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf`, `gemma-4-e2b-it.Q8_0.gguf`) verifying valid output token IDs and coherent text generation.

## 2. Architecture Directives
- **Zero-Copy Memory-Mapped Slicing**: Slices directly borrow from OS memory maps (`&'a [BlockQ4_K]`, `&'a [BlockQ8_0]`) without allocating intermediate heap vectors for model weights.
- **Rayon-Accelerated Multithreaded GEMV**: Block-tiled parallel loops utilizing Zen 4 AVX2 / AVX-512 FMA vector registers for real CPU inference throughput.
- **Complete Physical VRAM Allocations on CUDA**: Real device pointers for all 36 transformer layers allocated via `cudaMalloc` on the NVIDIA RTX 4060 dGPU, dispatching genuine FP16/INT4 Tensor Core kernels.
- **Lock-Free Continuous Batching**: Lock-free SPSC channels and static slot KV-cache blocks replacing coarse asynchronous mutex locks.
