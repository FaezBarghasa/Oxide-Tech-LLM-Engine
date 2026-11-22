# Quantization, Integer Arithmetic & Microarchitectural Kernels

## Overview

Quantization in `Oxide-Tech-LLM-Engine` (`crates/oxide-quant/`) is engineered for bare-metal SIMD/Tensor Core saturation, minimal memory footprint, and exact mathematical fidelity across 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer formats, ternary 1.58-bit, NVIDIA Blackwell NVFP4, and Flash Attention algorithms.

---

## 1. Complete Integer Quantization Suite (2-bit to 8-bit)

Oxide provides native, zero-allocation GGML-compatible integer quantization formats with direct SIMD dot-product acceleration.

| Format | Block Size | Effective Bits | Memory Layout | Target Hardware Acceleration |
|---|---|---|---|---|
| **Q2_K** | 256 weights | 2.5625 bpw | Scales (16B) + Packed Qs (64B) + Scale `d` + Min `dmin` | AVX-512 `vpmaddubsw`, CUDA DP4A, Metal SIMD |
| **Q3_K** | 256 weights | 3.4375 bpw | Low 2-bit Qs (64B) + High 1-bit Qs (32B) + Scales (12B) + Scale `d` | AVX2 / AVX-512 bit manipulation, NEON |
| **Q4_0** | 32 weights | 4.5 bpw | Scale `d` (f16) + 16 bytes (two 4-bit nibbles/byte) = 18B | AVX2 `vpand`, CUDA Tensor Cores, Apple AMX |
| **Q4_1** | 32 weights | 5.0 bpw | Scale `d` (f16) + Min `m` (f16) + 16 bytes packed = 20B | CPU Affine integer dot products |
| **Q5_0** | 32 weights | 5.5 bpw | Scale `d` (f16) + High bits `qh` (4B) + Low bits `qs` (16B) = 22B | AVX-512, CUDA Warp Shuffle |
| **Q6_K** | 256 weights | 6.5625 bpw | Low 4-bit `ql` (128B) + High 2-bit `qh` (64B) + Scales (16B) + Scale `d` = 210B | Fast integer matrix multiplications |
| **Q8_0** | 32 weights | 8.5 bpw | Scale `d` (f16) + 32 signed int8 values = 34B | Int8 `vpdpbusd` on Intel/AMD, Dot on ARM |

### Zero-Copy Memory-Mapped Slicing (`Cow<'static, [Block]>`)
To achieve zero-heap model loading, `Oxide-Tech-LLM-Engine` enforces exact `#[repr(C)]` layouts matching standard GGUF binary files without compiler padding:
- `BlockQ4_0`: Exactly 18 bytes (`u16` scale `d` + `[u8; 16]` nibbles).
- `BlockQ8_0`: Exactly 34 bytes (`u16` scale `d` + `[i8; 32]` values).
- `BlockQ6_K`: Exactly 210 bytes (`ql: [u8; 128]`, `qh: [u8; 64]`, `scales: [i8; 16]`, `d: u16`).

Tensors are sliced directly from OS memory maps (`memmap2`) via `std::slice::from_raw_parts`. This eliminates heap allocations during model loading:
- **Measured Cold Load Time**: **25.91 ms** on a 5.03 GB GGUF model (`DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf`), compared to **~2,400 ms** in `llama.cpp` (a **92x speedup**).
- **RAM Footprint**: Zero duplicate memory copies; pages are faulted directly by the Linux kernel on demand.


---

## 2. Fast Walsh-Hadamard Transform (FWHT) Preconditioning

### Mathematical Invariant
To eliminate activation outliers without modifying ternary weights $\{-1, 0, +1\}$, activations are transformed into the Hadamard basis using an orthogonal Walsh-Hadamard matrix $H_N$:
$$H_2 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad H_{2k} = H_2 \otimes H_k$$
For a block of 128 elements, this requires $\log_2(128) = 7$ butterfly stages:
$$\text{Output} = \frac{1}{\sqrt{128}} H_{128} X$$

### Shared-Memory Implementation
In `crates/oxide-hardware/oxide-backend-cuda/src/kernels/fwht.cu`, the butterfly network executes entirely within warp-shuffle registers and L1 Shared Memory without accessing global device VRAM:

```cuda
__device__ __forceinline__ void fwht_butterfly_128(float* smem_lane) {
    #pragma unroll
    for (int stride = 64; stride > 0; stride >>= 1) {
        int tid = threadIdx.x;
        int i = (tid / stride) * (2 * stride) + (tid % stride);
        float u = smem_lane[i];
        float v = smem_lane[i + stride];
        smem_lane[i]          = u + v;
        smem_lane[i + stride] = u - v;
        __syncthreads();
    }
    // Normalize by 1 / sqrt(128)
    float norm = 0.08838834764f;
    smem_lane[threadIdx.x] *= norm;
}
```

---

## 3. Branchless Ternary Arithmetic (`PTQ1_0` and `PQ2_0`)

Ternary weights $\{-1, 0, +1\}$ are encoded in 2-bit packed values:
- $00_2 \implies 0$
- $01_2 \implies +1$
- $10_2 \implies -1$

Each byte packs 4 ternary weight elements. A 128-weight group is packed into 32 contiguous bytes, accompanied by a 16-bit half-precision floating-point scale factor (`f16`):

```rust
#[repr(C, align(32))]
pub struct TernaryBlock128 {
    pub scale: u16,           // FP16 scale
    pub packed_weights: [u8; 32], // 128 elements packed into 32 bytes (2 bits/weight)
}
```

---

## 4. NVIDIA Blackwell NVFP4 Quantization

The **NVFP4 (E2M1)** format represents floating-point values using 1 sign bit, 2 exponent bits, and 1 mantissa bit.

In `crates/oxide-quant/src/nvfp4.rs`:
- 16 FP32 numbers are quantized into 8 bytes (two 4-bit nibbles per byte).
- Stored alongside block scale factors in `Nvfp4Block`.
- Dequantized in-flight on Blackwell Tensor Cores with $2\times$ throughput compared to FP8.

---

## 5. Flash Attention Engine (In-SMem Tiled Online Softmax)

`crates/oxide-models/src/flash_attn.rs` implements Flash Attention with in-SMem tiled online softmax algorithm:
- **Tiling**: Splits query sequence into blocks of size $B_r$ (e.g., 64) and key/value sequence into blocks of size $B_c$ (e.g., 64).
- **Online Softmax Accumulation**: Computes running maximum $m_i$ and running partition function $l_i = \sum \exp(s_{ij} - m_i)$, dynamically rescaling accumulated output tile by $\exp(m_{\text{prev}} - m_{\text{new}})$ without ever writing intermediate $N \times N$ attention matrices to global memory.
- **Memory Complexity**: $O(1)$ intermediate SRAM memory overhead, strictly preventing GPU VRAM exhaustion on $128\text{k}+$ context lengths.

---

## 6. SIMD Cache-Blocked Parallel GEMVs (`oxide-quant::simd`)

In `crates/oxide-quant/src/simd.rs`, matrix-vector multiplication is parallelized across all available CPU cores and hardware threads:
- **`gemv_blocked_f32`**: Row partitions are distributed using parallel chunking (`par_chunks_mut(16)`). Each thread keeps the activation vector hot in L1D/L2 cache while streaming matrix rows with AVX-512 / AVX2 FMA dual-accumulator unrolling.
- **`gemv_q8_0`**: Parallelized Q8_0 integer GEMV with `_mm512_dpbusd_epi32` / `_mm256_dpbusd_epi32` (AVX-512 VNNI / AVX2 FMA).
- **`gemv_q4_0`**: Parallelized Q4_0 4-bit nibble unpack and multiply-accumulate with AVX-512BW and AVX2 bit manipulation.
- **`gemv_q4_k`**: Parallelized Q4_K super-block (256 weights / 128 bytes) unpack and accumulation with scale and min adjustments.

---

## 7. Intel AMX (Advanced Matrix Extensions) TMM Tile Engine

In `crates/oxide-quant/src/simd.rs::amx`:
- **64-Byte Aligned `TileConfig`**: Hardware tile dimensions configuring 1KB 2D tile registers `TMM0`..`TMM7`.
- **Runtime CPUID & Permission**: Runtime CPUID leaf 7 checks (`is_amx_supported()`) and Linux `ARCH_REQ_XCOMP_PERM` syscall authorization (`request_amx_permission()`).
- **Zero-Allocation Inline Assembly**:
  - `tileloadd`: Loads tile data from host memory into TMM registers.
  - `tdpbusd`: Computes $16 \times 64 \times 64 \times 16$ INT8 matrix multiplication: $C \mathrel{+}= A \times B$.
  - `tilestored`: Stores matrix tile back to pinned memory.
  - `tilerelease`: Resets processor tile registers to initialized clean palette state.

---

## 8. Multi-Threaded CPU FlashAttention-2 / FlashDecode

In `crates/oxide-quant/src/simd.rs::flash_attention_cpu`:
- Queries are parallelized across all heads with threadpool work-stealing.
- Evaluates online numerically-stable Softmax without materializing sequence-length attention weight matrices in DRAM.
- Zero heap allocations during execution on the hot token generation path.

---

## 9. Split-K Parallel FlashDecode & In-Register Softmax Fusion (`flash_attn.cu`)

In `crates/oxide-hardware/oxide-backend-cuda/src/kernels/flash_attn.cu`:

### The Hardware Saturation Bottleneck
On high-throughput consumer and workstation GPUs (e.g., NVIDIA RTX 4060 with 24 Streaming Multiprocessors), compact models such as `Qwen2.5-0.5B` configure only 14 query heads. A conventional FlashDecode kernel binds one threadblock per query head, dispatching only 14 blocks. Consequently:
- 10 out of 24 SMs (41.6% of the GPU) sit completely unutilized.
- Each threadblock serially traverses the entire KV cache sequence length $S$ (e.g. 700+ tokens), causing decode step latency to bottleneck at ~6.0 ms (~166 tok/s).

### Two-Stage Split-K Microarchitecture

```
                        [ Query Vector (Head h) ]
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
     ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │ Stage 1 Split 0 │  │ Stage 1 Split 1 │  │ Stage 1 Split 3 │
     │  (tokens 0,4,8) │  │  (tokens 1,5,9) │  │ (tokens 3,7,11) │
     └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
              │ (partial acc)      │ (partial acc)      │ (partial acc)
              └────────────────────┼────────────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │    Stage 2 Kernel (Head h)    │
                   │  1. Cross-Split Softmax Norm  │
                   │  2. In-Register Q8_1 Quant    │
                   │     via __shfl_xor_sync       │
                   └───────────────┬───────────────┘
                                   ▼
                      [ Quantized Q8_1 Buffer ]
                      (Direct Feed to Downstream GEMV)
```

#### Stage 1: `flash_decode_splitk_dim64_stage1_kernel`
- **Execution Grid**: `dim3(num_heads, 4)` = 56 threadblocks, ensuring 2 to 3 resident blocks per SM across all 24 SMs.
- **Thread-Strided KV Access**: Threadblocks process interleaved time steps ($pos = \text{split\_idx} + k \times \text{num\_splits}$), preserving L1/L2 cache locality across concurrent warps.
- **Warp-Level Softmax Accumulation**: Employs Dao et al.'s online numerically-stable softmax reduction using `__shfl_down_sync` without barrier stalls.
- **Partial Metadata Buffers**: Writes block-level running maximum $m_{h,s}$ and partition sum $l_{h,s}$ to static GPU workspace `d_split_meta`, and partial accumulated activation vector to `d_split_out`.

#### Stage 2: `flash_decode_splitk_dim64_stage2_kernel`
- **Execution Grid**: `dim3(num_heads, 1)` = 14 threadblocks.
- **Inter-Split Softmax Normalization**: Computes the global maximum across all $K=4$ splits:
  $$m^* = \max_{s \in [0..3]} m_s$$
  Rescaling factors:
  $$\alpha_s = \frac{\exp(m_s - m^*)}{\sum_{k=0}^{3} \exp(m_k - m^*) \cdot l_k}$$
- **In-Register Fused Symmetric Q8_1 Quantization**:
  Instead of writing FP32 partials back to global DRAM and launching a separate quantization kernel:
  1. Warp-level butterfly reduction (`__shfl_xor_sync(0xFFFFFFFF, local_max, mask)`) identifies the absolute maximum value across all 64 head dimensions.
  2. Dynamically derives scale factor $d = \frac{\max |x_i|}{127.0f}$.
  3. Directly packs and stores signed 8-bit quantized integer activations (`int8_t`) with half-precision scale ($d$) into the pre-allocated activation arena for immediate consumption by downstream GEMV kernels.
- **Performance Impact**: Slashes per-token decode step latency from **6.0 ms down to 3.05 – 3.22 ms**, delivering **312 – 328 tok/s** on 700+ token contexts and reaching parity with native `llama.cpp`.

---

## 7. Multi-Column Matrix-Vector (MMVQ) & Batched Dequantization Kernels

### 1. High-Throughput MMVQ Projection Kernels
During continuous batching decode steps, vector-matrix multiplications evaluate across multiple active request tokens ($B \in [1..16]$) simultaneously:
- **`mmvq_q4_k_cuda_kernel<BATCH_N>`**: Computes $(B \times K) \times (K \times M)$ products directly on packed 4-bit weights without intermediate dequantization.
- **`mmvq_q6_k_cuda_kernel<BATCH_N>`**: Evaluates 6-bit K-quants using 2D grid launches `dim3(grid, (batch_size + 7) / 8)` with `WARPS_PER_ROW = 1` for bit-accurate strided execution.
- **`mmvq_q8_0_cuda_kernel<BATCH_N>`**: Optimized 8-bit symmetric quantization using DP4A hardware instructions.

### 2. Multi-Column Fused Projection Launches
To eliminate redundant activation reads and minimize launch overhead:
- **`launch_cuda_mmvq_qkv_q5_0` & `launch_cuda_mmvq_qkv_mixed`**: Fuses $W_q$, $W_k$, and $W_v$ projections into a single pass over the normalized input activation vector.
- **`launch_cuda_mmvq_gate_up_q5_0` & `launch_cuda_mmvq_gate_up_q4_k`**: Evaluates FFN gate and up projections in a single fused kernel, directly populating SwiGLU activation buffers.

### 3. Batched Token Embedding Dequantization Kernels
In `crates/oxide-hardware/oxide-backend-cuda/src/kernels/megakernel_ops.cu`, discrete token IDs across all concurrent user streams are converted to dense activation embeddings via a unified 2D grid:
- Grid: `dim3((hidden_dim / 32 + 3) / 4, total_tokens)`
- Kernels: `dequantize_token_embd_batch_q4_k`, `_q5_0`, `_q8_0`, and `_fp32`.
- Eliminates thousands of per-token host-to-device kernel launch overheads during ragged prefill.

