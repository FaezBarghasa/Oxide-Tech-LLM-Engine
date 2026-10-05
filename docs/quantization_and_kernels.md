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
| **Q4_0** | 32 weights | 4.5 bpw | Scale `d` (f16) + 16 bytes (two 4-bit nibbles/byte) | AVX2 `vpand`, CUDA Tensor Cores, Apple AMX |
| **Q4_1** | 32 weights | 5.0 bpw | Scale `d` (f16) + Min `m` (f16) + 16 bytes packed | CPU Affine integer dot products |
| **Q5_0** | 32 weights | 5.5 bpw | Scale `d` (f16) + High bits `qh` (4B) + Low bits `qs` (16B) | AVX-512, CUDA Warp Shuffle |
| **Q6_K** | 256 weights | 6.5625 bpw | Low 4-bit `ql` (128B) + High 2-bit `qh` (64B) + Scales (16B) + Scale `d` | Fast integer matrix multiplications |
| **Q8_0** | 32 weights | 8.5 bpw | Scale `d` (f16) + 32 signed int8 values | Int8 `vpdpbusd` on Intel/AMD, Dot on ARM |

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
