# Quantization & Microarchitectural Kernels

## Overview

Quantization in `Oxide-Tech-LLM-Engine` is engineered for bare-metal integer instruction saturation, minimal memory footprint, and exact mathematical fidelity. Oxide supports non-standard and emerging quantization formats including **Ternary 1.58-bit (`PTQ1_0`, `PQ2_0`)**, **Monarch block-diagonal factors (`CQ2`)**, and **NVIDIA Blackwell NVFP4**.

---

## 1. Fast Walsh-Hadamard Transform (FWHT) Preconditioning

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

## 2. Branchless Ternary Arithmetic (`PTQ1_0` and `PQ2_0`)

### Bit Layout
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

### Branchless Unpack & Inner Product
The inner product eliminates warp divergence using bit-shifting arithmetic:

```rust
#[inline(always)]
pub fn unpack_and_dot_group(block: &TernaryBlock128, activations: &[f32; 128]) -> f32 {
    let scale = f16_to_f32(block.scale);
    let mut sum: f32 = 0.0;

    for byte_idx in 0..32 {
        let b = block.packed_weights[byte_idx];
        for bit_offset in 0..4 {
            let code = (b >> (bit_offset * 2)) & 0b11;
            let act_idx = byte_idx * 4 + bit_offset;
            let act = activations[act_idx];

            let sign = match code {
                0b01 => 1.0,
                0b10 => -1.0,
                _ => 0.0,
            };
            sum += act * sign;
        }
    }

    sum * scale
}
```

On CUDA hardware, this unpack loop maps to `__dp4a` instructions; on AVX-512 CPU hardware, it lowers to `vpdpbusd`.

---

## 3. NVIDIA Blackwell NVFP4 Quantization

The **NVFP4 (E2M1)** format represents floating-point values using 1 sign bit, 2 exponent bits, and 1 mantissa bit.

In `crates/oxide-quant/src/nvfp4.rs`:
- 16 FP32 numbers are quantized into 8 bytes (two 4-bit nibbles per byte).
- Stored alongside block scale factors in `Nvfp4Block`.
- Dequantized in-flight on Blackwell Tensor Cores with $2\times$ throughput compared to FP8.

---

## 4. Mathematical Tolerance & Verification Assertions

Every quantized kernel must satisfy strict mathematical tolerance thresholds against golden FP32/BF16 reference tensors before integration:

$$\max_i |L_{\text{rust}, i} - L_{\text{ref}, i}| \le \epsilon$$
$$\text{CosineSimilarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} \ge 1 - \delta$$

- **FP16/BF16 Layers**: $\epsilon \le 1.5 \times 10^{-3}$, $\delta \le 1.0 \times 10^{-5}$
- **Ternary PTQ1_0 / CQ2 Layers**: $\epsilon \le 5.0 \times 10^{-2}$, $\delta \le 1.0 \times 10^{-3}$
