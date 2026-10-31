#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_1_Gpu {
  half2 ds;      // ds.x = d (scale = amax / 127.0f), ds.y = sum
  int8_t qs[32]; // 32 quantized signed 8-bit ints
};

/**
 * Custom CUDA Kernel: Dynamic Activation Quantization to Q8_1.
 * Symmetric per-32 group quantization with half2 (scale, sum) header.
 * Consumed by MMVQ INT8 Tensor/ALU kernels with hardware __dp4a.
 */
__global__ void quantize_f32_to_q8_1_cuda_kernel(const float *__restrict__ x,
                                                 BlockQ8_1_Gpu *__restrict__ y,
                                                 int n) {
  int block_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int num_blocks = n / 32;
  if (block_idx >= num_blocks)
    return;
  int lane_id = threadIdx.x % 32;

  float val = x[block_idx * 32 + lane_id];
  float amax = fabsf(val);
  float sum = val;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
  }

  float d = amax / 127.0f;
  float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
  int8_t q = (int8_t)roundf(val * id);
  y[block_idx].qs[lane_id] = q;

  if (lane_id == 0) {
    y[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
  }
}

/**
 * Custom CUDA Kernel: RMSNorm (Root Mean Square Normalization)
 * Standard normalization across hidden dimension (e.g. 4096 / 8192) in
 * Llama/Mistral/Qwen. Performs block-wide warp-shuffle reduction for zero
 * shared-memory overhead.
 */
__global__ void rms_norm_f32_kernel(float *__restrict__ output,
                                    const float *__restrict__ input,
                                    const float *__restrict__ weight,
                                    int hidden_dim, float eps,
                                    float weight_offset) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  const float *in_row = input + bid * hidden_dim;
  float *out_row = output + bid * hidden_dim;

  // Compute sum of squares with unrolled grid-stride loop
  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float val = in_row[i];
    sum_sq += val * val;
  }

  // Warp reduction using shuffle down
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
  }

  // Shared memory for block-level reduction
  __shared__ float s_mean;
  __shared__ float warp_sums[32];

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
    }
    if (lane_id == 0) {
      s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
    }
  }
  __syncthreads();

  float inv_rms = s_mean;
  // Apply scale and weights with offset (e.g. 1.0 for Gemma)
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float w = weight != nullptr ? (weight[i] + weight_offset) : 1.0f;
    out_row[i] = in_row[i] * inv_rms * w;
  }
}

/**
 * Custom CUDA Kernel: Fused RMSNorm + Dynamic Q8_1 Activation Quantization.
 * Computes root-mean-square normalization and directly emits quantized
 * BlockQ8_1_Gpu activations with half2 (scale, sum) headers, eliminating
 * separate quantization kernel launches and avoiding memory round-tripping of
 * float activations.
 */
__global__ void fused_rmsnorm_q8_1_cuda_kernel(
    float *__restrict__ output, BlockQ8_1_Gpu *__restrict__ y_q8_1,
    const float *__restrict__ input, const float *__restrict__ weight,
    int hidden_dim, float eps, float weight_offset) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  const float *in_row = input + bid * hidden_dim;
  float *out_row = output ? (output + bid * hidden_dim) : nullptr;
  BlockQ8_1_Gpu *out_q8_1 =
      y_q8_1 ? (y_q8_1 + bid * (hidden_dim / 32)) : nullptr;

  // 1. Compute sum of squares with grid-stride loop
  float sum_sq = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float val = in_row[i];
    sum_sq += val * val;
  }

  // Warp reduction using shuffle down
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
  }

  // Shared memory for block-level reduction
  __shared__ float s_mean;
  __shared__ float warp_sums[32];

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
    }
    if (lane_id == 0) {
      s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
    }
  }
  __syncthreads();

  float inv_rms = s_mean;

  // 2. Apply scale and weights, and immediately quantize each 32-element warp
  // block
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float w = weight != nullptr ? (weight[i] + weight_offset) : 1.0f;
    float norm_val = in_row[i] * inv_rms * w;
    if (out_row != nullptr) {
      out_row[i] = norm_val;
    }

    if (out_q8_1 != nullptr) {
      int block_32_idx = i / 32;
      float amax = fabsf(norm_val);
      float sum = norm_val;

#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
      }

      float d = amax / 127.0f;
      float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
      int8_t q = (int8_t)roundf(norm_val * id);
      out_q8_1[block_32_idx].qs[lane_id] = q;

      if (lane_id == 0) {
        out_q8_1[block_32_idx].ds =
            make_half2(__float2half(d), __float2half(sum));
      }
    }
  }
}

/**
 * Custom CUDA Kernel: Rotary Position Embedding (RoPE)
 * Applies rotary frequency phase rotation to Query and Key projections.
 */
__global__ void rope_embedding_kernel(float *__restrict__ q,
                                      float *__restrict__ k,
                                      const float *__restrict__ cos_table,
                                      const float *__restrict__ sin_table,
                                      int seq_len, int num_heads,
                                      int head_dim) {
  int token_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int half_dim = head_dim / 2;
  int d = threadIdx.x;

  if (d >= half_dim)
    return;

  int q_base = (token_idx * num_heads + head_idx) * head_dim;
  float cos_val = cos_table[token_idx * half_dim + d];
  float sin_val = sin_table[token_idx * half_dim + d];

  // Query rotation
  float q0 = q[q_base + d];
  float q1 = q[q_base + d + half_dim];
  q[q_base + d] = q0 * cos_val - q1 * sin_val;
  q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;

  // Key rotation (if k pointer provided)
  if (k != nullptr) {
    float k0 = k[q_base + d];
    float k1 = k[q_base + d + half_dim];
    k[q_base + d] = k0 * cos_val - k1 * sin_val;
    k[q_base + d + half_dim] = k0 * sin_val + k1 * cos_val;
  }
}

/**
 * Packed GGUF Quantization Super-Block Structures for physical GPU
 * memory-mapping.
 */
struct __attribute__((packed)) BlockQ8_0_Gpu {
  uint16_t scale;
  int8_t qs[32];
};

struct __attribute__((packed)) BlockQ6_K_Gpu {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d;
};

struct __attribute__((packed)) BlockPTQ1_0_Gpu {
  uint8_t qs[24];
  uint8_t qh[2];
  uint16_t d;
};

struct __attribute__((packed)) BlockQ2_K_Gpu {
  uint8_t scales[16];
  uint8_t qs[64];
  uint16_t d;
  uint16_t dmin;
};

struct __attribute__((packed)) BlockQ3_K_Gpu {
  uint8_t hmask[32];
  uint8_t qs[64];
  uint8_t scales[12];
  uint16_t d;
};

struct __attribute__((packed)) BlockQ4_0_Gpu {
  uint16_t d;
  uint8_t qs[16];
};

struct __attribute__((packed)) BlockQ5_0_Gpu {
  uint16_t d;
  uint8_t qh[4];
  uint8_t qs[16];
};

struct __attribute__((packed)) BlockTQ2_0_Gpu {
  uint8_t qs[64];
  uint16_t d;
};

struct __attribute__((packed)) BlockQ4_1_Gpu {
  uint16_t d;
  uint16_t m;
  uint8_t qs[16];
};

struct __attribute__((packed)) BlockQ5_1_Gpu {
  uint16_t d;
  uint16_t m;
  uint8_t qh[4];
  uint8_t qs[16];
};

struct __attribute__((packed)) BlockIQ4_NL_Gpu {
  uint16_t d;
  uint8_t qs[16];
};

__constant__ float c_kvalues_iq4nl[16] = {
    -31.0f, -22.0f, -15.0f, -10.0f, -6.0f, -3.0f, -1.0f, 0.0f,
    1.0f,   3.0f,   6.0f,   10.0f,  15.0f, 22.0f, 31.0f, 44.0f};

struct __attribute__((packed)) BlockQ5_K_Gpu {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};

static __device__ __forceinline__ int
ptq1_0_trit_gpu(const BlockPTQ1_0_Gpu *blk, int e) {
  uint8_t b;
  int n;
  if (e < 80) {
    b = blk->qs[e & 15];
    n = e >> 4;
  } else if (e < 120) {
    int t = e - 80;
    b = blk->qs[16 + (t & 7)];
    n = t >> 3;
  } else {
    int t = e - 120;
    b = blk->qh[t & 1];
    n = t >> 1;
  }
  uint32_t v = b;
  for (int i = 0; i < n; ++i) {
    v = (v * 3) & 0xFF;
  }
  return (int)((v * 3) >> 8) - 1;
}

/**
 * Custom CUDA Kernel: Quantized Q4_0 Matrix-Vector GEMV (18-byte Block x FP32
 * Activations). Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
__global__ void gemv_q4_0_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  const BlockQ4_0_Gpu *row_blocks =
      (const BlockQ4_0_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 18);

  float warp_dot = 0.0f;
  for (int b = lane_id; b < num_blocks_per_row; b += 32) {
    const BlockQ4_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));

#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      uint8_t byte = blk->qs[i];
      float q0 = (float)((int8_t)(byte & 0x0F) - 8);
      float q1 = (float)((int8_t)((byte >> 4) & 0x0F) - 8);

      float a0 = x[b * 32 + i];
      float a1 = x[b * 32 + i + 16];

      warp_dot += (q0 * a0 + q1 * a1) * d;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q5_0 Matrix-Vector GEMV (22-byte Block x FP32
 * Activations). Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
__global__ void gemv_q5_0_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  const BlockQ5_0_Gpu *row_blocks =
      (const BlockQ5_0_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 22);

  float warp_dot = 0.0f;
  for (int b = lane_id; b < num_blocks_per_row; b += 32) {
    const BlockQ5_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                  ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);

#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      uint8_t byte = blk->qs[i];
      uint8_t h0 = (qh >> i) & 1;
      uint8_t h1 = (qh >> (i + 16)) & 1;
      float q0 = (float)((int8_t)((byte & 0x0F) | (h0 << 4)) - 16);
      float q1 = (float)((int8_t)(((byte >> 4) & 0x0F) | (h1 << 4)) - 16);

      float a0 = x[b * 32 + i];
      float a1 = x[b * 32 + i + 16];

      warp_dot += (q0 * a0 + q1 * a1) * d;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized TQ2_0 (Ternary 2-bit, 2.06 bpw) Matrix-Vector
 * GEMV (66-byte Block x FP32 Activations). Warp-Per-Row architecture: 0 shared
 * memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_tq2_0_cuda_kernel(float *__restrict__ y,
                                       const void *__restrict__ raw_blocks,
                                       const float *__restrict__ x, int m,
                                       int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockTQ2_0_Gpu *row_blocks =
      (const BlockTQ2_0_Gpu *)((const uint8_t *)raw_blocks +
                               (size_t)row * num_blocks_per_row * 66);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockTQ2_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));

#pragma unroll 2
    for (int step = 0; step < 2; step++) {
      int i = lane_id + step * 32;
      uint8_t byte = blk->qs[i];
      float q0 = (float)((int8_t)(byte & 3) - 1);
      float q1 = (float)((int8_t)((byte >> 2) & 3) - 1);
      float q2 = (float)((int8_t)((byte >> 4) & 3) - 1);
      float q3 = (float)((int8_t)((byte >> 6) & 3) - 1);

      float a0 = x[b * 256 + i];
      float a1 = x[b * 256 + i + 64];
      float a2 = x[b * 256 + i + 128];
      float a3 = x[b * 256 + i + 192];

      warp_dot += (q0 * a0 + q1 * a1 + q2 * a2 + q3 * a3) * d;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q4_1 Matrix-Vector GEMV (20-byte Block x FP32
 * Activations). Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
__global__ void gemv_q4_1_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  const BlockQ4_1_Gpu *row_blocks =
      (const BlockQ4_1_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 20);

  float warp_dot = 0.0f;
  for (int b = lane_id; b < num_blocks_per_row; b += 32) {
    const BlockQ4_1_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float min_val = __half2float(*((const half *)&blk->m));

#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      uint8_t byte = blk->qs[i];
      float q0 = (float)(byte & 0x0F);
      float q1 = (float)((byte >> 4) & 0x0F);

      float a0 = x[b * 32 + i];
      float a1 = x[b * 32 + i + 16];

      warp_dot += (q0 * d + min_val) * a0 + (q1 * d + min_val) * a1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q5_1 Matrix-Vector GEMV (24-byte Block x FP32
 * Activations). Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
__global__ void gemv_q5_1_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  const BlockQ5_1_Gpu *row_blocks =
      (const BlockQ5_1_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 24);

  float warp_dot = 0.0f;
  for (int b = lane_id; b < num_blocks_per_row; b += 32) {
    const BlockQ5_1_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float min_val = __half2float(*((const half *)&blk->m));

    uint32_t qh = *((const uint32_t *)blk->qh);

#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      uint8_t byte = blk->qs[i];
      uint8_t h0 = (qh >> i) & 1;
      uint8_t h1 = (qh >> (i + 16)) & 1;

      float q0 = (float)((byte & 0x0F) | (h0 << 4));
      float q1 = (float)(((byte >> 4) & 0x0F) | (h1 << 4));

      float a0 = x[b * 32 + i];
      float a1 = x[b * 32 + i + 16];

      warp_dot += (q0 * d + min_val) * a0 + (q1 * d + min_val) * a1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Non-linear IQ4_NL Matrix-Vector GEMV (18-byte Block x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_iq4_nl_cuda_kernel(float *__restrict__ y,
                                        const void *__restrict__ raw_blocks,
                                        const float *__restrict__ x, int m,
                                        int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  const BlockIQ4_NL_Gpu *row_blocks =
      (const BlockIQ4_NL_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 18);

  float warp_dot = 0.0f;
  for (int b = lane_id; b < num_blocks_per_row; b += 32) {
    const BlockIQ4_NL_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));

#pragma unroll 16
    for (int i = 0; i < 16; i++) {
      uint8_t byte = blk->qs[i];
      uint8_t q0 = byte & 0x0F;
      uint8_t q1 = (byte >> 4) & 0x0F;

      float w0 = d * c_kvalues_iq4nl[q0];
      float w1 = d * c_kvalues_iq4nl[q1];

      float a0 = x[b * 32 + i];
      float a1 = x[b * 32 + i + 16];

      warp_dot += w0 * a0 + w1 * a1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized PTQ1_0 (Ternary 1.58-bit / 1.75-bit)
 * Matrix-Vector GEMV (28-byte block: 128 elements packed in base-3 x FP32
 * Activations). Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
__global__ void gemv_ptq1_0_cuda_kernel(float *__restrict__ y,
                                        const void *__restrict__ raw_blocks,
                                        const float *__restrict__ x, int m,
                                        int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 128;

  const BlockPTQ1_0_Gpu *row_blocks =
      (const BlockPTQ1_0_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 28);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockPTQ1_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float sum_e = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int e = lane_id + i * 32;
      int t = ptq1_0_trit_gpu(blk, e);
      float x_val = x[b * 128 + e];
      sum_e += (float)t * x_val;
    }
    warp_dot += d * sum_e;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q8_0 Multi-Column MMVQ (34-byte Block x Q8_1
 * Activations). Evaluates BATCH_N token vectors simultaneously while reading
 * weights once. Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(),
 * independent warp execution.
 */
template <int BATCH_N, int ADD_TO_Y = 0, int WARPS_PER_ROW = 1>
__global__
__launch_bounds__(256, (BATCH_N > 16 ? 3 : 4)) void mmvq_q8_0_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k,
    int actual_batch = BATCH_N) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);
  if (num_cols <= 0)
    return;

  const BlockQ8_0_Gpu *row_blocks =
      (const BlockQ8_0_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 34);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ8_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->scale));
    int8_t q = blk->qs[lane_id];

#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      const BlockQ8_1_Gpu *bq8_1 =
          &x_q8_1[(col_offset + col) * num_blocks_per_row + b];
      float d8 = __low2float(bq8_1->ds);
      int8_t u = bq8_1->qs[lane_id];
      thread_dot[col] += (d * d8) * (float)((int)q * (int)u);
    }
  }

#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    if (BATCH_N == 1 || col < num_cols) {
      float val = thread_dot[col];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
      }

      if (lane_id == 0) {
        if (ADD_TO_Y) {
          y[(col_offset + col) * m + row] += val;
        } else {
          y[(col_offset + col) * m + row] = val;
        }
      }
    }
  }
}

/**
 * Custom CUDA Kernel: Quantized Q6_K Multi-Column MMVQ (210-byte Super-block x
 * Q8_1 Activations via __dp4a). Evaluates BATCH_N token vectors simultaneously
 * while reading weights once. Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
template <int BATCH_N, int ADD_TO_Y = 0, int WARPS_PER_ROW = 1>
__global__
__launch_bounds__(256, (BATCH_N > 32 ? 1 : (BATCH_N > 16 ? 3 : 4))) void mmvq_q6_k_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k,
    int actual_batch = BATCH_N) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  bool valid_row = (row < m);

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  int num_q8_blocks_per_col = k / 32;

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);

  const BlockQ6_K_Gpu *row_blocks = valid_row
      ? (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 210)
      : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  int group = lane_id / 8;
  int idx = lane_id % 8;

  for (int b = 0; b < num_blocks_per_row; ++b) {
    if (valid_row) {
      bool thread_has_work = false;
#pragma unroll
      for (int col = 0; col < BATCH_N; ++col) {
        if (col < num_cols) {
          const BlockQ8_1_Gpu *bx =
              &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
          float d0 = __low2float(__ldg(&bx[group].ds));
          float d1 = __low2float(__ldg(&bx[4 + group].ds));
          if (d0 != 0.0f || d1 != 0.0f) {
            thread_has_work = true;
          }
        }
      }
      unsigned int warp_active = __ballot_sync(0xFFFFFFFF, thread_has_work);
      if (warp_active == 0) {
        continue;
      }

      const BlockQ6_K_Gpu *blk = &row_blocks[b];
      float d = __half2float(*((const half *)&blk->d));

#pragma unroll
      for (int half_idx = 0; half_idx < 2; half_idx++) {
        const uint16_t *ql16 = (const uint16_t *)&blk->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
        uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
        const uint16_t *qh16 = (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
        uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

        uint32_t q_low4 = (ql4 >> ((group & 2) * 2)) & 0x0F0F0F0Fu;
        uint32_t q_high4 = ((qh4 >> (group * 2)) & 0x03030303u) << 4;
        uint32_t vx = (q_low4 | q_high4) ^ 0x20202020u;
        uint32_t vm = vx & 0x20202020u;
        int v_signed = (int)(vx | (vm << 1) | (vm << 2));

        float d_sc =
            d * (float)blk->scales[half_idx * 8 + group * 2 + (idx >> 2)];
        int sub = half_idx * 4 + group;

#pragma unroll
        for (int col = 0; col < BATCH_N; ++col) {
          if (col < num_cols) {
            const BlockQ8_1_Gpu *bx =
                &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
            const BlockQ8_1_Gpu *bq8 = &bx[sub];
            float d8 = __low2float(__ldg(&bq8->ds));
            if (d8 != 0.0f) {
              int u = __ldg(&((const int *)bq8->qs)[idx]);
              int dot = __dp4a(v_signed, u, 0);
              thread_dot[col] += d_sc * d8 * (float)dot;
            }
          }
        }
      }
    }
  }

  if (valid_row) {
#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      if (BATCH_N == 1 || col < num_cols) {
        float val = thread_dot[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (lane_id == 0) {
          if (ADD_TO_Y) {
            y[(col_offset + col) * m + row] += val;
          } else {
            y[(col_offset + col) * m + row] = val;
          }
        }
      }
    }
  }
}

struct __attribute__((packed)) BlockQ4_K_Gpu {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};

__device__ __forceinline__ void
get_scale_min_k4_cuda(int j, const uint8_t *scales, float *sc, float *m) {
  if (j < 4) {
    *sc = (float)(scales[j] & 63);
    *m = (float)(scales[j + 4] & 63);
  } else {
    *sc = (float)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    *m = (float)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
  }
}

__device__ __forceinline__ void
get_scale_min_k4_from_regs(int chunk, uint32_t sc_0_3, uint32_t sc_4_7,
                           uint32_t sc_8_11, float *sc0, float *min0,
                           float *sc1, float *min1) {
  if (chunk < 2) {
    int shift = chunk * 16;
    uint32_t w0 = sc_0_3 >> shift;
    uint32_t w1 = sc_4_7 >> shift;
    uint8_t s0 = w0 & 0xFF;
    uint8_t s1 = (w0 >> 8) & 0xFF;
    uint8_t m0 = w1 & 0xFF;
    uint8_t m1 = (w1 >> 8) & 0xFF;
    *sc0 = (float)(s0 & 63);
    *min0 = (float)(m0 & 63);
    *sc1 = (float)(s1 & 63);
    *min1 = (float)(m1 & 63);
  } else {
    int shift = (chunk - 2) * 16;
    uint32_t w2 = sc_8_11 >> shift;
    uint32_t w0 = sc_0_3 >> shift;
    uint32_t w1 = sc_4_7 >> shift;
    uint8_t s8 = w2 & 0xFF;
    uint8_t s9 = (w2 >> 8) & 0xFF;
    uint8_t s0 = w0 & 0xFF;
    uint8_t s1 = (w0 >> 8) & 0xFF;
    uint8_t s4 = w1 & 0xFF;
    uint8_t s5 = (w1 >> 8) & 0xFF;
    *sc0 = (float)((s8 & 0x0F) | ((s0 >> 6) << 4));
    *min0 = (float)((s8 >> 4) | ((s4 >> 6) << 4));
    *sc1 = (float)((s9 & 0x0F) | ((s1 >> 6) << 4));
    *min1 = (float)((s9 >> 4) | ((s5 >> 6) << 4));
  }
}

/**
 * Custom CUDA Kernel: Quantized Q4_K Multi-Column MMVQ (INT4 Super-blocks x
 * Q8_1 Activations via __dp4a). Evaluates BATCH_N token vectors simultaneously
 * while reading weights once. Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution. 128-byte coalesced weight loads
 * + hardware INT8 dot products.
 */
template <int BATCH_N, int ADD_TO_Y = 0, int WARPS_PER_ROW = 1>
__global__
__launch_bounds__(256, (BATCH_N > 32 ? 1 : (BATCH_N > 16 ? 3 : 4))) void mmvq_q4_k_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k,
    int actual_batch = BATCH_N) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  bool valid_row = (row < m);

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  int num_q8_blocks_per_col = k / 32;

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);

  const BlockQ4_K_Gpu *row_blocks = valid_row
      ? (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 144)
      : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  for (int b = 0; b < num_blocks_per_row; b++) {
    if (valid_row) {
      bool thread_has_work = false;
#pragma unroll
      for (int col = 0; col < BATCH_N; ++col) {
        if (col < num_cols) {
          const BlockQ8_1_Gpu *bx =
              &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
          float d0 = __low2float(__ldg(&bx[sub0].ds));
          float d1 = __low2float(__ldg(&bx[sub1].ds));
          if (d0 != 0.0f || d1 != 0.0f) {
            thread_has_work = true;
          }
        }
      }
      unsigned int warp_active = __ballot_sync(0xFFFFFFFF, thread_has_work);
      if (warp_active == 0) {
        continue;
      }

      const BlockQ4_K_Gpu *blk = &row_blocks[b];
      uint4 hdr = ((const uint4 *)blk)[0];
      float d = __low2float(*((const half2 *)&hdr.x));
      float dmin = __high2float(*((const half2 *)&hdr.x));

      float sc0, min0, sc1, min1;
      get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                                 &min1);

      float d_sc0 = d * sc0;
      float d_sc1 = d * sc1;
      float dm_min0 = dmin * min0;
      float dm_min1 = dmin * min1;

      int v = ((const int *)blk->qs)[lane_id];
      int v0 = v & 0x0F0F0F0F;
      int v1 = (v >> 4) & 0x0F0F0F0F;

#pragma unroll
      for (int col = 0; col < BATCH_N; ++col) {
        if (col < num_cols) {
          const BlockQ8_1_Gpu *bx =
              &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
          const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
          const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

          float d8_0 = __low2float(__ldg(&bq8_0->ds));
          float d8_1 = __low2float(__ldg(&bq8_1->ds));

          if (d8_0 != 0.0f) {
            int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
            int dot0 = __dp4a(v0, u0, 0);
            int sum0 = __dp4a(0x01010101, u0, 0);
            float s0 = d_sc0 * d8_0;
            float m0 = dm_min0 * d8_0;
            thread_dot[col] += (s0 * (float)dot0 - m0 * (float)sum0);
          }

          if (d8_1 != 0.0f) {
            int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);
            int dot1 = __dp4a(v1, u1, 0);
            int sum1 = __dp4a(0x01010101, u1, 0);
            float s1 = d_sc1 * d8_1;
            float m1 = dm_min1 * d8_1;
            thread_dot[col] += (s1 * (float)dot1 - m1 * (float)sum1);
          }
        }
      }
    }
  }

  if (valid_row) {
#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      if (BATCH_N == 1 || col < num_cols) {
        float val = thread_dot[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (lane_id == 0) {
          if (ADD_TO_Y) {
            y[(col_offset + col) * m + row] += val;
          } else {
            y[(col_offset + col) * m + row] = val;
          }
        }
      }
    }
  }
}

/**
 * Grouped Multi-Column Q4_K Gate + Up MMVQ Kernel.
 * Computes both Gate (w1) and Up (w3) projections simultaneously in a single warp pass,
 * reading activations from shared memory ONCE for both projections.
 */
template <int BATCH_N, int WARPS_PER_ROW = 1>
__global__ __launch_bounds__(
    256,
    (BATCH_N > 32
         ? 1
         : (BATCH_N > 16 ? 3 : 4))) void mmvq_gate_up_q4_k_cuda_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k,
    int actual_batch = BATCH_N) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  bool valid_row = (row < inter_m);

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  int num_q8_blocks_per_col = k / 32;

  const BlockQ4_K_Gpu *row_w1 = valid_row
      ? (const BlockQ4_K_Gpu *)((const uint8_t *)raw_w1 +
                                (size_t)row * num_blocks_per_row * 144)
      : nullptr;
  const BlockQ4_K_Gpu *row_w3 = valid_row
      ? (const BlockQ4_K_Gpu *)((const uint8_t *)raw_w3 +
                                (size_t)row * num_blocks_per_row * 144)
      : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float dot_gate[BATCH_N];
  float dot_up[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    dot_gate[col] = 0.0f;
    dot_up[col] = 0.0f;
  }

  for (int b = 0; b < num_blocks_per_row; b++) {
    if (valid_row) {
      const BlockQ4_K_Gpu *blk1 = &row_w1[b];
      uint4 hdr1 = ((const uint4 *)blk1)[0];
      float d1 = __low2float(*((const half2 *)&hdr1.x));
      float dmin1 = __high2float(*((const half2 *)&hdr1.x));

      float sc0_1, min0_1, sc1_1, min1_1;
      get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_1, &min0_1, &sc1_1,
                                 &min1_1);

      float d_sc0_1 = d1 * sc0_1;
      float d_sc1_1 = d1 * sc1_1;
      float dm_min0_1 = dmin1 * min0_1;
      float dm_min1_1 = dmin1 * min1_1;

      int v1 = ((const int *)blk1->qs)[lane_id];
      int v0_1 = v1 & 0x0F0F0F0F;
      int v1_1 = (v1 >> 4) & 0x0F0F0F0F;

      const BlockQ4_K_Gpu *blk3 = &row_w3[b];
      uint4 hdr3 = ((const uint4 *)blk3)[0];
      float d3 = __low2float(*((const half2 *)&hdr3.x));
      float dmin3 = __high2float(*((const half2 *)&hdr3.x));

      float sc0_3, min0_3, sc1_3, min1_3;
      get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0_3, &min0_3, &sc1_3,
                                 &min1_3);

      float d_sc0_3 = d3 * sc0_3;
      float d_sc1_3 = d3 * sc1_3;
      float dm_min0_3 = dmin3 * min0_3;
      float dm_min1_3 = dmin3 * min1_3;

      int v3 = ((const int *)blk3->qs)[lane_id];
      int v0_3 = v3 & 0x0F0F0F0F;
      int v1_3 = (v3 >> 4) & 0x0F0F0F0F;

#pragma unroll
      for (int col = 0; col < BATCH_N; ++col) {
        if (col < num_cols) {
          const BlockQ8_1_Gpu *bx =
              &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
          const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
          const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

          float d8_0 = __low2float(__ldg(&bq8_0->ds));
          float d8_1 = __low2float(__ldg(&bq8_1->ds));

          int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
          int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

          int sum0 = __dp4a(0x01010101, u0, 0);
          int sum1 = __dp4a(0x01010101, u1, 0);

          int dot0_1 = __dp4a(v0_1, u0, 0);
          int dot1_1 = __dp4a(v1_1, u1, 0);
          float acc0_1 = d_sc0_1 * (float)dot0_1 - dm_min0_1 * (float)sum0;
          float acc1_1 = d_sc1_1 * (float)dot1_1 - dm_min1_1 * (float)sum1;
          dot_gate[col] += acc0_1 * d8_0 + acc1_1 * d8_1;

          int dot0_3 = __dp4a(v0_3, u0, 0);
          int dot1_3 = __dp4a(v1_3, u1, 0);
          float acc0_3 = d_sc0_3 * (float)dot0_3 - dm_min0_3 * (float)sum0;
          float acc1_3 = d_sc1_3 * (float)dot1_3 - dm_min1_3 * (float)sum1;
          dot_up[col] += acc0_3 * d8_0 + acc1_3 * d8_1;
        }
      }
    }
  }

  if (valid_row) {
#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      if (BATCH_N == 1 || col < num_cols) {
        float val_gate = dot_gate[col];
        float val_up = dot_up[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          val_gate += __shfl_down_sync(0xFFFFFFFF, val_gate, offset);
          val_up += __shfl_down_sync(0xFFFFFFFF, val_up, offset);
        }

        if (lane_id == 0) {
          y_gate[(col_offset + col) * inter_m + row] = val_gate;
          y_up[(col_offset + col) * inter_m + row] = val_up;
        }
      }
    }
  }
}

/**
 * Custom Fused Pair Q4_K Gate + Up GEMV Kernel (BATCH_N = 1).
 * Computes both Gate and Up rows simultaneously in a single warp pass,
 * reading activations x_q8_1 exactly ONCE for both projections.
 */
__global__ __launch_bounds__(256, 4) void gemv_gate_up_fused_pair_q4_k_cuda_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k) {
  __shared__ BlockQ8_1_Gpu s_x[384];
  int tid = threadIdx.x;
  int num_q8_blocks = k / 32;
  const uint32_t *x_u32 = (const uint32_t *)x_q8_1_raw;
  uint32_t *s_x_u32 = (uint32_t *)s_x;
  int total_words = num_q8_blocks * 9;
  for (int item = tid; item < total_words; item += blockDim.x) {
    s_x_u32[item] = x_u32[item];
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int rows_per_block = blockDim.x / 32;
  int row = blockIdx.x * rows_per_block + warp_id;
  if (row >= inter_m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_w1 =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_w1 +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ4_K_Gpu *row_w3 =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_w3 +
                              (size_t)row * num_blocks_per_row * 144);

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float dot_gate = 0.0f;
  float dot_up = 0.0f;

  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ8_1_Gpu *bx = &s_x[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    // 1. Gate (w1)
    {
      const BlockQ4_K_Gpu *blk1 = &row_w1[b];
      uint4 hdr1 = ((const uint4 *)blk1)[0];
      float d1 = __low2float(*((const half2 *)&hdr1.x));
      float dmin1 = __high2float(*((const half2 *)&hdr1.x));
      float sc0, min0, sc1, min1;
      get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0, &min0, &sc1, &min1);

      int v = ((const int *)blk1->qs)[lane_id];
      int v0 = v & 0x0F0F0F0F;
      int v1 = (v >> 4) & 0x0F0F0F0F;

      int dot0 = __dp4a(v0, u0, 0);
      int dot1 = __dp4a(v1, u1, 0);

      float acc0 = (d1 * sc0) * (float)dot0 - (dmin1 * min0) * (float)sum0;
      float acc1 = (d1 * sc1) * (float)dot1 - (dmin1 * min1) * (float)sum1;
      dot_gate += acc0 * d8_0 + acc1 * d8_1;
    }

    // 2. Up (w3) using identical loaded activations
    {
      const BlockQ4_K_Gpu *blk3 = &row_w3[b];
      uint4 hdr3 = ((const uint4 *)blk3)[0];
      float d3 = __low2float(*((const half2 *)&hdr3.x));
      float dmin3 = __high2float(*((const half2 *)&hdr3.x));
      float sc0, min0, sc1, min1;
      get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0, &min0, &sc1, &min1);

      int v = ((const int *)blk3->qs)[lane_id];
      int v0 = v & 0x0F0F0F0F;
      int v1 = (v >> 4) & 0x0F0F0F0F;

      int dot0 = __dp4a(v0, u0, 0);
      int dot1 = __dp4a(v1, u1, 0);

      float acc0 = (d3 * sc0) * (float)dot0 - (dmin3 * min0) * (float)sum0;
      float acc1 = (d3 * sc1) * (float)dot1 - (dmin3 * min1) * (float)sum1;
      dot_up += acc0 * d8_0 + acc1 * d8_1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_down_sync(0xFFFFFFFF, dot_gate, offset);
    dot_up += __shfl_down_sync(0xFFFFFFFF, dot_up, offset);
  }

  if (lane_id == 0) {
    y_gate[row] = dot_gate;
    y_up[row] = dot_up;
  }
}

/**
 * Pure Warp Single-Token QKV GEMV Kernel (BATCH_N = 1, 0 Shared Memory, 0 __syncthreads).
 * Computes Q (Q4_K), K (Q4_K), and V (Q4_K when V_QUANT == 1, or Q6_K when V_QUANT == 2)
 * in a single warp-synchronous dispatch, writing V directly to the FP16 KV cache.
 */
template <int V_QUANT>
__global__ __launch_bounds__(256, 4) void gemv_qkv_mixed_pure_warp_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    void *__restrict__ v_cache_fp16, int start_pos,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k, const int *__restrict__ d_start_pos = nullptr) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = blockDim.x / 32;
  int total_row = blockIdx.x * rows_per_block + warp_id;
  int total_m = q_m + 2 * kv_m;
  if (total_row >= total_m)
    return;

  int row = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (total_row < q_m) {
    row = total_row;
    raw_blocks = raw_wq;
    y = y_q;
  } else if (total_row < q_m + kv_m) {
    row = total_row - q_m;
    raw_blocks = raw_wk;
    y = y_k;
  } else {
    row = total_row - q_m - kv_m;
    raw_blocks = raw_wv;
    y = y_v;
    is_v = true;
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot = 0.0f;

  if (!is_v || V_QUANT == 1) {
    const BlockQ4_K_Gpu *row_blocks =
        (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 144);
    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    for (int b = 0; b < num_blocks_per_row; b++) {
      const BlockQ4_K_Gpu *blk = &row_blocks[b];
      uint4 hdr = ((const uint4 *)blk)[0];
      float d = __low2float(*((const half2 *)&hdr.x));
      float dmin = __high2float(*((const half2 *)&hdr.x));

      float sc0, min0, sc1, min1;
      get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                                 &min1);

      float d_sc0 = d * sc0;
      float d_sc1 = d * sc1;
      float dm_min0 = dmin * min0;
      float dm_min1 = dmin * min1;

      int v = ((const int *)blk->qs)[lane_id];
      int v0 = v & 0x0F0F0F0F;
      int v1 = (v >> 4) & 0x0F0F0F0F;

      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
      const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
      const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

      float d8_0 = __low2float(__ldg(&bq8_0->ds));
      float d8_1 = __low2float(__ldg(&bq8_1->ds));

      int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
      int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

      int dot0 = __dp4a(v0, u0, 0);
      int dot1 = __dp4a(v1, u1, 0);
      int sum0 = __dp4a(0x01010101, u0, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);

      float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
      float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

      thread_dot += acc0 * d8_0 + acc1 * d8_1;
    }
  } else {
    const BlockQ6_K_Gpu *row_blocks =
        (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 210);
    int group = lane_id / 8;
    int idx = lane_id % 8;

    for (int b = 0; b < num_blocks_per_row; b++) {
      const BlockQ6_K_Gpu *blk = &row_blocks[b];
      float d = __half2float(*((const half *)&blk->d));
      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];

#pragma unroll
      for (int half_idx = 0; half_idx < 2; half_idx++) {
        const uint16_t *ql16 = (const uint16_t *)&blk->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
        uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
        const uint16_t *qh16 = (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
        uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

        uint32_t q_low4 = (ql4 >> ((group & 2) * 2)) & 0x0F0F0F0Fu;
        uint32_t q_high4 = ((qh4 >> (group * 2)) & 0x03030303u) << 4;
        uint32_t vx = (q_low4 | q_high4) ^ 0x20202020u;
        uint32_t vm = vx & 0x20202020u;
        int v_signed = (int)(vx | (vm << 1) | (vm << 2));

        float d_sc = d * (float)blk->scales[half_idx * 8 + group * 2 + (idx >> 2)];
        int sub = half_idx * 4 + group;

        const BlockQ8_1_Gpu *bq8 = &bx[sub];
        float d8 = __low2float(__ldg(&bq8->ds));
        int u = __ldg(&((const int *)bq8->qs)[idx]);
        int dot = __dp4a(v_signed, u, 0);
        thread_dot += d_sc * d8 * (float)dot;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    if (y != nullptr) {
      y[row] = thread_dot;
    }
    if (is_v && v_cache_fp16 != nullptr) {
      int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
      ((__half *)v_cache_fp16)[(size_t)actual_pos * kv_m + row] =
          __float2half(thread_dot);
    }
  }
}

/**
 * Grouped Multi-Column Mixed-Quant QKV MMVQ Kernel + Direct FP16 V-Cache Write.
 * Supports Q (Q4_K), K (Q4_K), and V (Q4_K when V_QUANT == 1, or Q6_K when
 * V_QUANT == 2) in a single kernel launch, and optionally writes V directly to
 * the FP16 KV cache.
 */
template <int BATCH_N, int V_QUANT, int WARPS_PER_ROW = 1>
__global__
__launch_bounds__(256, (BATCH_N > 32 ? 1 : (BATCH_N > 16 ? 3 : 4))) void mmvq_qkv_mixed_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    void *__restrict__ v_cache_fp16, int start_pos,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k, int actual_batch = BATCH_N,
    const int *__restrict__ d_start_pos = nullptr) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int total_row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  int total_m = q_m + 2 * kv_m;
  bool valid_row = (total_row < total_m);

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);

  int row = 0;
  int out_stride = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (valid_row) {
    if (total_row < q_m) {
      row = total_row;
      out_stride = q_m;
      raw_blocks = raw_wq;
      y = y_q;
    } else if (total_row < q_m + kv_m) {
      row = total_row - q_m;
      out_stride = kv_m;
      raw_blocks = raw_wk;
      y = y_k;
    } else {
      row = total_row - q_m - kv_m;
      out_stride = kv_m;
      raw_blocks = raw_wv;
      y = y_v;
      is_v = true;
    }
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  int num_q8_blocks_per_col = k / 32;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;
  int group = lane_id / 8;
  int idx = lane_id % 8;

  for (int b = 0; b < num_blocks_per_row; b++) {
    if (valid_row) {
      if (!is_v || V_QUANT == 1) {
        const BlockQ4_K_Gpu *blk = &((const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks + (size_t)row * num_blocks_per_row * 144))[b];
        uint4 hdr = ((const uint4 *)blk)[0];
        float d = __low2float(*((const half2 *)&hdr.x));
        float dmin = __high2float(*((const half2 *)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                                   &min1);

        float d_sc0 = d * sc0;
        float d_sc1 = d * sc1;
        float dm_min0 = dmin * min0;
        float dm_min1 = dmin * min1;

        int v = ((const int *)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

#pragma unroll
        for (int col = 0; col < BATCH_N; ++col) {
          if (col < num_cols) {
            const BlockQ8_1_Gpu *bx =
                &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
            const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
            const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

            float d8_0 = __low2float(__ldg(&bq8_0->ds));
            float d8_1 = __low2float(__ldg(&bq8_1->ds));

            int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
            int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

            int dot0 = __dp4a(v0, u0, 0);
            int dot1 = __dp4a(v1, u1, 0);
            int sum0 = __dp4a(0x01010101, u0, 0);
            int sum1 = __dp4a(0x01010101, u1, 0);

            float s0 = d_sc0 * d8_0;
            float m0 = dm_min0 * d8_0;
            float s1 = d_sc1 * d8_1;
            float m1 = dm_min1 * d8_1;

            thread_dot[col] += (s0 * (float)dot0 - m0 * (float)sum0) + (s1 * (float)dot1 - m1 * (float)sum1);
          }
        }
      } else {
        const BlockQ6_K_Gpu *blk = &((const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks + (size_t)row * num_blocks_per_row * 210))[b];
        float d = __half2float(*((const half *)&blk->d));

#pragma unroll
        for (int half_idx = 0; half_idx < 2; half_idx++) {
          const uint16_t *ql16 = (const uint16_t *)&blk->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
          uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
          const uint16_t *qh16 = (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
          uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

          uint32_t q_low4 = (ql4 >> ((group & 2) * 2)) & 0x0F0F0F0Fu;
          uint32_t q_high4 = ((qh4 >> (group * 2)) & 0x03030303u) << 4;
          uint32_t vx = (q_low4 | q_high4) ^ 0x20202020u;
          uint32_t vm = vx & 0x20202020u;
          int v_signed = (int)(vx | (vm << 1) | (vm << 2));

          float d_sc =
              d * (float)blk->scales[half_idx * 8 + group * 2 + (idx >> 2)];
          int sub = half_idx * 4 + group;

#pragma unroll
          for (int col = 0; col < BATCH_N; ++col) {
            if (col < num_cols) {
              const BlockQ8_1_Gpu *bx =
                  &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
              const BlockQ8_1_Gpu *bq8 = &bx[sub];
              float d8 = __low2float(__ldg(&bq8->ds));
              int u = __ldg(&((const int *)bq8->qs)[idx]);
              int dot = __dp4a(v_signed, u, 0);
              thread_dot[col] += d_sc * d8 * (float)dot;
            }
          }
        }
      }
    }
  }

  if (valid_row) {
#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      if (BATCH_N == 1 || col < num_cols) {
        float val = thread_dot[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        if (lane_id == 0) {
          if (y != nullptr) {
            y[(col_offset + col) * out_stride + row] = val;
          }
          if (is_v && v_cache_fp16 != nullptr) {
            int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
            ((__half *)
                 v_cache_fp16)[(size_t)(actual_pos + col_offset + col) * kv_m +
                               row] = __float2half(val);
          }
        }
      }
    }
  }
}

/**
 * Grouped Q4_K Gate + Up GEMV Kernel.
 * Computes both Gate (w1) and Up (w3) projections in a single kernel dispatch.
 * Total rows = 2 * inter_m. Rows [0, inter_m) -> w1 -> y_gate. Rows [inter_m, 2
 * * inter_m) -> w3 -> y_up.
 */
__global__ void gemv_gate_up_q4_k_cuda_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (total_row >= 2 * inter_m)
    return;

  bool is_up = (total_row >= inter_m);
  int row = is_up ? (total_row - inter_m) : total_row;
  const void *raw_blocks = is_up ? raw_w3 : raw_w1;
  float *y = is_up ? y_up : y_gate;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ4_K_Gpu *blk = &row_blocks[b];
    uint4 hdr = ((const uint4 *)blk)[0];
    float d = __low2float(*((const half2 *)&hdr.x));
    float dmin = __high2float(*((const half2 *)&hdr.x));

    float sc0, min0, sc1, min1;
    get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                               &min1);

    float d_sc0 = d * sc0;
    float d_sc1 = d * sc1;
    float dm_min0 = dmin * min0;
    float dm_min1 = dmin * min1;

    int v = ((const int *)blk->qs)[lane_id];
    int v0 = v & 0x0F0F0F0F;
    int v1 = (v >> 4) & 0x0F0F0F0F;

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Grouped Q4_K QKV GEMV Kernel.
 * Computes Q (wq), K (wk), and V (wv) projections in a single kernel dispatch
 * when all are Q4_K. Total rows = q_m + 2 * kv_m. Rows [0, q_m) -> wq -> y_q.
 * Rows [q_m, q_m + kv_m) -> wk -> y_k.
 * Rows [q_m + kv_m, q_m + 2 * kv_m) -> wv -> y_v.
 */
__global__ void gemv_qkv_q4_k_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int total_m = q_m + 2 * kv_m;
  if (total_row >= total_m)
    return;

  int row;
  const void *raw_blocks;
  float *y;

  if (total_row < q_m) {
    row = total_row;
    raw_blocks = raw_wq;
    y = y_q;
  } else if (total_row < q_m + kv_m) {
    row = total_row - q_m;
    raw_blocks = raw_wk;
    y = y_k;
  } else {
    row = total_row - q_m - kv_m;
    raw_blocks = raw_wv;
    y = y_v;
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ4_K_Gpu *blk = &row_blocks[b];
    uint4 hdr = ((const uint4 *)blk)[0];
    float d = __low2float(*((const half2 *)&hdr.x));
    float dmin = __high2float(*((const half2 *)&hdr.x));

    float sc0, min0, sc1, min1;
    get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                               &min1);

    float d_sc0 = d * sc0;
    float d_sc1 = d * sc1;
    float dm_min0 = dmin * min0;
    float dm_min1 = dmin * min1;

    int v = ((const int *)blk->qs)[lane_id];
    int v0 = v & 0x0F0F0F0F;
    int v1 = (v >> 4) & 0x0F0F0F0F;

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Grouped Q4_K QK GEMV Kernel.
 * Computes Q (wq) and K (wk) projections in a single kernel dispatch when wv
 * has a different quant format. Total rows = q_m + kv_m.
 */
__global__ void gemv_qk_q4_k_cuda_kernel(float *__restrict__ y_q,
                                         float *__restrict__ y_k,
                                         const void *__restrict__ raw_wq,
                                         const void *__restrict__ raw_wk,
                                         const void *__restrict__ x_q8_1_raw,
                                         int q_m, int kv_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int total_m = q_m + kv_m;
  if (total_row >= total_m)
    return;

  bool is_k = (total_row >= q_m);
  int row = is_k ? (total_row - q_m) : total_row;
  const void *raw_blocks = is_k ? raw_wk : raw_wq;
  float *y = is_k ? y_k : y_q;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ4_K_Gpu *blk = &row_blocks[b];
    uint4 hdr = ((const uint4 *)blk)[0];
    float d = __low2float(*((const half2 *)&hdr.x));
    float dmin = __high2float(*((const half2 *)&hdr.x));

    float sc0, min0, sc1, min1;
    get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                               &min1);

    float d_sc0 = d * sc0;
    float d_sc1 = d * sc1;
    float dm_min0 = dmin * min0;
    float dm_min1 = dmin * min1;

    int v = ((const int *)blk->qs)[lane_id];
    int v0 = v & 0x0F0F0F0F;
    int v1 = (v >> 4) & 0x0F0F0F0F;

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q2_K Matrix-Vector GEMV (2-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q2_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ2_K_Gpu *row_blocks =
      (const BlockQ2_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 84);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ2_K_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float min = __half2float(*((const half *)&blk->dmin));
    const float *x_block = x + b * 256;

    int l_offset = (lane_id >= 16) ? 16 : 0;
    int l = lane_id & 15;
    int sc_lane_idx = (lane_id >= 16) ? 1 : 0;

#pragma unroll 2
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        int is = half_idx * 8 + j * 2 + sc_lane_idx;
        uint8_t sc = blk->scales[is];
        float dl = d * (float)(sc & 0x0F);
        float ml = min * (float)(sc >> 4);

        uint8_t q_byte = blk->qs[half_idx * 32 + l_offset + l];
        float q_val = (float)((q_byte >> (j * 2)) & 3);
        float w = dl * q_val - ml;

        float x_val = x_block[half_idx * 128 + j * 32 + lane_id];
        warp_dot += w * x_val;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q3_K Matrix-Vector GEMV (3-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q3_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ3_K_Gpu *row_blocks =
      (const BlockQ3_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 110);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ3_K_Gpu *blk = &row_blocks[b];
    float d_all = __half2float(*((const half *)&blk->d));
    const float *x_block = x + b * 256;

    uint32_t aux[4];
    memcpy(aux, blk->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t *scales = (const int8_t *)aux;

    int l_offset = (lane_id >= 16) ? 16 : 0;
    int l = lane_id & 15;
    int sc_lane_idx = (lane_id >= 16) ? 1 : 0;
    uint8_t hm_byte = blk->hmask[l_offset + l];

#pragma unroll 2
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        int is = half_idx * 8 + j * 2 + sc_lane_idx;
        float dl = d_all * (float)(scales[is] - 32);

        uint8_t m_bit = (uint8_t)(1 << (half_idx * 4 + j));
        int8_t h_sub = (hm_byte & m_bit) ? 0 : 4;

        uint8_t q_byte = blk->qs[half_idx * 32 + l_offset + l];
        int8_t q_low = (int8_t)((q_byte >> (j * 2)) & 3);
        float w = dl * (float)(q_low - h_sub);

        float x_val = x_block[half_idx * 128 + j * 32 + lane_id];
        warp_dot += w * x_val;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q5_K Matrix-Vector GEMV (5-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q5_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ5_K_Gpu *row_blocks =
      (const BlockQ5_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 176);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ5_K_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float dmin = __half2float(*((const half *)&blk->dmin));
    const float *x_block = x + b * 256;

#pragma unroll 4
    for (int chunk = 0; chunk < 4; chunk++) {
      float sc1, m1, sc2, m2;
      get_scale_min_k4_cuda(chunk * 2 + 0, blk->scales, &sc1, &m1);
      get_scale_min_k4_cuda(chunk * 2 + 1, blk->scales, &sc2, &m2);
      float d1 = d * sc1;
      float min1 = dmin * m1;
      float d2 = d * sc2;
      float min2 = dmin * m2;

      uint8_t u1 = 1 << (chunk * 2);
      uint8_t u2 = 2 << (chunk * 2);

      uint8_t ql = blk->qs[chunk * 32 + lane_id];
      uint8_t qh = blk->qh[lane_id];

      float w1 = d1 * (float)((ql & 0x0F) + ((qh & u1) ? 16 : 0)) - min1;
      float w2 = d2 * (float)((ql >> 4) + ((qh & u2) ? 16 : 0)) - min2;

      float x0 = x_block[chunk * 64 + lane_id];
      float x1 = x_block[chunk * 64 + 32 + lane_id];

      warp_dot += w1 * x0 + w2 * x1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = warp_dot;
  }
}

/**
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling +
 * Online Softmax)
 */
__global__ void vector_add_f32_kernel(float *__restrict__ a,
                                      const float *__restrict__ b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    a[idx] += b[idx];
  }
}

__global__ void vector_bias_add_batch_f32_kernel(float *__restrict__ a,
                                                const float *__restrict__ bias,
                                                int m, int total_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    int col = idx % m;
    a[idx] += bias[col];
  }
}

__global__ void vector_scale_f32_kernel(float *__restrict__ x, float scale,
                                        int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] *= scale;
  }
}

__global__ void swiglu_f32_kernel(float *__restrict__ out,
                                  const float *__restrict__ gate,
                                  const float *__restrict__ up, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * up[idx];
  }
}

/**
 * Custom CUDA Kernel: Fused SwiGLU + Dynamic Q8_1 Activation Quantization.
 * Directly transforms gate and up projections via silu(gate) * up and emits
 * quantized BlockQ8_1_Gpu blocks for the down-projection GEMV.
 */
__global__ void fused_swiglu_q8_1_cuda_kernel(
    BlockQ8_1_Gpu *__restrict__ y_q8_1, float *__restrict__ output_f32,
    const float *__restrict__ gate, const float *__restrict__ up, int n,
    float sparsity_threshold = 0.0f, int enable_2_4 = 0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  int lane_id = threadIdx.x % 32;
  int block_32_idx = i / 32;

  float g = gate[i];
  float u = up[i];
  float silu_g = g / (1.0f + expf(-g));
  float val = silu_g * u;

  if (output_f32 != nullptr) {
    output_f32[i] = val;
  }

  if (y_q8_1 != nullptr) {
    // 2:4 structured sparsity: retain 2 largest values per 4-element window
    if (enable_2_4) {
      int rank = 0;
      int base_lane = lane_id & ~3;
      float abs_val = fabsf(val);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float other_abs = fabsf(__shfl_sync(0xFFFFFFFF, val, base_lane + k));
        if (other_abs > abs_val || (other_abs == abs_val && ((lane_id & 3) < k))) {
          rank++;
        }
      }
      if (rank >= 2) {
        val = 0.0f;
      }
    }

    float amax = fabsf(val);
    float sum = val;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    if (amax < sparsity_threshold) {
      // Dynamic pruning: entire 32-element block zeroed out
      y_q8_1[block_32_idx].qs[lane_id] = 0;
      if (lane_id == 0) {
        y_q8_1[block_32_idx].ds = make_half2(__float2half(0.0f), __float2half(0.0f));
      }
    } else {
      float d = amax / 127.0f;
      float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
      int8_t q = (int8_t)roundf(val * id);
      y_q8_1[block_32_idx].qs[lane_id] = q;

      if (lane_id == 0) {
        y_q8_1[block_32_idx].ds = make_half2(__float2half(d), __float2half(sum));
      }
    }
  }
}

__global__ void gelu_f32_kernel(float *__restrict__ out,
                                const float *__restrict__ gate,
                                const float *__restrict__ up, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = gate[idx];
    const float SQRT_2_OVER_PI = 0.7978845608f;
    float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
    float gelu = 0.5f * g * (1.0f + tanhf(inner));
    out[idx] = gelu * up[idx];
  }
}

__global__ void logit_softcap_f32_kernel(float *__restrict__ logits,
                                         int vocab_size, float cap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < vocab_size) {
    float l = logits[idx];
    logits[idx] = cap * tanhf(l / cap);
  }
}

__device__ inline float
compute_rope_freq_yarn(int d, int head_dim, float theta, float scaling_factor,
                       int orig_ctx, float ext_factor = 1.0f,
                       float beta_fast = 32.0f, float beta_slow = 1.0f) {
  float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
  if (scaling_factor > 1.0f && orig_ctx > 0) {
    float wavelength = 6.28318530718f / freq;
    float ramp =
        ((float)orig_ctx / wavelength - beta_slow) / (beta_fast - beta_slow);
    ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
    float freq_interp = freq / scaling_factor;
    freq = (1.0f - ramp) * freq_interp + ramp * (freq * ext_factor);
  }
  return freq;
}

__global__ void convert_f32_to_f16_kernel(__half *__restrict__ dst,
                                          const float *__restrict__ src,
                                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = __float2half(src[idx]);
  }
}

__global__ void rope_single_token_kernel(
    float *__restrict__ q, float *__restrict__ k, void *__restrict__ k_out,
    int pos, int num_heads, int num_kv_heads, int head_dim, float theta,
    int is_fp16 = 0, float scaling_factor = 1.0f, int orig_ctx = 0) {
  int head_idx = blockIdx.x;
  int half_dim = head_dim / 2;
  int d = threadIdx.x;
  if (d >= half_dim)
    return;

  float freq =
      compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
  float angle = (float)pos * freq;
  float sin_val, cos_val;
  __sincosf(angle, &sin_val, &cos_val);

  if (head_idx < num_heads && q != nullptr) {
    int q_base = head_idx * head_dim;
    float q0 = q[q_base + d];
    float q1 = q[q_base + d + half_dim];
    q[q_base + d] = q0 * cos_val - q1 * sin_val;
    q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;
  }

  if (head_idx < num_kv_heads && k != nullptr) {
    int k_base = head_idx * head_dim;
    float k0 = k[k_base + d];
    float k1 = k[k_base + d + half_dim];
    float rot0 = k0 * cos_val - k1 * sin_val;
    float rot1 = k0 * sin_val + k1 * cos_val;
    k[k_base + d] = rot0;
    k[k_base + d + half_dim] = rot1;

    if (k_out != nullptr) {
      if (is_fp16) {
        __half *dst_h = (__half *)k_out;
        dst_h[k_base + d] = __float2half(rot0);
        dst_h[k_base + d + half_dim] = __float2half(rot1);
      } else {
        float *dst_f = (float *)k_out;
        dst_f[k_base + d] = rot0;
        dst_f[k_base + d + half_dim] = rot1;
      }
    }
  }
}

__global__ void
rope_batch_tokens_kernel(float *__restrict__ q, float *__restrict__ k,
                         void *__restrict__ k_cache, int start_pos,
                         int batch_size, int num_heads, int num_kv_heads,
                         int head_dim, float theta, int is_fp16 = 0,
                         float scaling_factor = 1.0f, int orig_ctx = 0) {
  int head_idx = blockIdx.x;
  int b = blockIdx.y;
  if (b >= batch_size)
    return;

  int half_dim = head_dim / 2;
  int d = threadIdx.x;
  if (d >= half_dim)
    return;

  int token_pos = start_pos + b;
  float freq =
      compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
  float angle = (float)token_pos * freq;
  float sin_val, cos_val;
  __sincosf(angle, &sin_val, &cos_val);

  if (head_idx < num_heads && q != nullptr) {
    int q_base = b * (num_heads * head_dim) + head_idx * head_dim;
    float q0 = q[q_base + d];
    float q1 = q[q_base + d + half_dim];
    q[q_base + d] = q0 * cos_val - q1 * sin_val;
    q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;
  }

  if (head_idx < num_kv_heads && k != nullptr) {
    int k_base = b * (num_kv_heads * head_dim) + head_idx * head_dim;
    float k0 = k[k_base + d];
    float k1 = k[k_base + d + half_dim];
    float rot0 = k0 * cos_val - k1 * sin_val;
    float rot1 = k0 * sin_val + k1 * cos_val;
    k[k_base + d] = rot0;
    k[k_base + d + half_dim] = rot1;

    if (k_cache != nullptr) {
      int cache_base =
          token_pos * (num_kv_heads * head_dim) + head_idx * head_dim;
      if (is_fp16) {
        __half *dst_h = (__half *)k_cache;
        dst_h[cache_base + d] = __float2half(rot0);
        dst_h[cache_base + d + half_dim] = __float2half(rot1);
      } else {
        float *dst_f = (float *)k_cache;
        dst_f[cache_base + d] = rot0;
        dst_f[cache_base + d + half_dim] = rot1;
      }
    }
  }
}

/**
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling +
 * Online Softmax + GQA) Warp-Per-Head architecture: 0 shared memory, 0
 * __syncthreads(), independent warp registers. Supports FP16 and FP32 KV cache.
 */
__global__ void flash_decode_paged_kernel(
    float *__restrict__ output, const float *__restrict__ q,
    const void *__restrict__ k_cache, const void *__restrict__ v_cache,
    int num_heads, int num_kv_heads, int head_dim, int num_kv_tokens,
    int start_token, float sm_scale, int is_fp16 = 0,
    const int *__restrict__ d_num_kv_tokens = nullptr) {
  int actual_tokens = (d_num_kv_tokens != nullptr) ? (*d_num_kv_tokens + 1) : num_kv_tokens;
  if (num_heads <= 0 || head_dim <= 0 || actual_tokens <= 0)
    return;
  int head_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (head_idx >= num_heads)
    return;
  int lane_id = threadIdx.x % 32;

  const float *q_ptr = q + head_idx * head_dim;
  float *out_ptr = output + head_idx * head_dim;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    acc_v[i] = 0.0f;

  // Cache Q values into thread-private registers once
  float q_reg[16];
  int q_items = 0;
  for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
    q_reg[q_items] = q_ptr[d];
  }

  int valid_start =
      (start_token >= 0 && start_token < actual_tokens) ? start_token : 0;

  // Grid-stride over past KV cached tokens
  for (int t = valid_start; t < actual_tokens; t++) {
    size_t tok_offset = (size_t)(t * kv_heads + kv_head_idx) * head_dim;

    // Compute Q . K dot product for this head from registers
    float score = 0.0f;
    if (is_fp16) {
      const __half *k_tok = (const __half *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * __half2float(k_tok[lane_id + i * 32]);
      }
    } else {
      const float *k_tok = (const float *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * k_tok[lane_id + i * 32];
      }
    }

// Warp reduction across the 32 lanes
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFF, score, offset);
    }

    // Broadcast total score to all 32 lanes in warp
    float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;

    // Online Softmax update with SFU expf bypass when maximum does not change
    float new_max = fmaxf(max_score, cur_score);
    float alpha =
        (t == valid_start)
            ? 0.0f
            : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);

    max_score = new_max;
    sum_exp = sum_exp * alpha + beta;

    if (is_fp16) {
      const __half *v_tok = (const __half *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] =
            acc_v[i] * alpha + beta * __half2float(v_tok[lane_id + i * 32]);
      }
    } else {
      const float *v_tok = (const float *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] = acc_v[i] * alpha + beta * v_tok[lane_id + i * 32];
      }
    }
  }

  // Final normalization
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
  for (int i = 0; i < q_items; i++) {
    out_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
  }
}

/**
 * Custom CUDA Kernel: Batched Causal FlashAttention Prefill Kernel
 * Evaluates causal attention for all B tokens in prompt chunk across past KV
 * cache in single pass. Grid: dim3 grid((num_heads + 7) / 8, batch_size) Block:
 * 256 threads (8 warps, 1 warp per head). Supports FP16 and FP32 KV cache.
 */
__global__ void flash_causal_batched_prefill_kernel(
    float *__restrict__ output, const float *__restrict__ q,
    const void *__restrict__ k_cache, const void *__restrict__ v_cache,
    int start_pos, int batch_size, int num_heads, int num_kv_heads,
    int head_dim, int kv_cap, float sm_scale, int is_fp16 = 0) {
  if (num_heads <= 0 || head_dim <= 0 || batch_size <= 0)
    return;
  int b = blockIdx.y;
  if (b >= batch_size)
    return;

  int head_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (head_idx >= num_heads)
    return;
  int lane_id = threadIdx.x % 32;

  int token_pos = start_pos + b;
  int num_kv_tokens = token_pos + 1;
  if (kv_cap > 0 && num_kv_tokens > kv_cap) {
    num_kv_tokens = kv_cap;
  }

  int q_dim = num_heads * head_dim;
  const float *q_ptr = q + b * q_dim + head_idx * head_dim;
  float *out_ptr = output + b * q_dim + head_idx * head_dim;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    acc_v[i] = 0.0f;

  // Cache Q values into thread-private registers once
  float q_reg[16];
  int q_items = 0;
  for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
    q_reg[q_items] = q_ptr[d];
  }

  // Grid-stride over past KV cached tokens (causal: up to token_pos)
  // Vectorized 8-token unrolled chunk loop for high-throughput prefill
  int t = 0;
  for (; t + 7 < num_kv_tokens; t += 8) {
    size_t tok_offset0 = (size_t)((t + 0) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset1 = (size_t)((t + 1) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset2 = (size_t)((t + 2) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset3 = (size_t)((t + 3) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset4 = (size_t)((t + 4) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset5 = (size_t)((t + 5) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset6 = (size_t)((t + 6) * kv_heads + kv_head_idx) * head_dim;
    size_t tok_offset7 = (size_t)((t + 7) * kv_heads + kv_head_idx) * head_dim;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    if (is_fp16) {
      const __half *k0 = (const __half *)k_cache + tok_offset0;
      const __half *k1 = (const __half *)k_cache + tok_offset1;
      const __half *k2 = (const __half *)k_cache + tok_offset2;
      const __half *k3 = (const __half *)k_cache + tok_offset3;
      const __half *k4 = (const __half *)k_cache + tok_offset4;
      const __half *k5 = (const __half *)k_cache + tok_offset5;
      const __half *k6 = (const __half *)k_cache + tok_offset6;
      const __half *k7 = (const __half *)k_cache + tok_offset7;
#pragma unroll
      for (int i = 0; i < q_items; i++) {
        int idx = lane_id + i * 32;
        float qr = q_reg[i];
        s0 += qr * __half2float(k0[idx]);
        s1 += qr * __half2float(k1[idx]);
        s2 += qr * __half2float(k2[idx]);
        s3 += qr * __half2float(k3[idx]);
        s4 += qr * __half2float(k4[idx]);
        s5 += qr * __half2float(k5[idx]);
        s6 += qr * __half2float(k6[idx]);
        s7 += qr * __half2float(k7[idx]);
      }
    } else {
      const float *k0 = (const float *)k_cache + tok_offset0;
      const float *k1 = (const float *)k_cache + tok_offset1;
      const float *k2 = (const float *)k_cache + tok_offset2;
      const float *k3 = (const float *)k_cache + tok_offset3;
      const float *k4 = (const float *)k_cache + tok_offset4;
      const float *k5 = (const float *)k_cache + tok_offset5;
      const float *k6 = (const float *)k_cache + tok_offset6;
      const float *k7 = (const float *)k_cache + tok_offset7;
#pragma unroll
      for (int i = 0; i < q_items; i++) {
        int idx = lane_id + i * 32;
        float qr = q_reg[i];
        s0 += qr * k0[idx];
        s1 += qr * k1[idx];
        s2 += qr * k2[idx];
        s3 += qr * k3[idx];
        s4 += qr * k4[idx];
        s5 += qr * k5[idx];
        s6 += qr * k6[idx];
        s7 += qr * k7[idx];
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      s0 += __shfl_down_sync(0xFFFFFFFF, s0, offset);
      s1 += __shfl_down_sync(0xFFFFFFFF, s1, offset);
      s2 += __shfl_down_sync(0xFFFFFFFF, s2, offset);
      s3 += __shfl_down_sync(0xFFFFFFFF, s3, offset);
      s4 += __shfl_down_sync(0xFFFFFFFF, s4, offset);
      s5 += __shfl_down_sync(0xFFFFFFFF, s5, offset);
      s6 += __shfl_down_sync(0xFFFFFFFF, s6, offset);
      s7 += __shfl_down_sync(0xFFFFFFFF, s7, offset);
    }

    float c0 = __shfl_sync(0xFFFFFFFF, s0, 0) * sm_scale;
    float c1 = __shfl_sync(0xFFFFFFFF, s1, 0) * sm_scale;
    float c2 = __shfl_sync(0xFFFFFFFF, s2, 0) * sm_scale;
    float c3 = __shfl_sync(0xFFFFFFFF, s3, 0) * sm_scale;
    float c4 = __shfl_sync(0xFFFFFFFF, s4, 0) * sm_scale;
    float c5 = __shfl_sync(0xFFFFFFFF, s5, 0) * sm_scale;
    float c6 = __shfl_sync(0xFFFFFFFF, s6, 0) * sm_scale;
    float c7 = __shfl_sync(0xFFFFFFFF, s7, 0) * sm_scale;

    float cur_max03 = fmaxf(fmaxf(c0, c1), fmaxf(c2, c3));
    float cur_max47 = fmaxf(fmaxf(c4, c5), fmaxf(c6, c7));
    float cur_max = fmaxf(cur_max03, cur_max47);
    float new_max = fmaxf(max_score, cur_max);
    float alpha = (max_score == -1e20f)
                      ? 0.0f
                      : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta0 = expf(c0 - new_max);
    float beta1 = expf(c1 - new_max);
    float beta2 = expf(c2 - new_max);
    float beta3 = expf(c3 - new_max);
    float beta4 = expf(c4 - new_max);
    float beta5 = expf(c5 - new_max);
    float beta6 = expf(c6 - new_max);
    float beta7 = expf(c7 - new_max);

    max_score = new_max;
    sum_exp = sum_exp * alpha + (beta0 + beta1 + beta2 + beta3 + beta4 + beta5 + beta6 + beta7);

    if (is_fp16) {
      const __half *v0 = (const __half *)v_cache + tok_offset0;
      const __half *v1 = (const __half *)v_cache + tok_offset1;
      const __half *v2 = (const __half *)v_cache + tok_offset2;
      const __half *v3 = (const __half *)v_cache + tok_offset3;
      const __half *v4 = (const __half *)v_cache + tok_offset4;
      const __half *v5 = (const __half *)v_cache + tok_offset5;
      const __half *v6 = (const __half *)v_cache + tok_offset6;
      const __half *v7 = (const __half *)v_cache + tok_offset7;
#pragma unroll
      for (int i = 0; i < q_items; i++) {
        int idx = lane_id + i * 32;
        acc_v[i] = acc_v[i] * alpha +
            (beta0 * __half2float(v0[idx]) +
             beta1 * __half2float(v1[idx]) +
             beta2 * __half2float(v2[idx]) +
             beta3 * __half2float(v3[idx]) +
             beta4 * __half2float(v4[idx]) +
             beta5 * __half2float(v5[idx]) +
             beta6 * __half2float(v6[idx]) +
             beta7 * __half2float(v7[idx]));
      }
    } else {
      const float *v0 = (const float *)v_cache + tok_offset0;
      const float *v1 = (const float *)v_cache + tok_offset1;
      const float *v2 = (const float *)v_cache + tok_offset2;
      const float *v3 = (const float *)v_cache + tok_offset3;
      const float *v4 = (const float *)v_cache + tok_offset4;
      const float *v5 = (const float *)v_cache + tok_offset5;
      const float *v6 = (const float *)v_cache + tok_offset6;
      const float *v7 = (const float *)v_cache + tok_offset7;
#pragma unroll
      for (int i = 0; i < q_items; i++) {
        int idx = lane_id + i * 32;
        acc_v[i] = acc_v[i] * alpha +
            (beta0 * v0[idx] +
             beta1 * v1[idx] +
             beta2 * v2[idx] +
             beta3 * v3[idx] +
             beta4 * v4[idx] +
             beta5 * v5[idx] +
             beta6 * v6[idx] +
             beta7 * v7[idx]);
      }
    }
  }

  // Remainder scalar loop for tail tokens
  for (; t < num_kv_tokens; t++) {
    size_t tok_offset = (size_t)(t * kv_heads + kv_head_idx) * head_dim;

    // Compute Q . K dot product for this head from registers
    float score = 0.0f;
    if (is_fp16) {
      const __half *k_tok = (const __half *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * __half2float(k_tok[lane_id + i * 32]);
      }
    } else {
      const float *k_tok = (const float *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * k_tok[lane_id + i * 32];
      }
    }

// Warp reduction across the 32 lanes
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFF, score, offset);
    }

    // Broadcast total score to all 32 lanes in warp
    float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;

    // Online Softmax update with SFU expf bypass
    float new_max = fmaxf(max_score, cur_score);
    float alpha =
        (max_score == -1e20f)
            ? 0.0f
            : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);

    max_score = new_max;
    sum_exp = sum_exp * alpha + beta;

    if (is_fp16) {
      const __half *v_tok = (const __half *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] =
            acc_v[i] * alpha + beta * __half2float(v_tok[lane_id + i * 32]);
      }
    } else {
      const float *v_tok = (const float *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] = acc_v[i] * alpha + beta * v_tok[lane_id + i * 32];
      }
    }
  }

  // Final normalization
  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
  for (int i = 0; i < q_items; i++) {
    out_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
  }
}

template <int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 4) void gemv_q6_k_pure_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k) {
  __shared__ BlockQ8_1_Gpu s_x[384];
  int tid = threadIdx.x;
  int num_q8_blocks = k / 32;
  const uint32_t *x_u32 = (const uint32_t *)x_q8_1_raw;
  uint32_t *s_x_u32 = (uint32_t *)s_x;
  int total_words = num_q8_blocks * 9;
  for (int item = tid; item < total_words; item += blockDim.x) {
    s_x_u32[item] = x_u32[item];
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int rows_per_block = blockDim.x / 32;
  int row = blockIdx.x * rows_per_block + warp_id;
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ6_K_Gpu *row_blocks =
      (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 210);

  int group = lane_id / 8;
  int idx = lane_id % 8;
  float thread_dot = 0.0f;

  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ8_1_Gpu *bx = &s_x[b * 8];
    float d0 = __low2float(bx[group].ds);
    float d1 = __low2float(bx[4 + group].ds);
    bool active = (d0 != 0.0f || d1 != 0.0f);
    if (__ballot_sync(0xFFFFFFFF, active) == 0) {
      continue;
    }

    const BlockQ6_K_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));

#pragma unroll
    for (int half_idx = 0; half_idx < 2; half_idx++) {
      int sub = half_idx * 4 + group;
      const BlockQ8_1_Gpu *bq8 = &bx[sub];
      float d8 = __low2float(bq8->ds);
      if (d8 != 0.0f) {
        const uint16_t *ql16 = (const uint16_t *)&blk->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
        uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
        const uint16_t *qh16 = (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
        uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

        uint32_t q_low4 = (ql4 >> ((group & 2) * 2)) & 0x0F0F0F0Fu;
        uint32_t q_high4 = ((qh4 >> (group * 2)) & 0x03030303u) << 4;
        uint32_t vx = (q_low4 | q_high4) ^ 0x20202020u;
        uint32_t vm = vx & 0x20202020u;
        int v_signed = (int)(vx | (vm << 1) | (vm << 2));

        float d_sc = d * (float)blk->scales[half_idx * 8 + group * 2 + (idx >> 2)];
        int u = ((const int *)bq8->qs)[idx];
        int dot = __dp4a(v_signed, u, 0);
        thread_dot += d_sc * d8 * (float)dot;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    if (ADD_TO_Y) {
      y[row] += thread_dot;
    } else {
      y[row] = thread_dot;
    }
  }
}

template <int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 4) void gemv_q4_k_pure_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k) {
  __shared__ BlockQ8_1_Gpu s_x[384];
  int tid = threadIdx.x;
  int num_q8_blocks = k / 32;
  const uint32_t *x_u32 = (const uint32_t *)x_q8_1_raw;
  uint32_t *s_x_u32 = (uint32_t *)s_x;
  int total_words = num_q8_blocks * 9;
  for (int item = tid; item < total_words; item += blockDim.x) {
    s_x_u32[item] = x_u32[item];
  }
  __syncthreads();

  int warp_id = threadIdx.x / 32;
  int rows_per_block = blockDim.x / 32;
  int row = blockIdx.x * rows_per_block + warp_id;
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

  for (int b = 0; b < num_blocks_per_row; b++) {
    const BlockQ8_1_Gpu *bx = &s_x[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    bool active = (d8_0 != 0.0f || d8_1 != 0.0f);
    if (__ballot_sync(0xFFFFFFFF, active) == 0) {
      continue;
    }

    const BlockQ4_K_Gpu *blk = &row_blocks[b];
    uint4 hdr = ((const uint4 *)blk)[0];
    float d = __low2float(*((const half2 *)&hdr.x));
    float dmin = __high2float(*((const half2 *)&hdr.x));

    float sc0, min0, sc1, min1;
    get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                               &min1);

    float d_sc0 = d * sc0;
    float d_sc1 = d * sc1;
    float dm_min0 = dmin * min0;
    float dm_min1 = dmin * min1;

    int v = ((const int *)blk->qs)[lane_id];
    int v0 = v & 0x0F0F0F0F;
    int v1 = (v >> 4) & 0x0F0F0F0F;

    if (d8_0 != 0.0f) {
      int u0 = ((const int *)bq8_0->qs)[sub_idx];
      int dot0 = __dp4a(v0, u0, 0);
      int sum0 = __dp4a(0x01010101, u0, 0);
      thread_dot += (d_sc0 * (float)dot0 - dm_min0 * (float)sum0) * d8_0;
    }

    if (d8_1 != 0.0f) {
      int u1 = ((const int *)bq8_1->qs)[sub_idx];
      int dot1 = __dp4a(v1, u1, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);
      thread_dot += (d_sc1 * (float)dot1 - dm_min1 * (float)sum1) * d8_1;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    if (ADD_TO_Y) {
      y[row] += thread_dot;
    } else {
      y[row] = thread_dot;
    }
  }
}

/**
 * 2D Tiled MMQ for Q4_K: Matrix Multiplication with 2D shared-memory tiling.
 * Computes: Y = W * X where W is M x K (Q4_K) and X is K x N (Q8_1).
 * Grid:   gridDim.x = (m + 63) / 64, gridDim.y = (batch_size + 63) / 64
 * Block:  blockDim.x = 32 (lanes), blockDim.y = 8 (warps) => 256 threads total.
 * Shared: 64 rows of Q4_K (9,216 B) + 64 tokens of Q8_1 (18,432 B) = 27,648 B.
 */
template <int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 1) void mmq_q4_k_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k, int batch_size) {
  int tid = threadIdx.y * 32 + threadIdx.x;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y; // 0..7

  int w_row = warp_id / 2; // 0..3 (16 rows each => 64 rows total per block)
  int w_col = warp_id % 2; // 0..1 (32 tokens each => 64 tokens total per block)

  int row_start = blockIdx.x * 64;
  int total_threads = blockDim.x * blockDim.y;

  int num_k_blocks = k / 256;
  int num_q8_blocks_per_col = k / 32;

  // Shared memory: 64 rows x 256 weights (s8) = 16,384 B
  __shared__ alignas(16) int8_t s_A[64][256];
  // Shared memory: 64 tokens x 8 sub-blocks of Q8_1 = 18,432 B
  __shared__ alignas(16) BlockQ8_1_Gpu s_x[64][8];

  const BlockQ4_K_Gpu *w_blks = (const BlockQ4_K_Gpu *)raw_blocks;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int r0 = w_row * 16 + lane_id / 4;
  int r1 = r0 + 8;
  int t0 = (lane_id % 4) * 2 + 0;
  int t1 = (lane_id % 4) * 2 + 1;

  float acc[4][4] = {{0.0f}}; // [tc 0..3][r0_t0, r0_t1, r1_t0, r1_t1]

  for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
    __shared__ int s_k_blk_active;
    if (tid == 0) s_k_blk_active = 0;
    __syncthreads();

    // 1. Load 64 tokens x 8 sub-blocks of Q8_1 into shared memory
#pragma unroll
    for (int item = tid; item < 64 * 8; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * 64 + t;
      if (actual_tok < batch_size) {
        BlockQ8_1_Gpu val_x = x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col + k_blk * 8 + sub];
        s_x[t][sub] = val_x;
        if (__low2float(val_x.ds) != 0.0f) {
          s_k_blk_active = 1;
        }
      } else {
        s_x[t][sub].ds = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 32; ++i) s_x[t][sub].qs[i] = 0;
      }
    }
    __syncthreads();

    if (s_k_blk_active == 0) {
      // Entire 256-element block is zero for all active tokens in this block!
      continue;
    }

    // 2. Unpack 64 rows x 256 elements = 2048 words into shared memory
#pragma unroll
    for (int item = tid; item < 64 * 32; item += total_threads) {
      int r = item / 32;
      int rem = item % 32;
      int chunk = rem / 8;
      int w = rem % 8;
      int actual_row = row_start + r;
      int v0 = 0, v1 = 0;
      if (actual_row < m) {
        const BlockQ4_K_Gpu *blk = &w_blks[(size_t)actual_row * num_k_blocks + k_blk];
        int val = ((const int *)&blk->qs[chunk * 32])[w];
        v0 = val & 0x0F0F0F0F;
        v1 = (val >> 4) & 0x0F0F0F0F;
      }
      ((int *)&s_A[r][(2 * chunk + 0) * 32])[w] = v0;
      ((int *)&s_A[r][(2 * chunk + 1) * 32])[w] = v1;
    }

    __syncthreads();

    // 3. Compute via Tensor Core MMA
    int actual_r0 = row_start + r0;
    int actual_r1 = row_start + r1;

    float d_r0 = 0.0f, dmin_r0 = 0.0f;
    float d_r1 = 0.0f, dmin_r1 = 0.0f;
    const BlockQ4_K_Gpu *blk0 = nullptr;
    const BlockQ4_K_Gpu *blk1 = nullptr;

    if (actual_r0 < m) {
      blk0 = &w_blks[(size_t)actual_r0 * num_k_blocks + k_blk];
      uint4 hdr0 = ((const uint4 *)blk0)[0];
      d_r0 = __low2float(*((const half2 *)&hdr0.x));
      dmin_r0 = __high2float(*((const half2 *)&hdr0.x));
    }
    if (actual_r1 < m) {
      blk1 = &w_blks[(size_t)actual_r1 * num_k_blocks + k_blk];
      uint4 hdr1 = ((const uint4 *)blk1)[0];
      d_r1 = __low2float(*((const half2 *)&hdr1.x));
      dmin_r1 = __high2float(*((const half2 *)&hdr1.x));
    }

    int row_in_tile = lane_id % 16;
    int col_byte = (lane_id / 16) * 16;
    int mem_row = w_row * 16 + row_in_tile;
    int sub_k = lane_id % 4;

#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      float sc0_r0 = 0.0f, min0_r0 = 0.0f, sc1_r0 = 0.0f, min1_r0 = 0.0f;
      float sc0_r1 = 0.0f, min0_r1 = 0.0f, sc1_r1 = 0.0f, min1_r1 = 0.0f;
      if (blk0) {
        uint4 hdr0 = ((const uint4 *)blk0)[0];
        get_scale_min_k4_from_regs(chunk, hdr0.y, hdr0.z, hdr0.w, &sc0_r0, &min0_r0, &sc1_r0, &min1_r0);
      }
      if (blk1) {
        uint4 hdr1 = ((const uint4 *)blk1)[0];
        get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_r1, &min0_r1, &sc1_r1, &min1_r1);
      }

#pragma unroll
      for (int sub_in_chunk = 0; sub_in_chunk < 2; ++sub_in_chunk) {
        int sub = 2 * chunk + sub_in_chunk;

        int a0, a1, a2, a3;
        const void *ptr = &s_A[mem_row][sub * 32 + col_byte];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(ptr));

        float cur_sc_r0 = (sub_in_chunk == 0) ? (d_r0 * sc0_r0) : (d_r0 * sc1_r0);
        float cur_min_r0 = (sub_in_chunk == 0) ? (dmin_r0 * min0_r0) : (dmin_r0 * min1_r0);
        float cur_sc_r1 = (sub_in_chunk == 0) ? (d_r1 * sc0_r1) : (d_r1 * sc1_r1);
        float cur_min_r1 = (sub_in_chunk == 0) ? (dmin_r1 * min0_r1) : (dmin_r1 * min1_r1);

#pragma unroll
        for (int tc = 0; tc < 4; ++tc) {
          int tok_base_tc = (w_col * 4 + tc) * 8;
          int col_b = tok_base_tc + lane_id / 4;

          const int *x_qs = (const int *)&s_x[col_b][sub].qs[0];
          int b0 = x_qs[sub_k];
          int b1 = x_qs[sub_k + 4];

          int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
              : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

          int cur_t0 = tok_base_tc + t0;
          int cur_t1 = tok_base_tc + t1;

          float d8_t0 = __low2float(s_x[cur_t0][sub].ds);
          float sum8_t0 = __high2float(s_x[cur_t0][sub].ds);
          float d8_t1 = __low2float(s_x[cur_t1][sub].ds);
          float sum8_t1 = __high2float(s_x[cur_t1][sub].ds);

          acc[tc][0] += (cur_sc_r0 * d8_t0 * (float)d0 - cur_min_r0 * sum8_t0);
          acc[tc][1] += (cur_sc_r0 * d8_t1 * (float)d1 - cur_min_r0 * sum8_t1);
          acc[tc][2] += (cur_sc_r1 * d8_t0 * (float)d2 - cur_min_r1 * sum8_t0);
          acc[tc][3] += (cur_sc_r1 * d8_t1 * (float)d3 - cur_min_r1 * sum8_t1);
        }
      }
    }

    __syncthreads();
  }

  int actual_r0 = row_start + r0;
  int actual_r1 = row_start + r1;

#pragma unroll
  for (int tc = 0; tc < 4; ++tc) {
    int tok_base_tc = blockIdx.y * 64 + (w_col * 4 + tc) * 8;
    int actual_t0 = tok_base_tc + t0;
    int actual_t1 = tok_base_tc + t1;

    if (actual_r0 < m) {
      if (actual_t0 < batch_size) {
        if (ADD_TO_Y) y[actual_t0 * m + actual_r0] += acc[tc][0];
        else y[actual_t0 * m + actual_r0] = acc[tc][0];
      }
      if (actual_t1 < batch_size) {
        if (ADD_TO_Y) y[actual_t1 * m + actual_r0] += acc[tc][1];
        else y[actual_t1 * m + actual_r0] = acc[tc][1];
      }
    }
    if (actual_r1 < m) {
      if (actual_t0 < batch_size) {
        if (ADD_TO_Y) y[actual_t0 * m + actual_r1] += acc[tc][2];
        else y[actual_t0 * m + actual_r1] = acc[tc][2];
      }
      if (actual_t1 < batch_size) {
        if (ADD_TO_Y) y[actual_t1 * m + actual_r1] += acc[tc][3];
        else y[actual_t1 * m + actual_r1] = acc[tc][3];
      }
    }
  }
}

__global__ __launch_bounds__(256, 1) void mmq_gate_up_q4_k_cuda_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k, int batch_size) {
  int tid = threadIdx.y * 32 + threadIdx.x;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  int row_start = blockIdx.x * 64;
  int tokens_per_block = blockDim.y * 8;
  int tok_base = blockIdx.y * tokens_per_block + warp_id * 8;
  int total_threads = blockDim.x * blockDim.y;

  int num_k_blocks = k / 256;
  int num_q8_blocks_per_col = k / 32;

  // Stride 36 uint32_t words per row gives 144 bytes = 16-byte aligned uint4 access
  alignas(16) __shared__ uint32_t s_w1_u32[64][36];
  alignas(16) __shared__ uint32_t s_w3_u32[64][36];
  alignas(16) __shared__ int s_x_qs[64][8][8];
  __shared__ half2 s_x_ds[64][8];

  float acc_gate[2][8] = {{0.0f}};
  float acc_up[2][8] = {{0.0f}};

  const uint4 *w1_u4 = (const uint4 *)raw_w1;
  const uint4 *w3_u4 = (const uint4 *)raw_w3;
  uint4 *s_w1_u4 = (uint4 *)s_w1_u32;
  uint4 *s_w3_u4 = (uint4 *)s_w3_u32;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
    // 1. Vectorized 128-bit load of 64 rows of W1 and W3 weights (576 uint4 vectors)
#pragma unroll
    for (int item = tid; item < 64 * 9; item += total_threads) {
      int r = item / 9;
      int vec_idx = item % 9;
      int actual_row = row_start + r;
      uint4 val1 = make_uint4(0, 0, 0, 0);
      uint4 val3 = make_uint4(0, 0, 0, 0);
      if (actual_row < inter_m) {
        size_t base_idx = ((size_t)actual_row * num_k_blocks + (size_t)k_blk) * 9 + vec_idx;
        val1 = w1_u4[base_idx];
        val3 = w3_u4[base_idx];
      }
      s_w1_u4[item] = val1;
      s_w3_u4[item] = val3;
    }

    // 2. Load tokens_per_block of Q8_1 activations with 16-byte guaranteed alignment
    int total_sub_blocks = tokens_per_block * 8;
#pragma unroll
    for (int item = tid; item < total_sub_blocks; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * tokens_per_block + t;
      if (actual_tok < batch_size) {
        const BlockQ8_1_Gpu *blk =
            &x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col + (size_t)k_blk * 8 + sub];
        s_x_ds[t][sub] = blk->ds;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_x_qs[t][sub][i] = ((const int *)blk->qs)[i];
        }
      } else {
        s_x_ds[t][sub] = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_x_qs[t][sub][i] = 0;
        }
      }
    }

    __syncthreads();

#pragma unroll
    for (int r = 0; r < 2; ++r) {
      int local_row = r * 32 + lane_id;
      const BlockQ4_K_Gpu *blk1 = (const BlockQ4_K_Gpu *)s_w1_u32[local_row];
      const BlockQ4_K_Gpu *blk3 = (const BlockQ4_K_Gpu *)s_w3_u32[local_row];
      uint4 hdr1 = ((const uint4 *)blk1)[0];
      float d1 = __low2float(*((const half2 *)&hdr1.x));
      float dmin1 = __high2float(*((const half2 *)&hdr1.x));
      uint4 hdr3 = ((const uint4 *)blk3)[0];
      float d3 = __low2float(*((const half2 *)&hdr3.x));
      float dmin3 = __high2float(*((const half2 *)&hdr3.x));

#pragma unroll
      for (int chunk = 0; chunk < 4; ++chunk) {
        int sub0 = 2 * chunk + 0;
        int sub1 = 2 * chunk + 1;

        float sc0_1, min0_1, sc1_1, min1_1;
        get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_1, &min0_1, &sc1_1, &min1_1);
        float sc0_3, min0_3, sc1_3, min1_3;
        get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0_3, &min0_3, &sc1_3, &min1_3);

        float d_sc0_1 = d1 * sc0_1;
        float d_sc1_1 = d1 * sc1_1;
        float dm_min0_1 = dmin1 * min0_1;
        float dm_min1_1 = dmin1 * min1_1;

        float d_sc0_3 = d3 * sc0_3;
        float d_sc1_3 = d3 * sc1_3;
        float dm_min0_3 = dmin3 * min0_3;
        float dm_min1_3 = dmin3 * min1_3;

        float d8_0[8], d8_1[8];
        float sum8_0[8], sum8_1[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int tok = warp_id * 8 + t;
          d8_0[t] = __low2float(s_x_ds[tok][sub0]);
          sum8_0[t] = __high2float(s_x_ds[tok][sub0]);
          d8_1[t] = __low2float(s_x_ds[tok][sub1]);
          sum8_1[t] = __high2float(s_x_ds[tok][sub1]);
        }

        int v0_1[8], v1_1[8];
        int v0_3[8], v1_3[8];
#pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
          int v1 = ((const int *)blk1->qs)[chunk * 8 + idx];
          v0_1[idx] = v1 & 0x0F0F0F0F;
          v1_1[idx] = (v1 >> 4) & 0x0F0F0F0F;

          int v3 = ((const int *)blk3->qs)[chunk * 8 + idx];
          v0_3[idx] = v3 & 0x0F0F0F0F;
          v1_3[idx] = (v3 >> 4) & 0x0F0F0F0F;
        }

        int idot0_1[8] = {0};
        int idot1_1[8] = {0};
        int idot0_3[8] = {0};
        int idot1_3[8] = {0};

#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int tok = warp_id * 8 + t;
          const int *q8_0 = s_x_qs[tok][sub0];
          const int *q8_1 = s_x_qs[tok][sub1];
#pragma unroll
          for (int idx = 0; idx < 8; ++idx) {
            int u0 = q8_0[idx];
            int u1 = q8_1[idx];

            idot0_1[t] = __dp4a(v0_1[idx], u0, idot0_1[t]);
            idot1_1[t] = __dp4a(v1_1[idx], u1, idot1_1[t]);
            idot0_3[t] = __dp4a(v0_3[idx], u0, idot0_3[t]);
            idot1_3[t] = __dp4a(v1_3[idx], u1, idot1_3[t]);
          }
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
          float term0_1 = d_sc0_1 * d8_0[t] * (float)idot0_1[t] - dm_min0_1 * sum8_0[t];
          float term1_1 = d_sc1_1 * d8_1[t] * (float)idot1_1[t] - dm_min1_1 * sum8_1[t];
          acc_gate[r][t] += term0_1 + term1_1;

          float term0_3 = d_sc0_3 * d8_0[t] * (float)idot0_3[t] - dm_min0_3 * sum8_0[t];
          float term1_3 = d_sc1_3 * d8_1[t] * (float)idot1_3[t] - dm_min1_3 * sum8_1[t];
          acc_up[r][t] += term0_3 + term1_3;
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int cur_row = row_start + r * 32 + lane_id;
    if (cur_row < inter_m) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        int cur_tok = tok_base + t;
        if (cur_tok < batch_size) {
          y_gate[cur_tok * inter_m + cur_row] = acc_gate[r][t];
          y_up[cur_tok * inter_m + cur_row] = acc_up[r][t];
        }
      }
    }
  }
}

static __device__ __forceinline__ int get_int_b2(const void *x, int i32) {
  const uint16_t *x16 = (const uint16_t *)x;
  return (int)x16[2 * i32 + 0] | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int unpack_q6_k_signed_word(const BlockQ6_K_Gpu *blk, int half_idx, int g, int idx) {
  const uint16_t *ql16 = (const uint16_t *)&blk->ql[half_idx * 64 + (g & 1) * 32 + idx * 4];
  uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
  const uint16_t *qh16 = (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
  uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

  uint32_t q_low4 = (ql4 >> ((g & 2) * 2)) & 0x0F0F0F0Fu;
  uint32_t q_high4 = ((qh4 >> (g * 2)) & 0x03030303u) << 4;
  uint32_t vx = (q_low4 | q_high4) ^ 0x20202020u;
  uint32_t vm = vx & 0x20202020u;
  return (int)(vx | (vm << 1) | (vm << 2));
}

template <int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 2) void mmq_mma_q6_k_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k, int batch_size) {
  int tid = threadIdx.y * 32 + threadIdx.x;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  int warp_row_tile = warp_id % 4; // 0..3 (16 rows each = 64 rows per block)
  int warp_col_tile = warp_id / 4; // 0..1 (8 tokens each = 16 tokens per block)

  int group = lane_id / 4;   // 0..7
  int sub_idx = lane_id % 4; // 0..3

  int local_r0 = warp_row_tile * 16 + group;
  int local_r1 = warp_row_tile * 16 + group + 8;
  int local_t0 = warp_col_tile * 8 + sub_idx * 2 + 0;
  int local_t1 = warp_col_tile * 8 + sub_idx * 2 + 1;

  int num_k_blocks = k / 256;
  int num_q8_blocks_per_col = k / 32;

  __shared__ uint16_t s_w_u16[64 * 105];
  __shared__ BlockQ8_1_Gpu s_x[16][8];

  float acc_r0_t0 = 0.0f;
  float acc_r0_t1 = 0.0f;
  float acc_r1_t0 = 0.0f;
  float acc_r1_t1 = 0.0f;

  const uint16_t *w_u16 = (const uint16_t *)raw_blocks;
  uint32_t *s_x_u32 = (uint32_t *)s_x;
  const uint32_t *x_u32 = (const uint32_t *)x_q8_1_raw;

  int idx0 = sub_idx * 2 + 0;
  int idx1 = sub_idx * 2 + 1;
  int mask0 = (sub_idx < 2) ? -1 : 0;
  int mask1 = (sub_idx >= 2) ? -1 : 0;

  for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
#pragma unroll
    for (int item = tid; item < 64 * 105; item += 256) {
      int r = item / 105;
      int word = item % 105;
      int actual_row = blockIdx.x * 64 + r;
      if (actual_row < m) {
        size_t idx = (size_t)actual_row * num_k_blocks * 105 + (size_t)k_blk * 105 + word;
        s_w_u16[item] = w_u16[idx];
      } else {
        s_w_u16[item] = 0;
      }
    }

#pragma unroll
    for (int item = tid; item < 16 * 72; item += 256) {
      int t = item / 72;
      int word = item % 72;
      int actual_tok = blockIdx.y * 16 + t;
      if (actual_tok < batch_size) {
        size_t idx = (size_t)actual_tok * num_q8_blocks_per_col * 9 + (size_t)k_blk * 72 + word;
        s_x_u32[item] = x_u32[idx];
      } else {
        s_x_u32[item] = 0;
      }
    }

    __syncthreads();

    const BlockQ6_K_Gpu *blk0 = (const BlockQ6_K_Gpu *)&s_w_u16[local_r0 * 105];
    const BlockQ6_K_Gpu *blk1 = (const BlockQ6_K_Gpu *)&s_w_u16[local_r1 * 105];

    float d0 = __half2float(*((const half *)&blk0->d));
    float d1 = __half2float(*((const half *)&blk1->d));

#pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
      int half_idx = sub / 4;
      int g = sub % 4;

      float sc0_0 = d0 * (float)blk0->scales[sub * 2 + 0];
      float sc1_0 = d0 * (float)blk0->scales[sub * 2 + 1];
      float sc0_1 = d1 * (float)blk1->scales[sub * 2 + 0];
      float sc1_1 = d1 * (float)blk1->scales[sub * 2 + 1];

      int v0_0 = unpack_q6_k_signed_word(blk0, half_idx, g, idx0);
      int v1_0 = unpack_q6_k_signed_word(blk0, half_idx, g, idx1);
      int v0_1 = unpack_q6_k_signed_word(blk1, half_idx, g, idx0);
      int v1_1 = unpack_q6_k_signed_word(blk1, half_idx, g, idx1);

      int a0_0 = v0_0 & mask0;
      int a1_0 = v1_0 & mask0;
      int a2_0 = v0_1 & mask0;
      int a3_0 = v1_1 & mask0;

      int a0_1 = v0_0 & mask1;
      int a1_1 = v1_0 & mask1;
      int a2_1 = v0_1 & mask1;
      int a3_1 = v1_1 & mask1;

      int b0 = ((const int *)s_x[local_t0][sub].qs)[group];
      int b1 = ((const int *)s_x[local_t1][sub].qs)[group];

      int d0_0, d0_1, d0_2, d0_3;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          : "=r"(d0_0), "=r"(d0_1), "=r"(d0_2), "=r"(d0_3)
          : "r"(a0_0), "r"(a1_0), "r"(a2_0), "r"(a3_0),
            "r"(b0), "r"(b1),
            "r"(0), "r"(0), "r"(0), "r"(0));

      int d1_0, d1_1, d1_2, d1_3;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
          "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
          : "=r"(d1_0), "=r"(d1_1), "=r"(d1_2), "=r"(d1_3)
          : "r"(a0_1), "r"(a1_1), "r"(a2_1), "r"(a3_1),
            "r"(b0), "r"(b1),
            "r"(0), "r"(0), "r"(0), "r"(0));

      float d8_t0 = __low2float(s_x[local_t0][sub].ds);
      float d8_t1 = __low2float(s_x[local_t1][sub].ds);

      acc_r0_t0 += (sc0_0 * (float)d0_0 + sc1_0 * (float)d1_0) * d8_t0;
      acc_r0_t1 += (sc0_0 * (float)d0_1 + sc1_0 * (float)d1_1) * d8_t1;
      acc_r1_t0 += (sc0_1 * (float)d0_2 + sc1_1 * (float)d1_2) * d8_t0;
      acc_r1_t1 += (sc0_1 * (float)d0_3 + sc1_1 * (float)d1_3) * d8_t1;
    }

    __syncthreads();
  }

  int r0 = blockIdx.x * 64 + local_r0;
  int r1 = blockIdx.x * 64 + local_r1;
  int t0 = blockIdx.y * 16 + local_t0;
  int t1 = blockIdx.y * 16 + local_t1;

  if (r0 < m) {
    if (t0 < batch_size) {
      if (ADD_TO_Y) y[t0 * m + r0] += acc_r0_t0;
      else y[t0 * m + r0] = acc_r0_t0;
    }
    if (t1 < batch_size) {
      if (ADD_TO_Y) y[t1 * m + r0] += acc_r0_t1;
      else y[t1 * m + r0] = acc_r0_t1;
    }
  }
  if (r1 < m) {
    if (t0 < batch_size) {
      if (ADD_TO_Y) y[t0 * m + r1] += acc_r1_t0;
      else y[t0 * m + r1] = acc_r1_t0;
    }
    if (t1 < batch_size) {
      if (ADD_TO_Y) y[t1 * m + r1] += acc_r1_t1;
      else y[t1 * m + r1] = acc_r1_t1;
    }
  }
}

template <int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 1) void mmq_q6_k_cuda_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k, int batch_size) {
  int tid = threadIdx.y * 32 + threadIdx.x;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;

  int row_start = blockIdx.x * 64;
  int tokens_per_block = blockDim.y * 8;
  int tok_base = blockIdx.y * tokens_per_block + warp_id * 8;
  int total_threads = blockDim.x * blockDim.y;

  int num_k_blocks = k / 256;
  int num_q8_blocks_per_col = k / 32;

  alignas(16) __shared__ int s_w_qs[64][65];
  __shared__ int8_t s_w_sc[64][20];
  __shared__ half s_w_d[64];
  alignas(16) __shared__ int s_x_qs[64][8][8];
  __shared__ half2 s_x_ds[64][8];

  float acc[2][8] = {{0.0f}};
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  for (int k_blk = 0; k_blk < num_k_blocks; ++k_blk) {
    // 1. Vectorized coalesced warp-level unpack of 64 rows of Q6_K weights into shared memory
    int warps_in_block = blockDim.y;
#pragma unroll
    for (int r0 = 0; r0 < 64; r0 += warps_in_block) {
      int r = r0 + warp_id;
      int actual_row = row_start + r;
      if (r < 64 && actual_row < m) {
        const BlockQ6_K_Gpu *blk = (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
            ((size_t)actual_row * num_k_blocks + (size_t)k_blk) * 210);

        int ql4 = get_int_b2(blk->ql, lane_id);
        int ql_low = ql4 & 0x0F0F0F0F;
        int ql_high = (ql4 >> 4) & 0x0F0F0F0F;

        int half = lane_id / 16;
        int idx = lane_id % 8;
        int is_second_32 = (lane_id / 8) % 2;

        int qh_idx = half * 8 + idx;
        int qh4 = get_int_b2(blk->qh, qh_idx);

        int g_low = (is_second_32 == 0) ? 0 : 1;
        int g_high = (is_second_32 == 0) ? 2 : 3;

        int qh_low = ((qh4 >> (g_low * 2)) & 0x03030303) << 4;
        int qh_high = ((qh4 >> (g_high * 2)) & 0x03030303) << 4;

        int v_low = __vsubss4(ql_low | qh_low, 0x20202020);
        int v_high = __vsubss4(ql_high | qh_high, 0x20202020);

        int sub_low = half * 4 + g_low;
        int sub_high = half * 4 + g_high;

        s_w_qs[r][sub_low * 8 + idx] = v_low;
        s_w_qs[r][sub_high * 8 + idx] = v_high;

        if (lane_id < 4) {
          ((int *)&s_w_sc[r][0])[lane_id] = get_int_b2(blk->scales, lane_id);
        }
        if (lane_id == 0) {
          s_w_d[r] = blk->d;
        }
      } else if (r < 64) {
        int half = lane_id / 16;
        int idx = lane_id % 8;
        int is_second_32 = (lane_id / 8) % 2;
        int g_low = (is_second_32 == 0) ? 0 : 1;
        int g_high = (is_second_32 == 0) ? 2 : 3;
        int sub_low = half * 4 + g_low;
        int sub_high = half * 4 + g_high;
        s_w_qs[r][sub_low * 8 + idx] = 0;
        s_w_qs[r][sub_high * 8 + idx] = 0;
        if (lane_id < 4) {
          ((int *)&s_w_sc[r][0])[lane_id] = 0;
        }
        if (lane_id == 0) {
          s_w_d[r] = __float2half(0.0f);
        }
      }
    }

    // 2. Load tokens_per_block of Q8_1 activations with 16-byte guaranteed alignment
    int total_sub_blocks = tokens_per_block * 8;
#pragma unroll
    for (int item = tid; item < total_sub_blocks; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * tokens_per_block + t;
      if (actual_tok < batch_size) {
        const BlockQ8_1_Gpu *blk =
            &x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col + (size_t)k_blk * 8 + sub];
        s_x_ds[t][sub] = blk->ds;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_x_qs[t][sub][i] = ((const int *)blk->qs)[i];
        }
      } else {
        s_x_ds[t][sub] = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          s_x_qs[t][sub][i] = 0;
        }
      }
    }

    __syncthreads();

    // 3. Ultra-fast DP4A compute loop reading unpacked INT8 weights with zero ALU twiddling
    for (int s = 0; s < 8; ++s) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        int local_row = r * 32 + lane_id;
        float d = __half2float(s_w_d[local_row]);
        float d_sc0 = d * (float)s_w_sc[local_row][s * 2 + 0];
        float d_sc1 = d * (float)s_w_sc[local_row][s * 2 + 1];

        int v[8];
#pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
          v[idx] = s_w_qs[local_row][s * 8 + idx];
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
          int tok = warp_id * 8 + t;
          float d8 = __low2float(s_x_ds[tok][s]);
          const int *q8 = s_x_qs[tok][s];

          int idot0 = 0;
          int idot1 = 0;

#pragma unroll
          for (int idx = 0; idx < 4; ++idx) {
            idot0 = __dp4a(v[idx], q8[idx], idot0);
          }
#pragma unroll
          for (int idx = 4; idx < 8; ++idx) {
            idot1 = __dp4a(v[idx], q8[idx], idot1);
          }

          acc[r][t] += (d_sc0 * (float)idot0 + d_sc1 * (float)idot1) * d8;
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int row = row_start + r * 32 + lane_id;
    if (row < m) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        int cur_tok = tok_base + t;
        if (cur_tok < batch_size) {
          if (ADD_TO_Y) {
            y[cur_tok * m + row] += acc[r][t];
          } else {
            y[cur_tok * m + row] = acc[r][t];
          }
        }
      }
    }
  }
}

// =========================================================================
// Hardware-Accelerated cuBLAS FP16 Tensor Core Prefill & Dequantization Engine
// =========================================================================

static cublasHandle_t g_cublas_handle = nullptr;

extern "C" int oxide_cuda_init_cublas(cudaStream_t stream) {
  if (!g_cublas_handle) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) return (int)status;
  }
  if (stream) {
    cublasSetStream(g_cublas_handle, stream);
  }
  return 0;
}

__global__ void dequantize_q4_k_to_f16_cuda_kernel(half *__restrict__ dst,
                                                   const void *__restrict__ raw_blocks,
                                                   int num_blocks) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int ib = blockIdx.x * (blockDim.x / 32) + warp_id;
  if (ib >= num_blocks) return;

  const BlockQ4_K_Gpu *blk = &((const BlockQ4_K_Gpu *)raw_blocks)[ib];
  uint4 hdr = ((const uint4 *)blk)[0];
  float dall = __low2float(*((const half2 *)&hdr.x));
  float dmin = __high2float(*((const half2 *)&hdr.x));

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;

  float sc0, min0, sc1, min1;
  get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

  float d1 = dall * sc0; float m1 = dmin * min0;
  float d2 = dall * sc1; float m2 = dmin * min1;

  int v = ((const int *)blk->qs)[lane_id];
  int v0 = v & 0x0F0F0F0F;
  int v1 = (v >> 4) & 0x0F0F0F0F;

  half2 out0_low = make_half2(__float2half(d1 * (float)(v0 & 0xFF) - m1),
                              __float2half(d1 * (float)((v0 >> 8) & 0xFF) - m1));
  half2 out0_high = make_half2(__float2half(d1 * (float)((v0 >> 16) & 0xFF) - m1),
                               __float2half(d1 * (float)((v0 >> 24) & 0xFF) - m1));

  half2 out1_low = make_half2(__float2half(d2 * (float)(v1 & 0xFF) - m2),
                              __float2half(d2 * (float)((v1 >> 8) & 0xFF) - m2));
  half2 out1_high = make_half2(__float2half(d2 * (float)((v1 >> 16) & 0xFF) - m2),
                               __float2half(d2 * (float)((v1 >> 24) & 0xFF) - m2));

  size_t base0 = (size_t)ib * 256 + (size_t)(2 * chunk + 0) * 32 + (size_t)sub_idx * 4;
  size_t base1 = (size_t)ib * 256 + (size_t)(2 * chunk + 1) * 32 + (size_t)sub_idx * 4;

  *((half2 *)&dst[base0 + 0]) = out0_low;
  *((half2 *)&dst[base0 + 2]) = out0_high;

  *((half2 *)&dst[base1 + 0]) = out1_low;
  *((half2 *)&dst[base1 + 2]) = out1_high;
}

__global__ void dequantize_q6_k_to_f16_cuda_kernel(half *__restrict__ dst,
                                                   const void *__restrict__ raw_blocks,
                                                   int num_blocks) {
  int sb_idx = threadIdx.x / 64;
  int tid = threadIdx.x % 64;
  int ib = blockIdx.x * (blockDim.x / 64) + sb_idx;
  if (ib >= num_blocks) return;

  const BlockQ6_K_Gpu *blk = &((const BlockQ6_K_Gpu *)raw_blocks)[ib];
  int ip = tid / 32;       // 0 or 1
  int il = tid % 32;       // 0..31
  int is_idx = 8 * ip + il / 16;

  float d = __half2float(*((const half *)&blk->d));
  const uint8_t *ql = blk->ql + 64 * ip + il;
  const uint8_t qh = blk->qh[32 * ip + il];
  const int8_t *sc = (const int8_t *)blk->scales + is_idx;

  int8_t q0 = (int8_t)((ql[ 0] & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
  int8_t q1 = (int8_t)((ql[32] & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
  int8_t q2 = (int8_t)((ql[ 0] >>   4) | (((qh >> 4) & 3) << 4)) - 32;
  int8_t q3 = (int8_t)((ql[32] >>   4) | (((qh >> 6) & 3) << 4)) - 32;

  size_t base = (size_t)ib * 256 + (size_t)128 * ip + (size_t)il;
  dst[base +  0] = __float2half(d * (float)sc[0] * (float)q0);
  dst[base + 32] = __float2half(d * (float)sc[2] * (float)q1);
  dst[base + 64] = __float2half(d * (float)sc[4] * (float)q2);
  dst[base + 96] = __float2half(d * (float)sc[6] * (float)q3);
}

extern "C" int launch_cuda_dequantize_q4_k_f16(void *dst_f16, const void *src_q4_k,
                                               int num_elements, cudaStream_t stream) {
  if (!dst_f16 || !src_q4_k || num_elements <= 0) return 0;
  int num_blocks = num_elements / 256;
  if (num_blocks <= 0) return 0;
  int block_dim = 256;
  int warps_per_block = block_dim / 32;
  int grid_dim = (num_blocks + warps_per_block - 1) / warps_per_block;
  dequantize_q4_k_to_f16_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
      (half *)dst_f16, src_q4_k, num_blocks);
  return cudaGetLastError();
}

extern "C" int launch_cuda_dequantize_q6_k_f16(void *dst_f16, const void *src_q6_k,
                                               int num_elements, cudaStream_t stream) {
  if (!dst_f16 || !src_q6_k || num_elements <= 0) return 0;
  int num_blocks = num_elements / 256;
  if (num_blocks <= 0) return 0;
  int block_dim = 256;
  int sb_per_block = block_dim / 64;
  int grid_dim = (num_blocks + sb_per_block - 1) / sb_per_block;
  dequantize_q6_k_to_f16_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
      (half *)dst_f16, src_q6_k, num_blocks);
  return cudaGetLastError();
}

extern "C" int launch_cuda_cublas_gemm_fp16_f32(
    float *d_y, const void *d_w_f16, const void *d_x_f16,
    int m, int k, int batch_size, int add_to_y, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0) return 0;
  if (oxide_cuda_init_cublas(stream) != 0) return -1;

  float alpha = 1.0f;
  float beta = add_to_y ? 1.0f : 0.0f;

  cublasStatus_t status = cublasGemmEx(
      g_cublas_handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      m, batch_size, k,
      &alpha,
      d_w_f16, CUDA_R_16F, k,
      d_x_f16, CUDA_R_16F, k,
      &beta,
      d_y, CUDA_R_32F, m,
      CUBLAS_COMPUTE_32F_FAST_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return (int)status;
}

// C-ABI Launch Wrappers for FFI

extern "C" {

int launch_cuda_rmsnorm(float *output, const float *input, const float *weight,
                        int num_tokens, int hidden_dim, float eps,
                        float weight_offset, cudaStream_t stream) {
  int block_size = 256;
  if (hidden_dim < 256)
    block_size = hidden_dim;
  rms_norm_f32_kernel<<<num_tokens, block_size, 0, stream>>>(
      output, input, weight, hidden_dim, eps, weight_offset);
  return cudaGetLastError();
}

int launch_cuda_fused_rmsnorm_q8_1(float *output, void *y_q8_1,
                                   const float *input, const float *weight,
                                   int num_tokens, int hidden_dim, float eps,
                                   float weight_offset, cudaStream_t stream) {
  if (hidden_dim <= 0 || num_tokens <= 0)
    return 0;
  int block_size = 256;
  if (hidden_dim >= 512 && hidden_dim % 512 == 0) {
    block_size = 512;
  }
  fused_rmsnorm_q8_1_cuda_kernel<<<num_tokens, block_size, 0, stream>>>(
      output, (BlockQ8_1_Gpu *)y_q8_1, input, weight, hidden_dim, eps,
      weight_offset);
  return cudaGetLastError();
}

int launch_cuda_rope(float *q, float *k, const float *cos_table,
                     const float *sin_table, int seq_len, int num_heads,
                     int head_dim, cudaStream_t stream) {
  dim3 grid(seq_len, num_heads);
  int block_size = head_dim / 2;
  rope_embedding_kernel<<<grid, block_size, 0, stream>>>(
      q, k, cos_table, sin_table, seq_len, num_heads, head_dim);
  return cudaGetLastError();
}

static inline void get_gemv_launch_dims(int m, int &grid, int &block) {
  const int warps_per_block = 8;
  block = warps_per_block * 32; // 256 threads
  grid = (m + warps_per_block - 1) / warps_per_block;
}

int launch_cuda_gemv_q4_0(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q4_0_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q5_0(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q5_0_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_tq2_0(float *y, const void *raw_blocks, const float *x,
                           int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_tq2_0_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_quantize_q8_1(const float *x, void *y_q8_1, int n,
                              cudaStream_t stream) {
  if (n <= 0)
    return 0;
  int num_blocks = n / 32;
  int warps_per_block = 8; // 256 threads
  int block_size = warps_per_block * 32;
  int grid = (num_blocks + warps_per_block - 1) / warps_per_block;
  quantize_f32_to_q8_1_cuda_kernel<<<grid, block_size, 0, stream>>>(
      x, (BlockQ8_1_Gpu *)y_q8_1, n);
  return cudaGetLastError();
}

int launch_cuda_mmvq_q8_0(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size == 1) {
    mmvq_q8_0_cuda_kernel<1>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
  } else if (batch_size <= 2) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 6) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<6><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 8) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<8><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 16) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<4, 0, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 20) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<5, 0, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 24) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<6, 0, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    int num_chunks = (batch_size + 15) / 16;
    dim3 grid_2d(grid * 4, num_chunks);
    mmvq_q8_0_cuda_kernel<4, 0, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_mmvq_q8_0_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size == 1) {
    mmvq_q8_0_cuda_kernel<1, 1>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
  } else if (batch_size <= 2) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<2, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<4, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 6) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<6, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 8) {
    dim3 grid_2d(grid, 1);
    mmvq_q8_0_cuda_kernel<8, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 16) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<4, 1, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 20) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<5, 1, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 24) {
    dim3 grid_2d(grid * 4, 1);
    mmvq_q8_0_cuda_kernel<6, 1, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    int num_chunks = (batch_size + 15) / 16;
    dim3 grid_2d(grid * 4, num_chunks);
    mmvq_q8_0_cuda_kernel<4, 1, 4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_q8_0(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream) {
  return launch_cuda_mmvq_q8_0(y, raw_blocks, x_q8_1, m, k, 1, stream);
}

int launch_cuda_gemv_q8_0_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream) {
  return launch_cuda_mmvq_q8_0_fused_add(y, raw_blocks, x_q8_1, m, k, 1,
                                         stream);
}

int launch_cuda_gemv_q6_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream);
int launch_cuda_gemv_q6_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream);
int launch_cuda_gemv_q4_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream);
int launch_cuda_gemv_q4_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream);

int launch_cuda_mmvq_q6_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    return launch_cuda_gemv_q6_k(y, raw_blocks, x_q8_1, m, k, stream);
  }
  if (batch_size >= 8) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q6_k_cuda_kernel<0><<<grid, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q6_k_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q6_k_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q6_k_cuda_kernel<8><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_mmvq_q6_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    return launch_cuda_gemv_q6_k_fused_add(y, raw_blocks, x_q8_1, m, k, stream);
  }
  if (batch_size >= 8) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q6_k_cuda_kernel<1><<<grid, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q6_k_cuda_kernel<2, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q6_k_cuda_kernel<4, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q6_k_cuda_kernel<8, 1><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_q6_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q6_k_pure_cuda_kernel<0><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q6_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q6_k_pure_cuda_kernel<1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
  return cudaGetLastError();
}

int launch_cuda_mmvq_q4_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    return launch_cuda_gemv_q4_k(y, raw_blocks, x_q8_1, m, k, stream);
  }
  if (batch_size >= 8) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0><<<grid, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q4_k_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q4_k_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_mmvq_q4_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    return launch_cuda_gemv_q4_k_fused_add(y, raw_blocks, x_q8_1, m, k, stream);
  }
  if (batch_size >= 8) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<1><<<grid, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q4_k_cuda_kernel<2, 1><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q4_k_cuda_kernel<4, 1><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_q4_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q4_k_pure_cuda_kernel<0><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q4_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q4_k_pure_cuda_kernel<1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
  return cudaGetLastError();
}

int launch_cuda_mmvq_gate_up_q4_k(float *y_gate, float *y_up,
                                  const void *raw_w1, const void *raw_w3,
                                  const void *x_q8_1, int inter_m, int k,
                                  int batch_size, cudaStream_t stream) {
  if (inter_m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    int g, b;
    get_gemv_launch_dims(inter_m, g, b);
    gemv_gate_up_fused_pair_q4_k_cuda_kernel<<<g, b, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k);
    return cudaGetLastError();
  }
  if (batch_size >= 8) {
    dim3 grid((inter_m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0><<<grid, block, 0, stream>>>(
        y_gate, raw_w1, x_q8_1, inter_m, k, batch_size);
    mmq_q4_k_cuda_kernel<0><<<grid, block, 0, stream>>>(
        y_up, raw_w3, x_q8_1, inter_m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(inter_m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_gate_up_q4_k_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_gate_up_q4_k_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_gate_up_q4_k(float *y_gate, float *y_up,
                                  const void *raw_w1, const void *raw_w3,
                                  const void *x_q8_1, int inter_m, int k,
                                  cudaStream_t stream) {
  int grid, block;
  get_gemv_launch_dims(inter_m, grid, block);
  gemv_gate_up_fused_pair_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(
      y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k);
  return cudaGetLastError();
}

int launch_cuda_convert_f32_to_f16(void *dst, const float *src, int n,
                                   cudaStream_t stream);

int launch_cuda_mmvq_qkv_mixed(float *y_q, float *y_k, float *y_v,
                               void *v_cache_fp16, int start_pos,
                               const void *raw_wq, const void *raw_wk,
                               const void *raw_wv, const void *x_q8_1, int q_m,
                               int kv_m, int k, int v_quant, int batch_size,
                               cudaStream_t stream) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0 || batch_size <= 0)
    return 0;

  if (batch_size >= 8) {
    dim3 grid_q((q_m + 63) / 64, (batch_size + 63) / 64);
    dim3 grid_k((kv_m + 63) / 64, (batch_size + 63) / 64);
    dim3 grid_v((kv_m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0><<<grid_q, block, 0, stream>>>(
        y_q, raw_wq, x_q8_1, q_m, k, batch_size);
    mmq_q4_k_cuda_kernel<0><<<grid_k, block, 0, stream>>>(
        y_k, raw_wk, x_q8_1, kv_m, k, batch_size);
    if (v_quant == 1) {
      mmq_q4_k_cuda_kernel<0><<<grid_v, block, 0, stream>>>(
          y_v, raw_wv, x_q8_1, kv_m, k, batch_size);
    } else {
      mmq_q6_k_cuda_kernel<0><<<grid_v, block, 0, stream>>>(
          y_v, raw_wv, x_q8_1, kv_m, k, batch_size);
    }
    if (v_cache_fp16 != nullptr) {
      int total_n = batch_size * kv_m;
      int b_size = 256;
      int g_size = (total_n + b_size - 1) / b_size;
      convert_f32_to_f16_kernel<<<g_size, b_size, 0, stream>>>(
          ((__half *)v_cache_fp16) + (size_t)start_pos * kv_m, y_v, total_n);
    }
    return cudaGetLastError();
  }

  int total_m = q_m + 2 * kv_m;
  int grid, block;
  get_gemv_launch_dims(total_m, grid, block);

  if (v_quant == 1) {
    if (batch_size == 1) {
      gemv_qkv_mixed_pure_warp_cuda_kernel<1><<<grid, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k);
    } else if (batch_size <= 2) {
      dim3 grid_2d(grid, (batch_size + 1) / 2);
      mmvq_qkv_mixed_cuda_kernel<2, 1><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    } else if (batch_size <= 4) {
      dim3 grid_2d(grid, (batch_size + 3) / 4);
      mmvq_qkv_mixed_cuda_kernel<4, 1><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    } else {
      dim3 grid_2d(grid, (batch_size + 7) / 8);
      mmvq_qkv_mixed_cuda_kernel<8, 1><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    }
  } else {
    if (batch_size == 1) {
      gemv_qkv_mixed_pure_warp_cuda_kernel<2><<<grid, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k);
    } else if (batch_size <= 2) {
      dim3 grid_2d(grid, (batch_size + 1) / 2);
      mmvq_qkv_mixed_cuda_kernel<2, 2><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    } else if (batch_size <= 4) {
      dim3 grid_2d(grid, (batch_size + 3) / 4);
      mmvq_qkv_mixed_cuda_kernel<4, 2><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    } else {
      dim3 grid_2d(grid, (batch_size + 7) / 8);
      mmvq_qkv_mixed_cuda_kernel<8, 2><<<grid_2d, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k, batch_size);
    }
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_qkv_q4_k(float *y_q, float *y_k, float *y_v,
                              const void *raw_wq, const void *raw_wk,
                              const void *raw_wv, const void *x_q8_1, int q_m,
                              int kv_m, int k, cudaStream_t stream) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0)
    return 0;
  int total_m = q_m + 2 * kv_m;
  int grid, block;
  get_gemv_launch_dims(total_m, grid, block);
  gemv_qkv_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(
      y_q, y_k, y_v, raw_wq, raw_wk, raw_wv, x_q8_1, q_m, kv_m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_qk_q4_k(float *y_q, float *y_k, const void *raw_wq,
                             const void *raw_wk, const void *x_q8_1, int q_m,
                             int kv_m, int k, cudaStream_t stream) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0)
    return 0;
  int total_m = q_m + kv_m;
  int grid, block;
  get_gemv_launch_dims(total_m, grid, block);
  gemv_qk_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(y_q, y_k, raw_wq, raw_wk,
                                                       x_q8_1, q_m, kv_m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_ptq1_0(float *y, const void *raw_blocks, const float *x,
                            int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_ptq1_0_cuda_kernel<<<grid, block, 0, stream>>>(
      y, (const uint8_t *)raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q4_1(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (!y || !raw_blocks || !x || m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q4_1_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q5_1(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (!y || !raw_blocks || !x || m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q5_1_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_iq4_nl(float *y, const void *raw_blocks, const float *x,
                            int m, int k, cudaStream_t stream) {
  if (!y || !raw_blocks || !x || m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_iq4_nl_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q2_k(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q2_k_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q3_k(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q3_k_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_q5_k(float *y, const void *raw_blocks, const float *x,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  gemv_q5_k_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
  return cudaGetLastError();
}

int launch_cuda_vector_add(float *a, const float *b, int n,
                           cudaStream_t stream) {
  int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  vector_add_f32_kernel<<<grid, block_size, 0, stream>>>(a, b, n);
  return cudaGetLastError();
}

int launch_cuda_vector_bias_add_batch(float *a, const float *bias, int m,
                                      int batch_size, cudaStream_t stream) {
  if (m <= 0 || batch_size <= 0 || a == nullptr || bias == nullptr)
    return 0;
  int total = m * batch_size;
  int block_size = 256;
  int grid = (total + block_size - 1) / block_size;
  vector_bias_add_batch_f32_kernel<<<grid, block_size, 0, stream>>>(a, bias, m, total);
  return cudaGetLastError();
}

int launch_cuda_vector_scale(float *x, float scale, int n,
                             cudaStream_t stream) {
  int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  vector_scale_f32_kernel<<<grid, block_size, 0, stream>>>(x, scale, n);
  return cudaGetLastError();
}

int launch_cuda_swiglu(float *out, const float *gate, const float *up, int n,
                       cudaStream_t stream) {
  int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  swiglu_f32_kernel<<<grid, block_size, 0, stream>>>(out, gate, up, n);
  return cudaGetLastError();
}

int launch_cuda_fused_swiglu_q8_1_sparse(void *y_q8_1, float *output_f32,
                                         const float *gate, const float *up,
                                         int n, float sparsity_threshold,
                                         int enable_2_4,
                                         cudaStream_t stream) {
  if (n <= 0)
    return 0;
  int block_size = 256;
  int grid = (n + block_size - 1) / block_size;
  fused_swiglu_q8_1_cuda_kernel<<<grid, block_size, 0, stream>>>(
      (BlockQ8_1_Gpu *)y_q8_1, output_f32, gate, up, n, sparsity_threshold,
      enable_2_4);
  return cudaGetLastError();
}

int launch_cuda_fused_swiglu_q8_1(void *y_q8_1, float *output_f32,
                                  const float *gate, const float *up, int n,
                                  cudaStream_t stream) {
  return launch_cuda_fused_swiglu_q8_1_sparse(y_q8_1, output_f32, gate, up, n,
                                              0.0f, 0, stream);
}

int launch_cuda_rope_single(float *q, float *k, void *k_out, int pos,
                            int num_heads, int num_kv_heads, int head_dim,
                            float theta, int is_fp16, float scaling_factor,
                            int orig_ctx, cudaStream_t stream) {
  int grid = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  int block_size = head_dim / 2;
  rope_single_token_kernel<<<grid, block_size, 0, stream>>>(
      q, k, k_out, pos, num_heads, num_kv_heads, head_dim, theta, is_fp16,
      scaling_factor, orig_ctx);
  return cudaGetLastError();
}

int launch_cuda_flash_decode(float *output, const float *q, const void *k_cache,
                             const void *v_cache, int num_heads,
                             int num_kv_heads, int head_dim, int num_kv_tokens,
                             int start_token, float sm_scale, int is_fp16,
                             cudaStream_t stream) {
  if (num_heads <= 0 || head_dim <= 0 || num_kv_tokens <= 0)
    return 0;
  const int block_size = 32; // 1 warp per block = maximum SM distribution
  int grid = num_heads;
  flash_decode_paged_kernel<<<grid, block_size, 0, stream>>>(
      output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim,
      num_kv_tokens, start_token, sm_scale, is_fp16);
  return cudaGetLastError();
}

int launch_cuda_rope_batch(float *q, float *k, void *k_cache, int start_pos,
                           int batch_size, int num_heads, int num_kv_heads,
                           int head_dim, float theta, int is_fp16,
                           float scaling_factor, int orig_ctx,
                           cudaStream_t stream) {
  if (batch_size <= 0 || head_dim <= 0)
    return 0;
  int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  dim3 grid(max_heads, batch_size);
  int block_size = head_dim / 2;
  rope_batch_tokens_kernel<<<grid, block_size, 0, stream>>>(
      q, k, k_cache, start_pos, batch_size, num_heads, num_kv_heads, head_dim,
      theta, is_fp16, scaling_factor, orig_ctx);
  return cudaGetLastError();
}

int launch_cuda_flash_causal_prefill(float *output, const float *q,
                                     const void *k_cache, const void *v_cache,
                                     int start_pos, int batch_size,
                                     int num_heads, int num_kv_heads,
                                     int head_dim, int kv_cap, float sm_scale,
                                     int is_fp16, cudaStream_t stream) {
  if (batch_size <= 0 || num_heads <= 0 || head_dim <= 0)
    return 0;
  const int block_size = 256; // 8 warps per block = 8 heads per block
  dim3 grid((num_heads + 7) / 8, batch_size);
  flash_causal_batched_prefill_kernel<<<grid, block_size, 0, stream>>>(
      output, q, k_cache, v_cache, start_pos, batch_size, num_heads,
      num_kv_heads, head_dim, kv_cap, sm_scale, is_fp16);
  return cudaGetLastError();
}

int launch_cuda_convert_f32_to_f16(void *dst, const float *src, int n,
                                   cudaStream_t stream) {
  if (!dst || !src || n <= 0)
    return 0;
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  convert_f32_to_f16_kernel<<<grid_size, block_size, 0, stream>>>((__half *)dst,
                                                                  src, n);
  return cudaGetLastError();
}

/**
 * Custom CUDA Kernel: Per-head QK-Norm (RMSNorm per head across head_dim).
 * Applied to Q and K separately before RoPE embedding.
 */
__global__ void qk_norm_f32_kernel(float *__restrict__ qk,
                                   const float *__restrict__ weight,
                                   int num_heads, int head_dim, float eps) {
  int head_idx = blockIdx.x;
  if (head_idx >= num_heads)
    return;

  int tid = threadIdx.x;
  float *head_vec = qk + head_idx * head_dim;

  float sum_sq = 0.0f;
  for (int i = tid; i < head_dim; i += blockDim.x) {
    float val = head_vec[i];
    sum_sq += val * val;
  }

  // Warp reduction
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
  }

  __shared__ float s_inv_rms;
  __shared__ float warp_sums[32];

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
    }
    if (lane_id == 0) {
      s_inv_rms = rsqrtf((b_sum / (float)head_dim) + eps);
    }
  }
  __syncthreads();

  float inv_rms = s_inv_rms;
  for (int i = tid; i < head_dim; i += blockDim.x) {
    head_vec[i] = head_vec[i] * inv_rms * weight[i];
  }
}

/**
 * Fused Per-Head QK-Norm + RoPE + Direct FP16 K-Cache Write Kernel.
 * 1 warp (32 threads) per head, 0 shared memory, 0 __syncthreads().
 * Grid: dim3(num_heads + num_kv_heads, batch_size), Block: 32.
 */
__global__ void fused_qknorm_rope_batch_kernel(
    float *__restrict__ q, float *__restrict__ k,
    const float *__restrict__ q_norm_weight,
    const float *__restrict__ k_norm_weight, void *__restrict__ k_cache,
    int start_pos, int batch_size, int num_heads, int num_kv_heads,
    int head_dim, float eps, float theta, int is_fp16, float scaling_factor,
    int orig_ctx, const int *__restrict__ d_start_pos = nullptr) {
  int total_head = blockIdx.x;
  int b = blockIdx.y;
  if (b >= batch_size || total_head >= (num_heads + num_kv_heads))
    return;

  int lane_id = threadIdx.x;
  bool is_k = (total_head >= num_heads);
  int head_idx = is_k ? (total_head - num_heads) : total_head;
  int stride_heads = is_k ? num_kv_heads : num_heads;
  float *base_ptr = is_k ? k : q;
  if (base_ptr == nullptr)
    return;

  float *head_vec = base_ptr + ((size_t)b * stride_heads + head_idx) * head_dim;
  const float *norm_w = is_k ? k_norm_weight : q_norm_weight;

  float inv_rms = 1.0f;
  if (norm_w != nullptr) {
    float sum_sq = 0.0f;
    for (int i = lane_id; i < head_dim; i += 32) {
      float v = head_vec[i];
      sum_sq += v * v;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);
    }
    inv_rms = rsqrtf((sum_sq / (float)head_dim) + eps);
  }

  int half_dim = head_dim / 2;
  int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
  int token_pos = actual_pos + b;

  for (int d = lane_id; d < half_dim; d += 32) {
    float v0 = head_vec[d];
    float v1 = head_vec[d + half_dim];
    if (norm_w != nullptr) {
      v0 = v0 * inv_rms * norm_w[d];
      v1 = v1 * inv_rms * norm_w[d + half_dim];
    }

    float freq =
        compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
    float angle = (float)token_pos * freq;
    float sin_val, cos_val;
    __sincosf(angle, &sin_val, &cos_val);

    float rot0 = v0 * cos_val - v1 * sin_val;
    float rot1 = v0 * sin_val + v1 * cos_val;

    head_vec[d] = rot0;
    head_vec[d + half_dim] = rot1;

    if (is_k && k_cache != nullptr) {
      size_t cache_base =
          ((size_t)token_pos * num_kv_heads + head_idx) * head_dim;
      if (is_fp16) {
        __half *dst_h = (__half *)k_cache;
        dst_h[cache_base + d] = __float2half(rot0);
        dst_h[cache_base + d + half_dim] = __float2half(rot1);
      } else {
        float *dst_f = (float *)k_cache;
        dst_f[cache_base + d] = rot0;
        dst_f[cache_base + d + half_dim] = rot1;
      }
    }
  }
}

int launch_cuda_fused_qknorm_rope_batch(
    float *q, float *k, const float *q_norm_weight, const float *k_norm_weight,
    void *k_cache, int start_pos, int batch_size, int num_heads,
    int num_kv_heads, int head_dim, float eps, float theta, int is_fp16,
    float scaling_factor, int orig_ctx, cudaStream_t stream) {
  if (batch_size <= 0 || head_dim <= 0 || (num_heads + num_kv_heads) <= 0)
    return 0;
  dim3 grid(num_heads + num_kv_heads, batch_size);
  fused_qknorm_rope_batch_kernel<<<grid, 32, 0, stream>>>(
      q, k, q_norm_weight, k_norm_weight, k_cache, start_pos, batch_size,
      num_heads, num_kv_heads, head_dim, eps, theta, is_fp16, scaling_factor,
      orig_ctx);
  return cudaGetLastError();
}

int launch_cuda_qk_norm(float *qk, const float *weight, int num_heads,
                        int head_dim, float eps, cudaStream_t stream) {
  if (!qk || !weight || num_heads <= 0 || head_dim <= 0)
    return 0;
  int block_size = 128;
  if (head_dim > 128)
    block_size = 256;
  if (head_dim < 128)
    block_size = head_dim;
  qk_norm_f32_kernel<<<num_heads, block_size, 0, stream>>>(
      qk, weight, num_heads, head_dim, eps);
  return cudaGetLastError();
}

int launch_cuda_gelu(float *out, const float *gate, const float *up, int n,
                     cudaStream_t stream) {
  if (!out || !gate || !up || n <= 0)
    return 0;
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  gelu_f32_kernel<<<grid_size, block_size, 0, stream>>>(out, gate, up, n);
  return cudaGetLastError();
}

int launch_cuda_logit_softcap(float *logits, int vocab_size, float cap,
                              cudaStream_t stream) {
  if (!logits || vocab_size <= 0 || cap <= 0.0f)
    return 0;
  int block_size = 256;
  int grid_size = (vocab_size + block_size - 1) / block_size;
  logit_softcap_f32_kernel<<<grid_size, block_size, 0, stream>>>(
      logits, vocab_size, cap);
  return cudaGetLastError();
}

/**
 * Custom CUDA Kernel: Single-pass Vocab Argmax Reduction
 * Computes argmax token index and maximum logit value directly on GPU.
 * Bypasses 1MB host-device memory transfers on every decode step.
 */
__global__ void argmax_f32_kernel(const float *__restrict__ logits,
                                  int vocab_size,
                                  uint32_t *__restrict__ out_token,
                                  float *__restrict__ out_max_val) {
  int token_idx = blockIdx.x;
  const float *token_logits = logits + (size_t)token_idx * vocab_size;

  __shared__ float s_max_val[32];
  __shared__ uint32_t s_max_idx[32];

  int tid = threadIdx.x;
  float max_v = -1e30f;
  uint32_t max_i = 0;

  // Thread stride across vocab
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    float v = token_logits[i];
    if (v > max_v) {
      max_v = v;
      max_i = (uint32_t)i;
    }
  }

// Warp-level reduction across 32 threads
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
    uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, max_i, offset);
    if (other_v > max_v) {
      max_v = other_v;
      max_i = other_i;
    }
  }

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    s_max_val[warp_id] = max_v;
    s_max_idx[warp_id] = max_i;
  }
  __syncthreads();

  // First warp reduces the 32 warp maximums
  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    float b_max_v = (lane_id < num_warps) ? s_max_val[lane_id] : -1e30f;
    uint32_t b_max_i = (lane_id < num_warps) ? s_max_idx[lane_id] : 0;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_v = __shfl_down_sync(0xFFFFFFFF, b_max_v, offset);
      uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, b_max_i, offset);
      if (other_v > b_max_v) {
        b_max_v = other_v;
        b_max_i = other_i;
      }
    }

    if (lane_id == 0) {
      if (out_token)
        out_token[token_idx] = b_max_i;
      if (out_max_val)
        out_max_val[token_idx] = b_max_v;
    }
  }
}

__device__ float g_argmax_scratch_val[32];
__device__ uint32_t g_argmax_scratch_idx[32];

__global__ void argmax_stage1_multiblock_kernel(const float *__restrict__ logits,
                                                int vocab_size) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  __shared__ float s_max_val[32];
  __shared__ uint32_t s_max_idx[32];

  float max_v = -1e30f;
  uint32_t max_i = 0;

  for (int i = bid * blockDim.x + tid; i < vocab_size; i += total_threads) {
    float v = logits[i];
    if (v > max_v) {
      max_v = v;
      max_i = (uint32_t)i;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
    uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, max_i, offset);
    if (other_v > max_v) {
      max_v = other_v;
      max_i = other_i;
    }
  }

  int warp_id = tid / 32;
  int lane_id = tid % 32;

  if (lane_id == 0) {
    s_max_val[warp_id] = max_v;
    s_max_idx[warp_id] = max_i;
  }
  __syncthreads();

  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    float b_max_v = (lane_id < num_warps) ? s_max_val[lane_id] : -1e30f;
    uint32_t b_max_i = (lane_id < num_warps) ? s_max_idx[lane_id] : 0;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_v = __shfl_down_sync(0xFFFFFFFF, b_max_v, offset);
      uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, b_max_i, offset);
      if (other_v > b_max_v) {
        b_max_v = other_v;
        b_max_i = other_i;
      }
    }

    if (lane_id == 0) {
      g_argmax_scratch_val[bid] = b_max_v;
      g_argmax_scratch_idx[bid] = b_max_i;
    }
  }
}

__global__ void argmax_stage2_multiblock_kernel(uint32_t *__restrict__ out_token,
                                                float *__restrict__ out_max_val,
                                                int num_blocks) {
  int lane_id = threadIdx.x;
  float max_v = (lane_id < num_blocks) ? g_argmax_scratch_val[lane_id] : -1e30f;
  uint32_t max_i = (lane_id < num_blocks) ? g_argmax_scratch_idx[lane_id] : 0;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
    uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, max_i, offset);
    if (other_v > max_v) {
      max_v = other_v;
      max_i = other_i;
    }
  }

  if (lane_id == 0) {
    if (out_token)
      out_token[0] = max_i;
    if (out_max_val)
      out_max_val[0] = max_v;
  }
}

int launch_cuda_argmax(const float *logits, int vocab_size, uint32_t *out_token,
                       float *out_max_val, int batch_size,
                       cudaStream_t stream) {
  if (!logits || vocab_size <= 0 || !out_token || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    int num_blocks = 24;
    argmax_stage1_multiblock_kernel<<<num_blocks, 256, 0, stream>>>(logits, vocab_size);
    argmax_stage2_multiblock_kernel<<<1, 32, 0, stream>>>(out_token, out_max_val, num_blocks);
    return cudaGetLastError();
  }
  int block_size = 1024;
  argmax_f32_kernel<<<batch_size, block_size, 0, stream>>>(
      logits, vocab_size, out_token, out_max_val);
  return cudaGetLastError();
}

/**
 * On-Device Q4_K Token Embedding Dequantization Kernel.
 * Dequantizes 1 row of Q4_K token embedding directly to d_hidden in registers,
 * completely eliminating host-side mmap reads and host-to-device memory copies.
 */
__global__ void dequantize_token_embd_q4_k_cuda_kernel(
    float *__restrict__ d_hidden, const void *__restrict__ raw_embd,
    int tok_idx, int hidden_dim, float scale,
    const int *__restrict__ d_tok_idx = nullptr) {
  int actual_tok_idx = (d_tok_idx != nullptr) ? *d_tok_idx : tok_idx;
  int num_blocks_per_row = hidden_dim / 256;
  int b = blockIdx.x;
  if (b >= num_blocks_per_row)
    return;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_embd +
                              (size_t)actual_tok_idx * num_blocks_per_row * 144);
  const BlockQ4_K_Gpu *blk = &row_blocks[b];

  int lane_id = threadIdx.x % 32;
  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  uint4 hdr = ((const uint4 *)blk)[0];
  float d = __low2float(*((const half2 *)&hdr.x)) * scale;
  float dmin = __high2float(*((const half2 *)&hdr.x)) * scale;

  float sc0, min0, sc1, min1;
  get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

  float d_sc0 = d * sc0;
  float d_sc1 = d * sc1;
  float dm_min0 = dmin * min0;
  float dm_min1 = dmin * min1;

  int v = ((const int *)blk->qs)[lane_id];
  int v0 = v & 0x0F0F0F0F;
  int v1 = (v >> 4) & 0x0F0F0F0F;

  int8_t *q0_bytes = (int8_t *)&v0;
  int8_t *q1_bytes = (int8_t *)&v1;

  #pragma unroll
  for (int i = 0; i < 4; i++) {
    d_hidden[b * 256 + sub0 * 32 + sub_idx * 4 + i] = (float)q0_bytes[i] * d_sc0 - dm_min0;
    d_hidden[b * 256 + sub1 * 32 + sub_idx * 4 + i] = (float)q1_bytes[i] * d_sc1 - dm_min1;
  }
}

int launch_cuda_dequantize_token_embd_q4_k(float *d_hidden,
                                           const void *d_token_embd,
                                           int tok_idx, int hidden_dim,
                                           float scale, cudaStream_t stream) {
  if (!d_hidden || !d_token_embd || hidden_dim <= 0)
    return 0;
  int num_blocks = hidden_dim / 256;
  dequantize_token_embd_q4_k_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
      d_hidden, d_token_embd, tok_idx, hidden_dim, scale);
  return cudaGetLastError();
}

int launch_cuda_dequantize_token_embd_q4_k_ptr(float *d_hidden,
                                               const void *d_token_embd,
                                               const int *d_tok_idx, int hidden_dim,
                                               float scale, cudaStream_t stream) {
  if (!d_hidden || !d_token_embd || hidden_dim <= 0)
    return 0;
  int num_blocks = hidden_dim / 256;
  dequantize_token_embd_q4_k_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
      d_hidden, d_token_embd, 0, hidden_dim, scale, d_tok_idx);
  return cudaGetLastError();
}

int launch_cuda_mmvq_qkv_mixed_ptr(float *y_q, float *y_k, float *y_v,
                                   void *v_cache_fp16, const int *d_start_pos,
                                   const void *raw_wq, const void *raw_wk,
                                   const void *raw_wv, const void *x_q8_1, int q_m,
                                   int kv_m, int k, int v_quant,
                                   cudaStream_t stream) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0)
    return 0;
  int total_m = q_m + 2 * kv_m;
  int grid, block;
  get_gemv_launch_dims(total_m, grid, block);

  if (v_quant == 1) {
    gemv_qkv_mixed_pure_warp_cuda_kernel<1><<<grid, block, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, 0, raw_wq, raw_wk, raw_wv,
        x_q8_1, q_m, kv_m, k, d_start_pos);
  } else {
    gemv_qkv_mixed_pure_warp_cuda_kernel<2><<<grid, block, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, 0, raw_wq, raw_wk, raw_wv,
        x_q8_1, q_m, kv_m, k, d_start_pos);
  }
  return cudaGetLastError();
}

int launch_cuda_fused_qknorm_rope_batch_ptr(
    float *q, float *k, const float *q_norm_weight, const float *k_norm_weight,
    void *k_cache, const int *d_start_pos, int num_heads,
    int num_kv_heads, int head_dim, float eps, float theta, int is_fp16,
    float scaling_factor, int orig_ctx, cudaStream_t stream) {
  if (head_dim <= 0 || (num_heads + num_kv_heads) <= 0)
    return 0;
  dim3 grid(num_heads + num_kv_heads, 1);
  fused_qknorm_rope_batch_kernel<<<grid, 32, 0, stream>>>(
      q, k, q_norm_weight, k_norm_weight, k_cache, 0, 1,
      num_heads, num_kv_heads, head_dim, eps, theta, is_fp16, scaling_factor,
      orig_ctx, d_start_pos);
  return cudaGetLastError();
}

int launch_cuda_flash_decode_ptr(float *output, const float *q, const void *k_cache,
                                const void *v_cache, int num_heads,
                                int num_kv_heads, int head_dim, const int *d_pos,
                                int start_token, float sm_scale, int is_fp16,
                                cudaStream_t stream) {
  if (num_heads <= 0 || head_dim <= 0)
    return 0;
  const int block_size = 32;
  int grid = num_heads;
  flash_decode_paged_kernel<<<grid, block_size, 0, stream>>>(
      output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim,
      0, start_token, sm_scale, is_fp16, d_pos);
  return cudaGetLastError();
}

__global__ void advance_decode_step_kernel(int *__restrict__ d_seq_pos,
                                           const uint32_t *__restrict__ d_sampled_token,
                                           int *__restrict__ d_input_token) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (d_seq_pos != nullptr) {
      *d_seq_pos += 1;
    }
    if (d_sampled_token != nullptr && d_input_token != nullptr) {
      *d_input_token = (int)(*d_sampled_token);
    }
  }
}

int launch_cuda_advance_decode_step(int *d_seq_pos,
                                    const uint32_t *d_sampled_token,
                                    int *d_input_token,
                                    cudaStream_t stream) {
  advance_decode_step_kernel<<<1, 1, 0, stream>>>(d_seq_pos, d_sampled_token, d_input_token);
  return cudaGetLastError();
}

__global__ void
rope_continuous_batch_kernel(float *__restrict__ q, float *__restrict__ k,
                             void *__restrict__ k_cache,
                             const int *__restrict__ d_seq_positions,
                             const int *__restrict__ d_slot_mapping,
                             const int *__restrict__ d_block_table,
                             int max_blocks_per_seq,
                             int batch_size, int num_heads, int num_kv_heads,
                             int head_dim, int slot_stride_tokens, float theta,
                             int is_fp16, float scaling_factor,
                             int orig_ctx) {
  int head_idx = blockIdx.x;
  int b = blockIdx.y;
  if (b >= batch_size)
    return;

  int half_dim = head_dim / 2;
  int d = threadIdx.x;
  if (d >= half_dim)
    return;

  int token_pos = d_seq_positions[b];
  float freq =
      compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
  float angle = (float)token_pos * freq;
  float sin_val, cos_val;
  __sincosf(angle, &sin_val, &cos_val);

  if (head_idx < num_heads && q != nullptr) {
    int q_base = b * (num_heads * head_dim) + head_idx * head_dim;
    float q0 = q[q_base + d];
    float q1 = q[q_base + d + half_dim];
    q[q_base + d] = q0 * cos_val - q1 * sin_val;
    q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;
  }

  if (head_idx < num_kv_heads && k != nullptr) {
    int k_base = b * (num_kv_heads * head_dim) + head_idx * head_dim;
    float k0 = k[k_base + d];
    float k1 = k[k_base + d + half_dim];
    float rot0 = k0 * cos_val - k1 * sin_val;
    float rot1 = k0 * sin_val + k1 * cos_val;
    k[k_base + d] = rot0;
    k[k_base + d + half_dim] = rot1;

    if (k_cache != nullptr) {
      size_t cache_base;
      if (d_block_table != nullptr) {
        int block_num = token_pos / 16;
        int in_block_t = token_pos % 16;
        int phys_block = d_block_table[b * max_blocks_per_seq + block_num];
        cache_base = (size_t)(phys_block * 16 + in_block_t) *
                         (num_kv_heads * head_dim) +
                     head_idx * head_dim;
      } else {
        int slot_idx = d_slot_mapping ? d_slot_mapping[b] : b;
        cache_base = (size_t)(slot_idx * slot_stride_tokens + token_pos) *
                         (num_kv_heads * head_dim) +
                     head_idx * head_dim;
      }
      if (is_fp16) {
        __half *dst_h = (__half *)k_cache;
        dst_h[cache_base + d] = __float2half(rot0);
        dst_h[cache_base + d + half_dim] = __float2half(rot1);
      } else {
        float *dst_f = (float *)k_cache;
        dst_f[cache_base + d] = rot0;
        dst_f[cache_base + d + half_dim] = rot1;
      }
    }
  }
}

__global__ void
write_v_cache_continuous_batch_kernel(void *__restrict__ v_cache,
                                      const float *__restrict__ v_src,
                                      const int *__restrict__ d_seq_positions,
                                      const int *__restrict__ d_slot_mapping,
                                      const int *__restrict__ d_block_table,
                                      int max_blocks_per_seq,
                                      int batch_size, int kv_dim,
                                      int slot_stride_tokens, int is_fp16) {
  int b = blockIdx.y;
  if (b >= batch_size)
    return;
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= kv_dim)
    return;

  int token_pos = d_seq_positions[b];
  size_t cache_base;
  if (d_block_table != nullptr) {
    int block_num = token_pos / 16;
    int in_block_t = token_pos % 16;
    int phys_block = d_block_table[b * max_blocks_per_seq + block_num];
    cache_base = (size_t)(phys_block * 16 + in_block_t) * kv_dim + d;
  } else {
    int slot_idx = d_slot_mapping ? d_slot_mapping[b] : b;
    cache_base =
        (size_t)(slot_idx * slot_stride_tokens + token_pos) * kv_dim + d;
  }
  float val = v_src[b * kv_dim + d];

  if (is_fp16) {
    ((__half *)v_cache)[cache_base] = __float2half(val);
  } else {
    ((float *)v_cache)[cache_base] = val;
  }
}

__global__ void split_k_flash_decode_continuous_batch_kernel(
    float *__restrict__ partial_out,
    float2 *__restrict__ partial_lse,
    float *__restrict__ final_out,
    const float *__restrict__ q,
    const void *__restrict__ k_cache, const void *__restrict__ v_cache,
    const int *__restrict__ d_seq_positions,
    const int *__restrict__ d_slot_mapping,
    const int *__restrict__ d_block_table,
    int max_blocks_per_seq,
    int batch_size,
    int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float sm_scale, int is_fp16,
    int num_splits,
    float *__restrict__ d_block_scores) {
  int head_idx = blockIdx.x;
  int b = blockIdx.y;
  int split_idx = blockIdx.z;
  int lane_id = threadIdx.x % 32;

  int actual_tokens = d_seq_positions[b] + 1;
  if (actual_tokens <= 0 || head_dim <= 0)
    return;

  int tokens_per_split = (actual_tokens + num_splits - 1) / num_splits;
  int t_start = split_idx * tokens_per_split;
  int t_end = (split_idx + 1) * tokens_per_split;
  if (t_end > actual_tokens) {
    t_end = actual_tokens;
  }

  size_t split_lse_idx = (size_t)(split_idx * batch_size + b) * num_heads + head_idx;
  float *split_part_ptr = partial_out ? (partial_out + split_lse_idx * head_dim) : nullptr;

  if (t_start >= actual_tokens) {
    if (num_splits > 1 && partial_lse != nullptr) {
      partial_lse[split_lse_idx] = make_float2(-1e20f, 0.0f);
      if (split_part_ptr != nullptr) {
        for (int d = lane_id; d < head_dim; d += 32) {
          split_part_ptr[d] = 0.0f;
        }
      }
    }
    return;
  }

  const float *q_ptr = q + (b * num_heads + head_idx) * head_dim;
  float *out_ptr = (num_splits == 1) ? (final_out + (b * num_heads + head_idx) * head_dim) : split_part_ptr;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    acc_v[i] = 0.0f;

  float q_reg[16];
  int q_items = 0;
  for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
    q_reg[q_items] = q_ptr[d];
  }

  int slot_idx = d_slot_mapping ? d_slot_mapping[b] : b;
  size_t slot_base_offset =
      (size_t)slot_idx * slot_stride_tokens * kv_heads * head_dim;

  int current_accum_block = -1;
  float current_block_beta_sum = 0.0f;

  for (int t = t_start; t < t_end; t++) {
    size_t tok_offset;
    if (d_block_table != nullptr) {
      int block_num = t / 16;
      int in_block_t = t % 16;
      int phys_block = d_block_table[b * max_blocks_per_seq + block_num];
      tok_offset = (size_t)(phys_block * 16 + in_block_t) * (kv_heads * head_dim) +
                   (size_t)kv_head_idx * head_dim;
    } else {
      tok_offset =
          slot_base_offset + (size_t)(t * kv_heads + kv_head_idx) * head_dim;
    }

    float score = 0.0f;
    if (is_fp16) {
      const __half *k_tok = (const __half *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * __half2float(k_tok[lane_id + i * 32]);
      }
    } else {
      const float *k_tok = (const float *)k_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        score += q_reg[i] * k_tok[lane_id + i * 32];
      }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFF, score, offset);
    }

    float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;
    float new_max = fmaxf(max_score, cur_score);
    float alpha = (t == t_start) ? 0.0f
                                 : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);

    if (d_block_scores != nullptr && d_block_table != nullptr) {
      int block_num = t / 16;
      if (lane_id == 0) {
        if (block_num != current_accum_block) {
          if (current_accum_block >= 0 && current_block_beta_sum > 0.0f) {
            atomicAdd(&d_block_scores[(size_t)b * max_blocks_per_seq + current_accum_block], current_block_beta_sum);
          }
          current_accum_block = block_num;
          current_block_beta_sum = beta;
        } else {
          current_block_beta_sum += beta;
        }
      }
    }

    max_score = new_max;
    sum_exp = sum_exp * alpha + beta;

    if (is_fp16) {
      const __half *v_tok = (const __half *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] =
            acc_v[i] * alpha + beta * __half2float(v_tok[lane_id + i * 32]);
      }
    } else {
      const float *v_tok = (const float *)v_cache + tok_offset;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] = acc_v[i] * alpha + beta * v_tok[lane_id + i * 32];
      }
    }
  }

  if (d_block_scores != nullptr && d_block_table != nullptr && lane_id == 0) {
    if (current_accum_block >= 0 && current_block_beta_sum > 0.0f) {
      atomicAdd(&d_block_scores[(size_t)b * max_blocks_per_seq + current_accum_block], current_block_beta_sum);
    }
  }

  if (num_splits == 1) {
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < q_items; i++) {
      out_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
    }
  } else {
    if (partial_lse != nullptr) {
      partial_lse[split_lse_idx] = make_float2(max_score, sum_exp);
    }
    if (out_ptr != nullptr) {
      for (int i = 0; i < q_items; i++) {
        out_ptr[lane_id + i * 32] = acc_v[i];
      }
    }
  }
}

__global__ void split_k_flash_decode_reduce_kernel(
    float *__restrict__ final_out,
    const float *__restrict__ partial_out,
    const float2 *__restrict__ partial_lse,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_splits) {
  int head_idx = blockIdx.x;
  int b = blockIdx.y;
  int lane_id = threadIdx.x % 32;

  float *final_ptr = final_out + (b * num_heads + head_idx) * head_dim;

  float global_max = -1e20f;
  for (int s = 0; s < num_splits; s++) {
    size_t split_lse_idx = (size_t)(s * batch_size + b) * num_heads + head_idx;
    float2 lse = partial_lse[split_lse_idx];
    if (lse.y > 0.0f) {
      global_max = fmaxf(global_max, lse.x);
    }
  }

  float global_sum_exp = 0.0f;
  float acc_v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    acc_v[i] = 0.0f;

  int q_items = 0;
  for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {}

  for (int s = 0; s < num_splits; s++) {
    size_t split_lse_idx = (size_t)(s * batch_size + b) * num_heads + head_idx;
    float2 lse = partial_lse[split_lse_idx];
    if (lse.y > 0.0f) {
      float weight = expf(lse.x - global_max);
      global_sum_exp += lse.y * weight;
      const float *part_ptr = partial_out + split_lse_idx * head_dim;
      for (int i = 0; i < q_items; i++) {
        acc_v[i] += part_ptr[lane_id + i * 32] * weight;
      }
    }
  }

  float inv_sum = global_sum_exp > 0.0f ? (1.0f / global_sum_exp) : 0.0f;
  for (int i = 0; i < q_items; i++) {
    final_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
  }
}

int launch_cuda_rope_paged_continuous_batch(
    float *q, float *k, void *k_cache, const int *d_seq_positions,
    const int *d_slot_mapping, const int *d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads,
    int head_dim, int slot_stride_tokens, float theta, int is_fp16,
    float scaling_factor, int orig_ctx, cudaStream_t stream) {
  if (batch_size <= 0 || head_dim <= 0)
    return 0;
  int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
  dim3 grid(max_heads, batch_size);
  int block_size = head_dim / 2;
  rope_continuous_batch_kernel<<<grid, block_size, 0, stream>>>(
      q, k, k_cache, d_seq_positions, d_slot_mapping, d_block_table,
      max_blocks_per_seq, batch_size, num_heads, num_kv_heads, head_dim,
      slot_stride_tokens, theta, is_fp16, scaling_factor, orig_ctx);
  return cudaGetLastError();
}

int launch_cuda_rope_continuous_batch(
    float *q, float *k, void *k_cache, const int *d_seq_positions,
    const int *d_slot_mapping, int batch_size, int num_heads, int num_kv_heads,
    int head_dim, int slot_stride_tokens, float theta, int is_fp16,
    float scaling_factor, int orig_ctx, cudaStream_t stream) {
  return launch_cuda_rope_paged_continuous_batch(
      q, k, k_cache, d_seq_positions, d_slot_mapping, nullptr, 0, batch_size,
      num_heads, num_kv_heads, head_dim, slot_stride_tokens, theta, is_fp16,
      scaling_factor, orig_ctx, stream);
}

int launch_cuda_write_v_cache_paged_continuous_batch(
    void *v_cache, const float *v_src, const int *d_seq_positions,
    const int *d_slot_mapping, const int *d_block_table, int max_blocks_per_seq,
    int batch_size, int kv_dim, int slot_stride_tokens, int is_fp16,
    cudaStream_t stream) {
  if (batch_size <= 0 || kv_dim <= 0)
    return 0;
  int block_size = 256;
  dim3 grid((kv_dim + block_size - 1) / block_size, batch_size);
  write_v_cache_continuous_batch_kernel<<<grid, block_size, 0, stream>>>(
      v_cache, v_src, d_seq_positions, d_slot_mapping, d_block_table,
      max_blocks_per_seq, batch_size, kv_dim, slot_stride_tokens, is_fp16);
  return cudaGetLastError();
}

int launch_cuda_write_v_cache_continuous_batch(
    void *v_cache, const float *v_src, const int *d_seq_positions,
    const int *d_slot_mapping, int batch_size, int kv_dim,
    int slot_stride_tokens, int is_fp16, cudaStream_t stream) {
  return launch_cuda_write_v_cache_paged_continuous_batch(
      v_cache, v_src, d_seq_positions, d_slot_mapping, nullptr, 0,
      batch_size, kv_dim, slot_stride_tokens, is_fp16, stream);
}

int launch_cuda_split_k_flash_decode_h2o(
    float *output,
    float *partial_out,
    void *partial_lse,
    const float *q, const void *k_cache, const void *v_cache,
    const int *d_seq_positions, const int *d_slot_mapping,
    const int *d_block_table, int max_blocks_per_seq,
    int batch_size,
    int num_heads, int num_kv_heads, int head_dim, int slot_stride_tokens,
    float sm_scale, int is_fp16, int num_splits,
    float *d_block_scores,
    cudaStream_t stream) {
  if (batch_size <= 0 || num_heads <= 0 || head_dim <= 0)
    return 0;

  int splits = num_splits > 0 ? num_splits : 1;
  const int block_size = 32;
  dim3 grid(num_heads, batch_size, splits);

  split_k_flash_decode_continuous_batch_kernel<<<grid, block_size, 0, stream>>>(
      partial_out, (float2 *)partial_lse, output, q, k_cache, v_cache,
      d_seq_positions, d_slot_mapping, d_block_table, max_blocks_per_seq,
      batch_size, num_heads, num_kv_heads, head_dim, slot_stride_tokens,
      sm_scale, is_fp16, splits, d_block_scores);

  if (splits > 1 && partial_out != nullptr && partial_lse != nullptr) {
    dim3 reduce_grid(num_heads, batch_size);
    split_k_flash_decode_reduce_kernel<<<reduce_grid, block_size, 0, stream>>>(
        output, partial_out, (const float2 *)partial_lse, batch_size,
        num_heads, head_dim, splits);
  }

  return cudaGetLastError();
}

int launch_cuda_split_k_flash_decode(
    float *output,
    float *partial_out,
    void *partial_lse,
    const float *q, const void *k_cache, const void *v_cache,
    const int *d_seq_positions, const int *d_slot_mapping,
    const int *d_block_table, int max_blocks_per_seq,
    int batch_size,
    int num_heads, int num_kv_heads, int head_dim, int slot_stride_tokens,
    float sm_scale, int is_fp16, int num_splits, cudaStream_t stream) {
  return launch_cuda_split_k_flash_decode_h2o(
      output, partial_out, partial_lse, q, k_cache, v_cache,
      d_seq_positions, d_slot_mapping, d_block_table, max_blocks_per_seq,
      batch_size, num_heads, num_kv_heads, head_dim, slot_stride_tokens,
      sm_scale, is_fp16, num_splits, nullptr, stream);
}

int launch_cuda_flash_decode_continuous_batch(
    float *output, const float *q, const void *k_cache, const void *v_cache,
    const int *d_seq_positions, const int *d_slot_mapping, int batch_size,
    int num_heads, int num_kv_heads, int head_dim, int slot_stride_tokens,
    float sm_scale, int is_fp16, cudaStream_t stream) {
  return launch_cuda_split_k_flash_decode(
      output, nullptr, nullptr, q, k_cache, v_cache, d_seq_positions,
      d_slot_mapping, nullptr, 0, batch_size, num_heads, num_kv_heads,
      head_dim, slot_stride_tokens, sm_scale, is_fp16, 1, stream);
}

void init_cuda_kernel_attributes() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaFuncSetAttribute(fused_rmsnorm_q8_1_cuda_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(fused_swiglu_q8_1_cuda_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(fused_qknorm_rope_batch_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(gemv_gate_up_q4_k_cuda_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(gemv_qkv_q4_k_cuda_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
  cudaFuncSetAttribute(gemv_qk_q4_k_cuda_kernel,
                       cudaFuncAttributePreferredSharedMemoryCarveout, 0);
}
}
