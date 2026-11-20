// Norms and RoPE Core Kernels
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
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  const float *in_row = input + bid * hidden_dim;
  float *out_row = output ? (output + bid * hidden_dim) : nullptr;
  BlockQ8_1_Gpu *out_q8_1 =
      y_q8_1 ? (y_q8_1 + bid * (hidden_dim / 32)) : nullptr;

  float sum_sq = 0.0f;
  int blocks_total = hidden_dim / 32;
  int warps_total = blockDim.x / 32;

  float val[16];
  int block_count = 0;
  for (int b = warp_id; b < blocks_total && block_count < 16;
       b += warps_total, block_count++) {
    int idx = b * 32 + lane_id;
    float v = in_row[idx];
    val[block_count] = v;
    sum_sq += v * v;
  }

  // Warp reduction using shuffle down
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
  }

  __shared__ float s_mean;
  __shared__ float warp_sums[32];

  if (lane_id == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float b_sum = (lane_id < warps_total) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
    }
    if (lane_id == 0) {
      s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
    }
  }
  __syncthreads();

  float inv_rms = s_mean;

  int b_idx = 0;
  for (int b = warp_id; b < blocks_total && b_idx < block_count;
       b += warps_total, b_idx++) {
    int idx = b * 32 + lane_id;
    float w = weight != nullptr ? (weight[idx] + weight_offset) : 1.0f;
    float norm_val = val[b_idx] * inv_rms * w;
    if (out_row != nullptr) {
      out_row[idx] = norm_val;
    }

    if (out_q8_1 != nullptr) {
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
      out_q8_1[b].qs[lane_id] = q;

      if (lane_id == 0) {
        out_q8_1[b].ds = make_half2(__float2half(d), __float2half(sum));
      }
    }
  }
}

/**
 * Custom CUDA Kernel: Fused Residual Add + RMSNorm + Dynamic Q8_1 Activation Quantization.
 * Adds residual in-place: residual[i] += input[i], then normalizes residual and emits
 * both normalized float (optional) and quantized BlockQ8_1_Gpu.
 * Saves 2 global memory roundtrips per transformer layer.
 */
__global__ void fused_residual_rmsnorm_q8_1_cuda_kernel(
    float *__restrict__ residual, float *__restrict__ output_norm,
    BlockQ8_1_Gpu *__restrict__ y_q8_1, const float *__restrict__ input,
    const float *__restrict__ weight, int hidden_dim, float eps,
    float weight_offset) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  float *res_row = residual + bid * hidden_dim;
  const float *in_row = input ? (input + bid * hidden_dim) : nullptr;
  float *out_row = output_norm ? (output_norm + bid * hidden_dim) : nullptr;
  BlockQ8_1_Gpu *out_q8_1 =
      y_q8_1 ? (y_q8_1 + bid * (hidden_dim / 32)) : nullptr;

  float sum_sq = 0.0f;
  int blocks_total = hidden_dim / 32;
  int warps_total = blockDim.x / 32;

  float val[16];
  int block_count = 0;
  for (int b = warp_id; b < blocks_total && block_count < 16;
       b += warps_total, block_count++) {
    int idx = b * 32 + lane_id;
    float r = res_row[idx];
    if (in_row != nullptr) {
      r += in_row[idx];
      res_row[idx] = r;
    }
    val[block_count] = r;
    sum_sq += r * r;
  }

  // Warp reduction using shuffle down
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
  }

  __shared__ float s_mean;
  __shared__ float warp_sums[32];

  if (lane_id == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  if (warp_id == 0) {
    float b_sum = (lane_id < warps_total) ? warp_sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
    }
    if (lane_id == 0) {
      s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
    }
  }
  __syncthreads();

  float inv_rms = s_mean;

  int b_idx = 0;
  for (int b = warp_id; b < blocks_total && b_idx < block_count;
       b += warps_total, b_idx++) {
    int idx = b * 32 + lane_id;
    float w = weight != nullptr ? (weight[idx] + weight_offset) : 1.0f;
    float norm_val = val[b_idx] * inv_rms * w;
    if (out_row != nullptr) {
      out_row[idx] = norm_val;
    }

    if (out_q8_1 != nullptr) {
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
      out_q8_1[b].qs[lane_id] = q;

      if (lane_id == 0) {
        out_q8_1[b].ds = make_half2(__float2half(d), __float2half(sum));
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
