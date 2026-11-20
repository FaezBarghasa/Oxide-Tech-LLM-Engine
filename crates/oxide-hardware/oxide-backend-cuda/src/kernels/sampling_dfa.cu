// Sampling and DFA Token Selection Kernels
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

__global__ void
argmax_stage1_multiblock_kernel(const float *__restrict__ logits,
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

__global__ void
argmax_stage2_multiblock_kernel(uint32_t *__restrict__ out_token,
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
    argmax_stage1_multiblock_kernel<<<num_blocks, 256, 0, stream>>>(logits,
                                                                    vocab_size);
    argmax_stage2_multiblock_kernel<<<1, 32, 0, stream>>>(
        out_token, out_max_val, num_blocks);
    return cudaGetLastError();
  }
  int block_size = 1024;
  argmax_f32_kernel<<<batch_size, block_size, 0, stream>>>(
      logits, vocab_size, out_token, out_max_val);
  return cudaGetLastError();
}

__global__ void
sample_stage1_multiblock_kernel(const float *__restrict__ logits,
                                int vocab_size, float inv_temp, uint64_t seed,
                                const int *__restrict__ d_seq_pos) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  __shared__ float s_max_val[32];
  __shared__ uint32_t s_max_idx[32];

  float max_v = -1e30f;
  uint32_t max_i = 0;

  uint32_t step = (d_seq_pos != nullptr) ? (uint32_t)(*d_seq_pos) : 0u;

  for (int i = bid * blockDim.x + tid; i < vocab_size; i += total_threads) {
    float v = logits[i] * inv_temp;
    if (inv_temp > 0.0f) {
      uint32_t h =
          (uint32_t)(seed ^ (seed >> 32)) ^ (step * 0x9e3779b9u) ^ (uint32_t)i;
      h ^= h >> 16;
      h *= 0x85ebca6bu;
      h ^= h >> 13;
      h *= 0xc2b2ae35u;
      h ^= h >> 16;
      float u = ((float)(h & 0x00FFFFFF) + 1.0f) / 16777218.0f;
      float g = -__logf(-__logf(u));
      v += g;
    }
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

int launch_cuda_sample_native(const float *logits, int vocab_size,
                              uint32_t *out_token, float *out_max_val,
                              float temperature, float top_p, uint64_t seed,
                              const int *d_seq_pos, cudaStream_t stream) {
  if (!logits || vocab_size <= 0 || !out_token)
    return 0;
  if (temperature <= 0.0f) {
    return launch_cuda_argmax(logits, vocab_size, out_token, out_max_val, 1,
                              stream);
  }
  int num_blocks = 24;
  float inv_temp = 1.0f / temperature;
  sample_stage1_multiblock_kernel<<<num_blocks, 256, 0, stream>>>(
      logits, vocab_size, inv_temp, seed, d_seq_pos);
  argmax_stage2_multiblock_kernel<<<1, 32, 0, stream>>>(out_token, out_max_val,
                                                        num_blocks);
  return cudaGetLastError();
}

/**
 * On-Device Q4_K Token Embedding Dequantization Kernel.
 * Dequantizes 1 row of Q4_K token embedding directly to d_hidden in registers,
 * completely eliminating host-side mmap reads and host-to-device memory copies.
 */
