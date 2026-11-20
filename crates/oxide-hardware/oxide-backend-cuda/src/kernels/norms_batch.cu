// Batch QK-Norm and RoPE Kernels
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
