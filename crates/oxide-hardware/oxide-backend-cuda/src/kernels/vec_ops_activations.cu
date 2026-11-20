// Vector Operations and Activations
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
  float val;
  if (up != nullptr) {
    float u = up[i];
    float silu_g = g / (1.0f + __expf(-g));
    val = silu_g * u;
  } else {
    val = g;
  }

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
        if (other_abs > abs_val ||
            (other_abs == abs_val && ((lane_id & 3) < k))) {
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
        y_q8_1[block_32_idx].ds =
            make_half2(__float2half(0.0f), __float2half(0.0f));
      }
    } else {
      float d = amax / 127.0f;
      float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
      int8_t q = (int8_t)roundf(val * id);
      y_q8_1[block_32_idx].qs[lane_id] = q;

      if (lane_id == 0) {
        y_q8_1[block_32_idx].ds =
            make_half2(__float2half(d), __float2half(sum));
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

__global__ void convert_f32_to_v_cache_fp16_kernel(
    void *__restrict__ v_cache_fp16, const float *__restrict__ y_v, int kv_m,
    const int *__restrict__ d_start_pos, int start_pos) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < kv_m) {
    int pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
    ((__half *)v_cache_fp16)[(size_t)pos * kv_m + idx] = __float2half(y_v[idx]);
  }
}

