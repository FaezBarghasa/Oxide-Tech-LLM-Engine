// Flash Attention and Flash Decoding Kernels
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
  int actual_tokens =
      (d_num_kv_tokens != nullptr) ? (*d_num_kv_tokens + 1) : num_kv_tokens;
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
 * Vectorized Unroll-2 Flash Decode Kernel for FP16 KV cache (head_dim = 128)
 * 8-byte uint2 vectorized loads, 2 tokens per loop iteration, warp-synchronous
 * online softmax.
 */
__global__ void flash_decode_vectorized_unroll2_kernel(
    float *__restrict__ output, BlockQ8_1_Gpu *__restrict__ out_q8_1,
    const float *__restrict__ q, const void *__restrict__ k_cache,
    const void *__restrict__ v_cache, int num_heads, int num_kv_heads,
    int head_dim, int num_kv_tokens, int start_token, float sm_scale,
    const int *__restrict__ d_num_kv_tokens = nullptr) {

  int actual_tokens =
      (d_num_kv_tokens != nullptr) ? (*d_num_kv_tokens + 1) : num_kv_tokens;
  if (actual_tokens <= start_token)
    return;
  int head_idx = blockIdx.x;
  if (head_idx >= num_heads)
    return;
  int lane_id = threadIdx.x;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  int d_base = lane_id * 4;
  float q0 = q[head_idx * head_dim + d_base + 0];
  float q1 = q[head_idx * head_dim + d_base + 1];
  float q2 = q[head_idx * head_dim + d_base + 2];
  float q3 = q[head_idx * head_dim + d_base + 3];

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v0 = 0.0f, acc_v1 = 0.0f, acc_v2 = 0.0f, acc_v3 = 0.0f;

  int t = start_token;
  for (; t + 1 < actual_tokens; t += 2) {
    size_t off0 = (size_t)((t + 0) * kv_heads + kv_head_idx) * head_dim;
    size_t off1 = (size_t)((t + 1) * kv_heads + kv_head_idx) * head_dim;

    const uint2 *k_ptr0 = (const uint2 *)((const __half *)k_cache + off0);
    const uint2 *k_ptr1 = (const uint2 *)((const __half *)k_cache + off1);
    uint2 k_u2_0 = __ldg(&k_ptr0[lane_id]);
    uint2 k_u2_1 = __ldg(&k_ptr1[lane_id]);
    const __half *kh0 = (const __half *)&k_u2_0;
    const __half *kh1 = (const __half *)&k_u2_1;

    float score0 = q0 * __half2float(kh0[0]) + q1 * __half2float(kh0[1]) +
                   q2 * __half2float(kh0[2]) + q3 * __half2float(kh0[3]);
    float score1 = q0 * __half2float(kh1[0]) + q1 * __half2float(kh1[1]) +
                   q2 * __half2float(kh1[2]) + q3 * __half2float(kh1[3]);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score0 += __shfl_down_sync(0xFFFFFFFF, score0, offset);
      score1 += __shfl_down_sync(0xFFFFFFFF, score1, offset);
    }

    float cur_score0 = __shfl_sync(0xFFFFFFFF, score0, 0) * sm_scale;
    float cur_score1 = __shfl_sync(0xFFFFFFFF, score1, 0) * sm_scale;

    // Token 0
    float new_max0 = fmaxf(max_score, cur_score0);
    float alpha0 =
        (t == start_token)
            ? 0.0f
            : ((new_max0 == max_score) ? 1.0f : expf(max_score - new_max0));
    float beta0 = expf(cur_score0 - new_max0);
    max_score = new_max0;
    sum_exp = sum_exp * alpha0 + beta0;

    const uint2 *v_ptr0 = (const uint2 *)((const __half *)v_cache + off0);
    uint2 v_u2_0 = __ldg(&v_ptr0[lane_id]);
    const __half *vh0 = (const __half *)&v_u2_0;
    acc_v0 = acc_v0 * alpha0 + beta0 * __half2float(vh0[0]);
    acc_v1 = acc_v1 * alpha0 + beta0 * __half2float(vh0[1]);
    acc_v2 = acc_v2 * alpha0 + beta0 * __half2float(vh0[2]);
    acc_v3 = acc_v3 * alpha0 + beta0 * __half2float(vh0[3]);

    // Token 1
    float new_max1 = fmaxf(max_score, cur_score1);
    float alpha1 = (new_max1 == max_score) ? 1.0f : expf(max_score - new_max1);
    float beta1 = expf(cur_score1 - new_max1);
    max_score = new_max1;
    sum_exp = sum_exp * alpha1 + beta1;

    const uint2 *v_ptr1 = (const uint2 *)((const __half *)v_cache + off1);
    uint2 v_u2_1 = __ldg(&v_ptr1[lane_id]);
    const __half *vh1 = (const __half *)&v_u2_1;
    acc_v0 = acc_v0 * alpha1 + beta1 * __half2float(vh1[0]);
    acc_v1 = acc_v1 * alpha1 + beta1 * __half2float(vh1[1]);
    acc_v2 = acc_v2 * alpha1 + beta1 * __half2float(vh1[2]);
    acc_v3 = acc_v3 * alpha1 + beta1 * __half2float(vh1[3]);
  }

  for (; t < actual_tokens; ++t) {
    size_t off = (size_t)(t * kv_heads + kv_head_idx) * head_dim;
    const uint2 *k_ptr = (const uint2 *)((const __half *)k_cache + off);
    uint2 k_u2 = __ldg(&k_ptr[lane_id]);
    const __half *kh = (const __half *)&k_u2;
    float score = q0 * __half2float(kh[0]) + q1 * __half2float(kh[1]) +
                  q2 * __half2float(kh[2]) + q3 * __half2float(kh[3]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFF, score, offset);
    }
    float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;
    float new_max = fmaxf(max_score, cur_score);
    float alpha =
        (t == start_token)
            ? 0.0f
            : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);
    max_score = new_max;
    sum_exp = sum_exp * alpha + beta;
    const uint2 *v_ptr = (const uint2 *)((const __half *)v_cache + off);
    uint2 v_u2 = __ldg(&v_ptr[lane_id]);
    const __half *vh = (const __half *)&v_u2;
    acc_v0 = acc_v0 * alpha + beta * __half2float(vh[0]);
    acc_v1 = acc_v1 * alpha + beta * __half2float(vh[1]);
    acc_v2 = acc_v2 * alpha + beta * __half2float(vh[2]);
    acc_v3 = acc_v3 * alpha + beta * __half2float(vh[3]);
  }

  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
  float v0 = acc_v0 * inv_sum;
  float v1 = acc_v1 * inv_sum;
  float v2 = acc_v2 * inv_sum;
  float v3 = acc_v3 * inv_sum;

  if (output != nullptr) {
    float *out_ptr = output + head_idx * head_dim + d_base;
    out_ptr[0] = v0;
    out_ptr[1] = v1;
    out_ptr[2] = v2;
    out_ptr[3] = v3;
  }

  if (out_q8_1 != nullptr) {
    int sub_block = lane_id / 8;
    int lane_in_sub = lane_id % 8;
    int block_idx = head_idx * 4 + sub_block;

    float local_amax =
        fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float local_sum = v0 + v1 + v2 + v3;

#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      local_amax =
          fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFF, local_amax, offset));
      local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }

    float d = local_amax / 127.0f;
    float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    int8_t q0 = (int8_t)roundf(v0 * id);
    int8_t q1 = (int8_t)roundf(v1 * id);
    int8_t q2 = (int8_t)roundf(v2 * id);
    int8_t q3 = (int8_t)roundf(v3 * id);

    int base_byte = lane_in_sub * 4;
    out_q8_1[block_idx].qs[base_byte + 0] = q0;
    out_q8_1[block_idx].qs[base_byte + 1] = q1;
    out_q8_1[block_idx].qs[base_byte + 2] = q2;
    out_q8_1[block_idx].qs[base_byte + 3] = q3;

    if (lane_in_sub == 0) {
      out_q8_1[block_idx].ds =
          make_half2(__float2half(d), __float2half(local_sum));
    }
  }
}

/**
 * Vectorized Unroll-2 Flash Decode Kernel for FP16 KV cache (head_dim = 64)
 * 2 elements per thread, 2 tokens per loop iteration, warp-synchronous online
 * softmax, and direct GPU register Q8_1 quantization.
 */
__global__ void flash_decode_vectorized_dim64_unroll2_kernel(
    float *__restrict__ output, BlockQ8_1_Gpu *__restrict__ out_q8_1,
    const float *__restrict__ q, const void *__restrict__ k_cache,
    const void *__restrict__ v_cache, int num_heads, int num_kv_heads,
    int head_dim, int num_kv_tokens, int start_token, float sm_scale,
    const int *__restrict__ d_num_kv_tokens = nullptr) {

  int actual_tokens =
      (d_num_kv_tokens != nullptr) ? (*d_num_kv_tokens + 1) : num_kv_tokens;
  if (actual_tokens <= start_token)
    return;
  int head_idx = blockIdx.x;
  if (head_idx >= num_heads)
    return;
  int lane_id = threadIdx.x;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  int d0 = lane_id;
  int d1 = lane_id + 32;
  float q0 = q[head_idx * 64 + d0];
  float q1 = q[head_idx * 64 + d1];

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v0 = 0.0f, acc_v1 = 0.0f;

  int t = start_token;
  for (; t + 1 < actual_tokens; t += 2) {
    size_t off0 = (size_t)((t + 0) * kv_heads + kv_head_idx) * 64;
    size_t off1 = (size_t)((t + 1) * kv_heads + kv_head_idx) * 64;

    const __half *k_tok0 = (const __half *)k_cache + off0;
    const __half *k_tok1 = (const __half *)k_cache + off1;

    float score0 = q0 * __half2float(k_tok0[d0]) + q1 * __half2float(k_tok0[d1]);
    float score1 = q0 * __half2float(k_tok1[d0]) + q1 * __half2float(k_tok1[d1]);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score0 += __shfl_down_sync(0xFFFFFFFF, score0, offset);
      score1 += __shfl_down_sync(0xFFFFFFFF, score1, offset);
    }

    float cur_score0 = __shfl_sync(0xFFFFFFFF, score0, 0) * sm_scale;
    float cur_score1 = __shfl_sync(0xFFFFFFFF, score1, 0) * sm_scale;

    // Token 0
    float new_max0 = fmaxf(max_score, cur_score0);
    float alpha0 =
        (t == start_token)
            ? 0.0f
            : ((new_max0 == max_score) ? 1.0f : expf(max_score - new_max0));
    float beta0 = expf(cur_score0 - new_max0);
    max_score = new_max0;
    sum_exp = sum_exp * alpha0 + beta0;

    const __half *v_tok0 = (const __half *)v_cache + off0;
    acc_v0 = acc_v0 * alpha0 + beta0 * __half2float(v_tok0[d0]);
    acc_v1 = acc_v1 * alpha0 + beta0 * __half2float(v_tok0[d1]);

    // Token 1
    float new_max1 = fmaxf(max_score, cur_score1);
    float alpha1 = (new_max1 == max_score) ? 1.0f : expf(max_score - new_max1);
    float beta1 = expf(cur_score1 - new_max1);
    max_score = new_max1;
    sum_exp = sum_exp * alpha1 + beta1;

    const __half *v_tok1 = (const __half *)v_cache + off1;
    acc_v0 = acc_v0 * alpha1 + beta1 * __half2float(v_tok1[d0]);
    acc_v1 = acc_v1 * alpha1 + beta1 * __half2float(v_tok1[d1]);
  }

  for (; t < actual_tokens; ++t) {
    size_t off = (size_t)(t * kv_heads + kv_head_idx) * 64;
    const __half *k_tok = (const __half *)k_cache + off;
    float score = q0 * __half2float(k_tok[d0]) + q1 * __half2float(k_tok[d1]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      score += __shfl_down_sync(0xFFFFFFFF, score, offset);
    }
    float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;
    float new_max = fmaxf(max_score, cur_score);
    float alpha =
        (t == start_token)
            ? 0.0f
            : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);
    max_score = new_max;
    sum_exp = sum_exp * alpha + beta;
    const __half *v_tok = (const __half *)v_cache + off;
    acc_v0 = acc_v0 * alpha + beta * __half2float(v_tok[d0]);
    acc_v1 = acc_v1 * alpha + beta * __half2float(v_tok[d1]);
  }

  float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
  float v0 = acc_v0 * inv_sum;
  float v1 = acc_v1 * inv_sum;

  if (output != nullptr) {
    output[head_idx * 64 + d0] = v0;
    output[head_idx * 64 + d1] = v1;
  }

  if (out_q8_1 != nullptr) {
    float amax0 = fabsf(v0);
    float sum0 = v0;
    float amax1 = fabsf(v1);
    float sum1 = v1;

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFF, amax0, offset));
      sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
      amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFF, amax1, offset));
      sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
    }

    float scale0 = amax0 / 127.0f;
    float scale1 = amax1 / 127.0f;
    float iscale0 = (scale0 > 0.0f) ? (1.0f / scale0) : 0.0f;
    float iscale1 = (scale1 > 0.0f) ? (1.0f / scale1) : 0.0f;

    int b0 = head_idx * 2 + 0;
    int b1 = head_idx * 2 + 1;
    out_q8_1[b0].qs[lane_id] = (int8_t)roundf(v0 * iscale0);
    out_q8_1[b1].qs[lane_id] = (int8_t)roundf(v1 * iscale1);

    if (lane_id == 0) {
      out_q8_1[b0].ds = make_half2(__float2half(scale0), __float2half(sum0));
      out_q8_1[b1].ds = make_half2(__float2half(scale1), __float2half(sum1));
    }
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
    float alpha =
        (max_score == -1e20f)
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
    sum_exp = sum_exp * alpha +
              (beta0 + beta1 + beta2 + beta3 + beta4 + beta5 + beta6 + beta7);

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
        acc_v[i] =
            acc_v[i] * alpha +
            (beta0 * __half2float(v0[idx]) + beta1 * __half2float(v1[idx]) +
             beta2 * __half2float(v2[idx]) + beta3 * __half2float(v3[idx]) +
             beta4 * __half2float(v4[idx]) + beta5 * __half2float(v5[idx]) +
             beta6 * __half2float(v6[idx]) + beta7 * __half2float(v7[idx]));
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
                   (beta0 * v0[idx] + beta1 * v1[idx] + beta2 * v2[idx] +
                    beta3 * v3[idx] + beta4 * v4[idx] + beta5 * v5[idx] +
                    beta6 * v6[idx] + beta7 * v7[idx]);
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
