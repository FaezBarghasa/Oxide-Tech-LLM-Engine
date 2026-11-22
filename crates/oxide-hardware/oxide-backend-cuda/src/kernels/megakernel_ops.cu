// Megakernel Continuous Batching and Embedding Kernels
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
                              (size_t)actual_tok_idx * num_blocks_per_row *
                                  144);
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
  get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                             &min1);

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
    d_hidden[b * 256 + sub0 * 32 + sub_idx * 4 + i] =
        (float)q0_bytes[i] * d_sc0 - dm_min0;
    d_hidden[b * 256 + sub1 * 32 + sub_idx * 4 + i] =
        (float)q1_bytes[i] * d_sc1 - dm_min1;
  }
}

__global__ void dequantize_token_embd_q5_0_cuda_kernel(
    float *__restrict__ d_hidden, const void *__restrict__ raw_embd,
    int tok_idx, int hidden_dim, float scale,
    const int *__restrict__ d_tok_idx = nullptr) {
  int actual_tok_idx = (d_tok_idx != nullptr) ? *d_tok_idx : tok_idx;
  int num_blocks_per_row = hidden_dim / 32;
  int b = blockIdx.x;
  if (b >= num_blocks_per_row)
    return;

  const BlockQ5_0_Gpu *row_blocks =
      (const BlockQ5_0_Gpu *)((const uint8_t *)raw_embd +
                              (size_t)actual_tok_idx * num_blocks_per_row * 22);
  const BlockQ5_0_Gpu *blk = &row_blocks[b];

  int lane_id = threadIdx.x % 32;
  float d = __half2float(*((const half *)&blk->d)) * scale;
  uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);

  if (lane_id < 16) {
    uint8_t byte = blk->qs[lane_id];
    uint8_t h0 = (qh >> lane_id) & 1;
    uint8_t h1 = (qh >> (lane_id + 16)) & 1;
    float q0 = (float)(((int)(byte & 0x0F) | ((int)h0 << 4)) - 16);
    float q1 = (float)(((int)((byte >> 4) & 0x0F) | ((int)h1 << 4)) - 16);

    d_hidden[b * 32 + lane_id] = q0 * d;
    d_hidden[b * 32 + lane_id + 16] = q1 * d;
  }
}

__global__ void dequantize_token_embd_q8_0_cuda_kernel(
    float *__restrict__ d_hidden, const void *__restrict__ raw_embd,
    int tok_idx, int hidden_dim, float scale,
    const int *__restrict__ d_tok_idx = nullptr) {
  int actual_tok_idx = (d_tok_idx != nullptr) ? *d_tok_idx : tok_idx;
  int num_blocks_per_row = hidden_dim / 32;
  int b = blockIdx.x;
  if (b >= num_blocks_per_row)
    return;

  const BlockQ8_0_Gpu *row_blocks =
      (const BlockQ8_0_Gpu *)((const uint8_t *)raw_embd +
                              (size_t)actual_tok_idx * num_blocks_per_row * 34);
  const BlockQ8_0_Gpu *blk = &row_blocks[b];

  int lane_id = threadIdx.x % 32;
  float d = __half2float(*((const half *)&blk->scale)) * scale;
  d_hidden[b * 32 + lane_id] = (float)blk->qs[lane_id] * d;
}

__global__ void dequantize_token_embd_fp32_cuda_kernel(
    float *__restrict__ d_hidden, const void *__restrict__ raw_embd,
    int tok_idx, int hidden_dim, float scale,
    const int *__restrict__ d_tok_idx = nullptr) {
  int actual_tok_idx = (d_tok_idx != nullptr) ? *d_tok_idx : tok_idx;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hidden_dim) {
    const float *row =
        ((const float *)raw_embd) + (size_t)actual_tok_idx * hidden_dim;
    d_hidden[idx] = row[idx] * scale;
  }
}

int launch_cuda_dequantize_token_embd(float *d_hidden, const void *d_token_embd,
                                      int tok_idx, int hidden_dim, float scale,
                                      int quant_type, cudaStream_t stream) {
  if (!d_hidden || !d_token_embd || hidden_dim <= 0)
    return 0;
  if (quant_type == 1) { // Q4_K
    int num_blocks = hidden_dim / 256;
    dequantize_token_embd_q4_k_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, tok_idx, hidden_dim, scale);
  } else if (quant_type == 2) { // Q5_0
    int num_blocks = hidden_dim / 32;
    dequantize_token_embd_q5_0_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, tok_idx, hidden_dim, scale);
  } else if (quant_type == 3) { // Q8_0
    int num_blocks = hidden_dim / 32;
    dequantize_token_embd_q8_0_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, tok_idx, hidden_dim, scale);
  } else if (quant_type == 0) { // FP32
    int num_blocks = (hidden_dim + 255) / 256;
    dequantize_token_embd_fp32_cuda_kernel<<<num_blocks, 256, 0, stream>>>(
        d_hidden, d_token_embd, tok_idx, hidden_dim, scale);
  }
  return cudaGetLastError();
}

int launch_cuda_dequantize_token_embd_ptr(float *d_hidden,
                                          const void *d_token_embd,
                                          const int *d_tok_idx, int hidden_dim,
                                          float scale, int quant_type,
                                          cudaStream_t stream) {
  if (!d_hidden || !d_token_embd || hidden_dim <= 0)
    return 0;
  if (quant_type == 1) { // Q4_K
    int num_blocks = hidden_dim / 256;
    dequantize_token_embd_q4_k_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, 0, hidden_dim, scale, d_tok_idx);
  } else if (quant_type == 2) { // Q5_0
    int num_blocks = hidden_dim / 32;
    dequantize_token_embd_q5_0_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, 0, hidden_dim, scale, d_tok_idx);
  } else if (quant_type == 3) { // Q8_0
    int num_blocks = hidden_dim / 32;
    dequantize_token_embd_q8_0_cuda_kernel<<<num_blocks, 32, 0, stream>>>(
        d_hidden, d_token_embd, 0, hidden_dim, scale, d_tok_idx);
  } else if (quant_type == 0) { // FP32
    int num_blocks = (hidden_dim + 255) / 256;
    dequantize_token_embd_fp32_cuda_kernel<<<num_blocks, 256, 0, stream>>>(
        d_hidden, d_token_embd, 0, hidden_dim, scale, d_tok_idx);
  }
  return cudaGetLastError();
}

int launch_cuda_dequantize_token_embd_q4_k(float *d_hidden,
                                           const void *d_token_embd,
                                           int tok_idx, int hidden_dim,
                                           float scale, cudaStream_t stream) {
  return launch_cuda_dequantize_token_embd(d_hidden, d_token_embd, tok_idx,
                                           hidden_dim, scale, 1, stream);
}

int launch_cuda_dequantize_token_embd_q4_k_ptr(float *d_hidden,
                                               const void *d_token_embd,
                                               const int *d_tok_idx,
                                               int hidden_dim, float scale,
                                               cudaStream_t stream) {
  return launch_cuda_dequantize_token_embd_ptr(
      d_hidden, d_token_embd, d_tok_idx, hidden_dim, scale, 1, stream);
}

int launch_cuda_mmvq_qkv_mixed_ptr(float *y_q, float *y_k, float *y_v,
                                   void *v_cache_fp16, const int *d_start_pos,
                                   const void *raw_wq, const void *raw_wk,
                                   const void *raw_wv, const void *x_q8_1,
                                   int q_m, int kv_m, int k, int v_quant,
                                   cudaStream_t stream, const float *q_bias,
                                   const float *k_bias, const float *v_bias) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0)
    return 0;
  int total_m = q_m + 2 * kv_m;
  int grid = (total_m + 3) / 4;

  if (v_quant == 1) {
    gemv_qkv_mixed_splitk_cuda_kernel<2, 1><<<grid, 256, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, 0, raw_wq, raw_wk, raw_wv, x_q8_1, q_m,
        kv_m, k, d_start_pos, q_bias, k_bias, v_bias);
  } else {
    gemv_qkv_mixed_splitk_cuda_kernel<2, 2><<<grid, 256, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, 0, raw_wq, raw_wk, raw_wv, x_q8_1, q_m,
        kv_m, k, d_start_pos, q_bias, k_bias, v_bias);
  }
  return cudaGetLastError();
}

int launch_cuda_mmvq_qkv_q5_0_ptr(float *y_q, float *y_k, float *y_v,
                                  void *v_cache_fp16, int start_pos,
                                  const int *d_start_pos, const void *raw_wq,
                                  const void *raw_wk, const void *raw_wv,
                                  const void *x_q8_1, int q_m, int kv_m, int k,
                                  int v_quant, cudaStream_t stream,
                                  const float *q_bias, const float *k_bias,
                                  const float *v_bias) {
  if (q_m <= 0 || kv_m <= 0 || k <= 0)
    return 0;
  int total_m = q_m + 2 * kv_m;
  int grid = (total_m + 3) / 4;

  if (v_quant == 3) {
    gemv_qkv_q5_0_splitk_cuda_kernel<2, 3><<<grid, 256, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv, x_q8_1,
        q_m, kv_m, k, d_start_pos, q_bias, k_bias, v_bias);
  } else {
    gemv_qkv_q5_0_splitk_cuda_kernel<2, 9><<<grid, 256, 0, stream>>>(
        y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv, x_q8_1,
        q_m, kv_m, k, d_start_pos, q_bias, k_bias, v_bias);
  }
  return cudaGetLastError();
}

__global__ void write_v_cache_ptr_kernel(void *__restrict__ v_cache_fp16,
                                         const float *__restrict__ y_v,
                                         const int *__restrict__ d_start_pos,
                                         int kv_m) {
  int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx2 >= kv_m)
    return;
  int pos = (d_start_pos != nullptr) ? *d_start_pos : 0;
  half2 h2;
  h2.x = __float2half(y_v[idx2]);
  h2.y = (idx2 + 1 < kv_m) ? __float2half(y_v[idx2 + 1]) : __float2half(0.0f);
  ((half2 *)v_cache_fp16)[((size_t)pos * kv_m + idx2) / 2] = h2;
}

int launch_cuda_write_v_cache_ptr(void *v_cache_fp16, const float *y_v,
                                  const int *d_start_pos, int kv_m,
                                  cudaStream_t stream) {
  if (!v_cache_fp16 || !y_v || kv_m <= 0)
    return 0;
  int block = 256;
  int grid = (kv_m + (block * 2) - 1) / (block * 2);
  write_v_cache_ptr_kernel<<<grid, block, 0, stream>>>(v_cache_fp16, y_v,
                                                       d_start_pos, kv_m);
  return cudaGetLastError();
}

__global__ void write_v_cache_kernel(void *__restrict__ v_cache_fp16,
                                     const float *__restrict__ y_v, int pos,
                                     int kv_m) {
  int idx2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  if (idx2 >= kv_m)
    return;
  half2 h2;
  h2.x = __float2half(y_v[idx2]);
  h2.y = (idx2 + 1 < kv_m) ? __float2half(y_v[idx2 + 1]) : __float2half(0.0f);
  ((half2 *)v_cache_fp16)[((size_t)pos * kv_m + idx2) / 2] = h2;
}

int launch_cuda_write_v_cache(void *v_cache_fp16, const float *y_v, int pos,
                              int kv_m, cudaStream_t stream) {
  if (!v_cache_fp16 || !y_v || kv_m <= 0)
    return 0;
  int block = 256;
  int grid = (kv_m + (block * 2) - 1) / (block * 2);
  write_v_cache_kernel<<<grid, block, 0, stream>>>(v_cache_fp16, y_v, pos,
                                                   kv_m);
  return cudaGetLastError();
}

int launch_cuda_fused_qknorm_rope_batch_ptr(
    float *q, float *k, const float *q_norm_weight, const float *k_norm_weight,
    void *k_cache, const int *d_start_pos, int num_heads, int num_kv_heads,
    int head_dim, float eps, float theta, int is_fp16, float scaling_factor,
    int orig_ctx, cudaStream_t stream) {
  if (head_dim <= 0 || (num_heads + num_kv_heads) <= 0)
    return 0;
  dim3 grid(num_heads + num_kv_heads, 1);
  fused_qknorm_rope_batch_kernel<<<grid, 32, 0, stream>>>(
      q, k, q_norm_weight, k_norm_weight, k_cache, 0, 1, num_heads,
      num_kv_heads, head_dim, eps, theta, is_fp16, scaling_factor, orig_ctx,
      d_start_pos);
  return cudaGetLastError();
}

int launch_cuda_flash_decode_ptr(float *output, void *out_q8_1, const float *q,
                                 const void *k_cache, const void *v_cache,
                                 int num_heads, int num_kv_heads, int head_dim,
                                 const int *d_pos, int start_token,
                                 float sm_scale, int is_fp16,
                                 cudaStream_t stream) {
  if (num_heads <= 0 || head_dim <= 0)
    return 0;
  if (head_dim == 128 && is_fp16) {
    flash_decode_vectorized_unroll2_kernel<<<num_heads, 32, 0, stream>>>(
        output, (BlockQ8_1_Gpu *)out_q8_1, q, k_cache, v_cache, num_heads,
        num_kv_heads, head_dim, 0, start_token, sm_scale, d_pos);
    return cudaGetLastError();
  }
  if (head_dim == 64 && is_fp16) {
    // When d_pos is provided (CUDA graphs or online decode), we use Split-K parallel stage1 + stage2
    // with static device workspace to fully saturate 24 SMs on RTX 4060.
    static float *d_split_out = nullptr;
    static float *d_split_meta = nullptr;
    if (d_split_out == nullptr) {
      // 128 max heads * 4 splits * 64 floats = 32,768 floats (128 KB)
      cudaMalloc(&d_split_out, 128 * 4 * 64 * sizeof(float));
      // 128 max heads * 4 splits * 2 floats = 1,024 floats (4 KB)
      cudaMalloc(&d_split_meta, 128 * 4 * 2 * sizeof(float));
    }
    dim3 grid_stage1(num_heads, 4);
    flash_decode_splitk_dim64_stage1_kernel<<<grid_stage1, 32, 0, stream>>>(
        d_split_out, d_split_meta, q, k_cache, v_cache, num_heads,
        num_kv_heads, head_dim, 0, start_token, sm_scale, d_pos);
    flash_decode_splitk_dim64_stage2_kernel<<<num_heads, 32, 0, stream>>>(
        output, (BlockQ8_1_Gpu *)out_q8_1, d_split_out, d_split_meta, num_heads);
    return cudaGetLastError();
  }
  const int block_size = 32;
  int grid = num_heads;
  flash_decode_paged_kernel<<<grid, block_size, 0, stream>>>(
      output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, 0,
      start_token, sm_scale, is_fp16, d_pos);
  if (out_q8_1 != nullptr && output != nullptr) {
    launch_cuda_quantize_q8_1(output, out_q8_1, num_heads * head_dim, stream);
  }
  return cudaGetLastError();
}

__global__ void advance_decode_step_kernel(
    int *__restrict__ d_seq_pos, const uint32_t *__restrict__ d_sampled_token,
    int *__restrict__ d_input_token, uint32_t *__restrict__ h_sampled_token) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (d_seq_pos != nullptr) {
      *d_seq_pos += 1;
    }
    if (d_sampled_token != nullptr && d_input_token != nullptr) {
      *d_input_token = (int)(*d_sampled_token);
    }
    if (d_sampled_token != nullptr && h_sampled_token != nullptr) {
      h_sampled_token[0] = *d_sampled_token;
      __threadfence_system();
      h_sampled_token[1] += 1;
    }
  }
}

int launch_cuda_advance_decode_step(int *d_seq_pos,
                                    const uint32_t *d_sampled_token,
                                    int *d_input_token,
                                    uint32_t *h_sampled_token,
                                    cudaStream_t stream) {
  advance_decode_step_kernel<<<1, 1, 0, stream>>>(
      d_seq_pos, d_sampled_token, d_input_token, h_sampled_token);
  return cudaGetLastError();
}

__global__ void rope_continuous_batch_kernel(
    float *__restrict__ q, float *__restrict__ k, void *__restrict__ k_cache,
    const int *__restrict__ d_seq_positions,
    const int *__restrict__ d_slot_mapping,
    const int *__restrict__ d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float theta, int is_fp16, float scaling_factor,
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
        cache_base =
            (size_t)(phys_block * 16 + in_block_t) * (num_kv_heads * head_dim) +
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

__global__ void write_v_cache_continuous_batch_kernel(
    void *__restrict__ v_cache, const float *__restrict__ v_src,
    const int *__restrict__ d_seq_positions,
    const int *__restrict__ d_slot_mapping,
    const int *__restrict__ d_block_table, int max_blocks_per_seq,
    int batch_size, int kv_dim, int slot_stride_tokens, int is_fp16) {
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
    float *__restrict__ partial_out, float2 *__restrict__ partial_lse,
    float *__restrict__ final_out, const float *__restrict__ q,
    const void *__restrict__ k_cache, const void *__restrict__ v_cache,
    const int *__restrict__ d_seq_positions,
    const int *__restrict__ d_slot_mapping,
    const int *__restrict__ d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float sm_scale, int is_fp16, int num_splits,
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

  size_t split_lse_idx =
      (size_t)(split_idx * batch_size + b) * num_heads + head_idx;
  float *split_part_ptr =
      partial_out ? (partial_out + split_lse_idx * head_dim) : nullptr;

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
  float *out_ptr = (num_splits == 1)
                       ? (final_out + (b * num_heads + head_idx) * head_dim)
                       : split_part_ptr;

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
      tok_offset =
          (size_t)(phys_block * 16 + in_block_t) * (kv_heads * head_dim) +
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
    float alpha =
        (t == t_start)
            ? 0.0f
            : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
    float beta = expf(cur_score - new_max);

    if (d_block_scores != nullptr && d_block_table != nullptr) {
      int block_num = t / 16;
      if (lane_id == 0) {
        if (block_num != current_accum_block) {
          if (current_accum_block >= 0 && current_block_beta_sum > 0.0f) {
            atomicAdd(&d_block_scores[(size_t)b * max_blocks_per_seq +
                                      current_accum_block],
                      current_block_beta_sum);
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
      atomicAdd(
          &d_block_scores[(size_t)b * max_blocks_per_seq + current_accum_block],
          current_block_beta_sum);
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
    float *__restrict__ final_out, const float *__restrict__ partial_out,
    const float2 *__restrict__ partial_lse, int batch_size, int num_heads,
    int head_dim, int num_splits) {
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
  for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
  }

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
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float theta, int is_fp16, float scaling_factor,
    int orig_ctx, cudaStream_t stream) {
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
      v_cache, v_src, d_seq_positions, d_slot_mapping, nullptr, 0, batch_size,
      kv_dim, slot_stride_tokens, is_fp16, stream);
}

__global__ void fused_split_k_flash_decode_continuous_batch_kernel(
    float *__restrict__ final_out, const float *__restrict__ q,
    const void *__restrict__ k_cache, const void *__restrict__ v_cache,
    const int *__restrict__ d_seq_positions,
    const int *__restrict__ d_slot_mapping,
    const int *__restrict__ d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float sm_scale, int is_fp16,
    float *__restrict__ d_block_scores) {
  const int NUM_WARPS = 4;
  int head_idx = blockIdx.x;
  int b = blockIdx.y;
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;

  int actual_tokens = d_seq_positions[b] + 1;
  if (actual_tokens <= 0 || head_dim <= 0)
    return;

  int tokens_per_split = (actual_tokens + NUM_WARPS - 1) / NUM_WARPS;
  int t_start = warp_id * tokens_per_split;
  int t_end = (warp_id + 1) * tokens_per_split;
  if (t_end > actual_tokens) {
    t_end = actual_tokens;
  }

  __shared__ float s_lse_max[NUM_WARPS];
  __shared__ float s_lse_sum[NUM_WARPS];
  __shared__ float s_acc_v[NUM_WARPS][128];

  const float *q_ptr = q + (b * num_heads + head_idx) * head_dim;

  int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
  int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
  int kv_head_idx = head_idx / heads_per_kv;
  if (kv_head_idx >= kv_heads)
    kv_head_idx = kv_heads - 1;

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc_v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    acc_v[i] = 0.0f;

  float q_reg[4];
  int q_items = 0;
  for (int d = lane_id; d < head_dim && q_items < 4; d += 32, q_items++) {
    q_reg[q_items] = q_ptr[d];
  }

  int slot_idx = d_slot_mapping ? d_slot_mapping[b] : b;
  size_t slot_base_offset =
      (size_t)slot_idx * slot_stride_tokens * kv_heads * head_dim;

  int current_accum_block = -1;
  float current_block_beta_sum = 0.0f;

  if (t_start < actual_tokens) {
    for (int t = t_start; t < t_end; t++) {
      size_t tok_offset;
      if (d_block_table != nullptr) {
        int block_num = t / 16;
        int in_block_t = t % 16;
        int phys_block = d_block_table[b * max_blocks_per_seq + block_num];
        tok_offset =
            (size_t)(phys_block * 16 + in_block_t) * (kv_heads * head_dim) +
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
      float alpha =
          (t == t_start)
              ? 0.0f
              : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
      float beta = expf(cur_score - new_max);

      if (d_block_scores != nullptr && d_block_table != nullptr) {
        int block_num = t / 16;
        if (lane_id == 0) {
          if (block_num != current_accum_block) {
            if (current_accum_block >= 0 && current_block_beta_sum > 0.0f) {
              atomicAdd(&d_block_scores[(size_t)b * max_blocks_per_seq +
                                        current_accum_block],
                        current_block_beta_sum);
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
        atomicAdd(
            &d_block_scores[(size_t)b * max_blocks_per_seq + current_accum_block],
            current_block_beta_sum);
      }
    }
  }

  if (lane_id == 0) {
    s_lse_max[warp_id] = max_score;
    s_lse_sum[warp_id] = sum_exp;
  }
  for (int i = 0; i < q_items; i++) {
    s_acc_v[warp_id][lane_id + i * 32] = acc_v[i];
  }

  __syncthreads();

  if (warp_id == 0) {
    float global_max = -1e20f;
#pragma unroll
    for (int s = 0; s < NUM_WARPS; s++) {
      if (s_lse_sum[s] > 0.0f) {
        global_max = fmaxf(global_max, s_lse_max[s]);
      }
    }

    float global_sum_exp = 0.0f;
    float reduced_v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      reduced_v[i] = 0.0f;
    }

#pragma unroll
    for (int s = 0; s < NUM_WARPS; s++) {
      if (s_lse_sum[s] > 0.0f) {
        float weight = expf(s_lse_max[s] - global_max);
        global_sum_exp += s_lse_sum[s] * weight;
        for (int i = 0; i < q_items; i++) {
          reduced_v[i] += s_acc_v[s][lane_id + i * 32] * weight;
        }
      }
    }

    float inv_sum = global_sum_exp > 0.0f ? (1.0f / global_sum_exp) : 0.0f;
    float *final_ptr = final_out + (b * num_heads + head_idx) * head_dim;
    for (int i = 0; i < q_items; i++) {
      final_ptr[lane_id + i * 32] = reduced_v[i] * inv_sum;
    }
  }
}

int launch_cuda_split_k_flash_decode_h2o(
    float *output, float *partial_out, void *partial_lse, const float *q,
    const void *k_cache, const void *v_cache, const int *d_seq_positions,
    const int *d_slot_mapping, const int *d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float sm_scale, int is_fp16, int num_splits,
    float *d_block_scores, cudaStream_t stream) {
  if (batch_size <= 0 || num_heads <= 0 || head_dim <= 0)
    return 0;

  int splits = num_splits > 0 ? num_splits : 1;

  if (splits > 1) {
    dim3 grid(num_heads, batch_size);
    fused_split_k_flash_decode_continuous_batch_kernel<<<grid, 128, 0, stream>>>(
        output, q, k_cache, v_cache, d_seq_positions, d_slot_mapping,
        d_block_table, max_blocks_per_seq, batch_size, num_heads, num_kv_heads,
        head_dim, slot_stride_tokens, sm_scale, is_fp16, d_block_scores);
  } else {
    dim3 grid(num_heads, batch_size, 1);
    split_k_flash_decode_continuous_batch_kernel<<<grid, 32, 0, stream>>>(
        nullptr, nullptr, output, q, k_cache, v_cache, d_seq_positions,
        d_slot_mapping, d_block_table, max_blocks_per_seq, batch_size,
        num_heads, num_kv_heads, head_dim, slot_stride_tokens, sm_scale,
        is_fp16, 1, d_block_scores);
  }

  return cudaGetLastError();
}

int launch_cuda_split_k_flash_decode(
    float *output, float *partial_out, void *partial_lse, const float *q,
    const void *k_cache, const void *v_cache, const int *d_seq_positions,
    const int *d_slot_mapping, const int *d_block_table, int max_blocks_per_seq,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int slot_stride_tokens, float sm_scale, int is_fp16, int num_splits,
    cudaStream_t stream) {
  return launch_cuda_split_k_flash_decode_h2o(
      output, partial_out, partial_lse, q, k_cache, v_cache, d_seq_positions,
      d_slot_mapping, d_block_table, max_blocks_per_seq, batch_size, num_heads,
      num_kv_heads, head_dim, slot_stride_tokens, sm_scale, is_fp16, num_splits,
      nullptr, stream);
}

int launch_cuda_flash_decode_continuous_batch(
    float *output, const float *q, const void *k_cache, const void *v_cache,
    const int *d_seq_positions, const int *d_slot_mapping, int batch_size,
    int num_heads, int num_kv_heads, int head_dim, int slot_stride_tokens,
    float sm_scale, int is_fp16, cudaStream_t stream) {
  return launch_cuda_split_k_flash_decode(
      output, nullptr, nullptr, q, k_cache, v_cache, d_seq_positions,
      d_slot_mapping, nullptr, 0, batch_size, num_heads, num_kv_heads, head_dim,
      slot_stride_tokens, sm_scale, is_fp16, 1, stream);
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
