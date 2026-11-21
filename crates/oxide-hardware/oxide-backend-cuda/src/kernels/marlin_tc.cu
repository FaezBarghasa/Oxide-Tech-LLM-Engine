// Marlin and 2:4 Sparse Tensor Core Kernels
__global__ void gemm_sparse_2_4_tensor_core_s4_kernel(
    int32_t *__restrict__ d, const int32_t *__restrict__ a_sparse,
    const int32_t *__restrict__ b_dense,
    const uint32_t *__restrict__ meta_sparse, int m, int k, int n) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_grid = (gridDim.x * blockDim.x) / 32;

  int num_m_tiles = (m + 15) / 16;
  int num_n_tiles = (n + 7) / 8;
  int total_tiles = num_m_tiles * num_n_tiles;
  int k_steps = k / 64;

  for (int tile_idx = warp_id; tile_idx < total_tiles;
       tile_idx += warps_per_grid) {
    int m_tile = tile_idx % num_m_tiles;
    int n_tile = tile_idx / num_m_tiles;
    int m_base = m_tile * 16;
    int n_base = n_tile * 8;

    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    for (int step = 0; step < k_steps; ++step) {
      int a_offset = (m_base * (k / 2) / 8) + step * 64 + lane_id * 2;
      int32_t a0 = a_sparse[a_offset];
      int32_t a1 = a_sparse[a_offset + 1];

      int b_offset = (n_base * k / 8) + step * 64 + lane_id * 2;
      int32_t b0 = b_dense[b_offset];
      int32_t b1 = b_dense[b_offset + 1];

      int meta_offset = (m_base * (k / 64)) + step * 32 + lane_id;
      uint32_t meta = meta_sparse[meta_offset];

      asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32."
                   "s4.s4.s32 "
                   "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11}, "
                   "%12, 0x0;\n"
                   : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                   : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(d0), "r"(d1),
                     "r"(d2), "r"(d3), "r"(meta));
    }

    int d_offset = (m_base * n + n_base) + lane_id * 4;
    d[d_offset + 0] = d0;
    d[d_offset + 1] = d1;
    d[d_offset + 2] = d2;
    d[d_offset + 3] = d3;
  }
}

int launch_cuda_gemm_sparse_2_4_tensor_core_s4(
    int32_t *d, const int32_t *a_sparse, const int32_t *b_dense,
    const uint32_t *meta_sparse, int m, int k, int n, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || n <= 0 || d == nullptr || a_sparse == nullptr ||
      b_dense == nullptr || meta_sparse == nullptr)
    return 0;
  int block_size = 128;
  int num_warps = ((m + 15) / 16) * ((n + 7) / 8);
  int grid_size = (num_warps * 32 + block_size - 1) / block_size;
  gemm_sparse_2_4_tensor_core_s4_kernel<<<grid_size, block_size, 0, stream>>>(
      d, a_sparse, b_dense, meta_sparse, m, k, n);
  return cudaGetLastError();
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

int launch_cuda_flash_decode(float *output, void *out_q8_1, const float *q,
                             const void *k_cache, const void *v_cache,
                             int num_heads, int num_kv_heads, int head_dim,
                             int num_kv_tokens, int start_token, float sm_scale,
                             int is_fp16, cudaStream_t stream) {
  if (num_heads <= 0 || head_dim <= 0 || num_kv_tokens <= 0)
    return 0;
  if (head_dim == 128 && is_fp16) {
    flash_decode_vectorized_unroll2_kernel<<<num_heads, 32, 0, stream>>>(
        output, (BlockQ8_1_Gpu *)out_q8_1, q, k_cache, v_cache, num_heads,
        num_kv_heads, head_dim, num_kv_tokens, start_token, sm_scale, nullptr);
    return cudaGetLastError();
  }
  if (head_dim == 64 && is_fp16) {
    flash_decode_vectorized_dim64_unroll2_kernel<<<num_heads, 32, 0, stream>>>(
        output, (BlockQ8_1_Gpu *)out_q8_1, q, k_cache, v_cache, num_heads,
        num_kv_heads, head_dim, num_kv_tokens, start_token, sm_scale, nullptr);
    return cudaGetLastError();
  }
  const int block_size = 32; // 1 warp per block = maximum SM distribution
  int grid = num_heads;
  flash_decode_paged_kernel<<<grid, block_size, 0, stream>>>(
      output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim,
      num_kv_tokens, start_token, sm_scale, is_fp16);
  if (out_q8_1 != nullptr && output != nullptr) {
    launch_cuda_quantize_q8_1(output, out_q8_1, num_heads * head_dim, stream);
  }
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
