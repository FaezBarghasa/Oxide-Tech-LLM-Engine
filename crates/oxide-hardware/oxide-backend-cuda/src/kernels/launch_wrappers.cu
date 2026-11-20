// C-ABI Launch Wrappers
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

int launch_cuda_mmvq_q5_0(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size == 1) {
    mmvq_q5_0_cuda_kernel<1, false>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1, 0);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q5_0_cuda_kernel<8, false><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size, -1);
  }
  return cudaGetLastError();
}

int launch_cuda_mmvq_q5_0_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size == 1) {
    mmvq_q5_0_cuda_kernel<1, true>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1, 0);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q5_0_cuda_kernel<8, true><<<grid_2d, block, 0, stream>>>(
        y, raw_blocks, x_q8_1, m, k, batch_size, -1);
  }
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
  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid((m + 63) / 64, (batch_size + 15) / 16);
    dim3 block(32, 8);
    mmq_mma_q6_k_cuda_kernel<0>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
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
  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid((m + 63) / 64, (batch_size + 15) / 16);
    dim3 block(32, 8);
    mmq_mma_q6_k_cuda_kernel<1>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
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
  if ((k % 256) == 0) {
    int grid = (m + 3) / 4;
    mmvq_q6_k_splitk_kernel<2, 0>
        <<<grid, 256, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  mmvq_q6_k_cuda_kernel<1, 0, 1>
      <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
  return cudaGetLastError();
}

int launch_cuda_gemv_q6_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  if ((k % 256) == 0) {
    int grid = (m + 3) / 4;
    mmvq_q6_k_splitk_kernel<2, 1>
        <<<grid, 256, 0, stream>>>(y, raw_blocks, x_q8_1, m, k);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  mmvq_q6_k_cuda_kernel<1, 1, 1>
      <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
  return cudaGetLastError();
}

int launch_cuda_mmvq_q4_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, int batch_size, cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    return launch_cuda_gemv_q4_k(y, raw_blocks, x_q8_1, m, k, stream);
  }
  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q4_k_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q4_k_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q4_k_cuda_kernel<8><<<grid_2d, block, 0, stream>>>(
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
  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid((m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<1>
        <<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_q4_k_cuda_kernel<2, 1><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_q4_k_cuda_kernel<4, 1><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_q4_k_cuda_kernel<8, 1><<<grid_2d, block, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_q4_k(float *y, const void *raw_blocks, const void *x_q8_1,
                          int m, int k, cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  if ((k % 256) == 0) {
    int grid = (m + 3) / 4;
    mmvq_q4_k_splitk_kernel<2, 0><<<grid, 256, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  mmvq_q4_k_cuda_kernel<1, 0, 1><<<grid, block, 0, stream>>>(
      y, (const uint8_t *)raw_blocks, x_q8_1, m, k, 1);
  return cudaGetLastError();
}

int launch_cuda_gemv_q4_k_fused_add(float *y, const void *raw_blocks,
                                    const void *x_q8_1, int m, int k,
                                    cudaStream_t stream) {
  if (m <= 0 || k <= 0)
    return 0;
  if ((k % 256) == 0) {
    int grid = (m + 3) / 4;
    mmvq_q4_k_splitk_kernel<2, 1><<<grid, 256, 0, stream>>>(
        y, (const uint8_t *)raw_blocks, x_q8_1, m, k);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(m, grid, block);
  mmvq_q4_k_cuda_kernel<1, 1, 1><<<grid, block, 0, stream>>>(
      y, (const uint8_t *)raw_blocks, x_q8_1, m, k, 1);
  return cudaGetLastError();
}

int launch_cuda_mmvq_gate_up_q4_k(float *y_gate, float *y_up,
                                  const void *raw_w1, const void *raw_w3,
                                  const void *x_q8_1, int inter_m, int k,
                                  int batch_size, cudaStream_t stream) {
  if (inter_m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (batch_size == 1) {
    int grid = (inter_m + 3) / 4;
    mmvq_gate_up_q4_k_2warp_kernel<2><<<grid, 256, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k);
    return cudaGetLastError();
  }
  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid((inter_m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0><<<grid, block, 0, stream>>>(y_gate, raw_w1, x_q8_1,
                                                        inter_m, k, batch_size);
    mmq_q4_k_cuda_kernel<0><<<grid, block, 0, stream>>>(y_up, raw_w3, x_q8_1,
                                                        inter_m, k, batch_size);
    return cudaGetLastError();
  }
  int grid, block;
  get_gemv_launch_dims(inter_m, grid, block);
  if (batch_size <= 2) {
    dim3 grid_2d(grid, (batch_size + 1) / 2);
    mmvq_gate_up_q4_k_cuda_kernel<2><<<grid_2d, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k, batch_size);
  } else if (batch_size <= 4) {
    dim3 grid_2d(grid, (batch_size + 3) / 4);
    mmvq_gate_up_q4_k_cuda_kernel<4><<<grid_2d, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k, batch_size);
  } else {
    dim3 grid_2d(grid, (batch_size + 7) / 8);
    mmvq_gate_up_q4_k_cuda_kernel<8><<<grid_2d, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k, batch_size);
  }
  return cudaGetLastError();
}

int launch_cuda_gemv_gate_up_q4_k(float *y_gate, float *y_up,
                                  const void *raw_w1, const void *raw_w3,
                                  const void *x_q8_1, int inter_m, int k,
                                  cudaStream_t stream) {
  int grid = (inter_m + 3) / 4;
  mmvq_gate_up_q4_k_2warp_kernel<2><<<grid, 256, 0, stream>>>(
      y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k);
  return cudaGetLastError();
}

int launch_cuda_gemv_gate_up_q5_0(float *y_gate, float *y_up,
                                  const void *raw_w1, const void *raw_w3,
                                  const void *x_q8_1, int inter_m, int k,
                                  cudaStream_t stream) {
  int grid = (inter_m + 3) / 4;
  mmvq_gate_up_q5_0_2warp_kernel<2><<<grid, 256, 0, stream>>>(
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

  int total_m = q_m + 2 * kv_m;
  int grid, block;
  get_gemv_launch_dims(total_m, grid, block);

  if (batch_size == 1) {
    if (v_quant == 1) {
      gemv_qkv_mixed_pure_warp_cuda_kernel<1><<<grid, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k);
    } else {
      gemv_qkv_mixed_pure_warp_cuda_kernel<2><<<grid, block, 0, stream>>>(
          y_q, y_k, y_v, v_cache_fp16, start_pos, raw_wq, raw_wk, raw_wv,
          x_q8_1, q_m, kv_m, k);
    }
    return cudaGetLastError();
  }

  if (false && batch_size >= 16 && (k % 256 == 0)) {
    dim3 grid_q((q_m + 63) / 64, (batch_size + 63) / 64);
    dim3 grid_k((kv_m + 63) / 64, (batch_size + 63) / 64);
    dim3 block(32, 8);
    mmq_q4_k_cuda_kernel<0>
        <<<grid_q, block, 0, stream>>>(y_q, raw_wq, x_q8_1, q_m, k, batch_size);
    mmq_q4_k_cuda_kernel<0><<<grid_k, block, 0, stream>>>(y_k, raw_wk, x_q8_1,
                                                          kv_m, k, batch_size);
    if (v_quant == 1) {
      dim3 grid_v((kv_m + 63) / 64, (batch_size + 63) / 64);
      mmq_q4_k_cuda_kernel<0><<<grid_v, block, 0, stream>>>(
          y_v, raw_wv, x_q8_1, kv_m, k, batch_size);
    } else if (v_quant == 2) {
      dim3 grid_v_q6((kv_m + 63) / 64, (batch_size + 15) / 16);
      mmq_mma_q6_k_cuda_kernel<0><<<grid_v_q6, block, 0, stream>>>(
          y_v, raw_wv, x_q8_1, kv_m, k, batch_size);
    }
    if (v_cache_fp16 != nullptr) {
      half *v_cache_dst = ((half *)v_cache_fp16) + (size_t)start_pos * kv_m;
      launch_cuda_convert_f32_to_f16(v_cache_dst, y_v, batch_size * kv_m,
                                     stream);
    }
    return cudaGetLastError();
  }

  if (v_quant == 1) {
    if (batch_size <= 2) {
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
    if (batch_size <= 2) {
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
  vector_bias_add_batch_f32_kernel<<<grid, block_size, 0, stream>>>(a, bias, m,
                                                                    total);
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
                                         int enable_2_4, cudaStream_t stream) {
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

/**
 * Hardware 2:4 Structured Sparsity INT4 Tensor Core Kernel via PTX mma.sp.
 * Computes D = A_sparse * B_dense + C using Ada Lovelace SM 8.9 sparse Tensor
 * Cores. Matrix A: M x K INT4 weights compressed to M x (K/2) with 2:4
 * sparsity. Matrix B: K x N dense INT4 activations (N=8). Matrix meta: M x
 * (K/64) 32-bit metadata registers encoding 2-bit position indices.
 */
