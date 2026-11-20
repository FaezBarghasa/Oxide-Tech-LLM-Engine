// Standard Quantized GEMV Kernels
// Quantized GEMV Kernels
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

template <int BATCH_N, bool ADD_TO_Y = false>
__global__ void mmvq_q5_0_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const void *__restrict__ x_q8_1_raw,
                                      int m, int k, int actual_batch,
                                      int col_offset) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 32;

  int effective_col_offset =
      (col_offset < 0) ? (blockIdx.y * BATCH_N) : col_offset;
  int num_cols = (actual_batch > effective_col_offset
                      ? ((actual_batch - effective_col_offset) < BATCH_N
                             ? (actual_batch - effective_col_offset)
                             : BATCH_N)
                      : 0);
  if (num_cols <= 0)
    return;

  const BlockQ5_0_Gpu *row_blocks =
      (const BlockQ5_0_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 22);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  int i = lane_id & 15;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ5_0_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                  ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);
    uint8_t byte = blk->qs[i];
    uint8_t h = (qh >> lane_id) & 1;
    int8_t q = (lane_id < 16)
                   ? (int8_t)(((int)(byte & 0x0F) | ((int)h << 4)) - 16)
                   : (int8_t)(((int)((byte >> 4) & 0x0F) | ((int)h << 4)) - 16);

#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      const BlockQ8_1_Gpu *bq8_1 =
          &x_q8_1[(effective_col_offset + col) * num_blocks_per_row + b];
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
          y[(effective_col_offset + col) * m + row] += val;
        } else {
          y[(effective_col_offset + col) * m + row] = val;
        }
      }
    }
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
__global__ __launch_bounds__(
    256,
    (BATCH_N > 32
         ? 1
         : (BATCH_N > 16
                ? 3
                : 4))) void mmvq_q6_k_cuda_kernel(float *__restrict__ y,
                                                  const void
                                                      *__restrict__ raw_blocks,
                                                  const void
                                                      *__restrict__ x_q8_1_raw,
                                                  int m, int k,
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

  const BlockQ6_K_Gpu *row_blocks =
      valid_row
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
        const uint16_t *ql16 =
            (const uint16_t *)&blk
                ->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
        uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
        const uint16_t *qh16 =
            (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
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
__global__ __launch_bounds__(
    256,
    (BATCH_N > 32
         ? 1
         : (BATCH_N > 16
                ? 3
                : 4))) void mmvq_q4_k_cuda_kernel(float *__restrict__ y,
                                                  const void
                                                      *__restrict__ raw_blocks,
                                                  const void
                                                      *__restrict__ x_q8_1_raw,
                                                  int m, int k,
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

  const BlockQ4_K_Gpu *row_blocks =
      valid_row
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
 * Split-K Multi-Warp Q4_K MMVQ Kernel (BATCH = 1).
 * Splits row reduction across WARPS_PER_ROW to saturate SM vector units and
 * DRAM bandwidth.
 */
template <int WARPS_PER_ROW = 2, int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 4) void mmvq_q4_k_splitk_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW;
  int row_in_block = warp_id / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + row_in_block;
  bool valid_row = (row < m);

  int warp_sub = warp_id % WARPS_PER_ROW;
  int num_blocks_per_row = k / 256;
  int blocks_per_warp =
      (num_blocks_per_row + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
  int b_start = min(warp_sub * blocks_per_warp, num_blocks_per_row);
  int b_end = min(b_start + blocks_per_warp, num_blocks_per_row);

  const BlockQ4_K_Gpu *row_blocks =
      valid_row
          ? (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                    (size_t)row * num_blocks_per_row * 144)
          : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = chunk * 2;
  int sub1 = sub0 + 1;

  float dot = 0.0f;

  if (valid_row) {
    for (int b = b_start; b < b_end; b++) {
      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
      const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
      const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

      float d8_0 = __low2float(__ldg(&bq8_0->ds));
      float d8_1 = __low2float(__ldg(&bq8_1->ds));

      int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
      int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);
      int sum0 = __dp4a(0x01010101, u0, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);

      const BlockQ4_K_Gpu *blk = &row_blocks[b];
      uint4 hdr = ((const uint4 *)blk)[0];
      float d = __low2float(*((const half2 *)&hdr.x));
      float dmin = __high2float(*((const half2 *)&hdr.x));
      float sc0, min0, sc1, min1;
      get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                                 &min1);

      int v = ((const int *)blk->qs)[lane_id];
      int v0 = v & 0x0F0F0F0F;
      int v1 = (v >> 4) & 0x0F0F0F0F;

      int dot0 = __dp4a(v0, u0, 0);
      int dot1 = __dp4a(v1, u1, 0);

      float sumf_d =
          (float)(dot0 * (int)sc0) * d8_0 + (float)(dot1 * (int)sc1) * d8_1;
      float sumf_m =
          (float)(sum0 * (int)min0) * d8_0 + (float)(sum1 * (int)min1) * d8_1;
      dot += d * sumf_d - dmin * sumf_m;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
  }

  if constexpr (WARPS_PER_ROW == 1) {
    if (lane_id == 0 && valid_row) {
      if (ADD_TO_Y) {
        y[row] += dot;
      } else {
        y[row] = dot;
      }
    }
  } else {
    __shared__ float s_dot_q4[8][WARPS_PER_ROW];

    if (lane_id == 0 && valid_row) {
      s_dot_q4[row_in_block][warp_sub] = dot;
    }
    __syncthreads();

    if (warp_sub == 0 && lane_id == 0 && valid_row) {
      float sum = 0.0f;
      for (int w = 0; w < WARPS_PER_ROW; ++w) {
        sum += s_dot_q4[row_in_block][w];
      }
      if (ADD_TO_Y) {
        y[row] += sum;
      } else {
        y[row] = sum;
      }
    }
  }
}

/**
 * Split-K Multi-Warp Q6_K MMVQ Kernel (BATCH = 1).
 * Splits row reduction across WARPS_PER_ROW to saturate SM vector units and
 * DRAM bandwidth.
 */
template <int WARPS_PER_ROW = 2, int ADD_TO_Y = 0>
__global__ __launch_bounds__(256, 4) void mmvq_q6_k_splitk_kernel(
    float *__restrict__ y, const void *__restrict__ raw_blocks,
    const void *__restrict__ x_q8_1_raw, int m, int k) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW;
  int row_in_block = warp_id / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + row_in_block;
  bool valid_row = (row < m);

  int warp_sub = warp_id % WARPS_PER_ROW;
  int num_blocks_per_row = k / 256;
  int blocks_per_warp =
      (num_blocks_per_row + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
  int b_start = min(warp_sub * blocks_per_warp, num_blocks_per_row);
  int b_end = min(b_start + blocks_per_warp, num_blocks_per_row);

  const BlockQ6_K_Gpu *row_blocks =
      valid_row
          ? (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                    (size_t)row * num_blocks_per_row * 210)
          : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int group = lane_id / 8;
  int idx = lane_id % 8;

  float dot = 0.0f;

  if (valid_row) {
    for (int b = b_start; b < b_end; ++b) {
      const BlockQ6_K_Gpu *blk = &row_blocks[b];
      float d = __half2float(*((const half *)&blk->d));
      float sum_b = 0.0f;

#pragma unroll
      for (int half_idx = 0; half_idx < 2; half_idx++) {
        const uint16_t *ql16 =
            (const uint16_t *)&blk
                ->ql[half_idx * 64 + (group & 1) * 32 + idx * 4];
        uint32_t ql4 = (uint32_t)ql16[0] | ((uint32_t)ql16[1] << 16);
        const uint16_t *qh16 =
            (const uint16_t *)&blk->qh[half_idx * 32 + idx * 4];
        uint32_t qh4 = (uint32_t)qh16[0] | ((uint32_t)qh16[1] << 16);

        uint32_t q_low4 = (ql4 >> ((group & 2) * 2)) & 0x0F0F0F0Fu;
        uint32_t q_high4 = ((qh4 >> (group * 2)) & 0x03030303u) << 4;
        int v_signed = (int)__vsubss4(q_low4 | q_high4, 0x20202020);

        int sc = (int)blk->scales[half_idx * 8 + group * 2 + (idx >> 2)];
        int sub = half_idx * 4 + group;

        const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
        const BlockQ8_1_Gpu *bq8 = &bx[sub];
        float d8 = __low2float(__ldg(&bq8->ds));
        int u = __ldg(&((const int *)bq8->qs)[idx]);
        int dot_int = __dp4a(v_signed, u, 0);
        sum_b += (float)(dot_int * sc) * d8;
      }
      dot += d * sum_b;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
  }

  if constexpr (WARPS_PER_ROW == 1) {
    if (lane_id == 0 && valid_row) {
      if (ADD_TO_Y) {
        y[row] += dot;
      } else {
        y[row] = dot;
      }
    }
  } else {
    __shared__ float s_dot_q6[8][WARPS_PER_ROW];

    if (lane_id == 0 && valid_row) {
      s_dot_q6[row_in_block][warp_sub] = dot;
    }
    __syncthreads();

    if (warp_sub == 0 && lane_id == 0 && valid_row) {
      float sum = 0.0f;
      for (int w = 0; w < WARPS_PER_ROW; ++w) {
        sum += s_dot_q6[row_in_block][w];
      }
      if (ADD_TO_Y) {
        y[row] += sum;
      } else {
        y[row] = sum;
      }
    }
  }
}

/**
 * Grouped Multi-Column Q4_K Gate + Up MMVQ Kernel.
 * Computes both Gate (w1) and Up (w3) projections simultaneously in a single
 * warp pass, reading activations from shared memory ONCE for both projections.
 */
template <int BATCH_N, int WARPS_PER_ROW = 1>
__global__ __launch_bounds__(
    256,
    (BATCH_N > 32
         ? 1
         : (BATCH_N > 16
                ? 3
                : 4))) void mmvq_gate_up_q4_k_cuda_kernel(float
                                                              *__restrict__ y_gate,
                                                          float
                                                              *__restrict__ y_up,
                                                          const void
                                                              *__restrict__ raw_w1,
                                                          const void
                                                              *__restrict__ raw_w3,
                                                          const void
                                                              *__restrict__ x_q8_1_raw,
                                                          int inter_m, int k,
                                                          int actual_batch =
                                                              BATCH_N) {
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

  const BlockQ4_K_Gpu *row_w1 =
      valid_row
          ? (const BlockQ4_K_Gpu *)((const uint8_t *)raw_w1 +
                                    (size_t)row * num_blocks_per_row * 144)
          : nullptr;
  const BlockQ4_K_Gpu *row_w3 =
      valid_row
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
      get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_1, &min0_1,
                                 &sc1_1, &min1_1);

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
      get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0_3, &min0_3,
                                 &sc1_3, &min1_3);

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
