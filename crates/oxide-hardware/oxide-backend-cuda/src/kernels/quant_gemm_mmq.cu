// Quantized MMQ Batched Prefill Kernels
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
    float sum_b = 0.0f;

#pragma unroll
    for (int half_idx = 0; half_idx < 2; half_idx++) {
      int sub = half_idx * 4 + group;
      const BlockQ8_1_Gpu *bq8 = &bx[sub];
      float d8 = __low2float(bq8->ds);
      if (d8 != 0.0f) {
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
        int u = ((const int *)bq8->qs)[idx];
        int dot = __dp4a(v_signed, u, 0);
        sum_b += (float)(dot * sc) * d8;
      }
    }
    thread_dot += d * sum_b;
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

    int v = ((const int *)blk->qs)[lane_id];
    int v0 = v & 0x0F0F0F0F;
    int v1 = (v >> 4) & 0x0F0F0F0F;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    if (d8_0 != 0.0f) {
      int u0 = ((const int *)bq8_0->qs)[sub_idx];
      int dot0 = __dp4a(v0, u0, 0);
      int sum0 = __dp4a(0x01010101, u0, 0);
      sumf_d += (float)(dot0 * (int)sc0) * d8_0;
      sumf_m += (float)(sum0 * (int)min0) * d8_0;
    }

    if (d8_1 != 0.0f) {
      int u1 = ((const int *)bq8_1->qs)[sub_idx];
      int dot1 = __dp4a(v1, u1, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);
      sumf_d += (float)(dot1 * (int)sc1) * d8_1;
      sumf_m += (float)(sum1 * (int)min1) * d8_1;
    }
    thread_dot += d * sumf_d - dmin * sumf_m;
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
    if (tid == 0)
      s_k_blk_active = 0;
    __syncthreads();

    // 1. Load 64 tokens x 8 sub-blocks of Q8_1 into shared memory
#pragma unroll
    for (int item = tid; item < 64 * 8; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * 64 + t;
      if (actual_tok < batch_size) {
        BlockQ8_1_Gpu val_x =
            x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col + k_blk * 8 +
                   sub];
        s_x[t][sub] = val_x;
        if (__low2float(val_x.ds) != 0.0f) {
          s_k_blk_active = 1;
        }
      } else {
        s_x[t][sub].ds = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 32; ++i)
          s_x[t][sub].qs[i] = 0;
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
        const BlockQ4_K_Gpu *blk =
            &w_blks[(size_t)actual_row * num_k_blocks + k_blk];
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
    int group = lane_id / 4;

#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) {
      float sc0_r0 = 0.0f, min0_r0 = 0.0f, sc1_r0 = 0.0f, min1_r0 = 0.0f;
      float sc0_r1 = 0.0f, min0_r1 = 0.0f, sc1_r1 = 0.0f, min1_r1 = 0.0f;
      if (blk0) {
        uint4 hdr0 = ((const uint4 *)blk0)[0];
        get_scale_min_k4_from_regs(chunk, hdr0.y, hdr0.z, hdr0.w, &sc0_r0,
                                   &min0_r0, &sc1_r0, &min1_r0);
      }
      if (blk1) {
        uint4 hdr1 = ((const uint4 *)blk1)[0];
        get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_r1,
                                   &min0_r1, &sc1_r1, &min1_r1);
      }

#pragma unroll
      for (int sub_in_chunk = 0; sub_in_chunk < 2; ++sub_in_chunk) {
        int sub = 2 * chunk + sub_in_chunk;

        int a0, a1, a2, a3;
        const void *ptr = &s_A[mem_row][sub * 32 + col_byte];
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                     : "l"(ptr));

        float cur_sc_r0 =
            (sub_in_chunk == 0) ? (d_r0 * sc0_r0) : (d_r0 * sc1_r0);
        float cur_min_r0 =
            (sub_in_chunk == 0) ? (dmin_r0 * min0_r0) : (dmin_r0 * min1_r0);
        float cur_sc_r1 =
            (sub_in_chunk == 0) ? (d_r1 * sc0_r1) : (d_r1 * sc1_r1);
        float cur_min_r1 =
            (sub_in_chunk == 0) ? (dmin_r1 * min0_r1) : (dmin_r1 * min1_r1);

#pragma unroll
        for (int tc = 0; tc < 4; ++tc) {
          int tok_base_tc = (w_col * 4 + tc) * 8;
          int cur_t0 = tok_base_tc + t0;
          int cur_t1 = tok_base_tc + t1;

          int b0 = ((const int *)s_x[cur_t0][sub].qs)[group];
          int b1 = ((const int *)s_x[cur_t1][sub].qs)[group];

          int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
          asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
              : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

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
        if (ADD_TO_Y)
          y[actual_t0 * m + actual_r0] += acc[tc][0];
        else
          y[actual_t0 * m + actual_r0] = acc[tc][0];
      }
      if (actual_t1 < batch_size) {
        if (ADD_TO_Y)
          y[actual_t1 * m + actual_r0] += acc[tc][1];
        else
          y[actual_t1 * m + actual_r0] = acc[tc][1];
      }
    }
    if (actual_r1 < m) {
      if (actual_t0 < batch_size) {
        if (ADD_TO_Y)
          y[actual_t0 * m + actual_r1] += acc[tc][2];
        else
          y[actual_t0 * m + actual_r1] = acc[tc][2];
      }
      if (actual_t1 < batch_size) {
        if (ADD_TO_Y)
          y[actual_t1 * m + actual_r1] += acc[tc][3];
        else
          y[actual_t1 * m + actual_r1] = acc[tc][3];
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

  // Stride 36 uint32_t words per row gives 144 bytes = 16-byte aligned uint4
  // access
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
    // 1. Vectorized 128-bit load of 64 rows of W1 and W3 weights (576 uint4
    // vectors)
#pragma unroll
    for (int item = tid; item < 64 * 9; item += total_threads) {
      int r = item / 9;
      int vec_idx = item % 9;
      int actual_row = row_start + r;
      uint4 val1 = make_uint4(0, 0, 0, 0);
      uint4 val3 = make_uint4(0, 0, 0, 0);
      if (actual_row < inter_m) {
        size_t base_idx =
            ((size_t)actual_row * num_k_blocks + (size_t)k_blk) * 9 + vec_idx;
        val1 = w1_u4[base_idx];
        val3 = w3_u4[base_idx];
      }
      s_w1_u4[item] = val1;
      s_w3_u4[item] = val3;
    }

    // 2. Load tokens_per_block of Q8_1 activations with 16-byte guaranteed
    // alignment
    int total_sub_blocks = tokens_per_block * 8;
#pragma unroll
    for (int item = tid; item < total_sub_blocks; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * tokens_per_block + t;
      if (actual_tok < batch_size) {
        const BlockQ8_1_Gpu *blk =
            &x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col +
                    (size_t)k_blk * 8 + sub];
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
        get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_1,
                                   &min0_1, &sc1_1, &min1_1);
        float sc0_3, min0_3, sc1_3, min1_3;
        get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0_3,
                                   &min0_3, &sc1_3, &min1_3);

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
          float term0_1 =
              d_sc0_1 * d8_0[t] * (float)idot0_1[t] - dm_min0_1 * sum8_0[t];
          float term1_1 =
              d_sc1_1 * d8_1[t] * (float)idot1_1[t] - dm_min1_1 * sum8_1[t];
          acc_gate[r][t] += term0_1 + term1_1;

          float term0_3 =
              d_sc0_3 * d8_0[t] * (float)idot0_3[t] - dm_min0_3 * sum8_0[t];
          float term1_3 =
              d_sc1_3 * d8_1[t] * (float)idot1_3[t] - dm_min1_3 * sum8_1[t];
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

__device__ __forceinline__ int unpack_q6_k_signed_word(const BlockQ6_K_Gpu *blk,
                                                       int half_idx, int g,
                                                       int idx) {
  const uint16_t *ql16 =
      (const uint16_t *)&blk->ql[half_idx * 64 + (g & 1) * 32 + idx * 4];
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
        size_t idx = (size_t)actual_row * num_k_blocks * 105 +
                     (size_t)k_blk * 105 + word;
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
        size_t idx = (size_t)actual_tok * num_q8_blocks_per_col * 9 +
                     (size_t)k_blk * 72 + word;
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
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, "
                   "%12, %13};\n"
                   : "=r"(d0_0), "=r"(d0_1), "=r"(d0_2), "=r"(d0_3)
                   : "r"(a0_0), "r"(a1_0), "r"(a2_0), "r"(a3_0), "r"(b0),
                     "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));

      int d1_0, d1_1, d1_2, d1_3;
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                   "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, "
                   "%12, %13};\n"
                   : "=r"(d1_0), "=r"(d1_1), "=r"(d1_2), "=r"(d1_3)
                   : "r"(a0_1), "r"(a1_1), "r"(a2_1), "r"(a3_1), "r"(b0),
                     "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));

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
      if (ADD_TO_Y)
        y[t0 * m + r0] += acc_r0_t0;
      else
        y[t0 * m + r0] = acc_r0_t0;
    }
    if (t1 < batch_size) {
      if (ADD_TO_Y)
        y[t1 * m + r0] += acc_r0_t1;
      else
        y[t1 * m + r0] = acc_r0_t1;
    }
  }
  if (r1 < m) {
    if (t0 < batch_size) {
      if (ADD_TO_Y)
        y[t0 * m + r1] += acc_r1_t0;
      else
        y[t0 * m + r1] = acc_r1_t0;
    }
    if (t1 < batch_size) {
      if (ADD_TO_Y)
        y[t1 * m + r1] += acc_r1_t1;
      else
        y[t1 * m + r1] = acc_r1_t1;
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
    // 1. Vectorized coalesced warp-level unpack of 64 rows of Q6_K weights into
    // shared memory
    int warps_in_block = blockDim.y;
#pragma unroll
    for (int r0 = 0; r0 < 64; r0 += warps_in_block) {
      int r = r0 + warp_id;
      int actual_row = row_start + r;
      if (r < 64 && actual_row < m) {
        const BlockQ6_K_Gpu *blk =
            (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                    ((size_t)actual_row * num_k_blocks +
                                     (size_t)k_blk) *
                                        210);

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

    // 2. Load tokens_per_block of Q8_1 activations with 16-byte guaranteed
    // alignment
    int total_sub_blocks = tokens_per_block * 8;
#pragma unroll
    for (int item = tid; item < total_sub_blocks; item += total_threads) {
      int t = item / 8;
      int sub = item % 8;
      int actual_tok = blockIdx.y * tokens_per_block + t;
      if (actual_tok < batch_size) {
        const BlockQ8_1_Gpu *blk =
            &x_q8_1[(size_t)actual_tok * num_q8_blocks_per_col +
                    (size_t)k_blk * 8 + sub];
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

    // 3. Ultra-fast DP4A compute loop reading unpacked INT8 weights with zero
    // ALU twiddling
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
