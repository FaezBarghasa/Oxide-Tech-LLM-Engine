// K-Quant and Multi-Warp GEMV Kernels
      }
    }
  }

  if (valid_row) {
#pragma unroll
    for (int col = 0; col < BATCH_N; ++col) {
      if (BATCH_N == 1 || col < num_cols) {
        float val_gate = dot_gate[col];
        float val_up = dot_up[col];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          val_gate += __shfl_down_sync(0xFFFFFFFF, val_gate, offset);
          val_up += __shfl_down_sync(0xFFFFFFFF, val_up, offset);
        }

        if (lane_id == 0) {
          y_gate[(col_offset + col) * inter_m + row] = val_gate;
          y_up[(col_offset + col) * inter_m + row] = val_up;
        }
      }
    }
  }
}

/**
 * Custom Fused Pair Q4_K Gate + Up GEMV Kernel (BATCH_N = 1).
 * Computes both Gate and Up rows simultaneously in a single warp pass,
 * reading activations x_q8_1 exactly ONCE for both projections.
 */
template <int WARPS_PER_ROW = 2>
__global__ __launch_bounds__(256, 4) void mmvq_gate_up_q4_k_2warp_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW; // 4 rows
  int row_in_block = warp_id / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + row_in_block;
  bool valid_row = (row < inter_m);

  int warp_sub = warp_id % WARPS_PER_ROW; // 0 or 1
  int num_blocks_per_row = k / 256;
  int blocks_per_warp =
      (num_blocks_per_row + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
  int b_start = min(warp_sub * blocks_per_warp, num_blocks_per_row);
  int b_end = min(b_start + blocks_per_warp, num_blocks_per_row);

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

  float dot_gate = 0.0f;
  float dot_up = 0.0f;

  if (valid_row) {
    for (int b = b_start; b < b_end; b++) {
      // 1. Load x_q8_1 once
      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
      const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
      const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

      float d8_0 = __low2float(__ldg(&bq8_0->ds));
      float d8_1 = __low2float(__ldg(&bq8_1->ds));

      int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
      int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);
      int sum0 = __dp4a(0x01010101, u0, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);

      // 2. Gate (w1)
      const BlockQ4_K_Gpu *blk1 = &row_w1[b];
      uint4 hdr1 = ((const uint4 *)blk1)[0];
      float d1 = __low2float(*((const half2 *)&hdr1.x));
      float dmin1 = __high2float(*((const half2 *)&hdr1.x));
      float sc0_1, min0_1, sc1_1, min1_1;
      get_scale_min_k4_from_regs(chunk, hdr1.y, hdr1.z, hdr1.w, &sc0_1, &min0_1,
                                 &sc1_1, &min1_1);
      int v1 = ((const int *)blk1->qs)[lane_id];
      int v0_1 = v1 & 0x0F0F0F0F;
      int v1_1 = (v1 >> 4) & 0x0F0F0F0F;
      int dot0_1 = __dp4a(v0_1, u0, 0);
      int dot1_1 = __dp4a(v1_1, u1, 0);
      float sumf_d_1 = (float)(dot0_1 * (int)sc0_1) * d8_0 +
                       (float)(dot1_1 * (int)sc1_1) * d8_1;
      float sumf_m_1 = (float)(sum0 * (int)min0_1) * d8_0 +
                       (float)(sum1 * (int)min1_1) * d8_1;
      dot_gate += d1 * sumf_d_1 - dmin1 * sumf_m_1;

      // 3. Up (w3)
      const BlockQ4_K_Gpu *blk3 = &row_w3[b];
      uint4 hdr3 = ((const uint4 *)blk3)[0];
      float d3 = __low2float(*((const half2 *)&hdr3.x));
      float dmin3 = __high2float(*((const half2 *)&hdr3.x));
      float sc0_3, min0_3, sc1_3, min1_3;
      get_scale_min_k4_from_regs(chunk, hdr3.y, hdr3.z, hdr3.w, &sc0_3, &min0_3,
                                 &sc1_3, &min1_3);
      int v3 = ((const int *)blk3->qs)[lane_id];
      int v0_3 = v3 & 0x0F0F0F0F;
      int v1_3 = (v3 >> 4) & 0x0F0F0F0F;
      int dot0_3 = __dp4a(v0_3, u0, 0);
      int dot1_3 = __dp4a(v1_3, u1, 0);
      float sumf_d_3 = (float)(dot0_3 * (int)sc0_3) * d8_0 +
                       (float)(dot1_3 * (int)sc1_3) * d8_1;
      float sumf_m_3 = (float)(sum0 * (int)min0_3) * d8_0 +
                       (float)(sum1 * (int)min1_3) * d8_1;
      dot_up += d3 * sumf_d_3 - dmin3 * sumf_m_3;
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_down_sync(0xFFFFFFFF, dot_gate, offset);
    dot_up += __shfl_down_sync(0xFFFFFFFF, dot_up, offset);
  }

  __shared__ float s_gate[4][WARPS_PER_ROW];
  __shared__ float s_up[4][WARPS_PER_ROW];

  if (lane_id == 0 && valid_row) {
    s_gate[row_in_block][warp_sub] = dot_gate;
    s_up[row_in_block][warp_sub] = dot_up;
  }
  __syncthreads();

  if (warp_sub == 0 && lane_id == 0 && valid_row) {
    float tot_g = 0.0f;
    float tot_u = 0.0f;
    for (int w = 0; w < WARPS_PER_ROW; ++w) {
      tot_g += s_gate[row_in_block][w];
      tot_u += s_up[row_in_block][w];
    }
    if (y_up != nullptr) {
      y_gate[row] = tot_g;
      y_up[row] = tot_u;
    } else {
      float silu_g = tot_g / (1.0f + __expf(-tot_g));
      y_gate[row] = silu_g * tot_u;
    }
  }
}

/**
 * Custom Fused Pair Q5_0 Gate + Up GEMV Kernel (BATCH_N = 1).
 * Computes both Gate (w1) and Up (w3) projections in a cooperative 2-warp pass,
 * reading activations x_q8_1 exactly ONCE for both projections, and fusing
 * SwiGLU.
 */
template <int WARPS_PER_ROW = 2>
__global__ __launch_bounds__(256, 4) void mmvq_gate_up_q5_0_2warp_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW; // 4 rows
  int row_in_block = warp_id / WARPS_PER_ROW;
  int row = blockIdx.x * rows_per_block + row_in_block;
  bool valid_row = (row < inter_m);

  int warp_sub = warp_id % WARPS_PER_ROW; // 0 or 1
  int num_blocks_per_row = k / 32;
  int blocks_per_warp = num_blocks_per_row / WARPS_PER_ROW;
  int b_start = warp_sub * blocks_per_warp;
  int b_end = (warp_sub == WARPS_PER_ROW - 1) ? num_blocks_per_row
                                              : (b_start + blocks_per_warp);

  const BlockQ5_0_Gpu *row_w1 =
      valid_row ? (const BlockQ5_0_Gpu *)((const uint8_t *)raw_w1 +
                                          (size_t)row * num_blocks_per_row * 22)
                : nullptr;
  const BlockQ5_0_Gpu *row_w3 =
      valid_row ? (const BlockQ5_0_Gpu *)((const uint8_t *)raw_w3 +
                                          (size_t)row * num_blocks_per_row * 22)
                : nullptr;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int i = lane_id & 15;
  float dot_gate = 0.0f;
  float dot_up = 0.0f;

  if (valid_row) {
    for (int b = b_start; b < b_end; b++) {
      // 1. Load x_q8_1 once for both gate and up
      const BlockQ8_1_Gpu *bq8_1 = &x_q8_1[b];
      float d8 = __low2float(bq8_1->ds);
      int8_t u = bq8_1->qs[lane_id];

      // 2. Gate (w1)
      const BlockQ5_0_Gpu *blk1 = &row_w1[b];
      float d1 = __half2float(*((const half *)&blk1->d));
      uint32_t qh1 = (uint32_t)blk1->qh[0] | ((uint32_t)blk1->qh[1] << 8) |
                     ((uint32_t)blk1->qh[2] << 16) |
                     ((uint32_t)blk1->qh[3] << 24);
      uint8_t byte1 = blk1->qs[i];
      uint8_t h1 = (qh1 >> lane_id) & 1;
      int8_t q1 =
          (lane_id < 16)
              ? (int8_t)(((int)(byte1 & 0x0F) | ((int)h1 << 4)) - 16)
              : (int8_t)(((int)((byte1 >> 4) & 0x0F) | ((int)h1 << 4)) - 16);
      dot_gate += (d1 * d8) * (float)((int)q1 * (int)u);

      // 3. Up (w3)
      const BlockQ5_0_Gpu *blk3 = &row_w3[b];
      float d3 = __half2float(*((const half *)&blk3->d));
      uint32_t qh3 = (uint32_t)blk3->qh[0] | ((uint32_t)blk3->qh[1] << 8) |
                     ((uint32_t)blk3->qh[2] << 16) |
                     ((uint32_t)blk3->qh[3] << 24);
      uint8_t byte3 = blk3->qs[i];
      uint8_t h3 = (qh3 >> lane_id) & 1;
      int8_t q3 =
          (lane_id < 16)
              ? (int8_t)(((int)(byte3 & 0x0F) | ((int)h3 << 4)) - 16)
              : (int8_t)(((int)((byte3 >> 4) & 0x0F) | ((int)h3 << 4)) - 16);
      dot_up += (d3 * d8) * (float)((int)q3 * (int)u);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot_gate += __shfl_down_sync(0xFFFFFFFF, dot_gate, offset);
    dot_up += __shfl_down_sync(0xFFFFFFFF, dot_up, offset);
  }

  __shared__ float s_gate[4][WARPS_PER_ROW];
  __shared__ float s_up[4][WARPS_PER_ROW];

  if (lane_id == 0 && valid_row) {
    s_gate[row_in_block][warp_sub] = dot_gate;
    s_up[row_in_block][warp_sub] = dot_up;
  }
  __syncthreads();

  if (warp_sub == 0 && lane_id == 0 && valid_row) {
    float tot_g = 0.0f;
    float tot_u = 0.0f;
    for (int w = 0; w < WARPS_PER_ROW; ++w) {
      tot_g += s_gate[row_in_block][w];
      tot_u += s_up[row_in_block][w];
    }
    if (y_up != nullptr) {
      y_gate[row] = tot_g;
      y_up[row] = tot_u;
    } else {
      float silu_g = tot_g / (1.0f + __expf(-tot_g));
      y_gate[row] = silu_g * tot_u;
    }
  }
}

/**
 * Pure Warp Single-Token QKV GEMV Kernel (BATCH_N = 1, 0 Shared Memory, 0
 * __syncthreads). Computes Q (Q4_K), K (Q4_K), and V (Q4_K when V_QUANT == 1,
 * or Q6_K when V_QUANT == 2) in a single warp-synchronous dispatch, writing V
 * directly to the FP16 KV cache.
 */
template <int V_QUANT>
__global__ __launch_bounds__(256, 4) void gemv_qkv_mixed_pure_warp_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    void *__restrict__ v_cache_fp16, int start_pos,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k, const int *__restrict__ d_start_pos = nullptr) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = blockDim.x / 32;
  int total_row = blockIdx.x * rows_per_block + warp_id;
  int total_m = q_m + 2 * kv_m;
  if (total_row >= total_m)
    return;

  int row = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (total_row < q_m) {
    row = total_row;
    raw_blocks = raw_wq;
    y = y_q;
  } else if (total_row < q_m + kv_m) {
    row = total_row - q_m;
    raw_blocks = raw_wk;
    y = y_k;
  } else {
    row = total_row - q_m - kv_m;
    raw_blocks = raw_wv;
    y = y_v;
    is_v = true;
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot = 0.0f;

  if (!is_v || V_QUANT == 1) {
    const BlockQ4_K_Gpu *row_blocks =
        (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 144);
    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    for (int b = 0; b < num_blocks_per_row; b++) {
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

      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
      const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
      const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

      float d8_0 = __low2float(__ldg(&bq8_0->ds));
      float d8_1 = __low2float(__ldg(&bq8_1->ds));

      int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
      int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

      int dot0 = __dp4a(v0, u0, 0);
      int dot1 = __dp4a(v1, u1, 0);
      int sum0 = __dp4a(0x01010101, u0, 0);
      int sum1 = __dp4a(0x01010101, u1, 0);

      float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
      float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

      thread_dot += acc0 * d8_0 + acc1 * d8_1;
    }
  } else {
    const BlockQ6_K_Gpu *row_blocks =
        (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                (size_t)row * num_blocks_per_row * 210);
    int group = lane_id / 8;
    int idx = lane_id % 8;

    for (int b = 0; b < num_blocks_per_row; b++) {
      const BlockQ6_K_Gpu *blk = &row_blocks[b];
      float d = __half2float(*((const half *)&blk->d));
      const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];

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

        const BlockQ8_1_Gpu *bq8 = &bx[sub];
        float d8 = __low2float(__ldg(&bq8->ds));
        int u = __ldg(&((const int *)bq8->qs)[idx]);
        int dot = __dp4a(v_signed, u, 0);
        thread_dot += d_sc * d8 * (float)dot;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    if (y != nullptr) {
      y[row] = thread_dot;
    }
    if (is_v && v_cache_fp16 != nullptr) {
      int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
      ((__half *)v_cache_fp16)[(size_t)actual_pos * kv_m + row] =
          __float2half(thread_dot);
    }
  }
}

/**
 * Split-K Multi-Warp Single-Token QKV GEMV Kernel (BATCH_N = 1).
 * Splits row reduction across WARPS_PER_ROW to saturate SM vector units and
 * DRAM bandwidth. Supports Q (Q4_K), K (Q4_K), and V (Q4_K when V_QUANT == 1,
 * or Q6_K when V_QUANT == 2) in a single dispatch, writing V directly to the
 * FP16 KV cache.
 */
template <int WARPS_PER_ROW = 2, int V_QUANT = 1>
__global__ __launch_bounds__(256, 4) void gemv_qkv_mixed_splitk_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    void *__restrict__ v_cache_fp16, int start_pos,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k, const int *__restrict__ d_start_pos = nullptr,
    const float *__restrict__ q_bias = nullptr,
    const float *__restrict__ k_bias = nullptr,
    const float *__restrict__ v_bias = nullptr) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW;
  int row_in_block = warp_id / WARPS_PER_ROW;
  int total_row = blockIdx.x * rows_per_block + row_in_block;
  int total_m = q_m + 2 * kv_m;
  bool valid_row = (total_row < total_m);

  int warp_sub = warp_id % WARPS_PER_ROW;
  int num_blocks_per_row = k / 256;
  int blocks_per_warp =
      (num_blocks_per_row + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
  int b_start = min(warp_sub * blocks_per_warp, num_blocks_per_row);
  int b_end = min(b_start + blocks_per_warp, num_blocks_per_row);

  int row = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (valid_row) {
    if (total_row < q_m) {
      row = total_row;
      raw_blocks = raw_wq;
      y = y_q;
    } else if (total_row < q_m + kv_m) {
      row = total_row - q_m;
      raw_blocks = raw_wk;
      y = y_k;
    } else {
      row = total_row - q_m - kv_m;
      raw_blocks = raw_wv;
      y = y_v;
      is_v = true;
    }
  }

  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;
  float thread_dot = 0.0f;

  if (valid_row) {
    if (!is_v || V_QUANT == 1) {
      const BlockQ4_K_Gpu *row_blocks =
          (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                  (size_t)row * num_blocks_per_row * 144);
      int chunk = lane_id / 8;
      int sub_idx = lane_id % 8;
      int sub0 = 2 * chunk + 0;
      int sub1 = 2 * chunk + 1;

      for (int b = b_start; b < b_end; b++) {
        const BlockQ4_K_Gpu *blk = &row_blocks[b];
        uint4 hdr = ((const uint4 *)blk)[0];
        float d = __low2float(*((const half2 *)&hdr.x));
        float dmin = __high2float(*((const half2 *)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0,
                                   &sc1, &min1);

        int v = ((const int *)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
        const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
        const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

        float d8_0 = __low2float(__ldg(&bq8_0->ds));
        float d8_1 = __low2float(__ldg(&bq8_1->ds));

        int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
        int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

        int dot0 = __dp4a(v0, u0, 0);
        int dot1 = __dp4a(v1, u1, 0);
        int sum0 = __dp4a(0x01010101, u0, 0);
        int sum1 = __dp4a(0x01010101, u1, 0);

        float sumf_d =
            (float)(dot0 * (int)sc0) * d8_0 + (float)(dot1 * (int)sc1) * d8_1;
        float sumf_m =
            (float)(sum0 * (int)min0) * d8_0 + (float)(sum1 * (int)min1) * d8_1;
        thread_dot += d * sumf_d - dmin * sumf_m;
      }
    } else {
      const BlockQ6_K_Gpu *row_blocks =
          (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                  (size_t)row * num_blocks_per_row * 210);
      int group = lane_id / 8;
      int idx = lane_id % 8;

      for (int b = b_start; b < b_end; b++) {
        const BlockQ6_K_Gpu *blk = &row_blocks[b];
        float d = __half2float(*((const half *)&blk->d));
        const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
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

          const BlockQ8_1_Gpu *bq8 = &bx[sub];
          float d8 = __low2float(__ldg(&bq8->ds));
          int u = __ldg(&((const int *)bq8->qs)[idx]);
          int dot_val = __dp4a(v_signed, u, 0);
          sum_b += (float)(dot_val * sc) * d8;
        }
        thread_dot += d * sum_b;
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  __shared__ float s_dot_qkv[4][WARPS_PER_ROW];

  if (lane_id == 0 && valid_row) {
    s_dot_qkv[row_in_block][warp_sub] = thread_dot;
  }
  __syncthreads();

  if (warp_sub == 0 && lane_id == 0 && valid_row) {
    float sum = 0.0f;
    for (int w = 0; w < WARPS_PER_ROW; ++w) {
      sum += s_dot_qkv[row_in_block][w];
    }
    if (total_row < q_m) {
      if (q_bias != nullptr)
        sum += q_bias[row];
    } else if (total_row < q_m + kv_m) {
      if (k_bias != nullptr)
        sum += k_bias[row];
    } else {
      if (v_bias != nullptr)
        sum += v_bias[row];
    }
    if (y != nullptr) {
      y[row] = sum;
    }
    if (is_v && v_cache_fp16 != nullptr) {
      int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
      ((__half *)v_cache_fp16)[(size_t)actual_pos * kv_m + row] =
          __float2half(sum);
    }
  }
}

/**
 * Split-K Multi-Warp Grouped QKV Kernel for Q5_0 Weights + Direct FP16 V-Cache
 * Write. Supports Q (Q5_0), K (Q5_0), and V (Q5_0 when V_QUANT == 9, or Q8_0
 * when V_QUANT == 3). Fuses Q, K, V GEMV, biases, and V-cache write into a
 * single kernel dispatch.
 */
template <int WARPS_PER_ROW = 2, int V_QUANT = 9>
__global__ __launch_bounds__(256, 4) void gemv_qkv_q5_0_splitk_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    void *__restrict__ v_cache_fp16, int start_pos,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k, const int *__restrict__ d_start_pos = nullptr,
    const float *__restrict__ q_bias = nullptr,
    const float *__restrict__ k_bias = nullptr,
    const float *__restrict__ v_bias = nullptr) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int rows_per_block = 8 / WARPS_PER_ROW;
  int row_in_block = warp_id / WARPS_PER_ROW;
  int total_row = blockIdx.x * rows_per_block + row_in_block;
  int total_m = q_m + 2 * kv_m;
  bool valid_row = (total_row < total_m);

  int warp_sub = warp_id % WARPS_PER_ROW;
  int num_blocks_per_row = k / 32;
  int blocks_per_warp =
      (num_blocks_per_row + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
  int b_start = min(warp_sub * blocks_per_warp, num_blocks_per_row);
  int b_end = min(b_start + blocks_per_warp, num_blocks_per_row);

  int row = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (valid_row) {
    if (total_row < q_m) {
      row = total_row;
      raw_blocks = raw_wq;
      y = y_q;
    } else if (total_row < q_m + kv_m) {
      row = total_row - q_m;
      raw_blocks = raw_wk;
      y = y_k;
    } else {
      row = total_row - q_m - kv_m;
      raw_blocks = raw_wv;
      y = y_v;
      is_v = true;
    }
  }

  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;
  float thread_dot = 0.0f;

  if (valid_row) {
    if (!is_v || V_QUANT == 9) {
      const BlockQ5_0_Gpu *row_blocks =
          (const BlockQ5_0_Gpu *)((const uint8_t *)raw_blocks +
                                  (size_t)row * num_blocks_per_row * 22);
      int i = lane_id & 15;
      for (int b = b_start; b < b_end; b++) {
        const BlockQ8_1_Gpu *bq8_1 = &x_q8_1[b];
        float d8 = __low2float(bq8_1->ds);
        int8_t u = bq8_1->qs[lane_id];

        const BlockQ5_0_Gpu *blk = &row_blocks[b];
        float d = __half2float(*((const half *)&blk->d));
        uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                      ((uint32_t)blk->qh[2] << 16) |
                      ((uint32_t)blk->qh[3] << 24);
        uint8_t byte_val = blk->qs[i];
        uint8_t h_bit = (qh >> lane_id) & 1;
        int8_t q =
            (lane_id < 16)
                ? (int8_t)(((int)(byte_val & 0x0F) | ((int)h_bit << 4)) - 16)
                : (int8_t)(((int)((byte_val >> 4) & 0x0F) | ((int)h_bit << 4)) -
                           16);
        thread_dot += (d * d8) * (float)((int)q * (int)u);
      }
    } else if (V_QUANT == 3) {
      const BlockQ8_0_Gpu *row_blocks =
          (const BlockQ8_0_Gpu *)((const uint8_t *)raw_blocks +
                                  (size_t)row * num_blocks_per_row * 34);
      for (int b = b_start; b < b_end; b++) {
        const BlockQ8_1_Gpu *bq8_1 = &x_q8_1[b];
        float d8 = __low2float(bq8_1->ds);
        int8_t u = bq8_1->qs[lane_id];

        const BlockQ8_0_Gpu *blk = &row_blocks[b];
        float d = __half2float(*((const half *)&blk->scale));
        int8_t q = blk->qs[lane_id];
        thread_dot += (d * d8) * (float)((int)q * (int)u);
      }
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  __shared__ float s_dot_qkv[4][WARPS_PER_ROW];

  if (lane_id == 0 && valid_row) {
    s_dot_qkv[row_in_block][warp_sub] = thread_dot;
  }
  __syncthreads();

  if (warp_sub == 0 && lane_id == 0 && valid_row) {
    float sum = 0.0f;
    for (int w = 0; w < WARPS_PER_ROW; ++w) {
      sum += s_dot_qkv[row_in_block][w];
    }
    if (total_row < q_m) {
      if (q_bias != nullptr)
        sum += q_bias[row];
    } else if (total_row < q_m + kv_m) {
      if (k_bias != nullptr)
        sum += k_bias[row];
    } else {
      if (v_bias != nullptr)
        sum += v_bias[row];
    }
    if (y != nullptr) {
      y[row] = sum;
    }
    if (is_v && v_cache_fp16 != nullptr) {
      int actual_pos = (d_start_pos != nullptr) ? *d_start_pos : start_pos;
      ((__half *)v_cache_fp16)[(size_t)actual_pos * kv_m + row] =
          __float2half(sum);
    }
  }
}

/**
 * Grouped Multi-Column Mixed-Quant QKV MMVQ Kernel + Direct FP16 V-Cache Write.
 * Supports Q (Q4_K), K (Q4_K), and V (Q4_K when V_QUANT == 1, or Q6_K when
 * V_QUANT == 2) in a single kernel launch, and optionally writes V directly to
 * the FP16 KV cache.
 */
template <int BATCH_N, int V_QUANT, int WARPS_PER_ROW = 1>
__global__ __launch_bounds__(
    256,
    (BATCH_N > 32
         ? 1
         : (BATCH_N > 16
                ? 3
                : 4))) void mmvq_qkv_mixed_cuda_kernel(float *__restrict__ y_q,
                                                       float *__restrict__ y_k,
                                                       float *__restrict__ y_v,
                                                       void
                                                           *__restrict__ v_cache_fp16,
                                                       int start_pos,
                                                       const void
                                                           *__restrict__ raw_wq,
                                                       const void
                                                           *__restrict__ raw_wk,
                                                       const void
                                                           *__restrict__ raw_wv,
                                                       const void
                                                           *__restrict__ x_q8_1_raw,
                                                       int q_m, int kv_m, int k,
                                                       int actual_batch =
                                                           BATCH_N,
                                                       const int
                                                           *__restrict__ d_start_pos =
                                                               nullptr) {
  int warp_id = threadIdx.x / 32;
  int rows_per_block = (blockDim.x / 32) / WARPS_PER_ROW;
  int total_row = blockIdx.x * rows_per_block + (warp_id / WARPS_PER_ROW);
  int total_m = q_m + 2 * kv_m;
  bool valid_row = (total_row < total_m);

  int warp_col_group = warp_id % WARPS_PER_ROW;
  int col_offset = (blockIdx.y * WARPS_PER_ROW + warp_col_group) * BATCH_N;
  int num_cols = (BATCH_N == 1 && WARPS_PER_ROW == 1)
                     ? 1
                     : ((actual_batch > col_offset)
                            ? ((actual_batch - col_offset < BATCH_N)
                                   ? (actual_batch - col_offset)
                                   : BATCH_N)
                            : 0);

  int row = 0;
  int out_stride = 0;
  const void *raw_blocks = nullptr;
  float *y = nullptr;
  bool is_v = false;

  if (valid_row) {
    if (total_row < q_m) {
      row = total_row;
      out_stride = q_m;
      raw_blocks = raw_wq;
      y = y_q;
    } else if (total_row < q_m + kv_m) {
      row = total_row - q_m;
      out_stride = kv_m;
      raw_blocks = raw_wk;
      y = y_k;
    } else {
      row = total_row - q_m - kv_m;
      out_stride = kv_m;
      raw_blocks = raw_wv;
      y = y_v;
      is_v = true;
    }
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;
  int num_q8_blocks_per_col = k / 32;
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  float thread_dot[BATCH_N];
#pragma unroll
  for (int col = 0; col < BATCH_N; ++col) {
    thread_dot[col] = 0.0f;
  }

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;
  int group = lane_id / 8;
  int idx = lane_id % 8;

  for (int b = 0; b < num_blocks_per_row; b++) {
    if (valid_row) {
      if (!is_v || V_QUANT == 1) {
        const BlockQ4_K_Gpu *blk = &(
            (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                                    (size_t)row * num_blocks_per_row * 144))[b];
        uint4 hdr = ((const uint4 *)blk)[0];
        float d = __low2float(*((const half2 *)&hdr.x));
        float dmin = __high2float(*((const half2 *)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0,
                                   &sc1, &min1);

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

            int u0 = __ldg(&((const int *)bq8_0->qs)[sub_idx]);
            int u1 = __ldg(&((const int *)bq8_1->qs)[sub_idx]);

            int dot0 = __dp4a(v0, u0, 0);
            int dot1 = __dp4a(v1, u1, 0);
            int sum0 = __dp4a(0x01010101, u0, 0);
            int sum1 = __dp4a(0x01010101, u1, 0);

            float s0 = d_sc0 * d8_0;
            float m0 = dm_min0 * d8_0;
            float s1 = d_sc1 * d8_1;
            float m1 = dm_min1 * d8_1;

            thread_dot[col] += (s0 * (float)dot0 - m0 * (float)sum0) +
                               (s1 * (float)dot1 - m1 * (float)sum1);
          }
        }
      } else {
        const BlockQ6_K_Gpu *blk = &(
            (const BlockQ6_K_Gpu *)((const uint8_t *)raw_blocks +
                                    (size_t)row * num_blocks_per_row * 210))[b];
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
          if (y != nullptr) {
            y[(col_offset + col) * out_stride + row] = val;
          }
          if (is_v && v_cache_fp16 != nullptr) {
            int actual_pos =
                (d_start_pos != nullptr) ? *d_start_pos : start_pos;
            ((__half *)
                 v_cache_fp16)[(size_t)(actual_pos + col_offset + col) * kv_m +
                               row] = __float2half(val);
          }
        }
      }
    }
  }
}

/**
 * Grouped Q4_K Gate + Up GEMV Kernel.
 * Computes both Gate (w1) and Up (w3) projections in a single kernel dispatch.
 * Total rows = 2 * inter_m. Rows [0, inter_m) -> w1 -> y_gate. Rows [inter_m, 2
 * * inter_m) -> w3 -> y_up.
 */
__global__ void gemv_gate_up_q4_k_cuda_kernel(
    float *__restrict__ y_gate, float *__restrict__ y_up,
    const void *__restrict__ raw_w1, const void *__restrict__ raw_w3,
    const void *__restrict__ x_q8_1_raw, int inter_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (total_row >= 2 * inter_m)
    return;

  bool is_up = (total_row >= inter_m);
  int row = is_up ? (total_row - inter_m) : total_row;
  const void *raw_blocks = is_up ? raw_w3 : raw_w1;
  float *y = is_up ? y_up : y_gate;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
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

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Grouped Q4_K QKV GEMV Kernel.
 * Computes Q (wq), K (wk), and V (wv) projections in a single kernel dispatch
 * when all are Q4_K. Total rows = q_m + 2 * kv_m. Rows [0, q_m) -> wq -> y_q.
 * Rows [q_m, q_m + kv_m) -> wk -> y_k.
 * Rows [q_m + kv_m, q_m + 2 * kv_m) -> wv -> y_v.
 */
__global__ void gemv_qkv_q4_k_cuda_kernel(
    float *__restrict__ y_q, float *__restrict__ y_k, float *__restrict__ y_v,
    const void *__restrict__ raw_wq, const void *__restrict__ raw_wk,
    const void *__restrict__ raw_wv, const void *__restrict__ x_q8_1_raw,
    int q_m, int kv_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int total_m = q_m + 2 * kv_m;
  if (total_row >= total_m)
    return;

  int row;
  const void *raw_blocks;
  float *y;

  if (total_row < q_m) {
    row = total_row;
    raw_blocks = raw_wq;
    y = y_q;
  } else if (total_row < q_m + kv_m) {
    row = total_row - q_m;
    raw_blocks = raw_wk;
    y = y_k;
  } else {
    row = total_row - q_m - kv_m;
    raw_blocks = raw_wv;
    y = y_v;
  }

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
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

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Grouped Q4_K QK GEMV Kernel.
 * Computes Q (wq) and K (wk) projections in a single kernel dispatch when wv
 * has a different quant format. Total rows = q_m + kv_m.
 */
__global__ void gemv_qk_q4_k_cuda_kernel(float *__restrict__ y_q,
                                         float *__restrict__ y_k,
                                         const void *__restrict__ raw_wq,
                                         const void *__restrict__ raw_wk,
                                         const void *__restrict__ x_q8_1_raw,
                                         int q_m, int kv_m, int k) {
  int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  int total_m = q_m + kv_m;
  if (total_row >= total_m)
    return;

  bool is_k = (total_row >= q_m);
  int row = is_k ? (total_row - q_m) : total_row;
  const void *raw_blocks = is_k ? raw_wk : raw_wq;
  float *y = is_k ? y_k : y_q;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ4_K_Gpu *row_blocks =
      (const BlockQ4_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 144);
  const BlockQ8_1_Gpu *x_q8_1 = (const BlockQ8_1_Gpu *)x_q8_1_raw;

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;
  int sub0 = 2 * chunk + 0;
  int sub1 = 2 * chunk + 1;

  float thread_dot = 0.0f;

#pragma unroll 2
  for (int b = 0; b < num_blocks_per_row; b++) {
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

    const BlockQ8_1_Gpu *bx = &x_q8_1[b * 8];
    const BlockQ8_1_Gpu *bq8_0 = &bx[sub0];
    const BlockQ8_1_Gpu *bq8_1 = &bx[sub1];

    float d8_0 = __low2float(bq8_0->ds);
    float d8_1 = __low2float(bq8_1->ds);

    int u0 = ((const int *)bq8_0->qs)[sub_idx];
    int u1 = ((const int *)bq8_1->qs)[sub_idx];

    int dot0 = __dp4a(v0, u0, 0);
    int dot1 = __dp4a(v1, u1, 0);
    int sum0 = __dp4a(0x01010101, u0, 0);
    int sum1 = __dp4a(0x01010101, u1, 0);

    float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
    float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

    thread_dot += acc0 * d8_0 + acc1 * d8_1;
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    thread_dot += __shfl_down_sync(0xFFFFFFFF, thread_dot, offset);
  }

  if (lane_id == 0) {
    y[row] = thread_dot;
  }
}

/**
 * Custom CUDA Kernel: Quantized Q2_K Matrix-Vector GEMV (2-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q2_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ2_K_Gpu *row_blocks =
      (const BlockQ2_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 84);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ2_K_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float min = __half2float(*((const half *)&blk->dmin));
    const float *x_block = x + b * 256;

    int l_offset = (lane_id >= 16) ? 16 : 0;
    int l = lane_id & 15;
    int sc_lane_idx = (lane_id >= 16) ? 1 : 0;

#pragma unroll 2
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        int is = half_idx * 8 + j * 2 + sc_lane_idx;
        uint8_t sc = blk->scales[is];
        float dl = d * (float)(sc & 0x0F);
        float ml = min * (float)(sc >> 4);

        uint8_t q_byte = blk->qs[half_idx * 32 + l_offset + l];
        float q_val = (float)((q_byte >> (j * 2)) & 3);
        float w = dl * q_val - ml;

        float x_val = x_block[half_idx * 128 + j * 32 + lane_id];
        warp_dot += w * x_val;
      }
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
 * Custom CUDA Kernel: Quantized Q3_K Matrix-Vector GEMV (3-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q3_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ3_K_Gpu *row_blocks =
      (const BlockQ3_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 110);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ3_K_Gpu *blk = &row_blocks[b];
    float d_all = __half2float(*((const half *)&blk->d));
    const float *x_block = x + b * 256;

    uint32_t aux[4];
    memcpy(aux, blk->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
    aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
    aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
    aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
    const int8_t *scales = (const int8_t *)aux;

    int l_offset = (lane_id >= 16) ? 16 : 0;
    int l = lane_id & 15;
    int sc_lane_idx = (lane_id >= 16) ? 1 : 0;
    uint8_t hm_byte = blk->hmask[l_offset + l];

#pragma unroll 2
    for (int half_idx = 0; half_idx < 2; half_idx++) {
#pragma unroll 4
      for (int j = 0; j < 4; j++) {
        int is = half_idx * 8 + j * 2 + sc_lane_idx;
        float dl = d_all * (float)(scales[is] - 32);

        uint8_t m_bit = (uint8_t)(1 << (half_idx * 4 + j));
        int8_t h_sub = (hm_byte & m_bit) ? 0 : 4;

        uint8_t q_byte = blk->qs[half_idx * 32 + l_offset + l];
        int8_t q_low = (int8_t)((q_byte >> (j * 2)) & 3);
        float w = dl * (float)(q_low - h_sub);

        float x_val = x_block[half_idx * 128 + j * 32 + lane_id];
        warp_dot += w * x_val;
      }
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
 * Custom CUDA Kernel: Quantized Q5_K Matrix-Vector GEMV (5-bit Super-blocks x
 * FP32 Activations). Warp-Per-Row architecture: 0 shared memory, 0
 * __syncthreads(), independent warp execution.
 */
__global__ void gemv_q5_k_cuda_kernel(float *__restrict__ y,
                                      const void *__restrict__ raw_blocks,
                                      const float *__restrict__ x, int m,
                                      int k) {
  int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
  if (row >= m)
    return;

  int lane_id = threadIdx.x % 32;
  int num_blocks_per_row = k / 256;

  const BlockQ5_K_Gpu *row_blocks =
      (const BlockQ5_K_Gpu *)((const uint8_t *)raw_blocks +
                              (size_t)row * num_blocks_per_row * 176);

  float warp_dot = 0.0f;
  for (int b = 0; b < num_blocks_per_row; ++b) {
    const BlockQ5_K_Gpu *blk = &row_blocks[b];
    float d = __half2float(*((const half *)&blk->d));
    float dmin = __half2float(*((const half *)&blk->dmin));
    const float *x_block = x + b * 256;

#pragma unroll 4
    for (int chunk = 0; chunk < 4; chunk++) {
      float sc1, m1, sc2, m2;
      get_scale_min_k4_cuda(chunk * 2 + 0, blk->scales, &sc1, &m1);
      get_scale_min_k4_cuda(chunk * 2 + 1, blk->scales, &sc2, &m2);
      float d1 = d * sc1;
      float min1 = dmin * m1;
      float d2 = d * sc2;
      float min2 = dmin * m2;

      uint8_t u1 = 1 << (chunk * 2);
      uint8_t u2 = 2 << (chunk * 2);

      uint8_t ql = blk->qs[chunk * 32 + lane_id];
      uint8_t qh = blk->qh[lane_id];

      float w1 = d1 * (float)((ql & 0x0F) + ((qh & u1) ? 16 : 0)) - min1;
      float w2 = d2 * (float)((ql >> 4) + ((qh & u2) ? 16 : 0)) - min2;

      float x0 = x_block[chunk * 64 + lane_id];
      float x1 = x_block[chunk * 64 + 32 + lane_id];

      warp_dot += w1 * x0 + w2 * x1;
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
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling +
 * Online Softmax)
 */
