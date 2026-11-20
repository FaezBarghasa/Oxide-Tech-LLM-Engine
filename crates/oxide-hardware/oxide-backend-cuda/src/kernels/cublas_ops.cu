// cuBLAS Operations and Dequantization
// =========================================================================

static cublasHandle_t g_cublas_handle = nullptr;

extern "C" int oxide_cuda_init_cublas(cudaStream_t stream) {
  if (!g_cublas_handle) {
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
      return (int)status;
  }
  if (stream) {
    cublasSetStream(g_cublas_handle, stream);
  }
  return 0;
}

__global__ void
dequantize_q4_k_to_f16_cuda_kernel(half *__restrict__ dst,
                                   const void *__restrict__ raw_blocks,
                                   int num_blocks) {
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int ib = blockIdx.x * (blockDim.x / 32) + warp_id;
  if (ib >= num_blocks)
    return;

  const BlockQ4_K_Gpu *blk = &((const BlockQ4_K_Gpu *)raw_blocks)[ib];
  uint4 hdr = ((const uint4 *)blk)[0];
  float dall = __low2float(*((const half2 *)&hdr.x));
  float dmin = __high2float(*((const half2 *)&hdr.x));

  int chunk = lane_id / 8;
  int sub_idx = lane_id % 8;

  float sc0, min0, sc1, min1;
  get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1,
                             &min1);

  float d1 = dall * sc0;
  float m1 = dmin * min0;
  float d2 = dall * sc1;
  float m2 = dmin * min1;

  int v = ((const int *)blk->qs)[lane_id];
  int v0 = v & 0x0F0F0F0F;
  int v1 = (v >> 4) & 0x0F0F0F0F;

  half2 out0_low =
      make_half2(__float2half(d1 * (float)(v0 & 0xFF) - m1),
                 __float2half(d1 * (float)((v0 >> 8) & 0xFF) - m1));
  half2 out0_high =
      make_half2(__float2half(d1 * (float)((v0 >> 16) & 0xFF) - m1),
                 __float2half(d1 * (float)((v0 >> 24) & 0xFF) - m1));

  half2 out1_low =
      make_half2(__float2half(d2 * (float)(v1 & 0xFF) - m2),
                 __float2half(d2 * (float)((v1 >> 8) & 0xFF) - m2));
  half2 out1_high =
      make_half2(__float2half(d2 * (float)((v1 >> 16) & 0xFF) - m2),
                 __float2half(d2 * (float)((v1 >> 24) & 0xFF) - m2));

  size_t base0 =
      (size_t)ib * 256 + (size_t)(2 * chunk + 0) * 32 + (size_t)sub_idx * 4;
  size_t base1 =
      (size_t)ib * 256 + (size_t)(2 * chunk + 1) * 32 + (size_t)sub_idx * 4;

  *((half2 *)&dst[base0 + 0]) = out0_low;
  *((half2 *)&dst[base0 + 2]) = out0_high;

  *((half2 *)&dst[base1 + 0]) = out1_low;
  *((half2 *)&dst[base1 + 2]) = out1_high;
}

__global__ void
dequantize_q6_k_to_f16_cuda_kernel(half *__restrict__ dst,
                                   const void *__restrict__ raw_blocks,
                                   int num_blocks) {
  int sb_idx = threadIdx.x / 64;
  int tid = threadIdx.x % 64;
  int ib = blockIdx.x * (blockDim.x / 64) + sb_idx;
  if (ib >= num_blocks)
    return;

  const BlockQ6_K_Gpu *blk = &((const BlockQ6_K_Gpu *)raw_blocks)[ib];
  int ip = tid / 32; // 0 or 1
  int il = tid % 32; // 0..31
  int is_idx = 8 * ip + il / 16;

  float d = __half2float(*((const half *)&blk->d));
  const uint8_t *ql = blk->ql + 64 * ip + il;
  const uint8_t qh = blk->qh[32 * ip + il];
  const int8_t *sc = (const int8_t *)blk->scales + is_idx;

  int8_t q0 = (int8_t)((ql[0] & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
  int8_t q1 = (int8_t)((ql[32] & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
  int8_t q2 = (int8_t)((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32;
  int8_t q3 = (int8_t)((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32;

  size_t base = (size_t)ib * 256 + (size_t)128 * ip + (size_t)il;
  dst[base + 0] = __float2half(d * (float)sc[0] * (float)q0);
  dst[base + 32] = __float2half(d * (float)sc[2] * (float)q1);
  dst[base + 64] = __float2half(d * (float)sc[4] * (float)q2);
  dst[base + 96] = __float2half(d * (float)sc[6] * (float)q3);
}

extern "C" int launch_cuda_dequantize_q4_k_f16(void *dst_f16,
                                               const void *src_q4_k,
                                               int num_elements,
                                               cudaStream_t stream) {
  if (!dst_f16 || !src_q4_k || num_elements <= 0)
    return 0;
  int num_blocks = num_elements / 256;
  if (num_blocks <= 0)
    return 0;
  int block_dim = 256;
  int warps_per_block = block_dim / 32;
  int grid_dim = (num_blocks + warps_per_block - 1) / warps_per_block;
  dequantize_q4_k_to_f16_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
      (half *)dst_f16, src_q4_k, num_blocks);
  return cudaGetLastError();
}

extern "C" int launch_cuda_dequantize_q6_k_f16(void *dst_f16,
                                               const void *src_q6_k,
                                               int num_elements,
                                               cudaStream_t stream) {
  if (!dst_f16 || !src_q6_k || num_elements <= 0)
    return 0;
  int num_blocks = num_elements / 256;
  if (num_blocks <= 0)
    return 0;
  int block_dim = 256;
  int sb_per_block = block_dim / 64;
  int grid_dim = (num_blocks + sb_per_block - 1) / sb_per_block;
  dequantize_q6_k_to_f16_cuda_kernel<<<grid_dim, block_dim, 0, stream>>>(
      (half *)dst_f16, src_q6_k, num_blocks);
  return cudaGetLastError();
}

extern "C" int launch_cuda_cublas_gemm_fp16_f32(float *d_y, const void *d_w_f16,
                                                const void *d_x_f16, int m,
                                                int k, int batch_size,
                                                int add_to_y,
                                                cudaStream_t stream) {
  if (m <= 0 || k <= 0 || batch_size <= 0)
    return 0;
  if (oxide_cuda_init_cublas(stream) != 0)
    return -1;

  float alpha = 1.0f;
  float beta = add_to_y ? 1.0f : 0.0f;

  cublasStatus_t status = cublasGemmEx(
      g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, batch_size, k, &alpha,
      d_w_f16, CUDA_R_16F, k, d_x_f16, CUDA_R_16F, k, &beta, d_y, CUDA_R_32F, m,
      CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return (int)status;
}

// C-ABI Launch Wrappers for FFI

