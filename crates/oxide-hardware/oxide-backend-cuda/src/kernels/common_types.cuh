// Common CUDA LLM Kernel Types
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_1_Gpu {
  half2 ds;      // ds.x = d (scale = amax / 127.0f), ds.y = sum
  int8_t qs[32]; // 32 quantized signed 8-bit ints
};

struct __attribute__((packed)) BlockQ8_0_Gpu {
  uint16_t scale;
  int8_t qs[32];
};

struct __attribute__((packed)) BlockQ5_0_Gpu {
  uint16_t d;
  uint8_t qh[4];
  uint8_t qs[16];
};

static __device__ __forceinline__ int get_int_b2(const void *x, int i32) {
  const uint16_t *x16 = (const uint16_t *)x;
  int x32 = (int)x16[2 * i32 + 0];
  x32 |= ((int)x16[2 * i32 + 1]) << 16;
  return x32;
}

static __device__ __forceinline__ int get_int_b4(const void *x, int i32) {
  return ((const int *)x)[i32];
}

static __device__ __forceinline__ float
compute_q5_0_q8_1_dot(const BlockQ5_0_Gpu *blk, const BlockQ8_1_Gpu *bq8_1) {
  int sumi = 0;
  uint32_t qh = (uint32_t)blk->qh[0] | ((uint32_t)blk->qh[1] << 8) |
                ((uint32_t)blk->qh[2] << 16) | ((uint32_t)blk->qh[3] << 24);

#pragma unroll
  for (int g = 0; g < 4; ++g) {
    int vl = get_int_b2(blk->qs, g);
    uint32_t vh = qh >> (4 * g);
    int u0 = get_int_b4(bq8_1->qs, g);
    int u1 = get_int_b4(bq8_1->qs, g + 4);

    int vi0 = (vl >> 0) & 0x0F0F0F0F;
    vi0 |= (vh << 4) & 0x00000010;
    vi0 |= (vh << 11) & 0x00001000;
    vi0 |= (vh << 18) & 0x00100000;
    vi0 |= (vh << 25) & 0x10000000;
    sumi = __dp4a(vi0, u0, sumi);

    int vi1 = (vl >> 4) & 0x0F0F0F0F;
    vi1 |= (vh >> 12) & 0x00000010;
    vi1 |= (vh >> 5) & 0x00001000;
    vi1 |= (vh << 2) & 0x00100000;
    vi1 |= (vh << 9) & 0x10000000;
    sumi = __dp4a(vi1, u1, sumi);
  }

  float d5 = __half2float(*((const half *)&blk->d));
  float2 ds8f = __half22float2(bq8_1->ds);
  return d5 * ((float)sumi * ds8f.x - 16.0f * ds8f.y);
}
