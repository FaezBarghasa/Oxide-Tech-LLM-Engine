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

/**
 * Custom CUDA Kernel: Dynamic Activation Quantization to Q8_1.
 * Symmetric per-32 group quantization with half2 (scale, sum) header.
 * Consumed by MMVQ INT8 Tensor/ALU kernels with hardware __dp4a.
 */
