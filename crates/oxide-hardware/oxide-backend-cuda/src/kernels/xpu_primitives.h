// Oxide-Engine Unified XPU Hardware Primitives Portability Header
// Supports: NVIDIA CUDA (sm_89+), AMD ROCm / HIP (gfx90a/gfx1100), Intel Level-Zero / SYCL, CPU
#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #define XPU_DEVICE __device__
  #define XPU_HOST __host__
  #define XPU_FORCEINLINE __forceinline__
  #define XPU_WARP_SIZE warpSize
  #define XPU_ACTIVE_MASK 0xFFFFFFFFFFFFFFFFull

  XPU_DEVICE XPU_FORCEINLINE uint64_t xpu_ballot(int predicate) {
      return __ballot(predicate);
  }

  XPU_DEVICE XPU_FORCEINLINE float xpu_shfl_down(float val, int delta, int width = 64) {
      return __shfl_down(val, delta, width);
  }

  XPU_DEVICE XPU_FORCEINLINE float xpu_shfl_xor(float val, int lane_mask, int width = 64) {
      return __shfl_xor(val, lane_mask, width);
  }

  XPU_DEVICE XPU_FORCEINLINE int xpu_lane_id() {
      return __lane_id();
  }

#elif defined(__CUDACC__)
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #define XPU_DEVICE __device__
  #define XPU_HOST __host__
  #define XPU_FORCEINLINE __forceinline__
  #define XPU_WARP_SIZE 32
  #define XPU_ACTIVE_MASK 0xFFFFFFFFu

  XPU_DEVICE XPU_FORCEINLINE uint32_t xpu_ballot(int predicate, unsigned int mask = XPU_ACTIVE_MASK) {
      return __ballot_sync(mask, predicate);
  }

  XPU_DEVICE XPU_FORCEINLINE float xpu_shfl_down(float val, int delta, int width = 32, unsigned int mask = XPU_ACTIVE_MASK) {
      return __shfl_down_sync(mask, val, delta, width);
  }

  XPU_DEVICE XPU_FORCEINLINE float xpu_shfl_xor(float val, int lane_mask, int width = 32, unsigned int mask = XPU_ACTIVE_MASK) {
      return __shfl_xor_sync(mask, val, lane_mask, width);
  }

  XPU_DEVICE XPU_FORCEINLINE int xpu_lane_id() {
      int lane;
      asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
      return lane;
  }

#elif defined(__SYCL_DEVICE_ONLY__)
  #include <sycl/sycl.hpp>
  #define XPU_DEVICE
  #define XPU_HOST
  #define XPU_FORCEINLINE inline
  #define XPU_WARP_SIZE 32
  #define XPU_ACTIVE_MASK 0xFFFFFFFFu

#else
  // Host CPU fallback
  #define XPU_DEVICE
  #define XPU_HOST
  #define XPU_FORCEINLINE inline
  #define XPU_WARP_SIZE 32
  #define XPU_ACTIVE_MASK 0xFFFFFFFFu
#endif
