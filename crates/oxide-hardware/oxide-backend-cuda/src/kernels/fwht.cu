#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" {

/**
 * In-Shared-Memory Fast Walsh-Hadamard Transform (FWHT) butterfly for 128 elements.
 * Executes 7 unrolled butterfly stages with zero global memory spills.
 */
__device__ __forceinline__ void fwht_butterfly_128(float* smem_lane) {
    #pragma unroll
    for (int stride = 64; stride > 0; stride >>= 1) {
        int tid = threadIdx.x;
        int i = (tid / stride) * (2 * stride) + (tid % stride);
        float u = smem_lane[i];
        float v = smem_lane[i + stride];
        smem_lane[i]          = u + v;
        smem_lane[i + stride] = u - v;
        __syncthreads();
    }
    // Normalize by 1 / sqrt(128)
    const float norm = 0.08838834764f;
    smem_lane[threadIdx.x] *= norm;
}

__global__ void fwht_kernel_128(float* __restrict__ d_out, const float* __restrict__ d_in, int num_blocks) {
    __shared__ float smem[128];
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    smem[tid] = d_in[block_id * 128 + tid];
    __syncthreads();

    fwht_butterfly_128(smem);

    d_out[block_id * 128 + tid] = smem[tid];
}

}
