#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_1_Gpu {
    half2 ds;       // ds.x = d (scale = amax / 127.0f), ds.y = sum
    int8_t qs[32];  // 32 quantized signed 8-bit ints
};

/**
 * Custom CUDA Kernel: Dynamic Activation Quantization to Q8_1.
 * Symmetric per-32 group quantization with half2 (scale, sum) header.
 * Consumed by MMVQ INT8 Tensor/ALU kernels with hardware __dp4a.
 */
__global__ void quantize_f32_to_q8_1_cuda_kernel(
    const float* __restrict__ x,
    BlockQ8_1_Gpu* __restrict__ y,
    int n
) {
    int block_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int num_blocks = n / 32;
    if (block_idx >= num_blocks) return;
    int lane_id = threadIdx.x % 32;

    float val = x[block_idx * 32 + lane_id];
    float amax = fabsf(val);
    float sum = val;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    float d = amax / 127.0f;
    float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    int8_t q = (int8_t)roundf(val * id);
    y[block_idx].qs[lane_id] = q;

    if (lane_id == 0) {
        y[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

/**
 * Custom CUDA Kernel: RMSNorm (Root Mean Square Normalization)
 * Standard normalization across hidden dimension (e.g. 4096 / 8192) in Llama/Mistral/Qwen.
 * Performs block-wide warp-shuffle reduction for zero shared-memory overhead.
 */
__global__ void rms_norm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden_dim,
    float eps,
    float weight_offset
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const float* in_row = input + bid * hidden_dim;
    float* out_row = output + bid * hidden_dim;

    // Compute sum of squares with unrolled grid-stride loop
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = in_row[i];
        sum_sq += val * val;
    }

    // Warp reduction using shuffle down
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Shared memory for block-level reduction
    __shared__ float s_mean;
    __shared__ float warp_sums[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
        }
        if (lane_id == 0) {
            s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_mean;
    // Apply scale and weights with offset (e.g. 1.0 for Gemma)
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float w = weight != nullptr ? (weight[i] + weight_offset) : 1.0f;
        out_row[i] = in_row[i] * inv_rms * w;
    }
}

/**
 * Custom CUDA Kernel: Fused RMSNorm + Dynamic Q8_1 Activation Quantization.
 * Computes root-mean-square normalization and directly emits quantized BlockQ8_1_Gpu
 * activations with half2 (scale, sum) headers, eliminating separate quantization kernel launches
 * and avoiding memory round-tripping of float activations.
 */
__global__ void fused_rmsnorm_q8_1_cuda_kernel(
    float* __restrict__ output,
    BlockQ8_1_Gpu* __restrict__ y_q8_1,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden_dim,
    float eps,
    float weight_offset
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const float* in_row = input + bid * hidden_dim;
    float* out_row = output ? (output + bid * hidden_dim) : nullptr;
    BlockQ8_1_Gpu* out_q8_1 = y_q8_1 ? (y_q8_1 + bid * (hidden_dim / 32)) : nullptr;

    // 1. Compute sum of squares with grid-stride loop
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = in_row[i];
        sum_sq += val * val;
    }

    // Warp reduction using shuffle down
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    // Shared memory for block-level reduction
    __shared__ float s_mean;
    __shared__ float warp_sums[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
        }
        if (lane_id == 0) {
            s_mean = rsqrtf((b_sum / (float)hidden_dim) + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_mean;

    // 2. Apply scale and weights, and immediately quantize each 32-element warp block
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float w = weight != nullptr ? (weight[i] + weight_offset) : 1.0f;
        float norm_val = in_row[i] * inv_rms * w;
        if (out_row != nullptr) {
            out_row[i] = norm_val;
        }

        if (out_q8_1 != nullptr) {
            int block_32_idx = i / 32;
            float amax = fabsf(norm_val);
            float sum = norm_val;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
            }

            float d = amax / 127.0f;
            float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
            int8_t q = (int8_t)roundf(norm_val * id);
            out_q8_1[block_32_idx].qs[lane_id] = q;

            if (lane_id == 0) {
                out_q8_1[block_32_idx].ds = make_half2(__float2half(d), __float2half(sum));
            }
        }
    }
}

/**
 * Custom CUDA Kernel: Rotary Position Embedding (RoPE)
 * Applies rotary frequency phase rotation to Query and Key projections.
 */
__global__ void rope_embedding_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int half_dim = head_dim / 2;
    int d = threadIdx.x;

    if (d >= half_dim) return;

    int q_base = (token_idx * num_heads + head_idx) * head_dim;
    float cos_val = cos_table[token_idx * half_dim + d];
    float sin_val = sin_table[token_idx * half_dim + d];

    // Query rotation
    float q0 = q[q_base + d];
    float q1 = q[q_base + d + half_dim];
    q[q_base + d]            = q0 * cos_val - q1 * sin_val;
    q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;

    // Key rotation (if k pointer provided)
    if (k != nullptr) {
        float k0 = k[q_base + d];
        float k1 = k[q_base + d + half_dim];
        k[q_base + d]            = k0 * cos_val - k1 * sin_val;
        k[q_base + d + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}



/**
 * Packed GGUF Quantization Super-Block Structures for physical GPU memory-mapping.
 */
struct __attribute__((packed)) BlockQ8_0_Gpu {
    uint16_t scale;
    int8_t qs[32];
};

struct __attribute__((packed)) BlockQ6_K_Gpu {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

struct __attribute__((packed)) BlockPTQ1_0_Gpu {
    uint8_t qs[24];
    uint8_t qh[2];
    uint16_t d;
};

struct __attribute__((packed)) BlockQ2_K_Gpu {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct __attribute__((packed)) BlockQ3_K_Gpu {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};

struct __attribute__((packed)) BlockQ4_0_Gpu {
    uint16_t d;
    uint8_t qs[16];
};

struct __attribute__((packed)) BlockQ5_0_Gpu {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct __attribute__((packed)) BlockTQ2_0_Gpu {
    uint8_t qs[64];
    uint16_t d;
};

struct __attribute__((packed)) BlockQ4_1_Gpu {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};

struct __attribute__((packed)) BlockQ5_1_Gpu {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct __attribute__((packed)) BlockIQ4_NL_Gpu {
    uint16_t d;
    uint8_t qs[16];
};

__constant__ float c_kvalues_iq4nl[16] = {
    -31.0f, -22.0f, -15.0f, -10.0f, -6.0f, -3.0f, -1.0f, 0.0f,
      1.0f,   3.0f,   6.0f,  10.0f, 15.0f, 22.0f, 31.0f, 44.0f
};


struct __attribute__((packed)) BlockQ5_K_Gpu {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

static __device__ __forceinline__ int ptq1_0_trit_gpu(const BlockPTQ1_0_Gpu* blk, int e) {
    uint8_t b; int n;
    if (e < 80) {
        b = blk->qs[e & 15];
        n = e >> 4;
    } else if (e < 120) {
        int t = e - 80;
        b = blk->qs[16 + (t & 7)];
        n = t >> 3;
    } else {
        int t = e - 120;
        b = blk->qh[t & 1];
        n = t >> 1;
    }
    uint32_t v = b;
    for (int i = 0; i < n; ++i) {
        v = (v * 3) & 0xFF;
    }
    return (int)((v * 3) >> 8) - 1;
}

/**
 * Custom CUDA Kernel: Quantized Q4_0 Matrix-Vector GEMV (18-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q4_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    const BlockQ4_0_Gpu* row_blocks = (const BlockQ4_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 18);

    float warp_dot = 0.0f;
    for (int b = lane_id; b < num_blocks_per_row; b += 32) {
        const BlockQ4_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));

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
 * Custom CUDA Kernel: Quantized Q5_0 Matrix-Vector GEMV (22-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q5_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    const BlockQ5_0_Gpu* row_blocks = (const BlockQ5_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 22);

    float warp_dot = 0.0f;
    for (int b = lane_id; b < num_blocks_per_row; b += 32) {
        const BlockQ5_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        uint32_t qh = *((const uint32_t*)blk->qh);

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

/**
 * Custom CUDA Kernel: Quantized TQ2_0 (Ternary 2-bit, 2.06 bpw) Matrix-Vector GEMV (66-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_tq2_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockTQ2_0_Gpu* row_blocks = (const BlockTQ2_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 66);

    float warp_dot = 0.0f;
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockTQ2_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));

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
 * Custom CUDA Kernel: Quantized Q4_1 Matrix-Vector GEMV (20-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q4_1_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    const BlockQ4_1_Gpu* row_blocks = (const BlockQ4_1_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 20);

    float warp_dot = 0.0f;
    for (int b = lane_id; b < num_blocks_per_row; b += 32) {
        const BlockQ4_1_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        float min_val = __half2float(*((const half*)&blk->m));

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
 * Custom CUDA Kernel: Quantized Q5_1 Matrix-Vector GEMV (24-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q5_1_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    const BlockQ5_1_Gpu* row_blocks = (const BlockQ5_1_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 24);

    float warp_dot = 0.0f;
    for (int b = lane_id; b < num_blocks_per_row; b += 32) {
        const BlockQ5_1_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        float min_val = __half2float(*((const half*)&blk->m));

        uint32_t qh = *((const uint32_t*)blk->qh);

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
 * Custom CUDA Kernel: Non-linear IQ4_NL Matrix-Vector GEMV (18-byte Block x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_iq4_nl_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    const BlockIQ4_NL_Gpu* row_blocks = (const BlockIQ4_NL_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 18);

    float warp_dot = 0.0f;
    for (int b = lane_id; b < num_blocks_per_row; b += 32) {
        const BlockIQ4_NL_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));

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
 * Custom CUDA Kernel: Quantized PTQ1_0 (Ternary 1.58-bit / 1.75-bit) Matrix-Vector GEMV
 * (28-byte block: 128 elements packed in base-3 x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_ptq1_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 128;

    const BlockPTQ1_0_Gpu* row_blocks = (const BlockPTQ1_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 28);

    float warp_dot = 0.0f;
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockPTQ1_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
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
 * Custom CUDA Kernel: Quantized Q8_0 Multi-Column MMVQ (34-byte Block x Q8_1 Activations).
 * Evaluates BATCH_N token vectors simultaneously while reading weights once.
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
template <int BATCH_N, int ADD_TO_Y = 0>
__global__ void mmvq_q8_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const void* __restrict__ x_q8_1_raw,
    int m,
    int k,
    int actual_batch = BATCH_N
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 32;

    int batch_chunk = blockIdx.y;
    int col_offset = batch_chunk * 32;
    int num_cols = (BATCH_N == 1) ? 1 : ((actual_batch > col_offset) ? ((actual_batch - col_offset < 32) ? (actual_batch - col_offset) : 32) : 0);
    if (num_cols <= 0) return;

    const BlockQ8_0_Gpu* row_blocks = (const BlockQ8_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 34);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    float thread_dot[BATCH_N];
    for (int col = 0; col < num_cols; ++col) {
        thread_dot[col] = 0.0f;
    }

    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockQ8_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->scale));
        int8_t q = blk->qs[lane_id];

        for (int col = 0; col < num_cols; ++col) {
            const BlockQ8_1_Gpu* bq8_1 = &x_q8_1[(col_offset + col) * num_blocks_per_row + b];
            float d8 = __low2float(bq8_1->ds);
            int8_t u = bq8_1->qs[lane_id];
            thread_dot[col] += (d * d8) * (float)((int)q * (int)u);
        }
    }

    for (int col = 0; col < num_cols; ++col) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_dot[col] += __shfl_down_sync(0xFFFFFFFF, thread_dot[col], offset);
        }

        if (lane_id == 0) {
            if (ADD_TO_Y) {
                y[(col_offset + col) * m + row] += thread_dot[col];
            } else {
                y[(col_offset + col) * m + row] = thread_dot[col];
            }
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q6_K Multi-Column MMVQ (210-byte Super-block x Q8_1 Activations).
 * Evaluates BATCH_N token vectors simultaneously while reading weights once.
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
template <int BATCH_N, int ADD_TO_Y = 0>
__global__ void mmvq_q6_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const void* __restrict__ x_q8_1_raw,
    int m,
    int k,
    int actual_batch = BATCH_N
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;
    int num_q8_blocks_per_col = k / 32;

    int batch_chunk = blockIdx.y;
    int col_offset = batch_chunk * 32;
    int num_cols = (BATCH_N == 1) ? 1 : ((actual_batch > col_offset) ? ((actual_batch - col_offset < 32) ? (actual_batch - col_offset) : 32) : 0);
    if (num_cols <= 0) return;

    const BlockQ6_K_Gpu* row_blocks = (const BlockQ6_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 210);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    float thread_dot[BATCH_N];
    for (int col = 0; col < num_cols; ++col) {
        thread_dot[col] = 0.0f;
    }

    int is = lane_id / 16;
    #pragma unroll 2
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockQ6_K_Gpu* blk = &row_blocks[b];
        float d;
        if (lane_id == 0) {
            d = __half2float(*((const half*)&blk->d));
        }
        d = __shfl_sync(0xFFFFFFFF, d, 0);

        #pragma unroll
        for (int half_idx = 0; half_idx < 2; half_idx++) {
            int ql_offset = half_idx * 64;
            int qh_offset = half_idx * 32;
            int sc_offset = half_idx * 8;

            uint8_t qh_byte = blk->qh[qh_offset + lane_id];
            uint8_t ql0 = blk->ql[ql_offset + lane_id];
            uint8_t ql32 = blk->ql[ql_offset + lane_id + 32];

            int q1 = (int)((ql0 & 0x0F) | (((qh_byte >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql32 & 0x0F) | (((qh_byte >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) - 32;

            uint32_t my_w = 0;
            if (lane_id == 0) {
                uint16_t p0 = *((const uint16_t*)&blk->scales[sc_offset + 0]);
                uint16_t p1 = *((const uint16_t*)&blk->scales[sc_offset + 2]);
                my_w = (uint32_t)p0 | ((uint32_t)p1 << 16);
            } else if (lane_id == 1) {
                uint16_t p2 = *((const uint16_t*)&blk->scales[sc_offset + 4]);
                uint16_t p3 = *((const uint16_t*)&blk->scales[sc_offset + 6]);
                my_w = (uint32_t)p2 | ((uint32_t)p3 << 16);
            }
            uint32_t w0 = __shfl_sync(0xFFFFFFFF, my_w, 0);
            uint32_t w1 = __shfl_sync(0xFFFFFFFF, my_w, 1);

            int shift = is * 8;
            float sc0 = (float)(int8_t)((w0 >> (shift + 0)) & 0xFF);
            float sc1 = (float)(int8_t)((w0 >> (shift + 16)) & 0xFF);
            float sc2 = (float)(int8_t)((w1 >> (shift + 0)) & 0xFF);
            float sc3 = (float)(int8_t)((w1 >> (shift + 16)) & 0xFF);

            float d_sc0 = d * sc0;
            float d_sc1 = d * sc1;
            float d_sc2 = d * sc2;
            float d_sc3 = d * sc3;

            for (int col = 0; col < num_cols; ++col) {
                const BlockQ8_1_Gpu* bx = &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
                const BlockQ8_1_Gpu* bq0 = &bx[half_idx * 4 + 0];
                const BlockQ8_1_Gpu* bq1 = &bx[half_idx * 4 + 1];
                const BlockQ8_1_Gpu* bq2 = &bx[half_idx * 4 + 2];
                const BlockQ8_1_Gpu* bq3 = &bx[half_idx * 4 + 3];

                float c0 = __low2float(bq0->ds) * d_sc0;
                float c1 = __low2float(bq1->ds) * d_sc1;
                float c2 = __low2float(bq2->ds) * d_sc2;
                float c3 = __low2float(bq3->ds) * d_sc3;

                int u0 = (int)bq0->qs[lane_id];
                int u1 = (int)bq1->qs[lane_id];
                int u2 = (int)bq2->qs[lane_id];
                int u3 = (int)bq3->qs[lane_id];

                thread_dot[col] += c0 * (float)(q1 * u0) + c1 * (float)(q2 * u1) + c2 * (float)(q3 * u2) + c3 * (float)(q4 * u3);
            }
        }
    }

    for (int col = 0; col < num_cols; ++col) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_dot[col] += __shfl_down_sync(0xFFFFFFFF, thread_dot[col], offset);
        }

        if (lane_id == 0) {
            if (ADD_TO_Y) {
                y[(col_offset + col) * m + row] += thread_dot[col];
            } else {
                y[(col_offset + col) * m + row] = thread_dot[col];
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

__device__ __forceinline__ void get_scale_min_k4_cuda(int j, const uint8_t* scales, float* sc, float* m) {
    if (j < 4) {
        *sc = (float)(scales[j] & 63);
        *m  = (float)(scales[j + 4] & 63);
    } else {
        *sc = (float)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        *m  = (float)((scales[j + 4] >> 4)   | ((scales[j] >> 6) << 4));
    }
}

__device__ __forceinline__ void get_scale_min_k4_from_regs(
    int chunk,
    uint32_t sc_0_3,
    uint32_t sc_4_7,
    uint32_t sc_8_11,
    float* sc0, float* min0,
    float* sc1, float* min1
) {
    if (chunk < 2) {
        int shift = chunk * 16;
        uint32_t w0 = sc_0_3 >> shift;
        uint32_t w1 = sc_4_7 >> shift;
        uint8_t s0 = w0 & 0xFF;
        uint8_t s1 = (w0 >> 8) & 0xFF;
        uint8_t m0 = w1 & 0xFF;
        uint8_t m1 = (w1 >> 8) & 0xFF;
        *sc0  = (float)(s0 & 63);
        *min0 = (float)(m0 & 63);
        *sc1  = (float)(s1 & 63);
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
        *sc0  = (float)((s8 & 0x0F) | ((s0 >> 6) << 4));
        *min0 = (float)((s8 >> 4)   | ((s4 >> 6) << 4));
        *sc1  = (float)((s9 & 0x0F) | ((s1 >> 6) << 4));
        *min1 = (float)((s9 >> 4)   | ((s5 >> 6) << 4));
    }
}

/**
 * Custom CUDA Kernel: Quantized Q4_K Multi-Column MMVQ (INT4 Super-blocks x Q8_1 Activations via __dp4a).
 * Evaluates BATCH_N token vectors simultaneously while reading weights once.
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 * 128-byte coalesced weight loads + hardware INT8 dot products.
 */
template <int BATCH_N, int ADD_TO_Y = 0>
__global__ void mmvq_q4_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const void* __restrict__ x_q8_1_raw,
    int m,
    int k,
    int actual_batch = BATCH_N
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;
    int num_q8_blocks_per_col = k / 32;

    int batch_chunk = blockIdx.y;
    int col_offset = batch_chunk * 32;
    int num_cols = (BATCH_N == 1) ? 1 : ((actual_batch > col_offset) ? ((actual_batch - col_offset < 32) ? (actual_batch - col_offset) : 32) : 0);
    if (num_cols <= 0) return;

    const BlockQ4_K_Gpu* row_blocks = (const BlockQ4_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 144);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    float thread_dot[BATCH_N];
    for (int col = 0; col < num_cols; ++col) {
        thread_dot[col] = 0.0f;
    }

    #pragma unroll 2
    for (int b = 0; b < num_blocks_per_row; b++) {
        const BlockQ4_K_Gpu* blk = &row_blocks[b];
        uint4 hdr = ((const uint4*)blk)[0];
        float d = __low2float(*((const half2*)&hdr.x));
        float dmin = __high2float(*((const half2*)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

        float d_sc0 = d * sc0;
        float d_sc1 = d * sc1;
        float dm_min0 = dmin * min0;
        float dm_min1 = dmin * min1;

        int v = ((const int*)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        for (int col = 0; col < num_cols; ++col) {
            const BlockQ8_1_Gpu* bx = &x_q8_1[(col_offset + col) * num_q8_blocks_per_col + b * 8];
            const BlockQ8_1_Gpu* bq8_0 = &bx[sub0];
            const BlockQ8_1_Gpu* bq8_1 = &bx[sub1];

            float d8_0 = __low2float(bq8_0->ds);
            float d8_1 = __low2float(bq8_1->ds);

            int u0 = ((const int*)bq8_0->qs)[sub_idx];
            int u1 = ((const int*)bq8_1->qs)[sub_idx];

            int dot0 = __dp4a(v0, u0, 0);
            int dot1 = __dp4a(v1, u1, 0);
            int sum0 = __dp4a(0x01010101, u0, 0);
            int sum1 = __dp4a(0x01010101, u1, 0);

            float acc0 = d_sc0 * (float)dot0 - dm_min0 * (float)sum0;
            float acc1 = d_sc1 * (float)dot1 - dm_min1 * (float)sum1;

            thread_dot[col] += acc0 * d8_0 + acc1 * d8_1;
        }
    }

    for (int col = 0; col < num_cols; ++col) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_dot[col] += __shfl_down_sync(0xFFFFFFFF, thread_dot[col], offset);
        }

        if (lane_id == 0) {
            if (ADD_TO_Y) {
                y[(col_offset + col) * m + row] += thread_dot[col];
            } else {
                y[(col_offset + col) * m + row] = thread_dot[col];
            }
        }
    }
}

/**
 * Grouped Q4_K Gate + Up GEMV Kernel.
 * Computes both Gate (w1) and Up (w3) projections in a single kernel dispatch.
 * Total rows = 2 * inter_m. Rows [0, inter_m) -> w1 -> y_gate. Rows [inter_m, 2 * inter_m) -> w3 -> y_up.
 */
__global__ void gemv_gate_up_q4_k_cuda_kernel(
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    const void* __restrict__ raw_w1,
    const void* __restrict__ raw_w3,
    const void* __restrict__ x_q8_1_raw,
    int inter_m,
    int k
) {
    int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (total_row >= 2 * inter_m) return;

    bool is_up = (total_row >= inter_m);
    int row = is_up ? (total_row - inter_m) : total_row;
    const void* raw_blocks = is_up ? raw_w3 : raw_w1;
    float* y = is_up ? y_up : y_gate;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockQ4_K_Gpu* row_blocks = (const BlockQ4_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 144);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    float thread_dot = 0.0f;

    #pragma unroll 2
    for (int b = 0; b < num_blocks_per_row; b++) {
        const BlockQ4_K_Gpu* blk = &row_blocks[b];
        uint4 hdr = ((const uint4*)blk)[0];
        float d = __low2float(*((const half2*)&hdr.x));
        float dmin = __high2float(*((const half2*)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

        float d_sc0 = d * sc0;
        float d_sc1 = d * sc1;
        float dm_min0 = dmin * min0;
        float dm_min1 = dmin * min1;

        int v = ((const int*)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        const BlockQ8_1_Gpu* bx = &x_q8_1[b * 8];
        const BlockQ8_1_Gpu* bq8_0 = &bx[sub0];
        const BlockQ8_1_Gpu* bq8_1 = &bx[sub1];

        float d8_0 = __low2float(bq8_0->ds);
        float d8_1 = __low2float(bq8_1->ds);

        int u0 = ((const int*)bq8_0->qs)[sub_idx];
        int u1 = ((const int*)bq8_1->qs)[sub_idx];

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
 * Computes Q (wq), K (wk), and V (wv) projections in a single kernel dispatch when all are Q4_K.
 * Total rows = q_m + 2 * kv_m.
 * Rows [0, q_m) -> wq -> y_q.
 * Rows [q_m, q_m + kv_m) -> wk -> y_k.
 * Rows [q_m + kv_m, q_m + 2 * kv_m) -> wv -> y_v.
 */
__global__ void gemv_qkv_q4_k_cuda_kernel(
    float* __restrict__ y_q,
    float* __restrict__ y_k,
    float* __restrict__ y_v,
    const void* __restrict__ raw_wq,
    const void* __restrict__ raw_wk,
    const void* __restrict__ raw_wv,
    const void* __restrict__ x_q8_1_raw,
    int q_m,
    int kv_m,
    int k
) {
    int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int total_m = q_m + 2 * kv_m;
    if (total_row >= total_m) return;

    int row;
    const void* raw_blocks;
    float* y;

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

    const BlockQ4_K_Gpu* row_blocks = (const BlockQ4_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 144);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    float thread_dot = 0.0f;

    #pragma unroll 2
    for (int b = 0; b < num_blocks_per_row; b++) {
        const BlockQ4_K_Gpu* blk = &row_blocks[b];
        uint4 hdr = ((const uint4*)blk)[0];
        float d = __low2float(*((const half2*)&hdr.x));
        float dmin = __high2float(*((const half2*)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

        float d_sc0 = d * sc0;
        float d_sc1 = d * sc1;
        float dm_min0 = dmin * min0;
        float dm_min1 = dmin * min1;

        int v = ((const int*)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        const BlockQ8_1_Gpu* bx = &x_q8_1[b * 8];
        const BlockQ8_1_Gpu* bq8_0 = &bx[sub0];
        const BlockQ8_1_Gpu* bq8_1 = &bx[sub1];

        float d8_0 = __low2float(bq8_0->ds);
        float d8_1 = __low2float(bq8_1->ds);

        int u0 = ((const int*)bq8_0->qs)[sub_idx];
        int u1 = ((const int*)bq8_1->qs)[sub_idx];

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
 * Computes Q (wq) and K (wk) projections in a single kernel dispatch when wv has a different quant format.
 * Total rows = q_m + kv_m.
 */
__global__ void gemv_qk_q4_k_cuda_kernel(
    float* __restrict__ y_q,
    float* __restrict__ y_k,
    const void* __restrict__ raw_wq,
    const void* __restrict__ raw_wk,
    const void* __restrict__ x_q8_1_raw,
    int q_m,
    int kv_m,
    int k
) {
    int total_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int total_m = q_m + kv_m;
    if (total_row >= total_m) return;

    bool is_k = (total_row >= q_m);
    int row = is_k ? (total_row - q_m) : total_row;
    const void* raw_blocks = is_k ? raw_wk : raw_wq;
    float* y = is_k ? y_k : y_q;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockQ4_K_Gpu* row_blocks = (const BlockQ4_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 144);
    const BlockQ8_1_Gpu* x_q8_1 = (const BlockQ8_1_Gpu*)x_q8_1_raw;

    int chunk = lane_id / 8;
    int sub_idx = lane_id % 8;
    int sub0 = 2 * chunk + 0;
    int sub1 = 2 * chunk + 1;

    float thread_dot = 0.0f;

    #pragma unroll 2
    for (int b = 0; b < num_blocks_per_row; b++) {
        const BlockQ4_K_Gpu* blk = &row_blocks[b];
        uint4 hdr = ((const uint4*)blk)[0];
        float d = __low2float(*((const half2*)&hdr.x));
        float dmin = __high2float(*((const half2*)&hdr.x));

        float sc0, min0, sc1, min1;
        get_scale_min_k4_from_regs(chunk, hdr.y, hdr.z, hdr.w, &sc0, &min0, &sc1, &min1);

        float d_sc0 = d * sc0;
        float d_sc1 = d * sc1;
        float dm_min0 = dmin * min0;
        float dm_min1 = dmin * min1;

        int v = ((const int*)blk->qs)[lane_id];
        int v0 = v & 0x0F0F0F0F;
        int v1 = (v >> 4) & 0x0F0F0F0F;

        const BlockQ8_1_Gpu* bx = &x_q8_1[b * 8];
        const BlockQ8_1_Gpu* bq8_0 = &bx[sub0];
        const BlockQ8_1_Gpu* bq8_1 = &bx[sub1];

        float d8_0 = __low2float(bq8_0->ds);
        float d8_1 = __low2float(bq8_1->ds);

        int u0 = ((const int*)bq8_0->qs)[sub_idx];
        int u1 = ((const int*)bq8_1->qs)[sub_idx];

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
 * Custom CUDA Kernel: Quantized Q2_K Matrix-Vector GEMV (2-bit Super-blocks x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q2_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockQ2_K_Gpu* row_blocks = (const BlockQ2_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 84);

    float warp_dot = 0.0f;
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockQ2_K_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        float min = __half2float(*((const half*)&blk->dmin));
        const float* x_block = x + b * 256;

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
 * Custom CUDA Kernel: Quantized Q3_K Matrix-Vector GEMV (3-bit Super-blocks x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q3_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockQ3_K_Gpu* row_blocks = (const BlockQ3_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 110);

    float warp_dot = 0.0f;
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockQ3_K_Gpu* blk = &row_blocks[b];
        float d_all = __half2float(*((const half*)&blk->d));
        const float* x_block = x + b * 256;

        uint32_t aux[4];
        memcpy(aux, blk->scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4);
        const int8_t* scales = (const int8_t*)aux;

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
 * Custom CUDA Kernel: Quantized Q5_K Matrix-Vector GEMV (5-bit Super-blocks x FP32 Activations).
 * Warp-Per-Row architecture: 0 shared memory, 0 __syncthreads(), independent warp execution.
 */
__global__ void gemv_q5_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= m) return;

    int lane_id = threadIdx.x % 32;
    int num_blocks_per_row = k / 256;

    const BlockQ5_K_Gpu* row_blocks = (const BlockQ5_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 176);

    float warp_dot = 0.0f;
    for (int b = 0; b < num_blocks_per_row; ++b) {
        const BlockQ5_K_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        float dmin = __half2float(*((const half*)&blk->dmin));
        const float* x_block = x + b * 256;

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
            float w2 = d2 * (float)((ql >> 4)   + ((qh & u2) ? 16 : 0)) - min2;

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
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling + Online Softmax)
 */
__global__ void vector_add_f32_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += b[idx];
    }
}

__global__ void vector_scale_f32_kernel(
    float* __restrict__ x,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= scale;
    }
}

__global__ void swiglu_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu = g / (1.0f + expf(-g));
        out[idx] = silu * up[idx];
    }
}

/**
 * Custom CUDA Kernel: Fused SwiGLU + Dynamic Q8_1 Activation Quantization.
 * Directly transforms gate and up projections via silu(gate) * up and emits
 * quantized BlockQ8_1_Gpu blocks for the down-projection GEMV.
 */
__global__ void fused_swiglu_q8_1_cuda_kernel(
    BlockQ8_1_Gpu* __restrict__ y_q8_1,
    float* __restrict__ output_f32,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int lane_id = threadIdx.x % 32;
    int block_32_idx = i / 32;

    float g = gate[i];
    float u = up[i];
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u;

    if (output_f32 != nullptr) {
        output_f32[i] = val;
    }

    if (y_q8_1 != nullptr) {
        float amax = fabsf(val);
        float sum = val;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset));
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }

        float d = amax / 127.0f;
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        int8_t q = (int8_t)roundf(val * id);
        y_q8_1[block_32_idx].qs[lane_id] = q;

        if (lane_id == 0) {
            y_q8_1[block_32_idx].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

__global__ void gelu_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        const float SQRT_2_OVER_PI = 0.7978845608f;
        float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        out[idx] = gelu * up[idx];
    }
}

__global__ void logit_softcap_f32_kernel(
    float* __restrict__ logits,
    int vocab_size,
    float cap
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        float l = logits[idx];
        logits[idx] = cap * tanhf(l / cap);
    }
}

__device__ inline float compute_rope_freq_yarn(
    int d,
    int head_dim,
    float theta,
    float scaling_factor,
    int orig_ctx,
    float ext_factor = 1.0f,
    float beta_fast = 32.0f,
    float beta_slow = 1.0f
) {
    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    if (scaling_factor > 1.0f && orig_ctx > 0) {
        float wavelength = 6.28318530718f / freq;
        float ramp = ((float)orig_ctx / wavelength - beta_slow) / (beta_fast - beta_slow);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        float freq_interp = freq / scaling_factor;
        freq = (1.0f - ramp) * freq_interp + ramp * (freq * ext_factor);
    }
    return freq;
}

__global__ void convert_f32_to_f16_kernel(
    __half* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void rope_single_token_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    void* __restrict__ k_out,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    int is_fp16 = 0,
    float scaling_factor = 1.0f,
    int orig_ctx = 0
) {
    int head_idx = blockIdx.x;
    int half_dim = head_dim / 2;
    int d = threadIdx.x;
    if (d >= half_dim) return;

    float freq = compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
    float angle = (float)pos * freq;
    float sin_val, cos_val;
    __sincosf(angle, &sin_val, &cos_val);

    if (head_idx < num_heads && q != nullptr) {
        int q_base = head_idx * head_dim;
        float q0 = q[q_base + d];
        float q1 = q[q_base + d + half_dim];
        q[q_base + d]            = q0 * cos_val - q1 * sin_val;
        q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;
    }

    if (head_idx < num_kv_heads && k != nullptr) {
        int k_base = head_idx * head_dim;
        float k0 = k[k_base + d];
        float k1 = k[k_base + d + half_dim];
        float rot0 = k0 * cos_val - k1 * sin_val;
        float rot1 = k0 * sin_val + k1 * cos_val;
        k[k_base + d]            = rot0;
        k[k_base + d + half_dim] = rot1;

        if (k_out != nullptr) {
            if (is_fp16) {
                __half* dst_h = (__half*)k_out;
                dst_h[k_base + d]            = __float2half(rot0);
                dst_h[k_base + d + half_dim] = __float2half(rot1);
            } else {
                float* dst_f = (float*)k_out;
                dst_f[k_base + d]            = rot0;
                dst_f[k_base + d + half_dim] = rot1;
            }
        }
    }
}

__global__ void rope_batch_tokens_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    void* __restrict__ k_cache,
    int start_pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    int is_fp16 = 0,
    float scaling_factor = 1.0f,
    int orig_ctx = 0
) {
    int head_idx = blockIdx.x;
    int b = blockIdx.y;
    if (b >= batch_size) return;

    int half_dim = head_dim / 2;
    int d = threadIdx.x;
    if (d >= half_dim) return;

    int token_pos = start_pos + b;
    float freq = compute_rope_freq_yarn(d, head_dim, theta, scaling_factor, orig_ctx);
    float angle = (float)token_pos * freq;
    float sin_val, cos_val;
    __sincosf(angle, &sin_val, &cos_val);

    if (head_idx < num_heads && q != nullptr) {
        int q_base = b * (num_heads * head_dim) + head_idx * head_dim;
        float q0 = q[q_base + d];
        float q1 = q[q_base + d + half_dim];
        q[q_base + d]            = q0 * cos_val - q1 * sin_val;
        q[q_base + d + half_dim] = q0 * sin_val + q1 * cos_val;
    }

    if (head_idx < num_kv_heads && k != nullptr) {
        int k_base = b * (num_kv_heads * head_dim) + head_idx * head_dim;
        float k0 = k[k_base + d];
        float k1 = k[k_base + d + half_dim];
        float rot0 = k0 * cos_val - k1 * sin_val;
        float rot1 = k0 * sin_val + k1 * cos_val;
        k[k_base + d]            = rot0;
        k[k_base + d + half_dim] = rot1;

        if (k_cache != nullptr) {
            int cache_base = token_pos * (num_kv_heads * head_dim) + head_idx * head_dim;
            if (is_fp16) {
                __half* dst_h = (__half*)k_cache;
                dst_h[cache_base + d]            = __float2half(rot0);
                dst_h[cache_base + d + half_dim] = __float2half(rot1);
            } else {
                float* dst_f = (float*)k_cache;
                dst_f[cache_base + d]            = rot0;
                dst_f[cache_base + d + half_dim] = rot1;
            }
        }
    }
}

/**
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling + Online Softmax + GQA)
 * Warp-Per-Head architecture: 0 shared memory, 0 __syncthreads(), independent warp registers.
 * Supports FP16 and FP32 KV cache.
 */
__global__ void flash_decode_paged_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_kv_tokens,
    int start_token,
    float sm_scale,
    int is_fp16 = 0
) {
    if (num_heads <= 0 || head_dim <= 0 || num_kv_tokens <= 0) return;
    int head_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (head_idx >= num_heads) return;
    int lane_id = threadIdx.x % 32;

    const float* q_ptr = q + head_idx * head_dim;
    float* out_ptr = output + head_idx * head_dim;

    int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
    int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
    int kv_head_idx = head_idx / heads_per_kv;
    if (kv_head_idx >= kv_heads) kv_head_idx = kv_heads - 1;

    float max_score = -1e20f;
    float sum_exp = 0.0f;
    float acc_v[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc_v[i] = 0.0f;

    // Cache Q values into thread-private registers once
    float q_reg[16];
    int q_items = 0;
    for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
        q_reg[q_items] = q_ptr[d];
    }

    int valid_start = (start_token >= 0 && start_token < num_kv_tokens) ? start_token : 0;

    // Grid-stride over past KV cached tokens
    for (int t = valid_start; t < num_kv_tokens; t++) {
        size_t tok_offset = (size_t)(t * kv_heads + kv_head_idx) * head_dim;

        // Compute Q . K dot product for this head from registers
        float score = 0.0f;
        if (is_fp16) {
            const __half* k_tok = (const __half*)k_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                score += q_reg[i] * __half2float(k_tok[lane_id + i * 32]);
            }
        } else {
            const float* k_tok = (const float*)k_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                score += q_reg[i] * k_tok[lane_id + i * 32];
            }
        }

        // Warp reduction across the 32 lanes
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        // Broadcast total score to all 32 lanes in warp
        float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;

        // Online Softmax update with SFU expf bypass when maximum does not change
        float new_max = fmaxf(max_score, cur_score);
        float alpha = (t == valid_start) ? 0.0f : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
        float beta = expf(cur_score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * alpha + beta;

        if (is_fp16) {
            const __half* v_tok = (const __half*)v_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                acc_v[i] = acc_v[i] * alpha + beta * __half2float(v_tok[lane_id + i * 32]);
            }
        } else {
            const float* v_tok = (const float*)v_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                acc_v[i] = acc_v[i] * alpha + beta * v_tok[lane_id + i * 32];
            }
        }
    }

    // Final normalization
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < q_items; i++) {
        out_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
    }
}

/**
 * Custom CUDA Kernel: Batched Causal FlashAttention Prefill Kernel
 * Evaluates causal attention for all B tokens in prompt chunk across past KV cache in single pass.
 * Grid: dim3 grid((num_heads + 7) / 8, batch_size)
 * Block: 256 threads (8 warps, 1 warp per head).
 * Supports FP16 and FP32 KV cache.
 */
__global__ void flash_causal_batched_prefill_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    int start_pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int kv_cap,
    float sm_scale,
    int is_fp16 = 0
) {
    if (num_heads <= 0 || head_dim <= 0 || batch_size <= 0) return;
    int b = blockIdx.y;
    if (b >= batch_size) return;

    int head_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (head_idx >= num_heads) return;
    int lane_id = threadIdx.x % 32;

    int token_pos = start_pos + b;
    int num_kv_tokens = token_pos + 1;
    if (kv_cap > 0 && num_kv_tokens > kv_cap) {
        num_kv_tokens = kv_cap;
    }

    int q_dim = num_heads * head_dim;
    const float* q_ptr = q + b * q_dim + head_idx * head_dim;
    float* out_ptr = output + b * q_dim + head_idx * head_dim;

    int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
    int heads_per_kv = (num_heads / kv_heads) > 0 ? (num_heads / kv_heads) : 1;
    int kv_head_idx = head_idx / heads_per_kv;
    if (kv_head_idx >= kv_heads) kv_head_idx = kv_heads - 1;

    float max_score = -1e20f;
    float sum_exp = 0.0f;
    float acc_v[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc_v[i] = 0.0f;

    // Cache Q values into thread-private registers once
    float q_reg[16];
    int q_items = 0;
    for (int d = lane_id; d < head_dim && q_items < 16; d += 32, q_items++) {
        q_reg[q_items] = q_ptr[d];
    }

    // Grid-stride over past KV cached tokens (causal: up to token_pos)
    for (int t = 0; t < num_kv_tokens; t++) {
        size_t tok_offset = (size_t)(t * kv_heads + kv_head_idx) * head_dim;

        // Compute Q . K dot product for this head from registers
        float score = 0.0f;
        if (is_fp16) {
            const __half* k_tok = (const __half*)k_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                score += q_reg[i] * __half2float(k_tok[lane_id + i * 32]);
            }
        } else {
            const float* k_tok = (const float*)k_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                score += q_reg[i] * k_tok[lane_id + i * 32];
            }
        }

        // Warp reduction across the 32 lanes
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        // Broadcast total score to all 32 lanes in warp
        float cur_score = __shfl_sync(0xFFFFFFFF, score, 0) * sm_scale;

        // Online Softmax update with SFU expf bypass
        float new_max = fmaxf(max_score, cur_score);
        float alpha = (t == 0) ? 0.0f : ((new_max == max_score) ? 1.0f : expf(max_score - new_max));
        float beta = expf(cur_score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * alpha + beta;

        if (is_fp16) {
            const __half* v_tok = (const __half*)v_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                acc_v[i] = acc_v[i] * alpha + beta * __half2float(v_tok[lane_id + i * 32]);
            }
        } else {
            const float* v_tok = (const float*)v_cache + tok_offset;
            for (int i = 0; i < q_items; i++) {
                acc_v[i] = acc_v[i] * alpha + beta * v_tok[lane_id + i * 32];
            }
        }
    }

    // Final normalization
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < q_items; i++) {
        out_ptr[lane_id + i * 32] = acc_v[i] * inv_sum;
    }
}

// C-ABI Launch Wrappers for FFI

extern "C" {

int launch_cuda_rmsnorm(
    float* output,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float eps,
    float weight_offset,
    cudaStream_t stream
) {
    int block_size = 256;
    if (hidden_dim < 256) block_size = hidden_dim;
    rms_norm_f32_kernel<<<num_tokens, block_size, 0, stream>>>(
        output, input, weight, hidden_dim, eps, weight_offset
    );
    return cudaGetLastError();
}

int launch_cuda_fused_rmsnorm_q8_1(
    float* output,
    void* y_q8_1,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float eps,
    float weight_offset,
    cudaStream_t stream
) {
    if (hidden_dim <= 0 || num_tokens <= 0) return 0;
    int block_size = 256;
    if (hidden_dim >= 512 && hidden_dim % 512 == 0) {
        block_size = 512;
    }
    fused_rmsnorm_q8_1_cuda_kernel<<<num_tokens, block_size, 0, stream>>>(
        output, (BlockQ8_1_Gpu*)y_q8_1, input, weight, hidden_dim, eps, weight_offset
    );
    return cudaGetLastError();
}

int launch_cuda_rope(
    float* q,
    float* k,
    const float* cos_table,
    const float* sin_table,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_heads);
    int block_size = head_dim / 2;
    rope_embedding_kernel<<<grid, block_size, 0, stream>>>(
        q, k, cos_table, sin_table, seq_len, num_heads, head_dim
    );
    return cudaGetLastError();
}

static inline void get_gemv_launch_dims(int m, int& grid, int& block) {
    const int warps_per_block = 8;
    block = warps_per_block * 32; // 256 threads
    grid = (m + warps_per_block - 1) / warps_per_block;
}

int launch_cuda_gemv_q4_0(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q4_0_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q5_0(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q5_0_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_tq2_0(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_tq2_0_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_quantize_q8_1(
    const float* x,
    void* y_q8_1,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return 0;
    int num_blocks = n / 32;
    int warps_per_block = 8; // 256 threads
    int block_size = warps_per_block * 32;
    int grid = (num_blocks + warps_per_block - 1) / warps_per_block;
    quantize_f32_to_q8_1_cuda_kernel<<<grid, block_size, 0, stream>>>(
        x, (BlockQ8_1_Gpu*)y_q8_1, n
    );
    return cudaGetLastError();
}

int launch_cuda_mmvq_q8_0(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    int batch_size,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0 || batch_size <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    if (batch_size == 1) {
        mmvq_q8_0_cuda_kernel<1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
    } else {
        int num_chunks = (batch_size + 31) / 32;
        dim3 grid_2d(grid, num_chunks);
        mmvq_q8_0_cuda_kernel<32><<<grid_2d, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
    }
    return cudaGetLastError();
}

int launch_cuda_gemv_q8_0(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    return launch_cuda_mmvq_q8_0(y, raw_blocks, x_q8_1, m, k, 1, stream);
}

int launch_cuda_gemv_q8_0_fused_add(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    mmvq_q8_0_cuda_kernel<1, 1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
    return cudaGetLastError();
}

int launch_cuda_mmvq_q6_k(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    int batch_size,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0 || batch_size <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    if (batch_size == 1) {
        mmvq_q6_k_cuda_kernel<1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
    } else {
        int num_chunks = (batch_size + 31) / 32;
        dim3 grid_2d(grid, num_chunks);
        mmvq_q6_k_cuda_kernel<32><<<grid_2d, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, batch_size);
    }
    return cudaGetLastError();
}

int launch_cuda_gemv_q6_k(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    return launch_cuda_mmvq_q6_k(y, raw_blocks, x_q8_1, m, k, 1, stream);
}

int launch_cuda_gemv_q6_k_fused_add(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    mmvq_q6_k_cuda_kernel<1, 1><<<grid, block, 0, stream>>>(y, raw_blocks, x_q8_1, m, k, 1);
    return cudaGetLastError();
}

int launch_cuda_mmvq_q4_k(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    int batch_size,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0 || batch_size <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    if (batch_size == 1) {
        mmvq_q4_k_cuda_kernel<1><<<grid, block, 0, stream>>>(y, (const uint8_t*)raw_blocks, x_q8_1, m, k, 1);
    } else {
        int num_chunks = (batch_size + 31) / 32;
        dim3 grid_2d(grid, num_chunks);
        mmvq_q4_k_cuda_kernel<32><<<grid_2d, block, 0, stream>>>(y, (const uint8_t*)raw_blocks, x_q8_1, m, k, batch_size);
    }
    return cudaGetLastError();
}

int launch_cuda_gemv_q4_k(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    return launch_cuda_mmvq_q4_k(y, raw_blocks, x_q8_1, m, k, 1, stream);
}

int launch_cuda_gemv_q4_k_fused_add(
    float* y,
    const void* raw_blocks,
    const void* x_q8_1,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    mmvq_q4_k_cuda_kernel<1, 1><<<grid, block, 0, stream>>>(y, (const uint8_t*)raw_blocks, x_q8_1, m, k, 1);
    return cudaGetLastError();
}

int launch_cuda_gemv_gate_up_q4_k(
    float* y_gate,
    float* y_up,
    const void* raw_w1,
    const void* raw_w3,
    const void* x_q8_1,
    int inter_m,
    int k,
    cudaStream_t stream
) {
    if (inter_m <= 0 || k <= 0) return 0;
    int total_m = 2 * inter_m;
    int grid, block;
    get_gemv_launch_dims(total_m, grid, block);
    gemv_gate_up_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y_gate, y_up, raw_w1, raw_w3, x_q8_1, inter_m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_qkv_q4_k(
    float* y_q,
    float* y_k,
    float* y_v,
    const void* raw_wq,
    const void* raw_wk,
    const void* raw_wv,
    const void* x_q8_1,
    int q_m,
    int kv_m,
    int k,
    cudaStream_t stream
) {
    if (q_m <= 0 || kv_m <= 0 || k <= 0) return 0;
    int total_m = q_m + 2 * kv_m;
    int grid, block;
    get_gemv_launch_dims(total_m, grid, block);
    gemv_qkv_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y_q, y_k, y_v, raw_wq, raw_wk, raw_wv, x_q8_1, q_m, kv_m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_qk_q4_k(
    float* y_q,
    float* y_k,
    const void* raw_wq,
    const void* raw_wk,
    const void* x_q8_1,
    int q_m,
    int kv_m,
    int k,
    cudaStream_t stream
) {
    if (q_m <= 0 || kv_m <= 0 || k <= 0) return 0;
    int total_m = q_m + kv_m;
    int grid, block;
    get_gemv_launch_dims(total_m, grid, block);
    gemv_qk_q4_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y_q, y_k, raw_wq, raw_wk, x_q8_1, q_m, kv_m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_ptq1_0(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_ptq1_0_cuda_kernel<<<grid, block, 0, stream>>>(
        y, (const uint8_t*)raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q4_1(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (!y || !raw_blocks || !x || m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q4_1_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
    return cudaGetLastError();
}

int launch_cuda_gemv_q5_1(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (!y || !raw_blocks || !x || m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q5_1_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
    return cudaGetLastError();
}

int launch_cuda_gemv_iq4_nl(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (!y || !raw_blocks || !x || m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_iq4_nl_cuda_kernel<<<grid, block, 0, stream>>>(y, raw_blocks, x, m, k);
    return cudaGetLastError();
}

int launch_cuda_gemv_q2_k(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q2_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q3_k(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q3_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q5_k(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return 0;
    int grid, block;
    get_gemv_launch_dims(m, grid, block);
    gemv_q5_k_cuda_kernel<<<grid, block, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_vector_add(
    float* a,
    const float* b,
    int n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    vector_add_f32_kernel<<<grid, block_size, 0, stream>>>(a, b, n);
    return cudaGetLastError();
}

int launch_cuda_vector_scale(
    float* x,
    float scale,
    int n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    vector_scale_f32_kernel<<<grid, block_size, 0, stream>>>(x, scale, n);
    return cudaGetLastError();
}

int launch_cuda_swiglu(
    float* out,
    const float* gate,
    const float* up,
    int n,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    swiglu_f32_kernel<<<grid, block_size, 0, stream>>>(out, gate, up, n);
    return cudaGetLastError();
}

int launch_cuda_fused_swiglu_q8_1(
    void* y_q8_1,
    float* output_f32,
    const float* gate,
    const float* up,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return 0;
    int block_size = 256;
    int grid = (n + block_size - 1) / block_size;
    fused_swiglu_q8_1_cuda_kernel<<<grid, block_size, 0, stream>>>(
        (BlockQ8_1_Gpu*)y_q8_1, output_f32, gate, up, n
    );
    return cudaGetLastError();
}

int launch_cuda_rope_single(
    float* q,
    float* k,
    void* k_out,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    int is_fp16,
    float scaling_factor,
    int orig_ctx,
    cudaStream_t stream
) {
    int grid = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    int block_size = head_dim / 2;
    rope_single_token_kernel<<<grid, block_size, 0, stream>>>(
        q, k, k_out, pos, num_heads, num_kv_heads, head_dim, theta, is_fp16, scaling_factor, orig_ctx
    );
    return cudaGetLastError();
}

int launch_cuda_flash_decode(
    float* output,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_kv_tokens,
    int start_token,
    float sm_scale,
    int is_fp16,
    cudaStream_t stream
) {
    if (num_heads <= 0 || head_dim <= 0 || num_kv_tokens <= 0) return 0;
    const int block_size = 32; // 1 warp per block = maximum SM distribution
    int grid = num_heads;
    flash_decode_paged_kernel<<<grid, block_size, 0, stream>>>(
        output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, num_kv_tokens, start_token, sm_scale, is_fp16
    );
    return cudaGetLastError();
}

int launch_cuda_rope_batch(
    float* q,
    float* k,
    void* k_cache,
    int start_pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    int is_fp16,
    float scaling_factor,
    int orig_ctx,
    cudaStream_t stream
) {
    if (batch_size <= 0 || head_dim <= 0) return 0;
    int max_heads = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    dim3 grid(max_heads, batch_size);
    int block_size = head_dim / 2;
    rope_batch_tokens_kernel<<<grid, block_size, 0, stream>>>(
        q, k, k_cache, start_pos, batch_size, num_heads, num_kv_heads, head_dim, theta, is_fp16, scaling_factor, orig_ctx
    );
    return cudaGetLastError();
}

int launch_cuda_flash_causal_prefill(
    float* output,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    int start_pos,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int kv_cap,
    float sm_scale,
    int is_fp16,
    cudaStream_t stream
) {
    if (batch_size <= 0 || num_heads <= 0 || head_dim <= 0) return 0;
    const int block_size = 256; // 8 warps per block = 8 heads per block
    dim3 grid((num_heads + 7) / 8, batch_size);
    flash_causal_batched_prefill_kernel<<<grid, block_size, 0, stream>>>(
        output, q, k_cache, v_cache, start_pos, batch_size, num_heads, num_kv_heads, head_dim, kv_cap, sm_scale, is_fp16
    );
    return cudaGetLastError();
}

int launch_cuda_convert_f32_to_f16(
    void* dst,
    const float* src,
    int n,
    cudaStream_t stream
) {
    if (!dst || !src || n <= 0) return 0;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    convert_f32_to_f16_kernel<<<grid_size, block_size, 0, stream>>>((__half*)dst, src, n);
    return cudaGetLastError();
}


/**
 * Custom CUDA Kernel: Per-head QK-Norm (RMSNorm per head across head_dim).
 * Applied to Q and K separately before RoPE embedding.
 */
__global__ void qk_norm_f32_kernel(
    float* __restrict__ qk,
    const float* __restrict__ weight,
    int num_heads,
    int head_dim,
    float eps
) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    int tid = threadIdx.x;
    float* head_vec = qk + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = head_vec[i];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float s_inv_rms;
    __shared__ float warp_sums[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float b_sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            b_sum += __shfl_down_sync(0xFFFFFFFF, b_sum, offset);
        }
        if (lane_id == 0) {
            s_inv_rms = rsqrtf((b_sum / (float)head_dim) + eps);
        }
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        head_vec[i] = head_vec[i] * inv_rms * weight[i];
    }
}

int launch_cuda_qk_norm(
    float* qk,
    const float* weight,
    int num_heads,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    if (!qk || !weight || num_heads <= 0 || head_dim <= 0) return 0;
    int block_size = 128;
    if (head_dim > 128) block_size = 256;
    if (head_dim < 128) block_size = head_dim;
    qk_norm_f32_kernel<<<num_heads, block_size, 0, stream>>>(
        qk, weight, num_heads, head_dim, eps
    );
    return cudaGetLastError();
}

int launch_cuda_gelu(
    float* out,
    const float* gate,
    const float* up,
    int n,
    cudaStream_t stream
) {
    if (!out || !gate || !up || n <= 0) return 0;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    gelu_f32_kernel<<<grid_size, block_size, 0, stream>>>(out, gate, up, n);
    return cudaGetLastError();
}

int launch_cuda_logit_softcap(
    float* logits,
    int vocab_size,
    float cap,
    cudaStream_t stream
) {
    if (!logits || vocab_size <= 0 || cap <= 0.0f) return 0;
    int block_size = 256;
    int grid_size = (vocab_size + block_size - 1) / block_size;
    logit_softcap_f32_kernel<<<grid_size, block_size, 0, stream>>>(logits, vocab_size, cap);
    return cudaGetLastError();
}

/**
 * Custom CUDA Kernel: Single-pass Vocab Argmax Reduction
 * Computes argmax token index and maximum logit value directly on GPU.
 * Bypasses 1MB host-device memory transfers on every decode step.
 */
__global__ void argmax_f32_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    uint32_t* __restrict__ out_token,
    float* __restrict__ out_max_val
) {
    int token_idx = blockIdx.x;
    const float* token_logits = logits + (size_t)token_idx * vocab_size;

    __shared__ float s_max_val[32];
    __shared__ uint32_t s_max_idx[32];

    int tid = threadIdx.x;
    float max_v = -1e30f;
    uint32_t max_i = 0;

    // Thread stride across vocab
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = token_logits[i];
        if (v > max_v) {
            max_v = v;
            max_i = (uint32_t)i;
        }
    }

    // Warp-level reduction across 32 threads
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_v = __shfl_down_sync(0xFFFFFFFF, max_v, offset);
        uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, max_i, offset);
        if (other_v > max_v) {
            max_v = other_v;
            max_i = other_i;
        }
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_max_val[warp_id] = max_v;
        s_max_idx[warp_id] = max_i;
    }
    __syncthreads();

    // First warp reduces the 32 warp maximums
    if (warp_id == 0) {
        int num_warps = blockDim.x / 32;
        float b_max_v = (lane_id < num_warps) ? s_max_val[lane_id] : -1e30f;
        uint32_t b_max_i = (lane_id < num_warps) ? s_max_idx[lane_id] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_v = __shfl_down_sync(0xFFFFFFFF, b_max_v, offset);
            uint32_t other_i = __shfl_down_sync(0xFFFFFFFF, b_max_i, offset);
            if (other_v > b_max_v) {
                b_max_v = other_v;
                b_max_i = other_i;
            }
        }

        if (lane_id == 0) {
            if (out_token) out_token[token_idx] = b_max_i;
            if (out_max_val) out_max_val[token_idx] = b_max_v;
        }
    }
}

int launch_cuda_argmax(
    const float* logits,
    int vocab_size,
    uint32_t* out_token,
    float* out_max_val,
    int batch_size,
    cudaStream_t stream
) {
    if (!logits || vocab_size <= 0 || !out_token || batch_size <= 0) return 0;
    int block_size = 1024;
    argmax_f32_kernel<<<batch_size, block_size, 0, stream>>>(logits, vocab_size, out_token, out_max_val);
    return cudaGetLastError();
}

void init_cuda_kernel_attributes() {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    cudaFuncSetAttribute(mmvq_q4_k_cuda_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q4_k_cuda_kernel<32>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q4_k_cuda_kernel<1, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q6_k_cuda_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q6_k_cuda_kernel<32>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q6_k_cuda_kernel<1, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q8_0_cuda_kernel<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q8_0_cuda_kernel<32>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(mmvq_q8_0_cuda_kernel<1, 1>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(fused_rmsnorm_q8_1_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(fused_swiglu_q8_1_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_gate_up_q4_k_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_qkv_q4_k_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaFuncSetAttribute(gemv_qk_q4_k_cuda_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
}

}


