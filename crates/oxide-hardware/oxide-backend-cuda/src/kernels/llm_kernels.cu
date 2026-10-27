#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" {

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
 * Custom CUDA Kernel: Quantized Q4_0 Matrix-Vector GEMV (INT4 Weights x FP32 Activations).
 * Each block computes 1 output logit/activation by streaming through 32-weight blocks.
 */
__global__ void gemv_q4_0_cuda_kernel(
    float* __restrict__ y,
    const uint8_t* __restrict__ weight_bytes,
    const float* __restrict__ x,
    const half* __restrict__ scales,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int num_blocks_per_row = k / 32;

    float acc = 0.0f;
    for (int b = tid; b < num_blocks_per_row; b += blockDim.x) {
        int weight_byte_offset = (row * num_blocks_per_row + b) * 16;
        float d = __half2float(scales[row * num_blocks_per_row + b]);

        #pragma unroll 16
        for (int i = 0; i < 16; i++) {
            uint8_t byte = weight_bytes[weight_byte_offset + i];
            float q0 = (float)((int8_t)(byte & 0x0F) - 8);
            float q1 = (float)((int8_t)((byte >> 4) & 0x0F) - 8);

            float a0 = x[b * 32 + i];
            float a1 = x[b * 32 + i + 16];

            acc += (q0 * a0 + q1 * a1) * d;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float s_warp_acc[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_warp_acc[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < (blockDim.x / 32)) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
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
 * Custom CUDA Kernel: Quantized PTQ1_0 (Ternary 1.58-bit / 1.75-bit) Matrix-Vector GEMV
 * (28-byte block: 128 elements packed in base-3 x FP32 Activations).
 */
__global__ void gemv_ptq1_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 128;

    const BlockPTQ1_0_Gpu* row_blocks = (const BlockPTQ1_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 28);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
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

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q8_0 Matrix-Vector GEMV (34-byte Interleaved Block x FP32 Activations).
 */
__global__ void gemv_q8_0_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 32;

    const BlockQ8_0_Gpu* row_blocks = (const BlockQ8_0_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 34);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
        const BlockQ8_0_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->scale));
        float q = (float)blk->qs[lane_id];
        float x_val = x[b * 32 + lane_id];
        warp_dot += d * q * x_val;
    }

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q6_K Matrix-Vector GEMV (210-byte Super-block x FP32 Activations).
 */
__global__ void gemv_q6_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 256;

    const BlockQ6_K_Gpu* row_blocks = (const BlockQ6_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 210);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
        const BlockQ6_K_Gpu* blk = &row_blocks[b];
        float d = __half2float(*((const half*)&blk->d));
        const float* x_block = x + b * 256;

        #pragma unroll 2
        for (int half_idx = 0; half_idx < 2; half_idx++) {
            int ql_offset = half_idx * 64;
            int qh_offset = half_idx * 32;
            int sc_offset = half_idx * 8;
            int y_offset = half_idx * 128;

            int is = lane_id / 16;
            uint8_t qh_byte = blk->qh[qh_offset + lane_id];
            uint8_t ql0 = blk->ql[ql_offset + lane_id];
            uint8_t ql32 = blk->ql[ql_offset + lane_id + 32];

            int8_t q1 = (int8_t)((ql0 & 0x0F) | (((qh_byte >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql32 & 0x0F) | (((qh_byte >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) - 32;

            float sc0 = (float)blk->scales[sc_offset + is + 0];
            float sc2 = (float)blk->scales[sc_offset + is + 2];
            float sc4 = (float)blk->scales[sc_offset + is + 4];
            float sc6 = (float)blk->scales[sc_offset + is + 6];

            float w0 = d * sc0 * (float)q1;
            float w1 = d * sc2 * (float)q2;
            float w2 = d * sc4 * (float)q3;
            float w3 = d * sc6 * (float)q4;

            float x0 = x_block[y_offset + lane_id + 0];
            float x1 = x_block[y_offset + lane_id + 32];
            float x2 = x_block[y_offset + lane_id + 64];
            float x3 = x_block[y_offset + lane_id + 96];

            warp_dot += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        }
    }

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
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

/**
 * Custom CUDA Kernel: Quantized Q4_K Matrix-Vector GEMV (INT4 Super-blocks x FP32 Activations).
 * Unpacks 144-byte Q4_K super-blocks directly across warp lanes with coalesced 32-byte loads.
 */
__global__ void gemv_q4_k_cuda_kernel(
    float* __restrict__ y,
    const uint8_t* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 256;

    const BlockQ4_K_Gpu* row_blocks = (const BlockQ4_K_Gpu*)(raw_blocks + (size_t)row * num_blocks_per_row * 144);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
        const BlockQ4_K_Gpu* blk = &row_blocks[b];
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

            uint8_t byte = blk->qs[chunk * 32 + lane_id];
            float q0 = (float)(byte & 0x0F);
            float q1 = (float)(byte >> 4);

            float w0 = d1 * q0 - min1;
            float w1 = d2 * q1 - min2;

            float x0 = x_block[chunk * 64 + lane_id];
            float x1 = x_block[chunk * 64 + 32 + lane_id];

            warp_dot += w0 * x0 + w1 * x1;
        }
    }

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q2_K Matrix-Vector GEMV (2-bit Super-blocks x FP32 Activations).
 * 84-byte super-blocks unpacked across 32 warp lanes.
 */
__global__ void gemv_q2_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 256;

    const BlockQ2_K_Gpu* row_blocks = (const BlockQ2_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 84);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
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

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q3_K Matrix-Vector GEMV (3-bit Super-blocks x FP32 Activations).
 * 110-byte super-blocks unpacked across 32 warp lanes.
 */
__global__ void gemv_q3_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 256;

    const BlockQ3_K_Gpu* row_blocks = (const BlockQ3_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 110);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
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

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
    }
}

/**
 * Custom CUDA Kernel: Quantized Q5_K Matrix-Vector GEMV (5-bit Super-blocks x FP32 Activations).
 * 176-byte super-blocks unpacked across 32 warp lanes.
 */
__global__ void gemv_q5_k_cuda_kernel(
    float* __restrict__ y,
    const void* __restrict__ raw_blocks,
    const float* __restrict__ x,
    int m,
    int k
) {
    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int num_blocks_per_row = k / 256;

    const BlockQ5_K_Gpu* row_blocks = (const BlockQ5_K_Gpu*)((const uint8_t*)raw_blocks + (size_t)row * num_blocks_per_row * 176);

    float warp_dot = 0.0f;

    for (int b = warp_id; b < num_blocks_per_row; b += num_warps) {
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

    // Warp reduction across the 32 lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_dot += __shfl_down_sync(0xFFFFFFFF, warp_dot, offset);
    }

    __shared__ float s_warp_acc[32];
    if (lane_id == 0) {
        s_warp_acc[warp_id] = warp_dot;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_acc = (lane_id < num_warps) ? s_warp_acc[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_acc += __shfl_down_sync(0xFFFFFFFF, block_acc, offset);
        }
        if (lane_id == 0) {
            y[row] = block_acc;
        }
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

__global__ void rope_single_token_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta
) {
    int head_idx = blockIdx.x;
    int half_dim = head_dim / 2;
    int d = threadIdx.x;
    if (d >= half_dim) return;

    float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

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
        k[k_base + d]            = k0 * cos_val - k1 * sin_val;
        k[k_base + d + half_dim] = k0 * sin_val + k1 * cos_val;
    }
}

/**
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling + Online Softmax + GQA)
 */
__global__ void flash_decode_paged_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_kv_tokens,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* q_ptr = q + head_idx * head_dim;
    float* out_ptr = output + head_idx * head_dim;

    int kv_heads = num_kv_heads > 0 ? num_kv_heads : num_heads;
    int kv_head_idx = head_idx / (num_heads / kv_heads);

    float max_score = -1e20f;
    float sum_exp = 0.0f;
    float acc_v = 0.0f;

    __shared__ float s_warp_scores[32];
    __shared__ float s_score;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Grid-stride over past KV cached tokens
    for (int t = 0; t < num_kv_tokens; t++) {
        const float* k_tok = k_cache + (t * kv_heads + kv_head_idx) * head_dim;

        // Compute Q . K dot product for this head across all warps
        float score = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            score += q_ptr[d] * k_tok[d];
        }

        // Warp reduction across the 32 lanes of this warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        if (lane_id == 0) {
            s_warp_scores[warp_id] = score;
        }
        __syncthreads();

        // Reduce across warps
        if (tid == 0) {
            int num_warps = blockDim.x / 32;
            float total_score = 0.0f;
            for (int w = 0; w < num_warps; w++) {
                total_score += s_warp_scores[w];
            }
            s_score = total_score * sm_scale;
        }
        __syncthreads();

        // Online Softmax update
        float cur_score = s_score;
        float new_max = fmaxf(max_score, cur_score);
        float alpha = (t == 0) ? 0.0f : expf(max_score - new_max);
        float beta = expf(cur_score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * alpha + beta;

        const float* v_tok = v_cache + (t * kv_heads + kv_head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            acc_v = acc_v * alpha + beta * v_tok[d];
        }
        __syncthreads();
    }

    // Final normalization
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_ptr[d] = acc_v * inv_sum;
        }
    } else {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_ptr[d] = 0.0f;
        }
    }
}

// C-ABI Launch Wrappers for FFI
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

int launch_cuda_gemv_q4_0(
    float* y,
    const uint8_t* weight_bytes,
    const float* x,
    const half* scales,
    int m,
    int k,
    cudaStream_t stream
) {
    gemv_q4_0_cuda_kernel<<<m, 256, 0, stream>>>(
        y, weight_bytes, x, scales, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q8_0(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    gemv_q8_0_cuda_kernel<<<m, 256, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q6_k(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    gemv_q6_k_cuda_kernel<<<m, 256, 0, stream>>>(
        y, raw_blocks, x, m, k
    );
    return cudaGetLastError();
}

int launch_cuda_gemv_q4_k(
    float* y,
    const void* raw_blocks,
    const float* x,
    int m,
    int k,
    cudaStream_t stream
) {
    gemv_q4_k_cuda_kernel<<<m, 256, 0, stream>>>(
        y, (const uint8_t*)raw_blocks, x, m, k
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
    gemv_ptq1_0_cuda_kernel<<<m, 256, 0, stream>>>(
        y, (const uint8_t*)raw_blocks, x, m, k
    );
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
    gemv_q2_k_cuda_kernel<<<m, 256, 0, stream>>>(
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
    gemv_q3_k_cuda_kernel<<<m, 256, 0, stream>>>(
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
    gemv_q5_k_cuda_kernel<<<m, 256, 0, stream>>>(
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

int launch_cuda_rope_single(
    float* q,
    float* k,
    int pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream
) {
    int grid = num_heads > num_kv_heads ? num_heads : num_kv_heads;
    int block_size = head_dim / 2;
    rope_single_token_kernel<<<grid, block_size, 0, stream>>>(
        q, k, pos, num_heads, num_kv_heads, head_dim, theta
    );
    return cudaGetLastError();
}

int launch_cuda_flash_decode(
    float* output,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int num_kv_tokens,
    float sm_scale,
    cudaStream_t stream
) {
    int block_size = 128;
    if (head_dim < 128) block_size = head_dim;
    flash_decode_paged_kernel<<<num_heads, block_size, 0, stream>>>(
        output, q, k_cache, v_cache, num_heads, num_kv_heads, head_dim, num_kv_tokens, sm_scale
    );
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

}

