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
    float eps
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
    // Apply scale and weights
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
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
 * Custom CUDA Kernel: Quantized Q8_0 Matrix-Vector GEMV (INT8 Weights x FP32 Activations).
 */
__global__ void gemv_q8_0_cuda_kernel(
    float* __restrict__ y,
    const int8_t* __restrict__ weight_bytes,
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
        int weight_offset = (row * num_blocks_per_row + b) * 32;
        float d = __half2float(scales[row * num_blocks_per_row + b]);

        float block_sum = 0.0f;
        #pragma unroll 32
        for (int i = 0; i < 32; i++) {
            block_sum += (float)weight_bytes[weight_offset + i] * x[b * 32 + i];
        }
        acc += block_sum * d;
    }

    // Warp reduction
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
 * Custom CUDA Kernel: FlashAttention-2 Paged Token Decode Kernel (Tiling + Online Softmax)
 */
__global__ void flash_decode_paged_kernel(
    float* __restrict__ output,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    int num_heads,
    int head_dim,
    int num_kv_tokens,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* q_ptr = q + head_idx * head_dim;
    float* out_ptr = output + head_idx * head_dim;

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    // Grid-stride over past KV cached tokens
    for (int t = 0; t < num_kv_tokens; t++) {
        const float* k_tok = k_cache + (t * num_heads + head_idx) * head_dim;

        // Compute Q . K dot product for this head
        float score = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            score += q_ptr[d] * k_tok[d];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            score += __shfl_down_sync(0xFFFFFFFF, score, offset);
        }

        __shared__ float s_score;
        if (tid == 0) {
            s_score = score * sm_scale;
        }
        __syncthreads();

        // Online Softmax update
        float cur_score = s_score;
        float new_max = fmaxf(max_score, cur_score);
        float alpha = expf(max_score - new_max);
        float beta = expf(cur_score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * alpha + beta;

        const float* v_tok = v_cache + (t * num_heads + head_idx) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_ptr[d] = out_ptr[d] * alpha + beta * v_tok[d];
        }
        __syncthreads();
    }

    // Final normalization
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_ptr[d] *= inv_sum;
        }
    }
}

}
