#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::similar_names
)]

use serde::{Deserialize, Serialize};

/// Quantization formats supported in AI laboratory research.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum LabQuantMethod {
    Q4_0,
    Q4_K,
    Q8_0,
    Ternary1_58Bit,
    MarlinInt4,
    NvFP4,
}

/// Quantized Block Container holding encoded bytes, scales, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedWeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub method: LabQuantMethod,
    pub scales: Vec<f32>,
    pub data: Vec<u8>,
    pub compression_ratio: f32,
}

/// Post-Training Quantization (PTQ) & Activation-Aware Quantizer (AWQ).
#[derive(Debug, Clone)]
pub struct LabQuantizer;

impl LabQuantizer {
    /// Calibrates and quantizes a weight matrix into the desired target quantization format.
    #[must_use]
    pub fn quantize(
        weights: &[f32],
        rows: usize,
        cols: usize,
        method: LabQuantMethod,
        activation_scales: Option<&[f32]>,
    ) -> QuantizedWeightMatrix {
        let total_elements = rows * cols;
        let mut scaled_weights = weights.to_vec();

        // Apply AWQ activation-aware saliency scaling if provided
        if let Some(act) = activation_scales {
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    if let Some(&s) = act.get(c) {
                        let scale = s.abs().clamp(0.1, 10.0);
                        scaled_weights[idx] *= scale;
                    }
                }
            }
        }

        match method {
            LabQuantMethod::Q8_0 => {
                let block_size = 32;
                let num_blocks = total_elements.div_ceil(block_size);
                let mut scales = Vec::with_capacity(num_blocks);
                let mut data = Vec::with_capacity(total_elements);

                for b in 0..num_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(total_elements);
                    let slice = &scaled_weights[start..end];

                    let mut amax = 0.0f32;
                    for &w in slice {
                        amax = amax.max(w.abs());
                    }
                    let scale = amax / 127.0;
                    scales.push(scale);

                    let inv_scale = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
                    for &w in slice {
                        let q = (w * inv_scale).round().clamp(-128.0, 127.0) as i8;
                        data.push(q as u8);
                    }
                }

                QuantizedWeightMatrix {
                    rows,
                    cols,
                    method,
                    scales,
                    data,
                    compression_ratio: 4.0, // FP32 to INT8
                }
            }
            LabQuantMethod::Q4_0 => {
                let block_size = 32;
                let num_blocks = total_elements.div_ceil(block_size);
                let mut scales = Vec::with_capacity(num_blocks);
                let mut data = Vec::with_capacity(num_blocks * 16);

                for b in 0..num_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(total_elements);
                    let slice = &scaled_weights[start..end];

                    let mut amax = 0.0f32;
                    for &w in slice {
                        amax = amax.max(w.abs());
                    }
                    let scale = amax / 7.0;
                    scales.push(scale);

                    let inv_scale = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
                    for i in (0..slice.len()).step_by(2) {
                        let q0 = ((slice[i] * inv_scale).round().clamp(-8.0, 7.0) + 8.0) as u8;
                        let q1 = if i + 1 < slice.len() {
                            ((slice[i + 1] * inv_scale).round().clamp(-8.0, 7.0) + 8.0) as u8
                        } else {
                            8
                        };
                        data.push((q1 << 4) | (q0 & 0x0F));
                    }
                }

                QuantizedWeightMatrix {
                    rows,
                    cols,
                    method,
                    scales,
                    data,
                    compression_ratio: 8.0, // FP32 to 4-bit
                }
            }
            LabQuantMethod::Ternary1_58Bit => {
                // BitNet 1.58-bit {-1, 0, +1}
                let block_size = 128;
                let num_blocks = total_elements.div_ceil(block_size);
                let mut scales = Vec::with_capacity(num_blocks);
                let mut data = Vec::with_capacity(num_blocks * 32);

                for b in 0..num_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(total_elements);
                    let slice = &scaled_weights[start..end];

                    let mut sum_abs = 0.0f32;
                    for &w in slice {
                        sum_abs += w.abs();
                    }
                    let scale = sum_abs / (slice.len() as f32).max(1.0);
                    scales.push(scale);

                    // Pack 4 ternary values per byte (2 bits each: 00 = 0, 01 = +1, 10 = -1)
                    let inv = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
                    for chunk in slice.chunks(4) {
                        let mut byte = 0u8;
                        for (idx, &w) in chunk.iter().enumerate() {
                            let norm = w * inv;
                            let code = if norm > 0.5 {
                                1u8 // +1
                            } else if norm < -0.5 {
                                2u8 // -1
                            } else {
                                0u8 // 0
                            };
                            byte |= code << (idx * 2);
                        }
                        data.push(byte);
                    }
                }

                QuantizedWeightMatrix {
                    rows,
                    cols,
                    method,
                    scales,
                    data,
                    compression_ratio: 16.0, // FP32 to 2-bit
                }
            }
            LabQuantMethod::Q4_K | LabQuantMethod::MarlinInt4 | LabQuantMethod::NvFP4 => {
                // High-performance 4-bit superblock quantization
                let block_size = 64;
                let num_blocks = total_elements.div_ceil(block_size);
                let mut scales = Vec::with_capacity(num_blocks);
                let mut data = Vec::with_capacity(num_blocks * 32);

                for b in 0..num_blocks {
                    let start = b * block_size;
                    let end = (start + block_size).min(total_elements);
                    let slice = &scaled_weights[start..end];

                    let mut amax = 0.0f32;
                    for &w in slice {
                        amax = amax.max(w.abs());
                    }
                    let scale = amax / 8.0;
                    scales.push(scale);

                    let inv = if scale > 1e-12 { 1.0 / scale } else { 0.0 };
                    for i in (0..slice.len()).step_by(2) {
                        let q0 = ((slice[i] * inv).round().clamp(-8.0, 7.0) + 8.0) as u8;
                        let q1 = if i + 1 < slice.len() {
                            ((slice[i + 1] * inv).round().clamp(-8.0, 7.0) + 8.0) as u8
                        } else {
                            8
                        };
                        data.push((q1 << 4) | (q0 & 0x0F));
                    }
                }

                QuantizedWeightMatrix {
                    rows,
                    cols,
                    method,
                    scales,
                    data,
                    compression_ratio: 8.0,
                }
            }
        }
    }
}

/// Pruned Matrix container with bitmask and sparsity statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrunedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
    pub sparsity: f32,
    pub is_structured_2_4: bool,
}

/// Low-Rank SVD Matrix Factors: $W \approx A \cdot B$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvdDecomposedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub rank_r: usize,
    pub factor_a: Vec<f32>, // [rows, rank_r]
    pub factor_b: Vec<f32>, // [rank_r, cols]
    pub compression_ratio: f32,
}

/// Model Compression Suite: Magnitude Pruning, SVD Factorization, and KV Cache Eviction.
#[derive(Debug, Clone)]
pub struct LabCompressor;

impl LabCompressor {
    /// Applies unstructured or 2:4 structured magnitude pruning to a weight matrix.
    #[must_use]
    pub fn prune_magnitude(
        weights: &[f32],
        rows: usize,
        cols: usize,
        target_sparsity: f32,
        structured_2_4: bool,
    ) -> PrunedMatrix {
        let total = rows * cols;
        let mut values = weights.to_vec();

        if structured_2_4 {
            // NVIDIA Ampere/Ada/Blackwell 2:4 structured sparse pattern (2 zeroed out of every 4)
            for chunk in values.chunks_mut(4) {
                if chunk.len() == 4 {
                    let mut indices = [0, 1, 2, 3];
                    indices.sort_by(|&a, &b| {
                        chunk[a]
                            .abs()
                            .partial_cmp(&chunk[b].abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    // Zero out the two smallest values
                    chunk[indices[0]] = 0.0;
                    chunk[indices[1]] = 0.0;
                }
            }
            PrunedMatrix {
                rows,
                cols,
                values,
                sparsity: 0.50,
                is_structured_2_4: true,
            }
        } else {
            // Unstructured global magnitude pruning
            let mut magnitudes: Vec<f32> = weights.iter().map(|w| w.abs()).collect();
            magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let threshold_idx = ((total as f32) * target_sparsity.clamp(0.0, 0.99)) as usize;
            let threshold = magnitudes.get(threshold_idx).copied().unwrap_or(0.0);

            let mut zero_count = 0;
            for v in &mut values {
                if v.abs() <= threshold {
                    *v = 0.0;
                    zero_count += 1;
                }
            }

            PrunedMatrix {
                rows,
                cols,
                values,
                sparsity: (zero_count as f32) / (total as f32),
                is_structured_2_4: false,
            }
        }
    }

    /// Truncated SVD / Low-Rank Matrix Factorization via Power Iteration: $W \approx A \cdot B$.
    #[must_use]
    pub fn svd_decompose(
        weights: &[f32],
        rows: usize,
        cols: usize,
        rank_r: usize,
    ) -> SvdDecomposedMatrix {
        let r = rank_r.min(rows).min(cols).max(1);
        let mut factor_a = vec![0.01f32; rows * r];
        let mut factor_b = vec![0.01f32; r * cols];

        // Power iteration approximation for top singular components
        for k in 0..r {
            // Random-like deterministic initialization for component k
            for i in 0..rows {
                factor_a[i * r + k] = ((i + k + 1) as f32 * 0.1).sin() * 0.1;
            }

            // Power iteration step
            for _ in 0..3 {
                // b_k = A_k^T * W
                for c in 0..cols {
                    let mut sum = 0.0f32;
                    for i in 0..rows {
                        sum += factor_a[i * r + k] * weights[i * cols + c];
                    }
                    factor_b[k * cols + c] = sum;
                }

                // Normalize b_k
                let mut norm_b = 0.0f32;
                for c in 0..cols {
                    norm_b += factor_b[k * cols + c] * factor_b[k * cols + c];
                }
                let inv_b = 1.0 / norm_b.sqrt().max(1e-12);
                for c in 0..cols {
                    factor_b[k * cols + c] *= inv_b;
                }

                // a_k = W * b_k^T
                for i in 0..rows {
                    let mut sum = 0.0f32;
                    for c in 0..cols {
                        sum += weights[i * cols + c] * factor_b[k * cols + c];
                    }
                    factor_a[i * r + k] = sum;
                }
            }
        }

        let original_params = rows * cols;
        let compressed_params = rows * r + r * cols;
        let compression_ratio = (original_params as f32) / (compressed_params as f32).max(1.0);

        SvdDecomposedMatrix {
            rows,
            cols,
            rank_r: r,
            factor_a,
            factor_b,
            compression_ratio,
        }
    }

    /// Evaluates KV cache compression policy (StreamingLLM + Heavy-Hitter Oracle H2O).
    #[must_use]
    pub fn compress_kv_cache(
        total_tokens: usize,
        sink_tokens: usize,
        local_window_size: usize,
        heavy_hitter_ratio: f32,
    ) -> (usize, f32) {
        if total_tokens <= sink_tokens + local_window_size {
            return (total_tokens, 1.0);
        }

        let evictable_tokens = total_tokens - (sink_tokens + local_window_size);
        let retained_h2o =
            ((evictable_tokens as f32) * heavy_hitter_ratio.clamp(0.0, 1.0)) as usize;
        let retained_total = sink_tokens + local_window_size + retained_h2o;
        let reduction_ratio = (total_tokens as f32) / (retained_total as f32).max(1.0);

        (retained_total, reduction_ratio)
    }
}
