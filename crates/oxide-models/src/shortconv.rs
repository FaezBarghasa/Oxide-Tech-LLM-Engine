//! Native Depthwise 1D Short-Convolution Engine for Liquid Foundation Models (LFM3) and hybrid architectures.
//!
//! Replaces standard quadratic self-attention in hybrid recurrent layers with a width-7 causal
//! depthwise 1D convolution and gated projections ($B \odot x$, $C \odot \text{conv1d}(u)$).

use crate::llama3::QuantizedTensor;

/// Internal state for causal depthwise 1D short convolution.
/// Retains the last `kernel_size - 1` historical activation values per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortConv1DState {
    pub kernel_size: usize,
    pub channels: usize,
    pub conv_state: Vec<f32>,
}

impl ShortConv1DState {
    /// Create an uninitialized zeroed state for a given channel count and kernel width (e.g. K=7).
    #[must_use]
    pub fn new(channels: usize, kernel_size: usize) -> Self {
        let history_len = kernel_size.saturating_sub(1);
        Self {
            kernel_size,
            channels,
            conv_state: vec![0.0f32; channels * history_len],
        }
    }

    /// Reset historical state to zeros.
    pub fn reset(&mut self) {
        self.conv_state.fill(0.0);
    }

    /// Single-step causal depthwise 1D convolution for decode.
    ///
    /// # Arguments
    /// - `bx`: input activation for the current token of size `channels`.
    /// - `conv_weights`: depthwise kernel weights of size `channels * kernel_size`.
    /// - `conv_bias`: optional per-channel bias of size `channels`.
    /// - `out`: buffer receiving the convolution output of size `channels`.
    pub fn step_causal(
        &mut self,
        bx: &[f32],
        conv_weights: &[f32],
        conv_bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let k = self.kernel_size;
        let h_len = k.saturating_sub(1);
        assert!(bx.len() >= self.channels);
        assert!(conv_weights.len() >= self.channels * k);
        assert!(out.len() >= self.channels);

        for c in 0..self.channels {
            let state_offset = c * h_len;
            let weight_offset = c * k;

            // Convolution over history: state[0..h_len] with conv_weights[0..h_len]
            let mut sum = 0.0f32;
            for i in 0..h_len {
                sum += self.conv_state[state_offset + i] * conv_weights[weight_offset + i];
            }
            // Contribution of current token with conv_weights[h_len]
            sum += bx[c] * conv_weights[weight_offset + h_len];

            if let Some(bias) = conv_bias {
                sum += bias[c];
            }
            out[c] = sum;

            // Shift state left by 1 and append bx[c]
            for i in 0..h_len.saturating_sub(1) {
                self.conv_state[state_offset + i] = self.conv_state[state_offset + i + 1];
            }
            if h_len > 0 {
                self.conv_state[state_offset + h_len - 1] = bx[c];
            }
        }
    }

    /// Multi-token causal depthwise 1D convolution for prefill chunk processing.
    ///
    /// # Arguments
    /// - `bx`: input activations of shape `[seq_len, channels]`.
    /// - `seq_len`: number of tokens in current prompt chunk.
    /// - `conv_weights`: depthwise kernel weights of size `channels * kernel_size`.
    /// - `conv_bias`: optional per-channel bias of size `channels`.
    /// - `out`: buffer receiving output of shape `[seq_len, channels]`.
    pub fn forward_prefill(
        &mut self,
        bx: &[f32],
        seq_len: usize,
        conv_weights: &[f32],
        conv_bias: Option<&[f32]>,
        out: &mut [f32],
    ) {
        let k = self.kernel_size;
        let h_len = k.saturating_sub(1);
        let c_dim = self.channels;

        for c in 0..c_dim {
            let state_offset = c * h_len;
            let weight_offset = c * k;

            for t in 0..seq_len {
                let mut sum = 0.0f32;
                for i in 0..k {
                    let lag = (k - 1) - i;
                    let val = if t >= lag {
                        bx[(t - lag) * c_dim + c]
                    } else {
                        // Historical input prior to prompt chunk boundary
                        let hist_idx = h_len.saturating_sub(lag - t);
                        if hist_idx < h_len {
                            self.conv_state[state_offset + hist_idx]
                        } else {
                            0.0f32
                        }
                    };
                    sum += val * conv_weights[weight_offset + i];
                }

                if let Some(bias) = conv_bias {
                    sum += bias[c];
                }
                out[t * c_dim + c] = sum;
            }

            // Update persistent state with trailing h_len tokens
            for i in 0..h_len {
                let lag = h_len - 1 - i;
                if seq_len > lag {
                    self.conv_state[state_offset + i] = bx[(seq_len - 1 - lag) * c_dim + c];
                }
            }
        }
    }
}

/// Hybrid Layer Short-Convolution Module for LFM3.
#[derive(Debug, Clone)]
pub struct Lfm3ShortConvLayer {
    /// Projection from `hidden_dim` to `3 * hidden_dim` (producing B, C, x).
    pub in_proj: QuantizedTensor,
    /// Depthwise 1D convolution weights of shape `[channels * kernel_size]`.
    pub conv_weights: Vec<f32>,
    /// Optional depthwise 1D convolution bias of shape `[channels]`.
    pub conv_bias: Option<Vec<f32>>,
    /// Dense linear projection from `hidden_dim` back to `hidden_dim`.
    pub out_proj: QuantizedTensor,
    /// Kernel size (typically 7).
    pub kernel_size: usize,
    /// Number of feature channels (typically hidden_dim = 2048).
    pub channels: usize,
}

impl Lfm3ShortConvLayer {
    /// Evaluates a single-token decode pass through the short-convolution block.
    pub fn forward_decode(
        &self,
        norm_in: &[f32],
        state: &mut ShortConv1DState,
        bcx: &mut [f32],
        conv_in: &mut [f32],
        conv_out: &mut [f32],
        out: &mut [f32],
    ) {
        let c = self.channels;
        assert!(norm_in.len() >= c);
        assert!(bcx.len() >= 3 * c);
        assert!(conv_in.len() >= c);
        assert!(conv_out.len() >= c);
        assert!(out.len() >= c);

        // 1. in_proj: maps norm_in [c] -> bcx [3 * c]
        self.in_proj.gemv(norm_in, &mut bcx[..3 * c]);

        let (b_vec, cx) = bcx[..3 * c].split_at(c);
        let (c_vec, x_vec) = cx.split_at(c);

        // 2. Input Gating: u = B * x
        for i in 0..c {
            conv_in[i] = b_vec[i] * x_vec[i];
        }

        // 3. Causal depthwise 1D conv
        state.step_causal(
            &conv_in[..c],
            &self.conv_weights,
            self.conv_bias.as_deref(),
            &mut conv_out[..c],
        );

        // 4. Output Gating: y = C * conv_out
        for i in 0..c {
            conv_out[i] *= c_vec[i];
        }

        // 5. out_proj: maps conv_out [c] -> out [c]
        self.out_proj.gemv(&conv_out[..c], &mut out[..c]);
    }

    /// Evaluates a multi-token prefill pass through the short-convolution block.
    pub fn forward_prefill(
        &self,
        norm_in: &[f32], // [seq_len, channels]
        seq_len: usize,
        state: &mut ShortConv1DState,
        batch_bcx: &mut [f32], // [seq_len, 3 * channels]
        batch_u: &mut [f32],   // [seq_len, channels]
        batch_y: &mut [f32],   // [seq_len, channels]
        out: &mut [f32],       // [seq_len, channels]
    ) {
        let c = self.channels;
        assert!(norm_in.len() >= seq_len * c);
        assert!(batch_bcx.len() >= seq_len * 3 * c);
        assert!(batch_u.len() >= seq_len * c);
        assert!(batch_y.len() >= seq_len * c);
        assert!(out.len() >= seq_len * c);

        // 1. in_proj per token
        for t in 0..seq_len {
            let in_t = &norm_in[t * c..(t + 1) * c];
            let bcx_t = &mut batch_bcx[t * 3 * c..(t + 1) * 3 * c];
            self.in_proj.gemv(in_t, bcx_t);

            let (b_vec, cx) = bcx_t.split_at(c);
            let (_c_vec, x_vec) = cx.split_at(c);

            // 2. Input Gating: u = B * x
            let u_t = &mut batch_u[t * c..(t + 1) * c];
            for i in 0..c {
                u_t[i] = b_vec[i] * x_vec[i];
            }
        }

        // 3. Multi-token causal depthwise 1D conv
        let mut conv_out = vec![0.0f32; seq_len * c];
        state.forward_prefill(
            &batch_u[..seq_len * c],
            seq_len,
            &self.conv_weights,
            self.conv_bias.as_deref(),
            &mut conv_out,
        );

        // 4. Output Gating & 5. out_proj per token
        for t in 0..seq_len {
            let bcx_t = &batch_bcx[t * 3 * c..(t + 1) * 3 * c];
            let c_vec = &bcx_t[c..2 * c];
            let y_t = &mut batch_y[t * c..(t + 1) * c];
            let cout_t = &conv_out[t * c..(t + 1) * c];

            for i in 0..c {
                y_t[i] = c_vec[i] * cout_t[i];
            }

            self.out_proj.gemv(y_t, &mut out[t * c..(t + 1) * c]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortconv_impulse_response() {
        let channels = 2;
        let k = 7;
        let mut state = ShortConv1DState::new(channels, k);

        // Weights: Channel 0 has weights 1.0..7.0; Channel 1 has weights 0.1..0.7
        let conv_weights: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // Channel 0
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // Channel 1
        ];

        let mut out = [0.0f32; 2];

        // Step 0: Input [1.0, 2.0]
        state.step_causal(&[1.0, 2.0], &conv_weights, None, &mut out);
        // Current token multiplies with conv_weights[6]
        assert!((out[0] - 7.0).abs() < 1e-5, "step 0 ch0: {}", out[0]);
        assert!((out[1] - 1.4).abs() < 1e-5, "step 0 ch1: {}", out[1]);

        // Step 1: Input [0.0, 0.0] -> previous input shifted to lag 1 (weight 6.0 / 0.6)
        state.step_causal(&[0.0, 0.0], &conv_weights, None, &mut out);
        assert!((out[0] - 6.0).abs() < 1e-5, "step 1 ch0: {}", out[0]);
        assert!((out[1] - 1.2).abs() < 1e-5, "step 1 ch1: {}", out[1]);

        // Step 2: Input [0.0, 0.0] -> lag 2 (weight 5.0 / 0.5)
        state.step_causal(&[0.0, 0.0], &conv_weights, None, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-5, "step 2 ch0: {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "step 2 ch1: {}", out[1]);
    }

    #[test]
    fn test_shortconv_prefill_and_decode_parity() {
        let channels = 4;
        let k = 7;
        let seq_len = 5;

        let mut prefill_state = ShortConv1DState::new(channels, k);
        let mut decode_state = ShortConv1DState::new(channels, k);

        let conv_weights: Vec<f32> = (0..channels * k).map(|i| (i as f32) * 0.1 + 0.5).collect();

        // Sequence of 5 tokens
        let mut prompt = vec![0.0f32; seq_len * channels];
        for t in 0..seq_len {
            for c in 0..channels {
                prompt[t * channels + c] = ((t + 1) * (c + 1)) as f32;
            }
        }

        // Prefill pass
        let mut prefill_out = vec![0.0f32; seq_len * channels];
        prefill_state.forward_prefill(
            &prompt,
            seq_len,
            &conv_weights,
            None,
            &mut prefill_out,
        );

        // Step-by-step decode pass over identical tokens
        let mut decode_out = vec![0.0f32; seq_len * channels];
        for t in 0..seq_len {
            let tok_in = &prompt[t * channels..(t + 1) * channels];
            let tok_out = &mut decode_out[t * channels..(t + 1) * channels];
            decode_state.step_causal(tok_in, &conv_weights, None, tok_out);
        }

        // Verify prefill outputs match sequential decode steps exactly
        for i in 0..seq_len * channels {
            assert!(
                (prefill_out[i] - decode_out[i]).abs() < 1e-5,
                "mismatch at {i}: prefill {} != decode {}",
                prefill_out[i],
                decode_out[i]
            );
        }

        // Verify resulting state parity
        for i in 0..prefill_state.conv_state.len() {
            assert!(
                (prefill_state.conv_state[i] - decode_state.conv_state[i]).abs() < 1e-5,
                "state mismatch at {i}: prefill {} != decode {}",
                prefill_state.conv_state[i],
                decode_state.conv_state[i]
            );
        }
    }
}
