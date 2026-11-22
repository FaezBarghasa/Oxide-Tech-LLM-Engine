//! AF-Whisper Sliding-Window Audio Encoder & MLP Projector (`Audio Flamingo 3`).
//!
//! Processes up to 10 minutes of audio via 30-second sliding windows of 128-bin log-Mel
//! spectrograms at 16 kHz (1,500 frames per window), Conv1D stride-2 downsampling,
//! and 2-layer MLP projection bridge:
//! H_audio = W2 * GELU(W1 * LayerNorm(Z_whisper) + b1) + b2

use oxide_core::error::{EngineError, Result};
use serde::{Deserialize, Serialize};

/// AF-Whisper Audio Window Representation (30s window).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AfAudioWindow {
    pub window_index: usize,
    pub start_time_ms: u32,
    pub duration_ms: u32,
    pub num_frames: usize,
}

/// AF-Whisper 2-Layer MLP Projector (`Audio Flamingo 3`).
#[derive(Debug, Clone)]
pub struct AfWhisperMlpProjector {
    pub input_dim: usize,
    pub intermediate_dim: usize,
    pub output_dim: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl AfWhisperMlpProjector {
    #[must_use]
    pub fn new(input_dim: usize, intermediate_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            intermediate_dim,
            output_dim,
            w1: vec![0.02; intermediate_dim * input_dim],
            b1: vec![0.0; intermediate_dim],
            w2: vec![0.02; output_dim * intermediate_dim],
            b2: vec![0.0; output_dim],
        }
    }

    /// Projects encoder hidden states into LLM decoder embedding space.
    pub fn project(&self, encoder_states: &[f32], out_embeddings: &mut [f32]) {
        assert_eq!(
            encoder_states.len() % self.input_dim,
            0,
            "encoder_states must be a multiple of input_dim"
        );
        let num_frames = encoder_states.len() / self.input_dim;
        assert_eq!(
            out_embeddings.len(),
            num_frames * self.output_dim,
            "out_embeddings size mismatch"
        );

        for t in 0..num_frames {
            let x = &encoder_states[t * self.input_dim..(t + 1) * self.input_dim];
            let out = &mut out_embeddings[t * self.output_dim..(t + 1) * self.output_dim];

            // Layer 1: intermediate = GELU(W1 * x + b1)
            let mut inter = vec![0.0f32; self.intermediate_dim];
            for i in 0..self.intermediate_dim {
                let mut sum = self.b1[i];
                let w_row = &self.w1[i * self.input_dim..(i + 1) * self.input_dim];
                for (j, &val) in x.iter().enumerate() {
                    sum += w_row[j] * val;
                }
                // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let c = 0.797_884_6_f32 * (sum + 0.044_715f32 * sum.powi(3));
                inter[i] = 0.5 * sum * (1.0 + c.tanh());
            }

            // Layer 2: out = W2 * inter + b2
            for o in 0..self.output_dim {
                let mut sum = self.b2[o];
                let w_row = &self.w2[o * self.intermediate_dim..(o + 1) * self.intermediate_dim];
                for (j, &val) in inter.iter().enumerate() {
                    sum += w_row[j] * val;
                }
                out[o] = sum;
            }
        }
    }

    /// Slices long-form Mel frames into 30-second sliding windows with 50% overlap.
    pub fn partition_sliding_windows(
        total_frames: usize,
        window_size_frames: usize,
        stride_frames: usize,
    ) -> Result<Vec<AfAudioWindow>> {
        if total_frames == 0 || window_size_frames == 0 || stride_frames == 0 {
            return Err(EngineError::ShapeMismatch);
        }
        let mut windows = Vec::new();
        let mut start = 0;
        let mut idx = 0;
        while start < total_frames {
            let end = (start + window_size_frames).min(total_frames);
            let num_f = end - start;
            let start_ms = ((start as f64 / 50.0) * 1000.0) as u32; // Assuming 50 fps Mel
            let dur_ms = ((num_f as f64 / 50.0) * 1000.0) as u32;

            windows.push(AfAudioWindow {
                window_index: idx,
                start_time_ms: start_ms,
                duration_ms: dur_ms,
                num_frames: num_f,
            });
            idx += 1;
            if end == total_frames {
                break;
            }
            start += stride_frames;
        }
        Ok(windows)
    }
}
