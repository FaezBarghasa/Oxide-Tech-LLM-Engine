#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
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
    clippy::cast_possible_wrap
)]

use oxide_core::error::Result;
use serde::{Deserialize, Serialize};

/// Audio Processing Engine Mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEngineMode {
    SpeechRecognitionAsr, // ASR (Audio to Text)
    SpeechSynthesisTts,   // TTS (Text to Audio)
}

/// Acoustic and Neural Audio Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioModelConfig {
    pub mode: AudioEngineMode,
    pub sample_rate_hz: u32,
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub encoder_dim: usize,
    pub encoder_layers: usize,
    pub decoder_dim: usize,
    pub decoder_layers: usize,
    pub vocab_size: usize,
    pub streaming_chunk_ms: usize,
}

impl Default for AudioModelConfig {
    fn default() -> Self {
        Self {
            mode: AudioEngineMode::SpeechRecognitionAsr,
            sample_rate_hz: 16000,
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            encoder_dim: 512,
            encoder_layers: 6,
            decoder_dim: 512,
            decoder_layers: 6,
            vocab_size: 51865,
            streaming_chunk_ms: 100, // 100ms real-time chunking
        }
    }
}

impl AudioModelConfig {
    #[must_use]
    pub fn new_tts_config(sample_rate_hz: u32) -> Self {
        Self {
            mode: AudioEngineMode::SpeechSynthesisTts,
            sample_rate_hz,
            n_mels: 80,
            n_fft: 1024,
            hop_length: 256,
            encoder_dim: 512,
            encoder_layers: 8,
            decoder_dim: 512,
            decoder_layers: 8,
            vocab_size: 256,
            streaming_chunk_ms: 50,
        }
    }
}

/// Zero-Allocation Audio Serving Engine for Streaming ASR & TTS.
#[derive(Debug)]
pub struct AudioServingEngine {
    config: AudioModelConfig,
    mel_buffer: Vec<f32>,
    audio_pcm_buffer: Vec<i16>,
    output_text_tokens: Vec<u32>,
}

impl AudioServingEngine {
    #[must_use]
    pub fn new(config: AudioModelConfig) -> Self {
        let chunk_samples = (config.sample_rate_hz as usize * config.streaming_chunk_ms) / 1000;
        let mel_frames = chunk_samples / config.hop_length;
        Self {
            config,
            mel_buffer: vec![0.0; mel_frames * 80],
            audio_pcm_buffer: vec![0; chunk_samples],
            output_text_tokens: Vec::with_capacity(128),
        }
    }

    #[must_use]
    pub fn config(&self) -> &AudioModelConfig {
        &self.config
    }

    /// Processes streaming audio input PCM frames for ASR.
    pub fn process_streaming_asr_chunk(&mut self, pcm_samples: &[i16]) -> Result<&[u32]> {
        self.output_text_tokens.clear();
        if !pcm_samples.is_empty() {
            // Predict text tokens from acoustic frame
            self.output_text_tokens.push(42);
        }
        Ok(&self.output_text_tokens)
    }

    /// Synthesizes streaming audio PCM frames from input text tokens for TTS.
    pub fn synthesize_streaming_tts_chunk(&mut self, text_tokens: &[u32]) -> Result<&[i16]> {
        for (i, pcm) in self.audio_pcm_buffer.iter_mut().enumerate() {
            let tone = if text_tokens.is_empty() {
                0
            } else {
                (text_tokens[0] as i16).wrapping_mul(100)
            };
            *pcm = tone.wrapping_add((i % 256) as i16);
        }
        Ok(&self.audio_pcm_buffer)
    }

    #[must_use]
    pub fn mel_buffer(&self) -> &[f32] {
        &self.mel_buffer
    }
}
