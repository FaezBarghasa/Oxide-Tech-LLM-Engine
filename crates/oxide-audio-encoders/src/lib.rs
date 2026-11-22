//! # `oxide-audio-encoders`
//!
//! Native Rust/CUDA SOTA Audio-LALM Encoders:
//! - AF-Whisper + 2-Layer MLP Projector (`Audio Flamingo 3`)
//! - MOSS-Audio-Encoder with 12.5 Hz Strided Continuous Latents (`MOSS-Music-8B-Thinking`)
//! - Qwen-Audio-Tower with 25 Hz TM-RoPE Alignment (`Qwen2.5/3-Omni`)

pub mod af_whisper;
pub mod moss_audio;
pub mod qwen_audio;

pub use af_whisper::{AfAudioWindow, AfWhisperMlpProjector};
pub use moss_audio::{MossAudioEncoder, MossAudioTokenMeta};
pub use qwen_audio::{QwenAudioTower, TmRopeGrid};

use serde::{Deserialize, Serialize};

/// Supported Multimodal Audio Encoder architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioEncoderKind {
    /// NVIDIA Audio Flamingo 3 sliding-window AF-Whisper + MLP.
    AfWhisperMlp,
    /// OpenMOSS MOSS-Music-8B continuous 12.5 Hz acoustic encoder.
    MossMusicAcoustic,
    /// Qwen-Omni 25 Hz audio-tower with TM-RoPE temporal alignment.
    QwenOmniTower,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_af_whisper_projection_shapes() {
        let proj = AfWhisperMlpProjector::new(16, 32, 64);
        let in_states = vec![1.0f32; 16 * 4];
        let mut out = vec![0.0f32; 64 * 4];
        proj.project(&in_states, &mut out);
        assert_eq!(out.len(), 256);
        assert!(out.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_moss_audio_12_5_hz_rate() {
        let encoder = MossAudioEncoder::new(16000, 128);
        let one_sec_pcm = vec![0.1f32; 16000];
        let mut latents = Vec::new();
        let tokens = encoder.encode_pcm_to_latents(&one_sec_pcm, &mut latents);
        // 16,000 / 1333 ~= 12 tokens
        assert!(tokens >= 11 && tokens <= 13);
        assert_eq!(latents.len(), tokens * 128);
    }

    #[test]
    fn test_tm_rope_grid_alignment() {
        let tower = QwenAudioTower::new(128);
        let grid = tower.generate_tm_rope_grid(10, 25, 5);
        assert_eq!(grid.len(), 40);
        assert_eq!(grid[0].modality_id, 0);
        assert_eq!(grid[10].modality_id, 1);
        assert_eq!(grid[35].modality_id, 0);
    }
}
