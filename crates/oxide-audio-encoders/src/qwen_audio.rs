//! Qwen-Omni 25 Hz Audio-Tower with TM-RoPE Alignment (`Qwen2.5/3-Omni`).
//!
//! Computes 25 Hz acoustic representations and 3D multi-axis Time-Aligned Multimodal RoPE
//! grids mapping text and continuous audio tokens into a single unified temporal coordinate space.

use serde::{Deserialize, Serialize};

/// Time-Aligned Multimodal RoPE Coordinate Grid (`TM-RoPE`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TmRopeGrid {
    /// Dimension 0: Batch position.
    pub batch_id: u32,
    /// Dimension 1: Modality channel (0 = text, 1 = audio, 2 = vision/spatial).
    pub modality_id: u32,
    /// Dimension 2: Physical wall-clock temporal offset (1 ID = 40 ms at 25 Hz).
    pub temporal_id: u32,
}

/// Qwen-Omni 25 Hz Audio Tower Modality Adapter.
#[derive(Debug, Clone)]
pub struct QwenAudioTower {
    pub hidden_dim: usize,
    pub sample_rate_hz: u32,
    pub frames_per_sec: u32,
}

impl QwenAudioTower {
    #[must_use]
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            sample_rate_hz: 16000,
            frames_per_sec: 25, // 40 ms per frame
        }
    }

    /// Generates TM-RoPE coordinate tuples for mixed text and continuous audio spans.
    #[must_use]
    pub fn generate_tm_rope_grid(
        &self,
        text_prefix_len: usize,
        audio_token_len: usize,
        text_suffix_len: usize,
    ) -> Vec<TmRopeGrid> {
        let total_len = text_prefix_len + audio_token_len + text_suffix_len;
        let mut grid = Vec::with_capacity(total_len);

        // Text prefix: modality 0, time starts at 0
        for i in 0..text_prefix_len {
            grid.push(TmRopeGrid {
                batch_id: 0,
                modality_id: 0,
                temporal_id: i as u32,
            });
        }

        // Audio tokens: modality 1, unified physical timeline (40ms steps)
        let audio_time_start = text_prefix_len as u32;
        for i in 0..audio_token_len {
            grid.push(TmRopeGrid {
                batch_id: 0,
                modality_id: 1,
                temporal_id: audio_time_start + i as u32,
            });
        }

        // Text suffix: modality 0, follows audio timeline
        let suffix_time_start = audio_time_start + audio_token_len as u32;
        for i in 0..text_suffix_len {
            grid.push(TmRopeGrid {
                batch_id: 0,
                modality_id: 0,
                temporal_id: suffix_time_start + i as u32,
            });
        }

        grid
    }
}
