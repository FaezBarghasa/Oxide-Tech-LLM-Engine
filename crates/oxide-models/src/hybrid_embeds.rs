//! Continuous multimodal audio-language embedding sequences.
//!
//! Provides the `InputSpan` abstraction enabling hybrid discrete-continuous sequences
//! (`inputs_embeds`) required for Audio Flamingo 3, MOSS-Music-8B-Thinking, and Qwen-Omni.

use serde::{Deserialize, Serialize};

/// A segment of an autoregressive model input sequence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputSpan {
    /// Standard discrete tokenizer token IDs.
    TextTokens(Vec<u32>),
    /// Continuous multimodal audio feature embeddings.
    ContinuousAudioFeatures {
        /// Blake3 cryptographic hash of the raw audio waveform for KV-prefix caching.
        blake3_audio_hash: [u8; 32],
        /// Flattened continuous embedding features `[num_audio_tokens, hidden_dim]`.
        features: Vec<f32>,
        /// Number of continuous audio tokens.
        num_tokens: usize,
        /// Audio hidden dimension.
        hidden_dim: usize,
        /// Optional temporal timestamp grid in milliseconds for TM-RoPE alignment.
        time_grid_ms: Option<Vec<u32>>,
    },
}

impl InputSpan {
    /// Creates a text token span.
    #[must_use]
    pub fn text(tokens: impl Into<Vec<u32>>) -> Self {
        Self::TextTokens(tokens.into())
    }

    /// Creates a continuous audio feature span.
    #[must_use]
    pub fn audio(
        blake3_audio_hash: [u8; 32],
        features: Vec<f32>,
        num_tokens: usize,
        hidden_dim: usize,
        time_grid_ms: Option<Vec<u32>>,
    ) -> Self {
        assert_eq!(
            features.len(),
            num_tokens * hidden_dim,
            "Audio feature length must equal num_tokens * hidden_dim"
        );
        Self::ContinuousAudioFeatures {
            blake3_audio_hash,
            features,
            num_tokens,
            hidden_dim,
            time_grid_ms,
        }
    }

    /// Returns the effective token count of this span.
    #[must_use]
    pub fn token_count(&self) -> usize {
        match self {
            Self::TextTokens(tokens) => tokens.len(),
            Self::ContinuousAudioFeatures { num_tokens, .. } => *num_tokens,
        }
    }
}

/// Spliced Hybrid Embedding Sequence for multimodal decoder forward prefill.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HybridEmbeddingSequence {
    pub spans: Vec<InputSpan>,
    pub total_token_count: usize,
}

impl HybridEmbeddingSequence {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_span(&mut self, span: InputSpan) {
        self.total_token_count += span.token_count();
        self.spans.push(span);
    }

    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_token_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_embedding_sequence_composition() {
        let mut seq = HybridEmbeddingSequence::new();
        seq.push_span(InputSpan::text(vec![1, 2, 3]));
        assert_eq!(seq.total_tokens(), 3);

        let audio_features = vec![0.5f32; 10 * 64];
        let audio_span = InputSpan::audio([7u8; 32], audio_features, 10, 64, None);
        seq.push_span(audio_span);
        assert_eq!(seq.total_tokens(), 13);
    }
}
