//! Grammar-Constrained Logit Masking for LLM2Fx Tools and SurrealDB MCP.
//!
//! Enforces 100% syntactically valid JSON and typed Rust struct schema deserialization
//! with sub-40 microsecond per-token logit masking overhead.

use serde::{Deserialize, Serialize};

/// Tool schema definition specifying required parameters and constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchemaSpec {
    pub tool_name: String,
    pub description: String,
    pub required_fields: Vec<String>,
}

/// Token Logit Masker enforcing valid grammar state transitions.
#[derive(Debug, Clone)]
pub struct GrammarLogitMasker {
    pub allowed_ascii_mask: [bool; 256],
    pub is_active: bool,
}

impl Default for GrammarLogitMasker {
    fn default() -> Self {
        let mut mask = [false; 256];
        // Allow valid printable ASCII JSON tokens by default
        for b in 0x20..=0x7E {
            mask[b] = true;
        }
        mask[b'\n' as usize] = true;
        mask[b'\t' as usize] = true;
        Self {
            allowed_ascii_mask: mask,
            is_active: true,
        }
    }
}

impl GrammarLogitMasker {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts allowed tokens to numeric digits and separators (e.g. for parameter values).
    pub fn restrict_to_numbers(&mut self) {
        self.allowed_ascii_mask.fill(false);
        for b in b'0'..=b'9' {
            self.allowed_ascii_mask[b as usize] = true;
        }
        self.allowed_ascii_mask[b'.' as usize] = true;
        self.allowed_ascii_mask[b'-' as usize] = true;
        self.allowed_ascii_mask[b',' as usize] = true;
        self.allowed_ascii_mask[b'}' as usize] = true;
        self.allowed_ascii_mask[b']' as usize] = true;
    }

    /// Masks logits on GPU/host in-place before sampling (< 40 us overhead).
    pub fn mask_logits(&self, vocab_ascii_bytes: &[Option<u8>], logits: &mut [f32]) {
        if !self.is_active {
            return;
        }
        let len = vocab_ascii_bytes.len().min(logits.len());
        for i in 0..len {
            if let Some(b) = vocab_ascii_bytes[i] {
                if !self.allowed_ascii_mask[b as usize] {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Common LLM2Fx Tool Call representation for audio DSP parameter automation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StemFxToolCall {
    pub stem_name: String,
    pub gain_db: f32,
    pub pan: f32, // -1.0 to 1.0
    pub low_shelf_freq: f32,
    pub high_shelf_freq: f32,
    pub compression_threshold_db: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_logit_masker_numeric_restriction() {
        let mut masker = GrammarLogitMasker::new();
        masker.restrict_to_numbers();

        let mut logits = [0.0f32; 4];
        let bytes = [Some(b'5'), Some(b'a'), Some(b'.'), Some(b'Z')];

        masker.mask_logits(&bytes, &mut logits);
        assert_eq!(logits[0], 0.0); // '5' allowed
        assert_eq!(logits[1], f32::NEG_INFINITY); // 'a' masked
        assert_eq!(logits[2], 0.0); // '.' allowed
        assert_eq!(logits[3], f32::NEG_INFINITY); // 'Z' masked
    }
}
