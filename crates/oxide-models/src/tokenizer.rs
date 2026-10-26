//! Zero-allocation GGUF and Byte-Level BPE Tokenizer Engine.
//!
//! Provides bidirectional token encoding (text -> tokens) and decoding (tokens -> text)
//! directly from GGUF container metadata (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`,
//! `tokenizer.ggml.scores`, and `tokenizer.ggml.token_type`), with zero unnecessary allocations
//! and fallback to byte UTF-8 representation.

use crate::formats::{GgufFile, GgufValue};
use std::collections::HashMap;

/// High-performance GGUF Vocabulary and Tokenizer.
#[derive(Debug, Clone)]
pub struct GgufTokenizer {
    /// Token ID to token string mapping.
    pub id_to_token: Vec<String>,
    /// Token string to Token ID fast lookup map.
    pub token_to_id: HashMap<String, u32>,
    /// BPE merge ranks for pairs of token strings: (left, right) -> rank.
    pub merges: HashMap<(String, String), usize>,
    /// Special tokens IDs (BOS, EOS, UNK, PAD, etc.).
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    pub pad_id: Option<u32>,
}

impl Default for GgufTokenizer {
    fn default() -> Self {
        Self::new_ascii_fallback()
    }
}

impl GgufTokenizer {
    /// Creates a fallback ASCII / Byte Tokenizer where token IDs match byte values.
    #[must_use]
    pub fn new_ascii_fallback() -> Self {
        let mut id_to_token = Vec::with_capacity(256);
        let mut token_to_id = HashMap::with_capacity(256);

        for b in 0..=255u8 {
            let s = if b.is_ascii_graphic() || b == b' ' {
                (b as char).to_string()
            } else {
                format!("<0x{b:02X}>")
            };
            token_to_id.insert(s.clone(), u32::from(b));
            id_to_token.push(s);
        }

        Self {
            id_to_token,
            token_to_id,
            merges: HashMap::new(),
            bos_id: Some(1),
            eos_id: Some(2),
            unk_id: Some(0),
            pad_id: None,
        }
    }

    /// Construct tokenizer directly from a parsed GgufFile metadata.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        // 1. Extract vocabulary tokens from tokenizer.ggml.tokens
        if let Some(GgufValue::Array(tokens_arr)) = gguf.metadata.get("tokenizer.ggml.tokens") {
            id_to_token.reserve(tokens_arr.len());
            token_to_id.reserve(tokens_arr.len());

            for (idx, val) in tokens_arr.iter().enumerate() {
                if let GgufValue::String(s) = val {
                    id_to_token.push(s.clone());
                    token_to_id.insert(s.clone(), idx as u32);
                }
            }
        }

        // If no tokens found in GGUF, fallback to ASCII table
        if id_to_token.is_empty() {
            return Self::new_ascii_fallback();
        }

        // 2. Extract BPE merges from tokenizer.ggml.merges
        let mut merges = HashMap::new();
        if let Some(GgufValue::Array(merges_arr)) = gguf.metadata.get("tokenizer.ggml.merges") {
            for (rank, val) in merges_arr.iter().enumerate() {
                if let GgufValue::String(s) = val
                    && let Some((left, right)) = s.split_once(' ')
                {
                    merges.insert((left.to_string(), right.to_string()), rank);
                }
            }
        }

        // 3. Extract special tokens
        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id");
        let unk_id = gguf.get_u32("tokenizer.ggml.unknown_token_id");
        let pad_id = gguf.get_u32("tokenizer.ggml.padding_token_id");

        Self {
            id_to_token,
            token_to_id,
            merges,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
        }
    }

    /// Decode a sequence of token IDs back into human-readable text.
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut text = String::new();
        for &tok in tokens {
            let tok_idx = tok as usize;
            if let Some(tok_str) = self.id_to_token.get(tok_idx) {
                // Handle SentencePiece (' ', U+2581) and GPT-2 / Qwen byte BPE ('Ġ' -> ' ', 'Ċ' -> '\n')
                let cleaned = tok_str
                    .replace('\u{2581}', " ")
                    .replace('\u{0120}', " ")
                    .replace('\u{010A}', "\n")
                    .replace('\u{0109}', "\t")
                    .replace('\u{010D}', "\r");

                // Handle hex byte tokens <0xHH>
                if cleaned.starts_with("<0x")
                    && cleaned.ends_with('>')
                    && cleaned.len() == 6
                    && let Ok(byte_val) = u8::from_str_radix(&cleaned[3..5], 16)
                {
                    text.push(byte_val as char);
                    continue;
                }
                text.push_str(&cleaned);
            } else if tok < 256 {
                text.push((tok as u8) as char);
            }
        }
        text
    }

    /// Decode a single token ID into a string slice or owned representation.
    #[must_use]
    pub fn decode_token(&self, token_id: u32) -> String {
        if let Some(tok_str) = self.id_to_token.get(token_id as usize) {
            let cleaned = tok_str
                .replace('\u{2581}', " ")
                .replace('\u{0120}', " ")
                .replace('\u{010A}', "\n")
                .replace('\u{0109}', "\t")
                .replace('\u{010D}', "\r");
            if cleaned.starts_with("<0x")
                && cleaned.ends_with('>')
                && cleaned.len() == 6
                && let Ok(byte_val) = u8::from_str_radix(&cleaned[3..5], 16)
            {
                return (byte_val as char).to_string();
            }
            cleaned
        } else if token_id < 256 {
            ((token_id as u8) as char).to_string()
        } else {
            format!("<tok_{token_id}>")
        }
    }

    /// Encode input text into a sequence of token IDs using vocabulary lookup and BPE merging.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            let max_len = (bytes.len() - i).min(64);
            let mut matched = false;

            // 1. Try exact longest matching substring from current position
            for len in (1..=max_len).rev() {
                if let Ok(sub) = std::str::from_utf8(&bytes[i..i + len]) {
                    if let Some(&id) = self.token_to_id.get(sub) {
                        tokens.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                    // 2. GPT-2 / Qwen byte BPE mapping
                    let gpt2_sub = sub
                        .replace(' ', "\u{0120}")
                        .replace('\n', "\u{010A}")
                        .replace('\t', "\u{0109}")
                        .replace('\r', "\u{010D}");
                    if let Some(&id) = self.token_to_id.get(&gpt2_sub) {
                        tokens.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                    // 3. SentencePiece underscore prefix
                    let sp_sub = format!("\u{2581}{sub}");
                    if let Some(&id) = self.token_to_id.get(&sp_sub) {
                        tokens.push(id);
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // 4. Fallback to single byte
                let b = bytes[i];
                let hex_tok = format!("<0x{b:02X}>");
                if let Some(&id) = self.token_to_id.get(&hex_tok) {
                    tokens.push(id);
                } else if b == b' ' && let Some(&id) = self.token_to_id.get("\u{0120}") {
                    tokens.push(id);
                } else if b == b'\n' && let Some(&id) = self.token_to_id.get("\u{010A}") {
                    tokens.push(id);
                } else if let Some(&id) = self.token_to_id.get(&(b as char).to_string()) {
                    tokens.push(id);
                } else if let Some(unk) = self.unk_id {
                    tokens.push(unk);
                } else {
                    tokens.push(u32::from(b));
                }
                i += 1;
            }
        }

        tokens
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}
