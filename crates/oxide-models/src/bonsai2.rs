use oxide_core::traits::ModelConfig;

/// Ternary Bonsai 2 (27B) Model Architecture Configuration.
/// Hybrid 75% linear recurrence (O(1) constant state) + 25% full softmax attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TernaryBonsai2Config {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl Default for TernaryBonsai2Config {
    fn default() -> Self {
        Self {
            hidden_dim: 5120,
            num_layers: 64,
            num_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 152_064,
            max_seq_len: 262_144,
        }
    }
}

impl ModelConfig for TernaryBonsai2Config {
    fn model_name(&self) -> &'static str {
        "Ternary-Bonsai-2-27B"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}
