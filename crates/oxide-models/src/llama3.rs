use oxide_core::traits::ModelConfig;

/// Dense Transformer baseline LLaMA 3.1 8B configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Llama3Config {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl Default for Llama3Config {
    fn default() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 128_256,
            max_seq_len: 131_072,
        }
    }
}

impl ModelConfig for Llama3Config {
    fn model_name(&self) -> &'static str {
        "Meta-Llama-3.1-8B"
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
