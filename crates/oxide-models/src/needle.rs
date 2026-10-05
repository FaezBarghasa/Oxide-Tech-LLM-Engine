use oxide_core::traits::ModelConfig;

/// Cactus Needle 3 subnetwork configuration with const-generic compile-time depth laddering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CactusNeedleConfig<const ACTIVE_LAYERS: usize> {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl<const ACTIVE_LAYERS: usize> Default for CactusNeedleConfig<ACTIVE_LAYERS> {
    fn default() -> Self {
        Self {
            hidden_dim: 2048,
            num_heads: 16,
            head_dim: 128,
            vocab_size: 32768,
            max_seq_len: 8192,
        }
    }
}

impl<const ACTIVE_LAYERS: usize> ModelConfig for CactusNeedleConfig<ACTIVE_LAYERS> {
    fn model_name(&self) -> &'static str {
        "Cactus-Needle-3"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn num_layers(&self) -> usize {
        ACTIVE_LAYERS
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn num_kv_heads(&self) -> usize {
        self.num_heads
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

/// Zero-branch compile-time subnetwork depth ladder slice (2L, 4L, 8L, 16L, 20L).
#[derive(Debug)]
pub struct NeedleSubnetwork<const ACTIVE_LAYERS: usize> {
    pub config: CactusNeedleConfig<ACTIVE_LAYERS>,
}

impl<const ACTIVE_LAYERS: usize> NeedleSubnetwork<ACTIVE_LAYERS> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CactusNeedleConfig::<ACTIVE_LAYERS>::default(),
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn active_layers(&self) -> usize {
        ACTIVE_LAYERS
    }
}

impl<const ACTIVE_LAYERS: usize> Default for NeedleSubnetwork<ACTIVE_LAYERS> {
    fn default() -> Self {
        Self::new()
    }
}
