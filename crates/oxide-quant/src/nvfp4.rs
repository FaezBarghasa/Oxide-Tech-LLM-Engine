use oxide_core::traits::QuantScheme;

/// NVIDIA FP4 (E2M1) Microscopic Floating Point Quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvFp4;

impl QuantScheme for NvFp4 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        32
    }
}
