use oxide_core::traits::QuantScheme;

/// Cactus Quantized 2-bit for Needle architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedleCQ2;

impl QuantScheme for NeedleCQ2 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        128
    }
}
