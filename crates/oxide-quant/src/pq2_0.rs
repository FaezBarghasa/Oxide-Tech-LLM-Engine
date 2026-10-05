use oxide_core::traits::QuantScheme;

/// Packed Quant 2-bit standard format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedQuant2_0;

impl QuantScheme for PackedQuant2_0 {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        2.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        64
    }
}
