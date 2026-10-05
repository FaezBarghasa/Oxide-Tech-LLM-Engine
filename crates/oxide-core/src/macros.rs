/// Static assertion macro verifying that a type's size in bytes does not exceed
/// a maximum threshold (default 4KB = 4096 bytes) to prevent thread stack overflows.
#[macro_export]
macro_rules! assert_stack_safe {
    ($t:ty) => {
        const _: () = {
            assert!(
                $crate::macros::size_of_val::<$t>() <= 4096,
                "Type exceeds 4KB thread stack safety threshold! Allocate in HostPinnedArena or Box."
            );
        };
    };
    ($t:ty, $max_bytes:expr) => {
        const _: () = {
            assert!(
                $crate::macros::size_of_val::<$t>() <= $max_bytes,
                "Type exceeds thread stack safety threshold! Allocate in HostPinnedArena or Box."
            );
        };
    };
}

/// Helper function to compute the byte size of a type in a const context.
#[must_use]
pub const fn size_of_val<T>() -> usize {
    std::mem::size_of::<T>()
}
