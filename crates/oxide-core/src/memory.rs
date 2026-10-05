use std::fmt;
use std::marker::PhantomData;

/// Zero-cost typed device pointer wrapping GPU / accelerator memory addresses.
///
/// Intentionally omits `Deref` and `DerefMut` implementations so that dereferencing
/// device VRAM on the CPU host is caught as an immediate compile-time failure.
#[repr(transparent)]
pub struct DevicePtr<T> {
    raw: *mut T,
    _marker: PhantomData<T>,
}

// SAFETY: DevicePtr wraps an opaque device address and does not dereference memory on host.
unsafe impl<T: Send> Send for DevicePtr<T> {}
// SAFETY: DevicePtr wraps an opaque device address and does not dereference memory on host.
unsafe impl<T: Sync> Sync for DevicePtr<T> {}

impl<T> Copy for DevicePtr<T> {}
impl<T> Clone for DevicePtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> fmt::Debug for DevicePtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DevicePtr").field(&self.raw).finish()
    }
}

impl<T> DevicePtr<T> {
    /// Creates a typed `DevicePtr` from a raw device address pointer.
    ///
    /// # Safety
    /// `raw` must be a valid, aligned device address allocated on the target hardware accelerator.
    #[inline(always)]
    pub const unsafe fn from_raw(raw: *mut T) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Creates a null `DevicePtr`.
    #[inline(always)]
    #[must_use]
    pub const fn null() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            _marker: PhantomData,
        }
    }

    /// Checks if the device pointer is null.
    #[inline(always)]
    #[must_use]
    pub fn is_null(self) -> bool {
        self.raw.is_null()
    }

    /// Returns the underlying raw pointer.
    #[inline(always)]
    #[must_use]
    pub fn as_raw(self) -> *mut T {
        self.raw
    }

    /// Returns the numerical 64-bit device memory address.
    #[inline(always)]
    #[must_use]
    pub fn as_device_address(self) -> u64 {
        self.raw as usize as u64
    }

    /// Computes an offset pointer in units of `T`.
    ///
    /// # Safety
    /// The offset must remain within the bounds of the allocated device buffer.
    #[inline(always)]
    #[must_use]
    pub unsafe fn offset(self, count: usize) -> Self {
        // SAFETY: Caller guarantees count remains within device allocation bounds.
        unsafe {
            Self {
                raw: self.raw.add(count),
                _marker: PhantomData,
            }
        }
    }

    /// Casts this `DevicePtr<T>` to `DevicePtr<U>`.
    #[inline(always)]
    #[must_use]
    pub fn cast<U>(self) -> DevicePtr<U> {
        DevicePtr {
            raw: self.raw.cast::<U>(),
            _marker: PhantomData,
        }
    }
}
