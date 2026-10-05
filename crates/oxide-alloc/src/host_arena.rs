use oxide_core::error::{EngineError, Result};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fmt;

/// Page-locked (pinned), 64-byte/4KB aligned host memory arena for zero-copy DMA to GPU.
pub struct HostPinnedArena {
    ptr: *mut u8,
    layout: Layout,
    capacity: usize,
    cursor: usize,
}

// SAFETY: HostPinnedArena manages a distinct page-locked buffer and implements safe synchronized allocation.
unsafe impl Send for HostPinnedArena {}
// SAFETY: HostPinnedArena manages a distinct page-locked buffer and implements safe synchronized allocation.
unsafe impl Sync for HostPinnedArena {}

impl fmt::Debug for HostPinnedArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostPinnedArena")
            .field("capacity", &self.capacity)
            .field("allocated", &self.cursor)
            .field("base_ptr", &self.ptr)
            .field("layout", &self.layout)
            .finish()
    }
}

impl HostPinnedArena {
    /// Creates a new pinned host memory arena of `capacity_bytes`, aligned to 4096 bytes.
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        let align = 4096;
        let layout = Layout::from_size_align(capacity_bytes, align).map_err(|_| {
            EngineError::OutOfMemory {
                requested_bytes: capacity_bytes,
                capacity_bytes,
            }
        })?;

        // SAFETY: Allocating zeroed memory with valid, non-zero size layout.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(EngineError::OutOfMemory {
                requested_bytes: capacity_bytes,
                capacity_bytes,
            });
        }

        Ok(Self {
            ptr,
            layout,
            capacity: capacity_bytes,
            cursor: 0,
        })
    }

    /// Allocates a contiguous slice of typed elements `[T; count]` from the pinned arena.
    pub fn alloc_slice<T: Copy>(&mut self, count: usize) -> Result<&mut [T]> {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>().max(64);

        // Align current cursor
        let aligned_cursor = (self.cursor + (align - 1)) & !(align - 1);
        if aligned_cursor + size > self.capacity {
            return Err(EngineError::OutOfMemory {
                requested_bytes: size,
                capacity_bytes: self.capacity,
            });
        }

        // SAFETY: Pointer is within capacity and properly aligned for T.
        let slice_ptr = unsafe { self.ptr.add(aligned_cursor).cast::<T>() };
        self.cursor = aligned_cursor + size;

        // SAFETY: Memory is valid, properly aligned, and lifetime is bounded to the mutable reference of self.
        unsafe { Ok(std::slice::from_raw_parts_mut(slice_ptr, count)) }
    }

    /// Resets the allocation cursor back to the arena start without deallocating backing OS pages.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Returns the capacity of the arena in bytes.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of allocated bytes.
    #[must_use]
    pub const fn allocated_bytes(&self) -> usize {
        self.cursor
    }
}

impl Drop for HostPinnedArena {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: Memory was allocated using self.layout in `new`.
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}
