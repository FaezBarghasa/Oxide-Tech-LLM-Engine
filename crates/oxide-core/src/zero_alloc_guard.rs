//! Zero-Allocation Guard for decode hot path verification.
//!
//! Provides counting allocator wrappers to assert zero heap allocations
//! during steady-state decode token forward passes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

/// Thread-safe allocation counter and tracking allocator.
#[derive(Debug, Default)]
pub struct CountingAlloc {
    inner: System,
    alloc_count: AtomicUsize,
    dealloc_count: AtomicUsize,
}

impl CountingAlloc {
    pub const fn new() -> Self {
        Self {
            inner: System,
            alloc_count: AtomicUsize::new(0),
            dealloc_count: AtomicUsize::new(0),
        }
    }

    pub fn reset(&self) {
        self.alloc_count.store(0, SeqCst);
        self.dealloc_count.store(0, SeqCst);
    }

    pub fn allocations(&self) -> usize {
        self.alloc_count.load(SeqCst)
    }

    pub fn deallocations(&self) -> usize {
        self.dealloc_count.load(SeqCst)
    }
}

// SAFETY: Forwarding to system allocator while tracking counts via atomics.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.alloc_count.fetch_add(1, SeqCst);
        // SAFETY: Delegating to inner GlobalAlloc.
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.dealloc_count.fetch_add(1, SeqCst);
        // SAFETY: Delegating to inner GlobalAlloc.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.alloc_count.fetch_add(1, SeqCst);
        // SAFETY: Delegating to inner GlobalAlloc.
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.alloc_count.fetch_add(1, SeqCst);
        // SAFETY: Delegating to inner GlobalAlloc.
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}
