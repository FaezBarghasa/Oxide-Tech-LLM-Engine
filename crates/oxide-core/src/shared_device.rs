//! Zero-copy shared GPU device handle and memory arena orchestration.
//!
//! Provides `SharedDeviceHandle` for sharing CUDA driver contexts between engines
//! (e.g. `crystal-mind`, `ort`, `burn`) and pinned host RAM memory hibernation
//! for 8 GB GPU operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Shared CUDA/accelerator device context handle.
#[derive(Debug, Clone)]
pub struct SharedDeviceHandle {
    pub raw_context_ptr: usize,
    pub vram_budget_bytes: usize,
    pub allocated_bytes: Arc<AtomicUsize>,
    pub is_hibernated: Arc<AtomicBool>,
}

impl SharedDeviceHandle {
    #[must_use]
    pub fn new(raw_context_ptr: usize, vram_budget_bytes: usize) -> Self {
        Self {
            raw_context_ptr,
            vram_budget_bytes,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            is_hibernated: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Checks if device memory can accommodate an allocation without exceeding budget.
    #[must_use]
    pub fn can_allocate(&self, bytes: usize) -> bool {
        self.allocated_bytes.load(Ordering::Relaxed) + bytes <= self.vram_budget_bytes
    }

    /// Records device allocation bytes.
    pub fn record_allocation(&self, bytes: usize) -> bool {
        let mut cur = self.allocated_bytes.load(Ordering::Relaxed);
        loop {
            if cur + bytes > self.vram_budget_bytes {
                return false;
            }
            match self.allocated_bytes.compare_exchange_weak(
                cur,
                cur + bytes,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(val) => cur = val,
            }
        }
    }

    /// Releases recorded device allocation bytes.
    pub fn release_allocation(&self, bytes: usize) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::SeqCst);
    }

    /// Sets the engine hibernation state (parking weights into host RAM during concurrent audio DSP).
    pub fn set_hibernated(&self, hibernated: bool) {
        self.is_hibernated.store(hibernated, Ordering::SeqCst);
    }

    #[must_use]
    pub fn is_hibernated(&self) -> bool {
        self.is_hibernated.load(Ordering::SeqCst)
    }
}

/// Unified Caching Memory Arena for fast weight parking and wake-up.
#[derive(Debug)]
pub struct UnifiedCachingArena {
    pub device_handle: SharedDeviceHandle,
    pub pinned_host_capacity: usize,
    pub pinned_host_used: usize,
}

impl UnifiedCachingArena {
    #[must_use]
    pub fn new(device_handle: SharedDeviceHandle, pinned_host_capacity: usize) -> Self {
        Self {
            device_handle,
            pinned_host_capacity,
            pinned_host_used: 0,
        }
    }

    /// Simulates fast streaming hibernation of weights from VRAM to pinned host RAM.
    pub fn hibernate_weights(&mut self, weight_bytes: usize) -> bool {
        if self.pinned_host_used + weight_bytes <= self.pinned_host_capacity {
            self.pinned_host_used += weight_bytes;
            self.device_handle.release_allocation(weight_bytes);
            self.device_handle.set_hibernated(true);
            true
        } else {
            false
        }
    }

    /// Simulates waking up parked weights back into VRAM from pinned host RAM.
    pub fn wake_weights(&mut self, weight_bytes: usize) -> bool {
        if self.pinned_host_used >= weight_bytes
            && self.device_handle.record_allocation(weight_bytes)
        {
            self.pinned_host_used -= weight_bytes;
            self.device_handle.set_hibernated(false);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_device_handle_budget_and_hibernation() {
        let handle = SharedDeviceHandle::new(0xDEAD_BEEF, 8 * 1024 * 1024 * 1024);
        assert!(handle.can_allocate(4 * 1024 * 1024 * 1024));
        assert!(handle.record_allocation(4 * 1024 * 1024 * 1024));
        assert_eq!(
            handle.allocated_bytes.load(Ordering::Relaxed),
            4 * 1024 * 1024 * 1024
        );

        let mut arena = UnifiedCachingArena::new(handle.clone(), 16 * 1024 * 1024 * 1024);
        assert!(arena.hibernate_weights(4 * 1024 * 1024 * 1024));
        assert!(handle.is_hibernated());
        assert_eq!(handle.allocated_bytes.load(Ordering::Relaxed), 0);

        assert!(arena.wake_weights(4 * 1024 * 1024 * 1024));
        assert!(!handle.is_hibernated());
        assert_eq!(
            handle.allocated_bytes.load(Ordering::Relaxed),
            4 * 1024 * 1024 * 1024
        );
    }
}
