use oxide_core::error::{EngineError, Result};
use oxide_core::memory::DevicePtr;
use std::fmt;
use std::marker::PhantomData;

/// Lifetime-parametric, pre-allocated static device memory arena for activations and KV pools.
pub struct DeviceMemoryArena<'arena> {
    base_ptr: DevicePtr<u8>,
    capacity_bytes: usize,
    slot_size_bytes: usize,
    num_slots: usize,
    slot_occupied: Vec<bool>,
    _marker: PhantomData<&'arena mut [u8]>,
}

// SAFETY: DeviceMemoryArena manages distinct device address offsets safely across threads.
unsafe impl Send for DeviceMemoryArena<'_> {}
// SAFETY: DeviceMemoryArena manages distinct device address offsets safely across threads.
unsafe impl Sync for DeviceMemoryArena<'_> {}

impl fmt::Debug for DeviceMemoryArena<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceMemoryArena")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("slot_size_bytes", &self.slot_size_bytes)
            .field("num_slots", &self.num_slots)
            .field("base_address", &self.base_ptr.as_device_address())
            .finish()
    }
}

impl DeviceMemoryArena<'_> {
    /// Creates a static arena from a pre-allocated device memory slab.
    ///
    /// # Safety
    /// `base_ptr` must be a valid, 128-byte aligned device allocation of at least `capacity_bytes`.
    pub unsafe fn from_raw_device_ptr(
        base_ptr: DevicePtr<u8>,
        capacity_bytes: usize,
        num_slots: usize,
    ) -> Result<Self> {
        if base_ptr.is_null() {
            return Err(EngineError::DeviceMemoryViolation { address: 0 });
        }
        let slot_size_bytes = capacity_bytes / num_slots.max(1);

        Ok(Self {
            base_ptr,
            capacity_bytes,
            slot_size_bytes,
            num_slots,
            slot_occupied: vec![false; num_slots],
            _marker: PhantomData,
        })
    }

    /// Binds an execution slot to a sequence request without dynamic allocation.
    pub fn bind_slot(&mut self, _sequence_id: u64, slot_idx: usize) -> Result<DevicePtr<u8>> {
        if slot_idx >= self.num_slots || self.slot_occupied[slot_idx] {
            return Err(EngineError::AllocationBoundsExceeded { slot_idx });
        }

        self.slot_occupied[slot_idx] = true;
        let byte_offset = slot_idx * self.slot_size_bytes;

        // SAFETY: Offset is bounded within pre-allocated static capacity.
        unsafe { Ok(self.base_ptr.offset(byte_offset)) }
    }

    /// Releases an execution slot back to the static pool.
    pub fn release_slot(&mut self, slot_idx: usize) -> Result<()> {
        if slot_idx >= self.num_slots {
            return Err(EngineError::AllocationBoundsExceeded { slot_idx });
        }
        self.slot_occupied[slot_idx] = false;
        Ok(())
    }

    /// Retrieves the device pointer for a given active slot.
    pub fn get_slot_ptr(&self, slot_idx: usize) -> Result<DevicePtr<u8>> {
        if slot_idx >= self.num_slots {
            return Err(EngineError::AllocationBoundsExceeded { slot_idx });
        }
        let byte_offset = slot_idx * self.slot_size_bytes;
        // SAFETY: Offset calculation is bounded by num_slots.
        unsafe { Ok(self.base_ptr.offset(byte_offset)) }
    }

    #[must_use]
    pub const fn slot_size_bytes(&self) -> usize {
        self.slot_size_bytes
    }

    #[must_use]
    pub const fn num_slots(&self) -> usize {
        self.num_slots
    }
}
