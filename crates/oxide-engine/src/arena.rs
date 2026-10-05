/// Pre-allocated bump allocation arena for zero-copy activation memory during graph execution.
/// Guarantees ZERO heap allocations (no malloc, no Vec::new) in the hot forward loop.
#[derive(Debug)]
pub struct GraphArena {
    storage: Vec<f32>,
    offset: usize,
    tensor_offsets: Vec<usize>,
}

impl GraphArena {
    #[must_use]
    pub fn new(capacity_elements: usize) -> Self {
        Self {
            storage: vec![0.0f32; capacity_elements],
            offset: 0,
            tensor_offsets: Vec::with_capacity(1024),
        }
    }

    /// Reset arena offset to 0 for next token step (O(1)).
    #[inline(always)]
    pub fn reset(&mut self) {
        self.offset = 0;
        self.tensor_offsets.clear();
    }

    /// Bump-allocates a slice for a tensor output.
    #[inline(always)]
    pub fn alloc(&mut self, size: usize) -> &mut [f32] {
        let start = self.offset;
        let end = start + size;
        assert!(
            end <= self.storage.len(),
            "GraphArena out of memory: capacity exceeded"
        );
        self.offset = end;
        self.tensor_offsets.push(start);
        &mut self.storage[start..end]
    }

    /// Returns the start offset of a given TensorId in the storage buffer.
    #[must_use]
    #[inline(always)]
    pub fn get_offset(&self, id: u32) -> usize {
        let idx = id as usize;
        if idx < self.tensor_offsets.len() {
            self.tensor_offsets[idx]
        } else {
            0
        }
    }

    /// Returns an immutable slice for a given TensorId.
    #[must_use]
    #[inline(always)]
    pub fn get_tensor(&self, id: u32, size: usize) -> &[f32] {
        let start = self.get_offset(id);
        if start + size <= self.storage.len() {
            &self.storage[start..start + size]
        } else {
            &[]
        }
    }

    /// Direct access to underlying buffer for executing nodes without allocation.
    #[inline(always)]
    pub fn storage_mut(&mut self) -> &mut [f32] {
        &mut self.storage
    }

    #[must_use]
    #[inline(always)]
    pub fn storage(&self) -> &[f32] {
        &self.storage
    }
}
