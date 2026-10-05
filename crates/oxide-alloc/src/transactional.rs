use std::fmt;

/// Atomic Transactional Block Table for speculative decoding rollbacks without device VRAM mutations.
pub struct TransactionalBlockTable {
    block_size: usize,
    physical_blocks: Vec<u32>,
    active_tokens: usize,
}

impl fmt::Debug for TransactionalBlockTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransactionalBlockTable")
            .field("block_size", &self.block_size)
            .field("num_blocks", &self.physical_blocks.len())
            .field("active_tokens", &self.active_tokens)
            .finish()
    }
}

impl TransactionalBlockTable {
    #[must_use]
    pub const fn new(block_size: usize) -> Self {
        Self {
            block_size,
            physical_blocks: Vec::new(),
            active_tokens: 0,
        }
    }

    /// Appends a new physical block to the table.
    pub fn push_block(&mut self, block_id: u32) {
        self.physical_blocks.push(block_id);
    }

    /// Commits accepted speculative tokens.
    pub fn commit_speculation(&mut self, accepted_tokens: usize) {
        self.active_tokens += accepted_tokens;
    }

    /// Performs an O(1) host metadata rollback when speculative tokens are rejected.
    /// Does NOT zero or mutate device VRAM; rejected tokens are overwritten by future steps.
    pub fn rollback(&mut self, accepted_tokens: usize, _total_speculated: usize) {
        let retained_blocks = if accepted_tokens == 0 {
            0
        } else {
            accepted_tokens.div_ceil(self.block_size)
        };

        self.physical_blocks.truncate(retained_blocks);
        self.active_tokens = accepted_tokens;
    }

    #[must_use]
    pub fn physical_blocks(&self) -> &[u32] {
        &self.physical_blocks
    }

    #[must_use]
    pub const fn active_tokens(&self) -> usize {
        self.active_tokens
    }

    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }
}
