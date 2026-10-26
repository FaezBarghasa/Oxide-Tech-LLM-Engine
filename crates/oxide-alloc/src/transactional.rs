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

    pub fn physical_blocks_mut(&mut self) -> &mut Vec<u32> {
        &mut self.physical_blocks
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

/// RAII guard that manages transactional KV page allocations for a slot or speculative draft.
/// If dropped before being committed (e.g. on client disconnect or speculative branch rejection),
/// it automatically rolls back uncommitted KV pages.
#[derive(Debug)]
pub struct TransactionalKvGuard<'a> {
    pub slot_id: usize,
    table: &'a mut TransactionalBlockTable,
    committed_tokens: usize,
    committed_blocks: usize,
    is_committed: bool,
}

impl<'a> TransactionalKvGuard<'a> {
    #[must_use]
    pub fn new(slot_id: usize, table: &'a mut TransactionalBlockTable) -> Self {
        let committed_tokens = table.active_tokens();
        let committed_blocks = table.physical_blocks().len();
        Self {
            slot_id,
            table,
            committed_tokens,
            committed_blocks,
            is_committed: false,
        }
    }

    pub fn push_draft_block(&mut self, block_id: u32) {
        self.table.push_block(block_id);
    }

    pub fn commit(mut self, accepted_tokens: usize) {
        self.table.commit_speculation(accepted_tokens);
        self.is_committed = true;
    }

    pub fn rollback_explicit(&mut self) {
        self.table
            .rollback(self.committed_tokens, self.table.active_tokens());
        self.table.physical_blocks_mut().truncate(self.committed_blocks);
        self.is_committed = true;
    }
}

impl Drop for TransactionalKvGuard<'_> {
    fn drop(&mut self) {
        if !self.is_committed {
            self.table
                .rollback(self.committed_tokens, self.table.active_tokens());
            self.table.physical_blocks_mut().truncate(self.committed_blocks);
        }
    }
}

/// Speculative draft transaction wrapping speculative token additions with atomic commit / rewind.
#[derive(Debug)]
pub struct KvTransaction {
    pub slot_id: usize,
    pub base_tokens: usize,
    pub speculated_tokens: usize,
    pub draft_blocks: Vec<u32>,
    pub committed: bool,
}

impl KvTransaction {
    #[must_use]
    pub fn begin(slot_id: usize, base_tokens: usize) -> Self {
        Self {
            slot_id,
            base_tokens,
            speculated_tokens: 0,
            draft_blocks: Vec::new(),
            committed: false,
        }
    }

    pub fn add_draft_token(&mut self, block_id: Option<u32>) {
        self.speculated_tokens += 1;
        if let Some(b) = block_id {
            self.draft_blocks.push(b);
        }
    }

    pub fn commit(&mut self, table: &mut TransactionalBlockTable, accepted_tokens: usize) {
        let actual_accepted = accepted_tokens.min(self.speculated_tokens);
        table.commit_speculation(actual_accepted);
        let retained_blocks = if (self.base_tokens + actual_accepted) == 0 {
            0
        } else {
            (self.base_tokens + actual_accepted).div_ceil(table.block_size())
        };
        table.physical_blocks_mut().truncate(retained_blocks);
        self.committed = true;
    }

    pub fn rollback(&mut self, table: &mut TransactionalBlockTable) {
        table.rollback(self.base_tokens, self.base_tokens + self.speculated_tokens);
        self.committed = true;
    }
}
