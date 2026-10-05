use oxide_core::error::Result;

/// Direct-mapped hashed n-gram Engram gather table (70.8M parameters) in pinned memory.
#[derive(Debug)]
pub struct EngramGatherTable {
    pub num_entries: usize,
    pub vector_dim: usize,
    pub table_data: Vec<f32>,
    pub prime_base: u64,
}

impl EngramGatherTable {
    #[must_use]
    pub fn new(num_entries: usize, vector_dim: usize) -> Self {
        Self {
            num_entries,
            vector_dim,
            table_data: vec![0.001f32; num_entries * vector_dim],
            prime_base: 31,
        }
    }

    /// Computes the direct hash slot for a token n-gram history.
    #[must_use]
    pub fn compute_slot(&self, tokens: &[u32]) -> usize {
        let mut hash_val: u64 = 0;
        let mut p: u64 = 1;

        for &tok in tokens.iter().rev() {
            hash_val = hash_val.wrapping_add(u64::from(tok).wrapping_mul(p));
            p = p.wrapping_mul(self.prime_base);
        }

        (hash_val % (self.num_entries as u64)) as usize
    }

    /// Performs O(1) direct gather lookup from the Engram parameter table into the destination buffer.
    pub fn gather(&self, tokens: &[u32], dest: &mut [f32]) -> Result<()> {
        assert_eq!(dest.len(), self.vector_dim);
        let slot = self.compute_slot(tokens);
        let offset = slot * self.vector_dim;

        dest.copy_from_slice(&self.table_data[offset..offset + self.vector_dim]);
        Ok(())
    }
}
