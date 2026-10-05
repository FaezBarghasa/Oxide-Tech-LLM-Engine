use oxide_core::error::Result;

/// Monarch Hadamard Block-Diagonal MLP.
/// Reduces parameter complexity from O(D^2) to O(D sqrt(D)) using block-diagonal matrices
/// B1, B2 interleaved with permutation operators P1, P2.
#[derive(Debug)]
pub struct MonarchMlp {
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub block_dim: usize,
    pub b1_weights: Vec<f32>,
    pub b2_weights: Vec<f32>,
}

impl MonarchMlp {
    #[must_use]
    pub fn new(hidden_dim: usize) -> Self {
        // sqrt(hidden_dim) block decomposition
        let num_blocks = (hidden_dim as f64).sqrt() as usize;
        let block_dim = hidden_dim / num_blocks;

        let b1_size = num_blocks * block_dim * block_dim;
        let b2_size = num_blocks * block_dim * block_dim;

        Self {
            hidden_dim,
            num_blocks,
            block_dim,
            b1_weights: vec![0.01f32; b1_size],
            b2_weights: vec![0.01f32; b2_size],
        }
    }

    /// Evaluates the forward pass of the Monarch factorized MLP: y = P2 B2 P1 B1 x
    pub fn forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        assert_eq!(input.len(), self.hidden_dim);
        assert_eq!(output.len(), self.hidden_dim);

        let mut intermediate = vec![0.0f32; self.hidden_dim];

        // Stage 1: Block-diagonal multiply B1
        for b in 0..self.num_blocks {
            let offset = b * self.block_dim;
            let weight_offset = b * self.block_dim * self.block_dim;

            for i in 0..self.block_dim {
                let mut sum = 0.0f32;
                for j in 0..self.block_dim {
                    let w = self.b1_weights[weight_offset + i * self.block_dim + j];
                    sum += input[offset + j] * w;
                }
                intermediate[offset + i] = sum;
            }
        }

        // Stage 2: Permutation P1 (transpose num_blocks x block_dim)
        let mut permuted = vec![0.0f32; self.hidden_dim];
        for i in 0..self.num_blocks {
            for j in 0..self.block_dim {
                permuted[j * self.num_blocks + i] = intermediate[i * self.block_dim + j];
            }
        }

        // Stage 3: Block-diagonal multiply B2
        for b in 0..self.num_blocks {
            let offset = b * self.block_dim;
            let weight_offset = b * self.block_dim * self.block_dim;

            for i in 0..self.block_dim {
                let mut sum = 0.0f32;
                for j in 0..self.block_dim {
                    let w = self.b2_weights[weight_offset + i * self.block_dim + j];
                    sum += permuted[offset + j] * w;
                }
                output[offset + i] = sum;
            }
        }

        Ok(())
    }
}
