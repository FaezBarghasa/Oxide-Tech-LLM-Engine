//! GPTQ and AWQ Post-Training Quantization Runtime Formats.
//!
//! Implements GPU-optimized packing layouts:
//! - GPTQ: Column-packed 4-bit integers in u32 (8 weights per word) with group-wise scales,
//!   zero-points, and activation-order (desc_act) permutation maps.
//! - AWQ: Interleaved nibble order [0, 4, 1, 5, 2, 6, 3, 7] enabling bank-conflict-free
//!   warp loading and Tensor Core GEMV throughput.

use oxide_core::traits::QuantScheme;

/// Group size options for GPTQ and AWQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantGroupSize {
    None,
    Group32,
    Group64,
    Group128,
}

impl QuantGroupSize {
    #[must_use]
    pub fn size(self) -> usize {
        match self {
            Self::None => usize::MAX,
            Self::Group32 => 32,
            Self::Group64 => 64,
            Self::Group128 => 128,
        }
    }
}

/// Supported bit-widths for GPTQ weight quantization (2, 3, 4, 8 bits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptqBitWidth {
    Bits2,
    Bits3,
    Bits4,
    Bits8,
}

impl GptqBitWidth {
    #[must_use]
    pub fn bits(self) -> usize {
        match self {
            Self::Bits2 => 2,
            Self::Bits3 => 3,
            Self::Bits4 => 4,
            Self::Bits8 => 8,
        }
    }
}

/// Precision modes for AWQ (W4A16, W4A8, W8A16, BF16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwqPrecisionMode {
    /// 4-bit weights, 16-bit activations (standard AWQ).
    W4A16,
    /// 4-bit weights, FP8 activations.
    W4A8,
    /// 8-bit weights, 16-bit activations.
    W8A16,
    /// BF16 precision.
    Bf16,
}

/// ZST marker for GPTQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptqQuant;

impl QuantScheme for GptqQuant {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        8
    }
}

/// GPTQ 4-Bit Packed Matrix Container.
#[derive(Debug, Clone, PartialEq)]
pub struct GptqWeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: QuantGroupSize,
    /// Packed weights: 8 weights per u32 (rows / 8 * cols elements).
    pub qweight: Vec<u32>,
    /// Scale factors per group: (cols * num_groups).
    pub scales: Vec<f32>,
    /// Zero points per group (packed into u32 or stored as u8).
    pub qzeros: Vec<u32>,
    /// Optional activation-order permutation vector (act_order / desc_act).
    pub perm: Option<Vec<u32>>,
}

impl GptqWeightMatrix {
    /// Unpacks and dequantizes an entire column or row of weights.
    #[must_use]
    pub fn dequantize_element(&self, row: usize, col: usize) -> f32 {
        let g_size = self.group_size.size();
        let group_idx = if g_size == usize::MAX { 0 } else { row / g_size };
        let scale_idx = group_idx * self.cols + col;
        let scale = self.scales.get(scale_idx).copied().unwrap_or(1.0);

        // Extract zero point (assuming 8 zeros packed per u32)
        let zero_word_idx = (group_idx / 8) * self.cols + col;
        let zero_shift = (group_idx % 8) * 4;
        let zero_val = if zero_word_idx < self.qzeros.len() {
            ((self.qzeros[zero_word_idx] >> zero_shift) & 0x0F) + 1
        } else {
            8
        };

        // Extract weight: 8 rows packed per u32 word
        let word_row = row / 8;
        let shift = (row % 8) * 4;
        let word_idx = word_row * self.cols + col;
        let qweight = if word_idx < self.qweight.len() {
            (self.qweight[word_idx] >> shift) & 0x0F
        } else {
            0
        };

        (qweight as i32 - zero_val as i32) as f32 * scale
    }

    /// Evaluates GEMV dot product: y = x * W
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);

        for c in 0..self.cols {
            let mut sum = 0.0f32;
            for r in 0..self.rows {
                let actual_r = self.perm.as_ref().map_or(r, |p| p[r] as usize);
                sum += x[actual_r] * self.dequantize_element(r, c);
            }
            y[c] = sum;
        }
    }
}

/// ZST marker for AWQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwqQuant;

impl QuantScheme for AwqQuant {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        8
    }
}

/// AWQ 4-Bit Packed Matrix Container.
///
/// AWQ interleaves nibbles to optimize warp coalescing:
/// byte 0: [w0, w4], byte 1: [w1, w5], byte 2: [w2, w6], byte 3: [w3, w7].
#[derive(Debug, Clone, PartialEq)]
pub struct AwqWeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub group_size: QuantGroupSize,
    pub qweight: Vec<u32>,
    pub scales: Vec<f32>,
    pub zeros: Vec<u32>,
}

/// AWQ nibble order mapping
const AWQ_REVERSE_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

impl AwqWeightMatrix {
    #[must_use]
    pub fn dequantize_element(&self, row: usize, col: usize) -> f32 {
        let g_size = self.group_size.size();
        let group_idx = if g_size == usize::MAX { 0 } else { row / g_size };
        let scale_idx = group_idx * self.cols + col;
        let scale = self.scales.get(scale_idx).copied().unwrap_or(1.0);

        let zero_word_idx = (group_idx / 8) * self.cols + col;
        let zero_shift = (group_idx % 8) * 4;
        let zero_val = if zero_word_idx < self.zeros.len() {
            (self.zeros[zero_word_idx] >> zero_shift) & 0x0F
        } else {
            0
        };

        let word_row = row / 8;
        let nibble_pos = row % 8;
        let awq_pos = AWQ_REVERSE_ORDER[nibble_pos];
        let shift = awq_pos * 4;
        let word_idx = word_row * self.cols + col;

        let qval = if word_idx < self.qweight.len() {
            (self.qweight[word_idx] >> shift) & 0x0F
        } else {
            0
        };

        (qval as i32 - zero_val as i32) as f32 * scale
    }

    /// Evaluates GEMV dot product: y = x * W
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.rows);
        assert_eq!(y.len(), self.cols);

        for c in 0..self.cols {
            let mut sum = 0.0f32;
            for r in 0..self.rows {
                sum += x[r] * self.dequantize_element(r, c);
            }
            y[c] = sum;
        }
    }
}
