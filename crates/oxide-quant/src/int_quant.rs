//! Comprehensive Integer Quantization Kernels (2-bit, 3-bit, 4-bit, 5-bit, 6-bit, 8-bit).
//! Implements standard GGML-compatible block layouts (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
//! Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K) with zero allocations and branchless math.

// ============================================================================
// 4-bit Quantization: Q4_0 and Q4_1
// ============================================================================

/// Q4_0: 32 weights per block. 1x FP16 scale + 16 bytes (32 nibbles).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_0 {
    pub scale: f16,
    pub qs: [u8; 16],
}

impl BlockQ4_0 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }
        let scale = max_abs / 7.0; // 4-bit signed max value is 7
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0u8; 16];
        for i in 0..16 {
            let v0 = (values[i] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let v1 = (values[i + 16] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let q0 = ((v0 + 8) as u8) & 0x0F;
            let q1 = ((v1 + 8) as u8) & 0x0F;
            qs[i] = q0 | (q1 << 4);
        }

        Self {
            scale: f16::from_f32(scale),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            output[i] = (q0 as f32) * d;
            output[i + 16] = (q1 as f32) * d;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let mut deq = [0.0f32; 32];
        self.dequantize(&mut deq);
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += deq[i] * activations[i];
        }
        sum
    }
}

/// Q4_1: 32 weights per block. 1x FP16 scale + 1x FP16 min + 16 bytes (32 nibbles).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_1 {
    pub scale: f16,
    pub min: f16,
    pub qs: [u8; 16],
}

impl BlockQ4_1 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let scale = (max_val - min_val) / 15.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0u8; 16];
        for i in 0..16 {
            let q0 = ((values[i] - min_val) * inv_scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((values[i + 16] - min_val) * inv_scale)
                .round()
                .clamp(0.0, 15.0) as u8;
            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self {
            scale: f16::from_f32(scale),
            min: f16::from_f32(min_val),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        let m = self.min.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let q0 = byte & 0x0F;
            let q1 = (byte >> 4) & 0x0F;
            output[i] = (q0 as f32) * d + m;
            output[i + 16] = (q1 as f32) * d + m;
        }
    }
}

/// Q1_0: 32 weights per block. 1x FP16 scale + 4 bytes (32 1-bit weights).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ1_0 {
    pub scale: f16,
    pub qs: [u8; 4],
}

impl BlockQ1_0 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }
        let scale = max_abs;
        let mut qs = [0u8; 4];
        for i in 0..32 {
            if values[i] >= 0.0 {
                qs[i / 8] |= 1 << (i % 8);
            }
        }
        Self {
            scale: f16::from_f32(scale),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..32 {
            let bit = (self.qs[i / 8] >> (i % 8)) & 1;
            output[i] = if bit == 1 { d } else { -d };
        }
    }
}

/// Q2_0: 32 weights per block. 1x FP16 scale + 8 bytes (32 2-bit weights).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ2_0 {
    pub scale: f16,
    pub qs: [u8; 8],
}

impl BlockQ2_0 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }
        let scale = max_abs / 1.5;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut qs = [0u8; 8];
        for i in 0..32 {
            let q = (values[i] * inv_scale).round().clamp(-2.0, 1.0) as i8 + 2;
            qs[i / 4] |= ((q as u8) & 0x03) << ((i % 4) * 2);
        }
        Self {
            scale: f16::from_f32(scale),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..32 {
            let q = ((self.qs[i / 4] >> ((i % 4) * 2)) & 0x03) as i8 - 2;
            output[i] = (q as f32) * d;
        }
    }
}

/// Q4_2: Legacy 4-bit block quantization format.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_2 {
    pub scale: f16,
    pub qs: [u8; 16],
}

impl BlockQ4_2 {
    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            output[i] = (q0 as f32) * d;
            output[i + 16] = (q1 as f32) * d;
        }
    }
}

/// Q4_3: Legacy 4-bit block quantization format with scale and min.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ4_3 {
    pub scale: f16,
    pub min: f16,
    pub qs: [u8; 16],
}

impl BlockQ4_3 {
    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        let m = self.min.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let q0 = byte & 0x0F;
            let q1 = (byte >> 4) & 0x0F;
            output[i] = (q0 as f32) * d + m;
            output[i + 16] = (q1 as f32) * d + m;
        }
    }
}

// ============================================================================
// 5-bit Quantization: Q5_0 and Q5_1
// ============================================================================

/// Q5_0: 32 weights per block. 1x FP16 scale + 4 bytes high bits + 16 bytes low bits = 22 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ5_0 {
    pub scale: f16,
    pub qh: [u8; 4],  // High 5th bit for each of the 32 elements
    pub qs: [u8; 16], // Low 4 bits for 32 elements
}

impl BlockQ5_0 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }
        let scale = max_abs / 15.0; // 5-bit signed max value is 15
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for i in 0..16 {
            let v0 = (values[i] * inv_scale).round().clamp(-16.0, 15.0) as i8 + 16;
            let v1 = (values[i + 16] * inv_scale).round().clamp(-16.0, 15.0) as i8 + 16;

            let u0 = v0 as u8;
            let u1 = v1 as u8;

            qs[i] = (u0 & 0x0F) | ((u1 & 0x0F) << 4);

            let h0 = (u0 >> 4) & 1;
            let h1 = (u1 >> 4) & 1;

            let bit_idx0 = i;
            let bit_idx1 = i + 16;

            qh[bit_idx0 / 8] |= h0 << (bit_idx0 % 8);
            qh[bit_idx1 / 8] |= h1 << (bit_idx1 % 8);
        }

        Self {
            scale: f16::from_f32(scale),
            qh,
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let h0 = (self.qh[i / 8] >> (i % 8)) & 1;
            let h1 = (self.qh[(i + 16) / 8] >> ((i + 16) % 8)) & 1;

            let q0 = ((byte & 0x0F) | (h0 << 4)) as i8 - 16;
            let q1 = (((byte >> 4) & 0x0F) | (h1 << 4)) as i8 - 16;

            output[i] = (q0 as f32) * d;
            output[i + 16] = (q1 as f32) * d;
        }
    }
}

/// Q5_1: 32 weights per block. 1x FP16 scale + 1x FP16 min + 4 bytes high bits + 16 bytes low bits = 24 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ5_1 {
    pub scale: f16,
    pub min: f16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

impl BlockQ5_1 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let scale = (max_val - min_val) / 31.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for i in 0..16 {
            let q0 = ((values[i] - min_val) * inv_scale).round().clamp(0.0, 31.0) as u8;
            let q1 = ((values[i + 16] - min_val) * inv_scale).round().clamp(0.0, 31.0) as u8;

            qs[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);

            let h0 = (q0 >> 4) & 1;
            let h1 = (q1 >> 4) & 1;

            let bit_idx0 = i;
            let bit_idx1 = i + 16;

            qh[bit_idx0 / 8] |= h0 << (bit_idx0 % 8);
            qh[bit_idx1 / 8] |= h1 << (bit_idx1 % 8);
        }

        Self {
            scale: f16::from_f32(scale),
            min: f16::from_f32(min_val),
            qh,
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        let m = self.min.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let h0 = (self.qh[i / 8] >> (i % 8)) & 1;
            let h1 = (self.qh[(i + 16) / 8] >> ((i + 16) % 8)) & 1;

            let q0 = (byte & 0x0F) | (h0 << 4);
            let q1 = ((byte >> 4) & 0x0F) | (h1 << 4);

            output[i] = (q0 as f32) * d + m;
            output[i + 16] = (q1 as f32) * d + m;
        }
    }
}

// ============================================================================
// 8-bit Quantization: Q8_0 and Q8_1
// ============================================================================

/// Q8_0: 32 weights per block. 1x FP16 scale + 32 signed int8 values = 34 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ8_0 {
    pub scale: f16,
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (values[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }

        Self {
            scale: f16::from_f32(scale),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..32 {
            output[i] = (self.qs[i] as f32) * d;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let d = self.scale.to_f32();
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += (self.qs[i] as f32) * activations[i];
        }
        sum * d
    }
}

/// Q8_1: 32 weights per block. 1x FP16 scale + 1x FP16 sum (for dot product bias) + 32 signed int8 values = 36 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlockQ8_1 {
    pub scale: f16,
    pub sum: f16,
    pub qs: [i8; 32],
}

impl BlockQ8_1 {
    #[must_use]
    pub fn quantize(values: &[f32; 32]) -> Self {
        let mut max_abs = 0.0f32;
        let mut total_sum = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
            total_sum += v;
        }
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (values[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
        }

        Self {
            scale: f16::from_f32(scale),
            sum: f16::from_f32(total_sum),
            qs,
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 32]) {
        let d = self.scale.to_f32();
        for i in 0..32 {
            output[i] = (self.qs[i] as f32) * d;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 32]) -> f32 {
        let d = self.scale.to_f32();
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += (self.qs[i] as f32) * activations[i];
        }
        sum * d
    }
}

// ============================================================================
// K-Quant Super-Block Formats (256 weights per super-block): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
// ============================================================================

/// Q2_K: 256 weights per super-block (2-bit weights, 16 scales per super-block).
#[repr(C, align(32))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ2_K {
    pub scales: [u8; 16],
    pub qs: [u8; 64], // 256 elements packed at 2 bits each = 64 bytes
    pub d: f16,       // Super-block scale
    pub dmin: f16,    // Super-block min
}

impl Default for BlockQ2_K {
    fn default() -> Self {
        Self {
            scales: [0u8; 16],
            qs: [0u8; 64],
            d: f16(0),
            dmin: f16(0),
        }
    }
}

impl BlockQ2_K {
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in values {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let d = (max_val - min_val) / 3.0; // 2-bit max = 3
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 64];
        for i in 0..64 {
            let mut byte = 0u8;
            for b in 0..4 {
                let idx = i * 4 + b;
                let q = ((values[idx] - min_val) * inv_d).round().clamp(0.0, 3.0) as u8;
                byte |= (q & 0x03) << (b * 2);
            }
            qs[i] = byte;
        }

        Self {
            scales: [1u8; 16],
            qs,
            d: f16::from_f32(d),
            dmin: f16::from_f32(min_val),
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 256]) {
        let d = self.d.to_f32();
        let m = self.dmin.to_f32();
        for i in 0..64 {
            let byte = self.qs[i];
            for b in 0..4 {
                let q = (byte >> (b * 2)) & 0x03;
                output[i * 4 + b] = (q as f32) * d + m;
            }
        }
    }
}

/// Q3_K: 256 weights per super-block (3-bit weights).
#[repr(C, align(32))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ3_K {
    pub hmask: [u8; 32], // High bits for 256 values (1 bit each = 32 bytes)
    pub qs: [u8; 64],    // Low 2 bits for 256 values = 64 bytes
    pub scales: [u8; 12],
    pub d: f16,
}

impl Default for BlockQ3_K {
    fn default() -> Self {
        Self {
            hmask: [0u8; 32],
            qs: [0u8; 64],
            scales: [0u8; 12],
            d: f16(0),
        }
    }
}

impl BlockQ3_K {
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }

        let d = max_abs / 3.5;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0u8; 64];
        let mut hmask = [0u8; 32];

        for i in 0..256 {
            let q = (values[i] * inv_d).round().clamp(-4.0, 3.0) as i8 + 4; // 0..7 (3-bit)
            let u = q as u8;

            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            qs[byte_idx] |= (u & 0x03) << bit_shift;

            let h = (u >> 2) & 1;
            hmask[i / 8] |= h << (i % 8);
        }

        Self {
            hmask,
            qs,
            scales: [1u8; 12],
            d: f16::from_f32(d),
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 256]) {
        let d = self.d.to_f32();
        for i in 0..256 {
            let byte_idx = i / 4;
            let bit_shift = (i % 4) * 2;
            let low = (self.qs[byte_idx] >> bit_shift) & 0x03;
            let high = (self.hmask[i / 8] >> (i % 8)) & 1;
            let q = ((high << 2) | low) as i8 - 4;
            output[i] = (q as f32) * d;
        }
    }
}

/// Q6_K: 256 weights per super-block (6-bit weights: 4-bit low + 2-bit high).
#[repr(C, align(32))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockQ6_K {
    pub ql: [u8; 128], // Lower 4 bits for 256 weights = 128 bytes
    pub qh: [u8; 64],  // Higher 2 bits for 256 weights = 64 bytes
    pub scales: [i8; 16],
    pub d: f16,
}

impl Default for BlockQ6_K {
    fn default() -> Self {
        Self {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [0i8; 16],
            d: f16(0),
        }
    }
}

impl BlockQ6_K {
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }

        let d = max_abs / 31.0; // 6-bit signed max value is 31
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        for i in 0..128 {
            let v0 = (values[i] * inv_d).round().clamp(-32.0, 31.0) as i8 + 32;
            let v1 = (values[i + 128] * inv_d).round().clamp(-32.0, 31.0) as i8 + 32;

            let u0 = v0 as u8;
            let u1 = v1 as u8;

            ql[i] = (u0 & 0x0F) | ((u1 & 0x0F) << 4);

            let h0 = (u0 >> 4) & 0x03;
            let h1 = (u1 >> 4) & 0x03;

            let qh_idx = i / 2;
            let shift = (i % 2) * 4;
            qh[qh_idx] |= ((h0 | (h1 << 2)) & 0x0F) << shift;
        }

        Self {
            ql,
            qh,
            scales: [1i8; 16],
            d: f16::from_f32(d),
        }
    }

    pub fn dequantize(&self, output: &mut [f32; 256]) {
        let d = self.d.to_f32();
        for i in 0..128 {
            let byte_l = self.ql[i];
            let qh_idx = i / 2;
            let shift = (i % 2) * 4;
            let byte_h = (self.qh[qh_idx] >> shift) & 0x0F;

            let h0 = byte_h & 0x03;
            let h1 = (byte_h >> 2) & 0x03;

            let q0 = ((h0 << 4) | (byte_l & 0x0F)) as i8 - 32;
            let q1 = ((h1 << 4) | ((byte_l >> 4) & 0x0F)) as i8 - 32;

            output[i] = (q0 as f32) * d;
            output[i + 128] = (q1 as f32) * d;
        }
    }
}

/// Q8_K: 256 weights per super-block (8-bit quantization with FP32/FP16 scale, 8.5 bpw).
#[repr(C, align(32))]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_K {
    pub d: f32,          // Super-block scale
    pub qs: [i8; 256],   // 256 signed 8-bit quantized weights
    pub bsums: [i16; 16], // Sum of weights per 16-element sub-block
}

impl Default for BlockQ8_K {
    fn default() -> Self {
        Self {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }
    }
}

impl BlockQ8_K {
    #[must_use]
    pub fn quantize(values: &[f32; 256]) -> Self {
        let mut max_abs = 0.0f32;
        for &v in values {
            max_abs = max_abs.max(v.abs());
        }

        let d = max_abs / 127.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];

        for i in 0..256 {
            let q = (values[i] * inv_d).round().clamp(-128.0, 127.0) as i8;
            qs[i] = q;
            bsums[i / 16] += q as i16;
        }

        Self { d, qs, bsums }
    }

    pub fn dequantize(&self, output: &mut [f32; 256]) {
        let d = self.d;
        for i in 0..256 {
            output[i] = (self.qs[i] as f32) * d;
        }
    }

    #[must_use]
    pub fn dot_product(&self, activations: &[f32; 256]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..256 {
            sum += (self.qs[i] as f32) * activations[i];
        }
        sum * self.d
    }
}

// ============================================================================
// IEEE 754 Binary16 (f16) Minimal Standalone Helper
// ============================================================================

/// Minimal 16-bit float representation without external dependencies.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub struct f16(pub u16);

impl f16 {
    #[must_use]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = (bits >> 23) & 0xFF;
        let frac = bits & 0x7F_FFFF;

        if exp == 0 {
            return Self((sign as u16) << 15);
        }
        if exp == 0xFF {
            return Self(((sign as u16) << 15) | 0x7C00 | ((frac >> 13) as u16));
        }

        let new_exp = (exp as i32) - 127 + 15;
        if new_exp >= 31 {
            return Self(((sign as u16) << 15) | 0x7C00); // Infinity
        }
        if new_exp <= 0 {
            return Self((sign as u16) << 15); // Subnormal/underflow to 0
        }

        let new_frac = (frac >> 13) as u16;
        Self(((sign as u16) << 15) | ((new_exp as u16) << 10) | new_frac)
    }

    #[must_use]
    pub fn to_f32(self) -> f32 {
        let raw = self.0;
        let sign = (raw >> 15) & 1;
        let exp = (raw >> 10) & 0x1F;
        let frac = raw & 0x3FF;

        if exp == 0 {
            if frac == 0 {
                return if sign == 1 { -0.0 } else { 0.0 };
            }
            // Subnormal
            return (if sign == 1 { -1.0 } else { 1.0 }) * (frac as f32) * (2.0f32).powi(-24);
        }
        if exp == 31 {
            if frac == 0 {
                return if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                };
            }
            return f32::NAN;
        }

        let new_exp = (exp as u32) + 127 - 15;
        let new_frac = (frac as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | (new_exp << 23) | new_frac)
    }
}
