//! Standard Uniform Symmetric and Asymmetric INT8 and INT4 Quantization.
//!
//! Provides enterprise-grade integer quantization algorithms compatible with standard
//! PyTorch, ONNX Runtime, TensorRT, and HuggingFace AutoRound/AWQ/GPTQ backends.

use oxide_core::traits::QuantScheme;

/// Symmetric Uniform INT8 (values in [-127, 127]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int8Sym {
    pub scale: f32,
}

impl QuantScheme for Int8Sym {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        1
    }
}

impl Int8Sym {
    #[must_use]
    pub fn quantize_slice(input: &[f32], output_q: &mut [i8]) -> f32 {
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        for (i, &v) in input.iter().enumerate() {
            output_q[i] = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }

        scale
    }

    pub fn dequantize_slice(q: &[i8], scale: f32, output: &mut [f32]) {
        for (i, &val) in q.iter().enumerate() {
            output[i] = (val as f32) * scale;
        }
    }

    #[must_use]
    pub fn dot_product(q: &[i8], activations: &[f32], scale: f32) -> f32 {
        let mut sum = 0.0f32;
        for (i, &val) in q.iter().enumerate() {
            sum += (val as f32) * activations[i];
        }
        sum * scale
    }
}

/// Asymmetric Uniform INT8 (values in [0, 255] with scale and zero-point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int8Asym {
    pub scale: f32,
    pub zero_point: i32,
}

impl QuantScheme for Int8Asym {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        8.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        1
    }
}

impl Int8Asym {
    #[must_use]
    pub fn quantize_slice(input: &[f32], output_q: &mut [u8]) -> Self {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in input {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let scale = if max_val > min_val {
            (max_val - min_val) / 255.0
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;
        let zero_point = (-min_val * inv_scale).round().clamp(0.0, 255.0) as i32;

        for (i, &v) in input.iter().enumerate() {
            let q = (v * inv_scale + zero_point as f32)
                .round()
                .clamp(0.0, 255.0) as u8;
            output_q[i] = q;
        }

        Self { scale, zero_point }
    }

    pub fn dequantize_slice(&self, q: &[u8], output: &mut [f32]) {
        for (i, &val) in q.iter().enumerate() {
            output[i] = ((val as i32 - self.zero_point) as f32) * self.scale;
        }
    }
}

/// Symmetric Uniform INT4 (values in [-8, 7], packed 2 nibbles per byte).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int4Sym {
    pub scale: f32,
}

impl QuantScheme for Int4Sym {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        2
    }
}

impl Int4Sym {
    /// Quantizes an even number of elements into packed bytes.
    #[must_use]
    pub fn quantize_slice(input: &[f32], output_packed: &mut [u8]) -> f32 {
        assert_eq!(input.len() % 2, 0);
        let mut max_abs = 0.0f32;
        for &v in input {
            max_abs = max_abs.max(v.abs());
        }

        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        for i in 0..output_packed.len() {
            let v0 = (input[i * 2] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let v1 = (input[i * 2 + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8;
            let q0 = ((v0 + 8) as u8) & 0x0F;
            let q1 = ((v1 + 8) as u8) & 0x0F;
            output_packed[i] = q0 | (q1 << 4);
        }

        scale
    }

    pub fn dequantize_slice(packed: &[u8], scale: f32, output: &mut [f32]) {
        for (i, &byte) in packed.iter().enumerate() {
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
            output[i * 2] = (q0 as f32) * scale;
            output[i * 2 + 1] = (q1 as f32) * scale;
        }
    }
}

/// Asymmetric Uniform INT4 (values in [0, 15], packed 2 nibbles per byte with scale and zero-point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int4Asym {
    pub scale: f32,
    pub zero_point: u8,
}

impl QuantScheme for Int4Asym {
    #[inline(always)]
    fn bits_per_weight() -> f32 {
        4.0
    }

    #[inline(always)]
    fn block_size() -> usize {
        2
    }
}

impl Int4Asym {
    #[must_use]
    pub fn quantize_slice(input: &[f32], output_packed: &mut [u8]) -> Self {
        assert_eq!(input.len() % 2, 0);
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &v in input {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let scale = if max_val > min_val {
            (max_val - min_val) / 15.0
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;
        let zero_point = (-min_val * inv_scale).round().clamp(0.0, 15.0) as u8;

        for i in 0..output_packed.len() {
            let q0 = (input[i * 2] * inv_scale + zero_point as f32)
                .round()
                .clamp(0.0, 15.0) as u8;
            let q1 = (input[i * 2 + 1] * inv_scale + zero_point as f32)
                .round()
                .clamp(0.0, 15.0) as u8;
            output_packed[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }

        Self { scale, zero_point }
    }

    pub fn dequantize_slice(&self, packed: &[u8], output: &mut [f32]) {
        for (i, &byte) in packed.iter().enumerate() {
            let q0 = (byte & 0x0F) as i32 - self.zero_point as i32;
            let q1 = ((byte >> 4) & 0x0F) as i32 - self.zero_point as i32;
            output[i * 2] = (q0 as f32) * self.scale;
            output[i * 2 + 1] = (q1 as f32) * self.scale;
        }
    }
}
