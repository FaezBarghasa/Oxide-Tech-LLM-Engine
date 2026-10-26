//! Fast Walsh-Hadamard Transform (FWHT) and Fused Operations.
//! Implements block-diagonal randomized Hadamard rotations (QuaRot / SpinQuant style)
//! for outlier suppression in low-bit quantized LLM activations (PTQ1_0, PQ2_0, Q4_0).

/// Performs an in-place Fast Walsh-Hadamard Transform on a slice whose length is a power of 2.
/// Butterfly structure:
/// For h = 1, 2, 4, ... < N:
///   u = x[i], v = x[i + h]
///   x[i] = u + v
///   x[i + h] = u - v
pub fn fwht_butterfly_in_place(buf: &mut [f32]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FWHT length must be a power of 2");

    let mut h = 1;
    while h < n {
        let step = h << 1;
        let mut i = 0;
        while i < n {
            for j in 0..h {
                let u = buf[i + j];
                let v = buf[i + j + h];
                buf[i + j] = u + v;
                buf[i + j + h] = u - v;
            }
            i += step;
        }
        h = step;
    }
}

/// Applies block-wise Fast Walsh-Hadamard Transform across a row of length `row.len()`,
/// where each sub-block has size `block_size` (must be a power of 2, e.g., 32, 64, 128, 256).
/// Multiplies the result by `scale` (defaults to `1.0 / sqrt(block_size)` if `None`).
pub fn fwht_row_in_place(row: &mut [f32], block_size: usize, scale: Option<f32>) {
    assert!(
        block_size.is_power_of_two(),
        "FWHT block_size must be a power of 2"
    );
    assert_eq!(
        row.len() % block_size,
        0,
        "Row length must be a multiple of block_size"
    );

    let norm_scale = scale.unwrap_or_else(|| 1.0 / (block_size as f32).sqrt());

    for chunk in row.chunks_exact_mut(block_size) {
        fwht_butterfly_in_place(chunk);
        if (norm_scale - 1.0).abs() > 1e-6 {
            for v in chunk.iter_mut() {
                *v *= norm_scale;
            }
        }
    }
}

/// Applies a signed Fast Walsh-Hadamard Transform:
/// Multiplies elements by Rademacher signs (`signs[i] \in {-1.0, 1.0}`) prior to the Hadamard butterfly,
/// dispersing activation outliers uniformly across the hidden channels.
pub fn fwht_signed_row_in_place(
    row: &mut [f32],
    signs: &[f32],
    block_size: usize,
    scale: Option<f32>,
) {
    assert_eq!(
        row.len(),
        signs.len(),
        "Row and signs slices must have identical length"
    );

    // In-place sign modulation
    for (x, &s) in row.iter_mut().zip(signs.iter()) {
        *x *= s;
    }

    fwht_row_in_place(row, block_size, scale);
}

/// Fuses Residual Addition + RMSNorm + Channel Scaling + Optional Rademacher Signs + FWHT
/// in a single cache-efficient pass across the hidden dimension.
pub fn fused_rmsnorm_fwht(
    hidden: &mut [f32],
    residual: &[f32],
    weight: &[f32],
    signs: Option<&[f32]>,
    eps: f32,
    block_size: usize,
) {
    let dim = hidden.len();
    assert_eq!(dim, residual.len());
    assert_eq!(dim, weight.len());
    if let Some(s) = signs {
        assert_eq!(dim, s.len());
    }

    // 1. Residual Add & compute sum of squares for RMSNorm
    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        hidden[i] += residual[i];
        sum_sq += hidden[i] * hidden[i];
    }

    let rms_scale = 1.0 / (sum_sq / (dim as f32) + eps).sqrt();

    // 2. Normalize and scale by RMSNorm weight (and optional sign vector)
    if let Some(s) = signs {
        for i in 0..dim {
            hidden[i] = (hidden[i] * rms_scale * weight[i]) * s[i];
        }
    } else {
        for i in 0..dim {
            hidden[i] = hidden[i] * rms_scale * weight[i];
        }
    }

    // 3. Block Hadamard rotation
    fwht_row_in_place(hidden, block_size, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht_orthogonality() {
        let block_size = 64;
        let mut original = vec![0.0f32; block_size];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 * 0.314).sin();
        }

        let mut transformed = original.clone();
        fwht_row_in_place(&mut transformed, block_size, None);

        // H * H = N * I, so applying twice with 1/sqrt(N) scale is an involution (identity)
        let mut reconstructed = transformed.clone();
        fwht_row_in_place(&mut reconstructed, block_size, None);

        for (i, (&orig, &rec)) in original.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-5,
                "Inversion failed at {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_signed_fwht_energy_preservation() {
        let block_size = 128;
        let mut x = vec![1.0f32; block_size];
        let signs: Vec<f32> = (0..block_size)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let energy_before: f32 = x.iter().map(|v| v * v).sum();
        fwht_signed_row_in_place(&mut x, &signs, block_size, None);
        let energy_after: f32 = x.iter().map(|v| v * v).sum();

        assert!(
            (energy_before - energy_after).abs() < 1e-4,
            "Energy changed: before={}, after={}",
            energy_before,
            energy_after
        );
    }
}
