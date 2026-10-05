use oxide_core::memory::DevicePtr;

/// Calculates Chebyshev distance (L_infinity norm): max_i |u_i - v_i|
pub fn chebyshev_distance(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len(), "Vectors must have identical length");
    u.iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Calculates Cosine Similarity: (u . v) / (||u||_2 * ||v||_2)
pub fn cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len(), "Vectors must have identical length");
    let mut dot = 0.0f32;
    let mut norm_u_sq = 0.0f32;
    let mut norm_v_sq = 0.0f32;

    for (&a, &b) in u.iter().zip(v.iter()) {
        dot += a * b;
        norm_u_sq += a * a;
        norm_v_sq += b * b;
    }

    if norm_u_sq == 0.0 || norm_v_sq == 0.0 {
        return 0.0;
    }

    dot / (norm_u_sq.sqrt() * norm_v_sq.sqrt())
}

#[test]
fn test_device_ptr_invariants() {
    let mut val: u64 = 0xDEAD_BEEF;
    let raw = &mut val as *mut u64;

    // SAFETY: Creating DevicePtr for testing address properties.
    let dev_ptr = unsafe { DevicePtr::from_raw(raw) };
    assert_eq!(dev_ptr.as_raw(), raw);
    assert_eq!(dev_ptr.as_device_address(), raw as usize as u64);
    assert!(!dev_ptr.is_null());

    let null_ptr: DevicePtr<f32> = DevicePtr::null();
    assert!(null_ptr.is_null());
}

#[test]
fn test_golden_tensor_similarity_metrics() {
    let ref_vec = vec![0.12f32, 0.45, -0.33, 0.89, -0.05, 0.72];
    let rust_vec = vec![0.1201f32, 0.4499, -0.3301, 0.8902, -0.0501, 0.7199];

    let l_inf = chebyshev_distance(&ref_vec, &rust_vec);
    let cos_sim = cosine_similarity(&ref_vec, &rust_vec);

    assert!(l_inf < 1.5e-3, "Chebyshev distance exceeds FP16 tolerance: {l_inf}");
    assert!(cos_sim >= 0.9999, "Cosine similarity below threshold: {cos_sim}");
}
