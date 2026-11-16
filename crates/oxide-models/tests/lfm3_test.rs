use oxide_models::formats::GgufFile;
use oxide_models::llama3::{Llama3Config, Llama3Model};
use oxide_models::tokenizer::GgufTokenizer;
use std::path::Path;

#[test]
fn test_lfm3_real_model_inference() {
    let model_path = Path::new("/home/jrad/models/LFM3-1.7B-Q4_K_M.gguf");
    if !model_path.exists() {
        println!("LFM3 model not found at {:?}, skipping", model_path);
        return;
    }

    let file = std::fs::File::open(model_path).expect("open LFM3");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap LFM3") };
    let static_slice: &'static [u8] =
        unsafe { std::slice::from_raw_parts(mmap.as_ptr(), mmap.len()) };

    let gguf = GgufFile::parse(static_slice).expect("parse GGUF");
    let cfg = Llama3Config::from_gguf(&gguf);
    assert!(cfg.is_lfm3, "Config should flag is_lfm3");
    assert_eq!(cfg.hidden_dim, 2048);
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.conv_l_cache, 7);

    let tok = GgufTokenizer::from_gguf(&gguf);

    let mut model = Llama3Model::new_empty(cfg);
    model
        .load_from_gguf(&gguf, static_slice)
        .expect("load_from_gguf");

    // Verify ShortConv layers vs Attention layers
    let mut shortconv_count = 0;
    let mut attn_count = 0;
    for (i, layer) in model.layers.iter().enumerate() {
        if layer.shortconv.is_some() {
            shortconv_count += 1;
            println!("  Layer {i}: ShortConv");
        } else if layer.q_proj.rows() > 0 {
            attn_count += 1;
            println!(
                "  Layer {i}: Attention (gated={})",
                layer.attn_gate.is_some()
            );
            assert!(
                layer.attn_gate.is_some(),
                "LFM3 attention layer {i} must have attn_gate"
            );
        }
    }
    println!(
        "LFM3 verified: {} ShortConv layers, {} Gated Attention layers",
        shortconv_count, attn_count
    );
    assert_eq!(shortconv_count + attn_count, 24);

    let prompt = "Liquid foundation models are";
    let prompt_tokens = tok.encode(prompt);
    println!("Prompt '{}' -> {:?}", prompt, prompt_tokens);

    // Test with 2 tokens first
    let two_tokens = &prompt_tokens[0..2];
    let mut decode_cache_2 = model.create_kv_cache();
    let mut scratch_2 = model.create_scratch();
    for (pos, &token_id) in two_tokens.iter().enumerate() {
        model
            .forward_step_with_scratch(token_id, pos, &mut decode_cache_2, &mut scratch_2)
            .expect("forward_step");
    }
    let mut prefill_cache_2 = model.create_kv_cache();
    let prefill_logits_2 = model
        .forward_batch(two_tokens, 0, &mut prefill_cache_2)
        .expect("forward_batch");
    let diff_2 = scratch_2
        .logits
        .iter()
        .zip(&prefill_logits_2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max logit difference for 2 tokens: {diff_2:.6}");

    // 1. Autoregressive decode pass
    let mut decode_cache = model.create_kv_cache();
    let mut scratch = model.create_scratch();
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        model
            .forward_step_with_scratch(token_id, pos, &mut decode_cache, &mut scratch)
            .expect("forward_step");
        let norm = oxide_quant::simd::dot_f32(&scratch.hidden, &scratch.hidden).sqrt();
        println!("  [Decode] pos {pos} (token {token_id}): hidden norm = {norm:.4}");
        assert!(!norm.is_nan(), "Hidden norm must not be NaN");
    }
    let decode_logits = scratch.logits.clone();

    // 2. Batched prefill pass
    let mut prefill_cache = model.create_kv_cache();
    let prefill_logits = model
        .forward_batch(&prompt_tokens, 0, &mut prefill_cache)
        .expect("forward_batch");

    // 3. Verify logits parity between sequential decode and batched prefill
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..decode_logits.len() {
        let diff = (decode_logits[i] - prefill_logits[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
    }
    let mean_diff = sum_diff / decode_logits.len() as f32;
    let cos_sim = {
        let dot = oxide_quant::simd::dot_f32(&decode_logits, &prefill_logits);
        let n1 = oxide_quant::simd::dot_f32(&decode_logits, &decode_logits).sqrt();
        let n2 = oxide_quant::simd::dot_f32(&prefill_logits, &prefill_logits).sqrt();
        dot / (n1 * n2)
    };
    println!(
        "Logit stats: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, cosine_similarity={cos_sim:.8}"
    );

    // Top 5 next tokens for Decode
    let mut indexed_dec: Vec<_> = decode_logits.iter().copied().enumerate().collect();
    indexed_dec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top 5 predicted tokens (Decode):");
    for (rank, (id, val)) in indexed_dec.iter().take(5).enumerate() {
        let text = tok.decode_token(*id as u32);
        println!("  #{rank}: token {id} ('{text}') = {val:.4}");
    }

    // Top 5 next tokens for Prefill
    let mut indexed_pref: Vec<_> = prefill_logits.iter().copied().enumerate().collect();
    indexed_pref.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top 5 predicted tokens (Prefill):");
    for (rank, (id, val)) in indexed_pref.iter().take(5).enumerate() {
        let text = tok.decode_token(*id as u32);
        println!("  #{rank}: token {id} ('{text}') = {val:.4}");
    }

    assert!(
        cos_sim > 0.95,
        "Cosine similarity must be > 0.95, got {cos_sim}"
    );

    let top3_dec: Vec<_> = indexed_dec.iter().take(3).map(|(id, _)| *id).collect();
    assert!(
        top3_dec.contains(&indexed_pref[0].0),
        "Top 1 prefill token ({}) must be in top 3 decode tokens ({:?})",
        indexed_pref[0].0,
        top3_dec
    );
}
