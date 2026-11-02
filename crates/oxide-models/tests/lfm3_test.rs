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
        } else if layer.q_proj.rows() > 0 {
            attn_count += 1;
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
    for i in 0..decode_logits.len() {
        let diff = (decode_logits[i] - prefill_logits[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("Max logit difference between prefill & decode: {max_diff:.6}");
    assert!(
        max_diff < 0.05,
        "Prefill and sequential decode logits must match! max_diff={max_diff}"
    );

    // Top 5 next tokens
    let mut indexed: Vec<_> = decode_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("Top 5 predicted tokens following '{}':", prompt);
    for (rank, (id, val)) in indexed.iter().take(5).enumerate() {
        let text = tok.decode_token(*id as u32);
        println!("  #{rank}: token {id} ('{text}') = {val:.4}");
    }
}
