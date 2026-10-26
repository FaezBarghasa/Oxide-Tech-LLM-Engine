use oxide_models::formats::{GgufFile, GgufQuantType};

#[test]
fn test_gguf_synthetic_binary_parsing() {
    let mut buffer = Vec::new();

    // 1. Magic 'GGUF' = [0x47, 0x47, 0x55, 0x46]
    buffer.extend_from_slice(b"GGUF");
    // 2. Version = 3 (u32)
    buffer.extend_from_slice(&3u32.to_le_bytes());
    // 3. Tensor count = 1 (u64)
    buffer.extend_from_slice(&1u64.to_le_bytes());
    // 4. Metadata KV count = 2 (u64)
    buffer.extend_from_slice(&2u64.to_le_bytes());

    // KV 1: "general.architecture" -> String "llama"
    let key1 = "general.architecture";
    buffer.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    buffer.extend_from_slice(key1.as_bytes());
    buffer.extend_from_slice(&8u32.to_le_bytes()); // Type 8 = String
    let val1 = "llama";
    buffer.extend_from_slice(&(val1.len() as u64).to_le_bytes());
    buffer.extend_from_slice(val1.as_bytes());

    // KV 2: "llama.context_length" -> UInt32 8192
    let key2 = "llama.context_length";
    buffer.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    buffer.extend_from_slice(key2.as_bytes());
    buffer.extend_from_slice(&4u32.to_le_bytes()); // Type 4 = UInt32
    buffer.extend_from_slice(&8192u32.to_le_bytes());

    // Tensor Info 1: "blk.0.attn_q.weight", 2 dims [4096, 4096], Q4_0, offset 0
    let t_name = "blk.0.attn_q.weight";
    buffer.extend_from_slice(&(t_name.len() as u64).to_le_bytes());
    buffer.extend_from_slice(t_name.as_bytes());
    buffer.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
    buffer.extend_from_slice(&4096u64.to_le_bytes()); // dim 0
    buffer.extend_from_slice(&4096u64.to_le_bytes()); // dim 1
    buffer.extend_from_slice(&2u32.to_le_bytes()); // Q4_0 quant type
    buffer.extend_from_slice(&0u64.to_le_bytes()); // offset

    let parsed = GgufFile::parse(&buffer).unwrap();
    let header = parsed.header.as_ref().unwrap();

    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 1);
    assert_eq!(header.metadata_kv_count, 2);

    assert_eq!(parsed.get_string("general.architecture"), Some("llama"));
    assert_eq!(parsed.get_u32("llama.context_length"), Some(8192));

    assert_eq!(parsed.tensors.len(), 1);
    let t = parsed.tensors.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(t.name, "blk.0.attn_q.weight");
    assert_eq!(t.dimensions, vec![4096, 4096]);
    assert_eq!(t.quant_type, GgufQuantType::Q4_0);
}

#[test]
fn test_real_gguf_models_in_home_dir() {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/jrad".to_string());
    let models_dir = std::path::Path::new(&home).join("models");
    if !models_dir.exists() {
        println!("No ~/models directory found, skipping real model parse test.");
        return;
    }

    let targets = [
        "Ornith-1.5-9B-Q4_K_M.gguf",
        "NVIDIA-Nemotron-3.5-Bolt-E4B-Uncensored-AD-Q4_K_M-00001-of-00001.gguf",
        "gemma-4-e2b-it.Q8_0.gguf",
        "Ternary-Bonsai-2-27B-Abliterated-PTQ1_0.gguf",
    ];

    for name in &targets {
        let candidate = models_dir.join(name);
        if candidate.exists() {
            println!("\n=== Inspecting: {} ===", name);
            let file = std::fs::File::open(&candidate).expect("Failed to open real GGUF model");
            let mmap = unsafe { memmap2::Mmap::map(&file).expect("Failed to mmap real GGUF model") };
            let parsed = GgufFile::parse(&mmap).expect("Failed to parse real GGUF model");
            let arch = parsed.architecture();
            println!("  Architecture: {arch}, tensor count: {}", parsed.tensors.len());
            if let Some(m) = parsed.get_string("tokenizer.ggml.model") {
                println!("  Tokenizer model: {m}");
            }
            let tok = oxide_models::GgufTokenizer::from_gguf(&parsed);
            let encoded = tok.encode("What is 2 + 2?");
            println!("  'What is 2 + 2?' -> {:?}", encoded);
            println!("  Decoded -> {:?}", tok.decode(&encoded));
            let cfg = oxide_models::llama3::Llama3Config::from_gguf(&parsed);
            println!("  Llama3Config: hidden={}, layers={}, heads={}, kv_heads={}, head_dim={}, inter={}, vocab={}, eps={:?}",
                cfg.hidden_dim, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate_dim, cfg.vocab_size, cfg.rms_norm_eps_f32());
            for (k, v) in &parsed.metadata {
                if k.starts_with(arch) || k.contains("rope") || k.contains("head") || k.contains("norm") {
                    println!("    meta: {k} = {:?}", v);
                }
            }
            println!("  qwen35 heads: head_count={:?}, head_count_kv={:?}, key_length={:?}",
                parsed.get_u32(&format!("{arch}.attention.head_count")),
                parsed.get_u32(&format!("{arch}.attention.head_count_kv")),
                parsed.get_u32(&format!("{arch}.attention.key_length")));
            for layer_idx in [0, 5, 14, 15, 20, 30] {
                for tname in &[
                    format!("blk.{layer_idx}.attn_q.weight"),
                    format!("blk.{layer_idx}.attn_output.weight"),
                    format!("blk.{layer_idx}.ffn_gate.weight"),
                    format!("blk.{layer_idx}.ffn_down.weight"),
                ] {
                    if let Some(t) = parsed.tensors.get(tname.as_str()) {
                        println!("    tensor {tname}: shape={:?}, type={:?}", t.dimensions, t.quant_type);
                    }
                }
            }
            if name == &"gemma-4-e2b-it.Q8_0.gguf" {
                println!("--- Testing Llama3Model forward step on {} ---", name);
                let mut model = oxide_models::llama3::Llama3Model::new_empty(cfg);
                let static_slice: &'static [u8] = unsafe {
                    std::slice::from_raw_parts(mmap.as_ptr(), mmap.len())
                };
                model.load_from_gguf(&parsed, static_slice).expect("load_from_gguf");
                let mut kv_cache = (0..model.config.num_layers)
                    .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
                    .collect::<Vec<_>>();
                let mut scratch = model.create_scratch();
                let prompt_tokens = tok.encode("What is 2 + 2?");
                println!("Prompt tokens: {:?}", prompt_tokens);
                for (pos, &t) in prompt_tokens.iter().enumerate() {
                    model.forward_step_with_scratch(t, pos, &mut kv_cache, &mut scratch).expect("forward_step");
                    let norm = oxide_quant::simd::dot_f32(&scratch.hidden, &scratch.hidden).sqrt();
                    println!("  pos {pos} (token {t}): hidden norm = {norm:.4}");
                }
                let mut indexed_logits: Vec<_> = scratch.logits.iter().copied().enumerate().collect();
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                println!("Top 5 logits after prompt:");
                for (rank, (idx, val)) in indexed_logits.iter().take(5).enumerate() {
                    let text = tok.decode_token(*idx as u32);
                    println!("  #{rank}: token {idx} ('{text}') = {val:.4}");
                }
            }
        } else {
            println!("Target {} not found in ~/models", name);
        }
    }
}
