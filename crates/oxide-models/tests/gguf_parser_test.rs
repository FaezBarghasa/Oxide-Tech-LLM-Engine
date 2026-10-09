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

    let candidate = models_dir.join("DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf");
    if candidate.exists() {
        println!("Testing parse of real model: {}", candidate.display());
        let file = std::fs::File::open(&candidate).expect("Failed to open real GGUF model");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("Failed to mmap real GGUF model") };
        let parsed = GgufFile::parse(&mmap).expect("Failed to parse real GGUF model");
        let arch = parsed.architecture();
        println!(
            "Parsed architecture: {arch}, tensor count: {}",
            parsed.tensors.len()
        );
        for (k, v) in &parsed.metadata {
            if k.contains("count") || k.contains("length") || k.contains("dim") {
                println!("  meta {k} = {v:?}");
            }
        }
        if let Some(t) = parsed.tensors.get("token_embd.weight") {
            println!(
                "token_embd.weight dims: {:?}, quant: {:?}",
                t.dimensions, t.quant_type
            );
        }
        if let Some(t) = parsed.tensors.get("output.weight") {
            println!(
                "output.weight dims: {:?}, quant: {:?}",
                t.dimensions, t.quant_type
            );
        }
        for (name, t) in &parsed.tensors {
            if name.starts_with("blk.0.") {
                println!(
                    "  {name} dims: {:?}, quant: {:?}",
                    t.dimensions, t.quant_type
                );
            }
        }
        assert!(!parsed.tensors.is_empty(), "Real model should have tensors");
        assert!(
            parsed.tensor_data_offset > 0,
            "Tensor data offset must be positive"
        );
    }
}
