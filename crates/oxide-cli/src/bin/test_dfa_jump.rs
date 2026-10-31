use memmap2::Mmap;
use oxide_core::dfa::DfaSchemaGrammar;
use oxide_engine::{ContinuousBatchingEngine, EngineHandle, SpecializedPipeline};
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== PHASE 5 VERIFICATION: Production DFA Jump-Forward Integration ===");

    // 1. Direct DFA lookahead verification
    let json_dfa = DfaSchemaGrammar::new_simple_json_validator();
    let (jump0, next0) = json_dfa.find_jump_forward_bytes(0, 16);
    println!(
        "DFA State 0 Jump: {:?} -> State {}",
        std::str::from_utf8(&jump0)?,
        next0
    );
    assert_eq!(jump0, b"{");
    assert_eq!(next0, 1);

    let (jump3, next3) = json_dfa.find_jump_forward_bytes(3, 16);
    println!(
        "DFA State 3 Jump: {:?} -> State {}",
        std::str::from_utf8(&jump3)?,
        next3
    );
    assert_eq!(jump3, b":");
    assert_eq!(next3, 4);

    let prefix_dfa = DfaSchemaGrammar::new_json_with_fixed_prefix("{\"answer\": ");
    let (jump_prefix, next_prefix) = prefix_dfa.find_jump_forward_bytes(0, 32);
    println!(
        "DFA Fixed Prefix Jump: {:?} -> State {}",
        std::str::from_utf8(&jump_prefix)?,
        next_prefix
    );
    assert_eq!(jump_prefix, b"{\"answer\": ");
    assert_eq!(next_prefix, 11);

    // 2. Hardware-backed verification on RTX 4060 dGPU
    let model_path = PathBuf::from("/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf");
    println!("Loading physical model weights from {:?}...", model_path);

    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(&gguf));

    let pipe = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        1,
        None,
    )?;
    let pipe_mutex = Arc::new(Mutex::new(pipe));

    // Warmup sampling
    {
        let mut p = pipe_mutex.lock().await;
        p.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        p.set_seq_pos(0, 0);
    }

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
    let engine = ContinuousBatchingEngine::new_with_pipeline(Arc::clone(&pipe_mutex), 1, cmd_rx)
        .with_tokenizer(Arc::clone(&tokenizer));
    engine.spawn();

    let client = EngineHandle::new(cmd_tx);

    let prompt = "Extract result as JSON:";
    let prompt_tokens = tokenizer.encode(prompt);
    println!(
        "Submitting request with prompt ({} tokens) and DFA fixed prefix constraint...",
        prompt_tokens.len()
    );

    let t_start = Instant::now();
    let mut token_rx = client
        .generate_stream_with_grammar(
            "req-dfa-verified".to_string(),
            prompt_tokens,
            12,
            0.0,
            1.0,
            Some(Arc::new(prefix_dfa)),
        )
        .await
        .map_err(|e| format!("Enqueue failed: {e}"))?;

    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let mut first_token_time = None;

    while let Some(event) = token_rx.recv().await {
        if first_token_time.is_none() {
            first_token_time = Some(t_start.elapsed());
        }
        let tok_str = tokenizer.decode_token(event.token_id);
        print!("{tok_str}");
        generated_text.push_str(&tok_str);
        generated_tokens.push(event.token_id);

        if event.is_terminal {
            break;
        }
    }
    println!();

    println!(
        "Generation complete! Total tokens: {}, First token latency: {:?}, Total duration: {:?}",
        generated_tokens.len(),
        first_token_time.unwrap_or_default(),
        t_start.elapsed()
    );

    println!("Full output text: {:?}", generated_text);
    assert!(
        generated_text.starts_with("{\"answer\": "),
        "Generated text must begin with deterministic DFA prefix"
    );

    println!(
        "=== SUCCESS: Production DFA Jump-Forward Integration verified on physical RTX 4060 dGPU! ==="
    );
    Ok(())
}
