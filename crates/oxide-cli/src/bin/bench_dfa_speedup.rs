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
    println!("=================================================================");
    println!("     STRUCTURED OUTPUT DFA JUMP-FORWARD EMPIRICAL SPEEDUP        ");
    println!("=================================================================");

    let model_path = PathBuf::from("/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf");
    println!("Model: {:?}", model_path);

    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = Arc::new(GgufTokenizer::from_gguf(&gguf));

    let pipe = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        2,
        None,
    )?;
    let pipe_mutex = Arc::new(Mutex::new(pipe));

    {
        let mut p = pipe_mutex.lock().await;
        p.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        p.set_seq_pos(0, 0);
    }

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel();
    let engine = ContinuousBatchingEngine::new_with_pipeline(Arc::clone(&pipe_mutex), 2, cmd_rx)
        .with_tokenizer(Arc::clone(&tokenizer));
    engine.spawn();

    let client = EngineHandle::new(cmd_tx);

    let prompt = "Extract answer as JSON:";
    let prompt_tokens = tokenizer.encode(prompt);
    let fixed_prefix = "{\"answer\": ";
    let num_tokens = 15;

    // Run A: With DFA Jump-Forward Enabled
    println!("\n[Run A] Evaluating with DFA Jump-Forward ENABLED...");
    let prefix_dfa = DfaSchemaGrammar::new_json_with_fixed_prefix(fixed_prefix);
    let t_start_a = Instant::now();
    let mut rx_a = client
        .generate_stream_with_grammar(
            "req-dfa-on".to_string(),
            prompt_tokens.clone(),
            num_tokens,
            0.0,
            1.0,
            Some(Arc::new(prefix_dfa)),
        )
        .await
        .map_err(|e| format!("Enqueue failed: {e}"))?;

    let mut tokens_a = Vec::new();
    let mut text_a = String::new();
    while let Some(event) = rx_a.recv().await {
        let s = tokenizer.decode_token(event.token_id);
        text_a.push_str(&s);
        tokens_a.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }
    let duration_a = t_start_a.elapsed();
    let tps_a = (tokens_a.len() as f64) / duration_a.as_secs_f64();
    println!("  Output A    : {:?}", text_a);
    println!("  Tokens A    : {}", tokens_a.len());
    println!("  Duration A  : {:?}", duration_a);
    println!("  Throughput A: {:.2} tok/s", tps_a);

    // Run B: Standard Autoregressive Decode (DFA Jump-Forward Disabled)
    println!("\n[Run B] Evaluating with DFA Jump-Forward DISABLED (standard autoregressive)...");
    {
        let mut p = pipe_mutex.lock().await;
        p.set_seq_pos(1, 0);
    }
    let t_start_b = Instant::now();
    let mut rx_b = client
        .generate_stream_with_grammar(
            "req-dfa-off".to_string(),
            prompt_tokens.clone(),
            num_tokens,
            0.0,
            1.0,
            None,
        )
        .await
        .map_err(|e| format!("Enqueue failed: {e}"))?;

    let mut tokens_b = Vec::new();
    let mut text_b = String::new();
    while let Some(event) = rx_b.recv().await {
        let s = tokenizer.decode_token(event.token_id);
        text_b.push_str(&s);
        tokens_b.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }
    let duration_b = t_start_b.elapsed();
    let tps_b = (tokens_b.len() as f64) / duration_b.as_secs_f64();
    println!("  Output B    : {:?}", text_b);
    println!("  Tokens B    : {}", tokens_b.len());
    println!("  Duration B  : {:?}", duration_b);
    println!("  Throughput B: {:.2} tok/s", tps_b);

    let speedup = duration_b.as_secs_f64() / duration_a.as_secs_f64().max(1e-6);
    println!("\n=================================================================");
    println!("                       SPEEDUP VERDICT                           ");
    println!("=================================================================");
    println!("  DFA Jump-Forward Throughput : {:.2} tok/s", tps_a);
    println!("  Standard Decode Throughput  : {:.2} tok/s", tps_b);
    println!("  Effective Wall-Clock Speedup: {:.2}x", speedup);
    println!("=================================================================");

    assert!(
        speedup >= 1.0,
        "DFA jump-forward must be faster than standard decode"
    );
    println!("PASS: Phase 6 Multi-Engine Empirical Verification Complete.");
    Ok(())
}
