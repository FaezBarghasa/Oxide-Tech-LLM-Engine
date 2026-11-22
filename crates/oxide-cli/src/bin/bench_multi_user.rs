use anyhow::Result;
use oxide_core::error::EngineError;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let prompt_file = "bench_large_prompt.txt";
    let base_prompt = std::fs::read_to_string(prompt_file)?;

    let users = 4;
    let prompts: Vec<String> = (0..users)
        .map(|i| {
            format!(
                "{}\n\n[USER REQUEST {}]: Focus on Section {} trade-offs and provide detailed microarchitectural recommendations.",
                base_prompt,
                i + 1,
                i + 1
            )
        })
        .collect();

    println!("============================================================");
    println!("Oxide-Engine Multi-User Benchmark (N={} concurrent users)", users);
    println!("Model: /home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    println!("Hardware: NVIDIA RTX 4060 Laptop GPU (CUDA Native, Zero-Copy)");
    println!("============================================================");

    // Load backend and engine
    let model_path = "/home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let mut engine = oxide_engine::pipeline::InferencePipeline::new_llama3_cuda(
        model_path,
        None,
        4096,
        99,
        "Auto-Detect / System Native",
        None,
        16, // max_slots
        1024,
        8192,
        32768,
    )?;

    // Tokenize prompts
    let tokenizer = oxide_models::llama3::Llama3Tokenizer::from_gguf_file(model_path)?;
    let mut tokenized_prompts: Vec<Vec<u32>> = Vec::new();
    let mut total_prompt_tokens = 0;
    for (i, p) in prompts.iter().enumerate() {
        let toks = tokenizer.encode(p, false, false);
        total_prompt_tokens += toks.len();
        println!("User {} Prompt Tokens: {}", i + 1, toks.len());
        tokenized_prompts.push(toks);
    }

    let gen_tokens = 128;
    println!("\nExecuting Multi-User Prefill & Generation (4 streams x {} tokens)...", gen_tokens);

    let t_start = Instant::now();
    let mut user_completions: Vec<Vec<u32>> = vec![Vec::new(); users];

    // 1. Concurrent Prefill for all 4 slots
    for (slot_idx, tokens) in tokenized_prompts.iter().enumerate() {
        engine.prefill(slot_idx as u64 + 1, tokens, slot_idx as u16)?;
    }
    let t_prefill = t_start.elapsed();
    println!("All {} prefill passes completed in {:.2} ms ({:.2} tok/s aggregate)",
        users,
        t_prefill.as_secs_f64() * 1000.0,
        total_prompt_tokens as f64 / t_prefill.as_secs_f64()
    );

    // 2. Decode Loop across all 4 slots
    let t_decode_start = Instant::now();
    let mut active_tokens: Vec<u32> = tokenized_prompts.iter().map(|p| *p.last().unwrap_or(&1)).collect();

    for step in 0..gen_tokens {
        for slot_idx in 0..users {
            let cmd = oxide_core::StepCommand::new(
                slot_idx as u64 + 1,
                active_tokens[slot_idx],
                slot_idx as u16,
                false,
            );
            let completion = engine.step(&cmd)?;
            active_tokens[slot_idx] = completion.sampled_token;
            user_completions[slot_idx].push(completion.sampled_token);
        }
    }
    let t_total = t_start.elapsed();
    let t_decode = t_decode_start.elapsed();
    let total_gen_tokens = users * gen_tokens;

    println!("============================================================");
    println!("Oxide-Engine Multi-User (N={}) Results:", users);
    println!("Total Wall-Clock Latency: {:.2} ms", t_total.as_secs_f64() * 1000.0);
    println!("Total Prompt Tokens: {}", total_prompt_tokens);
    println!("Total Completion Tokens: {}", total_gen_tokens);
    println!("Prefill Time: {:.2} ms ({:.2} tok/s)", t_prefill.as_secs_f64() * 1000.0, total_prompt_tokens as f64 / t_prefill.as_secs_f64());
    println!("Decode Time: {:.2} ms ({:.2} tok/s aggregate)", t_decode.as_secs_f64() * 1000.0, total_gen_tokens as f64 / t_decode.as_secs_f64());
    println!("Per-User Effective Latency: {:.2} ms", t_total.as_secs_f64() * 1000.0 / users as f64);
    for i in 0..users {
        let decoded = tokenizer.decode(&user_completions[i])?;
        println!("User {} Output (first 100 chars): {:?}", i + 1, &decoded[..decoded.len().min(100)]);
    }
    println!("============================================================");

    Ok(())
}
