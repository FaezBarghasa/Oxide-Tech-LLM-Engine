use memmap2::Mmap;
use oxide_engine::SpecializedPipeline;
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    let model_path = PathBuf::from("/home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    // Pass max_slots = 4 so each slot gets 8192 / 4 = 2048 tokens capacity!
    let mut pipe = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        4,
        None,
    )?;

    let mut tokenized_prompts: Vec<Vec<u32>> = Vec::new();
    let mut total_prompt_tokens = 0;
    for (i, p) in prompts.iter().enumerate() {
        let toks = tokenizer.encode(p);
        total_prompt_tokens += toks.len();
        println!("User {} Prompt Tokens: {}", i + 1, toks.len());
        tokenized_prompts.push(toks);
    }

    let gen_tokens = 128;
    println!("\nExecuting Multi-User Prefill & Generation (4 streams x {} tokens)...", gen_tokens);

    let t_start = Instant::now();
    let mut user_completions: Vec<Vec<u32>> = vec![Vec::new(); users];

    // 1. Prefill for all 4 user slots
    for (slot_idx, tokens) in tokenized_prompts.iter().enumerate() {
        pipe.prefill(slot_idx as u64 + 1, tokens, slot_idx as u16)?;
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

    for _step in 0..gen_tokens {
        for slot_idx in 0..users {
            let cmd = oxide_core::StepCommand::new(
                slot_idx as u64 + 1,
                active_tokens[slot_idx],
                slot_idx as u16,
                false,
            );
            let completion = pipe.step(&cmd)?;
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
    println!("Prefill Time: {:.2} ms ({:.2} tok/s aggregate)", t_prefill.as_secs_f64() * 1000.0, total_prompt_tokens as f64 / t_prefill.as_secs_f64());
    println!("Decode Time: {:.2} ms ({:.2} tok/s aggregate)", t_decode.as_secs_f64() * 1000.0, total_gen_tokens as f64 / t_decode.as_secs_f64());
    println!("Per-User Effective Latency: {:.2} ms", t_total.as_secs_f64() * 1000.0 / users as f64);
    for i in 0..users {
        let decoded = tokenizer.decode(&user_completions[i]);
        println!("User {} Output (first 100 chars): {:?}", i + 1, &decoded[..decoded.len().min(100)]);
    }
    println!("============================================================");

    Ok(())
}
