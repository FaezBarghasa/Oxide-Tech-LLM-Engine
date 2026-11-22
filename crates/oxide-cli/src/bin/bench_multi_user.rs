use memmap2::Mmap;
use oxide_engine::SpecializedPipeline;
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tracing_subscriber::fmt::try_init();
    let prompt_file = "bench_large_prompt.txt";
    let base_prompt = std::fs::read_to_string(prompt_file)?;

    let args: Vec<String> = std::env::args().collect();
    let users: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let gen_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

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
    println!(
        "Oxide-Engine Multi-User Benchmark (N={} concurrent users)",
        users
    );
    println!("Model: /home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    println!("Hardware: NVIDIA RTX 4060 Laptop GPU (CUDA Native, Zero-Copy)");
    println!("============================================================");

    let model_path = PathBuf::from("/home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    // Pass max_slots = users so each slot gets appropriate capacity!
    let mut pipe = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        users.max(4),
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

    println!(
        "\nExecuting Multi-User Prefill & Generation ({} streams x {} tokens)...",
        users, gen_tokens
    );

    let t_start = Instant::now();
    let mut user_completions: Vec<Vec<u32>> = vec![Vec::new(); users];

    // 1. Unified Ragged Batched Prefill across all user slots in a single GPU pass
    let prompt_seqs: Vec<(&[u32], usize)> = tokenized_prompts
        .iter()
        .enumerate()
        .map(|(slot_idx, toks)| (toks.as_slice(), slot_idx))
        .collect();
    let initial_sampled = pipe.prefill_ragged_batched(&prompt_seqs)?;
    let t_prefill = t_start.elapsed();
    println!(
        "All {} prefill passes completed via Ragged Batching in {:.2} ms ({:.2} tok/s aggregate)",
        users,
        t_prefill.as_secs_f64() * 1000.0,
        total_prompt_tokens as f64 / t_prefill.as_secs_f64()
    );

    // 2. Continuous Batched Decode Loop across all active slots in a single GPU forward pass
    let t_decode_start = Instant::now();
    let mut active_tokens = initial_sampled;
    for slot_idx in 0..users {
        user_completions[slot_idx].push(active_tokens[slot_idx]);
    }
    let mut batch_positions: Vec<usize> = (0..users)
        .map(|slot_idx| pipe.get_seq_pos(slot_idx as u64))
        .collect();
    let slot_indices: Vec<usize> = (0..users).collect();

    for _step in 1..gen_tokens {
        let sampled =
            pipe.step_continuous_batch(&active_tokens, &batch_positions, &slot_indices)?;
        for slot_idx in 0..users {
            active_tokens[slot_idx] = sampled[slot_idx];
            user_completions[slot_idx].push(sampled[slot_idx]);
            batch_positions[slot_idx] += 1;
        }
    }
    let t_total = t_start.elapsed();
    let t_decode = t_decode_start.elapsed();
    let total_gen_tokens = users * gen_tokens;

    println!("============================================================");
    println!("Oxide-Engine Multi-User (N={}) Results:", users);
    println!(
        "Total Wall-Clock Latency: {:.2} ms",
        t_total.as_secs_f64() * 1000.0
    );
    println!("Total Prompt Tokens: {}", total_prompt_tokens);
    println!("Total Completion Tokens: {}", total_gen_tokens);
    println!(
        "Prefill Time: {:.2} ms ({:.2} tok/s aggregate)",
        t_prefill.as_secs_f64() * 1000.0,
        total_prompt_tokens as f64 / t_prefill.as_secs_f64()
    );
    println!(
        "Decode Time: {:.2} ms ({:.2} tok/s aggregate)",
        t_decode.as_secs_f64() * 1000.0,
        total_gen_tokens as f64 / t_decode.as_secs_f64()
    );
    println!(
        "Per-User Effective Latency: {:.2} ms",
        t_total.as_secs_f64() * 1000.0 / users as f64
    );
    for i in 0..users {
        let decoded = tokenizer.decode(&user_completions[i]);
        let preview: String = decoded.chars().take(100).collect();
        println!(
            "User {} Output (first 100 chars): {:?}",
            i + 1,
            preview
        );
    }
    println!("============================================================");

    Ok(())
}
