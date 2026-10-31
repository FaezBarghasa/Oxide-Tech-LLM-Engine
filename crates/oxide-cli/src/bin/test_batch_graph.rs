use memmap2::Mmap;
use oxide_engine::SpecializedPipeline;
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = PathBuf::from("/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf");
    println!("Loading physical model from {:?}...", model_path);

    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    let mut pipe = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        16,
        None,
    )?;
    pipe.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);

    let base_prompts = [
        "Count from 1 to 5: 1, 2, 3,",
        "The capital of France is",
        "Write the color of the clear sky: The sky is",
        "The speed of light in vacuum is approximately",
        "In computer science, a binary tree is a",
        "The chemical symbol for water is",
        "The largest ocean on Earth is the",
        "Artificial intelligence is the field of",
    ];

    let mut prompts = Vec::new();
    for p in base_prompts {
        prompts.push(p);
    }
    for p in base_prompts {
        prompts.push(p);
    }

    let num_streams = prompts.len();
    let mut slot_tokens = Vec::new();
    let mut slot_positions = Vec::new();
    let slot_indices: Vec<usize> = (0..num_streams).collect();

    println!(
        "\n[1/3] Prefilling {} independent streams into physical KV cache...",
        num_streams
    );
    for (slot, &prompt) in prompts.iter().enumerate() {
        let tokens = tokenizer.encode(prompt);
        let prefill_slice = if tokens.len() > 1 {
            &tokens[..tokens.len() - 1]
        } else {
            &[]
        };
        if !prefill_slice.is_empty() {
            pipe.prefill(slot as u64, prefill_slice, slot as u16)?;
        }
        let last_token = *tokens.last().unwrap();
        slot_tokens.push(last_token);
        slot_positions.push(tokens.len().saturating_sub(1));
    }

    println!("\n[2/3] Executing Step 1 (Capture & Warmup of Bucket 16 CUDA Graph)...");
    let t0 = Instant::now();
    let step1_out = pipe.step_continuous_batch(&slot_tokens, &slot_positions, &slot_indices)?;
    let dt_step1 = t0.elapsed();
    println!("  Step 1 finished in {:?}:", dt_step1);

    // Update tokens & positions for decoding
    slot_tokens[..slot_indices.len()].copy_from_slice(&step1_out[..slot_indices.len()]);
    for p in slot_positions.iter_mut() {
        *p += 1;
    }

    println!("\n[3/3] Executing Steps 2..15 via Multi-Bucket CUDA Graph Replay (Bucket 16)...");
    let mut generated_texts = vec![String::new(); num_streams];
    for (i, &tok) in step1_out.iter().enumerate() {
        generated_texts[i].push_str(&tokenizer.decode_token(tok));
    }

    let mut replay_times = Vec::new();
    for _step in 2..=15 {
        let t_start = Instant::now();
        let out_tokens =
            pipe.step_continuous_batch(&slot_tokens, &slot_positions, &slot_indices)?;
        let t_step = t_start.elapsed();
        replay_times.push(t_step);

        for (i, &tok) in out_tokens.iter().enumerate() {
            generated_texts[i].push_str(&tokenizer.decode_token(tok));
            slot_tokens[i] = tok;
            slot_positions[i] += 1;
        }
    }

    let avg_replay = replay_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / replay_times.len() as f64;
    let min_replay = replay_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let throughput_tokens_per_sec = (slot_indices.len() as f64) / (avg_replay / 1000.0);

    println!("\n========================================================");
    println!("        CONTINUOUS BATCH CUDA GRAPH PERFORMANCE         ");
    println!("========================================================");
    println!("  Concurrent Streams : {}", slot_indices.len());
    println!("  Graph Bucket Used  : 16");
    println!("  Avg Step Latency   : {:.2} ms", avg_replay);
    println!("  Min Step Latency   : {:.2} ms", min_replay);
    println!(
        "  Serving Throughput : {:.2} tok/s",
        throughput_tokens_per_sec
    );
    println!("========================================================");
    println!("\nGenerated Text Samples per Stream:");
    for (i, text) in generated_texts.iter().enumerate() {
        assert!(!text.is_empty(), "Stream {} output was empty", i);
        if i < 8 {
            println!("  [Stream {}] Prompt: {:?} -> Output: {:?}", i, prompts[i], text.trim());
        }
    }

    Ok(())
}
