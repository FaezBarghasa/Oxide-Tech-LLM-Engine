use memmap2::Mmap;
use oxide_core::StepCommand;
use oxide_engine::SpecializedPipeline;
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;

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
        1,
        None,
    )?;
    pipe.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    pipe.set_seq_pos(0, 0);

    let prompt = "Count from 1 to 10: 1, 2, 3, 4, 5,";
    let tokens = tokenizer.encode(prompt);
    println!("Prefilling prompt with {} tokens...", tokens.len());
    pipe.prefill(0, &tokens, 0)?;

    let initial_pos = pipe.get_seq_pos(0);
    println!("Initial sequence position after prefill: {}", initial_pos);
    assert_eq!(initial_pos, tokens.len());

    // Generate 12 tokens
    let mut cur_token = *tokens.last().unwrap();
    let mut generated_text = String::new();
    println!("Decoding 12 tokens prior to H2O pruning...");
    for _ in 0..12 {
        let cmd = StepCommand::new(0, cur_token, 0, false);
        let completion = pipe.step(&cmd)?;
        cur_token = completion.sampled_token;
        let text = tokenizer.decode_token(cur_token);
        generated_text.push_str(&text);
        print!("{}", text);
    }
    println!();

    let pre_prune_pos = pipe.get_seq_pos(0);
    println!("Sequence position before H2O pruning: {}", pre_prune_pos);
    assert_eq!(pre_prune_pos, initial_pos + 12);

    // Apply H2O & StreamingLLM pruning:
    // Retain 10 tokens total budget: 2 sink tokens + 4 heavy-hitters + 4 rolling tokens
    let budget = 10;
    let sink = 2;
    let rolling = 4;
    println!(
        "\nPruning slot 0 KV cache using H2O: budget={}, sink={}, rolling={}...",
        budget, sink, rolling
    );
    let new_pos = pipe.prune_slot_kv_h2o(0, budget, sink, rolling)?;
    println!("Sequence position after H2O pruning: {}", new_pos);
    assert_eq!(new_pos, budget);
    assert_eq!(pipe.get_seq_pos(0), budget);

    // Continue decoding 10 tokens on top of pruned/compacted KV cache
    println!("\nDecoding 10 tokens post-H2O pruning on compacted physical KV cache...");
    let mut post_text = String::new();
    for _ in 0..10 {
        let cmd = StepCommand::new(0, cur_token, 0, false);
        let completion = pipe.step(&cmd)?;
        cur_token = completion.sampled_token;
        let text = tokenizer.decode_token(cur_token);
        post_text.push_str(&text);
        print!("{}", text);
    }
    println!();

    println!("\nPre-prune text:  {:?}", generated_text.trim());
    println!("Post-prune text: {:?}", post_text.trim());
    assert!(!post_text.is_empty(), "Post-prune output was empty");
    println!("\n[SUCCESS] Physical H2O & StreamingLLM KV cache pruning verified on RTX 4060 dGPU!");

    Ok(())
}
