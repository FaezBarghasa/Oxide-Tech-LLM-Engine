use memmap2::Mmap;
use oxide_engine::SpecializedPipeline;
use oxide_models::{GgufFile, GgufTokenizer};
use std::fs::File;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = PathBuf::from("/home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    let prompt = "<|im_start|>user\n2+2=<|im_end|>\n<|im_start|>assistant\n";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt tokens (len {}): {:?}", tokens.len(), tokens);

    // CPU run
    println!("=== CPU RUN ===");
    let mut pipe_cpu = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cpu",
        None,
        1,
        None,
    )?;
    pipe_cpu.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    if let SpecializedPipeline::Llama3Dense { ref model, .. } = pipe_cpu {
        let mut emb = vec![0.0f32; model.config.hidden_dim];
        if let Some(ref t) = model.token_embd_tensor {
            println!(
                "token_embd_tensor quant_type={}, shape={}x{}, bytes_len={}",
                t.quant_type(),
                t.rows(),
                t.cols(),
                t.as_raw_bytes().len()
            );
            t.extract_row(17, &mut emb);
            println!("CPU token 17 embedding (first 5): {:?}", &emb[..5]);
        }
    }
    let cmd = oxide_core::StepCommand::new(1, tokens[0], 0, false);
    let res_cpu = pipe_cpu.step(&cmd)?;
    println!(
        "CPU step 0: tok {} -> sampled {}",
        tokens[0], res_cpu.sampled_token
    );
    if let SpecializedPipeline::Llama3Dense { ref scratch, .. } = pipe_cpu {
        println!(
            "CPU hidden after all layers (first 5): {:?}",
            &scratch.hidden[..5]
        );
        let mut top: Vec<(usize, f32)> = scratch.logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("CPU top 5 logits: {:?}", &top[..5]);
    }

    // Compare CUDA step vs CUDA chunk for prefix lengths 1..=tokens.len()
    for len in 1..=tokens.len() {
        let prefix = &tokens[..len];
        let mut pipe_step = SpecializedPipeline::from_model_or_path(
            &model_path.to_string_lossy(),
            "cuda",
            None,
            1,
            None,
        )?;
        pipe_step.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        let mut step_sampled = 0;
        for (i, &tok) in prefix.iter().enumerate() {
            let is_prefill = i + 1 < prefix.len();
            let cmd = oxide_core::StepCommand::new(1, tok, 0, is_prefill);
            let res = pipe_step.step(&cmd)?;
            step_sampled = res.sampled_token;
        }

        let mut pipe_chunk_test = SpecializedPipeline::from_model_or_path(
            &model_path.to_string_lossy(),
            "cuda",
            None,
            1,
            None,
        )?;
        pipe_chunk_test.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        let mut chunk_sampled = 0;
        if let SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe_chunk_test {
            let res = eng.backend.dispatch_prefill_chunk(prefix, true)?;
            chunk_sampled = res.unwrap_or(0);
        }

        let match_str = if step_sampled == chunk_sampled {
            "MATCH"
        } else {
            "MISMATCH"
        };
        println!(
            "Len {}: step={} ({:?}) vs chunk={} ({:?}) -> {}",
            len,
            step_sampled,
            tokenizer.decode_token(step_sampled),
            chunk_sampled,
            tokenizer.decode_token(chunk_sampled),
            match_str
        );
    }

    Ok(())
}
