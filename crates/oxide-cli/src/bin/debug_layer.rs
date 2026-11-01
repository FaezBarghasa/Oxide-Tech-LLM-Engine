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

    let prompt = "2+2=";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt: {:?}, tokens: {:?}", prompt, tokens);

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
            t.extract_row(17, &mut emb);
            println!("CPU token 17 embedding (first 5): {:?}", &emb[..5]);
        }
    }
    let cmd = oxide_core::StepCommand::new(1, tokens[0], 0, false);
    let res_cpu = pipe_cpu.step(&cmd)?;
    println!("CPU step 0: tok {} -> sampled {}", tokens[0], res_cpu.sampled_token);
    if let SpecializedPipeline::Llama3Dense { ref scratch, .. } = pipe_cpu {
        println!("CPU hidden after all layers (first 5): {:?}", &scratch.hidden[..5]);
        let mut top: Vec<(usize, f32)> = scratch.logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("CPU top 5 logits: {:?}", &top[..5]);
    }

    // CUDA step run
    println!("\n=== CUDA STEP RUN ===");
    let mut pipe_cuda = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        1,
        None,
    )?;
    pipe_cuda.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    if let SpecializedPipeline::Llama3Cuda(ref eng) = pipe_cuda {
        let mut host_emb = vec![0.0f32; eng.backend.hidden_dim];
        if let Some(ref lookup) = eng.backend.embedding_lookup {
            lookup(17, &mut host_emb);
            println!("CUDA host embedding_lookup for 17 (first 5): {:?}", &host_emb[..5]);
        }
    }
    let res_cuda = pipe_cuda.step(&cmd)?;
    println!("CUDA step 0: tok {} -> sampled {}", tokens[0], res_cuda.sampled_token);
    if let SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe_cuda {
        let mut host_hid = vec![0.0f32; eng.backend.hidden_dim];
        if let (Some(d_hid), Some(stream)) = (&eng.backend.d_hidden, &eng.backend.stream) {
            let _ = d_hid.copy_to_host_async(&mut host_hid, stream.raw());
            let _ = stream.synchronize();
            println!("CUDA hidden after all layers (first 5): {:?}", &host_hid[..5]);
        }
        let mut host_logits = vec![0.0f32; eng.backend.vocab_size];
        if let (Some(d_logits), Some(stream)) = (&eng.backend.d_logits, &eng.backend.stream) {
            let _ = d_logits.copy_to_host_async(&mut host_logits, stream.raw());
            let _ = stream.synchronize();
            let mut top: Vec<(usize, f32)> = host_logits.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("CUDA top 5 logits: {:?}", &top[..5]);
        }
    }

    // CUDA chunk prefill run
    println!("\n=== CUDA CHUNK PREFILL RUN ===");
    let mut pipe_chunk = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        1,
        None,
    )?;
    pipe_chunk.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    if let SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe_chunk {
        let res = eng.backend.dispatch_prefill_chunk(&tokens, true)?;
        println!("CUDA chunk prefill -> sampled {:?}", res.map(|t| (t, tokenizer.decode_token(t))));
    }

    Ok(())
}
