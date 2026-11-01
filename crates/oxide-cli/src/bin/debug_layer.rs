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

    let prompt = "Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Now write down the first ten letters of the alphabet in uppercase:";
    let all_tokens = tokenizer.encode(prompt);
    let tokens = all_tokens[..8].to_vec();
    println!("Prompt len 8: tokens: {:?}", tokens);

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
            println!("token_embd_tensor quant_type={}, shape={}x{}, bytes_len={}", t.quant_type(), t.rows(), t.cols(), t.as_raw_bytes().len());
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
        let l0 = &eng.backend.layers[0];
        println!("Layer 0 quant types: q_quant={}, k_quant={}, v_quant={}, o_quant={}, gate_quant={}, up_quant={}, down_quant={}",
            l0.q_quant, l0.k_quant, l0.v_quant, l0.o_quant, l0.gate_quant, l0.up_quant, l0.down_quant);
        println!("Has biases: q_bias={}, k_bias={}, v_bias={}",
            l0.d_q_bias.is_some(), l0.d_k_bias.is_some(), l0.d_v_bias.is_some());
        println!("CUDA token_embd: is_some={}, quant_type={}",
            eng.backend.d_token_embd.is_some(), eng.backend.token_embd_quant_type);
    }
    for (i, &tok) in tokens.iter().enumerate() {
        let is_prefill = i + 1 < tokens.len();
        let cmd = oxide_core::StepCommand::new(1, tok, 0, is_prefill);
        let res = pipe_cuda.step(&cmd)?;
        println!("CUDA step {}: tok {} ({:?}) -> sampled {} ({:?})", i, tok, tokenizer.decode_token(tok), res.sampled_token, tokenizer.decode_token(res.sampled_token));
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
        println!("CUDA chunk prefill on {:?} -> sampled {:?}", tokens, res.map(|t| (t, tokenizer.decode_token(t))));
        let mut host_logits = vec![0.0f32; eng.backend.vocab_size];
        if let (Some(d_logits), Some(stream)) = (&eng.backend.d_logits, &eng.backend.stream) {
            let _ = d_logits.copy_to_host_async(&mut host_logits, stream.raw());
            let _ = stream.synchronize();
            let mut top: Vec<(usize, f32)> = host_logits.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("CUDA chunk {:?} top 5 logits: {:?}", tokens, &top[..5]);
        }
    }

    Ok(())
}
