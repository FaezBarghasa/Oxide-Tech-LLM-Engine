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

    let prompt =
        "<|im_start|>user\nWhat is 2+2? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt tokens (len {}): {:?}", tokens.len(), tokens);

    // CPU run layer by layer
    let mut pipe_cpu = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cpu",
        None,
        1,
        None,
    )?;
    pipe_cpu.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);

    // Let's inspect the embeddings on CPU for tokens
    if let SpecializedPipeline::Llama3Dense { ref model, .. } = pipe_cpu {
        let mut emb0 = vec![0.0f32; model.config.hidden_dim];
        if let Some(ref t) = model.token_embd_tensor {
            t.extract_row(tokens[0] as usize, &mut emb0);
            println!("CPU tok[0] emb first 4: {:?}", &emb0[..4]);
            let mut emb_last = vec![0.0f32; model.config.hidden_dim];
            t.extract_row(*tokens.last().unwrap() as usize, &mut emb_last);
            println!("CPU tok[last] emb first 4: {:?}", &emb_last[..4]);
        }
    }

    // CUDA run
    let mut pipe_cuda = SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        1,
        None,
    )?;
    pipe_cuda.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    if let SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe_cuda {
        let b = &mut eng.backend;
        let s = b.stream.as_ref().unwrap().raw();
        // Check embedding dequant on CUDA
        let h = b.hidden_dim;
        let tok_idx = tokens[0] as usize;
        let mut cuda_emb0 = vec![0.0f32; h];
        if let Some(ref d_emb) = b.d_token_embd {
            let d_tmp = oxide_backend_cuda::CudaDeviceBuffer::allocate(h * 4).unwrap();
            unsafe {
                oxide_backend_cuda::driver::launch_cuda_dequantize_token_embd(
                    d_tmp.as_typed_ptr::<f32>(),
                    d_emb.as_raw_ptr(),
                    tok_idx as i32,
                    h as i32,
                    1.0,
                    b.token_embd_quant_type as i32,
                    s,
                );
            }
            d_tmp.copy_to_host_async(&mut cuda_emb0, s).unwrap();
            b.stream.as_ref().unwrap().synchronize().unwrap();
            println!("CUDA tok[0] emb first 4: {:?}", &cuda_emb0[..4]);
        }
    }

    Ok(())
}
