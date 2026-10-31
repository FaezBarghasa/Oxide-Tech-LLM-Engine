use std::fs::File;
use std::path::PathBuf;
use memmap2::Mmap;
use oxide_models::{GgufFile, GgufTokenizer};
use oxide_engine::SpecializedPipeline;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model_path = PathBuf::from("/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf");
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let gguf = GgufFile::parse(&mmap)?;
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    let prompt = "Count from 1 to 10: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Now write down the first ten letters of the alphabet in uppercase:";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt: {:?}, tokens: {:?}", prompt, tokens);

    for len in [1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 48, 53] {
        let sub = &tokens[..len];
        // Step
        let mut pipe_s = SpecializedPipeline::from_model_or_path(&model_path.to_string_lossy(), "cuda", None, 1, None)?;
        pipe_s.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        let mut s_tok = 0u32;
        for (i, &tok) in sub.iter().enumerate() {
            let is_prefill = i + 1 < sub.len();
            let cmd = oxide_core::StepCommand::new(1, tok, 0, is_prefill);
            let res = pipe_s.step(&cmd)?;
            s_tok = res.sampled_token;
        }
        drop(pipe_s);

        // Chunk
        let mut pipe_c = SpecializedPipeline::from_model_or_path(&model_path.to_string_lossy(), "cuda", None, 1, None)?;
        pipe_c.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
        let mut c_tok = 0u32;
        let mut ret_some = false;
        if let SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe_c {
            match eng.backend.dispatch_prefill_chunk(sub, true) {
                Ok(Some(tok)) => {
                    c_tok = tok;
                    ret_some = true;
                }
                Ok(None) => {
                    println!("len {}: dispatch_prefill_chunk returned None!", len);
                }
                Err(e) => {
                    println!("len {}: dispatch_prefill_chunk returned Err: {:?}", len, e);
                }
            }
        }
        drop(pipe_c);

        println!("len {}: step {} ({:?}), chunk {} ({:?}) [ret_some={}] - match: {}",
            len, s_tok, tokenizer.decode_token(s_tok), c_tok, tokenizer.decode_token(c_tok), ret_some, s_tok == c_tok);
        if s_tok != c_tok {
            break;
        }
    }

    Ok(())
}
