use memmap2::Mmap;
use oxide_backend_cuda::driver;
use oxide_models::GgufFile;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();
    let model_path = PathBuf::from("/home/jrad/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf");
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let _gguf = GgufFile::parse(&mmap)?;

    let mut pipe = oxide_engine::SpecializedPipeline::from_model_or_path(
        &model_path.to_string_lossy(),
        "cuda",
        None,
        1,
        None,
    )?;

    if let oxide_engine::SpecializedPipeline::Llama3Cuda(ref mut eng) = pipe {
        let stream_obj = eng.backend.stream.as_ref().unwrap();
        let stream = stream_obj.raw();
        let h = eng.backend.hidden_dim;
        let layer0 = &eng.backend.layers[0];
        let layer4 = &eng.backend.layers[4]; // Q4_K W2
        let d_norm_ptr = eng.backend.d_scratch_norm.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_q_ptr = eng.backend.d_scratch_q.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_k_ptr = eng.backend.d_scratch_k.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_v_ptr = eng.backend.d_scratch_v.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_out_ptr = eng.backend.d_scratch_attn_out.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_gate_ptr = eng.backend.d_scratch_gate.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_up_ptr = eng.backend.d_scratch_up.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_hid_ptr = eng.backend.d_hidden.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_logits_ptr = eng.backend.d_logits.as_ref().unwrap().as_typed_ptr::<f32>();
        let d_q8_1_ptr = eng.backend.d_scratch_q8_1_a.as_ref().unwrap().as_raw_ptr();
        let d_sp_ptr = eng.backend.d_seq_pos.as_ref().unwrap().as_typed_ptr::<i32>();
        let v_cache_ptr = eng.backend.kv_v[0].as_raw_ptr();
        let k_cache_ptr = eng.backend.kv_k[0].as_raw_ptr();
        let d_lm_ptr = eng.backend.d_lm_head.as_ref().unwrap().as_raw_ptr();
        let d_sampled_ptr = eng.backend.d_sampled_token.as_ref().unwrap().as_raw_ptr() as *mut u32;
        let d_val_ptr = unsafe { d_sampled_ptr.add(1) as *mut f32 };
        let vocab_size = eng.backend.vocab_size;

        unsafe {
            let _ = stream_obj.synchronize();
            let iters = 100;

            // 1. RMSNorm
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_fused_rmsnorm_q8_1(
                    d_norm_ptr, d_q8_1_ptr, d_hid_ptr, layer0.d_attn_norm.as_typed_ptr::<f32>(),
                    1, h as i32, 1e-6, 0.0, stream,
                );
            }
            let _ = stream_obj.synchronize();
            let dt_rmsnorm = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 2. QKV
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_mmvq_qkv_mixed_ptr(
                    d_q_ptr, d_k_ptr, d_v_ptr, v_cache_ptr, d_sp_ptr,
                    layer0.d_wq.as_raw_ptr(), layer0.d_wk.as_raw_ptr(), layer0.d_wv.as_raw_ptr(),
                    d_q8_1_ptr, layer0.q_m as i32, layer0.kv_m as i32, layer0.q_k as i32, layer0.v_quant as i32, stream,
                );
            }
            let _ = stream_obj.synchronize();
            let dt_qkv = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 3. RoPE
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_fused_qknorm_rope_batch_ptr(
                    d_q_ptr, d_k_ptr, std::ptr::null(), std::ptr::null(), k_cache_ptr, d_sp_ptr,
                    32, 8, 128, 1e-6, 10000.0, 1, 1.0, 4096, stream,
                );
            }
            let _ = stream_obj.synchronize();
            let dt_rope = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 4. Flash Decode
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_flash_decode_ptr(
                    d_out_ptr, d_q8_1_ptr, d_q_ptr, k_cache_ptr, v_cache_ptr,
                    32, 8, 128, d_sp_ptr, 0, 1.0 / (128.0f32).sqrt(), 1, stream,
                );
            }
            let _ = stream_obj.synchronize();
            let dt_flash = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 5. Quantize Q8_1
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_quantize_q8_1(d_out_ptr, d_q8_1_ptr, 4096, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_quant_q8_1 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 6. WO (Q4_K fused add, m=4096, k=4096)
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, layer0.d_wo.as_raw_ptr(), d_q8_1_ptr, layer0.o_m as i32, layer0.o_k as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_wo = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 7. Gate + Up (Q4_K, inter_m=12288, k=4096)
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_gemv_gate_up_q4_k(d_gate_ptr, std::ptr::null_mut(), layer0.d_w1.as_raw_ptr(), layer0.d_w3.as_raw_ptr(), d_q8_1_ptr, layer0.inter_m as i32, h as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_gate_up = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_gemv_q4_k(d_gate_ptr, layer0.d_w1.as_raw_ptr(), d_q8_1_ptr, layer0.inter_m as i32, h as i32, stream);
                driver::launch_cuda_gemv_q4_k(d_up_ptr, layer0.d_w3.as_raw_ptr(), d_q8_1_ptr, layer0.inter_m as i32, h as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_two_gemv = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("Gate+Up fused vs two gemv: fused={:.2}us, two_gemv={:.2}us", dt_gate_up, dt_two_gemv);

            // 8. SwiGLU
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_fused_swiglu_q8_1_sparse(d_q8_1_ptr, std::ptr::null_mut(), d_gate_ptr, std::ptr::null(), layer0.inter_m as i32, 0.0, 0, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_swiglu = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 9. W2 Q6_K (m=4096, k=12288)
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_gemv_q6_k_fused_add(d_hid_ptr, layer0.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, layer0.inter_m as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_w2_q6 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 10. W2 Q4_K (m=4096, k=12288)
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, layer4.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, layer4.inter_m as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_w2_q4 = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            // 11. LM Head Q6_K (m=151936, k=4096)
            driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, eng.backend.d_output_norm.as_ref().unwrap().as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream);
            let _ = stream_obj.synchronize();
            let t0 = Instant::now();
            for _ in 0..20 {
                driver::launch_cuda_gemv_q6_k(d_logits_ptr, d_lm_ptr, d_q8_1_ptr, vocab_size as i32, h as i32, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_lm_head = t0.elapsed().as_secs_f64() * 1e6 / 20.0;

            // 12. Argmax
            let t0 = Instant::now();
            for _ in 0..iters {
                driver::launch_cuda_argmax(d_logits_ptr, vocab_size as i32, d_sampled_ptr, d_val_ptr, 1, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_argmax = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            println!("==================================================");
            println!("  COMPLETE DECODE BREAKDOWN (RTX 4060)");
            println!("==================================================");
            println!("RMSNorm (x2 per layer)          : {:>8.2} us", dt_rmsnorm * 2.0);
            println!("QKV GEMV                        : {:>8.2} us", dt_qkv);
            println!("RoPE + QKNorm                   : {:>8.2} us", dt_rope);
            println!("Flash Decode Attention          : {:>8.2} us", dt_flash);
            println!("Quantize Q8_1 (for WO)          : {:>8.2} us", dt_quant_q8_1);
            println!("WO Q4_K Fused Add               : {:>8.2} us", dt_wo);
            println!("Gate+Up Q4_K Fused              : {:>8.2} us", dt_gate_up);
            println!("SwiGLU Q8_1 Sparse              : {:>8.2} us", dt_swiglu);
            println!("W2 Q6_K Fused Add (18 layers)   : {:>8.2} us", dt_w2_q6);
            println!("W2 Q4_K Fused Add (18 layers)   : {:>8.2} us", dt_w2_q4);
            println!("LM Head Q6_K                    : {:>8.2} us ({:.2} ms)", dt_lm_head, dt_lm_head / 1000.0);
            println!("Argmax F32                      : {:>8.2} us", dt_argmax);
            println!("--------------------------------------------------");
            let per_layer_q6 = dt_rmsnorm * 2.0 + dt_qkv + dt_rope + dt_flash + dt_quant_q8_1 + dt_wo + dt_gate_up + dt_swiglu + dt_w2_q6;
            let per_layer_q4 = dt_rmsnorm * 2.0 + dt_qkv + dt_rope + dt_flash + dt_quant_q8_1 + dt_wo + dt_gate_up + dt_swiglu + dt_w2_q4;
            let total_layers = per_layer_q6 * 18.0 + per_layer_q4 * 18.0;
            let grand_total = total_layers + dt_rmsnorm + dt_lm_head + dt_argmax;
            println!("18 Q6_K layers total            : {:>8.2} ms ({:.2} us/layer)", per_layer_q6 * 18.0 / 1000.0, per_layer_q6);
            println!("18 Q4_K layers total            : {:>8.2} ms ({:.2} us/layer)", per_layer_q4 * 18.0 / 1000.0, per_layer_q4);
            println!("All 36 layers sum               : {:>8.2} ms", total_layers / 1000.0);
            println!("Grand Total Single-Token Decode : {:>8.2} ms ({:.2} tok/s)", grand_total / 1000.0, 1000.0 / (grand_total / 1000.0));
            println!("==================================================");

            // Now measure actual sequential stream execution of all 36 layers + LM head!
            let _ = stream_obj.synchronize();
            let t_real0 = Instant::now();
            let real_iters = 20;
            for _ in 0..real_iters {
                for l in 0..36 {
                    let lyr = &eng.backend.layers[l];
                    // 1. RMSNorm
                    driver::launch_cuda_fused_rmsnorm_q8_1(
                        d_norm_ptr, d_q8_1_ptr, d_hid_ptr, lyr.d_attn_norm.as_typed_ptr::<f32>(),
                        1, h as i32, 1e-6, 0.0, stream,
                    );
                    // 2. QKV
                    driver::launch_cuda_mmvq_qkv_mixed_ptr(
                        d_q_ptr, d_k_ptr, d_v_ptr, eng.backend.kv_v[l].as_raw_ptr(), d_sp_ptr,
                        lyr.d_wq.as_raw_ptr(), lyr.d_wk.as_raw_ptr(), lyr.d_wv.as_raw_ptr(),
                        d_q8_1_ptr, lyr.q_m as i32, lyr.kv_m as i32, lyr.q_k as i32, lyr.v_quant as i32, stream,
                    );
                    // 3. RoPE
                    driver::launch_cuda_fused_qknorm_rope_batch_ptr(
                        d_q_ptr, d_k_ptr, std::ptr::null(), std::ptr::null(), eng.backend.kv_k[l].as_raw_ptr(), d_sp_ptr,
                        32, 8, 128, 1e-6, 10000.0, 1, 1.0, 4096, stream,
                    );
                    // 4. Flash Decode
                    driver::launch_cuda_flash_decode_ptr(
                        d_out_ptr, d_q8_1_ptr, d_q_ptr, eng.backend.kv_k[l].as_raw_ptr(), eng.backend.kv_v[l].as_raw_ptr(),
                        32, 8, 128, d_sp_ptr, 0, 1.0 / (128.0f32).sqrt(), 1, stream,
                    );
                    // 5. WO
                    driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, lyr.d_wo.as_raw_ptr(), d_q8_1_ptr, lyr.o_m as i32, lyr.o_k as i32, stream);
                    // 6. FFN Norm
                    driver::launch_cuda_fused_rmsnorm_q8_1(
                        d_norm_ptr, d_q8_1_ptr, d_hid_ptr, lyr.d_ffn_norm.as_typed_ptr::<f32>(),
                        1, h as i32, 1e-6, 0.0, stream,
                    );
                    // 7. Gate + Up
                    driver::launch_cuda_gemv_gate_up_q4_k(d_gate_ptr, std::ptr::null_mut(), lyr.d_w1.as_raw_ptr(), lyr.d_w3.as_raw_ptr(), d_q8_1_ptr, lyr.inter_m as i32, h as i32, stream);
                    // 8. SwiGLU
                    driver::launch_cuda_fused_swiglu_q8_1_sparse(d_q8_1_ptr, std::ptr::null_mut(), d_gate_ptr, std::ptr::null(), lyr.inter_m as i32, 0.0, 0, stream);
                    // 9. W2
                    if lyr.down_quant == 2 {
                        driver::launch_cuda_gemv_q6_k_fused_add(d_hid_ptr, lyr.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, lyr.inter_m as i32, stream);
                    } else {
                        driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, lyr.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, lyr.inter_m as i32, stream);
                    }
                }
                // Final Norm + LM Head + Argmax
                driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, eng.backend.d_output_norm.as_ref().unwrap().as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream);
                driver::launch_cuda_gemv_q6_k(d_logits_ptr, d_lm_ptr, d_q8_1_ptr, vocab_size as i32, h as i32, stream);
                driver::launch_cuda_argmax(d_logits_ptr, vocab_size as i32, d_sampled_ptr, d_val_ptr, 1, stream);
            }
            let _ = stream_obj.synchronize();
            let dt_real_stream = t_real0.elapsed().as_secs_f64() * 1000.0 / real_iters as f64;
            // Measure each layer individually
            let mut layer_times = Vec::new();
            for l in 0..36 {
                let lyr = &eng.backend.layers[l];
                let _ = stream_obj.synchronize();
                let t_l = Instant::now();
                for _ in 0..10 {
                    driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, lyr.d_attn_norm.as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream);
                    driver::launch_cuda_mmvq_qkv_mixed_ptr(d_q_ptr, d_k_ptr, d_v_ptr, eng.backend.kv_v[l].as_raw_ptr(), d_sp_ptr, lyr.d_wq.as_raw_ptr(), lyr.d_wk.as_raw_ptr(), lyr.d_wv.as_raw_ptr(), d_q8_1_ptr, lyr.q_m as i32, lyr.kv_m as i32, lyr.q_k as i32, lyr.v_quant as i32, stream);
                    driver::launch_cuda_fused_qknorm_rope_batch_ptr(d_q_ptr, d_k_ptr, std::ptr::null(), std::ptr::null(), eng.backend.kv_k[l].as_raw_ptr(), d_sp_ptr, 32, 8, 128, 1e-6, 10000.0, 1, 1.0, 4096, stream);
                    driver::launch_cuda_flash_decode_ptr(d_out_ptr, d_q8_1_ptr, d_q_ptr, eng.backend.kv_k[l].as_raw_ptr(), eng.backend.kv_v[l].as_raw_ptr(), 32, 8, 128, d_sp_ptr, 0, 1.0 / (128.0f32).sqrt(), 1, stream);
                    driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, lyr.d_wo.as_raw_ptr(), d_q8_1_ptr, lyr.o_m as i32, lyr.o_k as i32, stream);
                    driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, lyr.d_ffn_norm.as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream);
                    driver::launch_cuda_gemv_gate_up_q4_k(d_gate_ptr, std::ptr::null_mut(), lyr.d_w1.as_raw_ptr(), lyr.d_w3.as_raw_ptr(), d_q8_1_ptr, lyr.inter_m as i32, h as i32, stream);
                    driver::launch_cuda_fused_swiglu_q8_1_sparse(d_q8_1_ptr, std::ptr::null_mut(), d_gate_ptr, std::ptr::null(), lyr.inter_m as i32, 0.0, 0, stream);
                    if lyr.down_quant == 2 {
                        driver::launch_cuda_gemv_q6_k_fused_add(d_hid_ptr, lyr.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, lyr.inter_m as i32, stream);
                    } else {
                        driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, lyr.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, lyr.inter_m as i32, stream);
                    }
                }
                let _ = stream_obj.synchronize();
                layer_times.push(t_l.elapsed().as_secs_f64() * 1000.0 / 10.0);
            }
            let mut q6_count = 0;
            let mut q4_count = 0;
            let mut wv_q6_count = 0;
            for l in 0..36 {
                let lyr = &eng.backend.layers[l];
                if lyr.down_quant == 2 { q6_count += 1; } else { q4_count += 1; }
                if lyr.v_quant == 2 { wv_q6_count += 1; }
            }
            let sum_layer_times: f64 = layer_times.iter().sum();
            println!("Layer breakdown: down_quant Q6_K={}, Q4_K={}, v_quant Q6_K={}", q6_count, q4_count, wv_q6_count);
            println!("Sum of individually measured 36 layers: {:>8.2} ms", sum_layer_times);

            // Cumulative step-by-step measurement of Layer 4
            let _ = stream_obj.synchronize();
            let mut cum_times = Vec::new();
            for step in 1..=9 {
                let _ = stream_obj.synchronize();
                let t_cum = Instant::now();
                for _ in 0..iters {
                    if step >= 1 { driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, layer4.d_attn_norm.as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream); }
                    if step >= 2 { driver::launch_cuda_mmvq_qkv_mixed_ptr(d_q_ptr, d_k_ptr, d_v_ptr, eng.backend.kv_v[4].as_raw_ptr(), d_sp_ptr, layer4.d_wq.as_raw_ptr(), layer4.d_wk.as_raw_ptr(), layer4.d_wv.as_raw_ptr(), d_q8_1_ptr, layer4.q_m as i32, layer4.kv_m as i32, layer4.q_k as i32, layer4.v_quant as i32, stream); }
                    if step >= 3 { driver::launch_cuda_fused_qknorm_rope_batch_ptr(d_q_ptr, d_k_ptr, std::ptr::null(), std::ptr::null(), eng.backend.kv_k[4].as_raw_ptr(), d_sp_ptr, 32, 8, 128, 1e-6, 10000.0, 1, 1.0, 4096, stream); }
                    if step >= 4 { driver::launch_cuda_flash_decode_ptr(d_out_ptr, d_q8_1_ptr, d_q_ptr, eng.backend.kv_k[4].as_raw_ptr(), eng.backend.kv_v[4].as_raw_ptr(), 32, 8, 128, d_sp_ptr, 0, 1.0 / (128.0f32).sqrt(), 1, stream); }
                    if step >= 5 { driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, layer4.d_wo.as_raw_ptr(), d_q8_1_ptr, layer4.o_m as i32, layer4.o_k as i32, stream); }
                    if step >= 6 { driver::launch_cuda_fused_rmsnorm_q8_1(d_norm_ptr, d_q8_1_ptr, d_hid_ptr, layer4.d_ffn_norm.as_typed_ptr::<f32>(), 1, h as i32, 1e-6, 0.0, stream); }
                    if step >= 7 { driver::launch_cuda_gemv_gate_up_q4_k(d_gate_ptr, std::ptr::null_mut(), layer4.d_w1.as_raw_ptr(), layer4.d_w3.as_raw_ptr(), d_q8_1_ptr, layer4.inter_m as i32, h as i32, stream); }
                    if step >= 8 { driver::launch_cuda_fused_swiglu_q8_1_sparse(d_q8_1_ptr, std::ptr::null_mut(), d_gate_ptr, std::ptr::null(), layer4.inter_m as i32, 0.0, 0, stream); }
                    if step >= 9 { driver::launch_cuda_gemv_q4_k_fused_add(d_hid_ptr, layer4.d_w2.as_raw_ptr(), d_q8_1_ptr, h as i32, layer4.inter_m as i32, stream); }
                }
                let _ = stream_obj.synchronize();
                cum_times.push(t_cum.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
            println!("Layer 4 cumulative steps:");
            let names = ["RMSNorm1", "+QKV", "+RoPE", "+FlashDec", "+WO", "+RMSNorm2", "+GateUp", "+SwiGLU", "+W2"];
            let mut prev = 0.0;
            for s in 0..9 {
                println!("  Step {} {:<12}: {:>6.2} us (delta: {:>6.2} us)", s + 1, names[s], cum_times[s], cum_times[s] - prev);
                prev = cum_times[s];
            }
            for l in 0..36 {
                println!("  Layer {:2}: {:>6.2} us (down_quant={}, v_quant={})", l, layer_times[l] * 1000.0, eng.backend.layers[l].down_quant, eng.backend.layers[l].v_quant);
            }
            println!("Actual 36 layers sequential stream dt: {:>8.2} ms ({:.2} tok/s)", dt_real_stream, 1000.0 / dt_real_stream);
            println!("==================================================");
        }
    }

    println!("\nNow running pipe.step() 10 decode steps to measure actual end-to-end decode step:");
    pipe.configure_sampling(0.0, 0.95, 40, 0.05, 1.1, 64, 42, true);
    let mut tok = 151644u32;
    for i in 0..10 {
        let t_step = Instant::now();
        let cmd = oxide_core::StepCommand::new(1, tok, 0, false);
        let res = pipe.step(&cmd)?;
        let dt = t_step.elapsed();
        tok = res.sampled_token;
        println!("Step {}: {:.2} ms ({:.2} tok/s), tok: {}", i, dt.as_secs_f64() * 1000.0, 1.0 / dt.as_secs_f64(), tok);
    }

    Ok(())
}
