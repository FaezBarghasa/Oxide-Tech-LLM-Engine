use oxide_lab::*;

#[test]
fn test_custom_model_builder_and_forward() {
    let model = CustomModelBuilder::new("research-transformer", 100, 64)
        .add_rmsnorm("norm_0")
        .add_attention("attn_0", 4, 2)
        .add_swiglu_mlp("mlp_0", 128)
        .add_moe("moe_0", 4, 2, 128)
        .add_mamba_ssm("ssm_0", 16)
        .add_latent_attention_mla("mla_0", 32, 4)
        .add_rmsnorm("norm_final")
        .build();

    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 7);

    let mut scratch = CustomModelScratch::new(64, 100);
    let tokens = [1u32, 42, 99];
    let logits = model.forward(&tokens, &mut scratch).unwrap();

    assert_eq!(logits.len(), 100);
    // Logits should be non-empty and non-NaN
    for &l in logits {
        assert!(!l.is_nan());
    }

    // Compile to compute graph DAG
    let graph = model.compile_to_graph();
    assert_eq!(graph.nodes.len(), 7);
}

#[test]
fn test_training_and_optimizer_suite() {
    // 1. Cross-Entropy Loss & Analytical Gradients
    let logits = vec![2.0, 1.0, 0.1, -1.0];
    let (loss, grad) = LossComputer::compute_cross_entropy(&logits, 0, 0.0);
    assert!(loss > 0.0);
    assert_eq!(grad.len(), 4);
    // Target gradient dL/dLogit_0 should be negative (prob - 1 < 0)
    assert!(grad[0] < 0.0);

    // 2. MSE Loss
    let preds = vec![1.0, 2.0, 3.0];
    let targets = vec![1.1, 1.9, 3.2];
    let (mse, grad_mse) = LossComputer::compute_mse(&preds, &targets);
    assert!(mse > 0.0);
    assert_eq!(grad_mse.len(), 3);

    // 3. DPO Loss
    let (dpo_loss, dpo_grad) = LossComputer::compute_dpo(1.2, 0.4, 0.8, 0.7, 0.1);
    assert!(dpo_loss > 0.0);
    assert!(dpo_grad < 0.0);

    // 4. Learning Rate Scheduler
    let scheduler = LrScheduler::new(1e-3, 10, 100);
    let lr_0 = scheduler.get_lr(0);
    let lr_warmup = scheduler.get_lr(10);
    let lr_mid = scheduler.get_lr(50);
    let lr_end = scheduler.get_lr(100);
    assert!(lr_0 < lr_warmup);
    assert_eq!(lr_warmup, 1e-3);
    assert!(lr_mid < lr_warmup);
    assert_eq!(lr_end, scheduler.min_lr);

    // 5. LoRA Fine-Tuner Forward & Backward Step
    let config = TrainingConfig {
        learning_rate: 1e-2,
        warmup_steps: 5,
        total_steps: 50,
        ..Default::default()
    };
    let mut tuner = LoraFineTuner::new(32, 32, 4, 8.0, config);
    let x = vec![0.5f32; 32];
    let mut inter = vec![0.0f32; 4];
    let mut delta = vec![0.0f32; 32];

    tuner.forward(&x, &mut inter, &mut delta);
    // Initially delta is 0 because lora_b is initialized to 0
    assert_eq!(delta[0], 0.0);

    let grad_out = vec![1.0f32; 32];
    tuner.backward(&x, &inter, &grad_out);
    tuner.step();

    // After step, lora_b was updated, so forward will now produce non-zero delta
    tuner.forward(&x, &mut inter, &mut delta);
    assert_ne!(delta[0], 0.0);
}

#[test]
fn test_quant_and_compression_suite() {
    let weights = vec![0.5, -0.3, 1.2, -0.9, 0.1, -0.05, 0.8, -0.4];

    // 1. Q8_0 Quantization
    let q8 = LabQuantizer::quantize(&weights, 2, 4, LabQuantMethod::Q8_0, None);
    assert_eq!(q8.rows, 2);
    assert_eq!(q8.cols, 4);
    assert_eq!(q8.compression_ratio, 4.0);

    // 2. Q4_0 Quantization
    let q4 = LabQuantizer::quantize(&weights, 2, 4, LabQuantMethod::Q4_0, None);
    assert_eq!(q4.compression_ratio, 8.0);

    // 3. Ternary 1.58-bit Quantization
    let ternary = LabQuantizer::quantize(&weights, 2, 4, LabQuantMethod::Ternary1_58Bit, None);
    assert_eq!(ternary.compression_ratio, 16.0);

    // 4. Magnitude Pruning (Unstructured)
    let pruned = LabCompressor::prune_magnitude(&weights, 2, 4, 0.50, false);
    assert!(pruned.sparsity >= 0.40);

    // 5. Magnitude Pruning (2:4 Structured)
    let structured = LabCompressor::prune_magnitude(&weights, 2, 4, 0.50, true);
    assert_eq!(structured.sparsity, 0.50);
    assert!(structured.is_structured_2_4);

    // 6. Truncated SVD Decomposition
    let svd = LabCompressor::svd_decompose(&weights, 2, 4, 1);
    assert_eq!(svd.rank_r, 1);
    assert_eq!(svd.factor_a.len(), 2);
    assert_eq!(svd.factor_b.len(), 4);

    // 7. KV Cache Compression Policy
    let (retained, ratio) = LabCompressor::compress_kv_cache(1000, 4, 256, 0.20);
    assert!(retained < 1000);
    assert!(ratio > 1.0);
}

#[test]
fn test_realtime_debug_and_drift_suite() {
    let tensor = vec![0.1, 0.2, -0.3, 0.4, 0.0, 0.0, 0.5];
    let stats = TensorDebugger::inspect("layer_0_act", &tensor);
    assert_eq!(stats.tensor_name, "layer_0_act");
    assert_eq!(stats.num_elements, 7);
    assert!(!stats.has_nan);
    assert!(!stats.has_inf);
    assert!(stats.sparsity_percentage > 20.0);

    // Safety check passing
    assert!(TensorDebugger::assert_safe("layer_0_act", &tensor).is_ok());

    // Safety check with NaN
    let bad_tensor = vec![0.1, f32::NAN, 0.3];
    assert!(TensorDebugger::assert_safe("bad_act", &bad_tensor).is_err());

    // Drift Detection
    let golden = vec![1.0, 2.0, 3.0, 4.0];
    let candidate = vec![0.99, 2.01, 2.98, 4.02];
    let drift = DriftDetector::compare("layer_attn", &golden, &candidate);
    assert!(drift.cosine_similarity > 0.999);
    assert!(drift.mean_squared_error < 0.001);

    // Perplexity Evaluation
    let seq_logits = vec![vec![5.0, 1.0, 0.0], vec![1.0, 6.0, 0.0]];
    let targets = vec![0, 1];
    let ppl = PerplexityAuditor::evaluate_ppl(&seq_logits, &targets);
    assert!(ppl >= 1.0);
}

#[test]
fn test_multimodal_generation_and_inputs() {
    let engine = MultiModalLabEngine;

    // 1. Text-to-Image Generation
    let img_out = engine
        .generate_image("A futuristic quantum computer in Rust", 64, 64, 5, 7.5)
        .unwrap();
    match img_out {
        MultiModalOutput::Image {
            pixels,
            width,
            height,
            channels,
            format,
        } => {
            assert_eq!(width, 64);
            assert_eq!(height, 64);
            assert_eq!(channels, 3);
            assert_eq!(pixels.len(), 64 * 64 * 3);
            assert_eq!(format, "RGB_F32");
        }
        _ => panic!("Expected Image output"),
    }

    // 2. Text-to-Video Generation
    let vid_out = engine
        .generate_video("Turbulent fluid flow simulation", 32, 32, 4, 24.0, 5)
        .unwrap();
    match vid_out {
        MultiModalOutput::Video {
            frames,
            width,
            height,
            num_frames,
            fps,
        } => {
            assert_eq!(width, 32);
            assert_eq!(height, 32);
            assert_eq!(num_frames, 4);
            assert_eq!(fps, 24.0);
            assert_eq!(frames.len(), 32 * 32 * 3 * 4);
        }
        _ => panic!("Expected Video output"),
    }

    // 3. Neural Speech Synthesis (TTS)
    let speech_out = engine
        .synthesize_speech("Oxide Tech LLM Engine online.", 0, 16000)
        .unwrap();
    match speech_out {
        MultiModalOutput::Audio {
            samples,
            sample_rate,
        } => {
            assert_eq!(sample_rate, 16000);
            assert!(!samples.is_empty());
        }
        _ => panic!("Expected Audio output"),
    }

    // 4. Speech Transcription (ASR)
    let audio_samples = vec![0.1f32; 1600];
    let transcription = engine.transcribe_speech(&audio_samples, 16000).unwrap();
    assert!(transcription.contains("Transcribed speech"));
}
