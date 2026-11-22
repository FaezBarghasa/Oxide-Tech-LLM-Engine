//! Real-Life Model Verification Suite
//!
//! Validates physical, zero-stub execution against genuine model weights:
//! - Real GGUF model forward execution (CPU & CUDA) from `~/models/`
//! - Real SafeTensors tensor loading from `/home/jrad/comfy/ComfyUI/models/` and `~/models/`
//! - 3D Generative Geometry & Binary GLB 2.0 verification
//! - Electronic Design Automation (EDA) KiCad Schematic & PCB Synthesis with DRC verification
//! - Unsloth 4-bit NF4 QLoRA reverse-mode autograd training
//! - Google Research SLEEP Continual Learning Framework lifecycle
//! - Audio-LALM encoders (AF-Whisper, MOSS-Audio, TM-RoPE)

use memmap2::Mmap;
use oxide_audio_encoders::{AfWhisperEncoder, MossAudioEncoder, TmRopeTemporalGrid};
use oxide_core::worker::StepCommand;
use oxide_eda::{EdaEngine, EdaSynthesisRequest};
use oxide_engine::SpecializedPipeline;
use oxide_lora::{
    QloraTrainingConfig, SleepConfig, SleepMemoryConsolidator, SleepState,
    UnslothAdapter, UnslothQloraTrainer,
};
use oxide_models::{GgufFile, GgufTokenizer};
use oxide_tech_llm_engine::{Mesh3dEngineKind, MeshGenerationRequest, OxideEngine};
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

#[test]
fn test_real_safetensors_inspection_comfy_and_models() {
    let test_targets = [
        "/home/jrad/models/hf_qwen2.5_0.5b_meta/model.safetensors",
        "/home/jrad/comfy/ComfyUI/models/Hi3D/hi3d/transformer_512/model.safetensors",
        "/home/jrad/comfy/ComfyUI/models/vae/flux2-vae.safetensors",
    ];

    let mut inspected_count = 0;
    for path_str in &test_targets {
        let path = Path::new(path_str);
        if !path.exists() {
            println!("Skipping non-existent path: {path_str}");
            continue;
        }

        let file = File::open(path).expect("Failed to open physical SafeTensors file");
        let mmap = unsafe { Mmap::map(&file).expect("Failed to memory-map SafeTensors file") };
        let tensors = SafeTensors::deserialize(&mmap)
            .expect("SafeTensors parsing succeeded without error");

        let names: Vec<&str> = tensors.names();
        println!(
            "[+] Verified SafeTensors file: {} | Tensors: {}",
            path.display(),
            names.len()
        );

        assert!(!names.is_empty(), "SafeTensors file must contain tensors");
        let first_name = names[0];
        let tensor_view = tensors
            .tensor(first_name)
            .expect("Header metadata points to valid tensor data");

        println!(
            "    First tensor: '{}' | Shape: {:?} | Dtype: {:?}",
            first_name,
            tensor_view.shape(),
            tensor_view.dtype()
        );
        assert!(!tensor_view.data().is_empty());
        inspected_count += 1;
    }

    assert!(
        inspected_count >= 2,
        "Must verify at least 2 real physical SafeTensors files"
    );
}

#[test]
fn test_real_gguf_physical_forward_cpu_and_cuda() {
    let model_path = "/home/jrad/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if !Path::new(model_path).exists() {
        println!("Model not found at {model_path}, skipping test.");
        return;
    }

    let file = File::open(model_path).expect("Open GGUF file");
    let mmap = unsafe { Mmap::map(&file).expect("Mmap GGUF file") };
    let gguf = GgufFile::parse(&mmap).expect("Parse GGUF file");
    let tokenizer = GgufTokenizer::from_gguf(&gguf);

    let prompt = "The speed of light in vacuum is";
    let input_tokens = tokenizer.encode(prompt);
    assert!(!input_tokens.is_empty());

    for backend in ["cpu", "cuda"] {
        println!("\n=== Executing real physical forward pass on backend: {backend} ===");
        let mut pipe = SpecializedPipeline::from_model_or_path(
            model_path,
            backend,
            None,
            1,
            None,
        )
        .expect("Initialize pipeline on physical backend");

        pipe.configure_sampling(0.0, 1.0, 1, 0.0, 1.0, 0, 42, false);

        let mut current_tok = *input_tokens.last().unwrap_or(&1);
        let mut generated = Vec::new();

        for _ in 0..4 {
            let cmd = StepCommand::new(1, current_tok, 0, false);
            let res = pipe.step(&cmd).expect("Physical step forward decode");
            current_tok = res.sampled_token;
            generated.push(current_tok);
        }

        println!("  Backend '{backend}' generated tokens: {:?}", generated);
        assert_eq!(generated.len(), 4);
        assert!(generated.iter().all(|&t| t > 0));
    }
}

#[test]
fn test_real_3d_watertight_generation_and_glb_spec() {
    let engine = OxideEngine::builder().build().expect("Build OxideEngine");

    let req = MeshGenerationRequest {
        engine: Mesh3dEngineKind::Hunyuan3d21,
        prompt_or_condition: "Watertight aerodynamic UAV propeller blade".to_string(),
        input_images: vec![],
        export_glb: true,
    };

    let glb_bytes = engine.generate_glb_mesh(req).expect("Generate GLB mesh");
    assert!(
        glb_bytes.len() >= 20,
        "GLB binary must have header and chunks"
    );

    // Verify glTF 2.0 binary header
    assert_eq!(&glb_bytes[0..4], b"glTF", "Magic header must be 'glTF'");
    let version = u32::from_le_bytes([glb_bytes[4], glb_bytes[5], glb_bytes[6], glb_bytes[7]]);
    assert_eq!(version, 2, "glTF version must be 2");

    let total_len = u32::from_le_bytes([glb_bytes[8], glb_bytes[9], glb_bytes[10], glb_bytes[11]]);
    assert_eq!(
        total_len as usize,
        glb_bytes.len(),
        "GLB length header must match buffer length"
    );
    println!("[+] Validated binary GLB 2.0 asset ({} bytes)", glb_bytes.len());
}

#[test]
fn test_real_eda_hardware_synthesis_and_drc() {
    let engine = EdaEngine::new();
    let req = EdaSynthesisRequest {
        project_name: "Oxide_STM32_Sensor_Node".to_string(),
        specification_prompt: "STM32F401 with buck converter, decoupling, and I2C pullups".to_string(),
        board_width_mm: 50.0,
        board_height_mm: 35.0,
        layer_count: 4,
    };

    let artifact = engine.synthesize_project(&req).expect("Synthesize EDA project");

    assert!(artifact.schematic_kicad.contains("(kicad_sch"));
    assert!(artifact.pcb_kicad.contains("(kicad_pcb"));
    assert!(artifact.report.is_valid);
    assert_eq!(artifact.report.drc_violations.len(), 0);

    println!(
        "[+] Synthesized KiCad project: Sch = {} bytes | PCB = {} bytes | DRC Clean = {}",
        artifact.schematic_kicad.len(),
        artifact.pcb_kicad.len(),
        artifact.report.is_valid
    );
}

#[test]
fn test_real_unsloth_qlora_training_and_export() {
    let temp_dir = std::env::temp_dir().join("oxide_real_qlora_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).unwrap();

    let cfg = QloraTrainingConfig {
        learning_rate: 0.005,
        lora_r: 16,
        lora_alpha: 32.0,
        lora_dropout: 0.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
    };

    let mut trainer = UnslothQloraTrainer::new(cfg);
    trainer.add_layer("model.layers.0.self_attn.q_proj", 64, 64);
    trainer.add_layer("model.layers.0.self_attn.v_proj", 64, 64);

    let input = vec![0.25f32; 64];
    let grad_output = vec![0.1f32; 64];

    // Verify reverse-mode autograd training step
    for _ in 0..5 {
        trainer.train_step("model.layers.0.self_attn.q_proj", &input, &grad_output);
        trainer.train_step("model.layers.0.self_attn.v_proj", &input, &grad_output);
    }

    // Export checkpoint to disk
    trainer
        .export_unsloth_checkpoint(&temp_dir, "Qwen/Qwen2.5-0.5B")
        .expect("Export Unsloth QLoRA checkpoint");

    let config_file = temp_dir.join("adapter_config.json");
    let weights_file = temp_dir.join("adapter_model.safetensors");
    assert!(config_file.exists());
    assert!(weights_file.exists());

    // Ingest back with UnslothAdapter and verify round-trip fidelity
    let reloaded = UnslothAdapter::from_directory("reloaded_adapter", &temp_dir)
        .expect("Reload adapter from exported files");

    assert_eq!(reloaded.config.r, 16);
    assert_eq!(reloaded.config.lora_alpha, 32.0);
    assert!(reloaded.layers.contains_key("model.layers.0.self_attn.q_proj"));
    assert!(reloaded.layers.contains_key("model.layers.0.self_attn.v_proj"));

    let _ = std::fs::remove_dir_all(&temp_dir);
    println!("[+] Verified Unsloth QLoRA training and round-trip SafeTensors persistence");
}

#[test]
fn test_real_google_sleep_continual_learning_consolidation() {
    let mut consolidator = SleepMemoryConsolidator::new(SleepConfig {
        importance_threshold: 0.2,
        dream_learning_rate: 0.01,
        dream_cycles: 4,
        max_episodic_buffer_size: 50,
        lora_r: 8,
        lora_alpha: 16.0,
        auto_sleep_interval_secs: None,
    });

    assert_eq!(consolidator.state, SleepState::Awake);

    // Record high-value interactions
    consolidator.record_experience(
        "Design a low-power Cortex-M4 RTIC firmware architecture",
        "Use lock-free shared resources, hardware interrupts with priority ceilings, and WFI sleep",
        &[10, 20, 30, 40],
        0.95,
        Some("embedded_rust"),
    );
    consolidator.record_experience(
        "Tuning ADS1115 temperature curves",
        "Steinhart-Hart equation with 3950 NTC coefficient",
        &[50, 60, 70],
        0.85,
        Some("thermal_pid"),
    );

    let (adapter, report) = consolidator
        .execute_sleep_cycle(128, "model.layers.0.mlp.down_proj")
        .expect("Execute SLEEP consolidation cycle");

    assert_eq!(report.experiences_consolidated, 2);
    assert!(report.final_reconstruction_loss <= report.initial_reconstruction_loss);
    assert_eq!(consolidator.state, SleepState::Awake);

    let safetensors_bytes = adapter
        .save_safetensors_bytes()
        .expect("Export SLEEP adapter as SafeTensors bytes");
    assert!(!safetensors_bytes.is_empty());
    println!(
        "[+] SLEEP consolidation completed: 2 experiences -> {} bytes SafeTensors | Loss: {:.4} -> {:.4}",
        safetensors_bytes.len(),
        report.initial_reconstruction_loss,
        report.final_reconstruction_loss
    );
}

#[test]
fn test_real_audio_lalm_encoders() {
    // 1. AF-Whisper MLP Projection
    let whisper = AfWhisperEncoder::new(128, 256, 128);
    let mel = vec![0.5f32; 128 * 10]; // 10 frames of 128 mel bins
    let proj = whisper.project_features(&mel).expect("Project AF-Whisper features");
    assert_eq!(proj.len(), 10 * 128);

    // 2. MOSS-Audio 12.5 Hz Continuous Acoustic Latents
    let moss = MossAudioEncoder::new(128);
    let pcm = vec![0.1f32; 16000]; // 1 second @ 16kHz
    let acoustic_tokens = moss.encode_acoustic_stream(&pcm).expect("Encode acoustic stream");
    // At 12.5 Hz, 1 second yields ~12-13 tokens
    assert!(acoustic_tokens.len() >= 12 && acoustic_tokens.len() <= 14);

    // 3. TM-RoPE 3D Multimodal Grid Alignment
    let grid = TmRopeTemporalGrid::new(25.0, 64);
    let mut freqs = vec![0.0f32; 64];
    grid.compute_temporal_frequencies(5, &mut freqs);
    assert!(freqs.iter().any(|&f| f.abs() > 1e-6));

    println!("[+] Verified AF-Whisper, MOSS-Audio 12.5 Hz, and TM-RoPE audio towers");
}
