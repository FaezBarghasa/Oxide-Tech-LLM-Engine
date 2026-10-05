use clap::Parser;
use oxide_cli::{BackendArg, Cli, ModelArg};

#[test]
fn test_cli_default_parsing() {
    let args = ["oxide-engine"];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.model, "llama3");
    assert_eq!(cli.backend, BackendArg::Cuda);
    assert_eq!(cli.max_slots, 64);
    assert_eq!(cli.ctx_size, 4096);
    assert_eq!(cli.n_gpu_layers, 0);
    assert_eq!(cli.serve.to_string(), "127.0.0.1:8080");
    assert_eq!(cli.kv_device_blocks, 1024);
    assert_eq!(cli.kv_host_blocks, 8192);
    assert_eq!(cli.kv_storage_blocks, 32768);
    assert!(cli.weights.is_none());
}

#[test]
fn test_cli_custom_options_parsing() {
    let args = [
        "oxide-engine",
        "-m",
        "./models/qwen2.5-7b.gguf",
        "--alias",
        "qwen-custom",
        "--models-dir",
        "./models",
        "-c",
        "8192",
        "--n-gpu-layers",
        "33",
        "--backend",
        "cpu",
        "--serve",
        "0.0.0.0:9090",
        "--max-slots",
        "128",
        "--gpu",
        "Apple M4 Max",
        "--weights",
        "models/llama3.safetensors",
    ];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.model, "./models/qwen2.5-7b.gguf");
    assert_eq!(cli.alias.as_deref(), Some("qwen-custom"));
    assert_eq!(cli.models_dir.as_deref(), Some("./models"));
    assert_eq!(cli.ctx_size, 8192);
    assert_eq!(cli.n_gpu_layers, 33);
    assert_eq!(cli.backend, BackendArg::Cpu);
    assert_eq!(cli.serve.to_string(), "0.0.0.0:9090");
    assert_eq!(cli.max_slots, 128);
    assert_eq!(cli.gpu.as_deref(), Some("Apple M4 Max"));
    assert_eq!(cli.weights.as_deref(), Some("models/llama3.safetensors"));
}

#[test]
fn test_cli_dynamic_model_queries() {
    let queries = [
        "qwen2.5-7b",
        "deepseek-r1-distill-qwen-8b",
        "mistral-7b-instruct",
        "gemma-2-9b",
        "bonsai2",
        "needle3",
        "diffusion",
        "audio-tts",
        "audio-asr",
        "kronos",
        "./models/custom.gguf",
        "weights.safetensors",
    ];
    for q in queries {
        let args = ["oxide-engine", "-m", q];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.model, q);
    }
}

#[test]
fn test_cli_all_backend_enums() {
    let backends = [
        ("cuda", BackendArg::Cuda),
        ("rocm", BackendArg::Rocm),
        ("tpu", BackendArg::Tpu),
        ("intel", BackendArg::Intel),
        ("metal", BackendArg::Metal),
        ("snapdragon", BackendArg::Snapdragon),
        ("rknn", BackendArg::Rknn),
        ("hailo", BackendArg::Hailo),
        ("cpu", BackendArg::Cpu),
    ];
    for (name, expected) in backends {
        let args = ["oxide-engine", "--backend", name];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.backend, expected);
        assert_eq!(cli.backend.as_str(), expected.as_str());
    }
}

#[test]
fn test_cli_safetensors_weights_loading() {
    let tmp_path = std::env::temp_dir().join("test_weights.safetensors");
    let header_json = r#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[128,32],"data_offsets":[0,16384]}}"#;
    let header_bytes = header_json.as_bytes();
    let header_len = (header_bytes.len() as u64).to_le_bytes();

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len);
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + 16384, 0x3f);

    std::fs::write(&tmp_path, &file_bytes).unwrap();

    let args = [
        "oxide-engine",
        "-m",
        tmp_path.to_str().unwrap(),
        "--backend",
        "cpu",
    ];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.model, tmp_path.to_str().unwrap());

    let _ = std::fs::remove_file(&tmp_path);
}
