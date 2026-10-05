use clap::Parser;
use oxide_cli::{BackendArg, Cli, ModelArg};

#[test]
fn test_cli_default_parsing() {
    let args = ["oxide-engine"];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.model, ModelArg::Bonsai2);
    assert_eq!(cli.backend, BackendArg::Cuda);
    assert_eq!(cli.max_slots, 64);
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
        "--model",
        "llama3",
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
    assert_eq!(cli.model, ModelArg::Llama3);
    assert_eq!(cli.backend, BackendArg::Cpu);
    assert_eq!(cli.serve.to_string(), "0.0.0.0:9090");
    assert_eq!(cli.max_slots, 128);
    assert_eq!(cli.gpu.as_deref(), Some("Apple M4 Max"));
    assert_eq!(cli.weights.as_deref(), Some("models/llama3.safetensors"));
}

#[test]
fn test_cli_all_model_enums() {
    let models = [
        ("bonsai2", ModelArg::Bonsai2),
        ("needle3", ModelArg::Needle3),
        ("llama3", ModelArg::Llama3),
        ("diffusion", ModelArg::Diffusion),
        ("audio-tts", ModelArg::AudioTts),
        ("audio-asr", ModelArg::AudioAsr),
        ("kronos", ModelArg::Kronos),
    ];
    for (name, expected) in models {
        let args = ["oxide-engine", "--model", name];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.model, expected);
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
        "--model",
        "llama3",
        "--backend",
        "cpu",
        "--weights",
        tmp_path.to_str().unwrap(),
    ];
    let cli = Cli::try_parse_from(args).unwrap();
    assert_eq!(cli.weights.as_deref(), Some(tmp_path.to_str().unwrap()));

    let _ = std::fs::remove_file(&tmp_path);
}
