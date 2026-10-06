use clap::Parser;
use oxide_cli::{BackendArg, Cli};

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
        ("cpu_igpu", BackendArg::CpuIgpu),
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

#[test]
fn test_cli_subcommands_chat_server_img() {
    use oxide_cli::Commands;

    // Chat subcommand with -ngl and -p
    let chat_args = [
        "oxide-engine",
        "chat",
        "-m",
        "/models/llama-3-8b-q4_k_m.gguf",
        "-ngl",
        "99",
        "-p",
        "Hello",
    ];
    let cli = Cli::parse_normalized(chat_args).unwrap();
    match cli.command {
        Some(Commands::Chat(chat)) => {
            assert_eq!(chat.model, "/models/llama-3-8b-q4_k_m.gguf");
            assert_eq!(chat.n_gpu_layers, 99);
            assert_eq!(chat.prompt.as_deref(), Some("Hello"));
        }
        _ => panic!("Expected Chat command"),
    }

    // Chat subcommand with Qwen model
    let qwen_args = [
        "oxide-engine",
        "chat",
        "-m",
        "/models/qwen2.5-14b-fp8.gguf",
        "-ngl",
        "99",
        "-p",
        "Hello",
    ];
    let cli_qwen = Cli::parse_normalized(qwen_args).unwrap();
    match cli_qwen.command {
        Some(Commands::Chat(chat)) => {
            assert_eq!(chat.model, "/models/qwen2.5-14b-fp8.gguf");
            assert_eq!(chat.n_gpu_layers, 99);
            assert_eq!(chat.prompt.as_deref(), Some("Hello"));
        }
        _ => panic!("Expected Chat command"),
    }

    // Server subcommand with --port
    let server_args = [
        "oxide-engine",
        "server",
        "-m",
        "/models/llama-3-8b.gguf",
        "--port",
        "8080",
    ];
    let cli_server = Cli::try_parse_from(server_args).unwrap();
    match cli_server.command {
        Some(Commands::Server(srv)) => {
            assert_eq!(srv.model, "/models/llama-3-8b.gguf");
            assert_eq!(srv.port, 8080);
        }
        _ => panic!("Expected Server command"),
    }

    // Img subcommand with prompt
    let img_args = [
        "oxide-engine",
        "img",
        "-m",
        "/models/sdxl-turbo.gguf",
        "-p",
        "A cyberpunk city",
    ];
    let cli_img = Cli::try_parse_from(img_args).unwrap();
    match cli_img.command {
        Some(Commands::Img(img)) => {
            assert_eq!(img.model, "/models/sdxl-turbo.gguf");
            assert_eq!(img.prompt, "A cyberpunk city");
        }
        _ => panic!("Expected Img command"),
    }
}

#[test]
fn test_cli_top_level_prompt_and_interactive_parity() {
    // 1. Direct prompt generation (llama.cpp parity: main -m model.gguf -p "Hello")
    let prompt_args = [
        "oxide-engine",
        "-m",
        "/models/llama-3-8b.gguf",
        "-p",
        "Explain quantum computing in one sentence.",
    ];
    let cli = Cli::parse_normalized(prompt_args).unwrap();
    assert_eq!(cli.model, "/models/llama-3-8b.gguf");
    assert_eq!(
        cli.prompt.as_deref(),
        Some("Explain quantum computing in one sentence.")
    );
    assert!(!cli.interactive);

    // 2. Direct interactive mode (llama.cpp parity: main -m model.gguf -i)
    let interactive_args = [
        "oxide-engine",
        "-m",
        "./models/qwen2.5-7b.gguf",
        "-i",
        "-ngl",
        "33",
    ];
    let cli_i = Cli::parse_normalized(interactive_args).unwrap();
    assert_eq!(cli_i.model, "./models/qwen2.5-7b.gguf");
    assert!(cli_i.interactive);
    assert_eq!(cli_i.n_gpu_layers, 33);
}
