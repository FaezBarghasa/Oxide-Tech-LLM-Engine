#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

use clap::{Parser, ValueEnum};
use oxide_alloc::HierarchicalKvCache;
use oxide_core::worker::StepCommand;
use oxide_engine::SpecializedPipeline;
use oxide_server::dfa::DfaSchemaGrammar;
use oxide_server::{ServerState, start_server};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

/// Legacy/Convenience Model enum preserved for backwards-compatible test assertions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum ModelArg {
    Bonsai2,
    Needle3,
    Llama3,
    Diffusion,
    AudioTts,
    AudioAsr,
    Kronos,
}

impl ModelArg {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bonsai2 => "bonsai2",
            Self::Needle3 => "needle3",
            Self::Llama3 => "llama3",
            Self::Diffusion => "diffusion",
            Self::AudioTts => "audio-tts",
            Self::AudioAsr => "audio-asr",
            Self::Kronos => "kronos",
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    Cuda,
    Rocm,
    Tpu,
    Intel,
    Metal,
    Snapdragon,
    Rknn,
    Hailo,
    Cpu,
    #[value(name = "cpu_igpu", alias = "cpu-igpu")]
    CpuIgpu,
    Apu,
    Epyc,
    #[value(name = "arm_npu", alias = "arm-npu")]
    ArmNpu,
    Hybrid,
}

impl BackendArg {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::Tpu => "tpu",
            Self::Intel => "intel",
            Self::Metal => "metal",
            Self::Snapdragon => "qualcomm",
            Self::Rknn => "rknn",
            Self::Hailo => "hailo",
            Self::Cpu => "cpu",
            Self::CpuIgpu => "cpu_igpu",
            Self::Apu => "apu",
            Self::Epyc => "epyc",
            Self::ArmNpu => "arm_npu",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum Commands {
    /// Interactive chat or single-prompt generation
    Chat(ChatArgs),
    /// Start OpenAI-compatible API server
    Server(ServerArgs),
    /// Image diffusion generation (e.g. sdxl-turbo, flux)
    Img(ImgArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct ChatArgs {
    /// Model name or path to GGUF/SafeTensors file
    #[arg(short = 'm', long, default_value = "llama3")]
    pub model: String,

    /// Number of layers to offload to GPU accelerator (-ngl / --n-gpu-layers)
    #[arg(long = "n-gpu-layers", alias = "ngl", default_value_t = 0)]
    pub n_gpu_layers: usize,

    /// Context window size (-c / --ctx-size)
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    pub ctx_size: usize,

    /// Prompt to run (if omitted, starts interactive chat)
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// Backend to use
    #[arg(short, long, value_enum, default_value = "cuda")]
    pub backend: BackendArg,

    /// Target hardware/device profile name
    #[arg(long)]
    pub gpu: Option<String>,

    /// Optional path to weights file
    #[arg(long)]
    pub weights: Option<String>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ServerArgs {
    /// Model name or path to GGUF/SafeTensors file
    #[arg(short = 'm', long, default_value = "llama3")]
    pub model: String,

    /// Server port (llama.cpp compatible --port)
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Server listen host
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Number of layers to offload to GPU accelerator (-ngl / --n-gpu-layers)
    #[arg(long = "n-gpu-layers", alias = "ngl", default_value_t = 0)]
    pub n_gpu_layers: usize,

    /// Context window size (-c / --ctx-size)
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    pub ctx_size: usize,

    /// Backend to use
    #[arg(short, long, value_enum, default_value = "cuda")]
    pub backend: BackendArg,

    /// Directory containing GGUF and SafeTensors model files for dynamic loading
    #[arg(long)]
    pub models_dir: Option<String>,

    /// Optional alias name for the loaded model in OpenAI API responses
    #[arg(long)]
    pub alias: Option<String>,

    /// Target hardware/device profile name
    #[arg(long)]
    pub gpu: Option<String>,

    /// Maximum concurrent slots
    #[arg(long, default_value_t = 64)]
    pub max_slots: usize,

    /// Optional path to weights file
    #[arg(long)]
    pub weights: Option<String>,

    /// Tier 1 Device VRAM KV cache capacity in blocks
    #[arg(long, default_value_t = 1024)]
    pub kv_device_blocks: usize,

    /// Tier 2 Host RAM KV cache capacity in blocks
    #[arg(long, default_value_t = 8192)]
    pub kv_host_blocks: usize,

    /// Tier 3 External Storage / NVMe KV cache capacity in blocks
    #[arg(long, default_value_t = 32768)]
    pub kv_storage_blocks: usize,
}

#[derive(clap::Args, Debug, Clone)]
pub struct ImgArgs {
    /// Model name or path to GGUF/SafeTensors diffusion model
    #[arg(short = 'm', long, default_value = "diffusion")]
    pub model: String,

    /// Prompt to generate
    #[arg(short = 'p', long)]
    pub prompt: String,

    /// Number of inference steps
    #[arg(long, default_value_t = 20)]
    pub steps: usize,

    /// Backend to use
    #[arg(short, long, value_enum, default_value = "cuda")]
    pub backend: BackendArg,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "oxide-engine",
    version = "0.1.0",
    about = "Oxide-Tech-LLM-Engine: Bare-metal high-throughput zero-allocation inference runtime"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Model name (e.g. "llama-3.1-8b", "qwen2.5-7b", "deepseek-r1", "mistral-7b", "bonsai2") or path to GGUF / SafeTensors file
    #[arg(short = 'm', long, default_value = "llama3")]
    pub model: String,

    /// Optional alias name for the loaded model in OpenAI API responses
    #[arg(long)]
    pub alias: Option<String>,

    /// Directory containing GGUF and SafeTensors model files for dynamic loading
    #[arg(long)]
    pub models_dir: Option<String>,

    /// Context window length (llama.cpp compatible -c / --ctx-size)
    #[arg(short = 'c', long, default_value_t = 4096)]
    pub ctx_size: usize,

    /// Number of layers to offload to GPU accelerator (llama.cpp compatible -ngl / --n-gpu-layers)
    #[arg(long = "n-gpu-layers", alias = "ngl", default_value_t = 0)]
    pub n_gpu_layers: usize,

    #[arg(short, long, value_enum, default_value = "cuda")]
    pub backend: BackendArg,

    #[arg(long, default_value = "127.0.0.1:8080")]
    pub serve: SocketAddr,

    #[arg(long)]
    pub weights: Option<String>,

    #[arg(long, default_value_t = 64)]
    pub max_slots: usize,

    /// Target hardware/device profile name (e.g. "RTX 4090", "H100", "Arc B580", "Xeon 6980P", "Apple M4 Max", "Snapdragon X Elite", "Orange Pi 6 Plus", "RPi5 with AI HAT+ 2")
    #[arg(long)]
    pub gpu: Option<String>,

    /// Prompt to run directly without entering server mode (llama.cpp -p / --prompt)
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// Enter interactive chat REPL mode directly (llama.cpp -i / --interactive)
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Tier 1 Device VRAM KV cache capacity in blocks
    #[arg(long, default_value_t = 1024)]
    pub kv_device_blocks: usize,

    /// Tier 2 Host RAM KV cache capacity in blocks
    #[arg(long, default_value_t = 8192)]
    pub kv_host_blocks: usize,

    /// Tier 3 External Storage / NVMe KV cache capacity in blocks
    #[arg(long, default_value_t = 32768)]
    pub kv_storage_blocks: usize,
}

impl Cli {
    pub fn parse_normalized<I, T>(itr: I) -> Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let normalized: Vec<std::ffi::OsString> = itr
            .into_iter()
            .map(|item| {
                let os_str: std::ffi::OsString = item.into();
                if let Some(s) = os_str.to_str() {
                    if s == "-ngl" {
                        return std::ffi::OsString::from("--n-gpu-layers");
                    }
                    if let Some(rest) = s.strip_prefix("-ngl=") {
                        return std::ffi::OsString::from(format!("--n-gpu-layers={rest}"));
                    }
                }
                os_str
            })
            .collect();
        Self::try_parse_from(normalized)
    }
}

pub async fn run_cli() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = match Cli::parse_normalized(std::env::args_os()) {
        Ok(c) => c,
        Err(e) => {
            e.exit();
        }
    };

    match cli.command {
        Some(Commands::Chat(chat)) => run_chat_session(
            &chat.model,
            chat.backend,
            chat.gpu.as_deref(),
            chat.weights.as_deref(),
            chat.prompt.as_deref(),
            None,
        ),
        Some(Commands::Img(img)) => {
            tracing::info!(
                "Starting Oxide-Tech Diffusion Engine | Model: {} | Backend: {:?}",
                img.model,
                img.backend
            );
            let mut pipeline = SpecializedPipeline::from_model_or_path(
                &img.model,
                img.backend.as_str(),
                None,
                1,
                None,
            )?;
            println!(
                "Generating image for prompt '{}' with {} steps...",
                img.prompt, img.steps
            );
            let cmd = StepCommand::new(1, 1, 0, true);
            let step_res = pipeline.step(&cmd)?;
            println!(
                "Diffusion completed. Output status token: {}",
                step_res.sampled_token
            );
            Ok(())
        }
        Some(Commands::Server(srv)) => {
            let addr: SocketAddr = format!("{}:{}", srv.host, srv.port).parse()?;
            run_server_with_options(
                &srv.model,
                srv.alias.as_deref(),
                srv.models_dir.as_deref(),
                srv.ctx_size,
                srv.n_gpu_layers,
                srv.backend,
                addr,
                srv.weights.as_deref(),
                srv.max_slots,
                srv.gpu.as_deref(),
                srv.kv_device_blocks,
                srv.kv_host_blocks,
                srv.kv_storage_blocks,
            )
            .await
        }
        None => {
            if cli.prompt.is_some() || cli.interactive {
                run_chat_session(
                    &cli.model,
                    cli.backend,
                    cli.gpu.as_deref(),
                    cli.weights.as_deref(),
                    cli.prompt.as_deref(),
                    cli.models_dir.as_deref(),
                )
            } else {
                run_server_with_options(
                    &cli.model,
                    cli.alias.as_deref(),
                    cli.models_dir.as_deref(),
                    cli.ctx_size,
                    cli.n_gpu_layers,
                    cli.backend,
                    cli.serve,
                    cli.weights.as_deref(),
                    cli.max_slots,
                    cli.gpu.as_deref(),
                    cli.kv_device_blocks,
                    cli.kv_host_blocks,
                    cli.kv_storage_blocks,
                )
                .await
            }
        }
    }
}

fn run_chat_session(
    initial_model: &str,
    backend: BackendArg,
    gpu: Option<&str>,
    weights: Option<&str>,
    prompt: Option<&str>,
    models_dir: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let backend_str = backend.as_str();
    let mut current_model = initial_model.to_string();
    tracing::info!(
        "Starting Oxide-Tech Chat | Model: {} | Backend: {:?} | Target Device: {:?}",
        current_model,
        backend,
        gpu.unwrap_or("Auto-Detect / System Native")
    );
    let mut pipeline =
        SpecializedPipeline::from_model_or_path(&current_model, backend_str, gpu, 1, weights)?;

    println!("Oxide-Tech-LLM-Engine | Model: {current_model} | Backend: {backend_str}");

    let tokenizer = {
        let p = std::path::Path::new(&current_model);
        if p.exists() {
            if let Ok(file) = std::fs::File::open(p) {
                // SAFETY: The model file is mapped read-only for metadata/tokenizer extraction and is immutable.
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&file) } {
                    if let Ok(gguf) = oxide_models::GgufFile::parse(&mmap) {
                        oxide_models::GgufTokenizer::from_gguf(&gguf)
                    } else {
                        oxide_models::GgufTokenizer::default()
                    }
                } else {
                    oxide_models::GgufTokenizer::default()
                }
            } else {
                oxide_models::GgufTokenizer::default()
            }
        } else {
            oxide_models::GgufTokenizer::default()
        }
    };

    if let Some(p) = prompt {
        println!("Prompt: {p}");
        let tokens = tokenizer.encode(p);
        let mut cur_token = tokens.last().copied().unwrap_or(1);
        print!("Assistant: ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let t_start = std::time::Instant::now();
        let mut ttft = None;
        let mut gen_count = 0usize;

        for _ in 0..64 {
            let cmd = StepCommand::new(1, cur_token, 0, false);
            let step_res = pipeline.step(&cmd)?;
            if ttft.is_none() {
                ttft = Some(t_start.elapsed());
            }
            gen_count += 1;
            cur_token = step_res.sampled_token;
            let text = tokenizer.decode_token(cur_token);
            print!("{text}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if step_res.is_terminal {
                break;
            }
        }
        let elapsed = t_start.elapsed();
        println!();
        if let Some(first_tok_time) = ttft {
            let gen_u32 = u32::try_from(gen_count).unwrap_or(u32::MAX);
            let tps = if elapsed.as_secs_f64() > 0.0 {
                f64::from(gen_u32) / elapsed.as_secs_f64()
            } else {
                0.0
            };
            println!(
                "--------------------------------------------------\n\
                 [Benchmark Metrics]\n\
                 Generated Tokens : {gen_count}\n\
                 Total Time       : {elapsed:.2?}\n\
                 TTFT             : {first_tok_time:.2?}\n\
                 Throughput       : {tps:.2} tokens/sec\n\
                 --------------------------------------------------"
            );
        }
        return Ok(());
    }

    println!("Interactive REPL mode (llama.cpp compatible).");
    println!("Type '/help' for command options or '/model <path_or_name>' to hot-swap models.");
    let stdin = std::io::stdin();
    loop {
        use std::io::Write;
        print!("\n[{current_model}] > ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("exit")
            || trimmed.eq_ignore_ascii_case("quit")
            || trimmed.eq_ignore_ascii_case("/exit")
            || trimmed.eq_ignore_ascii_case("/quit")
        {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("/help") {
            println!("Interactive Options:");
            println!(
                "  /model <name_or_path> - Hot-swap active model without restarting or recompiling"
            );
            println!("  /models or /list      - List available models from directory & catalog");
            println!("  /info                 - Show active model and hardware backend");
            println!("  /exit or /quit        - Exit chat session");
            continue;
        }
        if trimmed.eq_ignore_ascii_case("/info") {
            println!("Active Model: {current_model}");
            println!("Hardware Backend: {backend_str}");
            println!("Target Device Profile: {}", gpu.unwrap_or("Native / Auto"));
            continue;
        }
        if trimmed.eq_ignore_ascii_case("/models") || trimmed.eq_ignore_ascii_case("/list") {
            println!("Scanning discoverable models:");
            let mgr = oxide_engine::DynamicModelManager::new(
                &current_model,
                SpecializedPipeline::from_model_or_path(
                    &current_model,
                    backend_str,
                    gpu,
                    1,
                    weights,
                )?,
                backend_str,
                gpu.map(std::string::ToString::to_string),
                1,
                models_dir.map(std::path::PathBuf::from),
            );
            for (idx, m) in mgr.list_available().iter().enumerate() {
                let marker = if m == &current_model { "*" } else { " " };
                println!(" [{marker}] {idx}: {m}");
            }
            continue;
        }
        if let Some(target) = trimmed
            .strip_prefix("/model ")
            .or_else(|| trimmed.strip_prefix("/load "))
        {
            let target = target.trim();
            if target.is_empty() {
                println!("Usage: /model <model_name_or_path>");
                continue;
            }
            println!("Hot-swapping model to '{target}'...");
            match SpecializedPipeline::from_model_or_path(target, backend_str, gpu, 1, None) {
                Ok(new_pipe) => {
                    pipeline = new_pipe;
                    current_model = target.to_string();
                    println!("Successfully loaded and switched model to: {current_model}");
                }
                Err(err) => {
                    println!("Failed to load model '{target}': {err}");
                }
            }
            continue;
        }

        let tokens = tokenizer.encode(trimmed);
        let mut cur_token = tokens.last().copied().unwrap_or(1);
        print!("Assistant: ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        for _ in 0..64 {
            let cmd = StepCommand::new(1, cur_token, 0, false);
            let step_res = pipeline.step(&cmd)?;
            cur_token = step_res.sampled_token;
            let text = tokenizer.decode_token(cur_token);
            print!("{text}");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if step_res.is_terminal {
                break;
            }
        }
        println!();
    }

    Ok(())
}

async fn run_server_with_options(
    model: &str,
    alias: Option<&str>,
    models_dir: Option<&str>,
    _ctx_size: usize,
    _n_gpu_layers: usize,
    backend: BackendArg,
    serve: SocketAddr,
    weights: Option<&str>,
    max_slots: usize,
    gpu: Option<&str>,
    kv_device_blocks: usize,
    kv_host_blocks: usize,
    kv_storage_blocks: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing::info!(
        "Booting Oxide-Tech-LLM-Engine | Model: {} | Backend: {:?} | Target Device: {:?}",
        model,
        backend,
        gpu.unwrap_or("Auto-Detect / System Native")
    );

    let kv_cache = Arc::new(Mutex::new(HierarchicalKvCache::new(
        kv_device_blocks,
        kv_host_blocks,
        kv_storage_blocks,
    )));
    tracing::info!(
        "Initialized Hierarchical KV Cache | Tier 1 (Device): {} blks | Tier 2 (Host): {} blks | Tier 3 (Storage): {} blks",
        kv_device_blocks,
        kv_host_blocks,
        kv_storage_blocks
    );

    if let Some(weights_path) = weights {
        let path = std::path::Path::new(weights_path);
        if !path.exists() {
            return Err(format!("Model weights path does not exist: {weights_path}").into());
        }
        tracing::info!("Validated model weights path: {}", weights_path);
    }

    let backend_str = backend.as_str();

    let pipeline =
        SpecializedPipeline::from_model_or_path(model, backend_str, gpu, max_slots, weights)?;

    let models_dir_buf = models_dir.map(std::path::PathBuf::from);
    let model_alias = alias.map_or_else(|| model.to_string(), std::string::ToString::to_string);

    let model_manager_inst = oxide_engine::DynamicModelManager::new(
        model_alias,
        pipeline,
        backend_str,
        gpu.map(std::string::ToString::to_string),
        max_slots,
        models_dir_buf,
    );
    let default_pipeline = model_manager_inst.get_default_pipeline();
    let model_manager = Arc::new(Mutex::new(model_manager_inst));

    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let slot_manager = Arc::new(Mutex::new(
        oxide_engine::ContinuousBatchingSlotManager::new(max_slots),
    ));

    let tokenizer = {
        let p = std::path::Path::new(model);
        if p.exists() {
            if let Ok(file) = std::fs::File::open(p) {
                // SAFETY: The model file is mapped read-only for metadata/tokenizer extraction and is immutable.
                if let Ok(mmap) = unsafe { memmap2::Mmap::map(&file) } {
                    if let Ok(gguf) = oxide_models::GgufFile::parse(&mmap) {
                        oxide_models::GgufTokenizer::from_gguf(&gguf)
                    } else {
                        oxide_models::GgufTokenizer::default()
                    }
                } else {
                    oxide_models::GgufTokenizer::default()
                }
            } else {
                oxide_models::GgufTokenizer::default()
            }
        } else {
            oxide_models::GgufTokenizer::default()
        }
    };

    let state = ServerState {
        pipeline: default_pipeline,
        model_manager: Some(model_manager),
        dfa_grammar,
        slot_manager,
        kv_cache,
        tokenizer: Arc::new(tokenizer),
    };

    start_server(serve, state).await?;

    Ok(())
}
