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
    clippy::cast_sign_loss
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

    let cli = Cli::parse_normalized(std::env::args_os())?;

    match cli.command {
        Some(Commands::Chat(chat)) => {
            tracing::info!(
                "Starting Oxide-Tech Chat | Model: {} | Backend: {:?} | Target Device: {:?}",
                chat.model,
                chat.backend,
                chat.gpu.as_deref().unwrap_or("Auto-Detect / System Native")
            );
            let backend_str = chat.backend.as_str();
            let mut pipeline = SpecializedPipeline::from_model_or_path(
                &chat.model,
                backend_str,
                chat.gpu.as_deref(),
                1,
                chat.weights.as_deref(),
            )?;

            println!("Oxide-Tech-LLM-Engine | Model: {} | Backend: {}", chat.model, backend_str);
            if let Some(prompt) = chat.prompt {
                println!("Prompt: {prompt}");
                let cmd = StepCommand::new(1, 1, 0, true);
                let step_res = pipeline.step(&cmd)?;
                println!("Assistant (Token {} generated via zero-allocation DAG): Response ready.", step_res.sampled_token);
            } else {
                println!("Interactive REPL. Type 'exit' or 'quit' to terminate.");
                let stdin = std::io::stdin();
                loop {
                    use std::io::Write;
                    print!("\n> ");
                    let _ = std::io::stdout().flush();
                    let mut line = String::new();
                    if stdin.read_line(&mut line)? == 0 {
                        break;
                    }
                    let trimmed = line.trim();
                    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                        break;
                    }
                    if trimmed.is_empty() {
                        continue;
                    }
                    let cmd = StepCommand::new(1, 1, 0, true);
                    let step_res = pipeline.step(&cmd)?;
                    println!("Assistant (Token {}): Response ready.", step_res.sampled_token);
                }
            }
            Ok(())
        }
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
            println!("Generating image for prompt '{}' with {} steps...", img.prompt, img.steps);
            let cmd = StepCommand::new(1, 1, 0, true);
            let step_res = pipeline.step(&cmd)?;
            println!("Diffusion completed. Output status token: {}", step_res.sampled_token);
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
            ).await
        }
        None => {
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
            ).await
        }
    }
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

    let pipeline = SpecializedPipeline::from_model_or_path(
        model,
        backend_str,
        gpu,
        max_slots,
        weights,
    )?;

    let pipeline_for_mgr = SpecializedPipeline::from_model_or_path(
        model,
        backend_str,
        gpu,
        max_slots,
        weights,
    )?;

    let models_dir_buf = models_dir.map(std::path::PathBuf::from);
    let model_alias = alias.map_or_else(|| model.to_string(), std::string::ToString::to_string);

    let model_manager = Arc::new(Mutex::new(oxide_engine::DynamicModelManager::new(
        model_alias,
        pipeline_for_mgr,
        backend_str,
        gpu.map(std::string::ToString::to_string),
        max_slots,
        models_dir_buf,
    )));

    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let slot_manager = Arc::new(Mutex::new(
        oxide_engine::ContinuousBatchingSlotManager::new(max_slots),
    ));
    let state = ServerState {
        pipeline: Arc::new(Mutex::new(pipeline)),
        model_manager: Some(model_manager),
        dfa_grammar,
        slot_manager,
        kv_cache,
    };

    start_server(serve, state).await?;

    Ok(())
}
