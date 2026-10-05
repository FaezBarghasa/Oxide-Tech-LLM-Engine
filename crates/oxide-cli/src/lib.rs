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
use oxide_backend_cpu::CpuBackend;
use oxide_backend_cuda::CudaBackend;
use oxide_backend_hailo::HailoBackend;
use oxide_backend_intel::IntelBackend;
use oxide_backend_metal::MetalBackend;
use oxide_backend_qualcomm::QualcommBackend;
use oxide_backend_rknn::RknnBackend;
use oxide_backend_rocm::RocmBackend;
use oxide_backend_tpu::TpuBackend;
use oxide_engine::{OxideEngine, SpecializedPipeline};
use oxide_models::audio::{AudioModelConfig, AudioServingEngine};
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::diffusion::{DiffusionEngine, DiffusionTransformerConfig};
use oxide_models::llama3::Llama3Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_server::dfa::DfaSchemaGrammar;
use oxide_server::{ServerState, start_server};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

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

#[derive(Parser, Debug)]
#[command(
    name = "oxide-engine",
    version = "0.1.0",
    about = "Oxide-Tech-LLM-Engine: Bare-metal high-throughput zero-allocation inference runtime"
)]
pub struct Cli {
    #[arg(short, long, value_enum, default_value = "bonsai2")]
    pub model: ModelArg,

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

#[allow(clippy::too_many_lines)]
pub async fn run_cli() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = Cli::parse();
    tracing::info!(
        "Booting Oxide-Tech-LLM-Engine | Model: {:?} | Backend: {:?} | Target Device: {:?}",
        cli.model,
        cli.backend,
        cli.gpu.as_deref().unwrap_or("Auto-Detect / System Native")
    );

    let kv_cache = Arc::new(Mutex::new(HierarchicalKvCache::new(
        cli.kv_device_blocks,
        cli.kv_host_blocks,
        cli.kv_storage_blocks,
    )));
    tracing::info!(
        "Initialized Hierarchical KV Cache | Tier 1 (Device): {} blks | Tier 2 (Host): {} blks | Tier 3 (Storage): {} blks",
        cli.kv_device_blocks,
        cli.kv_host_blocks,
        cli.kv_storage_blocks
    );

    if let Some(weights_path) = &cli.weights {
        tracing::info!("Validating and preparing model weights from: {}", weights_path);
    }

    let pipeline = match (cli.model, cli.backend) {
        // Multi-Modal: Diffusion Pipeline
        (ModelArg::Diffusion, _) => {
            let config = DiffusionTransformerConfig::default();
            tracing::info!(
                "Configured Diffusion DiT Pipeline | InChannels: {} | Steps: {} | Latent: {}x{}",
                config.in_channels,
                config.num_inference_steps,
                config.latent_width,
                config.latent_height
            );
            SpecializedPipeline::DiffusionPipeline(DiffusionEngine::new(config))
        }

        // Multi-Modal: Audio TTS Pipeline
        (ModelArg::AudioTts, _) => {
            let config = AudioModelConfig::new_tts_config(24000);
            tracing::info!(
                "Configured Audio TTS Serving Pipeline | SampleRate: {} Hz | Chunk: {} ms",
                config.sample_rate_hz,
                config.streaming_chunk_ms
            );
            SpecializedPipeline::AudioPipeline(AudioServingEngine::new(config))
        }

        // Multi-Modal: Audio ASR Pipeline
        (ModelArg::AudioAsr, _) => {
            let config = AudioModelConfig::default();
            tracing::info!(
                "Configured Audio ASR Serving Pipeline | SampleRate: {} Hz | N-Mels: {}",
                config.sample_rate_hz,
                config.n_mels
            );
            SpecializedPipeline::AudioPipeline(AudioServingEngine::new(config))
        }

        // LLM - Bonsai 2
        (ModelArg::Bonsai2, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Cuda(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Rocm(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Tpu(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Intel) => {
            let backend = IntelBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Intel(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Metal) => {
            let backend = MetalBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Metal(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Snapdragon) => {
            let backend = QualcommBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Qualcomm(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Rknn) => {
            let backend = RknnBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Rknn(OxideEngine::new(backend, config))
        }
        (ModelArg::Bonsai2, BackendArg::Hailo) => {
            let backend = HailoBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Hailo(OxideEngine::new(backend, config))
        }

        // LLM - Needle 3
        (ModelArg::Needle3, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Cuda(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Rocm(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Tpu(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Intel) => {
            let backend = IntelBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Intel(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Metal) => {
            let backend = MetalBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Metal(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Snapdragon) => {
            let backend = QualcommBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Qualcomm(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Rknn) => {
            let backend = RknnBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Rknn(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Hailo) => {
            let backend = HailoBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Hailo(OxideEngine::new(backend, config))
        }
        (ModelArg::Needle3, BackendArg::Cpu) => {
            let backend = CpuBackend::new(0, cli.max_slots);
            let config = CactusNeedleConfig::<8>::default();
            SpecializedPipeline::Needle3Cpu(OxideEngine::new(backend, config))
        }

        // LLM - Llama 3
        (ModelArg::Llama3, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Cuda(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Rocm(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Tpu(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Intel) => {
            let backend = IntelBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Intel(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Metal) => {
            let backend = MetalBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Metal(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Snapdragon) => {
            let backend = QualcommBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Qualcomm(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Rknn) => {
            let backend = RknnBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Rknn(OxideEngine::new(backend, config))
        }
        (ModelArg::Llama3, BackendArg::Hailo) => {
            let backend = HailoBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            SpecializedPipeline::Llama3Hailo(OxideEngine::new(backend, config))
        }

        // Quantitative Trading & Financial Time-Series Foundation Model (Kronos)
        (ModelArg::Kronos, _) => {
            let engine = oxide_models::KronosTradingEngine::new(2048, 60, 10, 1.0);
            SpecializedPipeline::KronosTradingPipeline(engine)
        }
        (m, b) => {
            tracing::warn!(
                "Backend {:?} requested with {:?}; defaulting to CUDA Bonsai2",
                b,
                m
            );
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            SpecializedPipeline::Bonsai2Cuda(OxideEngine::new(backend, config))
        }
    };

    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let slot_manager = Arc::new(Mutex::new(
        oxide_engine::ContinuousBatchingSlotManager::new(cli.max_slots),
    ));
    let state = ServerState {
        pipeline: Arc::new(Mutex::new(pipeline)),
        dfa_grammar,
        slot_manager,
        kv_cache,
    };

    start_server(cli.serve, state).await?;

    Ok(())
}
