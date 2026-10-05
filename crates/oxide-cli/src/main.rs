use clap::{Parser, ValueEnum};
use oxide_backend_cpu::CpuBackend;
use oxide_backend_cuda::CudaBackend;
use oxide_backend_rocm::RocmBackend;
use oxide_backend_tpu::TpuBackend;
use oxide_engine::{OxideEngine, SpecializedPipeline};
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::llama3::Llama3Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_server::dfa::DfaSchemaGrammar;
use oxide_server::{ServerState, start_server};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ModelArg {
    Bonsai2,
    Needle3,
    Llama3,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Cuda,
    Rocm,
    Tpu,
    Cpu,
}

#[derive(Parser, Debug)]
#[command(
    name = "oxide",
    version = "0.1.0",
    about = "Oxide-Tech-LLM-Engine: Bare-metal high-throughput zero-allocation inference runtime"
)]
struct Cli {
    #[arg(short, long, value_enum, default_value = "bonsai2")]
    model: ModelArg,

    #[arg(short, long, value_enum, default_value = "cuda")]
    backend: BackendArg,

    #[arg(long, default_value = "127.0.0.1:8080")]
    serve: SocketAddr,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long, default_value_t = 64)]
    max_slots: usize,

    /// Target GPU/APU/TPU model name (e.g. "RTX 4090", "H100", "B200", "MI300X", "AI Max+ 395", "TPU v6e", "TPU v5p", "Coral Edge TPU")
    #[arg(long)]
    gpu: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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

    let pipeline = match (cli.model, cli.backend) {
        (ModelArg::Bonsai2, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured CUDA Profile: {} | Arch: {:?} ({}) | TensorCores: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.profile().tensor_core_gen
            );
            let config = TernaryBonsai2Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Bonsai2Cuda(engine)
        }
        (ModelArg::Bonsai2, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured ROCm Profile: {} | Arch: {:?} ({}) | MatrixEngine: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.profile().tensor_core_gen
            );
            let config = TernaryBonsai2Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Bonsai2Rocm(engine)
        }
        (ModelArg::Bonsai2, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured TPU Profile: {} | Arch: {:?} ({}) | MXU/SparseCore: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.profile().tensor_core_gen
            );
            let config = TernaryBonsai2Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Bonsai2Tpu(engine)
        }
        (ModelArg::Needle3, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured CUDA Profile: {} | Arch: {:?} ({}) | Plan: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.execution_plan()
            );
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Cuda(engine)
        }
        (ModelArg::Needle3, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured ROCm Profile: {} | Arch: {:?} ({}) | Plan: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.execution_plan()
            );
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Rocm(engine)
        }
        (ModelArg::Needle3, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            tracing::info!(
                "Configured TPU Profile: {} | Arch: {:?} ({}) | Plan: {:?}",
                backend.profile().name,
                backend.profile().architecture,
                backend.profile().compute_capability,
                backend.execution_plan()
            );
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Tpu(engine)
        }
        (ModelArg::Needle3, BackendArg::Cpu) => {
            let backend = CpuBackend::new(0, cli.max_slots);
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Cpu(engine)
        }
        (ModelArg::Llama3, BackendArg::Cuda) => {
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Llama3Cuda(engine)
        }
        (ModelArg::Llama3, BackendArg::Rocm) => {
            let backend = RocmBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Llama3Rocm(engine)
        }
        (ModelArg::Llama3, BackendArg::Tpu) => {
            let backend = TpuBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = Llama3Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Llama3Tpu(engine)
        }
        (m, b) => {
            tracing::warn!(
                "Backend {:?} requested with {:?}; defaulting to CUDA Bonsai2",
                b,
                m
            );
            let backend = CudaBackend::new_with_profile(0, cli.max_slots, cli.gpu.as_deref());
            let config = TernaryBonsai2Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Bonsai2Cuda(engine)
        }
    };

    let dfa_grammar = Arc::new(DfaSchemaGrammar::new_simple_json_validator());
    let state = ServerState {
        pipeline: Arc::new(Mutex::new(pipeline)),
        dfa_grammar,
    };

    start_server(cli.serve, state).await?;

    Ok(())
}
