use clap::{Parser, ValueEnum};
use oxide_backend_cuda::CudaBackend;
use oxide_backend_cpu::CpuBackend;
use oxide_engine::{OxideEngine, SpecializedPipeline};
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_models::llama3::Llama3Config;
use oxide_server::dfa::DfaSchemaGrammar;
use oxide_server::{start_server, ServerState};
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    let cli = Cli::parse();
    tracing::info!(
        "Booting Oxide-Tech-LLM-Engine | Model: {:?} | Backend: {:?}",
        cli.model,
        cli.backend
    );

    let pipeline = match (cli.model, cli.backend) {
        (ModelArg::Bonsai2, BackendArg::Cuda) => {
            let backend = CudaBackend::new(0, cli.max_slots);
            let config = TernaryBonsai2Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Bonsai2Cuda(engine)
        }
        (ModelArg::Needle3, BackendArg::Cuda) => {
            let backend = CudaBackend::new(0, cli.max_slots);
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Cuda(engine)
        }
        (ModelArg::Needle3, BackendArg::Cpu) => {
            let backend = CpuBackend::new(0, cli.max_slots);
            let config = CactusNeedleConfig::<8>::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Needle3Cpu(engine)
        }
        (ModelArg::Llama3, BackendArg::Cuda) => {
            let backend = CudaBackend::new(0, cli.max_slots);
            let config = Llama3Config::default();
            let engine = OxideEngine::new(backend, config);
            SpecializedPipeline::Llama3Cuda(engine)
        }
        (m, b) => {
            tracing::warn!("Backend {:?} requested with {:?}; defaulting to CUDA Bonsai2", b, m);
            let backend = CudaBackend::new(0, cli.max_slots);
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
