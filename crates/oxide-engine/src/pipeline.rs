use crate::engine::OxideEngine;
use oxide_backend_cpu::CpuBackend;
use oxide_backend_cuda::CudaBackend;
use oxide_core::error::Result;
use oxide_core::worker::{StepCommand, StepCompletion};
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::llama3::Llama3Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_quant::cq2::NeedleCQ2;
use oxide_quant::nvfp4::NvFp4;
use oxide_quant::ptq1_0::Ternary1_58Bit;

/// Closed-dispatch pipeline enum eliminating vtables (`dyn Trait`) from the hot loop.
/// LLVM lowers this into a flat direct jump table.
#[derive(Debug)]
pub enum SpecializedPipeline {
    Bonsai2Cuda(OxideEngine<CudaBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Needle3Cuda(OxideEngine<CudaBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Cpu(OxideEngine<CpuBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Llama3Cuda(OxideEngine<CudaBackend, Llama3Config, NvFp4>),
}

impl SpecializedPipeline {
    /// Hot-loop forward step executing strictly through direct branch jump table.
    #[inline(always)]
    pub fn step(&mut self, cmd: &StepCommand) -> Result<StepCompletion> {
        match self {
            Self::Bonsai2Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Cpu(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Cuda(engine) => engine.step_monomorphized(cmd),
        }
    }
}
