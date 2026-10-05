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

use crate::engine::OxideEngine;
use oxide_backend_cpu::CpuBackend;
use oxide_backend_cuda::CudaBackend;
use oxide_backend_hailo::HailoBackend;
use oxide_backend_intel::IntelBackend;
use oxide_backend_metal::MetalBackend;
use oxide_backend_qualcomm::QualcommBackend;
use oxide_backend_rknn::RknnBackend;
use oxide_backend_rocm::RocmBackend;
use oxide_backend_tpu::TpuBackend;
use oxide_core::error::Result;
use oxide_core::worker::{StepCommand, StepCompletion};
use oxide_models::audio::AudioServingEngine;
use oxide_models::bonsai2::TernaryBonsai2Config;
use oxide_models::diffusion::DiffusionEngine;
use oxide_models::llama3::Llama3Config;
use oxide_models::needle::CactusNeedleConfig;
use oxide_quant::cq2::NeedleCQ2;
use oxide_quant::nvfp4::NvFp4;
use oxide_quant::ptq1_0::Ternary1_58Bit;

/// Closed-dispatch pipeline enum eliminating vtables (`dyn Trait`) from the hot loop.
/// LLVM lowers this into a flat direct jump table.
#[derive(Debug)]
pub enum SpecializedPipeline {
    // LLM - Bonsai 2 (1.58-bit Ternary)
    Bonsai2Cuda(OxideEngine<CudaBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Rocm(OxideEngine<RocmBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Tpu(OxideEngine<TpuBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Intel(OxideEngine<IntelBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Metal(OxideEngine<MetalBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Qualcomm(OxideEngine<QualcommBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Rknn(OxideEngine<RknnBackend, TernaryBonsai2Config, Ternary1_58Bit>),
    Bonsai2Hailo(OxideEngine<HailoBackend, TernaryBonsai2Config, Ternary1_58Bit>),

    // LLM - Needle 3 (CQ2 Sub-byte Sparse)
    Needle3Cuda(OxideEngine<CudaBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Rocm(OxideEngine<RocmBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Tpu(OxideEngine<TpuBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Intel(OxideEngine<IntelBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Metal(OxideEngine<MetalBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Qualcomm(OxideEngine<QualcommBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Rknn(OxideEngine<RknnBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Hailo(OxideEngine<HailoBackend, CactusNeedleConfig<8>, NeedleCQ2>),
    Needle3Cpu(OxideEngine<CpuBackend, CactusNeedleConfig<8>, NeedleCQ2>),

    // LLM - Llama 3 (NVFP4 / FP8)
    Llama3Cuda(OxideEngine<CudaBackend, Llama3Config, NvFp4>),
    Llama3Rocm(OxideEngine<RocmBackend, Llama3Config, NvFp4>),
    Llama3Tpu(OxideEngine<TpuBackend, Llama3Config, NvFp4>),
    Llama3Intel(OxideEngine<IntelBackend, Llama3Config, NvFp4>),
    Llama3Metal(OxideEngine<MetalBackend, Llama3Config, NvFp4>),
    Llama3Qualcomm(OxideEngine<QualcommBackend, Llama3Config, NvFp4>),
    Llama3Rknn(OxideEngine<RknnBackend, Llama3Config, NvFp4>),
    Llama3Hailo(OxideEngine<HailoBackend, Llama3Config, NvFp4>),

    // Multi-Modal - Latent Diffusion, Audio Serving & Quantitative Trading
    DiffusionPipeline(DiffusionEngine),
    AudioPipeline(AudioServingEngine),
    KronosTradingPipeline(oxide_models::KronosTradingEngine),

    // Multi-Device & Heterogeneous Hybrid Pipeline (CPU+GPU, GPU+GPU+CPU)
    HybridMultiDevice(crate::hybrid::HybridMultiDevicePipeline),
}

impl SpecializedPipeline {
    /// Hot-loop forward step executing strictly through direct branch jump table.
    #[inline(always)]
    pub fn step(&mut self, cmd: &StepCommand) -> Result<StepCompletion> {
        match self {
            Self::Bonsai2Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Intel(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Metal(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Bonsai2Hailo(engine) => engine.step_monomorphized(cmd),

            Self::Needle3Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Intel(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Metal(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Hailo(engine) => engine.step_monomorphized(cmd),
            Self::Needle3Cpu(engine) => engine.step_monomorphized(cmd),

            Self::Llama3Cuda(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Rocm(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Tpu(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Intel(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Metal(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Qualcomm(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Rknn(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Hailo(engine) => engine.step_monomorphized(cmd),

            Self::DiffusionPipeline(engine) => {
                let step_idx = (cmd.input_token as usize) % engine.config().num_inference_steps;
                engine.step_denoise(step_idx)?;
                let is_terminal = step_idx + 1 >= engine.config().num_inference_steps;
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    cmd.input_token.wrapping_add(1),
                    is_terminal,
                ))
            }
            Self::AudioPipeline(engine) => {
                let pcm_chunk = [cmd.input_token as i16];
                let tokens = engine.process_streaming_asr_chunk(&pcm_chunk)?;
                let next_token = tokens.first().copied().unwrap_or(cmd.input_token + 1);
                Ok(StepCompletion::new(cmd.sequence_id, next_token, false))
            }
            Self::KronosTradingPipeline(engine) => {
                let dummy_bar = oxide_models::FinancialMarketBar {
                    timestamp_epoch_ms: 1_700_000_000_000,
                    open: 100.0,
                    high: 105.0,
                    low: 99.5,
                    close: 104.2,
                    volume: 15_000.0,
                    vwap: 102.8,
                    bid_ask_spread_bps: 1.5,
                    order_flow_imbalance: 0.35,
                };
                let bars = vec![dummy_bar; engine.context_bars.max(1)];
                let forecast = engine.evaluate_market_bars(&bars)?;
                let action_token = match forecast.signal {
                    oxide_models::TradingSignal::StrongBuy => 1,
                    oxide_models::TradingSignal::Buy => 2,
                    oxide_models::TradingSignal::Hold => 3,
                    oxide_models::TradingSignal::Sell => 4,
                    oxide_models::TradingSignal::StrongSell => 5,
                    oxide_models::TradingSignal::ClosePosition => 6,
                };
                Ok(StepCompletion::new(cmd.sequence_id, action_token, false))
            }
            Self::HybridMultiDevice(pipeline) => pipeline.step_hybrid(cmd),
        }
    }
}
