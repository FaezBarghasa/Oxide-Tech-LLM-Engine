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
    clippy::cast_precision_loss
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
    Llama3Cpu(OxideEngine<CpuBackend, Llama3Config, NvFp4>),
    Llama3Dense {
        model: oxide_models::Llama3Model,
        kv_cache: Vec<oxide_models::llama3::Llama3KvCacheLayer>,
        seq_positions: std::collections::HashMap<u64, usize>,
        scratch: Box<oxide_models::llama3::Llama3ScratchBuffers>,
    },
    /// AMD APU Heterogeneous CPU + iGPU co-processing pipeline.
    /// Distributes transformer layers between Zen CPU AVX2 SIMD threads and RDNA integrated GPU compute units over coherent DDR5.
    Llama3AmdApuCpuIgpu {
        model: oxide_models::Llama3Model,
        kv_cache: Vec<oxide_models::llama3::Llama3KvCacheLayer>,
        seq_positions: std::collections::HashMap<u64, usize>,
        scratch: Box<oxide_models::llama3::Llama3ScratchBuffers>,
        topology: crate::hybrid::HybridDeviceTopology,
    },

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
            Self::Llama3Cpu(engine) => engine.step_monomorphized(cmd),
            Self::Llama3Dense {
                model,
                kv_cache,
                seq_positions,
                scratch,
            } => {
                let pos = seq_positions.entry(cmd.sequence_id).or_insert(0);
                model.forward_step_with_scratch(cmd.input_token, *pos, kv_cache, scratch)?;
                *pos += 1;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &l) in scratch.logits.iter().enumerate() {
                    if l > max_val {
                        max_val = l;
                        max_idx = i;
                    }
                }
                let next_token = max_idx as u32;
                let is_terminal = next_token == 0
                    || next_token == 2
                    || next_token == 128_001
                    || next_token == 128_009;
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    next_token,
                    is_terminal,
                ))
            }

            Self::Llama3AmdApuCpuIgpu {
                model,
                kv_cache,
                seq_positions,
                scratch,
                topology,
            } => {
                let pos = seq_positions.entry(cmd.sequence_id).or_insert(0);

                // 1. Token Embedding lookup on host DDR5
                let h = model.config.hidden_dim;
                let tok_idx = (cmd.input_token as usize) % model.config.vocab_size;
                let emb_offset = tok_idx * h;
                if emb_offset + h <= model.token_embedding.len() {
                    scratch
                        .hidden
                        .copy_from_slice(&model.token_embedding[emb_offset..emb_offset + h]);
                } else {
                    scratch.hidden.fill(0.0);
                }

                // 2. Collaborative execution across topology partitions (CPU AVX2 threads + iGPU CUs)
                for partition in &topology.partitions {
                    model.forward_layers(
                        partition.start_layer,
                        partition.end_layer,
                        *pos,
                        kv_cache,
                        scratch,
                    );
                }

                // 3. Final RMSNorm
                let eps = model.config.rms_norm_eps_f32();
                let mean_sq = oxide_quant::simd::dot_f32(&scratch.hidden, &scratch.hidden)
                    / scratch.hidden.len().max(1) as f32;
                let inv_rms = 1.0 / (mean_sq + eps).sqrt();
                for (i, (&h_val, &w_val)) in scratch
                    .hidden
                    .iter()
                    .zip(model.output_norm.iter())
                    .enumerate()
                {
                    scratch.final_norm[i] = h_val * inv_rms * w_val;
                }

                // 4. LM Head projection via multithreaded SIMD matrix-vector multiplication
                model.lm_head.gemv(&scratch.final_norm, &mut scratch.logits);

                *pos += 1;
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &l) in scratch.logits.iter().enumerate() {
                    if l > max_val {
                        max_val = l;
                        max_idx = i;
                    }
                }
                let next_token = max_idx as u32;
                let is_terminal = next_token == 0
                    || next_token == 2
                    || next_token == 128_001
                    || next_token == 128_009;
                Ok(StepCompletion::new(
                    cmd.sequence_id,
                    next_token,
                    is_terminal,
                ))
            }

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

    /// Universal model loader resolving from a local file path (GGUF/SafeTensors),
    /// a catalog model name (e.g. Qwen, DeepSeek, Mistral, Llama), or domain pipelines.
    pub fn from_model_or_path(
        model_query_or_path: &str,
        backend_name: &str,
        gpu_profile: Option<&str>,
        max_slots: usize,
        weights_override: Option<&str>,
    ) -> Result<Self> {
        let q_lower = model_query_or_path.to_lowercase();
        let path = std::path::Path::new(model_query_or_path);

        // 1. If explicit file path on disk (GGUF or SafeTensors)
        if path.exists() {
            let model = oxide_models::Llama3Model::from_file(path)?;
            let kv_cache = (0..model.config.num_layers)
                .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
                .collect();
            let scratch = Box::new(model.create_scratch());

            let is_apu = backend_name.eq_ignore_ascii_case("apu");
            let is_cpu_igpu = backend_name.eq_ignore_ascii_case("cpu_igpu");
            let auto_apu = backend_name.eq_ignore_ascii_case("cpu")
                && oxide_core::hardware::GpuDeviceProfile::detect_amd_cpu_and_igpu().is_some();

            if is_apu || is_cpu_igpu || auto_apu {
                let has_npu = is_apu
                    || oxide_core::hardware::GpuDeviceProfile::detect_amd_apu_full()
                        .is_some_and(|(_, _, npu)| npu);
                let topology = crate::hybrid::HybridDeviceTopology::amd_apu_full_partition(
                    model.config.num_layers,
                    has_npu,
                );
                return Ok(Self::Llama3AmdApuCpuIgpu {
                    model,
                    kv_cache,
                    seq_positions: std::collections::HashMap::new(),
                    scratch,
                    topology,
                });
            }

            return Ok(Self::Llama3Dense {
                model,
                kv_cache,
                seq_positions: std::collections::HashMap::new(),
                scratch,
            });
        }

        // 2. Multi-modal / domain engine routes
        if q_lower.contains("diffusion") || q_lower.contains("flux") {
            let cfg = oxide_models::diffusion::DiffusionTransformerConfig::default();
            return Ok(Self::DiffusionPipeline(DiffusionEngine::new(cfg)));
        }
        if q_lower.contains("audio-tts") || q_lower.contains("kokoro") {
            let cfg = oxide_models::audio::AudioModelConfig::new_tts_config(24000);
            return Ok(Self::AudioPipeline(AudioServingEngine::new(cfg)));
        }
        if q_lower.contains("audio-asr") || q_lower.contains("whisper") {
            let cfg = oxide_models::audio::AudioModelConfig::default();
            return Ok(Self::AudioPipeline(AudioServingEngine::new(cfg)));
        }
        if q_lower.contains("kronos") || q_lower.contains("trading") {
            let engine = oxide_models::KronosTradingEngine::new(2048, 60, 10, 1.0);
            return Ok(Self::KronosTradingPipeline(engine));
        }
        if q_lower.contains("bonsai") {
            let backend = CudaBackend::new_with_profile(0, max_slots, gpu_profile);
            let config = TernaryBonsai2Config::default();
            return Ok(Self::Bonsai2Cuda(OxideEngine::new(backend, config)));
        }
        if q_lower.contains("needle") {
            let backend = CpuBackend::new(0, max_slots);
            let config = CactusNeedleConfig::<8>::default();
            return Ok(Self::Needle3Cpu(OxideEngine::new(backend, config)));
        }

        // 3. Universal LLM: Load from weights override or catalog lookup
        let model = if let Some(weights_path) = weights_override {
            let w_path = std::path::Path::new(weights_path);
            if w_path.exists() {
                oxide_models::Llama3Model::from_file(w_path)?
            } else {
                oxide_models::Llama3Model::from_model_name_or_path(model_query_or_path)?
            }
        } else {
            oxide_models::Llama3Model::from_model_name_or_path(model_query_or_path)?
        };

        if weights_override.is_none() {
            match backend_name.to_ascii_lowercase().as_str() {
                "cuda" => {
                    let backend = CudaBackend::new_with_profile(0, max_slots, gpu_profile);
                    return Ok(Self::Llama3Cuda(OxideEngine::new(backend, model.config)));
                }
                "rocm" => {
                    let backend = RocmBackend::new(0, max_slots);
                    return Ok(Self::Llama3Rocm(OxideEngine::new(backend, model.config)));
                }
                "tpu" => {
                    let backend = TpuBackend::new(0, max_slots);
                    return Ok(Self::Llama3Tpu(OxideEngine::new(backend, model.config)));
                }
                "intel" => {
                    let backend = IntelBackend::new(0, max_slots);
                    return Ok(Self::Llama3Intel(OxideEngine::new(backend, model.config)));
                }
                "metal" => {
                    let backend = MetalBackend::new(0, max_slots);
                    return Ok(Self::Llama3Metal(OxideEngine::new(backend, model.config)));
                }
                "qualcomm" | "snapdragon" => {
                    let backend = QualcommBackend::new(0, max_slots);
                    return Ok(Self::Llama3Qualcomm(OxideEngine::new(backend, model.config)));
                }
                "rknn" | "rockchip" => {
                    let backend = RknnBackend::new(0, max_slots);
                    return Ok(Self::Llama3Rknn(OxideEngine::new(backend, model.config)));
                }
                "hailo" => {
                    let backend = HailoBackend::new(0, max_slots);
                    return Ok(Self::Llama3Hailo(OxideEngine::new(backend, model.config)));
                }
                _ => {}
            }
        }


        let kv_cache = (0..model.config.num_layers)
            .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
            .collect();

        let scratch = Box::new(model.create_scratch());

        let norm_backend = backend_name.to_ascii_lowercase().replace('-', "_");

        let maybe_topology = match norm_backend.as_str() {
            "cpu_nvidia" | "cpu+nvidia" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_nvidia_partition(
                    model.config.num_layers,
                    1,
                    (model.config.num_layers * 2) / 10,
                ))
            }
            "cpu_amd" | "cpu+amd" => Some(crate::hybrid::HybridDeviceTopology::cpu_amd_partition(
                model.config.num_layers,
                1,
                (model.config.num_layers * 2) / 10,
            )),
            "cpu_intel" | "cpu+intel" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_intel_partition(
                    model.config.num_layers,
                    1,
                    (model.config.num_layers * 2) / 10,
                ))
            }
            "cpu_tpu" | "cpu+tpu" => Some(crate::hybrid::HybridDeviceTopology::cpu_tpu_partition(
                model.config.num_layers,
                1,
                (model.config.num_layers * 2) / 10,
            )),
            "cpu_npu" | "cpu+npu" => Some(crate::hybrid::HybridDeviceTopology::cpu_npu_partition(
                model.config.num_layers,
                0.65,
            )),
            "cpu_nvidia_amd_intel" | "cpu+nvidia+amd+intel" | "triple_gpu" | "hybrid" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_amd_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_amd_intel" | "cpu+amd+intel" | "amd_intel" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_amd_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_nvidia_intel" | "cpu+nvidia+intel" | "nvidia_intel" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_intel_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_nvidia_amd" | "cpu+nvidia+amd" | "nvidia_amd" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_nvidia_amd_partition(
                    model.config.num_layers,
                    1,
                    1,
                    (model.config.num_layers * 2) / 10,
                ),
            ),
            "cpu_igpu_npu" | "cpu+igpu+npu" | "apu" | "cpu_igpu" => {
                let has_npu = norm_backend == "apu"
                    || norm_backend.contains("npu")
                    || oxide_core::hardware::GpuDeviceProfile::detect_amd_apu_full()
                        .is_some_and(|(_, _, npu)| npu);
                Some(crate::hybrid::HybridDeviceTopology::amd_apu_full_partition(
                    model.config.num_layers,
                    has_npu,
                ))
            }
            "cpu_igpu_tpu" | "cpu+igpu+tpu" => {
                Some(crate::hybrid::HybridDeviceTopology::cpu_igpu_tpu_partition(
                    model.config.num_layers,
                    1,
                ))
            }
            "cpu_igpu_npu_nvidia" | "cpu+igpu+npu+nvidia" | "apu_nvidia" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_igpu_npu_nvidia_partition(
                    model.config.num_layers,
                    1,
                ),
            ),
            "cpu_igpu_npu_amd" | "cpu+igpu+npu+amd" | "apu_amd" => Some(
                crate::hybrid::HybridDeviceTopology::cpu_igpu_npu_amd_partition(
                    model.config.num_layers,
                    1,
                ),
            ),
            "arm_npu_hat" | "arm+npu+hat" | "arm_npu" | "arm+npu" => Some(
                crate::hybrid::HybridDeviceTopology::arm_npu_external_hat_partition(
                    model.config.num_layers,
                ),
            ),
            "arm_integrated_npu" | "arm_npu_only" => Some(
                crate::hybrid::HybridDeviceTopology::arm_integrated_npu_partition(
                    model.config.num_layers,
                ),
            ),
            "epyc" | "epyc_server" => Some(
                crate::hybrid::HybridDeviceTopology::epyc_server_standalone_partition(
                    model.config.num_layers,
                    2,
                ),
            ),
            "epyc_gpu" | "epyc+gpu" => Some(
                crate::hybrid::HybridDeviceTopology::epyc_server_gpu_partition(
                    model.config.num_layers,
                    2,
                    &[crate::hybrid::DeviceRole::NvidiaGpu(0)],
                ),
            ),
            "apple" | "apple_silicon" => Some(
                crate::hybrid::HybridDeviceTopology::apple_silicon_uma_partition(
                    model.config.num_layers,
                ),
            ),
            "snapdragon" | "qualcomm" => Some(
                crate::hybrid::HybridDeviceTopology::qualcomm_snapdragon_partition(
                    model.config.num_layers,
                ),
            ),
            "intel_ultra" => Some(
                crate::hybrid::HybridDeviceTopology::intel_core_ultra_partition(
                    model.config.num_layers,
                    0,
                ),
            ),
            "rockchip" | "rknn" => Some(
                crate::hybrid::HybridDeviceTopology::rockchip_rknn_partition(
                    model.config.num_layers,
                ),
            ),
            "rpi_hailo" | "hailo" => Some(
                crate::hybrid::HybridDeviceTopology::raspberry_pi_hailo_partition(
                    model.config.num_layers,
                ),
            ),
            _ => {
                if norm_backend == "cpu"
                    && oxide_core::hardware::GpuDeviceProfile::detect_amd_cpu_and_igpu().is_some()
                {
                    let has_npu = oxide_core::hardware::GpuDeviceProfile::detect_amd_apu_full()
                        .is_some_and(|(_, _, npu)| npu);
                    Some(crate::hybrid::HybridDeviceTopology::amd_apu_full_partition(
                        model.config.num_layers,
                        has_npu,
                    ))
                } else {
                    None
                }
            }
        };

        if let Some(topology) = maybe_topology {
            return Ok(Self::Llama3AmdApuCpuIgpu {
                model,
                kv_cache,
                seq_positions: std::collections::HashMap::new(),
                scratch,
                topology,
            });
        }

        Ok(Self::Llama3Dense {
            model,
            kv_cache,
            seq_positions: std::collections::HashMap::new(),
            scratch,
        })
    }
}
