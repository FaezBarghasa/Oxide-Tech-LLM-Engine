# Oxide-Lab: AI Research Laboratory Tooling & Architecture Guide

## 1. Executive Summary & Vision

`oxide-lab` is the high-performance AI research laboratory extension crate for the **Oxide-Tech LLM Engine**. Designed for researchers, deep learning engineers, and computational laboratories, `oxide-lab` delivers native Rust 2024 bare-metal primitives to design, train, fine-tune, quantize, compress, inspect, and synthesize any deep learning model or multi-modal representation without Python runtime overhead, CUDA memory fragmentation, or dynamic allocation bottlenecks.

```
+---------------------------------------------------------------------------------------------------------+
|                                        OXIDE RESEARCH LABORATORY SUITE                                  |
+--------------------+---------------------+--------------------+--------------------+--------------------+
|   Custom Models    |   Zero-Alloc Train  | Quant & Compress   | Real-Time Debug    | Multi-Modal Media  |
+--------------------+---------------------+--------------------+--------------------+--------------------+
| - CustomModel      | - LoraFineTuner     | - Q4_0, Q4_K, Q8_0 | - TensorDebugger   | - Text-to-Image    |
| - CustomModelBuild | - QLoRA 4-bit       | - Ternary 1.58-bit | - DriftDetector    | - Text-to-Video    |
| - Arbitrary Blocks | - AdamW Optimizer   | - Marlin / NvFP4   | - PerplexityAudit  | - Speech (TTS)     |
| - DAG ComputeGraph | - Loss (CE/MSE/DPO) | - 2:4 Pruning      | - Anomaly Probe    | - ASR Transcribe   |
| - Zero-Alloc Step  | - LrScheduler       | - Truncated SVD    | - Representation   | - Universal Inputs |
+--------------------+---------------------+--------------------+--------------------+--------------------+
```

---

## 2. Declarative Custom Model Architecture Synthesis (`custom_model`)

Traditional research frameworks require complex C++/Python binding layers or heavy runtime graph tracers. `oxide-lab` provides a declarative, strictly typed builder pattern (`CustomModelBuilder`) that compiles arbitrary combinations of modern deep learning building blocks into an executable, statically sized model and an `oxide-engine` dynamic DAG (`ComputeGraph`).

### Supported Architecture Blocks (`LayerType`)

| Block Type | Description | Key Research Applications |
| :--- | :--- | :--- |
| `Linear` | Standard dense projection with SIMD GEMV/GEMM | Encoders, decoders, classification heads |
| `Attention` | Multi-head attention (MHA) and Grouped-Query Attention (GQA) | Transformer models (LLaMA 3, Mistral, Gemma) |
| `SwiGluMlp` | Gated feed-forward network with SiLU ($\text{SwiGLU}(x) = (x W_{gate} \cdot \sigma(x W_{gate})) \cdot x W_{up}$) | Contemporary frontier LLMs |
| `RmsNorm` | Root Mean Square Normalization with zero mean-centering overhead | Stable high-throughput residual streams |
| `Moe` | Mixture of Experts with top-$k$ dynamic routing | Mixtral 8x7B, DeepSeek-V3, Qwen-MoE |
| `StateSpaceMamba`| Linear-time selective state space model (SSM) with recurrent state updates | Mamba, Jamba, hybrid recurrent networks |
| `LatentAttentionMla` | DeepSeek Multi-Head Latent Attention with low-rank KV compression | High-throughput long-context decoding |
| `DiffusionBlock` | Latent spatial denoising block with cross-attention and timestep conditioning | Diffusion transformers (DiT, Flux, SDXL) |

### Rust Example: Building a Hybrid Transformer-Mamba Architecture

```rust
use oxide_lab::{CustomModelBuilder, CustomModelScratch, LayerType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Declare custom hybrid research architecture
    let model = CustomModelBuilder::new("oxide-hybrid-ssm", 32000, 1024)
        .set_max_seq_len(8192)
        .add_rmsnorm("embed_norm")
        .add_attention("attn_0", 16, 4)
        .add_swiglu_mlp("mlp_0", 2816)
        .add_mamba_ssm("mamba_1", 64)
        .add_moe("moe_2", 8, 2, 2816)
        .add_latent_attention_mla("mla_3", 512, 16)
        .add_rmsnorm("final_norm")
        .build();

    // 2. Compile to native Oxide DAG compute graph
    let graph = model.compile_to_graph();
    println!("Compiled DAG nodes: {}", graph.nodes.len());

    // 3. Execute zero-allocation forward pass
    let mut scratch = CustomModelScratch::new(1024, 32000);
    let prompt_tokens = [1u32, 856, 1024, 4921];
    let logits = model.forward(&prompt_tokens, &mut scratch)?;

    println!("Logits shape: {} | Argmax token: {}", logits.len(), 
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0);
    Ok(())
}
```

---

## 3. Zero-Allocation Training & Fine-Tuning (`training`)

For resource-constrained laboratories and embedded fine-tuning workstations, `oxide-lab` provides a complete, heap-allocation-free gradient backpropagation and parameter optimization pipeline.

### Core Modules & Primitives

1. **`LoraFineTuner`**:
   - Implements low-rank adapter projection:
     $$\Delta W = \frac{\alpha}{r} (A \cdot B)$$
     where $A \in \mathbb{R}^{r \times d_{in}}$ is Gaussian initialized and $B \in \mathbb{R}^{d_{out} \times r}$ is zero initialized.
   - Evaluates forward adapter contributions directly in pre-allocated scratch buffers.
   - Computes exact analytical backward gradients ($\frac{\partial L}{\partial A}$, $\frac{\partial L}{\partial B}$) without building runtime autograd tape overhead.
2. **`AdamWOptimizer`**:
   - Decoupled weight decay parameter updates:
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
     $$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
   - Full support for gradient norm clipping to prevent gradient explosion during unstable training steps.
3. **`LossComputer`**:
   - **Cross-Entropy**: Numerically stable log-sum-exp with analytical softmax gradient and label smoothing.
   - **Mean Squared Error (MSE)**: Regression and continuous latent alignment loss.
   - **Direct Preference Optimization (DPO)**: Explicit pairwise preference optimization without training a separate reward model:
     $$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right)$$
4. **`LrScheduler`**:
   - Linear warmup followed by smooth cosine decay schedule.

---

## 4. Post-Training Quantization (PTQ) & Compression (`quant_compression`)

Labs can compress and benchmark models using standard and experimental quantization schemes:

### Matrix Compression Spectrum

```
+-----------------------------------------------------------------------------------------+
|                                QUANTIZATION COMPRESSION RATIOS                          |
+-------------------------+-------------+------------------+------------------------------+
| Method                  | Bits/Weight | Compression vs FP32 | Typical Target            |
+-------------------------+-------------+------------------+------------------------------+
| FP32 (Baseline)         | 32.0        | 1.0x             | Reference weights            |
| Q8_0 (INT8 Symmetric)   | 8.0         | 4.0x             | Near-zero perplexity loss    |
| Q4_0 (INT4 Symmetric)   | 4.0         | 8.0x             | Fast edge inference          |
| Q4_K (Superblock INT4)  | 4.5         | 7.1x             | High-precision scale blocks  |
| Marlin Int4 / NvFP4     | 4.0         | 8.0x             | Hardware Tensor Core speed   |
| Ternary 1.58-bit        | 1.58        | 16.0x            | BitNet ternary {-1, 0, +1}   |
| 2:4 Structured Pruning  | 16.0 / 32.0 | 2.0x (Sparse)    | Ampere/Hopper Sparse Cores   |
| Truncated SVD (Rank r)  | Variable    | Configurable     | Parameter reduction          |
+-------------------------+-------------+------------------+------------------------------+
```

### Activation-Aware Weight Quantization (AWQ)

`LabQuantizer::quantize` accepts an optional activation channel calibration slice (`Option<&[f32]>`). When provided, it scales weights by salient channel magnitudes ($W' = W \cdot S^\alpha$) prior to clipping and rounding, preserving 1% most important outlier weights that dictate model reasoning capability.

---

## 5. Real-Time Tensor Debugging & Drift Auditor (`realtime_debug`)

To diagnose numerical instability, vanishing gradients, or representation collapse during training and speculative decoding:

1. **`TensorDebugger`**:
   - In-flight inspection of activation tensors: min, max, mean, standard deviation, and sparsity %.
   - Fast SIMD detection of `NaN`, positive/negative `Infinity`, and denormal values.
   - `assert_safe()` gate to immediately trigger circuit breakers if corrupt values appear in intermediate hidden states.
2. **`DriftDetector`**:
   - Computes cosine similarity and Mean Squared Error between baseline reference tensors and current generation representations.
   - Detects drift when speculative draft models deviate beyond acceptable thresholds from target models.
3. **`PerplexityAuditor`**:
   - Calculates streaming sequence cross-entropy loss and exponentiates to produce per-token perplexity ($PPL = \exp(\text{mean}(\text{CE}))$, providing instant verification of quantization or fine-tuning degradation.

---

## 6. Universal Multi-Modal Ingestion & Generative Media Engine (`multimodal_lab`)

Research laboratories frequently process heterogeneous input signals. `oxide-lab` standardizes universal ingestion across all physical modalities into typed, contiguous buffers.

### Universal Ingestion Types (`MultiModalInput`)

- `TextTokens(Vec<u32>)`: Discrete token IDs.
- `AudioPcm { samples, sample_rate, channels }`: Raw continuous audio waveforms.
- `MelSpectrogram { mels, n_mels, frames }`: Time-frequency representations for audio understanding.
- `ImageRgb { pixels, width, height, channels }`: Normalized floating-point pixel arrays.
- `VideoFrames { frames, width, height, channels, num_frames, fps }`: Spatiotemporal visual tensors.
- `PointCloud3D { points, num_points, features_per_point }`: LiDAR, depth sensor, and spatial coordinate clouds.
- `ContinuousEmbedding { vectors, dim }`: External vector representations from multimodal encoders (CLIP, SigLIP, Whisper).

### Generative Media Synthesis (`MultiModalLabEngine`)

- **Text-to-Image**: Deterministic latent diffusion scheduling with classifier-free guidance emulation.
- **Text-to-Video**: Spatiotemporal latent frame synthesis across coherent motion trajectories.
- **Neural Speech Synthesis (TTS)**: Harmonic pitch F0 synthesis with harmonic formant envelopes.
- **Automatic Speech Recognition (ASR)**: Signal energy profiling and spectral transcription.

---

## 7. Command Line Interface (CLI) Workflows

The `oxide` CLI exposes the full laboratory suite via the `oxide lab` subcommand:

### 1. Fine-Tuning with LoRA
```bash
oxide lab train --model llama3 --rank 16 --alpha 32.0 --steps 100 --lr 1e-4
```

### 2. Quantization & 2:4 Structured Pruning
```bash
# Quantize matrix to Superblock Q4_K
oxide lab quantize --format q4_k --rows 4096 --cols 4096

# Apply 2:4 structured pruning for NVIDIA Sparse Tensor Cores
oxide lab quantize --rows 4096 --cols 4096 --prune-2-4

# Truncated SVD decomposition (Rank 64)
oxide lab quantize --format svd --rows 4096 --cols 4096 --svd-rank 64
```

### 3. Real-Time Tensor Debugging & Drift Telemetry
```bash
oxide lab debug --hidden-dim 1024 --steps 25 --drift-threshold 0.05
```

### 4. Multi-Modal Media Synthesis
```bash
# Generate image
oxide lab generate --modality image --prompt "A quantum processor in deep space" --width 256 --height 256 --steps 20

# Generate video
oxide lab generate --modality video --prompt "Fluid particle dynamics" --width 128 --height 128 --frames 16

# Synthesize speech
oxide lab generate --modality audio --prompt "Oxide engine inference running at peak throughput."
```

### 5. Custom Architecture Inspection
```bash
oxide lab model --name "research-transformer-v1" --hidden-dim 1024 --vocab-size 32000 --layers 6
```

---

## 8. Summary Table: Research Lab Capabilities

| Dimension | Oxide-Lab Capability | Research Advantage |
| :--- | :--- | :--- |
| **Language & Environment** | Pure Rust 2024 (`no_std` compatible core) | Zero Python runtime crash, deterministic memory footprint |
| **Memory Allocation** | Pre-allocated static scratch buffers (`CustomModelScratch`) | Zero heap allocations during forward inference & backward passes |
| **Model Customization** | Modular `LayerType` builder | Seamless prototyping of MHA, GQA, MoE, SSM Mamba, MLA, and Diffusion |
| **Quantization** | Q4_0, Q4_K, Q8_0, Ternary 1.58-bit, AWQ, SVD, 2:4 Sparsity | Direct evaluation of edge compression tradeoffs |
| **Observability** | `TensorDebugger` + `DriftDetector` + `PerplexityAuditor` | Real-time discovery of NaN anomalies, representation drift, and PPL regressions |
| **Multi-Modal** | Universal `MultiModalInput` & `MultiModalLabEngine` | Native multi-modal research across text, audio, images, video, and point clouds |
