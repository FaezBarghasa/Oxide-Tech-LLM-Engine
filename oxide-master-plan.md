# Oxide-Tech-LLM-Engine — Master Phased Task Checklist (`oxide-master-plan.md`)

> Detailed architectural rationale, roofline equations, and kernel designs live in the brain artifact `implementation_plan.md`.
> Execute one phase at a time; mark `[x]` only after running the phase's verification gate.

---

## Phase 1: Correctness First & Zero-Stub Purge (`AGENTS.md` Gate)

**Goal**: Eliminate all silent bugs causing garbled tokens, slot-0 KV corruption, discarded weights, failed-prefill advance, GGUF corruption, and synthetic `.sin()`/`.cos()` hardware/HTTP loops.
**Pre-requisites**: Rust 2024 toolchain (`1.85+`), CUDA toolkit (`nvcc`, `compute-sanitizer`), local GGUF models in `~/models/`.

- [x] **1.1 Fix double `self.seq_pos += 1` in CUDA `dispatch_step_kernel`** — Remove duplicate increment in `crates/oxide-hardware/oxide-backend-cuda/src/lib.rs` (lines 4624–4627 vs 5675–5678) and unify slot position tracking in `slot_positions[slot]`.
  - *Verify*: `cargo test -p oxide-backend-cuda -- --nocapture` (assert slot position increments by exactly `1` per decode step).
- [x] **1.2 Fix single-slot KV overwrite in CUDA non-graph path** — Thread explicit `slot * slot_stride` / paged block offsets into all `kv_k[layer]` / `kv_v[layer]` accesses in `crates/oxide-hardware/oxide-backend-cuda/src/lib.rs` (lines 4898, 5138, 5175).
  - *Verify*: Run concurrent 4-slot decode test comparing slot `1..3` outputs vs isolated single-slot run (exact token match).
- [x] **1.3 Propagate prefill errors & stop silent advance in `ContinuousBatchEngine`** — Fix `crates/oxide-engine/src/continuous_batching.rs` (lines 538–555) so `pipe.prefill(...)` `Err(e)` transitions the slot to failed state and frees resources instead of advancing `uncomputed_start`.
  - *Verify*: Unit test injecting a failing backend prefill error verifies slot aborts cleanly and does not enter `SequencePhase::Decode`.
- [x] **1.4 Pass weights to all non-CUDA pipelines & fix APU partition routing** — Update `crates/oxide-engine/src/pipeline.rs` (lines 1227–1257 and `Llama3AmdApuCpuIgpu` at lines 438–446) so every backend receives `model` weights cleanly or fails fast with `Err(EngineError::DeviceNotFound)` / `UnsupportedCapability` when absent.
  - *Verify*: `cargo check --workspace --all-targets`
- [x] **1.5 Purge all synthetic 128-dim `.sin()`/`.cos()` & CPU fake loops** — Replace stubs with real hardware probe or `Err(EngineError::DeviceNotFound)` across `oxide-backend-cuda/src/megakernel.rs` (757–1107), `oxide-backend-rocm/src/lib.rs` (150–186), `oxide-backend-{metal,intel,tpu,qualcomm,rknn,hailo}/src/kernels.rs`, and `oxide-server/src/lib.rs` (832–947).
  - *Verify*: `rg -n "\.sin\(\)|\.cos\(\)|HIDDEN_DIM: usize = 128" crates/` returns zero matches in inference/kernel/server handlers (RoPE math excluded).
- [x] **1.6 Fix GGUF quant enum conflict & eliminate `_ => F32` silent cast** — Map canonical GGML codes `0..=39` in `crates/oxide-models/src/formats.rs` (22–43, 277–298), relocate custom research quants (`PTQ1_0`, `PQ2_0`, `Nvfp4`) to `>= 200` + metadata key, and return `Err(EngineError::UnsupportedCapability)` for unknown IDs.
  - *Verify*: `cargo test -p oxide-models formats` verifying `Q8_K`, `IQ2_XXS..IQ4_XS`, `TQ1_0`, `TQ2_0`, `MXFP4`, `BF16` parse accurately.
- [x] **1.7 Expand `HardwareBackend` trait & dynamic stop-token extraction** — Extend `crates/oxide-core/src/traits.rs` (`probe_devices`, `upload_tensor`, `alloc_kv_pages`, `dispatch_prefill_batch`, `dispatch_decode_batch`, `dispatch_pool_embeddings`) and replace hardcoded EOS list (`continuous_batching.rs:596-602`) with GGUF/tokenizer metadata stop IDs.
  - *Verify*: `cargo clippy --workspace --all-targets -- -D warnings`

**Phase 1 Done Criteria**: Zero clippy warnings; `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf` and `gemma-4-e2b-it.Q8_0.gguf` produce coherent multi-turn text across slots `0..7` on CUDA; absent hardware backends return typed `DeviceNotFound`.

---

## Phase 2: NVIDIA CUDA Kernels & Graph Runtime (Beat vLLM & SGLang)

**Goal**: Achieve `0` register spills, $\ge 88\%$ RTX 4060 DRAM bandwidth on $B=1$ decode, $\ge 1.15\times$ vLLM/SGLang batched throughput ($B=32$), and sub-2µs CUDA Graph launches across all quants & samplers.
**Pre-requisites**: Phase 1 passing; `ncu` & `compute-sanitizer` available.

- [x] **2.1 Implement LUT-Free In-Register MMVQ kernels (`llm_kernels.cu`)** — Add vectorized 128-bit `uint4` loads (`ld.global.cs.v4.u32`) + 1-cycle `lop3.b32`/`prmt.b32` nibble unpack directly into FP16x2/`__dp4a` accumulators across all GGML quants (`Q4_0..Q8_0`, `Q2_K..Q8_K`, `IQ*`, `TQ*`, `MXFP4`), 4 rows/warp.
  - *Verify*: `nvcc --ptxas-options=-v` shows `0 bytes spill stores, 0 bytes spill loads`; golden logit cosine similarity $\ge 0.9995$ vs CPU `f32`.
- [x] **2.2 Implement Asynchronous Tensor Core `mma.sync` batched GEMM ($M > 8$ & prefill)** — Add SM80+ `cp.async.cg` 3-stage pipeline + `mma.sync.aligned.m16n8k16` with register-level weight dequantization in `crates/oxide-hardware/oxide-backend-cuda/src/kernels/`.
  - *Verify*: Benchmark $M = 16, 64, 512$ prefill projections vs `cublasHgemm` + dequant overhead ($\ge 2.5\times$ speedup over MMVQ).
- [x] **2.3 Implement FlashInfer-style Paged GQA Split-K Decode + Cascade Prefix** — Upgrade `split_k_flash_decode_continuous_batch_kernel` (`llm_kernels.cu`) to pack $G = N_q / N_{kv}$ heads sharing 1 KV load from `PagedKvArena` (`TOKENS_PER_PAGE = 16`) via `d_block_table`, add cascade shared-prefix stage, and optional FP8/Q8_0 KV dequant.
  - *Verify*: `compute-sanitizer --tool memcheck` clean across context lengths `512, 4096, 16384` at `B = 1..32`.
- [x] **2.4 Unlock 100%-Quant CUDA Graph Bucket Pool + In-Graph Sampling** — Remove `token_embd_quant_type == 1` & `d_per_layer_emb.is_none()` guards (`lib.rs:3240-3243`); pre-capture batch buckets `{1,2,4,8,16,32,64,128}` including GGUF/PLE embeddings and in-graph Philox-4x32-10 Top-K/Top-P/Min-P/Grammar bitmask sampling (`flash_sample_kernel`).
  - *Verify*: Capture & replay multi-step decode on both `DeepSeek-R1` (`Q4_K_M`) and `gemma-4-e2b-it` (`Q8_0` with PLE) with `cudaStreamSynchronize` called only at token commit.
- [x] **2.5 Build Real GPU Persistent Cooperative Megakernel (`megakernel.rs`)** — Replace CPU fake loop with real `cudaLaunchCooperativeKernel` fusing `RMSNorm -> QKV+RoPE -> PagedAttn -> OProj+Res -> RMSNorm -> GateUp+SwiGLU -> Down+Res` for low-latency small-batch decode.
  - *Verify*: Kernel parity matches step-by-step CUDA Graph logits ($\Delta \le 10^{-3}$); profiler confirms single grid launch per layer group.
- [x] **2.6 Implement Fused GPU MoE Grouped MMA + Atomic Bucketing** — Fuse `topk_softmax`/`sigmoid` routing, atomic token-to-expert permutation, grouped quantized MMA, shared-expert overlap, and unpermute in CUDA for `DeepSeek-MoE` / `Qwen3-MoE` / `Mixtral`.
  - *Verify*: Single MoE layer latency beats sequential per-expert dispatch by $\ge 4\times$.
- [x] **2.7 Wire Real `PagedKvArena` + `RadixPrefixCache` GPU Mirror** — Replace dummy `&[slot_id as u32]` insertion (`continuous_batching.rs:545-546`) with 16-token chunk hashing storing ordered physical page IDs, async `d_block_table` updates on `h2d_copy_stream`, and CoW `fork_slot_pages`.
  - *Verify*: Multi-request shared system prompt achieves $>95\%$ prefix cache hit rate and $\ge 3\times$ TTFT reduction.

**Phase 2 Done Criteria**: Live RTX 4060 `ncu` confirms $\ge 88\%$ DRAM peak BW on $B=1$ Q4_K/Q8_0, zero `compute-sanitizer` errors, and `oxide-lab` beats `llama.cpp` & `vLLM` on local throughput benchmarks.

**Phase 2 Done Criteria**: Live RTX 4060 `ncu` confirms $\ge 88\%$ DRAM peak BW on $B=1$ Q4_K/Q8_0, zero `compute-sanitizer` errors, and `oxide-lab` beats `llama.cpp` & `vLLM` on local throughput benchmarks.

---

## Phase 3: AMD ROCm HIP + AOT Triton HSACO + Ryzen AI XDNA

**Goal**: Deliver genuine AMD GPU (CDNA2/3/4, RDNA3/4) and APU SVM/XDNA execution adhering strictly to Wave64/Wave32 and ROCm Triton constraints.
**Pre-requisites**: Phase 1 passing; `libloading` in workspace dependencies.

- [x] **3.1 Implement `RocmDriver` (`libamdhip64.so` / `libhiprtc.so` / `hipBLASLt`)** — Build real runtime FFI loader, coarse/fine-grained SVM `hipHostMalloc`, async streams, events, and `hipGraph` capture in `crates/oxide-hardware/oxide-backend-rocm/src/driver.rs`.
  - *Verify*: `cargo test -p oxide-backend-rocm` probes live AMD GPU cleanly or returns `Err(EngineError::DeviceNotFound)` on NVIDIA-only hosts.
- [x] **3.2 Port CUDA MMVQ, Paged GQA & Sampling Kernels to Wave-Agnostic HIP** — Create shared `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` kernels replacing `32` with `__AMDGCN_WAVEFRONT_SIZE`, `uint64_t` ballot masks, `v_dot4_i32_i8`/`v_mfma_f32_16x16x16` (CDNA), `wmma` (RDNA3/4), CDNA3 FNUZ FP8 vs CDNA4 OCP FP8 + 32-XCD PID swizzle, and RDNA4 256B cacheline alignment.
  - *Verify*: If `hipcc` installed, offline compile to `.hsaco` succeeds cleanly for `--offload-arch=gfx942,gfx950,gfx1100,gfx1201` with zero warnings.
- [x] **3.3 Add AOT Triton `.hsaco` Compiler & Rust Loader (`tools/triton_aot/`)** — Generate `.hsaco` binaries + JSON launch specs for FlashPrefill, PagedDecode, RMSNorm, RoPE (`pid_s % seq_len`), and Fused MoE with `TRITON_HIP_USE_BLOCK_PINGPONG=1` & `TRITON_HIP_USE_ASYNC_COPY=1`.
  - *Verify*: CI static regex check confirms ZERO `tl.math.tanh`, ZERO `tl.libdevice`, ZERO `break`/`continue`, ZERO RMSNorm `BLOCK_D` autotune.
- [x] **3.4 Enable AMD APU Unified Memory (`Llama3AmdApuCpuIgpu`) & XDNA NPU** — Bind zero-copy HSA coherent buffers across CPU+iGPU in `oxide-engine/src/pipeline.rs` and add `libxrt_coreutil.so` / `amdxdna` IOCTL dispatch in `oxide-backend-rocm`.
  - *Verify*: Zero-copy buffer test passes pointer equality across CPU & HIP host-device mapping when ROCm SVM is available.

**Phase 3 Done Criteria**: `oxide-backend-rocm` compiles clean `.hsaco` kernels (where SDK present) or probes cleanly (`DeviceNotFound`), passes all banned-Triton static checks, and has zero synthetic code.

**Phase 3 Done Criteria**: `oxide-backend-rocm` compiles clean `.hsaco` kernels (where SDK present) or probes cleanly (`DeviceNotFound`), passes all banned-Triton static checks, and has zero synthetic code.

---

## Phase 4: Universal Hardware Ecosystem (Zero-Stub All 10 Targets + Hybrid Split)

**Goal**: Wire every remaining hardware crate (`Metal`, `Intel`, `TPU`, `Qualcomm`, `RKNN`, `Hailo`, `CPU`, `Vulkan`) to genuine OS/vendor drivers or clean errors, plus true heterogeneous CPU+GPU split execution.
**Pre-requisites**: Phase 1 passing; `libloading` and `ash` added to root `Cargo.toml`.

- [x] **4.1 Apple Silicon Metal Backend (`oxide-backend-metal`)** — Under `cfg(target_os = "macos")`, wire real `MTLDevice`, zero-copy `.storageModeShared` `MTLBuffer`, `.metallib` `simdgroup_matrix` MMA/PagedAttention shaders, and `MTLIndirectCommandBuffer`; return `DeviceNotFound` on non-macOS hosts.
  - *Verify*: `cargo test -p oxide-backend-metal` passes probe & compile cleanly on Linux (`DeviceNotFound`) and hardware dispatch on macOS.
- [x] **4.2 Intel Arc/PVC Level Zero + OpenVINO NPU + Xeon AMX (`oxide-backend-intel`)** — Wire `libze_loader.so` SPIR-V DPAS/ESIMD kernels for Intel GPUs, OpenVINO NPU C-API (`intel_vpu`), and `x86_64` `cpuid` + `syscall(ARCH_REQ_XCOMP_PERM)` AMX tile `tdpbf16ps` kernels.
  - *Verify*: CPU AMX parity test vs `f32` (on AMX-capable CPU) or `UnsupportedCapability`; Level Zero runtime probe clean.
- [x] **4.3 Google Cloud TPU PJRT & Coral EdgeTPU (`oxide-backend-tpu`)** — Wire `libtpu.so` `PJRT` C-API (`PJRT_Client_Create`, StableHLO MXU compilation/execution) and `libedgetpu.so` delegate.
  - *Verify*: Clean dynamic symbol resolution & `DeviceNotFound` without `libtpu.so`.
- [x] **4.4 Qualcomm FastRPC/QNN + Universal Vulkan Compute (`oxide-backend-qualcomm`)** — Implement `libcdsprpc.so` / `libQnnHtp.so` DSP/HTP execution AND build a real `ash` **Vulkan Compute (`VK_KHR_cooperative_matrix` / `VK_KHR_16bit_storage`) SPIR-V** backend supporting Adreno, Mali, PowerVR, and cross-vendor GPUs.
  - *Verify*: Run live Vulkan Compute SPIR-V matrix & RMSNorm kernels on local Linux Vulkan ICD (RTX 4060 / RADV / Mesa) and verify output parity vs CPU `f32` ($\Delta \le 10^{-3}$).
- [x] **4.5 Rockchip RKNN + Ethos-U & Hailo-8/10/15 Backends (`oxide-backend-rknn`, `oxide-backend-hailo`)** — Wire `librknnrt.so` (`rknn_init`, `rknn_create_mem_from_fd`, `rknn_run`) + `/dev/ethosu*` IOCTL and `libhailort.so` PCIe VStream pipelines.
  - *Verify*: ABI header checks & clean `DeviceNotFound` on non-NPU x86 hosts.
- [x] **4.6 Complete CPU AVX-512 / NEON / SVE2 SIMD Matrix (`oxide-quant/src/simd.rs`, `oxide-backend-cpu`)** — Implement block-tiled MT kernels across all 39 GGML quants using AVX-512 VNNI/BF16, AVX2 FMA, and ARM NEON DotProd/I8MM pinned to NUMA cores via `oxide_core::ThreadTeam`.
  - *Verify*: Property tests (`proptest`) validate every SIMD quant dot product against scalar `f32` within expected ULP/quant tolerances.
- [x] **4.7 True Heterogeneous CPU+GPU+NPU Split Execution (`oxide-engine/src/pipeline.rs`)** — Execute `HybridDeviceTopology` GPU layers on device backend and CPU/MoE cold experts on AVX-512/AMX with async pinned staging overlapping PCIe transfers.
  - *Verify*: Run a split `Llama3` / `DeepSeek` layer partition (50% CUDA / 50% CPU) end-to-end and verify coherent generation matching 100%-CUDA output.

**Phase 4 Done Criteria**: Zero hardware stubs remain across all 17 crates; Vulkan SPIR-V & Hybrid CPU+CUDA split run real weights cleanly on the local test machine.

**Phase 4 Done Criteria**: Zero hardware stubs remain across all 17 crates; Vulkan SPIR-V & Hybrid CPU+CUDA split run real weights cleanly on the local test machine.

---

## Phase 5: Lock-Free Overlap Scheduler, Versatility & Server Excellence

**Goal**: Complete end-to-end continuous batching overlap, SafeTensors/MLA/DeltaNet/VLM versatility, EAGLE-3 speculative decoding, XGrammar DFA, and real OpenAI/Anthropic/Embeddings/Rerank endpoints.
**Pre-requisites**: Phases 1–4 complete.

- [ ] **5.1 Replace `try_lock()` 200µs Poll with Lock-Free `rtrb` Overlap Scheduler (`continuous_batching.rs`)** — Move GPU dispatch to a dedicated core-pinned worker thread fed by `rtrb` SPSC rings; overlap step $N$ GPU execution with step $N+1$ batch packing, XGrammar bitmask computation, radix updates, and step $N-1$ detokenization.
  - *Verify*: Zero mutex lock stalls under 64 concurrent SSE client streams; P99 scheduling jitter $<50\ \mu\text{s}$.
- [ ] **5.2 Implement Unified Ragged Chunked Prefill + Decode Mixing** — Batch active chunked-prefill tokens and decode tokens together into one ragged `BatchExecDescriptor` bounded by configurable token budget (`max_num_batched_tokens`) so prefill never starves decode ITL.
  - *Verify*: 8k-prompt submission while 16 decode clients stream maintains decode ITL within SLA ($<1.3\times$ isolated baseline).
- [ ] **5.3 Complete SafeTensors Sharded Loader + EXL2/GPTQ/AWQ/FP8/NVFP4** — Load HuggingFace sharded `.safetensors` zero-copy with automatic Marlin/LUT-free repacking, plus full `cskv` hybrid/MLA cache serialization.
  - *Verify*: Load SafeTensors BF16/FP8 & GPTQ-INT4 model checkpoints cleanly and match HF reference logits.
- [ ] **5.4 Complete MLA Absorption, MoE Cold Offload, Gated DeltaNet & VLM** — Verify DeepSeek MLA `W_UK`/`W_UV` latent KV compression, KTransformers-style hybrid GPU-hot / DDR5-cold MoE routing, GPU Gated DeltaNet recurrence, and SigLIP/Qwen2.5-VL image patch tokenization.
  - *Verify*: End-to-end inference tests across Dense (`gemma-4`), MLA+MoE (`DeepSeek-R1`), and multimodal image input pass golden assertions.
- [ ] **5.5 GPU Tree Speculative Decoding (`speculative.rs`) & Zero-Overhead XGrammar DFA (`oxide-core::dfa`)** — Implement EAGLE-2/3, MTP, and N-gram lookup tree verification on GPU with atomic `PagedKvArena` tail rollback + overlap-computed JSON/CFG vocabulary bitmasking.
  - *Verify*: `>= 1.8x` decode tok/s speedup via speculative verification on coding/reasoning prompts; 100% JSON schema validity at `<2%` throughput overhead.
- [ ] **5.6 Finalize Production Server, Real Embeddings/Rerank/Vision & `oxide-lab` Benchmark Suite (`oxide-server`, `oxide-lab`)** — Eliminate duplicate `LeasedSlotGuard` (`oxide-server/src/lib.rs:429-466`), serve real pooled embeddings (`/v1/embeddings`), cross-encoder `/v1/rerank`, real VLM `/v1/vision/detect`, Prometheus `/metrics`, and automated `llama.cpp` / `vLLM` / `SGLang` comparison suite.
  - *Verify*: Full ShareGPT stress benchmark comparing TTFT, P99 ITL, and tok/s against `llama.cpp` / `vLLM` / `SGLang` produces reproducible JSON/HTML report confirming target SLA wins.

**Phase 5 Done Criteria**: Full workspace release build (`cargo build --release --workspace`), all end-to-end GGUF/SafeTensors/HTTP/Prometheus/benchmark checks green with reproducible superiority metrics over `vLLM`, `SGLang`, and `llama.cpp`.

---

## Optional SurrealDB Tracker Record (`/phase-breakdown-planner`)

If tracking execution in SurrealDB (`oxide_phases` namespace), persist each phase with dependencies and commands:

```surql
-- Optional SurrealDB phase tracking schema
DEFINE TABLE execution_phase SCHEMAFULL;
DEFINE FIELD project ON execution_phase TYPE string;
DEFINE FIELD phase_num ON execution_phase TYPE int;
DEFINE FIELD title ON execution_phase TYPE string;
DEFINE FIELD status ON execution_phase TYPE string DEFAULT 'pending';
DEFINE FIELD verify_cmd ON execution_phase TYPE string;
DEFINE FIELD done_criteria ON execution_phase TYPE string;

CREATE execution_phase:p1 SET project = 'Oxide-Tech-LLM-Engine', phase_num = 1,
  title = 'Correctness First & Zero-Stub Purge', status = 'pending',
  verify_cmd = 'cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace',
  done_criteria = 'Zero seq_pos double-inc, zero slot-0 KV overwrite, zero synthetic stubs, coherent multi-slot GGUF decode';
CREATE execution_phase:p2 SET project = 'Oxide-Tech-LLM-Engine', phase_num = 2,
  title = 'Beat-SGLang/vLLM CUDA Kernels & Graph Runtime', status = 'pending',
  verify_cmd = 'compute-sanitizer --tool memcheck cargo test -p oxide-backend-cuda --release',
  done_criteria = '0 register spills, >=88% DRAM BW B=1, >=1.15x vLLM/SGLang batched throughput';
CREATE execution_phase:p3 SET project = 'Oxide-Tech-LLM-Engine', phase_num = 3,
  title = 'AMD ROCm HIP + AOT Triton HSACO + XDNA', status = 'pending',
  verify_cmd = 'cargo check -p oxide-backend-rocm && ./scripts/audit_triton_rocm.sh',
  done_criteria = 'Wave64/Wave32 HIP + AOT .hsaco pipeline, zero banned Triton constructs, APU SVM unified buffer';
CREATE execution_phase:p4 SET project = 'Oxide-Tech-LLM-Engine', phase_num = 4,
  title = 'Universal Hardware Backends & Hybrid Split', status = 'pending',
  verify_cmd = 'cargo test -p oxide-backend-qualcomm -p oxide-quant --release',
  done_criteria = 'Zero stubs across all 10 hardware crates, live Vulkan SPIR-V & Hybrid CPU+CUDA parity passed';
CREATE execution_phase:p5 SET project = 'Oxide-Tech-LLM-Engine', phase_num = 5,
  title = 'Lock-Free Overlap Engine, Versatility & Benchmark Gate', status = 'pending',
  verify_cmd = 'cargo run --release -p oxide-lab -- bench-compare --model ~/models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf',
  done_criteria = 'rtrb overlap scheduler, chunked prefill, 39 quants, EAGLE-3 + XGrammar, beats vLLM/SGLang/llama.cpp';
```
