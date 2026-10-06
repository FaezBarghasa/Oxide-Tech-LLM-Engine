# Hardware Backends & Silicon Acceleration Matrix

## Overview

`Oxide-Tech-LLM-Engine` provides dedicated, bare-metal hardware backend crates in `crates/oxide-hardware/`. Each backend implements native kernel dispatch, zero-copy memory mapping, and hardware-specific synchronization mechanisms.

---

## 1. NVIDIA CUDA Backend (`oxide-backend-cuda`)

- **Target Architectures**: Blackwell (GB200 / B200 / B100), Hopper (H100 / H200), Ada Lovelace (RTX 4090 / L40S / RTX 4060 Mobile), Ampere (A100 / RTX 3090), Turing (RTX 2080 Ti).
- **Cluster Array Scaling (1 to 16 NVIDIA GPUs)**:
  - `CudaDeviceClusterArray`: Clustered execution topology orchestrating from a single card up to an array of 16 NVIDIA GPUs (`num_gpus: 1..=16`).
  - **NCCL Distributed Collective Communication**: Non-blocking `all_reduce_f32`, `all_gather_f32`, and in-place `all_reduce_slice` scaling across NVLink meshes and PCIe switches.
  - **Pipeline Parallelism Layer Splitting**: Automatic even and remainder partition allocation (`layer_partition_for_gpu`) distributing $L$ transformer layers across $N \le 16$ GPU ranks.
  - **Tensor Parallelism**: Distributed column/row projection splitting with AllReduce reduction over NVLink/PCIe ring topologies.
- **In-Shared-Memory FWHT**: 7-stage unrolled Fast Walsh-Hadamard Transform running inside warp-shuffle registers and L1 Shared Memory without global VRAM accesses.
- **Branchless Ternary GEMV**: `__dp4a` hardware integer dot-product lowering with group-128 FP16 scaling.
- **NVFP4 Tensor Cores**: Support for Blackwell 4-bit floating-point (E2M1) format.
- **CUDA Graphs (`CudaGraphManager`)**: Sub-microsecond forward decode kernel dispatch via captured and replayed execution graphs.
- **P2P NVLink DMA**: Direct peer-to-peer memory transfers via `cudaMemcpyPeerAsync`.

---

## 2. AMD ROCm & APU Backend (`oxide-backend-rocm`)

- **Target Architectures**: Instinct MI350X / MI355X (CDNA4), Instinct MI300X / MI325X (CDNA3), Radeon RX 7900 XTX / RX 7900 XT (RDNA3), Ryzen 7000/8000/9000 & Ryzen AI 300 / Strix Point / Strix Halo APUs.
- **Tri-Compute APU Co-Processing (`DeviceRole::Cpu` + `DeviceRole::Igpu` + `DeviceRole::Npu`)**:
  - Unified coherent DDR5 / LPDDR5X memory partitioning with zero PCIe transfer penalties.
  - Tripartite layer distribution: NPU handles dense GEMM/MLP layers, RDNA iGPU computes multi-head attention projections, and Zen CPU executes AVX2/AVX-512 prefill and head sampling.
  - Direct CLI support via `--backend apu` and `--backend cpu-igpu`.
- **Matrix Core MFMA**: High-throughput Matrix Fused Multiply-Add instructions.
- **APU Zero-Copy Unified Memory**: Direct CPU/GPU memory sharing on AMD APUs without PCIe transfer overhead.
- **HSA Signal Polling**: Asynchronous queue doorbell submission with low-overhead event completion querying.

---

## 3. Apple Silicon Metal Backend (`oxide-backend-metal`)

- **Target Architectures**: Apple M4 Max, M4 Pro, M3 Max, M2 Ultra, M1 Pro.
- **Unified Memory Architecture (UMA)**: CPU and GPU share physical LPDDR5X DRAM. Host token writing and Metal kernel execution operate directly on the same physical memory pointers.
- **Metal Shading Language (MSL) Shaders**: Threadgroup memory optimizations for FWHT butterfly networks and SIMD-group matrix multiplications (`simdgroup_matrix`).
- **Engram Table Direct Reads**: Unified memory placement of the 70.8M Engram parameter table without CPU-GPU synchronization stalls.

---

## 4. Google TPU Backend (`oxide-backend-tpu`)

- **Target Architectures**: TPU v6e (Trillium), TPU v5p, TPU v5e, TPU v4, Google Edge TPU (Coral USB / M.2 Accelerator).
- **Matrix Multiply Units (MXU)**: Systolic array acceleration for dense and ternary quantized projections.
- **Inter-Chip Interconnect (ICI)**: High-speed ring and torus collective fabrics enabling in-kernel direct AllReduce for tensor parallelism.

---

## 5. Intel Arc & Xeon Backend (`oxide-backend-intel`)

- **Target Architectures**:
  - **Discrete GPUs**: Intel Arc B580 / B570 (Battlemage / Xe2-HPG), Intel Arc A770 / A750 (Alchemist / Xe-HPG), Data Center GPU Max (Ponte Vecchio).
  - **Server CPUs**: Intel Xeon 6980P / 6780E (Granite Rapids / Sierra Forest), Intel Xeon Max Series (with on-package HBM2e).
- **XMX (Xe Matrix Extensions)**: Hardware matrix engines accelerating quantized GEMV and systolic multiplication.
- **AMX (Advanced Matrix Extensions)**: Tile Matrix Multiply (TMM) instructions on Xeon processors for INT8 and BF16 acceleration.
- **OneAPI Level-Zero Driver**: Direct low-level command queue submission and event signaling.

---

## 6. Qualcomm Snapdragon Backend (`oxide-backend-qualcomm`)

- **Target Architectures**: Snapdragon X Elite (X1E-84-100, X1E-80-100), Snapdragon X Plus (X1P-64-100), Snapdragon 8 Elite (Gen 4 Mobile Platform).
- **Hexagon Tensor Processor (HTP)**: 45 TOPS dedicated NPU offloading for micro-quantized neural weights.
- **FastRPC DMA Memory**: Zero-copy shared memory buffers between Kryo/Oryon CPU cores and Hexagon HTP vector extensions.

---

## 7. Rockchip RKNN Backend (`oxide-backend-rknn`)

- **Target Architectures**: Rockchip RK3588, RK3588S, RK3576, Orange Pi 6 Plus, Radxa Rock 5B.
- **Tri-Core NPU (6.0 TOPS)**: Multi-core asynchronous NPU task dispatch supporting INT4, INT8, and FP16 operations.
- **DMA-BUF Memory Sharing**: Direct kernel memory buffer sharing between video decoders, cameras, and neural acceleration cores.

---

## 8. Raspberry Pi & Hailo Backend (`oxide-backend-hailo`)

- **Target Architectures**: Raspberry Pi 5 with AI HAT+ (13 TOPS / Hailo-8L), Raspberry Pi 5 with AI HAT+ 2 (26 TOPS / Hailo-8), M.2 / PCIe Hailo accelerators.
- **Dataflow Processing Architecture**: Native dataflow execution with low power draw ($< 2.5\text{W}$).
- **PCIe Direct Circular Ringbuffers**: Wait-free streaming of activation frames directly over the Raspberry Pi 5 PCIe Gen 2/3 interface.

---

## 9. Vectorized & Multithreaded CPU Backend (`oxide-backend-cpu` & `oxide-quant::simd`)

- **Full Core & Thread Saturation (8 to 128 Cores)**:
  - `CpuThreadPool`: NUMA-aware, core-pinned worker thread pool using `core_affinity` to bind threads directly to physical hardware CPU cores and logical hardware threads.
  - Multi-threaded Cache-Blocked GEMV (`gemv_blocked_f32`): Partitions matrix rows across all available CPU cores and threads. Each thread executes vector dot products keeping the activation vector resident in L1D/L2 cache while streaming matrix rows.
  - Multi-threaded Quantized GEMVs: Parallelized execution for `gemv_q8_0` (AVX-512 VNNI / AVX2 FMA), `gemv_q4_0` (AVX-512BW nibble unpacking), and `gemv_q4_k` (AVX-512 / AVX2 super-block accumulation).
  - Multi-threaded CPU FlashAttention-2 / FlashDecode (`flash_attention_cpu`): Parallelizes query head computation across all threads with online, numerically-stable Softmax (Dao et al. / Milakov & Gimelshein).
- **Intel AMX (Advanced Matrix Extensions)**:
  - Hardware 1KB 2D tile registers (`TMM0`..`TMM7`) on Intel Xeon Sapphire Rapids, Emerald Rapids, and Granite Rapids.
  - OS XTILE context initialization via Linux `arch_prctl(ARCH_REQ_XCOMP_PERM)`.
  - Zero-allocation direct inline assembly instructions: `tileloadd`, `tdpbusd` (INT8 matrix multiplication), `tilestored`, and `tilerelease`.
- **x86_64 AVX-512 & AVX2 / FMA**:
  - `_mm512_dpbusd_epi32` and `_mm256_dpbusd_epi32` integer vector dot products for branchless quantized inference.
  - AVX2 FMA & AVX-512 FMA parallel float conversions with dual-accumulator unrolling.
  - Vectorized RMSNorm (`rmsnorm_f32`) and Rotary Position Embedding (`rope_f32`).
- **ARM Neon / SVE2**:
  - `vdotq_s32` vector dot products on ARM64 processors.
- **Cacheline Isolation**:
  - Thread context structures aligned to 64 bytes (`#[repr(C, align(64))]`) to prevent false sharing and L1/L2 cacheline bouncing across CPU cores.

---

## 10. AMD EPYC Server CPU High-Core Substrate (8 to 128 Cores)

- **Target Processors**: AMD EPYC 9004 / 9005 series (e.g., EPYC 9754 128-core, 9654 96-core, 9554 64-core, 9354 32-core, 9124 16-core, 9004 8-core).
- **NUMA & Memory Bandwidth**:
  - 12-channel DDR5-4800 / DDR5-6000 memory controllers delivering up to 460.8 GB/s sustained memory bandwidth per socket.
  - Dual-socket (2S) and single-socket (1S) NUMA domain isolation (`HardwareFormFactor::AmdEpycServerSocket`).
- **SIMD Vector Engine**: AVX-512 dual 256/512-bit VNNI execution (`TensorCoreGeneration::AmdAvx512Vnni`) for fast INT8/BF16/FP8 matrix-vector operations.
- **Infinity Fabric Interconnect**: 32–64 Gbps coherent interconnect for inter-socket tensor all-reduce and NUMA paging.

---

## 11. Heterogeneous Hardware Combination Matrix

`Oxide-Tech-LLM-Engine` provides native topology construction and pipeline orchestration for arbitrary combinations of silicon:

| Hardware Combination | Topology Constructor | Execution Mechanism |
| :--- | :--- | :--- |
| **CPU + NVIDIA GPUs** | `multi_vendor_gpu_partition(L, N, 0, 0, C)` | CUDA Tensor Cores + CPU SIMD offloading |
| **CPU + AMD GPUs** | `multi_vendor_gpu_partition(L, 0, A, 0, C)` | ROCm MFMA Matrix Cores + CPU SIMD offloading |
| **CPU + Intel GPUs** | `multi_vendor_gpu_partition(L, 0, 0, I, C)` | Intel Xe2/Xe1 XMX matrix engines + CPU SIMD |
| **CPU + Google TPU** | `HybridDeviceTopology` (`DeviceRole::Tpu`) | Cloud/Edge TPU systolic MXU + Host prefill |
| **CPU + NPU** | `HybridDeviceTopology` (`DeviceRole::Npu`) | Intel NPU / AMD XDNA tile systolic array |
| **CPU + NVIDIA + AMD + Intel** | `multi_vendor_gpu_partition(L, N, A, I, C)` | Triple-vendor dGPU array with AllReduce reduction |
| **CPU + AMD + Intel** | `multi_vendor_gpu_partition(L, 0, A, I, C)` | ROCm + Xe multi-vendor dGPU co-processing |
| **CPU + NVIDIA + Intel** | `multi_vendor_gpu_partition(L, N, 0, I, C)` | CUDA + Xe multi-vendor dGPU co-processing |
| **CPU + NVIDIA + AMD** | `multi_vendor_gpu_partition(L, N, A, 0, C)` | CUDA + ROCm cross-vendor dGPU co-processing |
| **CPU + iGPU + NPU (APU)** | `amd_apu_full_partition(L, true)` | Coherent 3-way unified DDR5 zero-copy memory |
| **CPU + iGPU + TPU** | `HybridDeviceTopology` (`Igpu` + `Tpu` + `Cpu`) | Integrated GPU attention + TPU systolic GEMM |
| **CPU + iGPU + NPU + NVIDIA** | `HybridDeviceTopology` (`Igpu` + `Npu` + `NvidiaGpu`) | APU local layers + discrete NVIDIA CUDA offload |
| **CPU + iGPU + NPU + AMD dGPU**| `HybridDeviceTopology` (`Igpu` + `Npu` + `AmdGpu`) | APU local layers + discrete AMD Radeon/Instinct |
| **ARM CPU + Integrated NPU + HAT** | `arm_npu_hat_partition(L, true)` | SoC NPU (Apple/RKNN/HTP) + PCIe/USB NPU HAT (Hailo/Coral) |
| **ARM CPU + Integrated NPU** | `arm_npu_hat_partition(L, false)` | Direct ARM SoC NPU + Neon SIMD offloading |
| **AMD EPYC (8 to 128 cores)** | `epyc_server_partition(L, sockets, &[])` | 12-channel DDR5 multi-socket AVX-512 VNNI scaling |
| **AMD EPYC + Any GPUs** | `epyc_server_partition(L, sockets, gpus)` | High-memory bandwidth server CPU + arbitrary dGPUs |

