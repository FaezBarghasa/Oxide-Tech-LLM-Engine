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

## 9. Vectorized CPU Backend (`oxide-backend-cpu`)

- **x86_64 AVX-512**:
  - `vpdpbusd` integer vector dot products for branchless ternary matrix-vector multiplication.
  - AVX-512 FMA parallel float conversion.
- **ARM Neon / SVE2**:
  - `vdotq_s32` vector dot products on ARM64 processors.
- **Cacheline Isolation**: Thread context structures aligned to 64 bytes (`#[repr(C, align(64))]`) to prevent L1/L2 cacheline bouncing across CPU cores.
