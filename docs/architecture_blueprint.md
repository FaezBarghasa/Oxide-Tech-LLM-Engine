# Architecture Blueprint & Deep Technical Specification

## Architectural Charter & Immutable Non-Negotiable Invariants

The **Oxide-Tech-LLM-Engine** (`oxide-engine`) eliminates the abstraction tax, scheduler jitter, and memory fragmentation inherent in generalized inference runtimes (such as standard vLLM and llama.cpp runtimes) while providing bare-metal hardware saturation for non-standard, emerging neural paradigms.

Every architectural stratum, kernel implementation, and subsystem across all developmental horizons enforces five fundamental invariants:

1. **Zero Dynamic Allocation on the Forward Path**: Once an execution lane enters the decode loop, zero calls to system memory allocators (`malloc`, `free`, Rust `alloc::*`, jemalloc) are permitted. All KV allocations, activation scratchpads, temporary buffers, and staging ring buffers reside in pre-allocated, lifetime-bounded static arenas (`'arena`).
2. **Compile-Time Device/Host Type Separation**: Host code never manipulates raw device pointers as untyped integers or raw host pointers. GPU memory is typed via `DevicePtr<T>` and verified through Zero-Sized Type (ZST) marker proofs, making dereferencing device VRAM on the CPU a compile-time failure.
3. **Monomorphized Hot Loop via Closed Dispatch**: Hot forward passes contain zero virtual dispatch (`dyn Trait` / vtables). Heterogeneous model and hardware execution is achieved via closed dispatch enums (`SpecializedPipeline`) matched once at pipeline initialization and lowered to flat direct jump tables.
4. **Thermal-Aware Adaptive Concurrency**: Worker threads never peg CPU cores at continuous $100\%$ active spin under idle or fluctuating load. Execution actors execute a strict three-phase cycle (micro-spin $\to$ `thread::yield_now()` $\to$ futex parking via `Parker`) to prevent package thermal throttling.
5. **Deterministic Bitwise Correctness**: Every quantized kernel (`PTQ1_0`, `PQ2_0`, `CQ2`, `NVFP4`) and custom layer (In-SMem FWHT, Engram table gather) passes automated bitwise tolerance tests against PyTorch/GGUF reference implementations before integration into the serving binary.

---

## Developmental Horizons Roadmap

```
+─────────────────────────────────────────────────────────────────────────────────────────────+
|                                    DEVELOPMENTAL TOPOLOGY                                   |
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON 0: Foundational Workspace Substrate, Memory Typing & Golden Oracle                 |
|  └── Precondition: Hermetic pure-Rust workspace, #![deny(unsafe_op_in_unsafe_fn)]           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 0: Zero LTO Warnings & Oracle Match]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON I: Hardware-Bound Memory Safety, Static Arenas & Core-Pinned Actors                |
|  └── Precondition: DevicePtr<T> invariants & stack guard macro verified                     |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 1: Zero Host Data Races & Zero Stack Overflow]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON II: Closed-Enum Pipeline Monomorphization & Zero-Copy Artifact Ingestion          |
|  └── Precondition: SpecializedPipeline jump tables validated with zero indirect calls        |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 2: Branchless Model Ingestion]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON III: Microarchitectural Kernels: In-SMem FWHT & Branchless Ternary GEMV             |
|  └── Precondition: Group-128 FP16 scales & unrolled butterfly networks stabilized           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 3: Layer Latency Saturation & Cosine Similarity]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON IV: Non-Euclidean Architectures: Needle Simple Attention, Monarch MLP & Engrams    |
|  └── Precondition: Subnetwork depth laddering & in-kernel DFA byte masks compiled           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 4: 100% DFA Schema Conformance]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON V: Dual-Topology Hybrid Memory Management & Transactional Rollback                 |
|  └── Precondition: Static recurrent O(1) state split from virtual paged KV pools           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 5: Zero VRAM Fragmentation Across Continuous Load]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VI: Directed Acyclic Graph (DAG) Tree-Native Reasoning & In-Flight Pruning         |
|  └── Precondition: Lock-free atomic reference-counted blocks (crossbeam-epoch)              |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 6: Zero-Copy Thought Branching]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VII: Cross-Silicon Portability: Heterogeneous Silicon & SIMD CPU Engines          |
|  └── Precondition: Unified memory zero-copy and AVX-512 / ARM Neon bitwise parity          |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 7: Cross-Platform Numerical Invariance]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON VIII: Multi-Device Distributed Topologies & Collective Fabrics                     |
|  └── Precondition: Hierarchical KV cache transfer & direct in-kernel collective bindings    |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 8: Near-Linear Multi-Device Scaling]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON IX: Profile-Guided Synthesis & Continuous Self-Tuning Autonomics                   |
|  └── Precondition: AutonomicPlan offline device registry & basic-block alignment           |
+──────────────────────────────────────────────┬──────────────────────────────────────────────+
                                               │ [Gate 9: Zero Instruction-Cache Thrashing]
                                               ▼
+─────────────────────────────────────────────────────────────────────────────────────────────+
|  HORIZON X: Physical Hardware Co-Design: Optical, In-Memory & Neuromorphic Substrates       |
|  └── Precondition: HardwareSupports<Backend, Model> compile-time marker proving            |
+─────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## Detailed Horizon Specifications

### Horizon 0: Foundational Workspace Substrate, Memory Typing & Golden Oracle
- Hermetic pure-Rust workspace with isolated crate boundaries (`oxide-core`, `oxide-alloc`, `oxide-hardware/*`, `oxide-models`, `oxide-quant`, `oxide-server`, `oxide-engine`, `oxide-cli`).
- `DevicePtr<T>` encapsulates raw GPU memory addresses without `Deref`/`DerefMut` to CPU memory.
- `#[assert_stack_safe(max_bytes = 4096)]` ensures no massive context arrays (> 4 KB) are allocated on the thread stack.
- Differential verification against PyTorch/GGUF reference implementations ($L_\infty \le 1.5 \times 10^{-3}$, cosine similarity $\ge 0.9999$).

### Horizon I: Hardware-Bound Memory Safety, Static Arenas & Core-Pinned Actors
- `HostPinnedArena`: Direct page-locked host memory mapped to GPU virtual address spaces (`cudaHostAllocMapped | cudaHostAllocWriteCombined`).
- `DeviceMemoryArena<'arena>`: Lifetime-bounded device memory allocation preventing runtime dynamic allocations.
- Typestate state machine enforcing sequence progression (`Unallocated` $\to$ `Allocated` $\to$ `Prefilling` $\to$ `Decoding` $\to$ `Terminal`).
- Wait-free 64-byte aligned SPSC ringbuffers (`rtrb`) isolating control plane from execution lanes.
- 3-Phase adaptive worker actor (`ExecutionWorker`) eliminating thread spin traps and CPU thermal throttling.

### Horizon II: Closed-Enum Pipeline Monomorphization & Zero-Copy Artifact Ingestion
- Closed enum `SpecializedPipeline` lowering match statements into flat direct jump tables.
- Aligned 64-byte `.oxide` binary container header specification (`OxideModelHeader`).
- Memory-mapped asynchronous model ingestion using `memmap2`.

### Horizon III: Microarchitectural Kernels: In-SMem FWHT & Branchless Ternary GEMV
- In-Shared-Memory Fast Walsh-Hadamard Transform (FWHT): 7-stage unrolled butterfly network in warp registers and shared memory.
- Branchless 2-bit ternary arithmetic (`PTQ1_0`, `PQ2_0`) lowering directly to `__dp4a` (CUDA) and `vpdpbusd` (AVX-512).
- Micro-benchmarked kernel latency $\le 18\,\mu\text{s}$ per layer with zero uncoalesced memory transactions.

### Horizon IV: Non-Euclidean Architectures: Needle Simple Attention, Monarch MLP & Engrams
- Monarch Hadamard block-diagonal matrix factorization lowering MLP parameter complexity to $O(D \sqrt{D})$.
- Const-generic dynamic subnetwork depth laddering (`NeedleSubnetwork<L>`) with zero-branch compile-time truncation.
- Direct-mapped 70.8M parameter Engram gather tables in pinned host/device memory.
- In-kernel byte-level DFA grammar masking for $100\%$ schema-conforming JSON generation.

### Horizon V: Dual-Topology Hybrid Memory Management & Transactional Rollback
- Dual-topology state memory management: $O(1)$ linear recurrent state matrix + virtual block-paged softmax KV cache.
- Transactional speculative rollback engine (`TransactionalBlockTable`) performing $O(1)$ metadata truncation on draft rejection.
- Sarathi chunked prefill co-scheduling $C=512$ chunks with ongoing decode iterations.

### Horizon VI: Directed Acyclic Graph (DAG) Tree-Native Reasoning & In-Flight Pruning
- Lock-free intrusive Copy-on-Write (CoW) block hierarchy (`SharedPhysicalBlock`, `TreeNode`) for $100,000+$ reasoning branches.
- Topological DAG attention mask synthesis for single-pass batched tree exploration.
- Asynchronous Process Reward Model (PRM) scoring and in-flight tree branch recycling.

### Horizon VII: Cross-Silicon Portability: Heterogeneous Silicon & SIMD CPU Engines
- Apple Silicon Metal backend (`oxide-backend-metal`) leveraging unified memory and MSL threadgroup shaders.
- Vectorized CPU backend (`oxide-backend-cpu`) with AVX-512 and ARM Neon intrinsics.
- Multi-silicon accelerators: AMD ROCm, Google TPU, Intel Level-Zero, Qualcomm Snapdragon HTP, Rockchip RKNN, Raspberry Pi 5 AI HAT+ 2 / Hailo.

### Horizon VIII: Multi-Device Distributed Topologies & Collective Fabrics
- 3-tier Hierarchical KV Cache (`HierarchicalKvCache`): Tier 1 Device VRAM, Tier 2 Host Pinned RAM, Tier 3 NVMe Storage.
- Content-addressed block sharing with Blake3 prefix hashing across distributed inference nodes.
- Direct in-kernel NCCL collective communications and P2P DMA over NVLink.

### Horizon IX: Profile-Guided Synthesis & Continuous Self-Tuning Autonomics
- Autonomic plan hardware registry (`AutonomicPlan`) configuring optimal threadblock tiles and warp limits.
- L1 Instruction Cache protection limiting compiler inlining bloat.

### Horizon X: Physical Hardware Co-Design: Optical, In-Memory & Neuromorphic Substrates
- Compile-time hardware capability proving via `HardwareSupports<Backend, Model>`.
- CXL.mem attached persistent memory descriptors and neuromorphic event-driven spiking execution.

---

## Dynamic ggml / llama.cpp Execution Architecture

To support any model dynamically without requiring the user to recompile the binary for different architectures or weights, `Oxide-Tech-LLM-Engine` adheres to the tripartite ggml / `llama.cpp` runtime paradigm:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. The Pre-compiled Binary (`oxide-engine` / `oxide`)                       │
│    - Exhaustive kernel library: MatMul, FlashAttention, RoPE, RMSNorm       │
│    - Quantization kernels: BlockQ2_K..BlockQ8_0, NvFP4, PTQ 1.58-bit        │
│    - Zero-allocation Bump Allocator (`GraphArena`) with O(1) reset          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ Ingests at runtime via mmap
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. The Universal Model File (GGUF / SafeTensors)                            │
│    - Metadata: architecture ("llama", "qwen2", "deepseek2", etc.)           │
│    - Hyperparameters: context_length, embedding_dim, block_count, heads     │
│    - Zero-copy weight tensors with -ngl GPU offloading (`WeightAllocator`)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ Dynamic DAG synthesis
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. The Execution Runtime                                                    │
│    - Synthesizes dynamic Compute Graph (`GraphNode`, `OpCode`)              │
│    - Applies graph-level kernel fusions (e.g. `RmsNorm` + `MulMat`)         │
│    - Lowers to `ComputeGraphExecutor` on pre-allocated scratch memory       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Streaming Universal Model Loader (`crates/oxide-models/src/loader.rs`)
- **Metadata Extraction**: Reads architecture keys, context windows, head counts, and vocabularies on the fly.
- **Zero-Copy Memory-Mapping**: Weights are accessed via `memmap2` with zero redundant heap copies.
- **Selective GPU Offloading**: `WeightAllocator` partitions layers between host RAM and GPU VRAM according to the user-specified `-ngl` / `--n-gpu-layers` parameter.

### 2. Dynamic Compute Graph (DAG) & Kernel Fusion (`crates/oxide-engine/src/graph.rs`)
- **`OpCode` Representation**: Compact `#[repr(u8)]` operation codes (`MulMat`, `RmsNorm`, `RoPE`, `Softmax`, `Add`, `FusedRmsMulMat`, etc.).
- **Graph Optimization Passes**: Consecutive dependent nodes (such as normalization immediately followed by projection matrix multiplication) are fused into specialized single-pass kernels (`FusedRmsMulMat`), reducing memory roundtrips and memory bus pressure.

### 3. Zero-Allocation `GraphArena` Bump Allocator (`crates/oxide-engine/src/arena.rs`)
- **Scratch Space Pre-allocation**: Activation tensors and intermediate layer buffers are carved out of a contiguous linear memory arena.
- **$O(1)$ Turnaround**: At the completion of each forward step, the arena offset resets in $O(1)$ time with zero calls to system memory allocators (`malloc`/`free`).
- **Bitwise Precision Guarantee**: Executed nodes pass strict cosine-similarity and absolute error bounds.

