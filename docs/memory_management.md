# Memory Management & Hierarchical KV Caching

## Overview

Inference performance and latency predictability in `Oxide-Tech-LLM-Engine` are fundamentally rooted in **deterministic, zero-dynamic-allocation memory management**. Standard runtimes suffer from memory fragmentation, allocator contention (`malloc`/`free` locks), and unpredictable Garbage Collection or allocator page faults.

Oxide eliminates all dynamic allocations during the decode loop through static memory arenas, typestate lifecycles, and a 3-tier hierarchical KV cache architecture.

---

## 1. Zero-Cost Compile-Time Device Pointer Typing (`DevicePtr<T>`)

GPU memory addresses must never be handled as raw integers (`usize`, `u64`) or raw CPU pointers (`*mut T`), which can lead to accidental host dereferencing or type safety bypasses.

In `crates/oxide-core/src/memory.rs`, `DevicePtr<T>` wraps raw addresses with Zero-Sized Type (ZST) marker proofs:

```rust
use std::marker::PhantomData;

#[repr(transparent)]
pub struct DevicePtr<T> {
    raw: *mut T,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for DevicePtr<T> {}
unsafe impl<T: Sync> Sync for DevicePtr<T> {}

impl<T> Copy for DevicePtr<T> {}
impl<T> Clone for DevicePtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self { *self }
}

impl<T> DevicePtr<T> {
    #[inline(always)]
    pub const unsafe fn from_raw(raw: *mut T) -> Self {
        Self { raw, _marker: PhantomData }
    }

    #[inline(always)]
    pub fn as_raw(self) -> *mut T { self.raw }

    #[inline(always)]
    pub fn as_device_address(self) -> u64 { self.raw as u64 }

    #[inline(always)]
    pub unsafe fn offset(self, count: usize) -> Self {
        Self {
            raw: self.raw.add(count),
            _marker: PhantomData,
        }
    }
}
```

Because `DevicePtr<T>` explicitly omits `Deref` and `DerefMut` implementations, dereferencing device VRAM on the CPU triggers an immediate compile-time error.

---

## 2. Static Lifetime-Bounded Memory Arenas (`'arena`)

### A. `HostPinnedArena`
Page-locked (pinned) host memory allocated via `cudaHostAlloc` with flags `cudaHostAllocMapped | cudaHostAllocWriteCombined`.
- Directly mapped into the device's virtual address space.
- Enables asynchronous PCIe/NVLink direct memory access (DMA) without explicit staging buffers.
- Pre-allocates sequence context arrays, bypassing OS thread stack boundaries (preventing stack guard page faults).

### B. `DeviceMemoryArena<'arena>`
Pre-allocates all activation scratchpads, temporary buffers, and KV pools in device VRAM at boot time.
- Lifetime parameter `'arena` guarantees that sequence allocations cannot outlive the underlying device memory pool.
- Guaranteed zero calls to `cudaMalloc` or `malloc` during token generation.

---

## 3. Typestate Request Lifecycle

The typestate pattern in `crates/oxide-core/src/typestate.rs` enforces that execution steps can only be invoked on requests that hold valid static physical reservations:

```rust
pub struct Unallocated;
pub struct Allocated;
pub struct Prefilling;
pub struct Decoding;
pub struct Terminal;

pub struct SequenceRequest<'arena, State> {
    pub sequence_id: u64,
    pub prompt_tokens: &'arena [u32],
    pub state_marker: PhantomData<State>,
}

impl<'arena> SequenceRequest<'arena, Unallocated> {
    pub fn allocate(
        self,
        slot_idx: usize,
        arena: &'arena mut DeviceMemoryArena<'arena>,
    ) -> Result<SequenceRequest<'arena, Allocated>, EngineError> {
        arena.bind_slot(self.sequence_id, slot_idx)?;
        Ok(SequenceRequest {
            sequence_id: self.sequence_id,
            prompt_tokens: self.prompt_tokens,
            state_marker: PhantomData,
        })
    }
}
```

---

## 4. Dual-Topology State & Cache Allocator

For hybrid architectures (such as Ternary Bonsai 2), memory is split by layer topology:

### Topology 1: Linear Recurrent State Matrix ($O(1)$ Constant Memory)
For linear attention layers (e.g., $75\%$ of layers in Bonsai 2), KV cache does not grow with context length. The recurrent state matrix updates as:
$$S_t = S_{t-1} + K_t^T V_t \quad \text{where} \quad S_t \in \mathbb{R}^{D_{\text{head}} \times D_{\text{head}}}$$
Memory is pre-allocated as a single static block:
$$\text{Bytes} = \text{Slots} \times \text{Layers}_{\text{linear}} \times D_{\text{head}} \times D_{\text{head}} \times 2\text{ bytes (BF16)}$$

### Topology 2: Virtual Block-Paged Softmax KV Pool ($O(N)$ Dynamic Memory)
For softmax attention layers, physical memory is mapped in fixed blocks ($B=16$ tokens). Physical page tables map logical sequence offsets to physical block IDs, eliminating external memory fragmentation.

---

## 5. 3-Tier Hierarchical KV Caching Engine

The **Hierarchical KV Cache** (`crates/oxide-alloc/src/hierarchical_kv.rs`) manages memory across three physical tiers:

```
[ Active Stream Request ]
         │
         ▼
┌─────────────────────────────────┐
│   Tier 1: Device VRAM Pool      │  ◄── Sub-microsecond latency (SRAM / HBM3 / GDDR6X)
│   (Contiguous Page Blocks)      │      Direct GPU kernel addressable
└────────────────┬────────────────┘
                 │ (Evict LRU / Prefetch Hit)
                 ▼
┌─────────────────────────────────┐
│   Tier 2: Host Pinned RAM Pool  │  ◄── Sub-millisecond latency (DDR5 / LPDDR5X)
│   (Zero-Copy DMA Mapped)        │      PCIe Gen5 / CXL.mem asynchronous transfer
└────────────────┬────────────────┘
                 │ (Deep Context Archive / Reuse)
                 ▼
┌─────────────────────────────────┐
│   Tier 3: NVMe Storage Pool     │  ◄── Disk I/O bound (O_DIRECT / io_uring)
│   (Persistent Content Addressing)│     Shared distributed cross-instance cache reuse
└─────────────────────────────────┘
```

- **Tier 1 (Device VRAM)**: Fast active pool for immediate token decode.
- **Tier 2 (Host Pinned RAM)**: High-capacity intermediate pool. Prefetched via asynchronous DMA streams.
- **Tier 3 (External NVMe Storage)**: Persistent content-addressed block tables with SHA-256 / Blake3 prefix hashing. Enables cross-request and cross-node prompt prefix reuse with zero redundant computation.

---

## 6. Atomic Transactional Speculative Rollback Engine

In speculative decoding, draft tokens may be partially or completely rejected by the verifier model. Traditional implementations zero or overwrite VRAM, incurring synchronization overhead.

In `crates/oxide-alloc/src/transactional.rs`, `TransactionalBlockTable` manages rollbacks at $O(1)$ host metadata speed:

```rust
pub struct TransactionalBlockTable {
    pub block_size: usize,
    pub physical_blocks: Vec<u32>,
    pub active_tokens: usize,
}

impl TransactionalBlockTable {
    pub fn commit_speculation(&mut self, accepted_tokens: usize) {
        self.active_tokens += accepted_tokens;
    }

    pub fn rollback(&mut self, accepted_tokens: usize, _total_speculated: usize) {
        let retained_blocks = (accepted_tokens + self.block_size - 1) / self.block_size;
        self.physical_blocks.truncate(retained_blocks);
        self.active_tokens = accepted_tokens;
    }
}
```

Device VRAM is never mutated during a rollback; subsequent forward passes simply overwrite unused block slots.
