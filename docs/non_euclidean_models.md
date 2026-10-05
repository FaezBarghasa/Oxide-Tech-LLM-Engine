# Non-Euclidean Architectures: Needle, Monarch & Engrams

## Overview

Modern efficient inference paradigms move beyond dense matrix multiplications ($O(D^2)$) and standard quadratic attention ($O(N^2)$). `Oxide-Tech-LLM-Engine` natively implements non-Euclidean architectures exemplified by **Cactus Needle 3**:
1. **Monarch Hadamard Block-Diagonal MLPs**
2. **Dynamic Const-Generic Depth Laddering**
3. **Direct-Mapped Hashed Engram Tables**
4. **In-Kernel Byte-DFA Schema Grammar Masking**

---

## 1. Monarch Hadamard Block-Diagonal MLP

### Structural Reformulation
Needle 3 replaces the standard dense MLP intermediate projection ($D_{\text{intermediate}} = 4 \times D_{\text{hidden}}$) with two block-diagonal matrices $M_1, M_2$ interleaved with permutation operations:
$$W_{\text{Monarch}} = P_1 B_1 P_2 B_2$$

This factorizes parameter complexity from $O(D^2)$ to $O(D \sqrt{D})$, keeping intermediate activation footprint compact enough to fit entirely inside GPU L1 Shared Memory and CPU L1 caches.

In `crates/oxide-models/src/monarch.rs`:
```rust
pub struct MonarchMlp {
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub block1_weights: Vec<f32>,
    pub block2_weights: Vec<f32>,
}
```

---

## 2. Dynamic Const-Generic Subnetwork Depth Laddering

Needle 3 supports dynamic computational laddering, evaluating 2, 4, 8, 16, or 20 active layers based on latency budgets.

In `crates/oxide-models/src/needle.rs`, depth laddering is implemented via const generics:

```rust
pub struct NeedleSubnetwork<const ACTIVE_LAYERS: usize> {
    pub config: CactusNeedleConfig<ACTIVE_LAYERS>,
    pub layers: [MonarchMlp; ACTIVE_LAYERS],
}
```

The forward execution loop unrolls and terminates at compile time without dynamic loop counter branching on the GPU.

---

## 3. Direct-Mapped Hashed Engram Gather Table

Dense language models spend significant FLOPs computing shallow n-gram associations. Needle 3 bypasses dense matrix multiplication for n-gram features by utilizing a **70.8M parameter direct-mapped Engram Table**.

In `crates/oxide-models/src/engram.rs`:
- Hashing token n-gram sequences:
  $$\text{Slot} = \left(\sum_{k=0}^{n-1} T_{t-k} \cdot p^k\right) \pmod{\text{TABLE\_SIZE}}$$
- Reading parameter vectors directly from pinned host/device memory (`HostPinnedArena`).
- Lookups execute with $O(1)$ memory loads and zero matrix multiply FLOPs.

---

## 4. In-Kernel Byte-DFA Schema Grammar Masking

When generating structured JSON output or tool-call arguments, host-side validation loops cause roundtrip latency penalties.

Oxide compiles JSON schemas into an offline Deterministic Finite Automaton (DFA) state transition matrix:
$$\delta: S \times \Sigma \to S$$
where $S$ is the DFA state and $\Sigma \in [0, 255]$ is the next byte.

In `crates/oxide-server/src/dfa.rs`:
```rust
pub struct DfaSchemaGrammar {
    pub transition_table: Vec<[u16; 256]>,
    pub accepting_states: Vec<bool>,
    pub start_state: u16,
}
```

Prior to sampling on the GPU, tokens with byte sequences that violate $\delta(S, \text{byte})$ have their logits clamped to $-\infty$. This guarantees $100\%$ valid JSON syntax on the very first token emitted.
