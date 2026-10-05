# Academic Sampling Algorithms & Generation Control Suite

## Overview

`Oxide-Tech-LLM-Engine` provides an exhaustive implementation of state-of-the-art and academic token sampling algorithms in `crates/oxide-core/src/sampler.rs` and `crates/oxide-engine/src/speculative.rs`.

---

## 1. Academic Sampling Algorithms

### A. Mirostat (v1 & v2)
- **Concept**: Active feedback-loop sampler that dynamically adjusts truncation parameter ($\tau$) or temperature at every decode step to maintain a constant target perplexity / entropy ($H = \tau$).
- **Mirostat v1**: Estimates surprise value using truncation threshold $\hat{k}$ and learning rate $\eta$.
- **Mirostat v2**: Uses max surprise threshold $\hat{s}$ and directly adjusts temperature $T_{\text{dyn}} = \mu / \tau$:
$$\mu \leftarrow \mu - \eta \cdot (s(x) - \tau)$$

---

### B. DRY (Don't Repeat Yourself) Sampling
- **Concept**: Advanced exponential repetition penalty that detects and breaks repeating multi-token n-gram cycles far more effectively than standard frequency/presence penalties.
- **Algorithm**: For every candidate token $t$, the sampler scans the recent token history backwards to locate matching prefixes of length $L \ge L_{\text{min}}$. When a match is found:
$$\text{logit}(t) \leftarrow \text{logit}(t) - \text{penalty\_multiplier} \cdot \text{base}^{L - L_{\text{min}}}$$

---

### C. XTC (Exclude Top Choices) & Min-P
- **Min-P Sampling**: Dynamically trims all tokens whose probability falls below a threshold relative to the top token:
$$p_i < p_{\max} \cdot p_{\text{base}} \implies \text{logit}_i = -\infty$$
- **XTC (Exclude Top Choices)**: In high-confidence scenarios ($p_{\max} > \text{threshold}$), XTC optionally masks out the top token with probability $p_{\text{xtc}}$, forcing the model to explore creative, high-quality secondary branches without output degradation.

---

### D. Tail-Free Sampling (TFS-Z) & Locally Typical Sampling
- **Tail-Free Sampling (TFS-Z)**: Computes the second derivative of the sorted probability curve to detect where the "elbow" of the distribution begins, cutting off the flat tail based on curvature without arbitrary fixed $P$ or $K$ cutoffs.
- **Locally Typical Sampling**: Selects tokens whose conditional information content $-\log_2 p_i$ is closest to the expected Shannon entropy of the distribution, ensuring mathematically consistent information density across long generations.

---

### E. GBNF Grammar & DFA Masking
- **GBNF Context-Free Grammars**: Constrains logits against arbitrary formal BNF grammars (JSON, SQL, Python, C++).
- **In-Kernel DFA**: Transforms schemas into deterministic finite automaton lookup tables for branchless single-cycle bitmask operations on GPU logits before softmax.

---

### F. Logit Bias, Token Bans & Penalize Newline
- **Logit Bias**: Arbitrary positive or negative offsets applied directly to designated token IDs.
- **Token Bans**: Instant hard masking ($\text{logit} = -\infty$) for forbidden token IDs.
- **Penalize Newline (`--penalize-nl`)**: Prevents premature line break spamming during dense code and JSON generation.

---

## 2. Speculative Decoding Engine (`crates/oxide-engine/src/speculative.rs`)

Oxide implements a zero-allocation Speculative Decoding engine:
1. **Draft Generation**: A small, fast draft model (e.g., Llama-3-1B or specialized draft head) autoregressively produces $K$ candidate draft tokens $(\tilde{x}_1, \dots, \tilde{x}_K)$ in single-token passes.
2. **Parallel Target Verification**: The primary target model executes a single batched prefill pass over all $K$ draft tokens in parallel.
3. **Acceptance Criterion**: Evaluates target logit vs draft logit probabilities using the speculative rejection sampling rule.
4. **Acceptance Rate Tracking**: Dynamically adjusts speculative lookahead window $K$ to maximize aggregate tokens/second throughput.

---

## 3. RoPE Scaling Engines (`crates/oxide-models/src/rope.rs`)

Supports long-context extrapolation without fine-tuning degradation:
- **YaRN (Yet another RoPE extensioN)**: Frequency-dependent interpolation/extrapolation with temperature scaling.
- **LongRoPE**: Per-head search-based non-uniform RoPE scaling factor vectors.
- **Llama-3 RoPE**: High/low frequency boundary factor interpolation.
- **Linear RoPE**: Standard context expansion scaling.

---

## 4. LoRA & QLoRA Hot-Swapping (`crates/oxide-models/src/lora.rs`)

- **Hot-Swappable Multi-Tenant Adapters**: Dynamic injection and eviction of Low-Rank Adaptation matrices ($A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$) with scaling factor $\alpha / r$.
- **QLoRA Quantized Base Support**: Applies high-precision FP32/FP16 LoRA deltas on top of 4-bit/8-bit base model quantized weights without dequantizing base weights in VRAM.

---

## 5. Multi-Modal Projectors (`crates/oxide-models/src/projector.rs`)

- **Linear Projector**: $W_{\text{proj}} \cdot x_{\text{modal}}$ cross-modal alignment.
- **MLP GELU Projector**: Multi-layer perceptron projection with GELU activation mapping image, audio, and sensor tokens directly into the LLM embedding manifold.

---

## 6. Chat Template Engine (`crates/oxide-models/src/chat_template.rs`)

Supports strict formatting and rendering for:
- `ChatML` (`<|im_start|>role\ncontent<|im_end|>`)
- `Llama-3` (`<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`)
- `DeepSeek` (`<|User|>...<|Assistant|>...`)
- `Mistral` (`[INST] ... [/INST]`)
- `Alpaca` (`### Instruction:\n...\n\n### Response:\n...`)
