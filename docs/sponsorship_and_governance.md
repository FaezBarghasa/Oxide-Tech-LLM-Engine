# Sponsorship, Grant Allocations & Governance 🏛️

> **Oxide-Tech LLM Engine Open-Source Foundation & Stewardship**  
> Dedicated to zero-allocation, local-first, vendor-neutral heterogeneous systems engineering.

---

## 1. Governance Creed

`Oxide-Tech-LLM-Engine` is developed and maintained under the **Apache License Version 2.0**. Our technical governance adheres to four core engineering rules:
1. **Zero Vendor Lock-In**: First-class hardware support for AMD, NVIDIA, Intel, Apple, Qualcomm, Rockchip, Hailo, and Google hardware. No single hardware vendor may dictate architectural concessions that compromise other targets.
2. **Deterministic Zero-Allocation Hot Paths**: The forward token decode loop must never allocate heap memory. Any PR or contribution introducing heap allocations on hot paths will be rejected.
3. **Safety & Pure Rust**: Rust 2024 Edition (`1.85+`), strictly eliminating unhandled crashes, data races, and undefined behavior.
4. **Transparent Sponsorship Stewardship**: All financial sponsorships received through GitHub Sponsors, Open Collective, Patreon, and direct corporate grants are tracked and allocated with 100% public visibility.

---

## 2. Allocation of Sponsorship Funds

Sponsorship contributions are allocated strictly according to the following fund partition:

```
┌─────────────────────────────────────────────────────────────┐
│             SPONSORSHIP FUND ALLOCATION MODEL               │
├──────────────────────────────┬──────────────────────────────┤
│ 60% Hardware Lab Bench       │ Acquisition & maintenance of │
│     Acquisition & Testing    │ physical GPUs, APUs, NPUs    │
├──────────────────────────────┼──────────────────────────────┤
│ 25% Bare-Metal CI &          │ Dedicated Linux servers,     │
│     Regression Test Rigs     │ continuous ASAN/Miri benches │
├──────────────────────────────┼──────────────────────────────┤
│ 15% Core Architecture        │ Technical stipends, code     │
│     Stipends & Maintenance   │ audits, and docs maintenance │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 3. Corporate Sustainer & Hardware Partner Program

For semiconductor manufacturers, cloud hyperscalers, and AI infrastructure startups:

### Custom Silicon Kernel Co-Development
If your organization is developing or deploying custom accelerators (e.g., custom ASIC, RISC-V neural accelerator, FPGA tensor engine, or proprietary NPU):
- We co-develop native `oxide-backend-<vendor>` crates implementing the `HardwareBackend` trait.
- We implement customized DMA ringbuffers, unified memory zero-copy mappings, and optimized GEMV/FlashAttention kernels.
- We validate your silicon in our automated continuous performance regression matrix.

### Contacting Governance & Invoicing
- **Lead Architect**: Faez Barghasa (`faez.barghasa@gmail.com`)
- **Direct Sponsors Page**: [SPONSORS.md](../SPONSORS.md)
- **GitHub Sponsors**: [github.com/sponsors/FaezBarghasa](https://github.com/sponsors/FaezBarghasa)
- **Open Collective**: [opencollective.com/oxide-tech](https://opencollective.com/oxide-tech)
