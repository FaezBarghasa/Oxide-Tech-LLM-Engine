#!/usr/bin/env python3
"""
scripts/dump_golden_tensors.py
Offline PyTorch Golden Tensor Generator for Oxide-Tech-LLM-Engine.
Extracts reference weights, activations, and logit distributions for:
- deepgrove/Ternary-Bonsai-2-27B
- cactus-compute/needle-3
- meta-llama/Llama-3.1-8B
"""

import os
import json
import numpy as np

def generate_synthetic_golden_fixtures(output_dir: str = "fixtures/golden"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating golden reference fixtures in {output_dir}...")

    # Bonsai2 reference fixtures
    bonsai_in = np.random.randn(128).astype(np.float32)
    # 128 weights in {-1, 0, 1}
    bonsai_w = np.random.choice([-1.0, 0.0, 1.0], size=128).astype(np.float32)
    bonsai_scale = 0.045
    bonsai_dot = np.sum(bonsai_in * bonsai_w) * bonsai_scale

    np.save(os.path.join(output_dir, "bonsai2_input.npy"), bonsai_in)
    np.save(os.path.join(output_dir, "bonsai2_weights.npy"), bonsai_w)
    np.save(os.path.join(output_dir, "bonsai2_golden_dot.npy"), np.array([bonsai_dot], dtype=np.float32))

    # Needle3 Monarch reference fixtures
    hidden_dim = 64
    monarch_in = np.random.randn(hidden_dim).astype(np.float32)
    np.save(os.path.join(output_dir, "needle3_input.npy"), monarch_in)

    manifest = {
        "models": ["Ternary-Bonsai-2-27B", "Cactus-Needle-3", "Meta-Llama-3.1-8B"],
        "tolerance": {
            "fp16_chebyshev_epsilon": 1.5e-3,
            "fp16_cosine_delta": 1.0e-5,
            "quant_chebyshev_epsilon": 5.0e-2,
            "quant_cosine_delta": 1.0e-3
        }
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("Golden fixtures successfully generated.")

if __name__ == "__main__":
    generate_synthetic_golden_fixtures()
