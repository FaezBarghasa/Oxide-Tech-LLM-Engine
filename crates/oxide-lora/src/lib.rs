//! # `oxide-lora`
//!
//! Native Unsloth Safetensors LoRA Loader, Dequant-Time Fusion, and S-LoRA Multi-Persona Hot-Swapping.
//!
//! Supports:
//! - Direct Safetensors ingestion without Python scripts
//! - Mode A: Zero-overhead weight fusion into dequantized base weights
//! - Mode B: Sub-millisecond S-LoRA / Punica multi-persona hot-swapping

pub mod fusion;
pub mod loader;
pub mod punica;

pub use loader::{UnslothAdapter, UnslothAdapterConfig, UnslothTensorPair};
pub use punica::UnslothLoraManager;

/// Unique identifier for a registered LoRA adapter persona.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoraId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsloth_tensor_pair_fusion_parity() {
        let in_dim = 8;
        let out_dim = 16;
        let rank = 4;
        let pair = UnslothTensorPair::new(in_dim, out_dim, rank, 16.0);

        let mut delta_w = vec![0.0f32; out_dim * in_dim];
        pair.compute_weight_delta(&mut delta_w);

        let input = vec![1.0f32; in_dim];
        let mut out_gemm = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += delta_w[o * in_dim + i] * input[i];
            }
            out_gemm[o] = sum;
        }

        let mut out_punica = vec![0.0f32; out_dim];
        pair.apply_lora_delta(&input, &mut out_punica);

        for (g, p) in out_gemm.iter().zip(out_punica.iter()) {
            assert!((g - p).abs() < 1e-4, "Fusion vs Punica mismatch: {g} vs {p}");
        }
    }

    #[test]
    fn test_lora_manager_hot_swap() {
        let mut manager = UnslothLoraManager::new();
        let adapter1 = UnslothAdapter::new("persona_mix", UnslothAdapterConfig::default());
        let adapter2 = UnslothAdapter::new("persona_theorist", UnslothAdapterConfig::default());

        manager.register_adapter(adapter1);
        manager.register_adapter(adapter2);

        assert!(manager.set_active_adapter(Some("persona_mix")));
        assert_eq!(manager.active_adapter().unwrap().name, "persona_mix");

        assert!(manager.set_active_adapter(Some("persona_theorist")));
        assert_eq!(manager.active_adapter().unwrap().name, "persona_theorist");

        assert!(manager.set_active_adapter(None));
        assert!(manager.active_adapter().is_none());
    }
}
