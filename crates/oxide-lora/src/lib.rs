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
pub mod qlora;

pub use loader::{UnslothAdapter, UnslothAdapterConfig, UnslothTensorPair};
pub use punica::UnslothLoraManager;
pub use qlora::{NF4_BLOCK_SIZE, NF4_VALUES, QloraLinearLayer, QloraTrainingConfig, UnslothQloraTrainer};

/// Unique identifier for a registered LoRA adapter persona.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoraId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_unsloth_qlora_training_and_export_lifecycle() {
        let temp_dir = std::env::temp_dir().join("oxide_test_qlora_export");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let cfg = QloraTrainingConfig {
            learning_rate: 1e-2,
            gradient_accumulation_steps: 2,
            lora_r: 4,
            lora_alpha: 8.0,
            ..Default::default()
        };

        let mut trainer = UnslothQloraTrainer::new(cfg);
        trainer.add_layer("model.layers.0.self_attn.q_proj", 16, 16);

        // Run forward step
        let input = vec![1.0f32; 16];
        let mut inter_r = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 16];

        let layer = trainer.layers.get("model.layers.0.self_attn.q_proj").unwrap();
        layer.forward(&input, &mut inter_r, &mut out);
        assert_eq!(out.len(), 16);

        // Run 2 micro-steps to trigger optimizer step
        let mut activations = HashMap::new();
        activations.insert("model.layers.0.self_attn.q_proj".to_string(), (input.clone(), inter_r.clone()));
        let mut grads = HashMap::new();
        grads.insert("model.layers.0.self_attn.q_proj".to_string(), vec![0.5f32; 16]);

        trainer.backward_micro_step(&activations, &grads);
        assert!(!trainer.maybe_optimizer_step()); // step 1 of 2

        trainer.backward_micro_step(&activations, &grads);
        assert!(trainer.maybe_optimizer_step()); // step 2 of 2 -> triggered optimizer!
        assert_eq!(trainer.step_count, 1);

        // Export to disk in Unsloth Safetensors format
        trainer.export_unsloth_checkpoint(&temp_dir, "test-model-4bit").expect("Checkpoint exported");
        assert!(temp_dir.join("adapter_config.json").exists());
        assert!(temp_dir.join("adapter_model.safetensors").exists());

        // Re-import exported adapter into in-memory UnslothAdapter
        let loaded = UnslothAdapter::from_directory("trained_persona", &temp_dir).expect("Adapter loaded from export");
        assert_eq!(loaded.name, "trained_persona");
        assert!(loaded.layers.contains_key("model.layers.0.self_attn.q_proj"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

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
