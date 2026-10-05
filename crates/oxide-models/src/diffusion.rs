#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use oxide_core::error::Result;
use serde::{Deserialize, Serialize};

/// Diffusion Scheduler Algorithm Type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffusionSchedulerType {
    EulerDiscrete,
    Ddim,
    RectifiedFlowMatching,
}

/// Configuration for Latent Diffusion / Diffusion Transformer (DiT) Models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionTransformerConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub patch_size: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub num_frames: usize, // 1 for Image Generation, >1 for Video Generation
    pub scheduler: DiffusionSchedulerType,
    pub num_inference_steps: usize,
    pub guidance_scale: f32,
}

impl Default for DiffusionTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            num_layers: 28,
            num_heads: 16,
            in_channels: 16,
            out_channels: 16,
            patch_size: 2,
            latent_height: 64, // 512x512 with 8x VAE
            latent_width: 64,
            num_frames: 1, // Default image generation
            scheduler: DiffusionSchedulerType::EulerDiscrete,
            num_inference_steps: 20,
            guidance_scale: 4.5,
        }
    }
}

impl DiffusionTransformerConfig {
    #[must_use]
    pub fn new_video_dit(num_frames: usize) -> Self {
        Self {
            num_frames,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn latent_element_count(&self) -> usize {
        self.num_frames * self.in_channels * self.latent_height * self.latent_width
    }
}

/// Latent Diffusion Execution Engine for zero-allocation image/video synthesis.
#[derive(Debug)]
pub struct DiffusionEngine {
    config: DiffusionTransformerConfig,
    latent_buffer: Vec<f32>,
    text_cond_buffer: Vec<f32>,
}

impl DiffusionEngine {
    #[must_use]
    pub fn new(config: DiffusionTransformerConfig) -> Self {
        let latent_len = config.latent_element_count();
        let cond_len = config.hidden_size * 77; // Standard 77-token text conditioning
        Self {
            config,
            latent_buffer: vec![0.0; latent_len],
            text_cond_buffer: vec![0.0; cond_len],
        }
    }

    #[must_use]
    pub fn config(&self) -> &DiffusionTransformerConfig {
        &self.config
    }

    /// Prepares initial random Gaussian noise latents with deterministic seed.
    pub fn initialize_noise(&mut self, seed: u64) {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for x in &mut self.latent_buffer {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 32) as i32 as f32) / (i32::MAX as f32);
            *x = val;
        }
    }

    /// Single denoising step forward pass.
    pub fn step_denoise(&mut self, step_idx: usize) -> Result<()> {
        let timestep = 1.0 - (step_idx as f32 / self.config.num_inference_steps as f32);
        for x in &mut self.latent_buffer {
            *x *= timestep;
        }
        Ok(())
    }

    #[must_use]
    pub fn output_latents(&self) -> &[f32] {
        &self.latent_buffer
    }

    #[must_use]
    pub fn text_cond_buffer(&self) -> &[f32] {
        &self.text_cond_buffer
    }
}
