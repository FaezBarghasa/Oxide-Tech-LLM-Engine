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
    predicted_noise_buffer: Vec<f32>,
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
            predicted_noise_buffer: vec![0.0; latent_len],
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
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = ((state >> 32) as i32 as f32) / (i32::MAX as f32);
            *x = val;
        }
    }

    /// Predicts noise field $\epsilon_\theta(x_t, t, c)$ across latents using DiT transformer block simulation
    fn predict_noise_dit(&mut self, t: f32) {
        let cond_norm = if self.text_cond_buffer.is_empty() {
            0.1f32
        } else {
            let mut s = 0.0f32;
            for &val in self.text_cond_buffer.iter().take(128) {
                s += val * val;
            }
            (s / 128.0).sqrt().max(0.01)
        };

        // DiT adaptive LayerNorm modulation & MLP projection over latent patches
        let patch_elem = self.config.patch_size * self.config.patch_size * self.config.in_channels;
        let p_elem = patch_elem.max(1);

        for (i, (&x_val, noise_out)) in self
            .latent_buffer
            .iter()
            .zip(self.predicted_noise_buffer.iter_mut())
            .enumerate()
        {
            let patch_idx = i / p_elem;
            let within_patch = (i % p_elem) as f32;
            let ada_ln_scale = 1.0 + 0.1 * ((within_patch * 0.31 + t).sin());
            let ada_ln_shift = 0.05 * ((patch_idx as f32 * 0.17 + t).cos());

            let norm_x = (x_val - ada_ln_shift) * ada_ln_scale;
            // Cross-attention conditioning interaction with text embeddings
            let attended = norm_x * cond_norm;
            // DiT SwiGLU projection approximation
            let activated = attended / (1.0 + (-attended).exp());
            *noise_out = activated * 0.8 + norm_x * 0.2;
        }
    }

    /// Single denoising step forward pass using Euler discrete or DDIM step update.
    pub fn step_denoise(&mut self, step_idx: usize) -> Result<()> {
        let total_steps = self.config.num_inference_steps.max(1);
        let t_curr = 1.0 - (step_idx as f32 / total_steps as f32);
        let t_next = 1.0 - ((step_idx + 1) as f32 / total_steps as f32);
        let dt = t_curr - t_next;

        // 1. Evaluate DiT transformer blocks to predict noise
        self.predict_noise_dit(t_curr);

        // 2. Scheduler Step Update
        match self.config.scheduler {
            DiffusionSchedulerType::EulerDiscrete
            | DiffusionSchedulerType::RectifiedFlowMatching => {
                // Euler / Flow-matching step: x_{t-1} = x_t - dt * v_\theta
                for (x, &noise) in self
                    .latent_buffer
                    .iter_mut()
                    .zip(self.predicted_noise_buffer.iter())
                {
                    *x -= dt * noise;
                }
            }
            DiffusionSchedulerType::Ddim => {
                // DDIM step: x_{t-1} = \sqrt{\alpha_{t-1}} * (x_t - \sqrt{1 - \alpha_t} * \epsilon) / \sqrt{\alpha_t} + \sqrt{1 - \alpha_{t-1}} * \epsilon
                let alpha_curr = (t_curr * std::f32::consts::FRAC_PI_2)
                    .cos()
                    .powi(2)
                    .max(1e-4);
                let alpha_next = (t_next * std::f32::consts::FRAC_PI_2)
                    .cos()
                    .powi(2)
                    .max(1e-4);
                let sqrt_alpha_curr = alpha_curr.sqrt();
                let sqrt_one_minus_alpha_curr = (1.0 - alpha_curr).max(0.0).sqrt();
                let sqrt_alpha_next = alpha_next.sqrt();
                let sqrt_one_minus_alpha_next = (1.0 - alpha_next).max(0.0).sqrt();

                for (x, &eps) in self
                    .latent_buffer
                    .iter_mut()
                    .zip(self.predicted_noise_buffer.iter())
                {
                    let x0_pred = (*x - sqrt_one_minus_alpha_curr * eps) / sqrt_alpha_curr;
                    *x = sqrt_alpha_next * x0_pred + sqrt_one_minus_alpha_next * eps;
                }
            }
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
