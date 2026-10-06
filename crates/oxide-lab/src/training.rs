#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
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
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::similar_names
)]

use serde::{Deserialize, Serialize};

/// Hyperparameter configuration for research lab model fine-tuning and training.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub max_grad_norm: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub batch_size: usize,
    pub is_qlora: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            max_grad_norm: 1.0,
            warmup_steps: 100,
            total_steps: 1000,
            batch_size: 1,
            is_qlora: false,
        }
    }
}

/// Loss function varieties supported in AI laboratory optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    CrossEntropy { label_smoothing: f32 },
    Mse,
    Contrastive { temperature: f32 },
    Dpo { beta: f32 },
}

/// Mathematical loss evaluator computing both scalar loss and backpropagated loss gradients.
#[derive(Debug, Clone)]
pub struct LossComputer;

impl LossComputer {
    /// Computes numerically-stable Cross-Entropy Loss and analytical softmax gradients.
    #[must_use]
    pub fn compute_cross_entropy(logits: &[f32], target_token: usize, label_smoothing: f32) -> (f32, Vec<f32>) {
        let vocab = logits.len();
        if vocab == 0 {
            return (0.0, Vec::new());
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; vocab];

        for (i, &l) in logits.iter().enumerate() {
            let exp = (l - max_logit).exp();
            probs[i] = exp;
            sum_exp += exp;
        }

        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for p in &mut probs {
            *p *= inv_sum;
        }

        let target_idx = target_token.min(vocab - 1);
        let target_prob = probs[target_idx].max(1e-12);
        let smooth = label_smoothing.clamp(0.0, 0.5);

        let loss = -(1.0 - smooth) * target_prob.ln() - (smooth / vocab as f32) * sum_exp.ln();

        // Analytical gradient: dL / dLogit = p_i - y_i
        let mut grad = probs;
        let target_target = 1.0 - smooth + (smooth / vocab as f32);
        grad[target_idx] -= target_target;

        (loss, grad)
    }

    /// Computes Mean Squared Error (MSE) loss and gradients.
    #[must_use]
    pub fn compute_mse(preds: &[f32], targets: &[f32]) -> (f32, Vec<f32>) {
        let n = preds.len().min(targets.len());
        if n == 0 {
            return (0.0, Vec::new());
        }

        let mut sum_sq = 0.0f32;
        let mut grad = vec![0.0f32; n];
        let factor = 2.0 / (n as f32);

        for i in 0..n {
            let diff = preds[i] - targets[i];
            sum_sq += diff * diff;
            grad[i] = factor * diff;
        }

        let loss = sum_sq / (n as f32);
        (loss, grad)
    }

    /// Computes Direct Preference Optimization (DPO) objective for LLM alignment without reinforcement learning.
    #[must_use]
    pub fn compute_dpo(
        policy_chosen_logp: f32,
        policy_rejected_logp: f32,
        ref_chosen_logp: f32,
        ref_rejected_logp: f32,
        beta: f32,
    ) -> (f32, f32) {
        let pi_logr = policy_chosen_logp - policy_rejected_logp;
        let ref_logr = ref_chosen_logp - ref_rejected_logp;
        let implicit_reward = beta * (pi_logr - ref_logr);

        // Loss = -log(sigmoid(implicit_reward))
        let sig = 1.0 / (1.0 + (-implicit_reward).exp());
        let loss = -sig.max(1e-12).ln();
        let grad = -beta * (1.0 - sig);

        (loss, grad)
    }
}

/// Zero-Allocation AdamW Optimizer with decoupled weight decay.
#[derive(Debug, Clone)]
pub struct AdamWOptimizer {
    pub lr: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub max_grad_norm: f32,
    pub step_count: usize,
    pub m: Vec<f32>, // First moment vector
    pub v: Vec<f32>, // Second moment vector
}

impl AdamWOptimizer {
    #[must_use]
    pub fn new(param_count: usize, config: &TrainingConfig) -> Self {
        Self {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            max_grad_norm: config.max_grad_norm,
            step_count: 0,
            m: vec![0.0; param_count],
            v: vec![0.0; param_count],
        }
    }

    /// Applies optimization step in-place to trainable parameters.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32], current_lr: f32) {
        let n = params.len().min(grads.len()).min(self.m.len());
        if n == 0 {
            return;
        }

        self.step_count += 1;
        let t = self.step_count as f32;

        // Gradient clipping: compute global L2 norm
        let mut total_norm_sq = 0.0f32;
        for &g in &grads[..n] {
            total_norm_sq += g * g;
        }
        let total_norm = total_norm_sq.sqrt();
        let clip_scale = if total_norm > self.max_grad_norm && total_norm > 1e-12 {
            self.max_grad_norm / total_norm
        } else {
            1.0
        };

        // Bias correction factors
        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        for i in 0..n {
            let g = grads[i] * clip_scale;

            // Decoupled weight decay
            params[i] -= current_lr * self.weight_decay * params[i];

            // Moments update
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;

            // Parameter update
            params[i] -= current_lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Cosine Annealing Learning Rate Scheduler with Linear Warmup.
#[derive(Debug, Clone)]
pub struct LrScheduler {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LrScheduler {
    #[must_use]
    pub fn new(base_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr: base_lr * 0.05,
            warmup_steps,
            total_steps,
        }
    }

    #[must_use]
    pub fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (step as f32 + 1.0) / (self.warmup_steps as f32).max(1.0)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            let cos_decay = f32::midpoint(1.0, (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.base_lr - self.min_lr) * cos_decay
        }
    }
}

/// Standalone Research LoRA Fine-Tuning Engine.
#[derive(Debug)]
pub struct LoraFineTuner {
    pub config: TrainingConfig,
    pub rank_r: usize,
    pub alpha: f32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub lora_a: Vec<f32>, // [rank_r, in_dim]
    pub lora_b: Vec<f32>, // [out_dim, rank_r]
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,
    pub optimizer_a: AdamWOptimizer,
    pub optimizer_b: AdamWOptimizer,
    pub scheduler: LrScheduler,
    pub current_step: usize,
}

impl LoraFineTuner {
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize, rank_r: usize, alpha: f32, config: TrainingConfig) -> Self {
        let size_a = rank_r * in_dim;
        let size_b = out_dim * rank_r;
        let scheduler = LrScheduler::new(config.learning_rate, config.warmup_steps, config.total_steps);
        let optimizer_a = AdamWOptimizer::new(size_a, &config);
        let optimizer_b = AdamWOptimizer::new(size_b, &config);

        Self {
            rank_r,
            alpha,
            in_dim,
            out_dim,
            lora_a: vec![0.01; size_a],
            lora_b: vec![0.0; size_b], // Initialized to zero so initial output is unchanged
            grad_a: vec![0.0; size_a],
            grad_b: vec![0.0; size_b],
            optimizer_a,
            optimizer_b,
            scheduler,
            current_step: 0,
            config,
        }
    }

    /// Forward pass through the low-rank adapter: delta = (alpha / r) * B * (A * x).
    pub fn forward(&self, x: &[f32], intermediate_r: &mut [f32], delta_out: &mut [f32]) {
        let scale = self.alpha / (self.rank_r as f32);

        // 1. intermediate = A * x (shape: [rank_r])
        for r in 0..self.rank_r {
            let mut dot = 0.0f32;
            let offset = r * self.in_dim;
            for i in 0..self.in_dim.min(x.len()) {
                dot += self.lora_a[offset + i] * x[i];
            }
            intermediate_r[r] = dot;
        }

        // 2. delta_out = scale * B * intermediate (shape: [out_dim])
        for o in 0..self.out_dim {
            let mut dot = 0.0f32;
            let offset = o * self.rank_r;
            for r in 0..self.rank_r {
                dot += self.lora_b[offset + r] * intermediate_r[r];
            }
            delta_out[o] = dot * scale;
        }
    }

    /// Backward pass computing analytical parameter gradients from upstream output gradient `grad_out`.
    pub fn backward(&mut self, x: &[f32], intermediate_r: &[f32], grad_out: &[f32]) {
        let scale = self.alpha / (self.rank_r as f32);

        // Gradient w.r.t lora_b: dL / dB_{o, r} = grad_out[o] * intermediate_r[r] * scale
        for o in 0..self.out_dim {
            let offset = o * self.rank_r;
            let g = grad_out.get(o).copied().unwrap_or(0.0) * scale;
            for r in 0..self.rank_r {
                self.grad_b[offset + r] += g * intermediate_r[r];
            }
        }

        // Upstream gradient through B: dL / d_intermediate[r] = sum_o (grad_out[o] * B_{o, r} * scale)
        let mut grad_inter = vec![0.0f32; self.rank_r];
        for r in 0..self.rank_r {
            let mut sum = 0.0f32;
            for o in 0..self.out_dim {
                let g = grad_out.get(o).copied().unwrap_or(0.0);
                sum += g * self.lora_b[o * self.rank_r + r];
            }
            grad_inter[r] = sum * scale;
        }

        // Gradient w.r.t lora_a: dL / dA_{r, i} = grad_inter[r] * x[i]
        for r in 0..self.rank_r {
            let offset = r * self.in_dim;
            let gi = grad_inter[r];
            for i in 0..self.in_dim.min(x.len()) {
                self.grad_a[offset + i] += gi * x[i];
            }
        }
    }

    /// Optimization step: updates parameters and resets gradient accumulators.
    pub fn step(&mut self) {
        let lr = self.scheduler.get_lr(self.current_step);
        self.optimizer_a.step(&mut self.lora_a, &self.grad_a, lr);
        self.optimizer_b.step(&mut self.lora_b, &self.grad_b, lr);

        self.grad_a.fill(0.0);
        self.grad_b.fill(0.0);
        self.current_step += 1;
    }
}
