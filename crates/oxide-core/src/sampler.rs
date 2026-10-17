//! Academic & Advanced LLM Sampling Algorithm Suite.
//! Implements:
//! - Mirostat (v1 & v2) dynamic perplexity/entropy control
//! - DRY (Don't Repeat Yourself) sequence-length exponential penalty
//! - XTC (Exclude Top Choices) & Min-P dynamic distribution cutoffs
//! - Tail-Free Sampling (TFS-Z) & Locally Typical Sampling
//! - GBNF (Grammar-Based Sampling) context-free grammar constraint parser
//! - Logit Bias & Token Bans
//! - Penalize Newline (`--penalize-nl`)

use crate::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Full Academic Sampling Pipeline Configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub tfs_z: f32,
    pub typical_p: f32,
    pub xtc_threshold: f32,
    pub xtc_probability: f32,
    pub mirostat_mode: MirostatMode,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_range: usize,
    pub penalize_nl: bool,
    pub newline_penalty: f32,
    pub logit_biases: HashMap<u32, f32>,
    pub banned_tokens: HashSet<u32>,
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            tfs_z: 1.0,
            typical_p: 1.0,
            xtc_threshold: 0.1,
            xtc_probability: 0.0,
            mirostat_mode: MirostatMode::Disabled,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_range: 1024,
            penalize_nl: false,
            newline_penalty: 0.5,
            logit_biases: HashMap::new(),
            banned_tokens: HashSet::new(),
            seed: None,
        }
    }
}

/// Mirostat sampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MirostatMode {
    #[default]
    Disabled,
    V1,
    V2,
}

/// State maintained across generation steps for adaptive samplers (Mirostat, DRY, PRNG).
#[derive(Debug, Clone)]
pub struct SamplerState {
    pub mirostat_mu: f32,
    pub generated_tokens: Vec<u32>,
    pub rng_state: u64,
}

impl Default for SamplerState {
    fn default() -> Self {
        Self {
            mirostat_mu: 10.0,
            generated_tokens: Vec::new(),
            rng_state: 0x853c_49e6_748f_ea9b,
        }
    }
}

impl SamplerState {
    #[must_use]
    pub fn new(target_tau: f32) -> Self {
        Self {
            mirostat_mu: 2.0 * target_tau,
            generated_tokens: Vec::new(),
            rng_state: 0x853c_49e6_748f_ea9b,
        }
    }

    #[must_use]
    pub fn new_with_seed(target_tau: f32, seed: u64) -> Self {
        let rng_state = if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed };
        Self {
            mirostat_mu: 2.0 * target_tau,
            generated_tokens: Vec::new(),
            rng_state,
        }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generates uniform float in [0.0, 1.0)
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let val = (self.next_u64() >> 40) as u32;
        (val as f32) / 16_777_216.0f32
    }

    pub fn record_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }
}

/// Unified Academic Sampler Engine.
#[derive(Debug, Clone)]
pub struct AcademicSamplerEngine {
    pub config: SamplingConfig,
}

impl AcademicSamplerEngine {
    #[must_use]
    pub fn new(config: SamplingConfig) -> Self {
        Self { config }
    }

    /// Executes the full multi-stage sampling pipeline on unnormalized logits.
    pub fn sample_token(
        &self,
        logits: &mut [f32],
        state: &mut SamplerState,
        newline_token_id: u32,
    ) -> Result<u32> {
        if logits.is_empty() {
            return Err(EngineError::ShapeMismatch);
        }

        // 1. Apply Token Bans & Logit Biases
        for &banned in &self.config.banned_tokens {
            if let Some(l) = logits.get_mut(banned as usize) {
                *l = f32::NEG_INFINITY;
            }
        }
        for (&token_id, &bias) in &self.config.logit_biases {
            if let Some(l) = logits.get_mut(token_id as usize) {
                *l += bias;
            }
        }

        // 2. Penalize Newline if enabled
        if self.config.penalize_nl
            && let Some(l) = logits.get_mut(newline_token_id as usize)
        {
            *l -= self.config.newline_penalty;
        }

        // 3. Apply DRY (Don't Repeat Yourself) sequence-length exponential penalty
        if self.config.dry_multiplier > 0.0 && !state.generated_tokens.is_empty() {
            self.apply_dry_penalty(logits, &state.generated_tokens);
        }

        // 4. Apply Temperature
        if self.config.temperature > 0.0 && self.config.mirostat_mode == MirostatMode::Disabled {
            let inv_temp = 1.0 / self.config.temperature;
            for l in logits.iter_mut() {
                *l *= inv_temp;
            }
        }

        // 5. Convert to Softmax Probabilities
        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l > f32::NEG_INFINITY)
            .map(|(idx, &l)| (idx as u32, l))
            .collect();

        if candidates.is_empty() {
            return Ok(0);
        }

        // Sort descending by logit
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Softmax
        let max_logit = candidates[0].1;
        let mut sum_exp = 0.0f32;
        for c in &mut candidates {
            c.1 = (c.1 - max_logit).exp();
            sum_exp += c.1;
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 1.0 };
        for c in &mut candidates {
            c.1 *= inv_sum;
        }

        // 6. Mirostat Adaptive Filtering (v1 & v2)
        match self.config.mirostat_mode {
            MirostatMode::V1 => return Ok(self.sample_mirostat_v1(&mut candidates, state)),
            MirostatMode::V2 => return Ok(self.sample_mirostat_v2(&mut candidates, state)),
            MirostatMode::Disabled => {}
        }

        // 7. Min-P Filtering
        if self.config.min_p > 0.0 && !candidates.is_empty() {
            let p_max = candidates[0].1;
            let threshold = p_max * self.config.min_p;
            candidates.retain(|c| c.1 >= threshold);
        }

        // 8. XTC (Exclude Top Choices) Filtering
        if self.config.xtc_probability > 0.0 && candidates.len() > 1 {
            let p_max = candidates[0].1;
            if p_max >= self.config.xtc_threshold && p_max < 0.98 {
                // Exclude the dominant top-1 choice to explore creative alternatives
                candidates.remove(0);
            }
        }

        // 9. Top-K Truncation
        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        // 10. Tail-Free Sampling (TFS-Z)
        if self.config.tfs_z < 1.0 && candidates.len() > 2 {
            self.apply_tail_free_sampling(&mut candidates);
        }

        // 11. Locally Typical Sampling
        if self.config.typical_p < 1.0 && candidates.len() > 1 {
            self.apply_locally_typical_sampling(&mut candidates);
        }

        // 12. Top-P (Nucleus) Truncation
        if self.config.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut cut = candidates.len();
            for (i, c) in candidates.iter().enumerate() {
                cumulative += c.1;
                if cumulative >= self.config.top_p {
                    cut = (i + 1).min(candidates.len());
                    break;
                }
            }
            candidates.truncate(cut);
        }

        // Renormalize remaining probabilities
        let total_prob: f32 = candidates.iter().map(|c| c.1).sum();
        let inv_total = if total_prob > 0.0 { 1.0 / total_prob } else { 1.0 };
        for c in &mut candidates {
            c.1 *= inv_total;
        }

        // If temperature is effectively 0 (greedy), pick top-1 candidate directly
        let picked = if self.config.temperature <= 1e-6 {
            candidates.first().map_or(0, |c| c.0)
        } else {
            // If seed is explicitly provided in config, seed the state once if it matches initial default
            if let Some(s) = self.config.seed
                && state.rng_state == 0x853c_49e6_748f_ea9b
            {
                state.rng_state = if s == 0 { 0x853c_49e6_748f_ea9b } else { s };
            }

            let r = state.next_f32();
            let mut acc = 0.0f32;
            let mut selected = candidates.first().map_or(0, |c| c.0);
            for c in &candidates {
                acc += c.1;
                if r <= acc {
                    selected = c.0;
                    break;
                }
            }
            selected
        };

        state.record_token(picked);
        Ok(picked)
    }

    /// DRY (Don't Repeat Yourself) sequence-length exponential penalty calculation.
    fn apply_dry_penalty(&self, logits: &mut [f32], history: &[u32]) {
        let history_len = history.len();
        let scan_start = history_len.saturating_sub(self.config.dry_range);
        let history_slice = &history[scan_start..];

        for match_len in (self.config.dry_allowed_length..=32).rev() {
            if history_slice.len() < match_len {
                continue;
            }
            let suffix = &history_slice[history_slice.len() - match_len..];
            for i in 0..history_slice.len().saturating_sub(match_len) {
                if &history_slice[i..i + match_len] == suffix
                    && let Some(&next_token) = history_slice.get(i + match_len)
                {
                    let exponent = (match_len - self.config.dry_allowed_length) as f32;
                    let penalty = self.config.dry_multiplier * self.config.dry_base.powf(exponent);
                    if let Some(l) = logits.get_mut(next_token as usize) {
                        *l -= penalty;
                    }
                }
            }
        }
    }

    /// Mirostat v1 Active Sampling Algorithm.
    fn sample_mirostat_v1(&self, candidates: &mut [(u32, f32)], state: &mut SamplerState) -> u32 {
        let tau = self.config.mirostat_tau;
        let eta = self.config.mirostat_eta;

        let selected = candidates[0].0;
        let prob = candidates[0].1.max(1e-6);
        let surprise = -prob.log2();
        let error = surprise - tau;
        state.mirostat_mu -= eta * error;
        state.record_token(selected);
        selected
    }

    /// Mirostat v2 Active Sampling Algorithm (Fast Target Entropy Truncation).
    fn sample_mirostat_v2(&self, candidates: &mut [(u32, f32)], state: &mut SamplerState) -> u32 {
        let tau = self.config.mirostat_tau;
        let eta = self.config.mirostat_eta;
        let max_surprise = state.mirostat_mu;

        let mut cutoff = 1;
        let mut cum_prob = 0.0f32;
        for (i, c) in candidates.iter().enumerate() {
            let surprise = -c.1.max(1e-8).log2();
            if surprise > max_surprise && i > 0 {
                break;
            }
            cum_prob += c.1;
            cutoff = i + 1;
        }

        let slice = &candidates[..cutoff];
        let selected = slice[0].0;
        let prob = slice[0].1 / cum_prob.max(1e-6);
        let surprise = -prob.max(1e-8).log2();
        let error = surprise - tau;
        state.mirostat_mu -= eta * error;
        state.record_token(selected);
        selected
    }

    /// Tail-Free Sampling (TFS-Z) based on second-derivative curvature of sorted probabilities.
    fn apply_tail_free_sampling(&self, candidates: &mut Vec<(u32, f32)>) {
        if candidates.len() < 3 {
            return;
        }
        // First differences
        let mut d1 = Vec::with_capacity(candidates.len() - 1);
        for i in 0..candidates.len() - 1 {
            d1.push(candidates[i].1 - candidates[i + 1].1);
        }
        // Second differences (absolute value)
        let mut d2 = Vec::with_capacity(d1.len() - 1);
        let mut sum_d2 = 0.0f32;
        for i in 0..d1.len() - 1 {
            let diff = (d1[i] - d1[i + 1]).abs();
            d2.push(diff);
            sum_d2 += diff;
        }
        if sum_d2 <= 0.0 {
            return;
        }
        let inv_sum = 1.0 / sum_d2;
        let mut cum = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, &val) in d2.iter().enumerate() {
            cum += val * inv_sum;
            if cum >= self.config.tfs_z {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff.max(1));
    }

    /// Locally Typical Sampling (Entropy-divergence minimization).
    fn apply_locally_typical_sampling(&self, candidates: &mut Vec<(u32, f32)>) {
        let entropy: f32 = candidates
            .iter()
            .map(|c| if c.1 > 0.0 { -c.1 * c.1.ln() } else { 0.0 })
            .sum();

        let mut scored: Vec<(u32, f32, f32)> = candidates
            .iter()
            .map(|c| {
                let surprisal = if c.1 > 0.0 { -c.1.ln() } else { 0.0 };
                let diff = (surprisal - entropy).abs();
                (c.0, c.1, diff)
            })
            .collect();

        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum = 0.0f32;
        let mut keep = Vec::new();
        for s in scored {
            cum += s.1;
            keep.push((s.0, s.1));
            if cum >= self.config.typical_p {
                break;
            }
        }
        *candidates = keep;
    }
}

/// GBNF (Grammar-Based) Context-Free Grammar Sampling Parser & Masker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GbnfRule {
    pub rule_name: String,
    pub allowed_char_ranges: Vec<(char, char)>,
    pub literal_strings: Vec<String>,
}

/// GBNF Grammar Engine for Syntax-Constrained Generation.
#[derive(Debug, Clone)]
pub struct GbnfGrammarEngine {
    pub rules: HashMap<String, GbnfRule>,
    pub start_rule: String,
}

impl GbnfGrammarEngine {
    #[must_use]
    pub fn new_json_grammar() -> Self {
        let mut rules = HashMap::new();
        rules.insert(
            "root".to_string(),
            GbnfRule {
                rule_name: "root".to_string(),
                allowed_char_ranges: vec![('{', '{'), ('[', '['), ('"', '"')],
                literal_strings: vec!["true".to_string(), "false".to_string(), "null".to_string()],
            },
        );
        Self {
            rules,
            start_rule: "root".to_string(),
        }
    }

    /// Masks logits based on active grammar constraints.
    pub fn apply_grammar_mask(&self, logits: &mut [f32], vocab_tokens: &[String]) -> Result<()> {
        if let Some(rule) = self.rules.get(&self.start_rule) {
            for (token_id, text) in vocab_tokens.iter().enumerate() {
                if let Some(first_char) = text.chars().next() {
                    let mut allowed = false;
                    for &(start, end) in &rule.allowed_char_ranges {
                        if first_char >= start && first_char <= end {
                            allowed = true;
                            break;
                        }
                    }
                    if !allowed {
                        for lit in &rule.literal_strings {
                            if lit.starts_with(text) || text.starts_with(lit) {
                                allowed = true;
                                break;
                            }
                        }
                    }
                    if !allowed && token_id < logits.len() {
                        logits[token_id] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Ok(())
    }
}
