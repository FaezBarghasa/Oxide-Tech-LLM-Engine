//! System 1 Decision Scoring Engine & Shared-Prefix Multi-Prompt Evaluator.
//!
//! Provides ultra-low-latency classification, decision scoring, and candidate label ranking
//! directly from model LM-head logits without autoregressive generation loops.
//! Implements longest common token prefix extraction (`shared_prefix_len`) to prefill
//! shared context once and branch for candidate evaluation.

use oxide_core::error::{EngineError, Result};
use std::collections::HashMap;

/// Maximum number of prompts scored in a single decision batch.
pub const MAX_LABEL_PROMPTS: usize = 256;

/// Maximum number of candidate labels evaluated per prompt.
pub const MAX_LABELS_PER_PROMPT: usize = 255;

/// Prompt specification with target candidate labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelPrompt {
    pub prompt: String,
    pub labels: Vec<String>,
}

/// Tokenized prompt with mapped candidate label token IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedLabelPrompt {
    pub prompt_tokens: Vec<u32>,
    pub label_token_ids: Vec<u32>,
}

/// Result of evaluating candidate labels for a single prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelPromptResult {
    /// Total tokens in this prompt, including any shared prefix.
    pub tokens: usize,
    /// Token IDs corresponding to the candidate labels.
    pub label_token_ids: Vec<u32>,
    /// Raw unnormalized LM-head logits at the prompt's terminal position.
    pub logits: Vec<f32>,
    /// Softmax-normalized probabilities over the candidate labels.
    pub probabilities: Vec<f32>,
    /// Index of the highest scoring candidate label.
    pub best_label_idx: usize,
}

/// Aggregate output of a decision batch evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelLogits {
    /// Evaluation results per prompt in corresponding order.
    pub results: Vec<LabelPromptResult>,
    /// Number of tokens in the common prefix evaluated only once.
    pub shared_tokens: usize,
}

/// Computes the length of the longest token prefix common to all prompts.
///
/// Guaranteed to leave at least one token for each prompt (since logits
/// must be read from each prompt's last position). Returns 0 if fewer
/// than two prompts are provided or if no common prefix exists.
#[must_use]
pub fn shared_prefix_len(prompts: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = prompts.split_first() else {
        return 0;
    };
    if rest.is_empty() {
        return 0;
    }
    let mut n = first.len().saturating_sub(1);
    for ids in rest {
        let common = first.iter().zip(ids).take_while(|(a, b)| a == b).count();
        n = n.min(common).min(ids.len().saturating_sub(1));
    }
    n
}

/// Computes numerically stable softmax over a slice of raw logits.
#[must_use]
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature <= 0.0 { 1.0 } else { temperature };
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

    let exps: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / temp).exp())
        .collect();
    let sum_exp: f32 = exps.iter().sum();

    if sum_exp > 0.0 {
        exps.into_iter().map(|x| x / sum_exp).collect()
    } else {
        vec![1.0 / (logits.len() as f32); logits.len()]
    }
}

/// System 1 Decision & Logit Extractor Engine.
#[derive(Debug, Clone, Default)]
pub struct DecisionScorer;

impl DecisionScorer {
    /// Constructs a new decision scorer instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Evaluates raw candidate label logits from a full vocabulary logit distribution.
    ///
    /// # Arguments
    /// * `prompt_tokens_len` - Full token length of the prompt.
    /// * `label_token_ids` - Slice of token IDs corresponding to the candidate labels.
    /// * `vocab_logits` - Complete vocabulary logits computed at the prompt's final token.
    /// * `temperature` - Temperature scaling factor for softmax normalization.
    pub fn score_labels_from_vocab(
        &self,
        prompt_tokens_len: usize,
        label_token_ids: &[u32],
        vocab_logits: &[f32],
        temperature: f32,
    ) -> Result<LabelPromptResult> {
        if label_token_ids.is_empty() {
            return Err(EngineError::BackendError(
                "candidate labels must not be empty".to_string(),
            ));
        }

        let mut logits = Vec::with_capacity(label_token_ids.len());
        for &token_id in label_token_ids {
            let idx = token_id as usize;
            if idx >= vocab_logits.len() {
                return Err(EngineError::BackendError(format!(
                    "label token ID {token_id} out of vocabulary bounds {}",
                    vocab_logits.len()
                )));
            }
            logits.push(vocab_logits[idx]);
        }

        let probabilities = softmax(&logits, temperature);

        let best_label_idx = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        Ok(LabelPromptResult {
            tokens: prompt_tokens_len,
            label_token_ids: label_token_ids.to_vec(),
            logits,
            probabilities,
            best_label_idx,
        })
    }

    /// Tokenizes and validates a batch of label prompts with single-token label validation.
    pub fn validate_and_tokenize<F>(
        &self,
        prompts: &[LabelPrompt],
        encode_fn: F,
    ) -> Result<Vec<TokenizedLabelPrompt>>
    where
        F: Fn(&str) -> Result<Vec<u32>>,
    {
        if prompts.is_empty() {
            return Err(EngineError::BackendError(
                "decision prompts must not be empty".to_string(),
            ));
        }
        if prompts.len() > MAX_LABEL_PROMPTS {
            return Err(EngineError::BackendError(format!(
                "exceeded MAX_LABEL_PROMPTS: {} > {MAX_LABEL_PROMPTS}",
                prompts.len()
            )));
        }

        let mut label_cache: HashMap<&str, u32> = HashMap::new();
        let mut tokenized = Vec::with_capacity(prompts.len());

        for (i, p) in prompts.iter().enumerate() {
            if p.labels.is_empty() || p.labels.len() > MAX_LABELS_PER_PROMPT {
                return Err(EngineError::BackendError(format!(
                    "prompt {i} has {} labels, must be 1 to {MAX_LABELS_PER_PROMPT}",
                    p.labels.len()
                )));
            }

            let prompt_tokens = encode_fn(&p.prompt)?;
            if prompt_tokens.is_empty() {
                return Err(EngineError::BackendError(format!(
                    "prompt {i} tokenized to 0 tokens"
                )));
            }

            let mut label_token_ids = Vec::with_capacity(p.labels.len());
            for label in &p.labels {
                let id = if let Some(&id) = label_cache.get(label.as_str()) {
                    id
                } else {
                    let toks = encode_fn(label)?;
                    if toks.len() != 1 {
                        return Err(EngineError::BackendError(format!(
                            "label {label:?} encoded to {} tokens, but must be exactly 1 token",
                            toks.len()
                        )));
                    }
                    let id = toks[0];
                    label_cache.insert(label.as_str(), id);
                    id
                };
                label_token_ids.push(id);
            }

            tokenized.push(TokenizedLabelPrompt {
                prompt_tokens,
                label_token_ids,
            });
        }

        Ok(tokenized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_prefix_len() {
        assert_eq!(shared_prefix_len(&[]), 0);
        assert_eq!(shared_prefix_len(&[vec![1, 2, 3]]), 0);

        let p1 = vec![100, 200, 300, 400, 501];
        let p2 = vec![100, 200, 300, 400, 502];
        let p3 = vec![100, 200, 300, 400, 503];
        // Common prefix is [100, 200, 300, 400] -> length 4, each prompt has 1 unique token.
        assert_eq!(shared_prefix_len(&[p1.clone(), p2.clone(), p3.clone()]), 4);

        // Even if all prompts are identical, must leave at least 1 token for each prompt
        let p_same1 = vec![1, 2, 3];
        let p_same2 = vec![1, 2, 3];
        assert_eq!(shared_prefix_len(&[p_same1, p_same2]), 2);

        // Disjoint prefixes
        let p_diff = vec![999, 200, 300];
        assert_eq!(shared_prefix_len(&[p1, p_diff]), 0);
    }

    #[test]
    fn test_softmax_and_scoring() {
        let logits = vec![2.0, 1.0, 0.1];
        let probs = softmax(&logits, 1.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);

        let scorer = DecisionScorer::new();
        let vocab_logits = vec![0.0, 10.0, 2.5, 8.0, -1.0];
        let label_token_ids = vec![1, 3]; // logits are 10.0 and 8.0
        let result = scorer
            .score_labels_from_vocab(10, &label_token_ids, &vocab_logits, 1.0)
            .expect("score labels");

        assert_eq!(result.tokens, 10);
        assert_eq!(result.label_token_ids, vec![1, 3]);
        assert_eq!(result.logits, vec![10.0, 8.0]);
        assert_eq!(result.best_label_idx, 0); // 10.0 > 8.0
        assert!(result.probabilities[0] > 0.85);
    }
}
