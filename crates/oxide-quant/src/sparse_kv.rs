//! SparseEngine: Adaptive KV-Cache Sparsification & Eviction Subsystem.
//!
//! Production implementation of core sparsity policies ported from SparseEngine:
//! - `AttentionSink` (StreamingLLM): Preserves initial sink tokens + local sliding window.
//! - `H2O` (Heavy Hitter Oracle): Tracks cumulative attention importance per layer/head,
//!   compacting storage to top heavy-hitters + recent suffix.
//! - `Quest` (Query-Aware Sparsity): Divides KV into pages, precomputes min/max key bounds,
//!   and selects top-K candidate pages per decode step without dense scans.

use oxide_core::error::{EngineError, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configured KV-cache sparsity policy.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SparseKvPolicy {
    /// Dense baseline (preserves all tokens).
    #[default]
    Vanilla,
    /// StreamingLLM attention sink: fixed initial tokens + sliding window.
    AttentionSink {
        num_sinks: usize,
        window_size: usize,
    },
    /// Heavy Hitter Oracle: top-K cumulative attention scores + recent window.
    H2O {
        heavy_hitters: usize,
        recent_window: usize,
    },
    /// Quest: query-aware page selection with min/max key bounding boxes.
    Quest {
        page_size: usize,
        top_k_pages: usize,
    },
}

/// Token attention score item for priority queue selection.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredToken {
    pos: usize,
    score: f32,
}

impl Eq for ScoredToken {}

impl Ord for ScoredToken {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredToken {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Heavy Hitter Oracle (H2O) cumulative attention tracker.
#[derive(Debug, Clone)]
pub struct H2OTracker {
    num_heads: usize,
    heavy_hitters: usize,
    recent_window: usize,
    /// Cumulative attention score per sequence position: shape `[num_heads, max_seq_len]`.
    cumulative_scores: Vec<f32>,
    current_seq_len: usize,
}

impl H2OTracker {
    #[must_use]
    pub fn new(num_heads: usize, heavy_hitters: usize, recent_window: usize) -> Self {
        Self {
            num_heads,
            heavy_hitters,
            recent_window,
            cumulative_scores: Vec::new(),
            current_seq_len: 0,
        }
    }

    /// Records newly observed attention scores of shape `[num_heads, current_seq_len]`.
    pub fn record_attention_step(&mut self, attn_weights: &[f32], seq_len: usize) -> Result<()> {
        let expected_len = self.num_heads * seq_len;
        if attn_weights.len() < expected_len {
            return Err(EngineError::ShapeMismatch);
        }

        if seq_len > self.current_seq_len {
            let additional_tokens = seq_len - self.current_seq_len;
            let total_capacity = self.num_heads * seq_len;
            if self.cumulative_scores.len() < total_capacity {
                self.cumulative_scores.resize(total_capacity, 0.0);
            }
            self.current_seq_len = seq_len;
            let _ = additional_tokens;
        }

        for h in 0..self.num_heads {
            let base_offset = h * seq_len;
            for p in 0..seq_len {
                self.cumulative_scores[base_offset + p] += attn_weights[base_offset + p];
            }
        }

        Ok(())
    }

    /// Determines the set of token positions to retain for head `h`.
    #[must_use]
    pub fn compute_retained_positions(&self, head_idx: usize) -> Vec<usize> {
        let seq_len = self.current_seq_len;
        if seq_len <= self.heavy_hitters + self.recent_window {
            return (0..seq_len).collect();
        }

        let base_offset = head_idx * seq_len;
        let recent_start = seq_len.saturating_sub(self.recent_window);

        // Recent window tokens are unconditionally retained
        let mut retained: Vec<usize> = (recent_start..seq_len).collect();

        // Candidates for heavy hitters are tokens before the recent window
        let mut heap = BinaryHeap::with_capacity(recent_start);
        for p in 0..recent_start {
            heap.push(ScoredToken {
                pos: p,
                score: self.cumulative_scores[base_offset + p],
            });
        }

        let num_hh = self.heavy_hitters.min(heap.len());
        for _ in 0..num_hh {
            if let Some(item) = heap.pop() {
                retained.push(item.pos);
            }
        }

        retained.sort_unstable();
        retained
    }
}

/// Quest: Page bounding box storing coordinate-wise min and max Key vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestKeyBoundingBox {
    pub page_idx: usize,
    pub start_pos: usize,
    pub num_tokens: usize,
    pub min_k: Vec<f32>,
    pub max_k: Vec<f32>,
}

/// Quest query-aware page manager.
#[derive(Debug, Clone)]
pub struct QuestPageManager {
    page_size: usize,
    top_k_pages: usize,
    head_dim: usize,
    pages: Vec<QuestKeyBoundingBox>,
    current_tokens: usize,
}

impl QuestPageManager {
    #[must_use]
    pub fn new(page_size: usize, top_k_pages: usize, head_dim: usize) -> Self {
        Self {
            page_size: page_size.max(1),
            top_k_pages,
            head_dim,
            pages: Vec::new(),
            current_tokens: 0,
        }
    }

    /// Appends incoming Key vectors `[num_tokens, head_dim]` and updates page bounding boxes.
    pub fn append_keys(&mut self, k_tokens: &[f32], num_tokens: usize) -> Result<()> {
        if k_tokens.len() < num_tokens * self.head_dim {
            return Err(EngineError::ShapeMismatch);
        }

        for t in 0..num_tokens {
            let global_pos = self.current_tokens + t;
            let page_idx = global_pos / self.page_size;
            let k_vec = &k_tokens[t * self.head_dim..(t + 1) * self.head_dim];

            if page_idx >= self.pages.len() {
                self.pages.push(QuestKeyBoundingBox {
                    page_idx,
                    start_pos: global_pos,
                    num_tokens: 1,
                    min_k: k_vec.to_vec(),
                    max_k: k_vec.to_vec(),
                });
            } else {
                let page = &mut self.pages[page_idx];
                page.num_tokens += 1;
                for d in 0..self.head_dim {
                    if k_vec[d] < page.min_k[d] {
                        page.min_k[d] = k_vec[d];
                    }
                    if k_vec[d] > page.max_k[d] {
                        page.max_k[d] = k_vec[d];
                    }
                }
            }
        }

        self.current_tokens += num_tokens;
        Ok(())
    }

    /// Selects top-K candidate pages by computing the upper bound of dot product with Query `q`.
    ///
    /// Formula for upper bound score of page $p$ with query $q$:
    /// $$\text{score}(p) = \sum_{d} \max(q[d] \cdot \min_k[d], q[d] \cdot \max_k[d])$$
    #[must_use]
    pub fn select_top_k_pages(&self, q: &[f32]) -> Vec<usize> {
        if self.pages.is_empty() || q.len() < self.head_dim {
            return Vec::new();
        }

        if self.pages.len() <= self.top_k_pages {
            return (0..self.pages.len()).collect();
        }

        let mut scored_pages = Vec::with_capacity(self.pages.len());
        for page in &self.pages {
            let mut bound_score = 0.0f32;
            for d in 0..self.head_dim {
                let q_val = q[d];
                let prod_min = q_val * page.min_k[d];
                let prod_max = q_val * page.max_k[d];
                bound_score += prod_min.max(prod_max);
            }
            scored_pages.push((page.page_idx, bound_score));
        }

        scored_pages.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut selected: Vec<usize> = scored_pages
            .into_iter()
            .take(self.top_k_pages)
            .map(|(idx, _)| idx)
            .collect();

        // Always include the latest active page to ensure recent context continuity
        let last_page = self.pages.len() - 1;
        if !selected.contains(&last_page) && !selected.is_empty() {
            let last_idx = selected.len() - 1;
            selected[last_idx] = last_page;
        }

        selected.sort_unstable();
        selected
    }

    #[must_use]
    pub fn pages(&self) -> &[QuestKeyBoundingBox] {
        &self.pages
    }
}

/// Evaluator for `AttentionSink` policy.
#[must_use]
pub fn compute_attention_sink_retained_positions(
    current_seq_len: usize,
    num_sinks: usize,
    window_size: usize,
) -> Vec<usize> {
    if current_seq_len <= num_sinks + window_size {
        return (0..current_seq_len).collect();
    }

    let mut retained = Vec::with_capacity(num_sinks + window_size);
    // Initial sink tokens
    for p in 0..num_sinks {
        retained.push(p);
    }
    // Recent sliding window
    let window_start = current_seq_len.saturating_sub(window_size);
    for p in window_start..current_seq_len {
        if p >= num_sinks {
            retained.push(p);
        }
    }

    retained
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_sink_retained_positions() {
        let seq_len = 100;
        let num_sinks = 4;
        let window_size = 8;

        let retained = compute_attention_sink_retained_positions(seq_len, num_sinks, window_size);
        assert_eq!(retained.len(), 4 + 8);
        assert_eq!(&retained[0..4], &[0, 1, 2, 3]);
        assert_eq!(&retained[4..], &[92, 93, 94, 95, 96, 97, 98, 99]);
    }

    #[test]
    fn test_h2o_tracker_selection() {
        let mut tracker = H2OTracker::new(1, 2, 2);

        // 6 tokens: attention scores [10.0, 1.0, 8.0, 2.0, 3.0, 4.0]
        // Recent window of size 2 -> tokens 4, 5 unconditionally kept.
        // Candidates for heavy hitters: tokens 0..4 (scores: 10.0, 1.0, 8.0, 2.0).
        // Top 2 heavy hitters: token 0 (10.0) and token 2 (8.0).
        // Retained: [0, 2, 4, 5].
        let scores = vec![10.0, 1.0, 8.0, 2.0, 3.0, 4.0];
        tracker.record_attention_step(&scores, 6).expect("record");

        let retained = tracker.compute_retained_positions(0);
        assert_eq!(retained, vec![0, 2, 4, 5]);
    }

    #[test]
    fn test_quest_page_selection() {
        let mut manager = QuestPageManager::new(2, 2, 2);

        // 6 tokens across 3 pages (page_size = 2):
        // Page 0: tokens 0, 1: keys [1.0, 1.0], [2.0, 2.0] -> min [1, 1], max [2, 2]
        // Page 1: tokens 2, 3: keys [10.0, 10.0], [12.0, 12.0] -> min [10, 10], max [12, 12]
        // Page 2: tokens 4, 5: keys [0.1, 0.1], [0.2, 0.2] -> min [0.1, 0.1], max [0.2, 0.2]
        let keys = vec![
            1.0, 1.0, 2.0, 2.0, 10.0, 10.0, 12.0, 12.0, 0.1, 0.1, 0.2, 0.2,
        ];
        manager.append_keys(&keys, 6).expect("append keys");
        assert_eq!(manager.pages().len(), 3);

        // Query q = [1.0, 1.0] -> page 1 produces highest dot product bounds
        let q = vec![1.0, 1.0];
        let selected = manager.select_top_k_pages(&q);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1)); // Page 1 has largest bound
        assert!(selected.contains(&2)); // Page 2 is the latest active page
    }
}
