//! SGLang-Inspired Radix Tree Prefix Cache for Zero-Overhead Prompt Reuse.
//!
//! Organizes KV-cache block allocations into an adaptive LRU Radix Tree.
//! Allows sub-millisecond discovery of the longest matching cached token prefix
//! across multi-turn dialogs, system prompts, and shared context windows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A node in the KV-cache Radix Tree representing a contiguous sequence of tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RadixNode {
    pub tokens: Vec<u32>,
    pub block_ids: Vec<u32>,
    pub children: HashMap<u32, RadixNode>,
    pub access_count: u64,
    pub last_accessed: u64,
}

impl Default for RadixNode {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            block_ids: Vec::new(),
            children: HashMap::new(),
            access_count: 0,
            last_accessed: 0,
        }
    }
}

impl RadixNode {
    #[must_use]
    pub fn new(tokens: Vec<u32>, block_ids: Vec<u32>, time: u64) -> Self {
        Self {
            tokens,
            block_ids,
            children: HashMap::new(),
            access_count: 1,
            last_accessed: time,
        }
    }
}

/// SGLang-Grade Radix Tree Prefix Cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RadixPrefixCache {
    root: RadixNode,
    total_tokens: usize,
    clock: u64,
}

impl RadixPrefixCache {
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: RadixNode::default(),
            total_tokens: 0,
            clock: 0,
        }
    }

    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Finds the longest matching prefix for a prompt sequence.
    /// Returns `(matched_token_count, accumulated_block_ids)`.
    #[must_use]
    pub fn match_longest_prefix(&mut self, prompt: &[u32]) -> (usize, Vec<u32>) {
        self.clock += 1;
        let now = self.clock;

        let mut matched_len = 0;
        let mut matched_blocks = Vec::new();
        let mut curr = &mut self.root;
        curr.last_accessed = now;
        curr.access_count += 1;

        let mut remaining = prompt;

        while !remaining.is_empty() {
            let first = remaining[0];
            if !curr.children.contains_key(&first) {
                break;
            }

            let child = curr.children.get_mut(&first).unwrap();
            child.last_accessed = now;
            child.access_count += 1;

            // Match common prefix between child.tokens and remaining
            let common_len = child
                .tokens
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == 0 {
                break;
            }

            if common_len == child.tokens.len() {
                // Entire child node matched, advance
                matched_len += common_len;
                matched_blocks.extend_from_slice(&child.block_ids);
                remaining = &remaining[common_len..];
                curr = child;
            } else {
                // Partial match within this node
                matched_len += common_len;
                let blocks_to_take = (child.block_ids.len() * common_len) / child.tokens.len();
                matched_blocks.extend_from_slice(&child.block_ids[..blocks_to_take]);
                break;
            }
        }

        (matched_len, matched_blocks)
    }

    /// Inserts a sequence of tokens and their associated KV-cache block IDs into the Radix Tree.
    pub fn insert(&mut self, prompt: &[u32], block_ids: &[u32]) {
        if prompt.is_empty() {
            return;
        }

        self.clock += 1;
        let now = self.clock;
        let mut curr = &mut self.root;
        curr.last_accessed = now;

        let mut remaining_tokens = prompt;
        let mut remaining_blocks = block_ids;

        while !remaining_tokens.is_empty() {
            let first = remaining_tokens[0];

            if !curr.children.contains_key(&first) {
                // No branch exists: insert remaining tokens as new leaf
                self.total_tokens += remaining_tokens.len();
                curr.children.insert(
                    first,
                    RadixNode::new(
                        remaining_tokens.to_vec(),
                        remaining_blocks.to_vec(),
                        now,
                    ),
                );
                return;
            }

            // Existing branch: calculate common prefix
            let child = curr.children.get_mut(&first).unwrap();
            let common = child
                .tokens
                .iter()
                .zip(remaining_tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common < child.tokens.len() {
                // Split child node
                let split_tokens = child.tokens[common..].to_vec();
                let split_blocks = child.block_ids[common.min(child.block_ids.len())..].to_vec();
                let split_first = split_tokens[0];

                let split_node = RadixNode {
                    tokens: split_tokens,
                    block_ids: split_blocks,
                    children: std::mem::take(&mut child.children),
                    access_count: child.access_count,
                    last_accessed: child.last_accessed,
                };

                child.tokens.truncate(common);
                child.block_ids.truncate(common.min(child.block_ids.len()));
                child.children.insert(split_first, split_node);
            }

            remaining_tokens = &remaining_tokens[common..];
            remaining_blocks = &remaining_blocks[common.min(remaining_blocks.len())..];
            curr = child;
        }
    }

    fn evict_one_oldest_leaf(node: &mut RadixNode) -> Option<usize> {
        if node.children.is_empty() {
            return None;
        }

        let mut oldest_leaf_key = None;
        let mut oldest_leaf_time = u64::MAX;

        for (&k, child) in &node.children {
            if child.children.is_empty() && child.last_accessed < oldest_leaf_time {
                oldest_leaf_time = child.last_accessed;
                oldest_leaf_key = Some(k);
            }
        }

        if let Some(k) = oldest_leaf_key {
            let removed = node.children.remove(&k)?;
            return Some(removed.tokens.len());
        }

        let mut candidate_key = None;
        let mut min_time = u64::MAX;
        for (&k, child) in &node.children {
            if child.last_accessed < min_time {
                min_time = child.last_accessed;
                candidate_key = Some(k);
            }
        }

        if let Some(k) = candidate_key {
            let child = node.children.get_mut(&k)?;
            return Self::evict_one_oldest_leaf(child);
        }

        None
    }

    /// Prune cache down to `max_token_budget` by evicting oldest leaf nodes across the tree.
    pub fn evict_to_budget(&mut self, max_token_budget: usize) {
        while self.total_tokens > max_token_budget && !self.root.children.is_empty() {
            if let Some(freed) = Self::evict_one_oldest_leaf(&mut self.root) {
                self.total_tokens = self.total_tokens.saturating_sub(freed);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_tree_prefix_matching_and_split() {
        let mut cache = RadixPrefixCache::new();

        // 1. Insert prompt A: [1, 2, 3, 4] with blocks [10, 20, 30, 40]
        cache.insert(&[1, 2, 3, 4], &[10, 20, 30, 40]);
        assert_eq!(cache.total_tokens(), 4);

        // 2. Query exact match
        let (len, blocks) = cache.match_longest_prefix(&[1, 2, 3, 4]);
        assert_eq!(len, 4);
        assert_eq!(blocks, vec![10, 20, 30, 40]);

        // 3. Query partial prefix: [1, 2, 3, 99]
        let (len_partial, blocks_partial) = cache.match_longest_prefix(&[1, 2, 3, 99]);
        assert_eq!(len_partial, 3);
        assert_eq!(blocks_partial, vec![10, 20, 30]);

        // 4. Insert branching prompt B: [1, 2, 5, 6] with blocks [10, 20, 50, 60]
        cache.insert(&[1, 2, 5, 6], &[10, 20, 50, 60]);

        // 5. Query both branches
        let (len_a, _) = cache.match_longest_prefix(&[1, 2, 3, 4, 7]);
        assert_eq!(len_a, 4);

        let (len_b, _) = cache.match_longest_prefix(&[1, 2, 5, 6, 8]);
        assert_eq!(len_b, 4);

        // 6. Test eviction
        cache.evict_to_budget(4);
        assert!(cache.total_tokens() <= 4);
    }
}
