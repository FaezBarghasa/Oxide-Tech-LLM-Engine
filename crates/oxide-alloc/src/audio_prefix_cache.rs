//! Blake3-Keyed Paged Audio KV-Prefix Caching.
//!
//! Avoids redundant prefill compute on multi-turn audio analysis queries by pinning
//! and reusing cached KV pages indexed by `(model_id, lora_id, blake3_audio_hash)`.

use std::collections::HashMap;

/// Cache key uniquely identifying an immutable audio sequence under an active model and LoRA adapter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioKvCacheKey {
    pub model_id: String,
    pub active_lora_id: Option<String>,
    pub blake3_audio_hash: [u8; 32],
}

/// Paged KV Cache block allocation entry for an audio prefix.
#[derive(Debug, Clone)]
pub struct CachedAudioPrefixEntry {
    pub page_block_ids: Vec<u32>,
    pub num_audio_tokens: usize,
    pub last_accessed_tick: u64,
}

/// Global Blake3 Audio KV-Prefix Cache.
#[derive(Debug, Default)]
pub struct Blake3AudioPrefixCache {
    entries: HashMap<AudioKvCacheKey, CachedAudioPrefixEntry>,
    current_tick: u64,
}

impl Blake3AudioPrefixCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up pre-allocated KV pages for an audio hash.
    #[must_use]
    pub fn lookup(
        &mut self,
        model_id: &str,
        active_lora_id: Option<&str>,
        blake3_audio_hash: &[u8; 32],
    ) -> Option<&[u32]> {
        self.current_tick += 1;
        let key = AudioKvCacheKey {
            model_id: model_id.to_string(),
            active_lora_id: active_lora_id.map(ToString::to_string),
            blake3_audio_hash: *blake3_audio_hash,
        };

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed_tick = self.current_tick;
            Some(&entry.page_block_ids)
        } else {
            None
        }
    }

    /// Stores or updates KV page block mappings for an audio prefix.
    pub fn insert(
        &mut self,
        model_id: &str,
        active_lora_id: Option<&str>,
        blake3_audio_hash: [u8; 32],
        page_block_ids: Vec<u32>,
        num_audio_tokens: usize,
    ) {
        self.current_tick += 1;
        let key = AudioKvCacheKey {
            model_id: model_id.to_string(),
            active_lora_id: active_lora_id.map(ToString::to_string),
            blake3_audio_hash,
        };

        let entry = CachedAudioPrefixEntry {
            page_block_ids,
            num_audio_tokens,
            last_accessed_tick: self.current_tick,
        };
        self.entries.insert(key, entry);
    }

    /// Evicts LRU audio cache entries if total capacity exceeds maximum entries.
    pub fn prune_lru(&mut self, max_cached_audios: usize) {
        if self.entries.len() <= max_cached_audios {
            return;
        }

        let mut sorted: Vec<(AudioKvCacheKey, u64)> = self
            .entries
            .iter()
            .map(|(k, v)| (k.clone(), v.last_accessed_tick))
            .collect();
        sorted.sort_by_key(|&(_, tick)| tick);

        let excess = self.entries.len() - max_cached_audios;
        for (k, _) in sorted.into_iter().take(excess) {
            self.entries.remove(&k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_audio_kv_cache_lifecycle() {
        let mut cache = Blake3AudioPrefixCache::new();
        let hash = [0xAB; 32];

        assert!(cache.lookup("qwen-audio", None, &hash).is_none());

        cache.insert("qwen-audio", None, hash, vec![101, 102, 103], 48);
        let blocks = cache.lookup("qwen-audio", None, &hash).expect("Found");
        assert_eq!(blocks, &[101, 102, 103]);

        // LoRA adapter separation test
        assert!(
            cache
                .lookup("qwen-audio", Some("mix_engineer"), &hash)
                .is_none()
        );
    }
}
