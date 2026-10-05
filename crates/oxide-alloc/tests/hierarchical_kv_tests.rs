use oxide_alloc::{CacheTierLocation, HierarchicalKvCache, TOKENS_PER_KV_BLOCK};

#[test]
fn test_hierarchical_kv_promotion_and_eviction() {
    // 2 Tier-1 blocks, 2 Tier-2 blocks, 2 Tier-3 blocks
    let mut cache = HierarchicalKvCache::new(2, 2, 2);

    let hash1 = HierarchicalKvCache::compute_prefix_hash(&[1, 2, 3, 4], 0);
    let hash2 = HierarchicalKvCache::compute_prefix_hash(&[5, 6, 7, 8], 0);
    let hash3 = HierarchicalKvCache::compute_prefix_hash(&[9, 10, 11, 12], 0);

    let blk1 = cache
        .allocate_block(hash1, 0, 64, vec![1.0; TOKENS_PER_KV_BLOCK * 64])
        .expect("blk1 allocated");
    let _blk2 = cache
        .allocate_block(hash2, 0, 64, vec![2.0; TOKENS_PER_KV_BLOCK * 64])
        .expect("blk2 allocated");

    assert_eq!(cache.tier1_count(), 2);
    assert_eq!(cache.tier2_count(), 0);

    // Allocating blk3 will evict blk1 to Tier 2
    let _blk3 = cache
        .allocate_block(hash3, 0, 64, vec![3.0; TOKENS_PER_KV_BLOCK * 64])
        .expect("blk3 allocated");

    assert_eq!(cache.tier1_count(), 2);
    assert_eq!(cache.tier2_count(), 1);

    // Finding hash1 promotes it back to Tier 1
    let (desc, data) = cache
        .find_or_promote_prefix(hash1)
        .expect("Prefix found and promoted");
    assert_eq!(desc.block_id, blk1);
    assert_eq!(desc.location, CacheTierLocation::Tier1DeviceVram);
    assert_eq!(data[0], 1.0);
}

#[test]
fn test_distributed_kv_export_import() {
    let mut cache_src = HierarchicalKvCache::new(4, 4, 4);
    let mut cache_dst = HierarchicalKvCache::new(4, 4, 4);

    let hash = HierarchicalKvCache::compute_prefix_hash(&[10, 20, 30], 0);
    let data = vec![0.5f32; TOKENS_PER_KV_BLOCK * 128]; // 64 K + 64 V
    let blk_id = cache_src
        .allocate_block(hash, 1, 64, data)
        .expect("Block allocated");

    let tokens = [1u32; TOKENS_PER_KV_BLOCK];
    let payload = cache_src
        .export_distributed_block(blk_id, tokens)
        .expect("Export successful");
    assert!(payload.verify_integrity());

    let imported_id = cache_dst
        .import_distributed_block(payload)
        .expect("Import successful");
    assert!(imported_id > 0);

    let (desc, imported_data) = cache_dst
        .find_or_promote_prefix(hash)
        .expect("Found imported prefix");
    assert_eq!(desc.prefix_hash, hash);
    assert_eq!(imported_data[0], 0.5);
}
