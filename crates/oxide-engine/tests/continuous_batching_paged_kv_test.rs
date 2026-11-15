use oxide_alloc::{PagedKvArena, RadixPrefixCache, SequenceBlockTable};
use oxide_engine::continuous_batching::{ContinuousBatchingEngine, EngineHandle};
use oxide_models::{Llama3Config, Llama3Model};
use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_paged_kv_arena_allocation_and_release() {
    let arena = PagedKvArena::new(8, 2, 2, 32);
    assert_eq!(arena.free_block_count(), 8);

    let mut table1 = SequenceBlockTable::new();
    let (b0, off0) = table1.advance_token(&arena).expect("Allocated block 0");
    assert_eq!(off0, 0);
    assert_eq!(table1.block_ids.len(), 1);
    assert_eq!(arena.free_block_count(), 7);

    // Advance 15 more tokens (fill up page of 16)
    for _ in 1..16 {
        let (b, _) = table1.advance_token(&arena).expect("Within block 0");
        assert_eq!(b, b0);
    }
    assert_eq!(table1.num_tokens, 16);
    assert_eq!(arena.free_block_count(), 7);

    // 17th token triggers new block allocation
    let (b1, off1) = table1.advance_token(&arena).expect("Allocated block 1");
    assert_ne!(b0, b1);
    assert_eq!(off1, 0);
    assert_eq!(table1.block_ids.len(), 2);
    assert_eq!(arena.free_block_count(), 6);

    // Release all blocks
    table1.release_all(&arena);
    assert_eq!(arena.free_block_count(), 8);
}

#[tokio::test]
async fn test_radix_prefix_cache_integration() {
    let mut radix = RadixPrefixCache::new();

    let prompt1 = vec![100, 200, 300, 400];
    let blocks1 = vec![1, 2];
    radix.insert(&prompt1, &blocks1);

    // Exact match
    let (matched, cached_blocks) = radix.match_longest_prefix(&prompt1);
    assert_eq!(matched, 4);
    assert_eq!(cached_blocks, vec![1, 2]);

    // Partial match
    let prompt2 = vec![100, 200, 300, 500];
    let (matched2, _) = radix.match_longest_prefix(&prompt2);
    assert_eq!(matched2, 3);
}

#[tokio::test]
async fn test_continuous_batching_engine_multi_client_streaming() {
    let config = Llama3Config::tiny_test_config();
    let model = Arc::new(Llama3Model::new(config));

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let engine = ContinuousBatchingEngine::new(model, 4, 64, cmd_rx);
    let _worker_handle = engine.spawn();

    let client = EngineHandle::new(cmd_tx);

    // Submit request 1
    let prompt1 = vec![1, 15, 25, 35];
    let mut stream1 = client
        .generate_stream("req-1".to_string(), prompt1, 5, 0.0, 1.0)
        .await
        .expect("Stream 1 accepted");

    // Submit request 2 concurrently
    let prompt2 = vec![1, 15, 25, 35, 45];
    let mut stream2 = client
        .generate_stream("req-2".to_string(), prompt2, 5, 0.0, 1.0)
        .await
        .expect("Stream 2 accepted");

    let mut tokens1 = Vec::new();
    while let Some(event) = stream1.recv().await {
        tokens1.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }

    let mut tokens2 = Vec::new();
    while let Some(event) = stream2.recv().await {
        tokens2.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }

    assert_eq!(tokens1.len(), 5);
    assert_eq!(tokens2.len(), 5);
}

#[tokio::test]
async fn test_radix_prefix_cache_multi_turn_reuse() {
    let config = Llama3Config::tiny_test_config();
    let model = Arc::new(Llama3Model::new(config));

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let engine = ContinuousBatchingEngine::new(model, 4, 64, cmd_rx);
    let _worker_handle = engine.spawn();

    let client = EngineHandle::new(cmd_tx);

    // Turn 1: System prompt + User question
    let system_and_user_turn_1 = vec![1, 100, 200, 300, 400];
    let mut stream1 = client
        .generate_stream(
            "turn-1".to_string(),
            system_and_user_turn_1.clone(),
            3,
            0.0,
            1.0,
        )
        .await
        .expect("Turn 1 accepted");

    let mut turn1_generated = Vec::new();
    while let Some(event) = stream1.recv().await {
        turn1_generated.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }
    assert_eq!(turn1_generated.len(), 3);

    // Turn 2: Extends Turn 1 prompt + response with new follow-up question
    let mut turn_2_prompt = system_and_user_turn_1.clone();
    turn_2_prompt.extend_from_slice(&[500, 600]);

    let mut stream2 = client
        .generate_stream("turn-2".to_string(), turn_2_prompt, 4, 0.0, 1.0)
        .await
        .expect("Turn 2 accepted with prefix hit");

    let mut turn2_generated = Vec::new();
    while let Some(event) = stream2.recv().await {
        turn2_generated.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }
    assert_eq!(turn2_generated.len(), 4);
}

#[tokio::test]
async fn test_iteration_level_chunked_prefill() {
    let config = Llama3Config::tiny_test_config();
    let model = Arc::new(Llama3Model::new(config));

    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let mut engine = ContinuousBatchingEngine::new(model, 4, 128, cmd_rx);
    // Configure small chunk budget to enforce chunked prefill across multiple steps
    engine.prefill_config.min_chunk_tokens = 4;
    engine.prefill_config.max_chunk_tokens = 8;
    let _worker_handle = engine.spawn();

    let client = EngineHandle::new(cmd_tx);

    // Sequence 1: Long prompt that exceeds max_chunk_tokens
    let long_prompt: Vec<u32> = (1..=25).collect();
    let mut stream1 = client
        .generate_stream("req-long".to_string(), long_prompt, 6, 0.0, 1.0)
        .await
        .expect("Long prompt accepted");

    // Sequence 2: Short prompt arriving concurrently
    let short_prompt = vec![1, 2, 3];
    let mut stream2 = client
        .generate_stream("req-short".to_string(), short_prompt, 4, 0.0, 1.0)
        .await
        .expect("Short prompt accepted");

    let mut tokens1 = Vec::new();
    while let Some(event) = stream1.recv().await {
        tokens1.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }

    let mut tokens2 = Vec::new();
    while let Some(event) = stream2.recv().await {
        tokens2.push(event.token_id);
        if event.is_terminal {
            break;
        }
    }

    assert_eq!(tokens1.len(), 6);
    assert_eq!(tokens2.len(), 4);
}

#[tokio::test]
async fn test_radix_cache_lru_90_percent_eviction() {
    let arena = PagedKvArena::new(10, 1, 1, 16);
    let mut radix = RadixPrefixCache::new();

    // Allocate 9 blocks (90% capacity)
    let p1 = vec![1, 2, 3, 4];
    let b1 = vec![
        arena.allocate_block().unwrap(),
        arena.allocate_block().unwrap(),
    ];
    radix.insert(&p1, &b1);

    let p2 = vec![1, 2, 3, 5];
    let b2 = vec![
        arena.allocate_block().unwrap(),
        arena.allocate_block().unwrap(),
    ];
    radix.insert(&p2, &b2);

    let p3 = vec![10, 20, 30, 40];
    let b3 = vec![
        arena.allocate_block().unwrap(),
        arena.allocate_block().unwrap(),
    ];
    radix.insert(&p3, &b3);

    let p4 = vec![50, 60, 70, 80];
    let b4 = vec![
        arena.allocate_block().unwrap(),
        arena.allocate_block().unwrap(),
        arena.allocate_block().unwrap(),
    ];
    radix.insert(&p4, &b4);

    assert_eq!(arena.free_block_count(), 1); // 9 out of 10 used (90%)

    // Trigger LRU eviction down to budget
    let target_budget = 4; // tokens
    let evicted_blocks = radix.evict_to_budget_collecting_blocks(target_budget);
    assert!(!evicted_blocks.is_empty());

    for b in evicted_blocks {
        arena.release_block(b);
    }

    // Capacity is now reclaimed below 90%
    assert!(arena.free_block_count() > 1);
}
