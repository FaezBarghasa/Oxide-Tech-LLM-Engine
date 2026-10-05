use oxide_engine::arena::GraphArena;
use oxide_engine::executor::ComputeGraphExecutor;
use oxide_engine::graph::{ComputeGraph, NodeParams, OpCode, build_graph};
use oxide_models::loader::ModelMetadata;
use std::collections::HashMap;

fn create_mock_metadata() -> ModelMetadata {
    ModelMetadata {
        arch: "llama".to_string(),
        vocab_size: 32000,
        hidden_size: 256,
        intermediate_size: 512,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        head_dim: 64,
        max_seq_len: 2048,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        tensor_info: HashMap::new(),
    }
}

#[test]
fn test_compute_graph_construction() {
    let meta = create_mock_metadata();
    let graph = build_graph(&meta.arch, &meta, 1);
    assert!(!graph.nodes.is_empty());
    assert!(graph.tensor_count > graph.nodes.len());

    // Verify operations present in graph
    let has_fused_norm = graph.nodes.iter().any(|n| n.op == OpCode::FusedRmsMulMat);
    let has_mulmat = graph.nodes.iter().any(|n| n.op == OpCode::MulMat);
    let has_attn = graph.nodes.iter().any(|n| n.op == OpCode::FlashAttn);
    let has_swiglu = graph.nodes.iter().any(|n| n.op == OpCode::SwiGlu);

    assert!(has_fused_norm, "Graph should contain FusedRmsMulMat nodes");
    assert!(has_mulmat, "Graph should contain MulMat nodes");
    assert!(has_attn, "Graph should contain FlashAttn nodes");
    assert!(has_swiglu, "Graph should contain SwiGlu nodes");
}

#[test]
fn test_compute_graph_runtime_fusion() {
    let mut graph = ComputeGraph::new();
    let norm_params = NodeParams {
        eps: 1e-5,
        in_dim: 64,
        ..Default::default()
    };
    let norm_id = graph.add_node(OpCode::RmsNorm, 0, 0, 64, None, norm_params);

    let mul_params = NodeParams {
        in_dim: 64,
        out_dim: 64,
        ..Default::default()
    };
    let _mul_id = graph.add_node(
        OpCode::MulMat,
        norm_id,
        0,
        64,
        Some("test.weight".to_string()),
        mul_params,
    );

    assert_eq!(graph.nodes.len(), 2);

    // Apply fusion optimization pass
    graph.optimize_fusions();

    // The two nodes must fuse into a single FusedRmsMulMat node
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.nodes[0].op, OpCode::FusedRmsMulMat);
    assert_eq!(
        graph.nodes[0].weight_name.as_deref(),
        Some("test.weight")
    );
}

#[test]
fn test_graph_arena_zero_alloc_bump_allocator() {
    let mut arena = GraphArena::new(1024 * 1024); // 1 MB arena
    assert_eq!(arena.offset(), 0);
    assert_eq!(arena.capacity(), 1024 * 1024);

    {
        let slice1 = arena.alloc(128);
        assert_eq!(slice1.len(), 128);
        slice1[0] = 1.0;
        slice1[1] = 2.0;
        slice1[2] = 3.0;
    }
    assert_eq!(arena.offset(), 128);

    {
        let slice2 = arena.alloc(256);
        assert_eq!(slice2.len(), 256);
    }
    assert_eq!(arena.offset(), 384);

    assert_eq!(&arena.storage()[0..3], &[1.0, 2.0, 3.0]);

    // O(1) reset - zero-allocation scratch arena invariant
    arena.reset();
    assert_eq!(arena.offset(), 0);

    {
        let slice3 = arena.alloc(64);
        assert_eq!(slice3.len(), 64);
    }
    assert_eq!(arena.offset(), 64);
}

#[test]
fn test_compute_graph_executor_step_execution() {
    let meta = create_mock_metadata();
    let mut graph = build_graph(&meta.arch, &meta, 1);
    graph.optimize_fusions();

    let mut executor = ComputeGraphExecutor::new(1024 * 1024);
    executor.cache_weights(&graph, None);

    let initial_input = vec![0.1f32; 256];
    let output_slice = executor
        .execute_step(&graph, &initial_input)
        .expect("Execution of compute graph step should succeed");

    assert!(!output_slice.is_empty());
    // Invariants: Output must be valid floating point values without NaNs
    for &val in output_slice {
        assert!(!val.is_nan(), "Output contains NaN values");
    }

    // Successive step reusing arena
    let step2_slice = executor
        .execute_step(&graph, &initial_input)
        .expect("Successive step with arena reset");
    assert!(!step2_slice.is_empty());
}
