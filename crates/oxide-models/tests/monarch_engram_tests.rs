use oxide_models::engram::EngramGatherTable;
use oxide_models::monarch::MonarchMlp;
use oxide_models::needle::NeedleSubnetwork;

#[test]
fn test_monarch_mlp_forward() {
    let mlp = MonarchMlp::new(64);
    assert_eq!(mlp.hidden_dim, 64);
    assert_eq!(mlp.num_blocks, 8);
    assert_eq!(mlp.block_dim, 8);

    let input = vec![1.0f32; 64];
    let mut output = vec![0.0f32; 64];

    mlp.forward(&input, &mut output).expect("Monarch forward pass failed");
    assert!(output.iter().all(|&v| v.is_finite()));
}

#[test]
fn test_engram_gather_table() {
    let engram = EngramGatherTable::new(1024, 64);
    let tokens = [101u32, 205, 304];

    let slot = engram.compute_slot(&tokens);
    assert!(slot < 1024);

    let mut gathered = vec![0.0f32; 64];
    engram.gather(&tokens, &mut gathered).expect("Engram gather failed");
    assert_eq!(gathered.len(), 64);
}

#[test]
fn test_needle_subnetwork_depth_ladder() {
    let subnet_2l = NeedleSubnetwork::<2>::new();
    assert_eq!(subnet_2l.active_layers(), 2);

    let subnet_8l = NeedleSubnetwork::<8>::new();
    assert_eq!(subnet_8l.active_layers(), 8);

    let subnet_20l = NeedleSubnetwork::<20>::new();
    assert_eq!(subnet_20l.active_layers(), 20);
}
