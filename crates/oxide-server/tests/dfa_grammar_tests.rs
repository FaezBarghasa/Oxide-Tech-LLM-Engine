use oxide_server::dfa::DfaSchemaGrammar;

#[test]
fn test_dfa_json_validator() {
    let dfa = DfaSchemaGrammar::new_simple_json_validator();

    // Valid JSON prefix: {"key":123}
    let valid_bytes = b"{\"key\":123}";
    let mut state = 0;
    for &b in valid_bytes {
        state = dfa.transition(state, b).expect("Valid byte transition failed");
    }
    assert!(dfa.accepting_states[state as usize]);

    // Invalid JSON: starts with number instead of {
    assert!(dfa.transition(0, b'5').is_err());
}

#[test]
fn test_dfa_logit_masking() {
    let dfa = DfaSchemaGrammar::new_simple_json_validator();
    let token_vocab = vec![
        b"{\"".to_vec(),   // token 0: valid after state 0
        b"123".to_vec(),   // token 1: invalid after state 0
        b"hello".to_vec(), // token 2: invalid after state 0
    ];

    let mut logits = vec![10.0f32, 10.0, 10.0];
    dfa.apply_dfa_mask(0, &token_vocab, &mut logits);

    assert_eq!(logits[0], 10.0);
    assert_eq!(logits[1], f32::NEG_INFINITY);
    assert_eq!(logits[2], f32::NEG_INFINITY);
}
