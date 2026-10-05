use oxide_core::sampler::{
    AcademicSamplerEngine, GbnfGrammarEngine, GbnfRule, MirostatMode, SamplerState, SamplingConfig,
};
use std::collections::HashMap;

#[test]
fn test_academic_sampler_temperature_and_top_k() {
    let mut logits = vec![1.0, 2.0, 5.0, 3.0, 0.5];
    let config = SamplingConfig {
        temperature: 0.5,
        top_k: 2,
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);

    let picked = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    assert_eq!(picked, 2); // Logit 5.0 is highest
    assert_eq!(state.generated_tokens, vec![2]);
}

#[test]
fn test_academic_sampler_min_p_and_xtc() {
    let mut logits = vec![10.0, 9.5, 2.0, 1.0];
    let config = SamplingConfig {
        min_p: 0.1,
        xtc_threshold: 0.5,
        xtc_probability: 1.0,
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);

    let picked = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    // With XTC active, top choice (token 0) is excluded, yielding token 1
    assert_eq!(picked, 1);
}

#[test]
fn test_dry_repetition_penalty() {
    let mut logits = vec![5.0, 5.0, 5.0];
    let config = SamplingConfig {
        dry_multiplier: 2.0,
        dry_base: 1.5,
        dry_allowed_length: 2,
        dry_range: 16,
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);
    state.generated_tokens = vec![0, 1, 0, 1, 0, 1]; // Repetitive pattern 0, 1

    let picked = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    // Sequence [0, 1] matches history, penalizing next token (0), so picked is not 0
    assert_ne!(picked, 0);
}

#[test]
fn test_mirostat_v2_sampling() {
    let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let config = SamplingConfig {
        mirostat_mode: MirostatMode::V2,
        mirostat_tau: 3.0,
        mirostat_eta: 0.1,
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(3.0);

    let picked = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    assert_eq!(picked, 4); // Highest probability token within target entropy
}

#[test]
fn test_gbnf_grammar_constrained_masking() {
    let mut logits = vec![10.0, 10.0, 10.0, 10.0];
    let mut rules = HashMap::new();
    rules.insert(
        "root".to_string(),
        GbnfRule {
            rule_name: "root".to_string(),
            allowed_char_ranges: vec![('{', '{')],
            literal_strings: vec!["true".to_string()],
        },
    );
    let grammar = GbnfGrammarEngine {
        rules,
        start_rule: "root".to_string(),
    };

    let vocab = vec![
        "\"key\"".to_string(), // not starting with { or true -> masked
        "{".to_string(),       // starts with { -> allowed
        "123".to_string(),     // masked
        "true".to_string(),    // matches literal -> allowed
    ];

    grammar.apply_grammar_mask(&mut logits, &vocab).unwrap();
    assert_eq!(logits[0], f32::NEG_INFINITY);
    assert_eq!(logits[1], 10.0);
    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert_eq!(logits[3], 10.0);
}

#[test]
fn test_logit_bias_and_token_bans() {
    let mut logits = vec![10.0, 5.0, 1.0];
    let mut config = SamplingConfig::default();
    config.banned_tokens.insert(0); // Ban top token 0
    config.logit_biases.insert(2, 20.0); // Boost token 2 by +20.0

    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);

    let picked = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    assert_eq!(picked, 2);
}

#[test]
fn test_penalize_newline() {
    let mut logits = vec![5.0, 5.0, 5.0];
    let config = SamplingConfig {
        penalize_nl: true,
        newline_penalty: 10.0,
        ..Default::default()
    };

    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);

    let picked = sampler.sample_token(&mut logits, &mut state, 1).unwrap();
    assert_ne!(picked, 1); // Token 1 (newline) was penalized
}
