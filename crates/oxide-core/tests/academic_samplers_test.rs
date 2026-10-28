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
fn test_sliding_window_repetition_penalty() {
    let mut logits = vec![10.0, 5.0, 2.0];
    let config = SamplingConfig {
        repeat_penalty: 2.0,
        repeat_last_n: 4,
        temperature: 0.0, // greedy
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(5.0);
    state.generated_tokens = vec![0]; // token 0 was just generated

    // Logit 0 starts at 10.0 -> divided by 2.0 -> 5.0
    // Logit 1 is 5.0
    // With repetition penalty applied to token 0, token 0 is penalized.
    let _ = sampler.sample_token(&mut logits, &mut state, 13).unwrap();
    assert_eq!(logits[0], 5.0);
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

#[test]
fn test_multinomial_seed_determinism_and_temperature() {
    // 1. Temperature = 0 gives strict greedy argmax
    let mut logits_greedy = vec![1.0, 2.0, 8.0, 3.0];
    let config_greedy = SamplingConfig {
        temperature: 0.0,
        ..Default::default()
    };
    let sampler_greedy = AcademicSamplerEngine::new(config_greedy);
    let mut state_greedy = SamplerState::new(5.0);
    assert_eq!(
        sampler_greedy
            .sample_token(&mut logits_greedy, &mut state_greedy, 0)
            .unwrap(),
        2
    );

    // 2. Fixed seed gives deterministic multinomial sequence
    let config_seeded1 = SamplingConfig {
        temperature: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let sampler_seeded1 = AcademicSamplerEngine::new(config_seeded1);
    let mut state_seeded1 = SamplerState::new_with_seed(5.0, 42);

    let mut draws1 = Vec::new();
    for _ in 0..50 {
        let mut l = vec![2.0, 2.0, 2.0];
        draws1.push(
            sampler_seeded1
                .sample_token(&mut l, &mut state_seeded1, 99)
                .unwrap(),
        );
    }

    let config_seeded2 = SamplingConfig {
        temperature: 1.0,
        seed: Some(42),
        ..Default::default()
    };
    let sampler_seeded2 = AcademicSamplerEngine::new(config_seeded2);
    let mut state_seeded2 = SamplerState::new_with_seed(5.0, 42);

    let mut draws2 = Vec::new();
    for _ in 0..50 {
        let mut l = vec![2.0, 2.0, 2.0];
        draws2.push(
            sampler_seeded2
                .sample_token(&mut l, &mut state_seeded2, 99)
                .unwrap(),
        );
    }

    assert_eq!(draws1, draws2);
    // Ensure all 3 equal logits get drawn in 50 trials (multinomial, not stuck on 0)
    assert!(draws1.contains(&0));
    assert!(draws1.contains(&1));
    assert!(draws1.contains(&2));
}
