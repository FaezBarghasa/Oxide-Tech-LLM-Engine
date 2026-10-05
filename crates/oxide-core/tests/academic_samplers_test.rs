use oxide_core::sampler::{
    AcademicSamplerEngine, GbnfGrammarEngine, GbnfRule, MirostatMode, SamplerState, SamplingConfig,
};

#[test]
fn test_academic_sampler_temperature_and_top_k() {
    let mut logits = vec![1.0, 2.0, 5.0, 3.0, 0.5];
    let config = SamplingConfig {
        temperature: 0.5,
        top_k: 2,
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    assert_eq!(picked, 2); // Logit 5.0 is highest
    assert_eq!(state.generated_history, vec![2]);
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
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
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
    let mut state = SamplerState::new(32);
    state.generated_history = vec![0, 1, 0, 1, 0, 1]; // Repetitive pattern 0, 1

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    // Sequence 0, 1, 0, 1 matches suffix [0, 1], penalizing token 0 heavily, yielding token 2
    assert_eq!(picked, 2);
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
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    assert_eq!(picked, 4); // Highest probability token within target entropy
}

#[test]
fn test_gbnf_grammar_constrained_sampling() {
    let mut logits = vec![10.0, 10.0, 10.0, 10.0];
    let grammar = GbnfGrammarEngine {
        rules: vec![GbnfRule {
            rule_id: 0,
            allowed_tokens: vec![1, 3], // Only tokens 1 and 3 are syntactically valid JSON/grammar
        }],
        current_rule_idx: 0,
    };

    let config = SamplingConfig {
        gbnf_grammar: Some(grammar),
        ..Default::default()
    };
    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    assert!(picked == 1 || picked == 3);
}

#[test]
fn test_logit_bias_and_token_bans() {
    let mut logits = vec![10.0, 5.0, 1.0];
    let mut config = SamplingConfig::default();
    config.banned_tokens.push(0); // Ban top token 0
    config.logit_bias.insert(2, 20.0); // Boost token 2 by +20.0

    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    assert_eq!(picked, 2);
}

#[test]
fn test_penalize_newline() {
    let mut logits = vec![5.0, 5.0, 5.0];
    let mut config = SamplingConfig {
        penalize_nl: true,
        newline_token_id: 1,
        ..Default::default()
    };
    config.logit_bias.insert(1, 0.0);

    let sampler = AcademicSamplerEngine::new(config);
    let mut state = SamplerState::new(32);

    let picked = sampler.sample(&mut logits, &mut state).unwrap();
    assert_ne!(picked, 1); // Token 1 (newline) was penalized
}
