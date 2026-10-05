use oxide_models::audio::{AudioEngineMode, AudioModelConfig, AudioServingEngine};
use oxide_models::diffusion::{DiffusionEngine, DiffusionSchedulerType, DiffusionTransformerConfig};

#[test]
fn test_diffusion_image_and_video_generation() {
    // Image DiT
    let image_config = DiffusionTransformerConfig::default();
    let mut image_engine = DiffusionEngine::new(image_config);
    image_engine.initialize_noise(12345);
    for step in 0..image_engine.config().num_inference_steps {
        assert!(image_engine.step_denoise(step).is_ok());
    }
    assert_eq!(
        image_engine.output_latents().len(),
        image_engine.config().latent_element_count()
    );

    // Video DiT (16 frames)
    let video_config = DiffusionTransformerConfig::new_video_dit(16);
    let mut video_engine = DiffusionEngine::new(video_config);
    assert_eq!(video_engine.config().num_frames, 16);
    assert_eq!(
        video_engine.config().scheduler,
        DiffusionSchedulerType::EulerDiscrete
    );
    video_engine.initialize_noise(54321);
    assert!(video_engine.step_denoise(0).is_ok());
}

#[test]
fn test_audio_tts_and_asr_serving() {
    // TTS
    let tts_config = AudioModelConfig::new_tts_config(24000);
    assert_eq!(tts_config.mode, AudioEngineMode::SpeechSynthesisTts);
    let mut tts_engine = AudioServingEngine::new(tts_config);
    let pcm = tts_engine
        .synthesize_streaming_tts_chunk(&[65, 66, 67])
        .expect("TTS chunk synthesized");
    assert!(!pcm.is_empty());

    // ASR
    let asr_config = AudioModelConfig::default();
    assert_eq!(asr_config.mode, AudioEngineMode::SpeechRecognitionAsr);
    let mut asr_engine = AudioServingEngine::new(asr_config);
    let pcm_input = vec![1000i16; 1600]; // 100ms at 16kHz
    let tokens = asr_engine
        .process_streaming_asr_chunk(&pcm_input)
        .expect("ASR chunk processed");
    assert!(!tokens.is_empty());
}
