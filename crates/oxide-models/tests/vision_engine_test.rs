//! Computer Vision Edge Engine Verification Tests.

use oxide_models::vision::{
    DetectedObject, EdgeVisionEngine, VisionModelType, VisionPreprocessConfig,
};

#[test]
fn test_edge_vision_engine_nms_and_iou() {
    let box_a = [0.1, 0.1, 0.5, 0.5];
    let box_b = [0.1, 0.1, 0.48, 0.48];
    let box_c = [0.7, 0.7, 0.9, 0.9];

    let iou_ab = EdgeVisionEngine::compute_iou(&box_a, &box_b);
    let iou_ac = EdgeVisionEngine::compute_iou(&box_a, &box_c);

    assert!(iou_ab > 0.8, "Overlapping boxes should have high IoU");
    assert_eq!(iou_ac, 0.0, "Disjoint boxes should have zero IoU");

    let engine = EdgeVisionEngine::new(
        VisionModelType::YoloWorld,
        VisionPreprocessConfig::default(),
    );

    let candidates = vec![
        DetectedObject {
            class_id: 0,
            class_name: "pedestrian".to_string(),
            score: 0.95,
            bbox: box_a,
        },
        DetectedObject {
            class_id: 0,
            class_name: "pedestrian".to_string(),
            score: 0.85,
            bbox: box_b,
        },
        DetectedObject {
            class_id: 2,
            class_name: "vehicle".to_string(),
            score: 0.90,
            bbox: box_c,
        },
    ];

    let kept = engine.non_maximum_suppression(candidates);
    assert_eq!(kept.len(), 2, "Duplicate pedestrian box should be suppressed by NMS");
    assert_eq!(kept[0].class_id, 0);
    assert_eq!(kept[1].class_id, 2);
}

#[test]
fn test_edge_vision_preprocessing_and_patch_extraction() {
    let mut config = VisionPreprocessConfig::default();
    config.target_width = 32;
    config.target_height = 32;

    let engine = EdgeVisionEngine::new(VisionModelType::MobileNetV4, config);

    // 16x16 dummy RGB image
    let raw_rgb = vec![128u8; 16 * 16 * 3];
    let mut chw = vec![0.0f32; 3 * 32 * 32];
    engine.preprocess_rgb_image(&raw_rgb, 16, 16, &mut chw);
    assert_eq!(chw.len(), 3 * 32 * 32);

    // Patch extraction with 8x8 patches -> (32/8)*(32/8) = 16 patches
    let patch_dim = 3 * 8 * 8; // 192
    let mut patch_tokens = vec![0.0f32; 16 * patch_dim];
    engine.extract_patch_embeddings(&chw, 8, &mut patch_tokens);
    assert_eq!(patch_tokens.len(), 16 * 192);
}
