//! End-to-end integration test validating simultaneous streaming compliance
//! across OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) protocols.

use axum::http::StatusCode;
use axum_test::TestServer;
use oxide_server::server::{MockEngineContext, create_engine_router};
use serde_json::json;

#[tokio::test]
async fn test_dual_protocol_streaming_conformance() {
    let context = MockEngineContext::new();
    let app = create_engine_router(context);
    let server = TestServer::new(app).expect("Failed to launch axum test server");

    // 1. Query OpenAI /v1/chat/completions with stream=true
    let openai_payload = json!({
        "model": "bonsai2-27b",
        "messages": [{"role": "user", "content": "Explain FWHT butterfly networks."}],
        "max_tokens": 4,
        "stream": true
    });

    let openai_response = server
        .post("/v1/chat/completions")
        .json(&openai_payload)
        .await;

    assert_eq!(openai_response.status_code(), StatusCode::OK);
    assert_eq!(openai_response.header("content-type"), "text/event-stream");
    let openai_body = openai_response.text();
    assert!(openai_body.contains("data: "));
    assert!(openai_body.contains("[DONE]"));

    // 2. Query Anthropic /v1/messages with stream=true
    let anthropic_payload = json!({
        "model": "bonsai2-27b",
        "messages": [{"role": "user", "content": "Explain FWHT butterfly networks."}],
        "max_tokens": 1024,
        "stream": true
    });

    let anthropic_response = server
        .post("/v1/messages")
        .add_header("anthropic-version", "2023-06-01")
        .json(&anthropic_payload)
        .await;

    assert_eq!(anthropic_response.status_code(), StatusCode::OK);
    assert_eq!(
        anthropic_response.header("content-type"),
        "text/event-stream"
    );
    let anthropic_body = anthropic_response.text();
    assert!(anthropic_body.contains("event: message_start"));
    assert!(anthropic_body.contains("event: content_block_delta"));
    assert!(anthropic_body.contains("event: message_stop"));
}
