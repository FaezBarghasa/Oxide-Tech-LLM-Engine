//! Anthropic Messages API (`POST /v1/messages`) Parity Server.
//!
//! Provides native wire-level compatibility for Anthropic Claude clients,
//! including system prompt handling, streaming SSE message events, and tool blocks.

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Sse, sse::Event},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::ServerState;

/// Anthropic Message Content Block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Anthropic Message Input Item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

/// Anthropic Messages Request (`POST /v1/messages`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: usize,
    pub system: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

/// Anthropic Usage Statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Anthropic Messages Response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

/// Handler for `POST /v1/messages`.
pub async fn messages_handler(
    State(state): State<ServerState>,
    Json(payload): Json<AnthropicMessagesRequest>,
) -> impl IntoResponse {
    let stream_mode = payload.stream.unwrap_or(false);

    let max_tokens = payload.max_tokens.min(4096);
    let mut prompt_text = String::new();
    if let Some(sys) = &payload.system {
        prompt_text.push_str(sys);
        prompt_text.push('\n');
    }
    for m in &payload.messages {
        prompt_text.push_str(&format!("{}: {}\n", m.role, m.content));
    }
    prompt_text.push_str("Assistant: ");

    let prompt_tokens = state.tokenizer.encode(&prompt_text);
    let prompt_len = prompt_tokens.len().max(1);
    let initial_token = prompt_tokens.last().copied().unwrap_or(1);

    let model_name = payload.model.clone();
    let target_pipeline = state.resolve_pipeline(&model_name).await;
    let tokenizer = Arc::clone(&state.tokenizer);

    let slot_request = crate::SlotRequest {
        request_id: "anthropic-msg-req".to_string(),
        prompt_tokens: if prompt_tokens.is_empty() {
            vec![1]
        } else {
            prompt_tokens.clone()
        },
        max_tokens,
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        stream: stream_mode,
    };

    let Some(slot_guard) = crate::LeasedSlotGuard::lease(&state.slot_manager, slot_request).await else {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "type": "error",
                    "message": "All inference slots busy"
                }
            })),
        )
            .into_response();
    };

    if stream_mode {
        let stream = async_stream::stream! {
            let guard = slot_guard;
            let msg_id = "msg_oxide_01".to_string();

            // 1. message_start event
            let start_json = serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": &msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": &payload.model,
                    "usage": { "input_tokens": prompt_len, "output_tokens": 0 }
                }
            });
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("message_start").data(start_json.to_string()));

            // 2. content_block_start event
            let block_start_json = serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            });
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("content_block_start").data(block_start_json.to_string()));

            let mut cur_token = initial_token;
            let mut out_tokens = 0;

            for i in 0..max_tokens {
                let cmd = oxide_core::StepCommand::new(1001, cur_token, guard.slot_id() as u16, false);
                let completion = match {
                    let mut pipeline = target_pipeline.lock().await;
                    pipeline.step(&cmd)
                } {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Anthropic inference step failed: {e}");
                        break;
                    }
                };
                cur_token = completion.sampled_token;
                let token_str = tokenizer.decode_token(cur_token);
                out_tokens += 1;

                let delta_json = serde_json::json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": token_str }
                });
                yield Ok::<Event, std::convert::Infallible>(Event::default().event("content_block_delta").data(delta_json.to_string()));

                if completion.is_terminal || i == max_tokens - 1 {
                    break;
                }
            }

            // 4. content_block_stop event
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("content_block_stop").data(serde_json::json!({
                "type": "content_block_stop",
                "index": 0
            }).to_string()));

            // 5. message_delta event
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("message_delta").data(serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": out_tokens }
            }).to_string()));

            // 6. message_stop event
            yield Ok::<Event, std::convert::Infallible>(Event::default().event("message_stop").data(serde_json::json!({
                "type": "message_stop"
            }).to_string()));
        };

        return Sse::new(stream).into_response();
    }

    let mut cur_token = initial_token;
    let mut generated_text = String::new();
    let mut out_tokens = 0;

    for i in 0..max_tokens {
        let cmd = oxide_core::StepCommand::new(1001, cur_token, slot_guard.slot_id() as u16, false);
        let completion = match {
            let mut pipeline = target_pipeline.lock().await;
            pipeline.step(&cmd)
        } {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Anthropic inference step failed: {e}");
                break;
            }
        };
        cur_token = completion.sampled_token;
        generated_text.push_str(&tokenizer.decode_token(cur_token));
        out_tokens += 1;

        if completion.is_terminal || i == max_tokens - 1 {
            break;
        }
    }

    let response = AnthropicMessagesResponse {
        id: "msg_oxide_01".to_string(),
        object_type: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![AnthropicContentBlock::Text {
            text: generated_text,
        }],
        model: payload.model,
        stop_reason: "end_turn".to_string(),
        usage: AnthropicUsage {
            input_tokens: prompt_len,
            output_tokens: out_tokens,
        },
    };

    Json(response).into_response()
}
