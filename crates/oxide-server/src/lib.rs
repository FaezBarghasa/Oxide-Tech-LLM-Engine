#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub mod dfa;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use dfa::DfaSchemaGrammar;
use oxide_core::worker::StepCommand;
use oxide_engine::{ContinuousBatchingSlotManager, SlotRequest, SpecializedPipeline};
use oxide_models::{ChatMessage, ChatTemplateFormat, ChatTemplateParser, ModelSpecification};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Debug, Deserialize)]
pub struct ChatMessageDto {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Option<Vec<ChatMessageDto>>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub json_schema: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: serde_json::Value, // String or Vec<String>
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoiceMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionChoiceMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct UsageStatistics {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: UsageStatistics,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkDelta {
    pub content: Option<String>,
    pub role: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStatistics,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: UsageStatistics,
}

#[derive(Clone, Debug)]
pub struct ServerState {
    pub pipeline: Arc<Mutex<SpecializedPipeline>>,
    pub dfa_grammar: Arc<DfaSchemaGrammar>,
    pub slot_manager: Arc<Mutex<ContinuousBatchingSlotManager>>,
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "Oxide-Tech-LLM-Engine OK"
}

async fn models_handler() -> Json<ModelsListResponse> {
    let catalog = ModelSpecification::catalog();
    let data = catalog
        .iter()
        .map(|spec| ModelCard {
            id: spec.identifier.to_string(),
            object: "model".to_string(),
            created: 1_728_000_000,
            owned_by: "oxide-engine".to_string(),
        })
        .collect();

    Json(ModelsListResponse {
        object: "list".to_string(),
        data,
    })
}

async fn chat_completions_handler(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let stream_mode = payload.stream.unwrap_or(false);
    let max_tokens = payload.max_tokens.unwrap_or(64);

    // Parse messages into prompt if provided
    let prompt_text = if let Some(msgs) = &payload.messages {
        let chat_msgs: Vec<ChatMessage> = msgs
            .iter()
            .map(|m| match m.role.as_str() {
                "system" => ChatMessage::system(&m.content),
                "assistant" => ChatMessage::assistant(&m.content),
                _ => ChatMessage::user(&m.content),
            })
            .collect();
        let parser = ChatTemplateParser::new(ChatTemplateFormat::ChatMl);
        parser.render_chat(&chat_msgs)
    } else {
        payload.prompt.clone().unwrap_or_default()
    };

    let prompt_len = prompt_text.split_whitespace().count().max(1);

    if stream_mode {
        let stream = async_stream::stream! {
            let mut cur_token: u32 = 42;

            for i in 0..max_tokens {
                let cmd = StepCommand::new(1001, cur_token, 0, false);
                let completion = {
                    let mut pipeline = state.pipeline.lock().await;
                    pipeline.step(&cmd).unwrap()
                };

                cur_token = completion.sampled_token;
                let token_str = format!("tok_{cur_token} ");
                let is_last = i == max_tokens - 1 || completion.is_terminal;

                let chunk = ChatCompletionChunk {
                    id: "chatcmpl-oxide-01".to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created: 1_728_000_000,
                    model: payload.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            content: Some(token_str),
                            role: if i == 0 { Some("assistant".to_string()) } else { None },
                        },
                        finish_reason: if is_last { Some("stop".to_string()) } else { None },
                    }],
                };

                let json_data = serde_json::to_string(&chunk).unwrap_or_default();
                yield Ok::<Event, Infallible>(Event::default().data(json_data));

                if is_last {
                    break;
                }
            }
            yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
        };

        Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
    } else {
        let mut generated_text = String::new();
        let mut cur_token: u32 = 42;
        let mut completion_tokens = 0;

        // Register with continuous batching slot manager
        {
            let mut slot_mgr = state.slot_manager.lock().await;
            slot_mgr.submit_request(SlotRequest {
                request_id: "req-oxide-01".to_string(),
                prompt_tokens: vec![42; prompt_len],
                max_tokens,
                temperature: payload.temperature.unwrap_or(0.7),
                top_p: payload.top_p.unwrap_or(0.9),
                stream: false,
            });
        }

        for i in 0..max_tokens {
            let cmd = StepCommand::new(1001, cur_token, 0, false);
            let completion = {
                let mut pipeline = state.pipeline.lock().await;
                pipeline.step(&cmd).unwrap()
            };

            cur_token = completion.sampled_token;
            generated_text.push_str(&format!("tok_{cur_token} "));
            completion_tokens += 1;

            if completion.is_terminal || i == max_tokens - 1 {
                break;
            }
        }

        let resp = ChatCompletionResponse {
            id: "chatcmpl-oxide-01".to_string(),
            object: "chat.completion".to_string(),
            created: 1_728_000_000,
            model: payload.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionChoiceMessage {
                    role: "assistant".to_string(),
                    content: generated_text,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: UsageStatistics {
                prompt_tokens: prompt_len,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
            },
        };

        Json(resp).into_response()
    }
}

async fn completions_handler(
    State(state): State<ServerState>,
    Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
    let max_tokens = payload.max_tokens.unwrap_or(64);
    let prompt_len = payload.prompt.split_whitespace().count().max(1);

    let mut generated_text = String::new();
    let mut cur_token: u32 = 42;
    let mut completion_tokens = 0;

    for i in 0..max_tokens {
        let cmd = StepCommand::new(1001, cur_token, 0, false);
        let completion = {
            let mut pipeline = state.pipeline.lock().await;
            pipeline.step(&cmd).unwrap()
        };

        cur_token = completion.sampled_token;
        generated_text.push_str(&format!("tok_{cur_token} "));
        completion_tokens += 1;

        if completion.is_terminal || i == max_tokens - 1 {
            break;
        }
    }

    let resp = CompletionResponse {
        id: "cmpl-oxide-01".to_string(),
        object: "text_completion".to_string(),
        created: 1_728_000_000,
        model: payload.model,
        choices: vec![CompletionChoice {
            text: generated_text,
            index: 0,
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStatistics {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    };

    Json(resp).into_response()
}

async fn embeddings_handler(
    Json(payload): Json<EmbeddingRequest>,
) -> Json<EmbeddingResponse> {
    let inputs: Vec<String> = match payload.input {
        serde_json::Value::String(s) => vec![s],
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect(),
        _ => vec!["default".to_string()],
    };

    let dim = 1024;
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0;

    for (idx, text) in inputs.iter().enumerate() {
        let tokens = text.split_whitespace().count().max(1);
        total_tokens += tokens;

        // Deterministic pseudo-embedding vector normalized to unit length
        let mut vec = vec![0.0; dim];
        for (i, v) in vec.iter_mut().enumerate().take(dim) {
            *v = ((i + 1) as f32 * 0.001 * (text.len() as f32)).sin();
        }
        let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-6);
        for v in &mut vec {
            *v /= norm;
        }

        data.push(EmbeddingData {
            object: "embedding".to_string(),
            index: idx,
            embedding: vec,
        });
    }

    Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: payload.model,
        usage: UsageStatistics {
            prompt_tokens: total_tokens,
            completion_tokens: 0,
            total_tokens,
        },
    })
}

pub async fn start_server(
    addr: SocketAddr,
    state: ServerState,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let router = create_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Oxide-Tech-LLM-Engine serving on http://{}", addr);
    axum::serve(listener, router).await?;
    Ok(())
}
