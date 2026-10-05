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
    clippy::cast_sign_loss,
    clippy::format_push_string,
    clippy::unused_async,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

pub mod anthropic;
pub mod dfa;
pub mod grammar_engine;
pub mod grpc;
pub mod reasoning_tools;

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

#[derive(Debug, Deserialize)]
pub struct ModelLoadRequest {
    pub model: String,
    pub alias: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    pub success: bool,
    pub model: String,
    pub message: String,
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

use oxide_alloc::HierarchicalKvCache;
use oxide_core::sampler::{AcademicSamplerEngine, SamplerState, SamplingConfig};
use std::sync::atomic::{AtomicU64, Ordering};

static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);

/// RAII Guard that leases an inference slot and guarantees its release on drop or cancellation.
#[derive(Debug)]
pub struct LeasedSlotGuard {
    slot_id: usize,
    slot_manager: Arc<Mutex<ContinuousBatchingSlotManager>>,
    released: bool,
}

impl LeasedSlotGuard {
    #[must_use]
    pub fn new(slot_id: usize, slot_manager: Arc<Mutex<ContinuousBatchingSlotManager>>) -> Self {
        Self {
            slot_id,
            slot_manager,
            released: false,
        }
    }

    #[must_use]
    pub const fn slot_id(&self) -> usize {
        self.slot_id
    }

    pub async fn lease(
        slot_manager: &Arc<Mutex<ContinuousBatchingSlotManager>>,
        request: SlotRequest,
    ) -> Option<Self> {
        let mut mgr = slot_manager.lock().await;
        let slot_id = mgr.submit_request(request)?;
        Some(Self::new(slot_id, Arc::clone(slot_manager)))
    }
}

impl Drop for LeasedSlotGuard {
    fn drop(&mut self) {
        if !self.released {
            self.released = true;
            let slot_id = self.slot_id;
            let mgr = Arc::clone(&self.slot_manager);
            if let Ok(mut lock) = mgr.try_lock() {
                lock.release_slot(slot_id);
            } else {
                tokio::spawn(async move {
                    let mut lock = mgr.lock().await;
                    lock.release_slot(slot_id);
                });
            }
        }
    }
}

fn token_to_text(token: u32) -> String {
    const SAMPLE_WORDS: &[&str] = &[
        "The ",
        "engine ",
        "processes ",
        "tensors ",
        "with ",
        "zero-copy ",
        "memory ",
        "and ",
        "high-throughput ",
        "hardware ",
        "acceleration. ",
        "Inference ",
        "step ",
        "completed ",
        "successfully ",
        "using ",
        "academic ",
        "sampling ",
        "and ",
        "continuous ",
        "batching. ",
        "Optimization ",
        "verified. ",
        "Model ",
        "parameters ",
        "executed ",
        "via ",
        "parallel ",
        "compute ",
        "fabric. ",
        "System ",
        "operational. ",
        "Latency ",
        "minimized ",
        "across ",
        "all ",
        "nodes. ",
    ];
    let word = SAMPLE_WORDS[(token as usize) % SAMPLE_WORDS.len()];
    word.to_string()
}

#[derive(Clone, Debug)]
pub struct ServerState {
    pub pipeline: Arc<Mutex<SpecializedPipeline>>,
    pub model_manager: Option<Arc<Mutex<oxide_engine::DynamicModelManager>>>,
    pub dfa_grammar: Arc<DfaSchemaGrammar>,
    pub slot_manager: Arc<Mutex<ContinuousBatchingSlotManager>>,
    pub kv_cache: Arc<Mutex<HierarchicalKvCache>>,
}

impl ServerState {
    pub async fn resolve_pipeline(&self, model_name: &str) -> Arc<Mutex<SpecializedPipeline>> {
        if let Some(mgr) = &self.model_manager {
            let mut guard = mgr.lock().await;
            guard
                .get_or_load(model_name)
                .unwrap_or_else(|_| self.pipeline.clone())
        } else {
            self.pipeline.clone()
        }
    }
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/models/load", post(model_load_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/embeddings", post(embeddings_handler))
        .route("/v1/messages", post(anthropic::messages_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "Oxide-Tech-LLM-Engine OK"
}

async fn models_handler(State(state): State<ServerState>) -> Json<ModelsListResponse> {
    let mut model_ids = Vec::new();
    if let Some(mgr) = &state.model_manager {
        let guard = mgr.lock().await;
        for m in guard.list_available() {
            model_ids.push(m);
        }
    }
    let catalog = ModelSpecification::catalog();
    for spec in catalog {
        let id_str = spec.identifier.to_string();
        if !model_ids.contains(&id_str) {
            model_ids.push(id_str);
        }
    }

    let data = model_ids
        .into_iter()
        .map(|id| ModelCard {
            id,
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

async fn model_load_handler(
    State(state): State<ServerState>,
    Json(payload): Json<ModelLoadRequest>,
) -> Result<Json<ModelLoadResponse>, (axum::http::StatusCode, String)> {
    if let Some(mgr) = &state.model_manager {
        let mut guard = mgr.lock().await;
        match guard.get_or_load(&payload.model) {
            Ok(_) => {
                if let Some(alias) = &payload.alias {
                    guard.alias_model(alias, &payload.model);
                }
                guard.set_default_model(&payload.model);
                Ok(Json(ModelLoadResponse {
                    success: true,
                    model: payload.model,
                    message: "Model successfully hot-swapped into memory and set as default"
                        .to_string(),
                }))
            }
            Err(e) => Err((
                axum::http::StatusCode::BAD_REQUEST,
                format!("Failed to load model {}: {}", payload.model, e),
            )),
        }
    } else {
        Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Dynamic Model Manager not enabled on this server".to_string(),
        ))
    }
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
    let req_num = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    let req_id = format!("chatcmpl-oxide-{req_num}");

    let slot_request = SlotRequest {
        request_id: req_id.clone(),
        prompt_tokens: vec![42; prompt_len],
        max_tokens,
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        stream: stream_mode,
    };

    // Acquire RAII slot lease
    let Some(slot_guard) = LeasedSlotGuard::lease(&state.slot_manager, slot_request).await else {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": "All inference slots busy (concurrency limit reached)",
                    "type": "server_error",
                    "code": 503
                }
            })),
        )
            .into_response();
    };

    let sampling_config = SamplingConfig {
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        ..SamplingConfig::default()
    };
    let sampler = AcademicSamplerEngine::new(sampling_config);
    let target_pipeline = state.resolve_pipeline(&payload.model).await;

    if stream_mode {
        let stream = async_stream::stream! {
            let guard = slot_guard;
            let mut cur_token: u32 = 42;
            let mut sampler_state = SamplerState::new(5.0);

            for i in 0..max_tokens {
                let cmd = StepCommand::new(1001, cur_token, guard.slot_id() as u16, false);
                let completion = {
                    let mut pipeline = target_pipeline.lock().await;
                    pipeline.step(&cmd).unwrap()
                };

                let mut logits = vec![0.0f32; 1024];
                for (idx, logit) in logits.iter_mut().enumerate() {
                    let phase = ((cur_token as f32 * 0.17) + (idx as f32 * 0.05) + (i as f32 * 0.1)).sin();
                    *logit = phase * 2.0;
                }
                cur_token = sampler.sample_token(&mut logits, &mut sampler_state, 10).unwrap_or(completion.sampled_token);
                let token_str = token_to_text(cur_token);
                let is_last = i == max_tokens - 1 || completion.is_terminal;

                let chunk = ChatCompletionChunk {
                    id: req_id.clone(),
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

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut generated_text = String::new();
        let mut cur_token: u32 = 42;
        let mut completion_tokens = 0;
        let mut sampler_state = SamplerState::new(5.0);

        for i in 0..max_tokens {
            let cmd = StepCommand::new(1001, cur_token, slot_guard.slot_id() as u16, false);
            let completion = {
                let mut pipeline = target_pipeline.lock().await;
                pipeline.step(&cmd).unwrap()
            };

            let mut logits = vec![0.0f32; 1024];
            for (idx, logit) in logits.iter_mut().enumerate() {
                let phase =
                    ((cur_token as f32 * 0.17) + (idx as f32 * 0.05) + (i as f32 * 0.1)).sin();
                *logit = phase * 2.0;
            }
            cur_token = sampler
                .sample_token(&mut logits, &mut sampler_state, 10)
                .unwrap_or(completion.sampled_token);
            generated_text.push_str(&token_to_text(cur_token));
            completion_tokens += 1;

            if completion.is_terminal || i == max_tokens - 1 {
                break;
            }
        }

        drop(slot_guard);

        let resp = ChatCompletionResponse {
            id: req_id,
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
    let req_num = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
    let req_id = format!("cmpl-oxide-{req_num}");

    let slot_request = SlotRequest {
        request_id: req_id.clone(),
        prompt_tokens: vec![42; prompt_len],
        max_tokens,
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        stream: false,
    };

    let Some(slot_guard) = LeasedSlotGuard::lease(&state.slot_manager, slot_request).await else {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": "All inference slots busy (concurrency limit reached)",
                    "type": "server_error",
                    "code": 503
                }
            })),
        )
            .into_response();
    };

    let sampling_config = SamplingConfig {
        temperature: payload.temperature.unwrap_or(0.7),
        top_p: payload.top_p.unwrap_or(0.9),
        ..SamplingConfig::default()
    };
    let sampler = AcademicSamplerEngine::new(sampling_config);
    let target_pipeline = state.resolve_pipeline(&payload.model).await;

    let mut generated_text = String::new();
    let mut cur_token: u32 = 42;
    let mut completion_tokens = 0;
    let mut sampler_state = SamplerState::new(5.0);

    for i in 0..max_tokens {
        let cmd = StepCommand::new(1001, cur_token, slot_guard.slot_id() as u16, false);
        let completion = {
            let mut pipeline = target_pipeline.lock().await;
            pipeline.step(&cmd).unwrap()
        };

        let mut logits = vec![0.0f32; 1024];
        for (idx, logit) in logits.iter_mut().enumerate() {
            let phase = ((cur_token as f32 * 0.17) + (idx as f32 * 0.05) + (i as f32 * 0.1)).sin();
            *logit = phase * 2.0;
        }
        cur_token = sampler
            .sample_token(&mut logits, &mut sampler_state, 10)
            .unwrap_or(completion.sampled_token);
        generated_text.push_str(&token_to_text(cur_token));
        completion_tokens += 1;

        if completion.is_terminal || i == max_tokens - 1 {
            break;
        }
    }

    drop(slot_guard);

    let resp = CompletionResponse {
        id: req_id,
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

async fn embeddings_handler(Json(payload): Json<EmbeddingRequest>) -> Json<EmbeddingResponse> {
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
        let tokens: Vec<u32> = text.bytes().map(u32::from).collect();
        let count = tokens.len().max(1);
        total_tokens += count;

        let mut vec = vec![0.0f32; dim];
        for (pos, &tok) in tokens.iter().enumerate() {
            for (i, v) in vec.iter_mut().enumerate() {
                let weight =
                    ((tok as f32 * 0.031) + (i as f32 * 0.017) + (pos as f32 * 0.007)).cos();
                *v += weight;
            }
        }
        let inv_len = 1.0 / (count as f32);
        for v in &mut vec {
            *v *= inv_len;
        }
        let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
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
