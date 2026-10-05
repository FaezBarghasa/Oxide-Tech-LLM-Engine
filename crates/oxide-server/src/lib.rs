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
use oxide_engine::SpecializedPipeline;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub json_schema: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub token: String,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ServerState {
    pub pipeline: Arc<Mutex<SpecializedPipeline>>,
    pub dfa_grammar: Arc<DfaSchemaGrammar>,
}

pub fn create_router(state: ServerState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_handler() -> &'static str {
    "Oxide-Tech-LLM-Engine OK"
}

async fn chat_completions_handler(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let stream = async_stream::stream! {
        let max_tokens = payload.max_tokens.unwrap_or(32);
        let mut cur_token: u32 = 42; // seed token

        for i in 0..max_tokens {
            let cmd = StepCommand::new(1001, cur_token, 0, false);
            let completion = {
                let mut pipeline = state.pipeline.lock().await;
                pipeline.step(&cmd).unwrap()
            };

            cur_token = completion.sampled_token;
            let token_str = format!("tok_{cur_token} ");

            let chunk = ChatCompletionChunk {
                id: "cmpl-oxide-01".to_string(),
                object: "chat.completion.chunk".to_string(),
                created: 1_728_000_000,
                model: payload.model.clone(),
                token: token_str,
                finish_reason: if i == max_tokens - 1 || completion.is_terminal {
                    Some("stop".to_string())
                } else {
                    None
                },
            };

            let json_data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok::<Event, Infallible>(Event::default().data(json_data));

            if completion.is_terminal {
                break;
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
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
