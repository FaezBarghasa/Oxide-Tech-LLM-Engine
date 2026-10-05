//! gRPC High-Throughput Streaming Protocol Service.
//!
//! Provides typed RPC definitions for streaming inference, embeddings, and health status.

use serde::{Deserialize, Serialize};

/// gRPC Inference Request Message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrpcInferenceRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub stream: bool,
}

/// gRPC Single Token Response Message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrpcTokenResponse {
    pub token_id: u32,
    pub text: String,
    pub log_prob: f32,
    pub is_finished: bool,
}

/// gRPC Embedding Request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrpcEmbeddingRequest {
    pub model: String,
    pub texts: Vec<String>,
}

/// gRPC Embedding Vector Response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrpcEmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
}

/// gRPC Service Handler.
#[derive(Debug, Clone, Default)]
pub struct OxideGrpcService;

impl OxideGrpcService {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Processes an RPC inference step.
    #[must_use]
    pub fn process_step(&self, _req: &GrpcInferenceRequest, step_token: u32, text: &str) -> GrpcTokenResponse {
        GrpcTokenResponse {
            token_id: step_token,
            text: text.to_string(),
            log_prob: -0.05,
            is_finished: false,
        }
    }
}
