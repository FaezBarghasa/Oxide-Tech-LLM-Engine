//! Server router factory and mock context for testing and production serving.

use crate::{create_router, ServerState};
use axum::Router;
use oxide_alloc::HierarchicalKvCache;
use oxide_engine::{ContinuousBatchingSlotManager, SpecializedPipeline};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Mock / test context for starting the engine server.
#[derive(Debug, Clone)]
pub struct MockEngineContext {
    pub state: ServerState,
}

impl Default for MockEngineContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MockEngineContext {
    #[must_use]
    pub fn new() -> Self {
        let pipeline = SpecializedPipeline::from_model_or_path("llama3", "cpu", None, 16, None)
            .unwrap_or_else(|_| {
                let cfg = oxide_models::llama3::Llama3Config::tiny_test_config();
                let model = oxide_models::Llama3Model::new(cfg);
                let kv_cache = (0..model.config.num_layers)
                    .map(|_| oxide_models::llama3::Llama3KvCacheLayer::default())
                    .collect();
                SpecializedPipeline::Llama3Dense {
                    model,
                    kv_cache,
                    seq_positions: std::collections::HashMap::new(),
                }
            });

        let kv_cache = Arc::new(Mutex::new(HierarchicalKvCache::new(128, 512, 1024)));
        let dfa_grammar = Arc::new(crate::dfa::DfaSchemaGrammar::new_simple_json_validator());
        let slot_manager = Arc::new(Mutex::new(ContinuousBatchingSlotManager::new(16)));

        let state = ServerState {
            pipeline: Arc::new(Mutex::new(pipeline)),
            model_manager: None,
            dfa_grammar,
            slot_manager,
            kv_cache,
        };

        Self { state }
    }
}

/// Creates the Axum router configured for OpenAI and Anthropic API conformance.
#[must_use]
pub fn create_engine_router(context: MockEngineContext) -> Router {
    create_router(context.state)
}
