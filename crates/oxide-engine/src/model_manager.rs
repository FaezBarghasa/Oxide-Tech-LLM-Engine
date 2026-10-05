use crate::SpecializedPipeline;
use oxide_core::error::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Dynamic runtime model manager allowing hot-loading, swapping, and caching models without recompiling.
#[derive(Debug)]
pub struct DynamicModelManager {
    models: HashMap<String, Arc<Mutex<SpecializedPipeline>>>,
    default_model: String,
    models_dir: Option<PathBuf>,
    backend_kind: String,
    gpu_profile: Option<String>,
    max_slots: usize,
}

impl DynamicModelManager {
    #[must_use]
    pub fn new(
        default_model: impl Into<String>,
        initial_pipeline: SpecializedPipeline,
        backend_kind: impl Into<String>,
        gpu_profile: Option<String>,
        max_slots: usize,
        models_dir: Option<PathBuf>,
    ) -> Self {
        let def = default_model.into();
        let mut models = HashMap::new();
        models.insert(def.clone(), Arc::new(Mutex::new(initial_pipeline)));
        Self {
            models,
            default_model: def,
            models_dir,
            backend_kind: backend_kind.into(),
            gpu_profile,
            max_slots,
        }
    }

    /// Retrieve an existing loaded pipeline, or dynamically load it on-demand.
    pub async fn get_or_load(
        &mut self,
        model_name_or_path: &str,
    ) -> Result<Arc<Mutex<SpecializedPipeline>>> {
        let target = if model_name_or_path.is_empty() {
            &self.default_model
        } else {
            model_name_or_path
        };

        if let Some(pipe) = self.models.get(target) {
            return Ok(Arc::clone(pipe));
        }

        // Check if model file exists in models_dir
        let candidate_path = if let Some(dir) = &self.models_dir {
            let p1 = dir.join(target);
            let p2 = dir.join(format!("{target}.gguf"));
            let p3 = dir.join(format!("{target}.safetensors"));
            if p1.exists() {
                Some(p1.to_string_lossy().to_string())
            } else if p2.exists() {
                Some(p2.to_string_lossy().to_string())
            } else if p3.exists() {
                Some(p3.to_string_lossy().to_string())
            } else {
                None
            }
        } else {
            None
        };

        let resolved_target = candidate_path.as_deref().unwrap_or(target);
        let pipeline = SpecializedPipeline::from_model_or_path(
            resolved_target,
            &self.backend_kind,
            self.gpu_profile.as_deref(),
            self.max_slots,
            None,
        )?;

        let arc_pipe = Arc::new(Mutex::new(pipeline));
        self.models
            .insert(target.to_string(), Arc::clone(&arc_pipe));
        Ok(arc_pipe)
    }

    #[must_use]
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    #[must_use]
    pub fn list_loaded(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    #[must_use]
    pub fn list_available(&self) -> Vec<String> {
        let mut list = self.list_loaded();
        if let Some(dir) = &self.models_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if ext.eq_ignore_ascii_case("gguf") || ext.eq_ignore_ascii_case("safetensors") {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            if !list.contains(&stem.to_string()) {
                                list.push(stem.to_string());
                            }
                        }
                    }
                }
            }
        }
        list
    }
}
