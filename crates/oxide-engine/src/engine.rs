use oxide_core::error::Result;
use oxide_core::traits::{HardwareBackend, ModelConfig, QuantScheme};
use oxide_core::worker::{StepCommand, StepCompletion};
use std::fmt;
use std::marker::PhantomData;

/// Monomorphized execution engine generic over hardware backend B, model config M, and quant scheme Q.
pub struct OxideEngine<B: HardwareBackend, M: ModelConfig, Q: QuantScheme> {
    backend: B,
    config: M,
    _quant_marker: PhantomData<Q>,
}

impl<B: HardwareBackend, M: ModelConfig, Q: QuantScheme> fmt::Debug for OxideEngine<B, M, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OxideEngine")
            .field("model", &self.config.model_name())
            .field("backend", &self.backend)
            .finish()
    }
}

impl<B: HardwareBackend, M: ModelConfig, Q: QuantScheme> OxideEngine<B, M, Q> {
    #[must_use]
    pub const fn new(backend: B, config: M) -> Self {
        Self {
            backend,
            config,
            _quant_marker: PhantomData,
        }
    }

    /// Monomorphized hot-path decode forward step. Contains zero virtual dispatch (dyn Trait).
    #[inline(always)]
    pub fn step_monomorphized(&mut self, cmd: &StepCommand) -> Result<StepCompletion> {
        let event = self.backend.dispatch_step_kernel(cmd)?;

        while !self.backend.query_event_completed(event) {
            std::hint::spin_loop();
        }

        let sampled_token = self.backend.read_sampled_token_host(cmd.slot_idx);
        let is_terminal = sampled_token == 0 || sampled_token == 2; // EOS check

        Ok(StepCompletion::new(
            cmd.sequence_id,
            sampled_token,
            is_terminal,
        ))
    }

    #[must_use]
    pub const fn config(&self) -> &M {
        &self.config
    }
}
