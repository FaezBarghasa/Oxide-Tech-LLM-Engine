use crate::error::Result;
use crate::memory::DevicePtr;
use crate::worker::StepCommand;
use std::fmt::Debug;

/// Core hardware backend abstraction.
pub trait HardwareBackend: Send + Sync + 'static + Debug {
    type Event: Copy + Send + Sync + Debug;

    /// Dispatches a single token forward step to the hardware execution stream.
    fn dispatch_step_kernel(&mut self, cmd: &StepCommand) -> Result<Self::Event>;

    /// Non-blocking query to check if a hardware event has completed.
    fn query_event_completed(&self, event: Self::Event) -> bool;

    /// Reads the sampled output token from host-mapped pinned buffer.
    fn read_sampled_token_host(&self, slot_idx: u16) -> u32;

    /// Synchronizes the hardware device.
    fn synchronize(&self) -> Result<()>;
}

/// Static model configuration descriptor.
pub trait ModelConfig: Send + Sync + 'static + Debug + Clone {
    fn model_name(&self) -> &'static str;
    fn hidden_dim(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn max_seq_len(&self) -> usize;
}

/// Quantization scheme marker trait.
pub trait QuantScheme: Send + Sync + 'static + Debug + Copy {
    fn bits_per_weight() -> f32;
    fn block_size() -> usize;
}

/// Optical compute fabric capability.
pub trait OpticalComputeFabric: HardwareBackend {
    /// Dispatches optical GEMV acceleration kernel.
    ///
    /// # Safety
    /// Device pointers must be valid and matrix_id must exist in optical register bank.
    unsafe fn dispatch_optical_gemv(
        &self,
        input: DevicePtr<f32>,
        matrix_id: u64,
        output: DevicePtr<f32>,
    ) -> Result<()>;
}

/// Compile-time marker asserting that backend `B` supports model `M`.
pub trait HardwareSupports<B: HardwareBackend, M: ModelConfig> {}
