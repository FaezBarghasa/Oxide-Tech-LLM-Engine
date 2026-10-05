use thiserror::Error;

pub type Result<T, E = EngineError> = std::result::Result<T, E>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum EngineError {
    #[error(
        "Out of static memory in arena: requested {requested_bytes} bytes, capacity {capacity_bytes} bytes"
    )]
    OutOfMemory {
        requested_bytes: usize,
        capacity_bytes: usize,
    },

    #[error("Static arena allocation bounds exceeded for slot {slot_idx}")]
    AllocationBoundsExceeded { slot_idx: usize },

    #[error("Device memory violation: invalid access or null pointer at {address:#x}")]
    DeviceMemoryViolation { address: u64 },

    #[error("Alignment fault: pointer {address:#x} does not satisfy alignment {alignment}")]
    AlignmentFault { address: u64, alignment: usize },

    #[error("DFA schema mismatch during structured decoding: state {state}, byte {byte:#04x}")]
    SchemaMismatch { state: u32, byte: u8 },

    #[error("Invalid model artifact header or magic mismatch")]
    InvalidArtifactHeader,

    #[error("Hardware backend error: {0}")]
    BackendError(String),

    #[error("Channel full or command queue overflow")]
    QueueOverflow,

    #[error("Channel empty or completion queue underflow")]
    QueueUnderflow,

    #[error("Sequence {sequence_id} not found in active batch")]
    SequenceNotFound { sequence_id: u64 },

    #[error("Invalid state transition for request {sequence_id}")]
    InvalidStateTransition { sequence_id: u64 },

    #[error("Unsupported hardware capability or model configuration")]
    UnsupportedCapability,

    #[error("Tensor shape or dimension mismatch")]
    ShapeMismatch,
}
