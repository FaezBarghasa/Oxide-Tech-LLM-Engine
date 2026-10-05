//! Validates speculative decoding token rejection and GPU sequence length synchronization.
//! Ensures rejected draft tokens decrement block tables and commit accepted tokens to seq_lens_d.

use oxide_engine::speculative::SpeculativeEngineContext;

#[test]
fn test_speculative_draft_rejection_synchronizes_gpu_counters() {
    let mut context = SpeculativeEngineContext::new(1024);
    let sequence_id = 9999u64;

    // 1. Establish initial prompt sequence length (Length = 128)
    context
        .register_sequence(sequence_id, 128)
        .expect("Failed to register test sequence");

    // 2. Propose K=4 draft tokens (Speculative length reaches 132)
    let draft_tokens = [101u32, 202u32, 303u32, 404u32];
    context
        .append_draft_tokens(sequence_id, &draft_tokens)
        .expect("Failed to append draft tokens");
    assert_eq!(context.get_host_sequence_length(sequence_id), 132);

    // 3. Verification step: Target model accepts 2 tokens, rejects 2 tokens
    let accepted_count = 2usize; // Only tokens 101 and 202 accepted
    context
        .rollback_speculative_state(sequence_id, 128 + accepted_count, draft_tokens.len())
        .expect("Failed to execute speculative rollback");

    // 4. Assert host state truncated to exactly 130 tokens
    assert_eq!(context.get_host_sequence_length(sequence_id), 130);

    // 5. Assert device-side sequence counter buffer (seq_lens_d) matches host state exactly
    let device_length = context
        .read_device_sequence_length(sequence_id)
        .expect("Failed to query device sequence length buffer");
    assert_eq!(
        device_length, 130,
        "FATAL: Device sequence length desynchronized! Host=130, Device={device_length}",
    );
}
