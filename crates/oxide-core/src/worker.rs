use crate::error::Result;
use crate::traits::HardwareBackend;
use crossbeam_utils::sync::Parker;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// 64-byte aligned hardware step execution command.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct StepCommand {
    pub sequence_id: u64,
    pub input_token: u32,
    pub slot_idx: u16,
    pub is_prefill_chunk: bool,
    _pad: [u8; 45],
}

impl StepCommand {
    #[must_use]
    pub const fn new(
        sequence_id: u64,
        input_token: u32,
        slot_idx: u16,
        is_prefill_chunk: bool,
    ) -> Self {
        Self {
            sequence_id,
            input_token,
            slot_idx,
            is_prefill_chunk,
            _pad: [0u8; 45],
        }
    }
}

/// 64-byte aligned step completion event.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct StepCompletion {
    pub sequence_id: u64,
    pub sampled_token: u32,
    pub is_terminal: bool,
    _pad: [u8; 51],
}

impl StepCompletion {
    #[must_use]
    pub const fn new(sequence_id: u64, sampled_token: u32, is_terminal: bool) -> Self {
        Self {
            sequence_id,
            sampled_token,
            is_terminal,
            _pad: [0u8; 51],
        }
    }
}

/// Dedicated hardware worker actor with 3-phase thermal-aware adaptive scheduling loop.
#[derive(Debug)]
pub struct ExecutionWorker<B: HardwareBackend> {
    backend: B,
    running: Arc<AtomicBool>,
    parker: Parker,
}

impl<B: HardwareBackend> ExecutionWorker<B> {
    #[must_use]
    pub fn new(backend: B, running: Arc<AtomicBool>) -> Self {
        Self {
            backend,
            running,
            parker: Parker::new(),
        }
    }

    /// Pin the execution worker to a specific physical CPU core.
    pub fn pin_to_core(&self, core_id: usize) -> bool {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        if let Some(core) = core_ids.into_iter().find(|c| c.id == core_id) {
            core_affinity::set_for_current(core)
        } else {
            false
        }
    }

    /// Executes the 3-phase adaptive worker loop reading commands from SPSC consumer
    /// and producing completions to SPSC producer.
    pub fn run_loop(
        &mut self,
        mut cmd_rx: rtrb::Consumer<StepCommand>,
        mut completion_tx: rtrb::Producer<StepCompletion>,
    ) -> Result<()> {
        let mut idle_spins: u64 = 0;

        while self.running.load(Ordering::Relaxed) {
            if let Ok(cmd) = cmd_rx.pop() {
                idle_spins = 0;

                // 1. Submit hardware step asynchronously to compute stream
                let step_event = self.backend.dispatch_step_kernel(&cmd)?;

                // 2. Non-blocking event query prevents host data race
                while !self.backend.query_event_completed(step_event) {
                    std::hint::spin_loop();
                }

                // 3. Extract sampled token from pinned host-mapped memory
                let sampled_token = self.backend.read_sampled_token_host(cmd.slot_idx);
                let completion = StepCompletion::new(cmd.sequence_id, sampled_token, false);

                // 4. Push completion to async plane
                while completion_tx.push(completion).is_err() {
                    if !self.running.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    std::hint::spin_loop();
                }
            } else {
                idle_spins = idle_spins.saturating_add(1);

                // Three-Phase Adaptive Scheduling
                if idle_spins <= 200 {
                    // Phase 1: Micro-spin for sub-microsecond latency bursts
                    std::hint::spin_loop();
                } else if idle_spins <= 2000 {
                    // Phase 2: Coarse yield to prevent CPU starvation
                    std::thread::yield_now();
                } else {
                    // Phase 3: Park thread via futex to eliminate thermal throttling
                    self.parker.park_timeout(Duration::from_micros(50));
                }
            }
        }

        Ok(())
    }
}
