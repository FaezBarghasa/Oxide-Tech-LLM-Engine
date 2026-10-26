//! Gated Delta Net (GDN) with Rollback by Replay.
//! Pure Rust implementation of Recurrent Linear Attention with Delta Rule and
//! memory-efficient speculative rollback (Prism ML / llamADALL architecture).
//!
//! # Architecture:
//! Standard recurrent state rollback in speculative decoding requires saving entire
//! $(d_v \times d_k)$ state matrices per speculative step.
//!
//! GDN Rollback by Replay maintains:
//! 1. A single **Base State** matrix $S_{\text{base}} \in \mathbb{R}^{d_v \times d_k}$.
//! 2. A compact **Record Ledger** storing per-token update vectors $(u_t, k_t, g_t)$
//!    where $u_t = v_t - \beta_t (S_{t-1} k_t)$ of size $O(d_v + d_k)$ rather than $O(d_v \times d_k)$.
//!
//! Rolling back $r$ tokens simply drops $r$ records from the ledger without restoring heavy matrices.

/// Per-token delta update record for replay.
#[derive(Debug, Clone, PartialEq)]
pub struct GdnUpdateRecord {
    pub u: Vec<f32>, // Size d_v: effective innovation vector
    pub k: Vec<f32>, // Size d_k: key vector
    pub g: f32,      // Gate decay scalar
}

/// Recurrent state for Gated Delta Net with Rollback by Replay.
#[derive(Debug, Clone)]
pub struct GdnReplayState {
    pub d_v: usize,
    pub d_k: usize,
    /// Base recurrent matrix of shape `[d_v, d_k]`.
    pub base_state: Vec<f32>,
    /// Live working state of shape `[d_v, d_k]`.
    pub working_state: Vec<f32>,
    /// Compact record ledger for pending speculative tokens.
    pub record_ledger: Vec<GdnUpdateRecord>,
}

impl GdnReplayState {
    /// Creates a new zeroed recurrent state.
    pub fn new(d_v: usize, d_k: usize) -> Self {
        let size = d_v * d_k;
        Self {
            d_v,
            d_k,
            base_state: vec![0.0f32; size],
            working_state: vec![0.0f32; size],
            record_ledger: Vec::new(),
        }
    }

    /// Resets the state to zero.
    pub fn reset(&mut self) {
        self.base_state.fill(0.0);
        self.working_state.fill(0.0);
        self.record_ledger.clear();
    }

    /// Commits the current working state into the base state and clears the record ledger.
    /// Called when speculative tokens are verified and accepted.
    pub fn commit(&mut self) {
        self.base_state.copy_from_slice(&self.working_state);
        self.record_ledger.clear();
    }

    /// Rolls back `r` unconfirmed speculative tokens by replaying only the retained records
    /// onto the base state, avoiding saving/restoring full $(d_v \times d_k)$ matrices.
    pub fn rollback(&mut self, r: usize) {
        if r >= self.record_ledger.len() {
            // Drop all unconfirmed tokens: revert to base state
            self.working_state.copy_from_slice(&self.base_state);
            self.record_ledger.clear();
            return;
        }

        let retain_count = self.record_ledger.len() - r;
        self.record_ledger.truncate(retain_count);

        // Replay retained records from base state
        self.working_state.copy_from_slice(&self.base_state);
        let d_v = self.d_v;
        let d_k = self.d_k;

        for rec in &self.record_ledger {
            // S = g * S + u * k^T
            for row in 0..d_v {
                let u_val = rec.u[row];
                let row_offset = row * d_k;
                for col in 0..d_k {
                    let s_val = self.working_state[row_offset + col];
                    self.working_state[row_offset + col] = rec.g * s_val + u_val * rec.k[col];
                }
            }
        }
    }
}

/// Gated Delta Net Layer parameters.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    pub d_in: usize,
    pub d_v: usize,
    pub d_k: usize,
}

/// Inputs for a single GDN forward step.
#[derive(Debug, Clone, Copy)]
pub struct GdnStepInputs<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub decay_gate: f32,
    pub delta_beta: f32,
}

impl GatedDeltaNet {
    pub fn new(d_in: usize, d_v: usize, d_k: usize) -> Self {
        Self { d_in, d_v, d_k }
    }

    /// Performs one forward step of Gated Delta Net:
    /// - Computes attention output: $o_t = S_{t-1} q_t$
    /// - Updates recurrent state via Delta rule:
    ///   $u_t = v_t - \beta_t (S_{t-1} k_t)$
    ///   $S_t = g_t \odot S_{t-1} + u_t k_t^T$
    /// - Appends update record to the replay ledger for zero-copy rollback.
    pub fn step(&self, inputs: GdnStepInputs<'_>, state: &mut GdnReplayState, output: &mut [f32]) {
        let d_v = self.d_v;
        let d_k = self.d_k;
        assert_eq!(inputs.query.len(), d_k);
        assert_eq!(inputs.key.len(), d_k);
        assert_eq!(inputs.value.len(), d_v);
        assert_eq!(output.len(), d_v);

        // 1. Output projection: o = S_{t-1} * q
        for row in 0..d_v {
            let mut accum = 0.0f32;
            let row_offset = row * d_k;
            for col in 0..d_k {
                accum += state.working_state[row_offset + col] * inputs.query[col];
            }
            output[row] = accum;
        }

        // 2. Compute prediction of current value: S_{t-1} * k
        let mut predicted_v = vec![0.0f32; d_v];
        for row in 0..d_v {
            let mut accum = 0.0f32;
            let row_offset = row * d_k;
            for col in 0..d_k {
                accum += state.working_state[row_offset + col] * inputs.key[col];
            }
            predicted_v[row] = accum;
        }

        // 3. Compute error / innovation: u = v - beta * (S * k)
        let mut innovation_u = vec![0.0f32; d_v];
        for row in 0..d_v {
            innovation_u[row] = inputs.value[row] - inputs.delta_beta * predicted_v[row];
        }

        // 4. Update working state: S_t = g * S_{t-1} + u * k^T
        for row in 0..d_v {
            let u_val = innovation_u[row];
            let row_offset = row * d_k;
            for col in 0..d_k {
                let s_val = state.working_state[row_offset + col];
                state.working_state[row_offset + col] =
                    inputs.decay_gate * s_val + u_val * inputs.key[col];
            }
        }

        // 5. Append compact record to replay ledger
        state.record_ledger.push(GdnUpdateRecord {
            u: innovation_u,
            k: inputs.key.to_vec(),
            g: inputs.decay_gate,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_rollback_by_replay_identity() {
        let d_v = 8;
        let d_k = 8;
        let gdn = GatedDeltaNet::new(16, d_v, d_k);
        let mut state = GdnReplayState::new(d_v, d_k);

        // Initial 2 warmup tokens
        let q = vec![0.1f32; d_k];
        let k = vec![0.2f32; d_k];
        let v = vec![0.3f32; d_v];
        let mut out = vec![0.0f32; d_v];

        let step_inputs = GdnStepInputs {
            query: &q,
            key: &k,
            value: &v,
            decay_gate: 0.95,
            delta_beta: 0.5,
        };

        gdn.step(step_inputs, &mut state, &mut out);
        gdn.step(step_inputs, &mut state, &mut out);
        // Confirmed base state
        state.commit();

        let state_after_commit = state.working_state.clone();

        // Now speculate 3 draft tokens
        gdn.step(step_inputs, &mut state, &mut out);
        gdn.step(step_inputs, &mut state, &mut out);
        gdn.step(step_inputs, &mut state, &mut out);

        assert_eq!(state.record_ledger.len(), 3);

        // Roll back all 3 speculative tokens
        state.rollback(3);

        assert_eq!(state.record_ledger.len(), 0);
        for (i, (&actual, &expected)) in state
            .working_state
            .iter()
            .zip(state_after_commit.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "State mismatch at {}: actual={}, expected={}",
                i,
                actual,
                expected
            );
        }
    }
}
