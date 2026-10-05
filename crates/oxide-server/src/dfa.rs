use oxide_core::error::{EngineError, Result};

pub const DFA_ERROR_STATE: u32 = u32::MAX;

/// Deterministic Finite Automaton (DFA) state transition table for structured JSON / grammar constraints.
#[derive(Debug, Clone)]
pub struct DfaSchemaGrammar {
    pub num_states: usize,
    pub transitions: Vec<u32>, // Flat array of shape [num_states, 256]
    pub accepting_states: Vec<bool>,
}

impl DfaSchemaGrammar {
    /// Compiles a JSON-validating state machine.
    #[must_use]
    pub fn new_simple_json_validator() -> Self {
        let num_states = 8;
        let mut transitions = vec![DFA_ERROR_STATE; num_states * 256];
        let mut accepting_states = vec![false; num_states];

        // State 0: expects '{' -> goes to State 1
        transitions[b'{' as usize] = 1;

        // State 1: inside object after '{' or ',', expects '"' -> State 2, or '}' -> State 7
        transitions[1 * 256 + (b'"' as usize)] = 2;
        transitions[1 * 256 + (b'}' as usize)] = 7;

        // State 2: inside string key, any byte except '"' stays in State 2; '"' -> State 3
        for b in 0..256 {
            transitions[2 * 256 + b] = 2;
        }
        transitions[2 * 256 + (b'"' as usize)] = 3;

        // State 3: after key quote, expects ':' -> State 4
        transitions[3 * 256 + (b':' as usize)] = 4;

        // State 4: expects value: digit -> State 5, '"' -> State 6
        for b in b'0'..=b'9' {
            transitions[4 * 256 + (b as usize)] = 5;
        }
        transitions[4 * 256 + (b'"' as usize)] = 6;

        // State 5: inside number value, digits stay in State 5; ',' -> State 1; '}' -> State 7
        for b in b'0'..=b'9' {
            transitions[5 * 256 + (b as usize)] = 5;
        }
        transitions[5 * 256 + (b',' as usize)] = 1;
        transitions[5 * 256 + (b'}' as usize)] = 7;

        // State 6: inside string value, any byte except '"' stays in State 6; '"' -> State 5/accept-ready
        for b in 0..256 {
            transitions[6 * 256 + b] = 6;
        }
        transitions[6 * 256 + (b'"' as usize)] = 5;

        // State 7: Terminal accepting state
        accepting_states[7] = true;

        Self {
            num_states,
            transitions,
            accepting_states,
        }
    }

    /// Evaluates state transition on next input byte.
    pub fn transition(&self, current_state: u32, byte: u8) -> Result<u32> {
        if current_state as usize >= self.num_states {
            return Err(EngineError::SchemaMismatch {
                state: current_state,
                byte,
            });
        }

        let next_state = self.transitions[current_state as usize * 256 + byte as usize];
        if next_state == DFA_ERROR_STATE {
            return Err(EngineError::SchemaMismatch {
                state: current_state,
                byte,
            });
        }

        Ok(next_state)
    }

    /// Applies in-kernel DFA logit mask, setting invalid token logits to -inf.
    pub fn apply_dfa_mask(&self, current_state: u32, token_bytes: &[Vec<u8>], logits: &mut [f32]) {
        for (tok_id, bytes) in token_bytes.iter().enumerate() {
            let mut s = current_state;
            let mut valid = true;

            for &b in bytes {
                match self.transition(s, b) {
                    Ok(next_s) => s = next_s,
                    Err(_) => {
                        valid = false;
                        break;
                    }
                }
            }

            if !valid && tok_id < logits.len() {
                logits[tok_id] = f32::NEG_INFINITY;
            }
        }
    }
}
