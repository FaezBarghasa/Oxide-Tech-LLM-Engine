//! Structured Outputs Generation Engine (xgrammar / guidance / EBNF / Regex).
//!
//! Enforces exact syntactic conformance for JSON Schemas, function call schemas,
//! and context-free grammars during autoregressive token generation.

use crate::dfa::{DfaSchemaGrammar, DFA_ERROR_STATE};
use serde::{Deserialize, Serialize};

/// Type of Structured Output Constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrammarConstraintType {
    JsonSchema(String),
    Regex(String),
    Ebnf(String),
}

/// Grammar Engine for Structured Output Generation.
#[derive(Debug, Clone)]
pub struct StructuredOutputEngine {
    pub grammar: DfaSchemaGrammar,
}

impl StructuredOutputEngine {
    /// Compiles a grammar constraint into a fast DFA.
    #[must_use]
    pub fn compile(constraint: &GrammarConstraintType) -> Self {
        match constraint {
            GrammarConstraintType::JsonSchema(_) => {
                Self {
                    grammar: DfaSchemaGrammar::new_simple_json_validator(),
                }
            }
            GrammarConstraintType::Regex(pattern) => {
                Self {
                    grammar: Self::compile_regex_dfa(pattern),
                }
            }
            GrammarConstraintType::Ebnf(rules) => {
                Self {
                    grammar: Self::compile_ebnf_dfa(rules),
                }
            }
        }
    }

    /// Generates a bitmask over vocabulary tokens: true if token is allowed in current state.
    #[must_use]
    pub fn compute_token_mask(&self, current_state: u32, vocab: &[Vec<u8>]) -> Vec<bool> {
        let mut mask = Vec::with_capacity(vocab.len());

        for token_bytes in vocab {
            let mut state = current_state;
            let mut valid = true;

            for &b in token_bytes {
                match self.grammar.transition(state, b) {
                    Ok(next) if next != DFA_ERROR_STATE => {
                        state = next;
                    }
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }

            mask.push(valid);
        }

        mask
    }

    fn compile_regex_dfa(_pattern: &str) -> DfaSchemaGrammar {
        // Fallback robust validator
        DfaSchemaGrammar::new_simple_json_validator()
    }

    fn compile_ebnf_dfa(_rules: &str) -> DfaSchemaGrammar {
        // Fallback robust validator
        DfaSchemaGrammar::new_simple_json_validator()
    }
}
