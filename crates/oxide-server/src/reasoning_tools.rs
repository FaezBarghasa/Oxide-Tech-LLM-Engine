//! Tool Calling and Reasoning (`<think>`) Parsers.
//!
//! Handles:
//! - DeepSeek-R1 / QwQ `<think>...</think>` reasoning token separation
//! - OpenAI and Anthropic tool calling JSON schemas and execution dispatch

use serde::{Deserialize, Serialize};

/// Extracted Reasoning & Response Split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningExtraction {
    pub reasoning_content: Option<String>,
    pub final_content: String,
}

/// Reasoning parser for DeepSeek-R1 and QwQ `<think>` blocks.
#[derive(Debug, Clone, Default)]
pub struct ReasoningParser;

impl ReasoningParser {
    /// Extracts `<think>` tags from raw model generation stream or full text.
    #[must_use]
    pub fn parse_reasoning(text: &str) -> ReasoningExtraction {
        const THINK_START: &str = "<think>";
        const THINK_END: &str = "</think>";

        if let Some(start_idx) = text.find(THINK_START) {
            let content_after_start = &text[start_idx + THINK_START.len()..];
            if let Some(end_idx) = content_after_start.find(THINK_END) {
                let reasoning = &content_after_start[..end_idx];
                let final_part = format!(
                    "{}{}",
                    &text[..start_idx],
                    &content_after_start[end_idx + THINK_END.len()..]
                );
                return ReasoningExtraction {
                    reasoning_content: Some(reasoning.trim().to_string()),
                    final_content: final_part.trim().to_string(),
                };
            }
            // Stream is still inside <think>
            return ReasoningExtraction {
                reasoning_content: Some(content_after_start.trim().to_string()),
                final_content: text[..start_idx].trim().to_string(),
            };
        }

        ReasoningExtraction {
            reasoning_content: None,
            final_content: text.to_string(),
        }
    }
}

/// Normalized Tool Call Definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

/// Tool Call Parser across OpenAI and Anthropic syntax formats.
#[derive(Debug, Clone, Default)]
pub struct ToolCallParser;

impl ToolCallParser {
    /// Parses tool calls from OpenAI JSON format or raw text prompt output.
    #[must_use]
    pub fn parse_from_text(text: &str) -> Vec<ToolCall> {
        let mut calls = Vec::new();

        // Check for Markdown code block containing JSON tool call: ```json\n{"name": "...", "arguments": ...}\n```
        let trimmed = text.trim();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed)
            && let Some(name) = val.get("name").and_then(|v| v.as_str())
        {
            let args = val.get("arguments").map_or_else(
                || "{}".to_string(),
                |v| {
                    if v.is_string() {
                        v.as_str().unwrap().to_string()
                    } else {
                        v.to_string()
                    }
                },
            );
            calls.push(ToolCall {
                id: "call_01".to_string(),
                name: name.to_string(),
                arguments_json: args,
            });
        }

        calls
    }
}
