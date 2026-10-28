#![deny(
    unsafe_op_in_unsafe_fn,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use serde::{Deserialize, Serialize};

/// Chat Message Role.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Single Chat Message in a Conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    #[must_use]
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Standardized Chat Template Formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatTemplateFormat {
    /// ChatML (`<|im_start|>role\ncontent<|im_end|>\n`) used by Qwen, Yi, DeepSeek-Chat.
    ChatMl,
    /// Llama 3 (`<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`).
    Llama3,
    /// DeepSeek R1 / V3 Reasoning format (`<|User|>...\n<|Assistant|><think>...\n</think>...`).
    DeepSeek,
    /// Mistral (`[INST] prompt [/INST]`).
    Mistral,
    /// Alpaca (`### Instruction:\n...\n\n### Response:\n`).
    Alpaca,
    /// Gemma 2 / Gemma 4 (`<start_of_turn>user\ncontent<end_of_turn>\n<start_of_turn>model\n`).
    Gemma,
    /// Raw unformatted plaintext concat.
    RawText,
}

impl ChatTemplateFormat {
    /// Detects the recommended chat template format from model architecture or filename.
    #[must_use]
    pub fn detect(model_arch: &str, model_name: &str) -> Self {
        let arch = model_arch.to_lowercase();
        let name = model_name.to_lowercase();
        if arch.contains("qwen")
            || name.contains("qwen")
            || arch.contains("yi")
            || name.contains("yi")
        {
            Self::ChatMl
        } else if arch.contains("llama") || name.contains("llama") {
            Self::Llama3
        } else if arch.contains("gemma") || name.contains("gemma") {
            Self::Gemma
        } else if arch.contains("deepseek") || name.contains("deepseek") {
            Self::ChatMl
        } else if arch.contains("mistral") || name.contains("mistral") {
            Self::Mistral
        } else {
            Self::ChatMl
        }
    }

    /// Returns the standard stop token IDs for this template format.
    #[must_use]
    pub fn default_stop_token_ids(&self) -> &'static [u32] {
        match self {
            Self::ChatMl => &[151643, 151645], // <|endoftext|>, <|im_end|>
            Self::Llama3 => &[128001, 128009], // <|end_of_text|>, <|eot_id|>
            Self::Gemma => &[1, 2, 107],       // <bos>, <eos>, <end_of_turn>
            Self::DeepSeek => &[100001, 151643, 151645],
            Self::Mistral | Self::Alpaca | Self::RawText => &[2],
        }
    }
}

/// Chat Template Parser and Prompt Serializer.
#[derive(Debug, Clone)]
pub struct ChatTemplateParser {
    pub format: ChatTemplateFormat,
    pub add_generation_prompt: bool,
}

impl ChatTemplateParser {
    #[must_use]
    pub fn new(format: ChatTemplateFormat) -> Self {
        Self {
            format,
            add_generation_prompt: true,
        }
    }

    /// Formats a list of structured chat messages into a unified prompt string for the model.
    #[must_use]
    pub fn render_chat(&self, messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();

        match self.format {
            ChatTemplateFormat::ChatMl => {
                for msg in messages {
                    let role_str = match msg.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::Tool => "tool",
                    };
                    prompt.push_str(&format!(
                        "<|im_start|>{role_str}\n{}<|im_end|>\n",
                        msg.content
                    ));
                }
                if self.add_generation_prompt {
                    prompt.push_str("<|im_start|>assistant\n");
                }
            }
            ChatTemplateFormat::Llama3 => {
                prompt.push_str("<|begin_of_text|>");
                for msg in messages {
                    let role_str = match msg.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::Tool => "ipython",
                    };
                    prompt.push_str(&format!(
                        "<|start_header_id|>{role_str}<|end_header_id|>\n\n{}<|eot_id|>",
                        msg.content
                    ));
                }
                if self.add_generation_prompt {
                    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
            }
            ChatTemplateFormat::DeepSeek => {
                for msg in messages {
                    match msg.role {
                        ChatRole::System => {
                            prompt.push_str(&format!("System: {}\n\n", msg.content));
                        }
                        ChatRole::User => {
                            prompt.push_str(&format!("<|User|>{}\n", msg.content));
                        }
                        ChatRole::Assistant => {
                            prompt.push_str(&format!("<|Assistant|>{}\n", msg.content));
                        }
                        ChatRole::Tool => {
                            prompt.push_str(&format!("<|Tool|>{}\n", msg.content));
                        }
                    }
                }
                if self.add_generation_prompt {
                    prompt.push_str("<|Assistant|>");
                }
            }
            ChatTemplateFormat::Mistral => {
                for msg in messages {
                    match msg.role {
                        ChatRole::System => {
                            prompt.push_str(&format!(
                                "[SYSTEM_PROMPT] {} [/SYSTEM_PROMPT]",
                                msg.content
                            ));
                        }
                        ChatRole::User => {
                            prompt.push_str(&format!("[INST] {} [/INST]", msg.content));
                        }
                        ChatRole::Assistant => {
                            prompt.push_str(&format!("{}</s>", msg.content));
                        }
                        ChatRole::Tool => {
                            prompt.push_str(&format!(
                                "[TOOL_RESULTS] {} [/TOOL_RESULTS]",
                                msg.content
                            ));
                        }
                    }
                }
            }
            ChatTemplateFormat::Alpaca => {
                for msg in messages {
                    match msg.role {
                        ChatRole::System => {
                            prompt.push_str(&format!("### System:\n{}\n\n", msg.content));
                        }
                        ChatRole::User => {
                            prompt.push_str(&format!("### Instruction:\n{}\n\n", msg.content));
                        }
                        ChatRole::Assistant => {
                            prompt.push_str(&format!("### Response:\n{}\n\n", msg.content));
                        }
                        ChatRole::Tool => {
                            prompt.push_str(&format!("### Tool Result:\n{}\n\n", msg.content));
                        }
                    }
                }
                if self.add_generation_prompt {
                    prompt.push_str("### Response:\n");
                }
            }
            ChatTemplateFormat::Gemma => {
                prompt.push_str("<bos>");
                for msg in messages {
                    let role_str = match msg.role {
                        ChatRole::System | ChatRole::User | ChatRole::Tool => "user",
                        ChatRole::Assistant => "model",
                    };
                    prompt.push_str(&format!(
                        "<start_of_turn>{role_str}\n{}<end_of_turn>\n",
                        msg.content
                    ));
                }
                if self.add_generation_prompt {
                    prompt.push_str("<start_of_turn>model\n");
                }
            }
            ChatTemplateFormat::RawText => {
                for msg in messages {
                    prompt.push_str(&msg.content);
                    prompt.push('\n');
                }
            }
        }

        prompt
    }
}
