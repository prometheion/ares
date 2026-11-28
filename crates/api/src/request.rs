//! What a client asks the engine to generate.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Json, Message, Sampling, Structure};

/// Identifies a request from [`Engine::submit`](crate::Engine::submit) or
/// [`Engine::decide`](crate::Engine::decide) until its last event or its
/// outcome, and is the handle for [`Engine::cancel`](crate::Engine::cancel).
///
/// The front end makes ids up; the engine must treat them as opaque keys that
/// are unique among the requests it currently holds, of both kinds. They are
/// not dense and not an index into anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub u64);

impl fmt::Display for RequestId {
    /// Fixed-width hex, so that ids line up in logs.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// The input of a generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Prompt {
    /// A conversation. The engine renders it with the model's chat template.
    Chat {
        messages: Vec<Message>,
        /// Tool definitions as the client sent them, for the template.
        /// `None` when the client sent none.
        tools: Option<Vec<Json>>,
        /// Variables of the chat template for this request
        /// (`chat_template_kwargs`): they come on top of the ones the engine
        /// was started with and win over them.
        variables: Vec<(String, Json)>,
        /// The last message is the assistant's, and the reply continues it:
        /// the prompt ends with the text of that message as the template
        /// writes it, and not with the end of its turn and the start of a new
        /// one. With an empty content and some reasoning, the prompt ends
        /// inside the reasoning, which the reply continues.
        #[serde(default)]
        continue_final: bool,
    },
    /// Text to continue as it is: no template, no added roles. Special tokens
    /// the model needs (a beginning-of-text token, say) are still the
    /// engine's to add, and the text of a control token (`<|im_start|>`) is
    /// read as that token, as llama-server reads the prompt of a completion.
    Text(String),
}

/// One request to generate text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Which model to run, as listed by [`Engine::models`](crate::Engine::models).
    pub model: String,
    pub prompt: Prompt,
    pub sampling: Sampling,
    /// Upper bound on the generated tokens. `None` means "until the model
    /// stops or the context is full".
    pub max_tokens: Option<u32>,
    /// Generation ends before any of these strings would be emitted; the
    /// matched string itself is not part of the output.
    pub stop: Vec<String>,
    /// Whether the engine may verify drafts for this request. `false` makes
    /// the request a plain one-token-per-step generation, to compare with.
    pub draft: bool,
    /// Whether the request uses the prompt cache: starts from what it holds
    /// of the prompt and leaves what it computed there. `false` (llama.cpp's
    /// `cache_prompt: false`) computes the whole prompt and leaves nothing.
    #[serde(default = "yes")]
    pub cache: bool,
    /// A rule the reply must follow (a grammar, a JSON schema), or `None` for
    /// text that is free.
    pub structure: Option<Structure>,
    /// Who goes first when the engine is busy: a lower number before a
    /// higher one (vLLM's numbers), 0 for a request that does not say. A
    /// request is let in, given rows and read before the requests of a worse
    /// priority, and those give way to it when it does not fit.
    #[serde(default)]
    pub priority: i32,
    pub id: RequestId,
}

fn yes() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Interrupt, Role, Shape, Start};

    #[test]
    fn ids_print_in_fixed_width_hex() {
        assert_eq!(RequestId(255).to_string(), "00000000000000ff");
        assert_eq!(RequestId(u64::MAX).to_string(), "ffffffffffffffff");
    }

    #[test]
    fn a_request_survives_serialization() {
        let request = GenerateRequest {
            model: "m".into(),
            prompt: Prompt::Chat {
                messages: vec![Message::new(Role::User, "hi")],
                tools: Some(vec![Json::Object(vec![(
                    "type".into(),
                    Json::String("function".into()),
                )])]),
                variables: vec![("enable_thinking".into(), Json::Bool(false))],
                continue_final: true,
            },
            sampling: Sampling { seed: Some(7), ..Sampling::default() },
            max_tokens: Some(16),
            stop: vec!["\n\n".into()],
            draft: true,
            cache: false,
            structure: Some(Structure {
                shape: Shape::JsonSchema(r#"{"type":"object"}"#.into()),
                start: Start::AfterReasoning { open: None, close: "</think>".into() },
                trigger: Some("<tool_call>".into()),
                interrupt: Some(Interrupt {
                    text: "<tool_call>".into(),
                    shape: Shape::Lark("start: \"a\"".into()),
                }),
            }),
            priority: -3,
            id: RequestId(9),
        };
        let text = serde_json::to_string(&request).unwrap();
        assert_eq!(serde_json::from_str::<GenerateRequest>(&text).unwrap(), request);
    }
}
