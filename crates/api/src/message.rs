//! Chat messages.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::Json;

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// The lowercase name chat templates use (`"user"`, `"assistant"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One message of a conversation.
///
/// `content` is plain text: a front end that accepts richer content (a list
/// of parts) flattens it before it gets here, or refuses what it cannot
/// flatten. The other four fields exist for the chat template and are never
/// interpreted on the way; they are `None` when the client did not send them,
/// which is not the same as an empty value and templates can tell the
/// difference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// The participant name (`name` in the OpenAI format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// On a [`Role::Tool`] message: the call this message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// On an [`Role::Assistant`] message: the tool calls the assistant made,
    /// each as the client sent it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Json>>,
    /// On an [`Role::Assistant`] message: what the assistant thought before it
    /// answered, as an earlier reply of the engine gave it. The template writes
    /// it back into the prompt, so that a reply that is sent back is the text
    /// the model generated and the prefix cache continues at its end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    /// A message with only a role and a text.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_use_the_lowercase_names_of_chat_templates() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let text = serde_json::to_string(&role).unwrap();
            assert_eq!(text, format!("\"{}\"", role.as_str()));
            assert_eq!(serde_json::from_str::<Role>(&text).unwrap(), role);
        }
        assert!(serde_json::from_str::<Role>("\"developer\"").is_err());
    }

    #[test]
    fn tool_calls_come_back_exactly_as_they_went_in() {
        let text = r#"{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"z\":1,\"a\":2}"}}]}"#;
        let message: Message = serde_json::from_str(text).unwrap();
        assert_eq!(message.tool_calls.as_ref().map(Vec::len), Some(1));
        assert_eq!(serde_json::to_string(&message).unwrap(), text);
    }

    #[test]
    fn the_reasoning_of_a_reply_comes_back_exactly_as_it_went_in() {
        let text = r#"{"role":"assistant","content":"4","reasoning_content":"2 and 2\n"}"#;
        let message: Message = serde_json::from_str(text).unwrap();
        assert_eq!(message.reasoning_content.as_deref(), Some("2 and 2\n"));
        assert_eq!(serde_json::to_string(&message).unwrap(), text);
    }

    #[test]
    fn absent_optional_fields_stay_absent() {
        let message: Message = serde_json::from_str(r#"{"role":"user","content":"hi"}"#).unwrap();
        assert_eq!(message, Message::new(Role::User, "hi"));
        assert_eq!(serde_json::to_string(&message).unwrap(), r#"{"role":"user","content":"hi"}"#);
    }
}
