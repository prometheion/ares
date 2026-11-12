//! What a model's chat template declares, as far as the front end needs it.

use serde::{Deserialize, Serialize};

/// What a model's chat template declares.
///
/// The engine reads it off the template (nothing here is known by the name of
/// a model) and the front end uses it to understand the text the model
/// generates and to refuse what the template cannot do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatFormat {
    /// How the model thinks, or `None` if its template has no thinking.
    pub thinking: Option<ThinkingFormat>,
    /// How the model writes a call of a tool, or `None` if its template
    /// writes calls in no way the engine knows (or has no tools at all).
    pub tools: Option<ToolFormat>,
}

/// How a model writes the calls of tools into its reply: every call between
/// tags, the name of the tool in a tag of its own, and each argument between
/// a tag that names it and a tag that closes it. The template of Qwen3.8
/// writes
///
/// ```text
/// <tool_call>
/// <function=NAME>
/// <parameter=KEY>
/// VALUE
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// A value that is a string stands as it is, without quotes; every other
/// value is JSON. The pieces are what the template writes around the name,
/// the keys and the values, white space included.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolFormat {
    /// From the start of a call to the name of the tool
    /// (`"<tool_call>\n<function="`).
    pub call_open: String,
    /// After the name (`">\n"`).
    pub name_close: String,
    /// Before the key of an argument (`"<parameter="`).
    pub param_open: String,
    /// Between the key and the value (`">\n"`).
    pub key_close: String,
    /// After the value (`"\n</parameter>\n"`).
    pub param_close: String,
    /// After the last argument, to the end of the call
    /// (`"</function>\n</tool_call>"`).
    pub call_close: String,
    /// What the template writes between the text of a message and its first
    /// call (`"\n\n"`).
    pub before_calls: String,
    /// What it writes between two calls (`"\n"`).
    pub between_calls: String,
}

impl ToolFormat {
    /// The text that tells a call has begun: the start of `call_open` up to
    /// its first white space (`"<tool_call>"`).
    pub fn trigger(&self) -> &str {
        let end = self.call_open.find(|c: char| c.is_ascii_whitespace());
        &self.call_open[..end.unwrap_or(self.call_open.len())]
    }

    /// What follows the trigger in `call_open` (`"\n<function="`).
    pub fn after_trigger(&self) -> &str {
        &self.call_open[self.trigger().len()..]
    }
}

/// How a model thinks before it answers: reasoning between two tags, whose
/// opening the template may already have written into the prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingFormat {
    /// What the template writes before the reasoning of an assistant message:
    /// the opening tag and the white space after it (`"<think>\n"`).
    pub start: String,
    /// What it writes after the reasoning: the white space before the closing
    /// tag, the tag and the white space after it (`"\n</think>\n\n"`).
    pub end: String,
    /// The end of the prompt from the opening tag on when the model thinks
    /// (`"<think>\n"`; empty if the prompt has no tag and the model writes the
    /// opening itself).
    pub prompt_on: String,
    /// The same when thinking is off (`"<think>\n\n</think>\n\n"`).
    pub prompt_off: String,
    /// Whether the model thinks when a request does not say: the template's
    /// own answer, changed by the variables the engine was started with.
    pub on_by_default: bool,
    /// Whether the template reads `reasoning_effort`.
    pub effort: bool,
}

impl ThinkingFormat {
    /// The opening tag without the white space around it.
    pub fn start_tag(&self) -> &str {
        self.start.trim_matches(|c: char| c.is_ascii_whitespace())
    }

    /// The closing tag without the white space around it.
    pub fn end_tag(&self) -> &str {
        self.end.trim_matches(|c: char| c.is_ascii_whitespace())
    }

    /// Whether `enable_thinking` changes the end of the prompt.
    pub fn switchable(&self) -> bool {
        self.prompt_on != self.prompt_off
    }
}
