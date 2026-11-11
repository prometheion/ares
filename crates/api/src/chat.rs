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
