//! What a chat template says about thinking.
//!
//! A model that thinks writes its reasoning between two tags before it answers,
//! and its chat template knows the tags: it writes them around the reasoning of
//! an earlier turn, and it may open the block itself when it ends the prompt.
//! [`ReasoningFormat::detect`] reads that off the template by rendering small
//! conversations and comparing the texts, the way llama.cpp's autoparser does,
//! so that nothing is known by the name of a model:
//!
//! - an assistant message with `reasoning_content` is rendered, and the text
//!   around the reasoning is the pair of tags with the white space the template
//!   puts next to them;
//! - the end of the prompt is rendered with `enable_thinking` on and off, and
//!   the part of it that starts at the opening tag says whether the model
//!   continues inside the block and whether the template closes an empty one;
//! - the reasoning of an earlier turn is rendered with `preserve_reasoning` on,
//!   and a few values of `reasoning_effort` are tried, to see which of these
//!   two variables the template reads.
//!
//! A template that does not write a pair of tags around the reasoning of an
//! assistant message is taken to have no thinking: there is nothing to find.

use crate::chat::{ChatOptions, ChatTemplate, Message};
use crate::json::Json;

/// A reasoning text that no template contains by accident.
const REASONING: &str = "REASONING_PROBE_7f3a";
/// An answer text, likewise.
const ANSWER: &str = "ANSWER_PROBE_c91d";

/// The values of `reasoning_effort` that are tried, none of them is meant to be
/// the default of any template.
const EFFORTS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

/// How a chat template handles thinking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReasoningFormat {
    /// What the template writes before the reasoning of an assistant message:
    /// the opening tag and the white space after it (`"<think>\n"`).
    pub start: String,
    /// What it writes after the reasoning: the white space before the closing
    /// tag, the tag, and the white space after it (`"\n</think>\n\n"`).
    pub end: String,
    /// The end of the prompt with thinking on, from the opening tag on
    /// (`"<think>\n"`), or empty if the prompt does not contain the tag.
    pub prompt_on: String,
    /// The same with thinking off (`"<think>\n\n</think>\n\n"`).
    pub prompt_off: String,
    /// Whether the template keeps the reasoning of earlier turns in the
    /// history when `preserve_reasoning` is on.
    pub keeps_history: bool,
    /// Whether the template reads `reasoning_effort`.
    pub effort: bool,
}

impl ReasoningFormat {
    /// Reads the format off `template`, or `None` if the template has no
    /// thinking.
    pub fn detect(template: &ChatTemplate) -> Option<Self> {
        let (start, end) = tags(template)?;
        let tag = without_space(&start).to_owned();
        // The end of the prompt, from the opening tag on.
        let prompt = |thinking| {
            let tail = prompt_tail(template, thinking)?;
            Some(tail.find(&tag).map(|at| tail[at..].to_owned()).unwrap_or_default())
        };
        Some(Self {
            start,
            end,
            prompt_on: prompt(true)?,
            prompt_off: prompt(false)?,
            keeps_history: keeps_history(template),
            effort: reads_effort(template),
        })
    }

    /// The opening tag without the white space around it.
    pub fn start_tag(&self) -> &str {
        without_space(&self.start)
    }

    /// The closing tag without the white space around it.
    pub fn end_tag(&self) -> &str {
        without_space(&self.end)
    }

    /// Whether the prompt with thinking on ends inside the block: the model
    /// then continues with the reasoning.
    pub fn opens(&self) -> bool {
        !self.prompt_on.is_empty() && !self.prompt_on.contains(self.end_tag())
    }

    /// Whether `enable_thinking` changes the end of the prompt.
    pub fn switchable(&self) -> bool {
        self.prompt_on != self.prompt_off
    }
}

/// `text` without the ASCII white space at both ends, which is what templates
/// put next to their tags.
fn without_space(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_ascii_whitespace())
}

/// The tags the template writes around the reasoning of an assistant message.
fn tags(template: &ChatTemplate) -> Option<(String, String)> {
    let mut assistant = Message::new("assistant", ANSWER);
    assistant.reasoning_content = REASONING.to_owned();
    let text = template
        .render(
            &[Message::new("user", "Hi"), assistant],
            &[],
            &ChatOptions { add_generation_prompt: false, ..ChatOptions::default() },
        )
        .ok()?;
    let at = text.find(REASONING)?;
    let (before, after) = (&text[..at], &text[at + REASONING.len()..]);

    // The tag is the last marker before the reasoning and the first one after
    // it, with nothing but white space between them and the reasoning.
    let opening = before.trim_end_matches(|c: char| c.is_ascii_whitespace());
    let open_at = opening.len() - marker_before(opening)?;
    let start = before[open_at..].to_owned();

    let gap = after.len() - after.trim_start_matches(|c: char| c.is_ascii_whitespace()).len();
    let closing = &after[gap..];
    let tag_len = marker_after(closing)?;
    let rest = &closing[tag_len..];
    let tail = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_whitespace()).len();
    // The answer follows at once: the tag closes the reasoning.
    if !rest[tail..].starts_with(ANSWER) {
        return None;
    }
    Some((start, after[..gap + tag_len + tail].to_owned()))
}

/// The longest marker (`<think>`, `<|think|>`, `[THINK]`) that ends `text`:
/// its length, or `None` if `text` does not end in one.
fn marker_before(text: &str) -> Option<usize> {
    let (open, close) = match text.chars().last()? {
        '>' => ('<', '>'),
        ']' => ('[', ']'),
        _ => return None,
    };
    let at = text.rfind(open)?;
    let marker = &text[at..];
    marker_is_whole(marker, close).then_some(marker.len())
}

/// The length of the marker that starts `text`, or `None`.
fn marker_after(text: &str) -> Option<usize> {
    let close = match text.chars().next()? {
        '<' => '>',
        '[' => ']',
        _ => return None,
    };
    let len = text.find(close)? + 1;
    marker_is_whole(&text[..len], close).then_some(len)
}

/// A marker is short and has no white space in it.
fn marker_is_whole(marker: &str, close: char) -> bool {
    marker.len() <= 64
        && marker.len() > 2
        && marker.ends_with(close)
        && !marker.contains(|c: char| c.is_ascii_whitespace())
}

/// What the template appends to the prompt of a first user message to start
/// the answer, with thinking `on` or off: the text the prompt has with the
/// start of the answer and does not have without it.
fn prompt_tail(template: &ChatTemplate, thinking: bool) -> Option<String> {
    let user = [Message::new("user", "Hi")];
    let with = ChatOptions { enable_thinking: thinking, ..ChatOptions::default() };
    let without = ChatOptions { add_generation_prompt: false, ..with.clone() };
    let full = template.render(&user, &[], &with).ok()?;
    let base = template.render(&user, &[], &without).ok()?;
    let mut common = full.bytes().zip(base.bytes()).take_while(|(a, b)| a == b).count();
    while !full.is_char_boundary(common) {
        common -= 1;
    }
    Some(full[common..].to_owned())
}

/// Whether the reasoning of an earlier turn is in the prompt when
/// `preserve_reasoning` is on.
fn keeps_history(template: &ChatTemplate) -> bool {
    let mut assistant = Message::new("assistant", ANSWER);
    assistant.reasoning_content = REASONING.to_owned();
    let conversation = [Message::new("user", "Hi"), assistant, Message::new("user", "Again")];
    let options = ChatOptions {
        add_generation_prompt: false,
        variables: vec![("preserve_reasoning".to_owned(), Json::Bool(true))],
        ..ChatOptions::default()
    };
    template.render(&conversation, &[], &options).is_ok_and(|text| text.contains(REASONING))
}

/// Whether some value of `reasoning_effort` changes the prompt. A value the
/// template refuses is not an answer; a template that accepts only some of
/// them still shows itself by those.
fn reads_effort(template: &ChatTemplate) -> bool {
    let user = [Message::new("user", "Hi")];
    let render = |variables: Vec<(String, Json)>| {
        template.render(&user, &[], &ChatOptions { variables, ..ChatOptions::default() })
    };
    let Ok(plain) = render(Vec::new()) else { return false };
    EFFORTS.iter().any(|effort| {
        render(vec![("reasoning_effort".to_owned(), Json::String((*effort).to_owned()))])
            .is_ok_and(|text| text != plain)
    })
}
