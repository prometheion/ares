//! What a chat template says about thinking.
//!
//! A model that thinks writes its reasoning between two tags before it answers,
//! and its chat template knows the tags: it writes them around the reasoning of
//! an earlier turn, and it may open the block itself when it ends the prompt.
//! [`ChatTemplate::thinking_format`] reads that off the template into a
//! [`ThinkingFormat`] by rendering small
//! conversations and comparing the texts, the way llama.cpp's autoparser does,
//! so that nothing is known by the name of a model:
//!
//! - an assistant message with `reasoning_content` is rendered, and the text
//!   around the reasoning is the pair of tags with the white space the template
//!   puts next to them;
//! - the end of the prompt is rendered with `enable_thinking` on and off, and
//!   the part of it that starts at the opening tag says whether the model
//!   continues inside the block and whether the template closes an empty one;
//! - a few values of `reasoning_effort` are tried, to see whether the template
//!   reads the variable and which values it refuses.
//!
//! A template that does not write a pair of tags around the reasoning of an
//! assistant message is taken to have no thinking: there is nothing to find.

use ares_api::ThinkingFormat;

use crate::chat::{ChatOptions, ChatTemplate, Message};
use crate::json::Json;

/// A reasoning text that no template contains by accident.
const REASONING: &str = "REASONING_PROBE_7f3a";
/// An answer text, likewise.
const ANSWER: &str = "ANSWER_PROBE_c91d";

/// The values of `reasoning_effort` that are tried, none of them is meant to be
/// the default of any template.
const EFFORTS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

/// Reads how `template` handles thinking, or `None` if it has no thinking
/// ([`ChatTemplate::thinking_format`]). Whether the model thinks when a
/// request does not say is what [`ChatOptions`] say by default; the engine
/// sets it from the options it renders with.
pub(crate) fn detect(template: &ChatTemplate) -> Option<ThinkingFormat> {
    let (start, end) = tags(template)?;
    let tag = without_space(&start).to_owned();
    // The end of the prompt, from the opening tag on.
    let prompt = |thinking| {
        let tail = prompt_tail(template, thinking)?;
        Some(tail.find(&tag).map(|at| tail[at..].to_owned()).unwrap_or_default())
    };
    Some(ThinkingFormat {
        start,
        end,
        prompt_on: prompt(true)?,
        prompt_off: prompt(false)?,
        on_by_default: ChatOptions::default().enable_thinking,
        effort: reads_effort(template),
        refused_efforts: refused_efforts(template),
    })
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

/// The usual values of `reasoning_effort` that a template which reads the
/// variable refuses; none for a template that does not read it.
fn refused_efforts(template: &ChatTemplate) -> Vec<String> {
    if !reads_effort(template) {
        return Vec::new();
    }
    let user = [Message::new("user", "Hi")];
    EFFORTS
        .iter()
        .filter(|effort| {
            let variables =
                vec![("reasoning_effort".to_owned(), Json::String((**effort).to_owned()))];
            template
                .render(&user, &[], &ChatOptions { variables, ..ChatOptions::default() })
                .is_err()
        })
        .map(|effort| (*effort).to_owned())
        .collect()
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
