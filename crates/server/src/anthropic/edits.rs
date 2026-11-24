//! The edits of `context_management`: thinking and results of tools that are
//! cleared from the conversation before it reaches the template, as
//! Anthropic's service clears them, and what a response says about them
//! (`context_management.applied_edits`).
//!
//! The edits are read with the body (`request::context_management`) and
//! carried out on the chat request after its model is known, because
//! `clear_tool_uses_20250919` is decided by the tokens of the prompt, which
//! only the engine counts. Each edit that changes the prompt is reported with
//! what it cleared and the tokens of the prompt it took away, which costs a
//! count of the prompt before and after it (a rendering and a tokenization,
//! no device); an edit that changes nothing costs no count.

use ares_api::{Json, Message, Prompt, Role};

use crate::error::ApiError;
use crate::generate;
use crate::models::Lease;
use crate::openai::request::Parsed;

/// What a result of a tool that is cleared becomes: Anthropic's service
/// writes "placeholder text indicating to Claude that it was removed" and does
/// not publish it.
pub(crate) const CLEARED_RESULT: &str = "[tool result cleared to save context]";

/// One edit of `context_management.edits`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Edit {
    /// `clear_thinking_20251015`: the thinking of the turns before the last
    /// `keep` ones leaves the prompt.
    ClearThinking { keep: Keep },
    /// `clear_tool_uses_20250919`.
    ClearToolUses(ToolUses),
}

/// How many turns of the assistant keep their thinking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Keep {
    All,
    /// The last this many turns, the one in progress among them: Anthropic's
    /// `{"type": "thinking_turns", "value": n}`. A turn begins with a message
    /// of the user that is not only results of tools, and it counts whether
    /// it has thinking or not.
    Turns(usize),
}

/// `clear_tool_uses_20250919`: once the prompt is past `trigger`, the results
/// of the oldest calls leave the prompt, all but the last `keep` calls and the
/// calls of the tools in `exclude`; with `clear_inputs` the arguments of those
/// calls too. Nothing is cleared when that would take away fewer than
/// `clear_at_least` tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolUses {
    pub trigger: Trigger,
    pub keep: usize,
    pub clear_at_least: Option<u32>,
    pub exclude: Vec<String>,
    pub clear_inputs: bool,
}

/// When `clear_tool_uses_20250919` applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Trigger {
    /// The prompt has more tokens than this.
    InputTokens(u32),
    /// The conversation has more calls of tools than this.
    ToolUses(u32),
}

impl Default for ToolUses {
    /// Anthropic's defaults: past 100 000 tokens, the last three calls kept.
    fn default() -> Self {
        Self {
            trigger: Trigger::InputTokens(100_000),
            keep: 3,
            clear_at_least: None,
            exclude: Vec::new(),
            clear_inputs: false,
        }
    }
}

/// An edit that changed the prompt, as a response reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Applied {
    Thinking { turns: u32, tokens: u32 },
    ToolUses { uses: u32, tokens: u32 },
}

impl Applied {
    /// The member of `applied_edits` in Anthropic's words.
    pub fn json(&self) -> String {
        match *self {
            Applied::Thinking { turns, tokens } => format!(
                r#"{{"type":"clear_thinking_20251015","cleared_thinking_turns":{turns},"cleared_input_tokens":{tokens}}}"#
            ),
            Applied::ToolUses { uses, tokens } => format!(
                r#"{{"type":"clear_tool_uses_20250919","cleared_tool_uses":{uses},"cleared_input_tokens":{tokens}}}"#
            ),
        }
    }
}

/// The member `context_management` of a response: the edits that changed the
/// prompt, which may be none.
pub(crate) fn response_member(applied: &[Applied]) -> String {
    let edits: Vec<String> = applied.iter().map(Applied::json).collect();
    format!(r#"{{"applied_edits":[{}]}}"#, edits.join(","))
}

/// What the edits did to a request: the edits that changed the prompt, and
/// the tokens of the prompt before any of them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Outcome {
    pub applied: Vec<Applied>,
    pub original_tokens: Option<u32>,
}

/// The turns of a conversation as Anthropic counts them for its edits: a
/// turn begins with a message of the user that is not only results of tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Turns {
    /// The turn of each message of the assistant, in order.
    pub of_assistant: Vec<usize>,
    /// The turn in progress: the one of the last message of the user, which
    /// the reply answers (or continues).
    pub current: usize,
}

/// Carries out `edits` on the conversation of `parsed`, whose model is the
/// one of `lease`; `turns` are its turns as the body counts them
/// (`request::conversation`).
pub(crate) async fn apply(
    edits: &[Edit],
    turns: &Turns,
    parsed: &mut Parsed,
    lease: &Lease,
) -> Result<Outcome, ApiError> {
    let mut outcome = Outcome::default();
    // The tokens of the prompt as it is now, counted when an edit needs them.
    let mut tokens: Option<u32> = None;
    for edit in edits {
        match edit {
            Edit::ClearThinking { keep } => {
                let before = parsed.request.prompt.clone();
                let turns = clear_thinking(messages(parsed)?, turns, *keep);
                if turns == 0 {
                    continue;
                }
                let was = match tokens {
                    Some(tokens) => tokens,
                    None => count(lease, parsed, before).await?,
                };
                outcome.original_tokens.get_or_insert(was);
                let now = count(lease, parsed, parsed.request.prompt.clone()).await?;
                tokens = Some(now);
                let tokens = was.saturating_sub(now);
                outcome.applied.push(Applied::Thinking { turns, tokens });
            }
            Edit::ClearToolUses(rule) => {
                let calls = calls(messages(parsed)?);
                if let Trigger::ToolUses(limit) = rule.trigger
                    && calls.len() <= limit as usize
                {
                    continue;
                }
                let cleared: Vec<usize> = {
                    let keep_from = calls.len().saturating_sub(rule.keep);
                    (0..keep_from).filter(|&i| !rule.exclude.contains(&calls[i].name)).collect()
                };
                if cleared.is_empty() {
                    continue;
                }
                let was = match tokens {
                    Some(tokens) => tokens,
                    None => count(lease, parsed, parsed.request.prompt.clone()).await?,
                };
                tokens = Some(was);
                if let Trigger::InputTokens(limit) = rule.trigger
                    && was <= limit
                {
                    continue;
                }
                let before = parsed.request.prompt.clone();
                clear_calls(messages(parsed)?, &calls, &cleared, rule.clear_inputs);
                let now = count(lease, parsed, parsed.request.prompt.clone()).await?;
                let taken = was.saturating_sub(now);
                if rule.clear_at_least.is_some_and(|least| taken < least) {
                    parsed.request.prompt = before;
                    continue;
                }
                outcome.original_tokens.get_or_insert(was);
                tokens = Some(now);
                let uses = cleared.len() as u32;
                outcome.applied.push(Applied::ToolUses { uses, tokens: taken });
            }
        }
    }
    Ok(outcome)
}

fn messages(parsed: &mut Parsed) -> Result<&mut Vec<Message>, ApiError> {
    match &mut parsed.request.prompt {
        Prompt::Chat { messages, .. } => Ok(messages),
        Prompt::Text(_) => Err(ApiError::internal("a Messages request without a conversation")),
    }
}

/// The tokens of the prompt `prompt` for the request of `parsed`.
async fn count(lease: &Lease, parsed: &Parsed, prompt: Prompt) -> Result<u32, ApiError> {
    let mut request = parsed.request.clone();
    request.prompt = prompt;
    generate::count_prompt(lease, request).await
}

/// Takes the thinking out of the messages of the assistant whose turn is
/// before the last `keep` turns, and returns how many turns lost some.
fn clear_thinking(messages: &mut [Message], turns: &Turns, keep: Keep) -> u32 {
    let Keep::Turns(keep) = keep else { return 0 };
    let first_kept = (turns.current + 1).saturating_sub(keep);
    let mut cleared: Vec<usize> = Vec::new();
    let assistant = messages.iter_mut().filter(|m| m.role == Role::Assistant);
    for (message, &turn) in assistant.zip(&turns.of_assistant) {
        if turn < first_kept
            && message.reasoning_content.take().is_some()
            && !cleared.contains(&turn)
        {
            cleared.push(turn);
        }
    }
    cleared.len() as u32
}

/// A call of a tool in the conversation: the message of the assistant that
/// made it, its place there, its name and id.
#[derive(Debug, Clone)]
struct Call {
    message: usize,
    place: usize,
    name: String,
    id: String,
}

/// The calls of the conversation, oldest first.
fn calls(messages: &[Message]) -> Vec<Call> {
    let mut found = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        let Some(calls) = &message.tool_calls else { continue };
        for (place, call) in calls.iter().enumerate() {
            let text = |value: Option<&Json>| match value {
                Some(Json::String(text)) => text.clone(),
                _ => String::new(),
            };
            let name = text(call.get("function").and_then(|f| f.get("name")));
            found.push(Call { message: i, place, name, id: text(call.get("id")) });
        }
    }
    found
}

/// Clears the results of the calls `cleared` (places in `calls`), and with
/// `inputs` their arguments.
fn clear_calls(messages: &mut [Message], calls: &[Call], cleared: &[usize], inputs: bool) {
    for &k in cleared {
        let call = &calls[k];
        for message in messages.iter_mut().skip(call.message + 1) {
            if message.role == Role::Tool && message.tool_call_id.as_deref() == Some(&call.id) {
                CLEARED_RESULT.clone_into(&mut message.content);
                break;
            }
        }
        if inputs
            && let Some(Json::Object(members)) = messages[call.message]
                .tool_calls
                .as_mut()
                .and_then(|calls| calls.get_mut(call.place))
                .and_then(|call| match call {
                    Json::Object(members) => {
                        members.iter_mut().find(|(name, _)| name == "function").map(|(_, f)| f)
                    }
                    _ => None,
                })
        {
            for (name, value) in members.iter_mut() {
                if name == "arguments" {
                    *value = Json::Object(Vec::new());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(reasoning: Option<&str>, calls: &[(&str, &str)]) -> Message {
        let mut message = Message::new(Role::Assistant, "answer");
        message.reasoning_content = reasoning.map(str::to_owned);
        if !calls.is_empty() {
            message.tool_calls = Some(
                calls
                    .iter()
                    .map(|(id, name)| {
                        Json::Object(vec![
                            ("id".into(), Json::String((*id).into())),
                            ("type".into(), Json::String("function".into())),
                            (
                                "function".into(),
                                Json::Object(vec![
                                    ("name".into(), Json::String((*name).into())),
                                    (
                                        "arguments".into(),
                                        Json::Object(vec![("x".into(), Json::Int(1))]),
                                    ),
                                ]),
                            ),
                        ])
                    })
                    .collect(),
            );
        }
        message
    }

    fn result(id: &str) -> Message {
        let mut message = Message::new(Role::Tool, "a long result");
        message.tool_call_id = Some(id.into());
        message
    }

    #[test]
    fn thinking_is_cleared_by_turns_the_one_in_progress_counted() {
        // Turn 0: a question and an answer that thinks; turn 1: a question, a
        // call that thinks, its result and an answer without thinking; turn
        // 2: the question being answered.
        let mut messages = vec![
            Message::new(Role::User, "q0"),
            assistant(Some("t0"), &[]),
            Message::new(Role::User, "q1"),
            assistant(Some("t1"), &[("c1", "read")]),
            result("c1"),
            assistant(None, &[]),
            Message::new(Role::User, "q2"),
        ];
        let turns = Turns { of_assistant: vec![0, 1, 1], current: 2 };
        let mut kept = messages.clone();
        assert_eq!(clear_thinking(&mut kept, &turns, Keep::All), 0);
        assert_eq!(kept, messages);
        // Two turns: the one in progress and turn 1.
        let mut two = messages.clone();
        assert_eq!(clear_thinking(&mut two, &turns, Keep::Turns(2)), 1);
        assert_eq!(two[1].reasoning_content, None);
        assert_eq!(two[3].reasoning_content.as_deref(), Some("t1"));
        // One turn: the one in progress, which has no answer yet.
        assert_eq!(clear_thinking(&mut messages, &turns, Keep::Turns(1)), 2);
        assert!(messages.iter().all(|m| m.reasoning_content.is_none()));
    }

    #[test]
    fn the_oldest_results_are_cleared_but_the_last_ones_and_those_of_excluded_tools() {
        let mut messages = vec![
            Message::new(Role::User, "q"),
            assistant(None, &[("a", "read"), ("b", "search")]),
            result("a"),
            result("b"),
            assistant(None, &[("c", "read")]),
            result("c"),
            assistant(None, &[("d", "read")]),
            result("d"),
        ];
        let found = calls(&messages);
        assert_eq!(found.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        // Keep the last two; `search` is never cleared.
        let cleared: Vec<usize> =
            (0..2).filter(|&i| !["search".to_owned()].contains(&found[i].name)).collect();
        clear_calls(&mut messages, &found, &cleared, true);
        assert_eq!(messages[2].content, CLEARED_RESULT);
        assert_eq!(messages[3].content, "a long result");
        assert_eq!(messages[5].content, "a long result");
        let first = &messages[1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(
            first.get("function").and_then(|f| f.get("arguments")),
            Some(&Json::Object(vec![]))
        );
        let second = &messages[1].tool_calls.as_ref().unwrap()[1];
        assert_ne!(
            second.get("function").and_then(|f| f.get("arguments")),
            Some(&Json::Object(vec![]))
        );
    }

    #[test]
    fn a_response_lists_what_was_applied() {
        assert_eq!(response_member(&[]), r#"{"applied_edits":[]}"#);
        let both = [
            Applied::Thinking { turns: 3, tokens: 150 },
            Applied::ToolUses { uses: 8, tokens: 5000 },
        ];
        let json: serde_json::Value = serde_json::from_str(&response_member(&both)).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"applied_edits": [
                {"type": "clear_thinking_20251015", "cleared_thinking_turns": 3, "cleared_input_tokens": 150},
                {"type": "clear_tool_uses_20250919", "cleared_tool_uses": 8, "cleared_input_tokens": 5000}
            ]})
        );
    }
}
