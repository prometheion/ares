//! Thinking in requests and replies.
//!
//! A model that thinks writes its reasoning between two tags before it
//! answers. This module is where the server deals with that, driven by what the
//! model's chat template declares ([`ThinkingFormat`], which the engine reads
//! off the template) and not by the name of a model:
//!
//! - [`asked`] reads the fields of a request that are about thinking, and
//!   [`apply`] turns them into variables of the chat template and into the
//!   [`Splitter`] that sorts the generated text;
//! - the [`Splitter`] puts the reasoning of a reply in `reasoning_content` and
//!   the rest in `content`, streamed or whole.
//!
//! # The controls
//!
//! They are those of llama-server, under its names:
//!
//! | control | where | here |
//! |---|---|---|
//! | `--reasoning on\|off\|auto` | start | `enable_thinking` as a chat variable |
//! | `--reasoning-format` | start; `reasoning_format` per request | [`ReasoningFormat`] |
//! | `--reasoning-effort` | start; `reasoning_effort` per request | `reasoning_effort` as a chat variable |
//! | `--reasoning-preserve` | start (on); `chat_template_kwargs` | `preserve_reasoning` as a chat variable |
//! | `chat_template_kwargs` | per request | chat variables |
//! | `--reasoning-budget`, `--reasoning-budget-message` | start; per request | refused: see below |
//!
//! llama-server enforces a budget of thinking tokens in its sampler, which
//! forces the tokens of a message and of the closing tag into the sequence once
//! the budget is spent. This engine has no way to put tokens into a sequence
//! that the model did not sample, so a request that asks for a budget, or for a
//! message to be injected, is refused by name and not answered without it.
//!
//! # The split
//!
//! The text the model generates is what follows the end of the prompt. When the
//! prompt ends inside the block (`<think>\n`), the generated text starts with
//! the reasoning; otherwise it starts with the content, or with the opening tag
//! if the model writes it itself. The reasoning ends at the closing tag. The
//! white space that the template writes around the reasoning is not part of
//! it: the template writes `<think>\n`, the reasoning, `\n</think>\n\n`, and
//! the content, so that is what is taken off again. llama-server leaves the
//! newline before the closing tag in `reasoning_content`; a template that does
//! not trim would then write it twice.

use ares_api::{Engine, GenerateRequest, Json, Prompt, Start, ThinkingFormat};

use crate::error::ApiError;
use crate::wire::{Fields, Parsed, wrong_type};

/// The members of a request body that are about thinking.
pub(crate) const FIELDS: &[&str] = &[
    "reasoning_format",
    "reasoning_effort",
    "chat_template_kwargs",
    "reasoning_budget_tokens",
    "thinking_budget_tokens",
    "reasoning_budget_message",
    "reasoning_control",
];

/// Variables the server sets itself; a request may not.
const RESERVED: &[&str] = &["messages", "tools", "add_generation_prompt", "bos_token", "eos_token"];

/// How the reasoning of a reply is returned (`--reasoning-format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningFormat {
    /// The raw text: the thinking stays in `content`, tags and all.
    None,
    /// The reasoning goes to `reasoning_content`, in streamed deltas too.
    #[default]
    Auto,
    /// The same as `Auto`.
    Deepseek,
    /// The same as `Auto`: llama-server accepts the name and parses the reply
    /// the same way (its description says the tags stay in `content` of a
    /// stream, its code does not do it).
    DeepseekLegacy,
}

impl ReasoningFormat {
    /// The format with this name, as llama-server spells them.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Self::None,
            "auto" => Self::Auto,
            "deepseek" => Self::Deepseek,
            "deepseek-legacy" => Self::DeepseekLegacy,
            _ => return None,
        })
    }

    /// Whether the reasoning is taken out of the text.
    fn extracts(self) -> bool {
        self != Self::None
    }
}

/// What a request says about thinking.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Asked {
    format: Option<ReasoningFormat>,
    /// `chat_template_kwargs`, in the order the client wrote them.
    variables: Vec<(String, Json)>,
    effort: Option<String>,
}

/// Reads the fields of a chat request that are about thinking, and refuses
/// the ones that ask for what the engine cannot do.
pub(crate) fn asked(fields: &Fields<'_>) -> Result<Asked, ApiError> {
    let format = match fields.string("reasoning_format")? {
        None => None,
        Some(name) => Some(ReasoningFormat::from_name(name).ok_or_else(|| {
            ApiError::invalid_param(
                "reasoning_format",
                format!(
                    "reasoning_format must be none, auto, deepseek or deepseek-legacy, got '{name}'"
                ),
            )
        })?),
    };
    let effort = fields.string("reasoning_effort")?.filter(|e| !e.is_empty()).map(str::to_owned);
    let variables = match fields.get("chat_template_kwargs") {
        None => Vec::new(),
        Some(Json::Object(members)) => members.clone(),
        Some(other) => return Err(wrong_type("chat_template_kwargs", "an object", other)),
    };
    for (name, value) in &variables {
        let at = format!("chat_template_kwargs.{name}");
        if RESERVED.contains(&name.as_str()) {
            return Err(ApiError::invalid_param(
                "chat_template_kwargs",
                format!("{at} is set by the server and cannot be a chat template variable"),
            ));
        }
        if name == "enable_thinking" && !matches!(value, Json::Bool(_) | Json::Null) {
            return Err(ApiError::invalid_param(
                "chat_template_kwargs",
                format!("{at} must be a boolean, got {}", value.kind()),
            ));
        }
    }
    refuse_budget(fields)?;
    Ok(Asked { format, variables, effort })
}

/// A budget of thinking tokens, and what goes with it, cannot be served.
fn refuse_budget(fields: &Fields<'_>) -> Result<(), ApiError> {
    let why = "a budget for thinking is not built: the engine cannot put the tokens that end the \
               thinking into a sequence, so -1 (no budget) is the only value";
    for name in ["reasoning_budget_tokens", "thinking_budget_tokens"] {
        match fields.get(name) {
            None | Some(Json::Int(-1)) => {}
            Some(Json::Float(f)) if *f == -1.0 => {}
            Some(Json::Int(_) | Json::Uint(_) | Json::Float(_)) => {
                return Err(ApiError::invalid_param(
                    name,
                    format!("{name} is not supported: {why}"),
                ));
            }
            Some(other) => return Err(wrong_type(name, "a number", other)),
        }
    }
    match fields.get("reasoning_budget_message") {
        None => {}
        Some(Json::String(text)) if text.is_empty() => {}
        Some(Json::String(_)) => {
            return Err(ApiError::invalid_param(
                "reasoning_budget_message",
                format!("reasoning_budget_message is not supported: {why}"),
            ));
        }
        Some(other) => return Err(wrong_type("reasoning_budget_message", "a string", other)),
    }
    match fields.get("reasoning_control") {
        None | Some(Json::Bool(false)) => Ok(()),
        Some(Json::Bool(true)) => Err(ApiError::invalid_param(
            "reasoning_control",
            format!("reasoning_control is not supported: {why}"),
        )),
        Some(other) => Err(wrong_type("reasoning_control", "a boolean", other)),
    }
}

/// What a request comes to for its reply.
#[derive(Debug)]
pub(crate) struct Applied {
    /// Sorts the text of the reply into reasoning and content.
    pub splitter: Splitter,
    /// Where the answer of the reply starts: where a rule for it binds.
    pub start: Start,
    /// Whether the reasoning stays in the content, tags and all.
    pub raw: bool,
}

/// Applies what `parsed` asks about thinking to its request and returns the
/// splitter for the reply and where its answer starts.
///
/// The variables go into the prompt of a chat request, on top of the ones the
/// engine was started with. `default` is the format the server was started
/// with (`--reasoning-format`).
pub(crate) fn apply(
    parsed: &mut Parsed,
    engine: &dyn Engine,
    default: ReasoningFormat,
) -> Result<Applied, ApiError> {
    let GenerateRequest { model, prompt, structure, .. } = &mut parsed.request;
    let Prompt::Chat { variables, .. } = prompt else {
        return Ok(Applied { splitter: Splitter::plain(), start: Start::Reply, raw: false });
    };
    let asked = std::mem::take(&mut parsed.reasoning);
    let format = engine.chat_format(model);
    let thinking = format.as_ref().and_then(|format| format.thinking.as_ref());
    let plan = plan(&asked, thinking, default)?;
    *variables = plan.variables;
    if let Some(structure) = structure {
        structure.start = plan.start.clone();
    }
    let raw = thinking.is_some() && !asked.format.unwrap_or(default).extracts();
    Ok(Applied { splitter: plan.splitter, start: plan.start, raw })
}

/// What a request comes to once the template's thinking is known.
#[derive(Debug)]
struct Plan {
    variables: Vec<(String, Json)>,
    splitter: Splitter,
    /// Where a grammar of the request starts to bind.
    start: Start,
}

fn plan(
    asked: &Asked,
    thinking: Option<&ThinkingFormat>,
    default: ReasoningFormat,
) -> Result<Plan, ApiError> {
    let mut variables = asked.variables.clone();
    let mut set = |name: &str, value: Json| {
        variables.retain(|(known, _)| known != name);
        variables.push((name.to_owned(), value));
    };
    // The request says whether to think, in the variable or in the effort.
    let mut asks_thinking = asked
        .variables
        .iter()
        .rev()
        .find_map(|(name, value)| (name == "enable_thinking").then_some(value))
        .and_then(|value| match value {
            Json::Bool(on) => Some(*on),
            _ => None,
        });
    match asked.effort.as_deref() {
        None => {}
        // OpenAI's way to say that the model should not think.
        Some("none") => {
            if let Some(format) = thinking
                && !format.switchable()
            {
                return Err(ApiError::invalid_param(
                    "reasoning_effort",
                    "reasoning_effort 'none' asks the model not to think, which the chat \
                     template of this model cannot switch off (it ignores enable_thinking)",
                ));
            }
            asks_thinking = Some(false);
        }
        Some(effort) => {
            if !thinking.is_some_and(|format| format.effort) {
                return Err(ApiError::invalid_param(
                    "reasoning_effort",
                    "reasoning_effort is not supported: the chat template of this model does \
                     not use it",
                ));
            }
            set("reasoning_effort", Json::String(effort.to_owned()));
        }
    }
    if let Some(on) = asks_thinking {
        set("enable_thinking", Json::Bool(on));
    }

    let format = asked.format.unwrap_or(default);
    let (splitter, start) = match thinking {
        None => (Splitter::plain(), Start::Reply),
        Some(thinking) => {
            let on = asks_thinking.unwrap_or(thinking.on_by_default);
            let prompt = if on { &thinking.prompt_on } else { &thinking.prompt_off };
            let splitter = if format.extracts() {
                Splitter::extracting(thinking, prompt)
            } else {
                Splitter::raw(prompt)
            };
            (splitter, answer_starts(thinking, prompt))
        }
    };
    Ok(Plan { variables, splitter, start })
}

/// Where the answer of a reply starts, for a prompt that ends with `prompt`
/// (from the opening tag on): llama-server binds a grammar to the text after
/// the reasoning, and so does this server, whatever the format the reasoning
/// is returned in.
///
/// - The prompt has closed the block (thinking is off): the reply is the
///   answer from its first token.
/// - The prompt has opened it: the reply is reasoning until the closing tag.
/// - The prompt wrote no tag: the reply may open the block itself, or answer.
fn answer_starts(thinking: &ThinkingFormat, prompt: &str) -> Start {
    let (open, close) = (thinking.start_tag(), thinking.end_tag());
    if close.is_empty() || (prompt.is_empty() && open.is_empty()) || prompt.contains(close) {
        return Start::Reply;
    }
    Start::AfterReasoning {
        open: prompt.is_empty().then(|| open.to_owned()),
        close: close.to_owned(),
    }
}

/// Text a splitter has sorted: the reasoning and the content of a piece.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Split {
    pub reasoning: String,
    pub content: String,
}

impl Split {
    pub fn is_empty(&self) -> bool {
        self.reasoning.is_empty() && self.content.is_empty()
    }
}

/// Sorts the text of a reply into reasoning and content as it comes.
///
/// [`push`](Splitter::push) takes the pieces in order and returns what can be
/// told so far; text that may still turn out to be part of a tag, or white
/// space the template writes around the reasoning, is held back until the
/// next piece decides it. [`finish`](Splitter::finish) returns what is left.
#[derive(Debug, Clone)]
pub(crate) struct Splitter {
    state: State,
    /// Text that has come and has not been sorted.
    buffer: String,
    layout: Layout,
    /// Whether the reasoning so far has been more than white space.
    started: bool,
    finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Everything is content.
    Plain,
    /// The raw text is returned as it is; the first piece is preceded by the
    /// end of the prompt.
    Raw,
    /// The prompt did not open the block: the reply may begin with the tag.
    Lead,
    Reasoning,
    /// The closing tag has come; the white space after it is to be skipped.
    AfterEnd,
    Content,
}

/// The tags and the white space the template writes around the reasoning.
#[derive(Debug, Default, Clone)]
struct Layout {
    start_tag: String,
    after_start: String,
    before_end: String,
    end_tag: String,
    after_end: String,
    /// What precedes the generated text in a raw reply.
    seed: String,
}

fn is_space(c: char) -> bool {
    c.is_ascii_whitespace()
}

impl Splitter {
    /// Everything is content: a request without thinking.
    pub fn plain() -> Self {
        Self::new(State::Plain, Layout::default())
    }

    /// The raw text, preceded by the end of the prompt from the tag on, which
    /// is what llama-server returns when it is not asked to extract.
    fn raw(prompt: &str) -> Self {
        Self::new(State::Raw, Layout { seed: prompt.to_owned(), ..Layout::default() })
    }

    /// Sorts a reply to a prompt that ends with `prompt` (from the opening tag
    /// on) into reasoning and content.
    fn extracting(format: &ThinkingFormat, prompt: &str) -> Self {
        let start = format.start.trim_start_matches(is_space);
        let end = format.end.as_str();
        let layout = Layout {
            start_tag: format.start_tag().to_owned(),
            after_start: start[start.trim_end_matches(is_space).len()..].to_owned(),
            before_end: end[..end.len() - end.trim_start_matches(is_space).len()].to_owned(),
            end_tag: format.end_tag().to_owned(),
            after_end: end[end.trim_end_matches(is_space).len()..].to_owned(),
            seed: String::new(),
        };
        let open = !prompt.is_empty() && !prompt.contains(&layout.end_tag);
        Self::new(if open { State::Reasoning } else { State::Lead }, layout)
    }

    fn new(state: State, layout: Layout) -> Self {
        Self { state, buffer: String::new(), layout, started: false, finished: false }
    }

    /// Takes the next piece of the reply.
    pub fn push(&mut self, text: &str) -> Split {
        self.buffer.push_str(text);
        self.sort()
    }

    /// Takes the end of the reply: what was held back is sorted as it is.
    pub fn finish(&mut self) -> Split {
        self.finished = true;
        self.sort()
    }

    fn sort(&mut self) -> Split {
        let mut split = Split::default();
        loop {
            match self.state {
                State::Plain | State::Content => {
                    split.content.push_str(&std::mem::take(&mut self.buffer));
                    return split;
                }
                State::Raw => {
                    split.content.push_str(&std::mem::take(&mut self.layout.seed));
                    split.content.push_str(&std::mem::take(&mut self.buffer));
                    return split;
                }
                State::Lead => {
                    if !self.lead() {
                        return split;
                    }
                }
                State::Reasoning => {
                    if !self.reasoning(&mut split) {
                        return split;
                    }
                }
                State::AfterEnd => {
                    let Some(skip) = skip(&self.buffer, &self.layout.after_end, self.finished)
                    else {
                        return split;
                    };
                    self.buffer.drain(..skip);
                    self.state = State::Content;
                }
            }
        }
    }

    /// The start of the reply: the tag, or the first text of the content.
    /// False when more text is needed to tell.
    fn lead(&mut self) -> bool {
        let tag = &self.layout.start_tag;
        if let Some(rest) = self.buffer.strip_prefix(tag.as_str()) {
            let Some(skip) = skip(rest, &self.layout.after_start, self.finished) else {
                return false;
            };
            self.buffer.drain(..tag.len() + skip);
            self.state = State::Reasoning;
            true
        } else if !self.finished && tag.starts_with(self.buffer.as_str()) {
            // Empty so far, or the start of the tag.
            false
        } else {
            self.state = State::Content;
            true
        }
    }

    /// The reasoning: up to the closing tag. False when more text is needed.
    fn reasoning(&mut self, split: &mut Split) -> bool {
        let (end_tag, before_end) = (&self.layout.end_tag, &self.layout.before_end);
        if let Some(at) = self.buffer.find(end_tag.as_str()) {
            // The white space the template writes before the tag is not the
            // reasoning's.
            let before = &self.buffer[..at];
            let reasoning = before.strip_suffix(before_end.as_str()).map_or(at, str::len);
            let tag_ends = at + end_tag.len() - reasoning;
            self.emit(split, reasoning, true);
            self.buffer.drain(..tag_ends);
            self.state = State::AfterEnd;
            return true;
        }
        // The end of the text may be the start of what ends the reasoning.
        let held = if self.finished {
            0
        } else {
            let with_space = format!("{before_end}{end_tag}");
            partial_suffix(&self.buffer, end_tag).max(partial_suffix(&self.buffer, &with_space))
        };
        let ready = self.buffer.len() - held;
        self.emit(split, ready, self.finished);
        false
    }

    /// Moves the first `len` bytes of the buffer to the reasoning of `split`,
    /// except white space that comes before any other reasoning: it is kept
    /// until a character that is not white space follows, or dropped when
    /// `last` says no more is coming. A reply that thinks of nothing has no
    /// reasoning.
    fn emit(&mut self, split: &mut Split, len: usize, last: bool) {
        let piece = &self.buffer[..len];
        if !self.started && piece.chars().all(is_space) {
            if last {
                self.buffer.drain(..len);
            }
            return;
        }
        self.started = true;
        split.reasoning.push_str(piece);
        self.buffer.drain(..len);
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of
/// `pattern`: what may still become `pattern`.
fn partial_suffix(text: &str, pattern: &str) -> usize {
    (1..pattern.len().min(text.len() + 1))
        .rev()
        .filter(|len| pattern.is_char_boundary(*len) && text.is_char_boundary(text.len() - len))
        .find(|len| text.ends_with(&pattern[..*len]))
        .unwrap_or(0)
}

/// How much of the start of `text` is the white space of `layout`, which
/// the template writes after a tag. Each character of the layout is matched
/// if it comes next and passed over if it does not, as an optional character
/// is. `None` when `text` ends before the layout is decided and more may come.
fn skip(text: &str, layout: &str, finished: bool) -> Option<usize> {
    let mut at = 0;
    for expected in layout.chars() {
        match text[at..].chars().next() {
            Some(c) if c == expected => at += c.len_utf8(),
            Some(_) => {}
            None if finished => break,
            None => return None,
        }
    }
    Some(at)
}

#[cfg(test)]
mod templates;
#[cfg(test)]
mod tests;
