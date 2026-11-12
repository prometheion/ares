//! The calls of tools in the text of a reply.
//!
//! [`CallSplitter`] takes the answer of a reply as it comes (what the
//! reasoning's splitter calls content) and sorts it into content and calls,
//! by the format the model's chat template writes calls in. The arguments of
//! a call go out as fragments of one JSON object, as OpenAI streams them: a
//! string of the format is written as a JSON string, any other value as the
//! JSON the model wrote.
//!
//! Text that may still turn out to be markup is held back until the next
//! piece decides it, so the pieces of a stream are the same text however the
//! tokens cut it, and no markup reaches the content. The white space between
//! the content and a call, and at the end of a reply that has calls, is the
//! template's layout and belongs to neither.
//!
//! A reply that is held to the grammar of calls has no other shape than the
//! one read here. Text that does not continue a call the way the format says
//! (from an engine that did not hold it) ends the call and is content.

use std::collections::HashMap;

use ares_api::ToolFormat;
use ares_grammar::calls::ValueTypes;
use serde_json::Value;

/// What a splitter found in a piece of the answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Part {
    Content(String),
    /// A call begins: its place among the calls of the reply, its id and the
    /// name of the tool.
    Call {
        index: usize,
        id: String,
        name: String,
    },
    /// The next fragment of the arguments of the call `index`.
    Arguments {
        index: usize,
        json: String,
    },
}

/// Sorts the answer of a reply into content and calls as it comes.
#[derive(Debug, Clone)]
pub(crate) struct CallSplitter {
    format: ToolFormat,
    /// The schema of the arguments of each tool, by its name.
    tools: HashMap<String, Option<Value>>,
    /// What the ids of the calls start with.
    id: String,
    state: State,
    /// Text that has come and has not been sorted.
    buffer: String,
    /// Text that comes before the answer and is passed on as content (the
    /// reasoning of a reply that is returned raw), up to this tag.
    gate: Option<String>,
    /// How many calls have begun.
    calls: usize,
    /// The schema of the tool being called.
    schema: Option<Value>,
    /// How many arguments the call has so far.
    arguments: usize,
    finished: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Content, until the trigger.
    Text,
    /// After the trigger: the rest of the opening and the name.
    Name,
    /// Between arguments: the next one, or the end of the call.
    Arguments,
    /// In a value that is written as a JSON string; `open` once its opening
    /// quote is out.
    String { open: bool },
    /// In a value that is JSON as the model writes it.
    Json,
    /// In a value that is a string or JSON: it is read whole, then told.
    Either(ValueTypes),
}

fn is_space(c: char) -> bool {
    c.is_ascii_whitespace()
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

/// `text` as the inside of a JSON string.
fn escaped(text: &str) -> String {
    let quoted = Value::String(text.to_owned()).to_string();
    quoted[1..quoted.len() - 1].to_owned()
}

impl CallSplitter {
    /// A splitter for replies that may call `tools` (the name and the schema
    /// of the arguments of each). `id` is what the ids of the calls start
    /// with. `gate` is a tag the answer starts after: the text up to it is
    /// passed on as content.
    pub fn new(
        format: ToolFormat,
        tools: HashMap<String, Option<Value>>,
        id: String,
        gate: Option<String>,
    ) -> Self {
        Self {
            format,
            tools,
            id,
            state: State::Text,
            buffer: String::new(),
            gate: gate.filter(|tag| !tag.is_empty()),
            calls: 0,
            schema: None,
            arguments: 0,
            finished: false,
        }
    }

    /// The same splitter for another reply, whose calls have ids that start
    /// with `id`.
    pub fn with_id(&self, id: String) -> Self {
        Self { id, ..self.clone() }
    }

    /// Takes the next piece of the answer.
    pub fn push(&mut self, text: &str) -> Vec<Part> {
        self.buffer.push_str(text);
        self.sort()
    }

    /// Takes the end of the answer: what was held back is sorted as it is. A
    /// call that the end cuts stays as far as it came.
    pub fn finish(&mut self) -> Vec<Part> {
        self.finished = true;
        self.sort()
    }

    fn sort(&mut self) -> Vec<Part> {
        let mut parts = Vec::new();
        while self.step(&mut parts) {}
        parts
    }

    fn content(parts: &mut Vec<Part>, text: &str) {
        if text.is_empty() {
            return;
        }
        match parts.last_mut() {
            Some(Part::Content(last)) => last.push_str(text),
            _ => parts.push(Part::Content(text.to_owned())),
        }
    }

    fn fragment(&self, parts: &mut Vec<Part>, json: &str) {
        if json.is_empty() {
            return;
        }
        let index = self.calls - 1;
        match parts.last_mut() {
            Some(Part::Arguments { index: last, json: text }) if *last == index => {
                text.push_str(json);
            }
            _ => parts.push(Part::Arguments { index, json: json.to_owned() }),
        }
    }

    /// Sorts what the buffer allows. False when more text is needed, or all
    /// is done.
    fn step(&mut self, parts: &mut Vec<Part>) -> bool {
        match self.state {
            State::Text => self.text(parts),
            State::Name => self.name(parts),
            State::Arguments => self.arguments(parts),
            State::String { open } => self.string(parts, open),
            State::Json => self.json(parts),
            State::Either(kinds) => self.either(parts, kinds),
        }
    }

    fn text(&mut self, parts: &mut Vec<Part>) -> bool {
        if let Some(gate) = &self.gate {
            // Before the answer: everything up to the tag is content.
            let Some(at) = self.buffer.find(gate.as_str()) else {
                let held = if self.finished { 0 } else { partial_suffix(&self.buffer, gate) };
                let ready = self.buffer.len() - held;
                Self::content(parts, &self.buffer[..ready]);
                self.buffer.drain(..ready);
                return false;
            };
            let end = at + gate.len();
            Self::content(parts, &self.buffer[..end]);
            self.buffer.drain(..end);
            self.gate = None;
        }
        let trigger = self.format.trigger();
        if let Some(at) = self.buffer.find(trigger) {
            // The white space before a call is the template's.
            Self::content(parts, self.buffer[..at].trim_end_matches(is_space));
            self.buffer.drain(..at + trigger.len());
            self.state = State::Name;
            return true;
        }
        if self.finished {
            // After a call nothing but its layout follows.
            let rest = std::mem::take(&mut self.buffer);
            if !(self.calls > 0 && rest.chars().all(is_space)) {
                Self::content(parts, &rest);
            }
            return false;
        }
        // The end may be the start of the trigger, after white space that
        // would then be the layout before a call.
        let held = partial_suffix(&self.buffer, trigger);
        let text = &self.buffer[..self.buffer.len() - held];
        let ready = text.trim_end_matches(is_space).len();
        Self::content(parts, &self.buffer[..ready]);
        self.buffer.drain(..ready);
        false
    }

    /// The buffer is not a call after all: `seen`, the markup read so far, is
    /// content, and so is what follows.
    fn not_a_call(&mut self, parts: &mut Vec<Part>, seen: &str) -> bool {
        Self::content(parts, seen);
        self.state = State::Text;
        true
    }

    /// Whether the buffer starts with `piece`: `None` when it is the start of
    /// `piece` and too short to tell.
    fn starts_with(&self, piece: &str) -> Option<bool> {
        if self.buffer.starts_with(piece) {
            Some(true)
        } else if piece.starts_with(self.buffer.as_str()) {
            None
        } else {
            Some(false)
        }
    }

    /// More text is needed. At the end of the reply there is none: the end
    /// cut the markup in the buffer, which is dropped.
    fn wait(&mut self) -> bool {
        if self.finished {
            self.buffer.clear();
        }
        false
    }

    fn name(&mut self, parts: &mut Vec<Part>) -> bool {
        let open = self.format.after_trigger().to_owned();
        let trigger = self.format.trigger().to_owned();
        match self.starts_with(&open) {
            None => return self.wait(),
            Some(false) => return self.not_a_call(parts, &trigger),
            Some(true) => {}
        }
        let Some(len) = self.buffer[open.len()..].find(&self.format.name_close) else {
            // At the end: the name is cut, and there is no call to return.
            return self.wait();
        };
        let name = self.buffer[open.len()..open.len() + len].to_owned();
        self.buffer.drain(..open.len() + len + self.format.name_close.len());
        self.schema = self.tools.get(&name).cloned().flatten();
        let index = self.calls;
        self.calls += 1;
        self.arguments = 0;
        parts.push(Part::Call { index, id: format!("{}{index}", self.id), name });
        self.fragment(parts, "{");
        self.state = State::Arguments;
        true
    }

    fn arguments(&mut self, parts: &mut Vec<Part>) -> bool {
        let (open, close) = (self.format.param_open.clone(), self.format.call_close.clone());
        match (self.starts_with(&close), self.starts_with(&open)) {
            (Some(true), _) => {
                self.buffer.drain(..close.len());
                self.fragment(parts, "}");
                self.state = State::Text;
                true
            }
            (_, Some(true)) => {
                let Some(len) = self.buffer[open.len()..].find(&self.format.key_close) else {
                    return self.wait();
                };
                let key = self.buffer[open.len()..open.len() + len].to_owned();
                self.buffer.drain(..open.len() + len + self.format.key_close.len());
                let separator = if self.arguments > 0 { "," } else { "" };
                self.arguments += 1;
                self.fragment(parts, &format!("{separator}{}:", Value::String(key.clone())));
                let kinds = ValueTypes::of(self.schema.as_ref(), &key);
                self.state = if kinds.only_string() {
                    State::String { open: false }
                } else if kinds.has_string() {
                    State::Either(kinds)
                } else {
                    State::Json
                };
                true
            }
            (Some(false), Some(false)) => {
                // Neither an argument nor the end of the call: the call ends
                // here, and what follows is content.
                self.fragment(parts, "}");
                self.state = State::Text;
                true
            }
            _ => self.wait(),
        }
    }

    /// The value up to the tag that closes it, as far as it has come: the
    /// text that is ready and whether the tag was reached.
    fn value(&mut self) -> (String, bool) {
        let close = &self.format.param_close;
        if let Some(at) = self.buffer.find(close.as_str()) {
            let text = self.buffer[..at].to_owned();
            self.buffer.drain(..at + close.len());
            return (text, true);
        }
        // The end of the text may be the start of the tag. At the end of the
        // reply it is taken for that: the limit cut the tag.
        let held = partial_suffix(&self.buffer, close);
        let ready = self.buffer.len() - held;
        let text = self.buffer[..ready].to_owned();
        self.buffer.drain(..ready);
        if self.finished {
            self.buffer.clear();
        }
        (text, false)
    }

    fn string(&mut self, parts: &mut Vec<Part>, open: bool) -> bool {
        let (text, closed) = self.value();
        let mut json = String::new();
        if !open && (closed || !text.is_empty()) {
            json.push('"');
        }
        json.push_str(&escaped(&text));
        if closed {
            json.push('"');
        }
        self.fragment(parts, &json);
        self.state = if closed {
            State::Arguments
        } else {
            State::String { open: open || !text.is_empty() }
        };
        closed
    }

    fn json(&mut self, parts: &mut Vec<Part>) -> bool {
        let (text, closed) = self.value();
        self.fragment(parts, &text);
        if closed {
            self.state = State::Arguments;
        }
        closed
    }

    fn either(&mut self, parts: &mut Vec<Part>, kinds: ValueTypes) -> bool {
        let close = self.format.param_close.clone();
        let Some(at) = self.buffer.find(&close) else {
            if self.finished {
                // Cut inside the value: it cannot be told, and stays a string
                // that does not close.
                let (text, _) = self.value();
                if !text.is_empty() {
                    self.fragment(parts, &format!("\"{}", escaped(&text)));
                }
            }
            return false;
        };
        let text = self.buffer[..at].to_owned();
        self.buffer.drain(..at + close.len());
        // JSON of a kind the schema allows is that value; anything else is
        // the string it is written as.
        let json = match serde_json::from_str::<Value>(&text) {
            Ok(value) if !value.is_string() && kinds.allows(&value) => text,
            _ => Value::String(text).to_string(),
        };
        self.fragment(parts, &json);
        self.state = State::Arguments;
        true
    }
}
