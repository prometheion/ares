//! A reply as Anthropic's API writes it: the blocks of content of a message,
//! and the events of a stream.
//!
//! Both are made from the same thing. [`Blocks`] turns the pieces of a reply
//! ([`Piece`]) into the events of its blocks, in the order the model wrote
//! them: a block opens when a piece of another kind comes, gets its deltas,
//! and closes when the next one opens or the reply ends. A stream writes the
//! events out; a message that is returned whole is the same events put
//! together ([`Content`]), so the two cannot differ.

use ares_api::Usage;
use serde::Serialize;

use crate::reply::Piece;

/// What every thinking block of a reply carries as its signature.
///
/// Anthropic signs a thinking block so that it can tell its own thinking when
/// a client sends it back. This server has no secret to sign with and no use
/// for a proof: thinking that comes back is text for the prompt, whoever
/// wrote it. So the signature says that there is none, in a text that is not
/// empty, because a client may drop a block whose signature is missing.
pub(crate) const SIGNATURE: &str = "unsigned";

/// An event of the blocks of a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BlockEvent {
    Thinking {
        index: usize,
    },
    Text {
        index: usize,
    },
    ToolUse {
        index: usize,
        id: String,
        name: String,
    },
    ThinkingDelta {
        index: usize,
        text: String,
    },
    TextDelta {
        index: usize,
        text: String,
    },
    /// A fragment of the JSON text of a call's input.
    InputDelta {
        index: usize,
        json: String,
    },
    /// The signature of a thinking block, before it closes.
    Signature {
        index: usize,
    },
    Stop {
        index: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Thinking,
    Text,
    ToolUse,
}

/// Turns the pieces of a reply into the events of its blocks.
#[derive(Debug, Default)]
pub(crate) struct Blocks {
    /// The block that is open.
    open: Option<Kind>,
    /// How many blocks have begun.
    begun: usize,
    /// The block of each call of the reply.
    calls: Vec<usize>,
}

impl Blocks {
    /// The events of the next pieces.
    pub fn push(&mut self, pieces: Vec<Piece>) -> Vec<BlockEvent> {
        let mut events = Vec::new();
        for piece in pieces {
            match piece {
                Piece::Text(split) => {
                    if !split.reasoning.is_empty() {
                        let index = self.open(Kind::Thinking, &mut events);
                        events.push(BlockEvent::ThinkingDelta { index, text: split.reasoning });
                    }
                    if !split.content.is_empty() {
                        let index = self.open(Kind::Text, &mut events);
                        events.push(BlockEvent::TextDelta { index, text: split.content });
                    }
                }
                Piece::Call { id, name, .. } => {
                    self.close(&mut events);
                    let index = self.begin(Kind::ToolUse);
                    self.calls.push(index);
                    events.push(BlockEvent::ToolUse { index, id, name });
                }
                Piece::Arguments { index, json } => {
                    if let Some(&index) = self.calls.get(index)
                        && !json.is_empty()
                    {
                        events.push(BlockEvent::InputDelta { index, json });
                    }
                }
            }
        }
        events
    }

    /// The events of the end of the reply: the open block closes.
    pub fn finish(&mut self) -> Vec<BlockEvent> {
        let mut events = Vec::new();
        self.close(&mut events);
        events
    }

    /// How many calls the reply has begun.
    pub fn calls(&self) -> usize {
        self.calls.len()
    }

    /// The index of the open block of `kind`, which is opened if another
    /// kind, or none, is open.
    fn open(&mut self, kind: Kind, events: &mut Vec<BlockEvent>) -> usize {
        if self.open != Some(kind) {
            self.close(events);
            let index = self.begin(kind);
            events.push(match kind {
                Kind::Thinking => BlockEvent::Thinking { index },
                _ => BlockEvent::Text { index },
            });
        }
        self.begun - 1
    }

    fn begin(&mut self, kind: Kind) -> usize {
        self.open = Some(kind);
        self.begun += 1;
        self.begun - 1
    }

    fn close(&mut self, events: &mut Vec<BlockEvent>) {
        let Some(kind) = self.open.take() else {
            return;
        };
        let index = self.begun - 1;
        if kind == Kind::Thinking {
            events.push(BlockEvent::Signature { index });
        }
        events.push(BlockEvent::Stop { index });
    }
}

/// A block of content of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Block {
    Thinking(String),
    Text(String),
    /// A call of a tool; `input` is the JSON text of its arguments as far as
    /// the model wrote it.
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

/// The content of a message, put together from the events of its blocks.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Content(pub Vec<Block>);

impl Content {
    pub fn take(&mut self, events: Vec<BlockEvent>) {
        for event in events {
            match event {
                BlockEvent::Thinking { .. } => self.0.push(Block::Thinking(String::new())),
                BlockEvent::Text { .. } => self.0.push(Block::Text(String::new())),
                BlockEvent::ToolUse { id, name, .. } => {
                    self.0.push(Block::ToolUse { id, name, input: String::new() });
                }
                BlockEvent::ThinkingDelta { index, text: more } => {
                    if let Some(Block::Thinking(text)) = self.0.get_mut(index) {
                        text.push_str(&more);
                    }
                }
                BlockEvent::TextDelta { index, text: more } => {
                    if let Some(Block::Text(text)) = self.0.get_mut(index) {
                        text.push_str(&more);
                    }
                }
                BlockEvent::InputDelta { index, json } => {
                    if let Some(Block::ToolUse { input, .. }) = self.0.get_mut(index) {
                        input.push_str(&json);
                    }
                }
                BlockEvent::Signature { .. } | BlockEvent::Stop { .. } => {}
            }
        }
    }
}

/// Why a reply ended, in the words of this API. `finish` is the engine's
/// word and `calls` the number of calls the reply has.
///
/// The engine has one word for a reply the model ended and for one a stop
/// string ended, so `stop_sequence` is never the reason here.
pub(crate) fn stop_reason(finish: &str, calls: usize) -> &'static str {
    match finish {
        "length" => "max_tokens",
        _ if calls > 0 => "tool_use",
        _ => "end_turn",
    }
}

/// The fields every object of one reply repeats.
#[derive(Debug, Clone)]
pub(crate) struct Head {
    pub id: String,
    pub model: String,
}

/// The token counts of a reply.
///
/// `input_tokens` are the tokens of the prompt that were computed and
/// `cache_read_input_tokens` the ones the prefix cache gave, as at Anthropic,
/// where the two add up to the prompt. The server writes no cache at a
/// client's marker (it keeps what it computes either way), so the count of
/// tokens written at markers is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct UsageBody {
    input_tokens: u32,
    cache_creation_input_tokens: u32,
    cache_read_input_tokens: u32,
    output_tokens: u32,
}

impl From<Usage> for UsageBody {
    fn from(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens.saturating_sub(usage.cached_prompt_tokens),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: usage.cached_prompt_tokens,
            output_tokens: usage.completion_tokens,
        }
    }
}

fn quoted(text: &str) -> String {
    serde_json::to_string(text).expect("a string serializes")
}

fn to_json(value: &impl Serialize) -> String {
    serde_json::to_string(value).expect("response objects serialize")
}

/// The input of a call as the JSON object it is: the text the model wrote,
/// byte for byte, so that a client that sends the call back sends what was
/// generated. A call the limit of tokens cut has no whole object, and its
/// input is the empty one.
fn input_object(text: &str) -> &str {
    match serde_json::from_str::<ares_api::Json>(text) {
        Ok(ares_api::Json::Object(_)) => text,
        _ => "{}",
    }
}

impl Head {
    /// The message of a reply that is returned whole.
    pub fn message(&self, content: &Content, stop_reason: &str, usage: Usage) -> String {
        let blocks: Vec<String> = content
            .0
            .iter()
            .map(|block| match block {
                Block::Thinking(text) => format!(
                    r#"{{"type":"thinking","thinking":{},"signature":{}}}"#,
                    quoted(text),
                    quoted(SIGNATURE)
                ),
                Block::Text(text) => format!(r#"{{"type":"text","text":{}}}"#, quoted(text)),
                Block::ToolUse { id, name, input } => format!(
                    r#"{{"type":"tool_use","id":{},"name":{},"input":{}}}"#,
                    quoted(id),
                    quoted(name),
                    input_object(input)
                ),
            })
            .collect();
        format!(
            r#"{{"id":{},"type":"message","role":"assistant","model":{},"content":[{}],"stop_reason":{},"stop_sequence":null,"usage":{}}}"#,
            quoted(&self.id),
            quoted(&self.model),
            blocks.join(","),
            quoted(stop_reason),
            to_json(&UsageBody::from(usage))
        )
    }

    /// The first event of a stream: the message, empty. The engine counts
    /// the prompt when the reply ends, so the usage here says nothing yet and
    /// the last `message_delta` has all of it.
    pub fn message_start(&self) -> (&'static str, String) {
        let data = format!(
            r#"{{"type":"message_start","message":{{"id":{},"type":"message","role":"assistant","model":{},"content":[],"stop_reason":null,"stop_sequence":null,"usage":{}}}}}"#,
            quoted(&self.id),
            quoted(&self.model),
            to_json(&UsageBody::from(Usage::default()))
        );
        ("message_start", data)
    }
}

/// An event of a stream as it is written: the name and the data.
pub(crate) type Written = (&'static str, String);

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum StartBody<'a> {
    Thinking { thinking: &'a str, signature: &'a str },
    Text { text: &'a str },
    ToolUse { id: &'a str, name: &'a str, input: Empty },
}

/// `{}`.
#[derive(Serialize)]
struct Empty {}

#[derive(Serialize)]
#[serde(tag = "type")]
enum DeltaBody<'a> {
    #[serde(rename = "thinking_delta")]
    Thinking { thinking: &'a str },
    #[serde(rename = "text_delta")]
    Text { text: &'a str },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: &'a str },
    #[serde(rename = "signature_delta")]
    Signature { signature: &'a str },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EventBody<'a> {
    ContentBlockStart { index: usize, content_block: StartBody<'a> },
    ContentBlockDelta { index: usize, delta: DeltaBody<'a> },
    ContentBlockStop { index: usize },
    MessageDelta { delta: Ending<'a>, usage: UsageBody },
    MessageStop {},
}

#[derive(Serialize)]
struct Ending<'a> {
    stop_reason: &'a str,
    stop_sequence: Option<&'a str>,
}

/// An event of a block as a stream writes it.
pub(crate) fn block_event(event: &BlockEvent) -> Written {
    fn start(index: usize, content_block: StartBody<'_>) -> (&'static str, EventBody<'_>) {
        ("content_block_start", EventBody::ContentBlockStart { index, content_block })
    }
    fn delta(index: usize, delta: DeltaBody<'_>) -> (&'static str, EventBody<'_>) {
        ("content_block_delta", EventBody::ContentBlockDelta { index, delta })
    }
    let (name, body) = match event {
        BlockEvent::Thinking { index } => {
            start(*index, StartBody::Thinking { thinking: "", signature: "" })
        }
        BlockEvent::Text { index } => start(*index, StartBody::Text { text: "" }),
        BlockEvent::ToolUse { index, id, name } => {
            start(*index, StartBody::ToolUse { id, name, input: Empty {} })
        }
        BlockEvent::ThinkingDelta { index, text } => {
            delta(*index, DeltaBody::Thinking { thinking: text })
        }
        BlockEvent::TextDelta { index, text } => delta(*index, DeltaBody::Text { text }),
        BlockEvent::InputDelta { index, json } => {
            delta(*index, DeltaBody::InputJson { partial_json: json })
        }
        BlockEvent::Signature { index } => {
            delta(*index, DeltaBody::Signature { signature: SIGNATURE })
        }
        BlockEvent::Stop { index } => {
            ("content_block_stop", EventBody::ContentBlockStop { index: *index })
        }
    };
    (name, to_json(&body))
}

/// The two events that end a stream: why the reply ended and what it cost,
/// and the end of the message.
pub(crate) fn message_end(stop_reason: &str, usage: Usage) -> [Written; 2] {
    let delta = EventBody::MessageDelta {
        delta: Ending { stop_reason, stop_sequence: None },
        usage: usage.into(),
    };
    [("message_delta", to_json(&delta)), ("message_stop", to_json(&EventBody::MessageStop {}))]
}

/// The data of the `ping` event a stream sends while the engine has nothing
/// to say.
pub(crate) const PING: &str = r#"{"type":"ping"}"#;
