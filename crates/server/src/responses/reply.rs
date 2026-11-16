//! A reply as the Responses API writes it: the output items of a response,
//! the response object, and the events of a stream.
//!
//! All of it is made from the same thing. [`Output`] takes the pieces of a
//! reply ([`Piece`]) and keeps the items they make, in the order the model
//! wrote them: an item opens when a piece of another kind comes, grows, and
//! closes when the next one opens or the reply ends. Every step is a
//! [`Change`]. A stream writes the changes out as events and ends with the
//! response object; a response that is returned whole is that same object,
//! so the two cannot differ.

use std::collections::VecDeque;

use ares_api::{Json, Usage};
use serde::{Serialize, Serializer};

use super::request::Echo;
use crate::bridge::Timings;
use crate::error::ApiError;
use crate::reply::Piece;
use crate::wire::TimingsBody;

/// `[]`: a list this server never has anything for (the summary of a
/// reasoning item, the annotations and the probabilities of a text).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Nothing;

impl Serialize for Nothing {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::empty::<()>())
    }
}

/// Where an item stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    InProgress,
    Completed,
    /// The limit of tokens cut the reply inside the item.
    Incomplete,
}

/// A part of the content of an item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Part {
    /// The reasoning itself, as the model wrote it.
    ReasoningText {
        text: String,
    },
    OutputText {
        text: String,
        annotations: Nothing,
        logprobs: Nothing,
    },
}

impl Part {
    fn text(&self) -> &str {
        match self {
            Part::ReasoningText { text } | Part::OutputText { text, .. } => text,
        }
    }

    fn text_mut(&mut self) -> &mut String {
        match self {
            Part::ReasoningText { text } | Part::OutputText { text, .. } => text,
        }
    }
}

/// An item of the output of a response.
///
/// A reasoning item has the reasoning in `content`, in the clear, and no
/// summary: the server writes none. Sent back, the item is the reasoning of
/// its turn again (see the `request` module).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Item {
    Reasoning {
        id: String,
        summary: Nothing,
        content: Vec<Part>,
        status: Status,
    },
    Message {
        id: String,
        role: &'static str,
        status: Status,
        content: Vec<Part>,
    },
    /// A call of a tool; `arguments` is the JSON text of its arguments as
    /// far as the model wrote it.
    FunctionCall {
        id: String,
        status: Status,
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl Item {
    fn id(&self) -> &str {
        match self {
            Item::Reasoning { id, .. }
            | Item::Message { id, .. }
            | Item::FunctionCall { id, .. } => id,
        }
    }

    fn status_mut(&mut self) -> &mut Status {
        match self {
            Item::Reasoning { status, .. }
            | Item::Message { status, .. }
            | Item::FunctionCall { status, .. } => status,
        }
    }

    /// The one part of the content of an item that has text.
    fn part(&self) -> Option<&Part> {
        match self {
            Item::Reasoning { content, .. } | Item::Message { content, .. } => content.first(),
            Item::FunctionCall { .. } => None,
        }
    }

    /// The item as it was when it opened: in progress and empty.
    fn opened(&self) -> Item {
        let status = Status::InProgress;
        match self {
            Item::Reasoning { id, .. } => {
                Item::Reasoning { id: id.clone(), summary: Nothing, content: Vec::new(), status }
            }
            Item::Message { id, role, .. } => {
                Item::Message { id: id.clone(), role, status, content: Vec::new() }
            }
            Item::FunctionCall { id, call_id, name, .. } => Item::FunctionCall {
                id: id.clone(),
                status,
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: String::new(),
            },
        }
    }
}

/// A step of the output of a reply. The numbers are places in the output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Change {
    /// The item opened.
    Added(usize),
    /// More text of a reasoning item or of a message.
    Text { index: usize, delta: String },
    /// A fragment of the JSON text of a call's arguments.
    Arguments { index: usize, delta: String },
    /// The item closed.
    Done(usize),
}

/// The output of a reply: its items so far.
#[derive(Debug)]
pub(crate) struct Output {
    /// The id of the request, which the ids of the items carry.
    request: String,
    items: Vec<Item>,
    /// Whether the last item is open.
    open: bool,
    /// The item of each call of the reply.
    calls: Vec<usize>,
}

impl Output {
    pub fn new(request: String) -> Self {
        Self { request, items: Vec::new(), open: false, calls: Vec::new() }
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// Takes the next pieces of the reply.
    pub fn push(&mut self, pieces: Vec<Piece>) -> Vec<Change> {
        let mut changes = Vec::new();
        for piece in pieces {
            match piece {
                Piece::Text(split) => {
                    if !split.reasoning.is_empty() {
                        self.text(true, split.reasoning, &mut changes);
                    }
                    if !split.content.is_empty() {
                        self.text(false, split.content, &mut changes);
                    }
                }
                Piece::Call { id, name, .. } => {
                    self.close(Status::Completed, &mut changes);
                    let index = self.items.len();
                    self.calls.push(index);
                    self.items.push(Item::FunctionCall {
                        id: format!("fc_{}_{index}", self.request),
                        status: Status::InProgress,
                        call_id: id,
                        name,
                        arguments: String::new(),
                    });
                    self.open = true;
                    changes.push(Change::Added(index));
                }
                Piece::Arguments { index, json } => {
                    // The arguments of a call come before the next item opens.
                    if let Some(&index) = self.calls.get(index)
                        && self.open
                        && index + 1 == self.items.len()
                        && !json.is_empty()
                        && let Item::FunctionCall { arguments, .. } = &mut self.items[index]
                    {
                        arguments.push_str(&json);
                        changes.push(Change::Arguments { index, delta: json });
                    }
                }
            }
        }
        changes
    }

    /// Takes the end of the reply: the open item closes, as incomplete when
    /// the limit of tokens `cut` the reply.
    pub fn finish(&mut self, cut: bool) -> Vec<Change> {
        let mut changes = Vec::new();
        self.close(if cut { Status::Incomplete } else { Status::Completed }, &mut changes);
        changes
    }

    /// Appends text to the open reasoning item (or message), which is
    /// opened if an item of another kind, or none, is open.
    fn text(&mut self, reasoning: bool, delta: String, changes: &mut Vec<Change>) {
        let continues = self.open
            && match self.items.last() {
                Some(Item::Reasoning { .. }) => reasoning,
                Some(Item::Message { .. }) => !reasoning,
                _ => false,
            };
        if !continues {
            self.close(Status::Completed, changes);
            let index = self.items.len();
            let status = Status::InProgress;
            self.items.push(if reasoning {
                Item::Reasoning {
                    id: format!("rs_{}_{index}", self.request),
                    summary: Nothing,
                    content: vec![Part::ReasoningText { text: String::new() }],
                    status,
                }
            } else {
                let part = Part::OutputText {
                    text: String::new(),
                    annotations: Nothing,
                    logprobs: Nothing,
                };
                Item::Message {
                    id: format!("msg_{}_{index}", self.request),
                    role: "assistant",
                    status,
                    content: vec![part],
                }
            });
            self.open = true;
            changes.push(Change::Added(index));
        }
        let index = self.items.len() - 1;
        if let Item::Reasoning { content, .. } | Item::Message { content, .. } =
            &mut self.items[index]
        {
            content[0].text_mut().push_str(&delta);
        }
        changes.push(Change::Text { index, delta });
    }

    fn close(&mut self, status: Status, changes: &mut Vec<Change>) {
        if !self.open {
            return;
        }
        self.open = false;
        let index = self.items.len() - 1;
        *self.items[index].status_mut() = status;
        changes.push(Change::Done(index));
    }
}

/// How a reply ended.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ending {
    /// `stop` or `length`.
    pub finish: &'static str,
    pub usage: Usage,
    pub timings: Timings,
}

impl Ending {
    /// Whether the limit of tokens cut the reply.
    pub fn cut(&self) -> bool {
        self.finish == "length"
    }
}

/// The token counts of a response. `input_tokens` are all tokens of the
/// prompt and `cached_tokens` the ones of them the prefix cache gave, as at
/// OpenAI. `output_tokens_details.reasoning_tokens` are the tokens of the
/// thinking, for a model that thinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct UsageBody {
    input_tokens: u32,
    input_tokens_details: InputDetails,
    output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens_details: Option<OutputDetails>,
    total_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct OutputDetails {
    reasoning_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
struct InputDetails {
    cached_tokens: u32,
}

impl From<Usage> for UsageBody {
    fn from(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            input_tokens_details: InputDetails { cached_tokens: usage.cached_prompt_tokens },
            output_tokens: usage.completion_tokens,
            output_tokens_details: usage
                .reasoning_tokens
                .map(|reasoning_tokens| OutputDetails { reasoning_tokens }),
            total_tokens: u64::from(usage.prompt_tokens) + u64::from(usage.completion_tokens),
        }
    }
}

#[derive(Serialize)]
struct IncompleteDetails {
    reason: &'static str,
}

#[derive(Serialize)]
struct ReasoningEcho<'a> {
    effort: &'a Json,
    /// The server writes no summaries.
    summary: Option<()>,
}

#[derive(Serialize)]
struct TextEcho<'a> {
    format: &'a Json,
}

/// The response object.
#[derive(Serialize)]
struct ResponseBody<'a> {
    id: &'a str,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    background: bool,
    error: Option<()>,
    incomplete_details: Option<IncompleteDetails>,
    instructions: &'a Json,
    max_output_tokens: &'a Json,
    model: &'a str,
    output: &'a [Item],
    parallel_tool_calls: bool,
    previous_response_id: Option<()>,
    reasoning: ReasoningEcho<'a>,
    /// This route keeps no responses.
    store: bool,
    temperature: &'a Json,
    text: TextEcho<'a>,
    tool_choice: &'a Json,
    tools: &'a Json,
    top_p: &'a Json,
    truncation: &'static str,
    usage: Option<UsageBody>,
    user: &'a Json,
    metadata: &'a Json,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<TimingsBody>,
}

/// An event of a stream as it is written: the name and the data.
pub(crate) type Written = (&'static str, String);

/// What an event says, next to its type and its number.
#[derive(Serialize)]
#[serde(untagged)]
enum Body<'a> {
    Response {
        response: &'a ResponseBody<'a>,
    },
    Item {
        output_index: usize,
        item: &'a Item,
    },
    Part {
        item_id: &'a str,
        output_index: usize,
        content_index: usize,
        part: &'a Part,
    },
    ReasoningDelta {
        item_id: &'a str,
        output_index: usize,
        content_index: usize,
        delta: &'a str,
    },
    ReasoningDone {
        item_id: &'a str,
        output_index: usize,
        content_index: usize,
        text: &'a str,
    },
    TextDelta {
        item_id: &'a str,
        output_index: usize,
        content_index: usize,
        delta: &'a str,
        logprobs: Nothing,
    },
    TextDone {
        item_id: &'a str,
        output_index: usize,
        content_index: usize,
        text: &'a str,
        logprobs: Nothing,
    },
    ArgumentsDelta {
        item_id: &'a str,
        output_index: usize,
        delta: &'a str,
    },
    ArgumentsDone {
        item_id: &'a str,
        output_index: usize,
        name: &'a str,
        arguments: &'a str,
    },
    Error {
        code: &'a str,
        message: &'a str,
        param: Option<&'a str>,
    },
}

#[derive(Serialize)]
struct Event<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    #[serde(flatten)]
    body: Body<'a>,
}

fn to_json(value: &impl Serialize) -> String {
    serde_json::to_string(value).expect("response objects serialize")
}

/// Writes the event `kind` with the next number of the stream.
fn written(kind: &'static str, sequence: &mut u64, body: Body<'_>) -> Written {
    let data = to_json(&Event { kind, sequence_number: *sequence, body });
    *sequence += 1;
    (kind, data)
}

/// The fields every object of one response repeats.
#[derive(Debug, Clone)]
pub(crate) struct Head {
    /// `resp_` and the id of the request.
    pub id: String,
    pub model: String,
    pub created_at: u64,
    pub echo: Echo,
}

impl Head {
    fn body<'a>(&'a self, output: &'a [Item], ending: Option<&Ending>) -> ResponseBody<'a> {
        let echo = &self.echo;
        let status = match ending {
            None => "in_progress",
            Some(ending) if ending.cut() => "incomplete",
            Some(_) => "completed",
        };
        let cut = ending.is_some_and(Ending::cut);
        ResponseBody {
            id: &self.id,
            object: "response",
            created_at: self.created_at,
            status,
            background: false,
            error: None,
            incomplete_details: cut.then_some(IncompleteDetails { reason: "max_output_tokens" }),
            instructions: &echo.instructions,
            max_output_tokens: &echo.max_output_tokens,
            model: &self.model,
            output,
            parallel_tool_calls: echo.parallel_tool_calls,
            previous_response_id: None,
            reasoning: ReasoningEcho { effort: &echo.effort, summary: None },
            store: false,
            temperature: &echo.temperature,
            text: TextEcho { format: &echo.format },
            tool_choice: &echo.tool_choice,
            tools: &echo.tools,
            top_p: &echo.top_p,
            truncation: "disabled",
            usage: ending.map(|ending| ending.usage.into()),
            user: &echo.user,
            metadata: &echo.metadata,
            timings: ending.map(|ending| ending.timings.into()),
        }
    }

    /// The response of a reply that has ended.
    pub fn response(&self, output: &[Item], ending: &Ending) -> String {
        to_json(&self.body(output, Some(ending)))
    }

    /// The first two events of a stream: the response, in progress and
    /// empty. The engine counts the prompt when the reply ends, so there is
    /// no usage yet.
    pub fn begin(&self, sequence: &mut u64) -> [Written; 2] {
        let response = &self.body(&[], None);
        ["response.created", "response.in_progress"]
            .map(|kind| written(kind, sequence, Body::Response { response }))
    }

    /// The last event of a stream: the response as it is returned whole,
    /// under the name its status gives it.
    pub fn end(&self, output: &[Item], ending: &Ending, sequence: &mut u64) -> Written {
        let kind = if ending.cut() { "response.incomplete" } else { "response.completed" };
        let response = &self.body(output, Some(ending));
        written(kind, sequence, Body::Response { response })
    }
}

/// Writes a change of `output` as the events of a stream, onto `events`.
///
/// `output` is the output after the change, and possibly after later ones.
/// An item that opened is written as it was then, and an item that closed
/// does not change any more.
pub(crate) fn change_events(
    output: &Output,
    change: &Change,
    sequence: &mut u64,
    events: &mut VecDeque<Written>,
) {
    let (index, content_index) = match change {
        Change::Added(index) | Change::Done(index) => (*index, 0),
        Change::Text { index, .. } | Change::Arguments { index, .. } => (*index, 0),
    };
    let item = &output.items()[index];
    let (item_id, output_index) = (item.id(), index);
    let mut write = |kind, body| events.push_back(written(kind, sequence, body));
    match (change, item) {
        (Change::Added(_), _) => {
            let opened = item.opened();
            write("response.output_item.added", Body::Item { output_index, item: &opened });
            let part = match item {
                Item::Reasoning { .. } => Part::ReasoningText { text: String::new() },
                Item::Message { .. } => Part::OutputText {
                    text: String::new(),
                    annotations: Nothing,
                    logprobs: Nothing,
                },
                Item::FunctionCall { .. } => return,
            };
            let body = Body::Part { item_id, output_index, content_index, part: &part };
            write("response.content_part.added", body);
        }
        (Change::Text { delta, .. }, Item::Reasoning { .. }) => write(
            "response.reasoning_text.delta",
            Body::ReasoningDelta { item_id, output_index, content_index, delta },
        ),
        (Change::Text { delta, .. }, _) => write(
            "response.output_text.delta",
            Body::TextDelta { item_id, output_index, content_index, delta, logprobs: Nothing },
        ),
        (Change::Arguments { delta, .. }, _) => write(
            "response.function_call_arguments.delta",
            Body::ArgumentsDelta { item_id, output_index, delta },
        ),
        (Change::Done(_), Item::FunctionCall { name, arguments, .. }) => {
            write(
                "response.function_call_arguments.done",
                Body::ArgumentsDone { item_id, output_index, name, arguments },
            );
            write("response.output_item.done", Body::Item { output_index, item });
        }
        (Change::Done(_), _) => {
            if let Some(part) = item.part() {
                let text = part.text();
                match part {
                    Part::ReasoningText { .. } => write(
                        "response.reasoning_text.done",
                        Body::ReasoningDone { item_id, output_index, content_index, text },
                    ),
                    Part::OutputText { .. } => write(
                        "response.output_text.done",
                        Body::TextDone {
                            item_id,
                            output_index,
                            content_index,
                            text,
                            logprobs: Nothing,
                        },
                    ),
                }
                let body = Body::Part { item_id, output_index, content_index, part };
                write("response.content_part.done", body);
            }
            write("response.output_item.done", Body::Item { output_index, item });
        }
    }
}

/// The event of a failure after a stream began. The stream ends with it.
pub(crate) fn error_event(error: &ApiError, sequence: &mut u64) -> Written {
    let body =
        Body::Error { code: error.code, message: &error.message, param: error.param.as_deref() };
    written("error", sequence, body)
}
