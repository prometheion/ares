//! How the Messages and the Responses routes answer a reply: whole, as one
//! JSON body, or streamed, as server-sent events. The two APIs differ only
//! in what they write ([`Writer`]); waiting for the engine, the order of the
//! events and the end of a stream are the same and are here, and so are the
//! segments of a reply ([`Segments`]), which one API writes as blocks and the
//! other as items.

use std::collections::VecDeque;
use std::convert::Infallible;

use ares_api::Progress;
use axum::response::Response;
use axum::response::sse::{Event as SseEvent, KeepAlive};

use crate::error::ApiError;
use crate::generate::{End, Generation, Step};
use crate::models::Lease;
use crate::openai::request::Parsed;
use crate::routes::{App, json_response, sse_response, with_limit};
use crate::sorter::{Piece, Sorter};

/// One event of a stream: its name and its data.
pub(crate) type Written = (&'static str, String);

/// What one API writes of a reply.
pub(crate) trait Writer: Send + 'static {
    /// The events that open a stream, before anything of the reply.
    fn begin(&mut self, out: &mut VecDeque<Written>);
    /// The events of pieces of a streamed reply.
    fn pieces(&mut self, pieces: Vec<Piece>, out: &mut VecDeque<Written>);
    /// The events of the end of a streamed reply: what the sorter still held,
    /// and what closes the stream.
    fn end(&mut self, end: End, out: &mut VecDeque<Written>);
    /// The event of a failure after the stream started, its last one.
    fn error(&mut self, error: &ApiError) -> Written;
    /// The events that say how far the prompt has got, for a request that
    /// asked (llama.cpp's `return_progress`); none from an API that has no
    /// place for it.
    fn progress(&mut self, _progress: &Progress, _out: &mut VecDeque<Written>) {}
    /// Takes pieces of a reply that is answered whole.
    fn take(&mut self, pieces: Vec<Piece>);
    /// The body of a reply that is answered whole, at its end.
    fn whole(self, end: End) -> String;
}

/// Starts the request `parsed` on the model `lease` holds and answers it
/// with `writer`, whole or streamed as the request asks; a stream sends
/// `keep_alive` while it waits.
pub(crate) async fn answer<W: Writer>(
    app: &App,
    lease: &Lease,
    parsed: Parsed,
    sorter: Sorter,
    writer: W,
    keep_alive: KeepAlive,
) -> Result<Response, ApiError> {
    let (streamed, lowered) = (parsed.stream, parsed.lowered);
    let progress = parsed.stream && parsed.progress;
    let live = parsed.stream && parsed.timings_per_token;
    let keep_alive = crate::routes::pings(keep_alive, parsed.ping);
    let reply =
        Generation::start(app, lease, parsed.request, sorter)?.reporting(progress).timing(live);
    let response = if streamed {
        stream(reply, writer, app, keep_alive).await
    } else {
        whole(reply, writer).await
    };
    Ok(with_limit(response?, lowered))
}

/// The answer that is not streamed. Neither API has a field that asks for
/// the probabilities of tokens.
async fn whole<W: Writer>(mut reply: Generation, mut writer: W) -> Result<Response, ApiError> {
    loop {
        match reply.next().await? {
            Step::Pieces(pieces) => writer.take(pieces),
            Step::Probs(_) | Step::Progress(_) => {}
            Step::End(end) => return Ok(json_response(writer.whole(end))),
        }
    }
}

/// The streamed answer.
///
/// As on the chat route the response starts when the engine has produced its
/// first step, so that a request the engine refuses gets an HTTP status, or
/// when `--stream-wait` has passed without it (`Generation::first`): then the
/// opening events and the keep-alives go out while the prompt is computed. A
/// failure after the start is the writer's error event, and the stream ends
/// with it.
async fn stream<W: Writer>(
    mut reply: Generation,
    mut writer: W,
    app: &App,
    keep_alive: Option<KeepAlive>,
) -> Result<Response, ApiError> {
    let first = reply.first(app.config.stream_wait).await?;
    let mut queue = VecDeque::new();
    writer.begin(&mut queue);
    let state = Events { reply, writer, queue, next: first, over: false };
    let events = futures_util::stream::unfold(state, |mut state| async move {
        let (name, data) = state.step().await?;
        Some((Ok::<_, Infallible>(SseEvent::default().event(name).data(data)), state))
    });
    Ok(sse_response(events, keep_alive))
}

/// The state of a stream.
#[derive(Debug)]
struct Events<W> {
    reply: Generation,
    writer: W,
    /// Events that are ready and have not gone out.
    queue: VecDeque<Written>,
    /// A step of the reply that has come and is not written yet.
    next: Option<Step>,
    over: bool,
}

impl<W: Writer> Events<W> {
    /// The next event, or `None` when the stream is over. Dropping the
    /// stream half way drops the reply, which cancels the request in the
    /// engine.
    async fn step(&mut self) -> Option<Written> {
        loop {
            if let Some(event) = self.queue.pop_front() {
                return Some(event);
            }
            if self.over {
                return None;
            }
            let step = match self.next.take() {
                Some(step) => Ok(step),
                None => self.reply.next().await,
            };
            match step {
                Ok(Step::Pieces(pieces)) => {
                    let before = self.queue.len();
                    self.writer.pieces(pieces, &mut self.queue);
                    // llama.cpp's `timings_per_token`: the data of the step's last
                    // event has the timings so far.
                    if self.queue.len() > before
                        && let (Some(timings), Some((_, data))) =
                            (self.reply.timings_now(), self.queue.back_mut())
                    {
                        crate::openai::reply::with_timings(data, timings);
                    }
                }
                Ok(Step::Progress(progress)) => self.writer.progress(&progress, &mut self.queue),
                Ok(Step::Probs(_)) => {}
                Ok(Step::End(end)) => {
                    self.writer.end(end, &mut self.queue);
                    self.over = true;
                }
                Err(error) => {
                    error.tell("the stream failed");
                    self.over = true;
                    return Some(self.writer.error(&error));
                }
            }
        }
    }
}

/// What a segment of a reply holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Reasoning,
    Text,
    Call,
}

/// A step of the segments of a reply ([`Segments`]); `index` is the
/// segment's place in the reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Segment {
    /// A segment of reasoning or of text opens.
    Open { index: usize, kind: Kind },
    /// A segment of a call opens: its id and the name of its function.
    Call { index: usize, id: String, name: String },
    /// Text of the open segment of reasoning or of text.
    Text { index: usize, kind: Kind, text: String },
    /// A fragment of the JSON text of the open call's arguments.
    Arguments { index: usize, json: String },
    /// The open segment closes.
    Close { index: usize, kind: Kind },
}

/// The pieces of a reply as its segments: reasoning and text go on in the
/// segment of their kind that is open, and open one when another kind, or
/// none, is; a call is a segment of its own, with the fragments of its
/// arguments while it is open; a segment closes when the next opens or the
/// reply ends.
#[derive(Debug, Default)]
pub(crate) struct Segments {
    /// The kind of the segment that is open.
    open: Option<Kind>,
    /// How many segments have begun.
    begun: usize,
    /// The segment of each call of the reply.
    calls: Vec<usize>,
}

impl Segments {
    /// The steps of the next pieces.
    pub fn push(&mut self, pieces: Vec<Piece>) -> Vec<Segment> {
        let mut steps = Vec::new();
        for piece in pieces {
            match piece {
                Piece::Text(split) => {
                    for (kind, text) in
                        [(Kind::Reasoning, split.reasoning), (Kind::Text, split.content)]
                    {
                        if !text.is_empty() {
                            let index = self.open(kind, &mut steps);
                            steps.push(Segment::Text { index, kind, text });
                        }
                    }
                }
                Piece::Call { id, name, .. } => {
                    self.close(&mut steps);
                    let index = self.begin(Kind::Call);
                    self.calls.push(index);
                    steps.push(Segment::Call { index, id, name });
                }
                Piece::Arguments { index, json } => {
                    // The arguments of a call come while its segment is open.
                    if let Some(&index) = self.calls.get(index)
                        && self.open == Some(Kind::Call)
                        && index + 1 == self.begun
                        && !json.is_empty()
                    {
                        steps.push(Segment::Arguments { index, json });
                    }
                }
            }
        }
        steps
    }

    /// The steps of the end of the reply: the open segment closes.
    pub fn finish(&mut self) -> Vec<Segment> {
        let mut steps = Vec::new();
        self.close(&mut steps);
        steps
    }

    /// How many calls the reply has begun.
    pub fn calls(&self) -> usize {
        self.calls.len()
    }

    /// The index of the open segment of `kind`, which is opened if another
    /// kind, or none, is open.
    fn open(&mut self, kind: Kind, steps: &mut Vec<Segment>) -> usize {
        if self.open != Some(kind) {
            self.close(steps);
            let index = self.begin(kind);
            steps.push(Segment::Open { index, kind });
        }
        self.begun - 1
    }

    fn begin(&mut self, kind: Kind) -> usize {
        self.open = Some(kind);
        self.begun += 1;
        self.begun - 1
    }

    fn close(&mut self, steps: &mut Vec<Segment>) {
        if let Some(kind) = self.open.take() {
            steps.push(Segment::Close { index: self.begun - 1, kind });
        }
    }
}
