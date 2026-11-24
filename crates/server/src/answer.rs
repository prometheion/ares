//! How the Messages and the Responses routes answer a reply: whole, as one
//! JSON body, or streamed, as server-sent events. The two APIs differ only
//! in what they write ([`Writer`]); waiting for the engine, the order of the
//! events and the end of a stream are the same and are here.

use std::collections::VecDeque;
use std::convert::Infallible;

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
    let reply = Generation::start(app, lease, parsed.request, sorter)?;
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
            Step::Probs(_) => {}
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
    keep_alive: KeepAlive,
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
                Ok(Step::Pieces(pieces)) => self.writer.pieces(pieces, &mut self.queue),
                Ok(Step::Probs(_)) => {}
                Ok(Step::End(end)) => {
                    self.writer.end(end, &mut self.queue);
                    self.over = true;
                }
                Err(error) => {
                    self.over = true;
                    return Some(self.writer.error(&error));
                }
            }
        }
    }
}
