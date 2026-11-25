//! The thread that decides.
//!
//! Everything the engine knows about its requests lives here, as plain data
//! that only this thread touches: sequences and their page tables, the
//! admission accounting and the line of requests that wait, the plan of each
//! step, the step time model, the corpus the drafts are looked up in, the
//! state of every request. Other threads reach it through one channel
//! ([`Command`]) and are reached through sinks and the executor, so there is
//! nothing to lock.
//!
//! One turn of the loop is one step:
//!
//! ```text
//! intake -> admit -> plan -> claim pages -> write record -> dispatch ... collect -> commit -> events
//!           (who     (drafts  (drafts cut,                  (devices work)         (accept, cut,
//!            starts)  first)   someone gives way,                                   publish pages,
//!                              cached pages evicted)                                capture state,
//!                                                                                   corpus, update)
//! ```
//!
//! # Rounds and pieces
//!
//! A step is a *round* of generation or a *piece* of prompt, never both
//! kinds of rows at their full size ([`ares_sched::Share`] decides which, and
//! the module docs there say why). A round carries the newest token of every
//! generating sequence and the drafts that pay ([`ares_sched::choose`], from
//! the price of a step by its rows and the acceptance measured); a piece
//! carries up to [`Model::prompt_rows`] rows of the prompts that have started,
//! shortest first, and the newest token of every generating sequence without
//! drafts. Before the first request [`Conductor::warm_up`] measures the price
//! of a step for every row count a window can have, and of a few above.
//!
//! # Who runs
//!
//! Nothing is reserved for a request's longest possible length and nothing is
//! refused for load. A request is put in line ([`ares_sched::Admission`]) and
//! starts when the pages of its prompt and the output it is expected to
//! produce fit next to what the running requests are expected to need; a
//! request that is alone always starts. Every step claims the pages its rows
//! write. When they do not fit, cached pages nobody holds are evicted, then
//! draft rows are dropped; then the request with the fewest generated tokens
//! gives way: it lets go of its place on the devices and waits at the front of
//! the line, and what it computed stays in the prefix cache, so that it comes
//! back as a prefix hit. The one request left in the pool is never made to
//! wait: when the pool cannot hold it any longer it ends, as it would at the
//! end of the context.
//!
//! What bounds the number of requests: the ones that run are bounded by the
//! places the devices have ([`RuntimeConfig::max_slots`]: a recurrent state,
//! a page table and a row of logits each) and by the pool; the ones that wait
//! are not bounded.
//!
//! # Images
//!
//! A prompt's images take cells of their own ([`crate::ImageSpan`]), whose
//! rows the model's image tower computes. The tower runs between steps, on the
//! device that holds it, as prompt work: a piece of prompt starts with as many
//! units of the tower's pass as fit the time of a piece while sequences
//! generate, and with the whole pass while none does. Its time is charged to
//! the share like a piece's. A prompt is read up to the first image whose rows
//! are not in the store ([`crate::ImageStore`]); the step that reaches an
//! image copies its rows into the image stage, and every row is written at
//! its rotary position. The prefix cache names an image's cells by the image
//! and the row ([`crate::cell_name`]), so a conversation with an image resumes
//! from its pages like any other.
//!
//! # The prefix cache
//!
//! For a model that takes part ([`Model::prefix_cache`]) the pages and the
//! recurrent state a request computes outlive it, and a request whose prompt
//! starts with text the engine has computed starts from them.
//!
//! - **Lookup.** A request is looked up when it arrives (the prompt without
//!   its last token, whose logits are needed) and again when it starts. It
//!   shares the cached pages and a tail, restores the recurrent state from the
//!   last checkpoint on that path, **replays** the rows between that checkpoint
//!   and the end of the cached cells (they read the cells and write none) and
//!   computes the rest. A prompt whose first missing page is being computed by
//!   a sequence that runs is held back in line until that page is published or
//!   the sequence lets go of it.
//! - **Publish.** At the commit that fills a page it enters the index, so a
//!   second prompt shares it at once. A prefill step ends where a checkpoint is
//!   due ([`ares_state::CheckpointPolicy`]), and so does the window of a
//!   generating sequence, so that the state is exactly there when the
//!   checkpoint is taken.
//! - **Retire.** When a request ends, gives way or is cancelled its partly used
//!   last page is published as a tail and its live state becomes that tail's
//!   checkpoint. The copy of the state runs with the next record, before
//!   anything of that record writes a slot, so the slot is free at once.
//! - **Evict.** When a step's claims need room, cached pages nobody holds go
//!   from the ends of chains, after the requests that start took the pages they
//!   share. Checkpoints go when their budget is full
//!   ([`RuntimeConfig::cache_ram`]) or with the page they hang on.
//!
//! # Files
//!
//! This file holds the conductor's state, the loop and the step's plan; the
//! rest is by concern: `lifecycle.rs` (a request queued, started, given way,
//! ended), `drafting.rs` (what a round asks for, the step that only drafts,
//! the drafts verified), `images.rs` (the image tower), `step_record.rs`
//! (the record written and run on the devices, the paused step),
//! `commit.rs` (what a step came to) and `measure.rs` (the warm-up).

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

use ares_api::{EngineError, Event, EventSink, FinishReason, RequestId, Sampling, Usage};
use ares_sampler::{SampleParams, SampleRow, TokenMask};
use ares_sched::{
    Acceptance, Admission, AdmissionConfig, Arrival, Choice, Drafting, Estimate, Limits, Load,
    NGRAM, NgramCorpus, NgramDrafter, Offer, Outcome, PartKind, Pending, PoolView, PriceTable,
    Request, SeqId, Share, Turn, UNASKED_SEQUENCE, Want, Wish, plan,
};
use ares_state::{
    Boundary, Chain, ChainHash, Owner, PagePool, PageState, PrefixIndex, Priority, Resume, Sequence,
};

use crate::cache::Cache;
use crate::config::RuntimeConfig;
use crate::exec::{Clock, ExecError, Executor, StepMeta, StepOutput};
use crate::image::{ImageInput, ImageName, ImageRows, ImageSpan, ImageStore, Spans, cell_name};
use crate::model::{ContextSpec, Drafter, Model, TowerJob, Vision};
use crate::record::{self, Commit, ImageRun, RecordWriter, Row, TableUpdate};
use crate::sampling::{Context, RequestSampler, StepRows, TokenFilter};
use crate::timing::{Calibration, DRAFT_POSITIONS, Phase, RequestReport, Snapshot, Stats};
use crate::watch::Watch;
use decisions::Decision;
pub(crate) use decisions::{DecisionSubmission, Role};

mod commit;
mod decisions;
mod drafting;
mod images;
mod lifecycle;
mod measure;
mod step_record;

/// How long the conductor sleeps on its channel when requests wait and none
/// can start. Admission starts the first in line whenever nothing runs, so
/// this is a guard against spinning and not a path a request takes.
const STALLED: Duration = Duration::from_millis(1);

/// When the step in flight was handed to the devices, for the watchdog
/// (`RuntimeConfig::watchdog`), which reads it from a thread of its own.
pub(crate) struct Heartbeat {
    epoch: Instant,
    /// Nanoseconds after `epoch`, plus one, at which the step in flight was
    /// dispatched; 0 while no step is in flight.
    since: std::sync::atomic::AtomicU64,
}

impl Heartbeat {
    pub(crate) fn new() -> Self {
        Self { epoch: Instant::now(), since: std::sync::atomic::AtomicU64::new(0) }
    }

    fn begin(&self) {
        let ns = self.epoch.elapsed().as_nanos() as u64 + 1;
        self.since.store(ns, std::sync::atomic::Ordering::Release);
    }

    fn end(&self) {
        self.since.store(0, std::sync::atomic::Ordering::Release);
    }

    /// How long the step in flight has run, if one is.
    pub(crate) fn busy(&self) -> Option<Duration> {
        let since = self.since.load(std::sync::atomic::Ordering::Acquire);
        (since != 0).then(|| self.epoch.elapsed().saturating_sub(Duration::from_nanos(since - 1)))
    }
}

/// What other threads ask of the conductor.
pub(crate) enum Command {
    Submit(Box<Submission>),
    /// A decision ([`decisions`]).
    Decide(Box<DecisionSubmission>),
    Cancel(RequestId),
    Stats {
        reset: bool,
        reply: SyncSender<StatsCopy>,
    },
    Inspect {
        reply: SyncSender<Vec<SequenceInfo>>,
    },
    /// The checkpoints may hold less page-locked memory than they do: the
    /// conductor gives the coldest up, also while nothing runs.
    Trim,
    Shutdown,
}

/// The conductor's measurements, copied out. The percentiles are computed from
/// the copy by whoever asked, so that sorting the samples never takes the
/// conductor's time.
pub(crate) struct StatsCopy {
    stats: Stats,
    calibration: Option<Calibration>,
    step_ms: Vec<(u32, f64)>,
    chain_ms: Vec<Vec<(u32, f64)>>,
    pages_free: u32,
    pages_capacity: u32,
    cached_pages: u32,
    checkpoint_ram: u64,
    active: u32,
    waiting: u32,
    ssd_problem: Option<String>,
}

impl StatsCopy {
    pub(crate) fn into_snapshot(self) -> Snapshot {
        let mut snapshot = Snapshot::new(
            &self.stats,
            self.pages_free,
            self.pages_capacity,
            self.cached_pages,
            self.checkpoint_ram,
            self.active,
            self.waiting,
        );
        snapshot.calibration = self.calibration;
        snapshot.step_ms = self.step_ms;
        snapshot.chain_ms = self.chain_ms;
        snapshot.ssd_problem = self.ssd_problem;
        snapshot
    }
}

/// A request that has been checked.
pub(crate) struct Submission {
    pub id: RequestId,
    pub prompt: Vec<u32>,
    /// Where the parts of the conversation end in the prompt, for the
    /// checkpoints of the prefix cache.
    pub boundaries: Vec<Boundary>,
    /// The images of the prompt, ascending.
    pub images: Vec<ImageSpan>,
    pub max_tokens: Option<u32>,
    /// The stop strings, and what else is watched in the reply's text.
    pub stop: Watch,
    /// How the request's tokens are picked.
    pub sampling: Sampling,
    /// Whether drafts may be verified for the request.
    pub draft: bool,
    /// A rule about which tokens the request may pick, if it has one.
    pub filter: Option<Box<dyn TokenFilter>>,
    /// Who goes first: a lower number before a higher one.
    pub priority: i32,
    /// The line was full when the request came: it takes the place of the
    /// worst waiting request of a worse priority that has not started, which
    /// is refused as overloaded, or is refused itself.
    pub displace: bool,
    pub sink: EventSink,
    /// What the request is for: a client's, or a part of a decision.
    pub role: Role,
}

/// What the conductor knows about a running sequence, for tests and logs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceInfo {
    pub id: RequestId,
    pub slot: u32,
    /// Tokens of the prompt and the output so far.
    pub tokens: u32,
    pub prompt_tokens: u32,
    /// Cells computed: the length of the page table's content.
    pub cells: u64,
    pub pages: Vec<u32>,
}

/// Where a request is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// Running, in this slot of the devices.
    Slot(u32),
    /// In line.
    Waiting,
}

/// A request the engine holds: running in a slot of the devices, or waiting
/// in line.
struct Active {
    id: RequestId,
    /// The request's name in the admission accounting, while it waits and
    /// while it runs.
    seq: SeqId,
    sink: EventSink,
    /// The page table. Its length is the number of cells computed, which is
    /// the number of tokens less one once the prompt is read: the newest
    /// token has no cell until a step computes it. A request that starts from
    /// the prefix cache starts with the cells it shares.
    cells: Sequence,
    /// The prompt and the output. An image's cells have the model's pad
    /// token.
    tokens: Vec<u32>,
    /// The names of the cells of `tokens` for the prefix cache: a text cell's
    /// is its token, an image cell's its own ([`cell_name`]).
    names: Vec<u64>,
    /// The images of the prompt.
    spans: Spans,
    /// Per token, the host-side context state (see [`ContextSpec`]).
    states: Vec<u32>,
    prompt_len: u32,
    /// Where the parts of the conversation end in the prompt: the system
    /// prompt, the tool list, the messages. The prefix cache keeps
    /// checkpoints there.
    boundaries: Vec<Boundary>,
    /// How many of `tokens` are computed as certain rows before the sequence
    /// generates: the prompt, and for a request that gave way and came back
    /// everything it had read and written by then.
    known: u32,
    /// The names of the pages the sequence has, for the prefix index.
    chain: Chain,
    /// The sequence's name in the prefix index, for its announcements and
    /// the checkpoints it publishes.
    owner: Owner,
    /// Whether the request has run (started once at least).
    ran: bool,
    /// The cells that start shared, and of them the rows that are replayed
    /// and still to run: the recurrent state is at `replay_end - replay_left`
    /// until the replay is done, and at the end of the cells after.
    replay_end: u32,
    replay_left: u32,
    /// Where the last checkpoint on this sequence's own path is: restored or
    /// taken; 0 for none.
    checkpoint_at: u64,
    /// The first page of the prompt that is not cached is being computed by
    /// another sequence: the request is held back until that changes.
    waits_for: Option<ChainHash>,
    /// Whether the request was held back once at least.
    was_held: bool,
    /// Whether pages of its prompt came back from RAM while it waited. It
    /// brings pages back once: two requests that wait would otherwise take
    /// the pool's pages from each other at every step.
    brought_back: bool,
    /// The number of evictions the accounting was last told about the cache
    /// at.
    looked_at: u64,
    /// For the request's report: what its first start took from the cache,
    /// and the rows it replayed and computed itself.
    cached_prompt: u32,
    replayed: u32,
    computed: u32,
    /// The sequence's place in the corpus of all text. Its tokens enter the
    /// corpus when they are certain: prompt tokens when the step that
    /// computed their rows is back, generated tokens when they are accepted.
    drafter: NgramDrafter,
    /// How the drafts of each of the model's drafters have done for this
    /// sequence, per position.
    model_acceptance: Vec<Acceptance>,
    /// Whether the request lets drafts be verified for it.
    draft_allowed: bool,
    /// How the sequence's tokens are picked, and what that keeps from step
    /// to step.
    sampler: RequestSampler,
    stop: Watch,
    max_tokens: Option<u32>,
    completion: u32,
    waited_steps: u32,
    /// Who goes first: a lower number before a higher one.
    priority: i32,
    /// Whether the request was counted as one that had to wait.
    counted_waiting: bool,
    /// Drafts for the coming step; cut to the rows the plan grants. When one
    /// of the model's drafters makes them (`own`, its place in the model's
    /// list), placeholders until it has run (the step that only drafts), and
    /// then the drafts it made that the round verifies, with the probability
    /// it gave each (`model_probs`; empty from a drafter that does not say).
    draft: Vec<u32>,
    model_drafter: Option<usize>,
    model_probs: Vec<f32>,
    /// What the n-gram drafter proposes for the coming step, with the chance of each
    /// token (the n-gram drafter still learns from what turns out to be right when
    /// the model's drafts were taken instead), the chance of each position of
    /// each model drafter's draft for the step (`offered`), and of the one
    /// taken (`model_chances`).
    ngram: Vec<u32>,
    ngram_chances: Vec<f64>,
    offered: Vec<Vec<f64>>,
    model_chances: Vec<f64>,
    /// For a window that takes the model's own drafts: the reach its chain
    /// needs for each link ([`ares_sched::need`]).
    model_needs: Vec<f32>,
    /// Bytes of generated text that are not whole characters yet.
    pending: Vec<u8>,
    /// Text produced this step, not yet delivered.
    emitted: String,
    finished: Option<FinishReason>,
    /// What the request is for: a client's, or a part of a decision
    /// ([`decisions`]), which reads its prompt and generates nothing.
    role: Role,
}

impl Active {
    fn generating(&self) -> bool {
        self.cells.len() >= u64::from(self.known)
    }

    fn usage(&self) -> Usage {
        Usage {
            prompt_tokens: self.prompt_len,
            completion_tokens: self.completion,
            cached_prompt_tokens: self.cached_prompt,
            output_hash: token_hash(&self.tokens[self.prompt_len as usize..]),
            reasoning_tokens: self.stop.reasoning_tokens(),
            queued: self.stop.queued(),
        }
    }

    /// Where the recurrent state is: the end of the cells, or while cached
    /// cells are replayed, where the replay has got to.
    fn state_pos(&self) -> u64 {
        if self.replay_left > 0 {
            u64::from(self.replay_end - self.replay_left)
        } else {
            self.cells.len()
        }
    }

    /// The rows of the prompt that are left to run: replayed and computed.
    fn prefill_left(&self) -> u32 {
        self.known.saturating_sub(self.cells.len() as u32) + self.replay_left
    }

    /// The first image of the prompt from cell `from` on whose rows `store`
    /// does not have.
    fn missing_image(&self, store: &ImageStore, from: u32) -> Option<Arc<ImageInput>> {
        let i = self.spans.from(from)?;
        self.spans.spans()[i..]
            .iter()
            .take_while(|s| s.first < self.known)
            .find(|s| !store.contains(&s.image.name))
            .map(|s| s.image.clone())
    }

    /// How many of `rows` rows from cell `from` can run now: those before the
    /// first image whose rows `store` does not have.
    fn ready_rows(&self, from: u64, rows: u32, store: &ImageStore) -> u32 {
        let from = from as u32;
        let Some(i) = self.spans.from(from) else { return rows };
        for span in &self.spans.spans()[i..] {
            if span.first >= from + rows {
                break;
            }
            if !store.contains(&span.image.name) {
                return span.first.saturating_sub(from).min(rows);
            }
        }
        rows
    }

    /// How many of `rows` rows from cell `from` fit `room` rows of the image
    /// stage, and the image rows among them.
    fn rows_within(&self, from: u64, rows: u32, room: u32) -> (u32, u32) {
        let from = from as u32;
        let Some(i) = self.spans.from(from) else { return (rows, 0) };
        let mut used = 0;
        for span in &self.spans.spans()[i..] {
            if span.first >= from + rows {
                break;
            }
            let (lo, hi) = (span.first.max(from), span.end().min(from + rows));
            if used + (hi - lo) > room {
                return (lo + (room - used) - from, room);
            }
            used += hi - lo;
        }
        (rows, used)
    }

    /// The most tokens the request may still generate in a context of
    /// `context_length`: its limit, or the room the context leaves.
    fn max_new_tokens(&self, context_length: u32) -> u32 {
        let room = context_length.saturating_sub(self.prompt_len);
        self.max_tokens.map_or(room, |max| max.min(room))
    }

    /// Takes the tokens the model produced for this request, in order, up to
    /// and including the one that ends it.
    fn accept(
        &mut self,
        model: &dyn Model,
        spec: &ContextSpec,
        context_length: u32,
        corpus: &mut NgramCorpus,
        picked: &[u32],
    ) {
        for (i, &token) in picked.iter().enumerate() {
            self.completion += 1;
            if model.ends(token) {
                self.stop.token(&[]);
                return self.end_reply(FinishReason::Stop);
            }
            let previous = self.states.last().copied().unwrap_or_else(|| spec.start());
            self.tokens.push(token);
            self.names.push(u64::from(token));
            self.states.push(spec.advance(previous, &self.tokens));
            self.drafter.push(corpus, token);
            let piece_at = self.pending.len();
            model.piece(token, &mut self.pending);
            self.stop.token(&self.pending[piece_at..]);
            if self.sampler.wants_probs() {
                self.sampler.token_taken(model, i, token, &self.pending[piece_at..]);
            }
            let (stop, emitted) = (&mut self.stop, &mut self.emitted);
            if take_characters(&mut self.pending, |ch| stop.push(ch, emitted)) {
                self.finished = Some(self.stop.finish());
                return;
            }
            // A reply caught in a loop ends as a limit ends it (`guard`).
            if self.sampler.cuts(&self.tokens, self.prompt_len as usize)
                || self.max_tokens.is_some_and(|max| self.completion >= max)
                || self.tokens.len() as u32 >= context_length
            {
                return self.end_reply(FinishReason::Length);
            }
        }
    }

    /// The reply ends for `finish`: the text the stop strings held back goes
    /// out, and the sampler's rules learn that the reply is over.
    fn end_reply(&mut self, finish: FinishReason) {
        self.stop.flush(&mut self.emitted);
        self.sampler.reply_ended();
        self.finished = Some(finish);
    }
}

impl Drop for Conductor {
    /// A conductor that stops any other way than through `run` (a panic, a
    /// failed start) still ends its requests, so no client waits for ever.
    fn drop(&mut self) {
        let gone = EngineError::Internal("the engine's conductor stopped".into());
        for slot in &mut self.slots {
            if let Some(a) = slot.take() {
                let _ = a.sink.send(Event::Error(gone.clone()));
                a.cells.release(&mut self.pool);
            }
        }
        for a in self.waiting.drain(..) {
            let _ = a.sink.send(Event::Error(gone.clone()));
            a.cells.release(&mut self.pool);
        }
        for (_, decision) in self.decisions.drain() {
            decision.fail(gone.clone());
        }
        while let Ok(command) = self.rx.try_recv() {
            match command {
                Command::Submit(submission) => {
                    let _ = submission.sink.send(Event::Error(gone.clone()));
                }
                Command::Decide(submission) => {
                    submission.sink.send(ares_api::DecisionOutcome::Failed(gone.clone()))
                }
                _ => {}
            }
        }
    }
}

/// A hash of token ids (FNV-1a over their four bytes each, little end first):
/// what two replies are compared by without keeping their tokens.
pub fn token_hash(tokens: &[u32]) -> u64 {
    tokens.iter().flat_map(|t| t.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Hands the whole characters at the start of `pending` to `push`, in order,
/// and removes them; bytes that cannot be part of a character become U+FFFD.
/// What is left is the start of a character that later bytes complete. Stops
/// early, and returns true, when `push` returns true.
fn take_characters(pending: &mut Vec<u8>, mut push: impl FnMut(char) -> bool) -> bool {
    let mut taken = 0;
    let mut stopped = false;
    while taken < pending.len() && !stopped {
        let (text, bad) = match std::str::from_utf8(&pending[taken..]) {
            Ok(text) => (text, Some(0)),
            Err(error) => {
                let valid = &pending[taken..taken + error.valid_up_to()];
                // SAFETY: `valid_up_to` bytes are valid UTF-8 by its contract.
                (unsafe { std::str::from_utf8_unchecked(valid) }, error.error_len())
            }
        };
        let len = text.len();
        stopped = text.chars().any(&mut push);
        taken += len;
        match bad {
            // The tail is an unfinished character: wait for its other bytes.
            None => break,
            Some(0) => {}
            Some(n) => {
                stopped = stopped || push(char::REPLACEMENT_CHARACTER);
                taken += n;
            }
        }
    }
    pending.drain(..taken);
    stopped
}

/// A pass of the image tower under way: its image, the next unit and the
/// buffer its rows come back in.
#[derive(Debug)]
struct TowerPass {
    image: Arc<ImageInput>,
    next: u32,
    rows: Vec<f32>,
    /// What the units run so far took, in milliseconds.
    ms: f64,
}

/// What a step record is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordKind {
    /// A round or a piece of prompt, every window with its drafts as tokens;
    /// a step the model may pause after `pause_ms` (0: it may not).
    Round { pause_ms: u32 },
    /// The paused step goes on, for at most `pause_ms` more: its rows were
    /// counted and its saves inside named when it was first written.
    Resume { pause_ms: u32 },
    /// The step that only drafts ([`Conductor::chains_first`]): the windows
    /// whose drafts a model's drafter makes, as far as their chains are
    /// asked, with their draft bars.
    DraftsOnly,
}

/// One sequence's rows in the step being run.
#[derive(Clone, Copy, Debug)]
struct StepPart {
    slot: u32,
    /// The sequence that was in the slot when the step was planned.
    seq: SeqId,
    rows: u32,
    kind: PartKind,
    /// Where the sequence's recurrent state is before the rows, which is the
    /// position of the first row.
    from: u64,
    /// The rows are a replay: they rebuild recurrent state from cached cells
    /// and write no cell.
    replay: bool,
}

impl StepPart {
    /// The model's drafter that makes the drafts of this part, of sequence
    /// `a`: a window with drafts whose sequence took a model drafter's offer.
    fn model_drafter(&self, a: &Active) -> Option<usize> {
        a.model_drafter.filter(|_| self.rows > 1 && matches!(self.kind, PartKind::Window))
    }
}

/// The fewest rows a piece of prompt is cut to by its time bound
/// ([`RuntimeConfig::piece_ms`]): the smallest step of prompt the warm-up
/// measures.
const MIN_PIECE: u32 = 16;

/// The load of `rows` consecutive rows of a sequence that start at position
/// `context`, for a model whose rows read at most `span` cells each.
fn load_of(span: u32, context: u64, rows: u32) -> Load {
    let span = u64::from(span);
    // The rows that still read less than the span, and the ones at it.
    let below = span.saturating_sub(context).min(u64::from(rows)) as u32;
    let head = Load::chunk(context, below);
    Load { rows, cells: head.cells + u64::from(rows - below) * span }
}

/// What the admission accounting may ask about the pool.
struct View<'a> {
    pool: &'a PagePool,
    slots: &'a [Option<Active>],
    index: Option<&'a mut PrefixIndex>,
}

impl PoolView for View<'_> {
    fn evictable(&mut self) -> u32 {
        self.index.as_mut().map_or(0, |index| index.evictable(self.pool))
    }

    fn releasable(&mut self, seq: SeqId) -> u32 {
        self.slots
            .iter()
            .flatten()
            .find(|a| a.seq == seq)
            .map_or(0, |a| a.cells.releasable(self.pool))
    }
}

/// What a step of the warm-up is shaped like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    /// Rows of a prompt: certain, and the last one wanted.
    Prompt,
    /// A window that verifies drafts: not certain, every row wanted.
    Window,
    /// The drafts of the model's drafter (its place in the model's list) for
    /// a window: a step that only drafts.
    Chain(usize),
}

pub(crate) struct Conductor {
    model: Arc<dyn Model>,
    spec: ContextSpec,
    config: RuntimeConfig,
    context_length: u32,
    max_rows: u32,
    /// The most cells a row reads through attention.
    span: u32,
    exec: Box<dyn Executor>,
    /// The executor's clock: device threads stamp their events with it.
    clock: Clock,
    rx: Receiver<Command>,
    pool: PagePool,
    /// The prefix cache, for a model that takes part in it.
    cache: Option<Cache>,
    /// The image tower of a model that reads images, the rows it computed,
    /// the pass under way, and what a unit of a pass took per row of its
    /// image the last time (milliseconds).
    vision: Option<Vision>,
    images: ImageStore,
    pass: Option<TowerPass>,
    unit_ms_per_row: Option<f64>,
    /// The rows of the image stage, and the image runs of the record being
    /// written with the rows that go into the stage.
    stage_rows: u32,
    image_runs: Vec<ImageRun>,
    staged: Vec<(Arc<ImageRows>, u32, u32)>,
    /// Who runs, who waits and who gives way.
    admission: Admission,
    /// The running requests, by their slot on the devices.
    slots: Vec<Option<Active>>,
    free_slots: Vec<u32>,
    /// The requests in line. Their order is the accounting's business.
    waiting: Vec<Active>,
    by_id: HashMap<RequestId, Place>,
    next_seq: u32,
    active: usize,
    /// What a step costs by its rows, measured at load and kept up to date.
    prices: PriceTable,
    /// The model's drafters, in its order, and what the engine knows of them:
    /// what their drafts are worth and cost, the bars of their links and the
    /// worth of a token.
    drafters: Vec<Drafter>,
    drafting: Drafting,
    /// Whose turn it is, rounds of generation or pieces of prompt.
    share: Share,
    /// The rows of each prompt that has started and is not read yet, for the
    /// share's decision.
    prompt_rows: Vec<u32>,
    /// The tokens the windows of the step that committed last produced.
    round_tokens: u32,
    /// The text of all sequences, which drafts are looked up in.
    corpus: NgramCorpus,
    stats: Stats,
    step_no: u64,
    after_idle: bool,
    /// When the last device finished the previous step, on the executor's clock.
    last_done_ns: Option<u64>,
    failed: Option<String>,
    /// `failed`, for whoever asks from another thread ([`Conductor::failure`]).
    failure: std::sync::Arc<std::sync::OnceLock<String>>,
    shutting_down: bool,
    /// Requests wait and the last pass could start none of them.
    stalled: bool,
    /// When the step in flight began, for the watchdog.
    heartbeat: Arc<Heartbeat>,
    /// The step the model paused between two of its pieces (a long prompt read layer by
    /// layer, [`StepOutput::paused`]): its one part, which runs again, alone, when it is its
    /// turn; its sequence's cells are claimed and its state is the model's until it ends.
    paused: Option<StepPart>,
    /// The paused step's sequence ended: the next record says so.
    give_up: bool,
    /// The rows of the images among the paused step's rows, held for it so that the
    /// image store cannot let them go before it goes on: its record stages them again.
    paused_images: Vec<(ImageName, Arc<ImageRows>)>,
    /// What the engine has been asked so far ([`Executor::asks`]), and what of it had been
    /// when the conductor last took its commands: a model that pauses long steps pauses
    /// one when it has been asked more since.
    asks: Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
    asked: u32,
    /// Checkpoints due inside the rows of the step being planned, for a model that saves
    /// them there ([`Model::saves_inside_steps`]): (slot, position); then the places taken
    /// for them, which the record names and the commit attaches; and those of the paused
    /// step.
    inner_due: Vec<(u32, u64)>,
    inner: Vec<crate::InnerSave>,
    paused_inner: Vec<crate::InnerSave>,
    /// The checkpoints that were due inside the paused step's rows (with a place or not).
    paused_due: Vec<(u32, u64)>,
    // Scratch that lives across steps so that a step allocates nothing for it.
    requests: Vec<Request>,
    parts: Vec<StepPart>,
    updates: Vec<(u32, Range<usize>)>,
    /// The pages the pool freed since the last record, for the devices.
    freed: Vec<u32>,
    runs: Vec<(u32, u32)>,
    /// The outcome of this step's draft windows, for the next step's record.
    commits: Vec<Commit>,
    /// The draft bars of the step's windows that take the model's own drafts.
    bars: Vec<f32>,
    /// What every row of the step is worth in milliseconds (the record's row
    /// worth).
    worth: Vec<f32>,
    /// The window a step of the warm-up left uncommitted.
    pending_probe: Option<Commit>,
    /// What the warm-up measured, for the snapshot.
    calibration: Option<Calibration>,
    output: StepOutput,
    scratch_tokens: Vec<u32>,
    /// How the wanted rows of the step are picked, in their order, and the
    /// corrections and masks they name.
    samplers: Vec<SampleRow>,
    step_rows: StepRows,
    /// The words of one mask.
    mask_words: usize,
    /// What requests share that takes a pass over the vocabulary to find.
    sampling_context: Context,
    /// The decisions held, by their ids ([`decisions`]).
    decisions: HashMap<RequestId, Decision>,
    /// The errors the requests of decisions were ended with, by the
    /// request's id, until the decision is ended with it.
    part_errors: Arc<std::sync::Mutex<Vec<(RequestId, EngineError)>>>,
    /// The logits the record's scores name, for the step being written:
    /// `(wanted row, token)`.
    score_pairs: Vec<(u32, u32)>,
}

impl Conductor {
    pub(crate) fn new(
        model: Arc<dyn Model>,
        config: RuntimeConfig,
        mut exec: Box<dyn Executor>,
        rx: Receiver<Command>,
    ) -> Self {
        let asks = exec.asks();
        let max_rows = match config.max_rows {
            0 => model.max_rows(),
            n => n.min(model.max_rows()),
        };
        let slots = config.max_slots as usize;
        let share = Share::new(config.generation_floor);
        let drafters = model.drafters();
        // A step that waits for what it fetches is dear because of what it
        // fetched, not because the machine was busy.
        let drafting = Drafting::new(drafters.len(), config.window_rows);
        let estimate =
            if model.steps_wait_for_fetches() { Estimate::Mean } else { Estimate::WithoutLargest };
        let admission = AdmissionConfig {
            output_clip: config.output_clip,
            initial_ratio: config.output_ratio,
            max_running: config.max_slots,
            ..AdmissionConfig::new(config.pages, config.cells_per_page)
        };
        let vision = model.vision();
        let stage_rows =
            vision.map_or(0, |v| (exec.image_stage().len() / v.width.max(1) as usize) as u32);
        Self {
            vision,
            images: ImageStore::new(config.image_rows),
            pass: None,
            unit_ms_per_row: None,
            stage_rows,
            image_runs: Vec::new(),
            staged: Vec::new(),
            spec: model.context(),
            context_length: model.card().context_length,
            span: model.attention_span(),
            pool: PagePool::new(config.pages, config.cells_per_page),
            cache: model.prefix_cache().map(|spec| {
                let durable = model.durable(config.cells_per_page);
                Cache::new(&model.card().id, &spec, durable.as_ref(), &config)
                    .with_reach(model.prompt_rows().min(max_rows))
                    .with_ram(exec.ram())
                    .with_far(model.far_pages())
            }),
            admission: Admission::new(admission),
            slots: (0..slots).map(|_| None).collect(),
            free_slots: (0..config.max_slots).rev().collect(),
            waiting: Vec::new(),
            by_id: HashMap::with_capacity(slots),
            next_seq: 0,
            corpus: NgramCorpus::new(config.ngram),
            stats: Stats::new(exec.devices(), config.stats_window),
            output: StepOutput {
                paused: false,
                picks: Vec::with_capacity(max_rows as usize),
                logprobs: Vec::new(),
                scores: Vec::new(),
                drafts: Vec::with_capacity(max_rows as usize),
                draft_probs: Vec::with_capacity(max_rows as usize),
                draft_ms: 0.0,
                inexact: Vec::new(),
                devices: Vec::with_capacity(exec.devices()),
                step_ms: None,
            },
            samplers: Vec::with_capacity(max_rows as usize),
            step_rows: StepRows::default(),
            mask_words: TokenMask::words_for(model.vocab()),
            sampling_context: Context::default(),
            decisions: HashMap::new(),
            part_errors: Arc::default(),
            score_pairs: Vec::new(),
            requests: Vec::with_capacity(slots),
            parts: Vec::with_capacity(max_rows as usize),
            updates: Vec::with_capacity(slots),
            freed: Vec::with_capacity(record::FREED_PER_RECORD),
            runs: Vec::with_capacity(max_rows as usize),
            commits: Vec::with_capacity(slots),
            pending_probe: None,
            calibration: None,
            scratch_tokens: Vec::with_capacity(64),
            model,
            config,
            max_rows,
            clock: exec.clock(),
            exec,
            rx,
            active: 0,
            prices: PriceTable::with_estimate(estimate),
            drafting,
            drafters,
            bars: Vec::with_capacity(max_rows as usize),
            worth: Vec::with_capacity(max_rows as usize),
            share,
            prompt_rows: Vec::with_capacity(slots),
            round_tokens: 0,
            paused: None,
            paused_images: Vec::new(),
            give_up: false,
            asks,
            asked: 0,
            inner_due: Vec::new(),
            inner: Vec::new(),
            paused_inner: Vec::new(),
            paused_due: Vec::new(),
            step_no: 0,
            after_idle: true,
            last_done_ns: None,
            failed: None,
            failure: std::sync::Arc::default(),
            shutting_down: false,
            stalled: false,
            heartbeat: Arc::new(Heartbeat::new()),
        }
    }

    /// The loop. Returns after a shutdown, when every request has ended.
    pub(crate) fn run(mut self) {
        loop {
            // Between steps, in this order. First the levels of the cache
            // below the devices move what is due: what left the devices goes
            // into RAM, what leaves RAM goes to the drive (a checkpoint out of
            // its place first), and what comes back from the drive takes
            // pages of the pool and a place. Then what the page-locked memory
            // has no room for any more goes, and only now are the places of
            // checkpoints that went called unused, so that no transfer reads
            // or writes memory that is given back. Every transfer of the pump
            // has returned by then, and no step runs.
            let moving = match &mut self.cache {
                Some(cache) => {
                    cache.pump(&mut self.pool, &mut *self.exec);
                    let dropped = cache.trim(&mut self.pool, &mut *self.exec);
                    self.stats.counters.checkpoints_dropped += u64::from(dropped);
                    cache.busy()
                }
                None => false,
            };
            if self.active == 0 && !self.shutting_down {
                // Nothing runs. With nobody in line either, sleep until
                // somebody asks for something.
                let command = if self.waiting.is_empty() && !moving {
                    self.rx.recv().map(Some).map_err(|_| RecvTimeoutError::Disconnected)
                } else if self.stalled || self.waiting.is_empty() {
                    self.rx.recv_timeout(STALLED).map(Some)
                } else {
                    // The last running request has just ended: the next in
                    // line starts in this turn.
                    Ok(None)
                };
                match command {
                    Ok(Some(command)) => {
                        self.after_idle = true;
                        self.handle(command);
                    }
                    Ok(None) | Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            let began = Instant::now();
            // What was asked before the commands are taken: a request that comes after
            // this makes a step that pauses pause.
            if let Some(asks) = &self.asks {
                self.asked = asks.load(std::sync::atomic::Ordering::Acquire) as u32;
            }
            while let Ok(command) = self.rx.try_recv() {
                self.handle(command);
            }
            if self.shutting_down {
                break;
            }
            self.stats.record(Phase::Intake, began.elapsed().as_nanos() as u64);
            if self.active == 0 && self.waiting.is_empty() {
                continue;
            }
            self.step(began);
        }
        self.finish_everything(FinishReason::Cancelled);
        // `Runtime::shutdown` stops the front end from sending before it
        // queues the shutdown command, so nothing can arrive after it; whatever
        // was queued behind it is ended here all the same.
        while let Ok(command) = self.rx.try_recv() {
            self.handle(command);
        }
        self.finish_everything(FinishReason::Cancelled);
        // What is worth keeping on SSD is written before the devices go: what
        // only RAM holds, and the pages and checkpoints of the devices.
        if let (Some(cache), None) = (&mut self.cache, &self.failed) {
            cache.drain(&mut self.pool, &mut *self.exec);
        }
        self.exec.shutdown();
    }

    fn handle(&mut self, command: Command) {
        match command {
            Command::Submit(submission) => self.enqueue(*submission),
            Command::Decide(submission) => self.take_decision(*submission),
            Command::Cancel(id) if self.decisions.contains_key(&id) => {
                self.end_decision(id, ares_api::DecisionOutcome::Cancelled)
            }
            Command::Cancel(id) => match self.by_id.get(&id) {
                Some(Place::Slot(slot)) => self.retire(*slot, FinishReason::Cancelled),
                Some(Place::Waiting) => {
                    if let Some(i) = self.waiting.iter().position(|a| a.id == id) {
                        self.dismiss(i, FinishReason::Cancelled);
                    }
                }
                None => {}
            },
            Command::Stats { reset, reply } => {
                self.stats.counters.device_wakes = self.exec.device_wakes();
                if let Some(cache) = &mut self.cache {
                    self.stats.counters.ssd = cache.ssd_counters();
                    self.stats.counters.ram = cache.ram_counters();
                }
                let copy = StatsCopy {
                    stats: self.stats.clone(),
                    calibration: self.calibration.clone(),
                    step_ms: self.prices.points().collect(),
                    chain_ms: self.drafting.chains.iter().map(|c| c.points().collect()).collect(),
                    pages_free: self.pool.free_pages(),
                    pages_capacity: self.pool.capacity(),
                    cached_pages: self.cache.as_ref().map_or(0, |cache| cache.index.len() as u32),
                    checkpoint_ram: self.exec.checkpoint_ram(),
                    active: self.active as u32,
                    waiting: self.waiting.len() as u32,
                    ssd_problem: self.cache.as_ref().and_then(|cache| cache.ssd_problem.clone()),
                };
                if reset {
                    self.stats.reset();
                }
                let _ = reply.send(copy);
            }
            Command::Inspect { reply } => {
                let _ = reply.send(self.inspect());
            }
            // The loop trims between steps, at its top; this woke it.
            Command::Trim => {}
            Command::Shutdown => self.shutting_down = true,
        }
    }

    fn inspect(&self) -> Vec<SequenceInfo> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, a)| a.as_ref().map(|a| (slot, a)))
            .map(|(slot, a)| SequenceInfo {
                id: a.id,
                slot: slot as u32,
                tokens: a.tokens.len() as u32,
                prompt_tokens: a.prompt_len,
                cells: a.cells.len(),
                pages: a.cells.pages().iter().map(|p| p.0).collect(),
            })
            .collect()
    }

    /// Where the conductor writes why it accepts no more requests, once its
    /// devices have failed.
    pub(crate) fn failure(&self) -> std::sync::Arc<std::sync::OnceLock<String>> {
        self.failure.clone()
    }

    /// When the step in flight began, for the watchdog.
    pub(crate) fn heartbeat(&self) -> Arc<Heartbeat> {
        self.heartbeat.clone()
    }

    /// One pass of the admission accounting: the pages the parts in
    /// `self.parts` claim, then who starts.
    fn settle(&mut self) -> Outcome {
        let mut claims = self.admission.begin(self.pool.free_pages());
        // The sequence of a paused step is in the middle of its rows.
        if let Some(p) = self.paused {
            claims.keep(p.seq);
        }
        for part in &self.parts {
            let Some(a) = &self.slots[part.slot as usize] else { continue };
            // A replay writes no cell, so it claims no page.
            let writes = if part.replay { 0 } else { part.rows };
            // A prompt's rows and a sequence's newest token must run; draft
            // rows are given up before anybody has to wait.
            let (must, prompt_rows) = match part.kind {
                PartKind::Prefill { .. } => (writes, writes),
                PartKind::Window | PartKind::Score { .. } => (1, 0),
            };
            let required = a.cells.pages_to_append(&self.pool, u64::from(must));
            let all = a.cells.pages_to_append(&self.pool, u64::from(writes));
            claims.claim(part.seq, required, all.saturating_sub(required), prompt_rows);
        }
        let index = self.cache.as_mut().map(|cache| &mut cache.index);
        claims.settle(&mut View { pool: &self.pool, slots: &self.slots, index })
    }

    /// Gives up `pages` cached pages that no sequence holds, from the ends of
    /// chains, and the checkpoints that hung on them.
    fn evict(&mut self, pages: u32) {
        let Some(cache) = &mut self.cache else { return };
        let given = cache.evict(&mut self.pool, pages);
        self.stats.counters.cache_pages_evicted += u64::from(given.pages);
        self.stats.counters.checkpoints_dropped += u64::from(given.checkpoints);
    }

    /// Tells the accounting what the cache has now for the requests in line,
    /// where that changed: a held request whose page was published or given
    /// up, and every request after pages were evicted.
    fn refresh_waiting(&mut self) {
        let Some(cache) = &mut self.cache else { return };
        for a in &mut self.waiting {
            let changed = match a.waits_for {
                Some(hash) => !matches!(cache.index.state(hash), PageState::Computing(_)),
                None => a.looked_at != cache.evictions,
            };
            if !changed {
                continue;
            }
            a.looked_at = cache.evictions;
            // Before the step is planned: what RAM has comes back now, once.
            let names = &a.names[..a.known as usize];
            let exec: Option<&mut dyn Executor> =
                if a.brought_back { None } else { Some(&mut *self.exec) };
            let probe = cache.probe(&mut self.pool, exec, &a.chain, names);
            a.brought_back |= probe.brought > 0;
            if a.ran {
                // It gave way and comes back: the rows the cache does not hold.
                let rows = probe.resume.compute.end.saturating_sub(probe.resume.compute.start);
                self.admission.set_recompute(a.seq, rows as u32, probe.idle);
                continue;
            }
            self.admission.refresh(a.seq, probe.resume.cached_cells as u32, probe.idle);
            a.waits_for = probe.found.computing.map(|(hash, _)| hash);
            if a.waits_for.is_some() {
                self.admission.hold(a.seq);
            }
        }
    }

    /// Carries out who gives way and who starts.
    fn seat(&mut self, outcome: &Outcome) {
        for seq in &outcome.spill {
            self.suspend(*seq);
        }
        self.stats.counters.requests_outranked += u64::from(outcome.outranked);
        for seq in outcome.resume.iter().chain(&outcome.admit) {
            self.start(*seq);
        }
        // The requests that start have taken the cached pages they share, so
        // what is given up now is not theirs.
        if outcome.evict > 0 {
            self.evict(outcome.evict);
        }
    }

    /// One turn of the loop with something to do.
    fn step(&mut self, began: Instant) {
        let t_plan = Instant::now();
        // Those who wait come in before the plan, so that a request that
        // fits shares this step.
        self.parts.clear();
        self.inner_due.clear();
        if !self.waiting.is_empty() {
            self.refresh_waiting();
            let outcome = self.settle();
            self.seat(&outcome);
            for a in self.waiting.iter_mut().filter(|a| !a.counted_waiting) {
                a.counted_waiting = true;
                self.stats.counters.requests_waited += 1;
            }
        }
        self.stalled = self.active == 0;
        if self.stalled {
            return;
        }
        // A step the model paused goes on alone when nobody else runs.
        let paused = self.paused;
        if paused.is_some() && self.active == 1 {
            return self.resume(began, 0);
        }
        let paused_slot = paused.map(|p| p.slot as usize);
        // Whose turn it is. The price of a piece for the share is a step of
        // that many rows at the depth of the deepest prompt. The paused step's
        // rows are a prompt's, still to read.
        self.prompt_rows.clear();
        self.prompt_rows.extend(self.slots.iter().enumerate().filter_map(|(slot, a)| {
            let a = a.as_ref()?;
            match paused {
                Some(p) if Some(slot) == paused_slot => Some(p.rows),
                _ => (!a.generating()).then(|| a.prefill_left()),
            }
        }));
        let generating = self.generating_besides(paused_slot);
        let deepest = self
            .slots
            .iter()
            .flatten()
            .filter(|a| !a.generating())
            .map(|a| a.cells.len())
            .max()
            .unwrap_or(0);
        let (span, prices) = (self.span, &self.prices);
        // A piece is what a step for prompts takes while another request
        // waits: within the model's own memory, and as large as fits the time
        // bound by the price of a step (halved until it does; a price not yet
        // measured fits).
        let mut piece = self.model.prompt_rows().min(self.model.own_rows()).min(self.max_rows);
        while piece > MIN_PIECE
            && prices.ms(load_of(span, deepest, piece)).is_some_and(|ms| ms > self.config.piece_ms)
        {
            piece = (piece / 2).max(MIN_PIECE);
        }
        // A prompt of a better priority than every sequence that generates
        // owes them nothing.
        let best = |generating: bool| {
            let slots = self.slots.iter().flatten();
            slots.filter(|a| a.generating() == generating).map(|a| a.priority).min()
        };
        let ahead = matches!((best(false), best(true)), (Some(p), Some(g)) if p < g);
        let pending = Pending {
            generating,
            prompts: &self.prompt_rows,
            piece_rows: piece,
            max_rows: self.max_rows,
            ahead,
        };
        let turn = self
            .share
            .next(&pending, |rows| prices.ms(load_of(span, deepest, rows)).unwrap_or(0.0));
        // A piece starts with the image tower's work for the prompts that wait
        // for an image's rows, charged as prompt work.
        if let Turn::Piece { .. } = turn
            && self.vision.is_some()
        {
            match self.run_tower(generating) {
                Ok(ms) if ms > 0.0 => {
                    let crowded = !self.waiting.is_empty();
                    self.share.prompts_ran(ms, 0.0, generating, crowded);
                }
                Ok(_) => {}
                Err(message) => return self.fail(message),
            }
        }
        // The paused step's turn for prompts, when no other prompt waits for a
        // piece: it goes on for about the time of a piece.
        if let (Some(p), Turn::Piece { .. }) = (paused, turn) {
            let others = self.slots.iter().enumerate().any(|(slot, a)| {
                slot != p.slot as usize
                    && a.as_ref().is_some_and(|a| !a.generating() && a.prefill_left() > 0)
            });
            if !others {
                return self.resume(began, self.config.piece_ms.max(1.0) as u32);
            }
        }
        self.want(turn);
        let drafting =
            self.requests.iter().any(|r| matches!(r.want, Want::Window { drafts } if drafts > 0));
        let mut limits = Limits {
            max_rows: if drafting {
                self.max_rows.min(self.model.draft_rows())
            } else {
                self.max_rows
            },
            prefill_rows_with_windows: match turn {
                Turn::Piece { rows } => rows,
                Turn::Round => 0,
            },
            starvation_steps: self.config.starvation_steps,
        };
        // A long prompt next to generating sequences, for a model that pauses its
        // wide steps (a prompt read layer by layer): the prompt's step is as wide
        // as when it is read alone and has its rows alone, and it pauses once the
        // time of a piece has passed; the rounds run between its pieces. Every
        // expert is then read once for all its rows, and not once a piece.
        // The step borrows VRAM for its rows from the expert slots, which the rounds
        // between its pieces then do without: it is half as wide as the widest, so
        // that the slots keep at least half of what they could lend (a pass of all the
        // rows they can lend for stalled Flash-Next's rounds on one card).
        let wide_rows = self.max_rows / 2;
        let wide = Limits { max_rows: wide_rows, prefill_rows_with_windows: wide_rows, ..limits };
        let pass = matches!(turn, Turn::Piece { .. })
            && generating > 0
            && paused.is_none()
            && wide_rows > piece
            && self.model.pauses_wide_steps()
            && {
                let prompts: Vec<Request> = self
                    .requests
                    .iter()
                    .filter(|r| matches!(r.want, Want::Prefill { .. }))
                    .copied()
                    .collect();
                // The prompt the planner reads first, when it has more than a
                // piece left.
                match plan(&prompts, &wide).parts.first() {
                    Some(first) if first.rows > piece => {
                        self.requests.retain(|r| r.seq == first.seq);
                        true
                    }
                    _ => false,
                }
            };
        if pass {
            limits = wide;
        }
        let planned = plan(&self.requests, &limits);
        for request in &self.requests {
            if let Some(a) = self.slots[request.seq.0 as usize].as_mut() {
                a.waited_steps = a.waited_steps.saturating_add(1);
            }
        }
        let mut stage_room = self.stage_rows;
        for part in &planned.parts {
            let Some(a) = &self.slots[part.seq.0 as usize] else { continue };
            let mut step = StepPart {
                slot: part.seq.0,
                seq: a.seq,
                rows: part.rows,
                kind: part.kind,
                from: a.state_pos(),
                replay: false,
            };
            if let PartKind::Prefill { .. } = part.kind {
                // Replayed rows and rows that write are steps of their own: a
                // segment is one or the other. And a step ends where a
                // checkpoint is due, so that the state is exactly there.
                step.replay = a.replay_left > 0;
                if step.replay {
                    step.rows = step.rows.min(a.replay_left);
                }
                // Up to an image whose rows are not there, and as many image
                // rows as the stage has room for; before the checkpoints, so
                // that those due inside the step lie inside the rows it keeps.
                if !a.spans.is_empty() {
                    step.rows = a.ready_rows(step.from, step.rows, &self.images);
                    step.rows = a.rows_within(step.from, step.rows, stage_room).0;
                    if step.rows == 0 {
                        continue;
                    }
                }
                let to = step.from + u64::from(step.rows);
                // A model that saves the state inside a step wider than its own rows (a
                // prompt read layer by layer) takes the checkpoints due inside its rows there,
                // and the step goes on.
                let margin = self.model.saves_inside_steps();
                let inside = !step.replay && margin > 0 && step.rows > self.model.own_rows();
                let named = self.inner_due.len();
                if let Some(cache) = &self.cache {
                    let (mut last, mut from) = (a.checkpoint_at, step.from);
                    while let Some(at) = cache.capture(a.prompt_len, &a.boundaries, last, from, to)
                    {
                        if inside
                            && at >= step.from + u64::from(margin)
                            && at + u64::from(margin) <= to
                        {
                            self.inner_due.push((step.slot, at));
                            (last, from) = (at, at);
                            continue;
                        }
                        step.rows = step.rows.min((at - step.from) as u32);
                        break;
                    }
                }
                // A cut after them may have left the saves inside the step too close to its
                // end, or the step no wider than the model's own rows (no pass, which saves
                // nothing inside): it then ends at the first of them instead.
                let end = step.from + u64::from(step.rows);
                if let Some(&(_, at)) = self.inner_due.get(named)
                    && (step.rows <= self.model.own_rows()
                        || self.inner_due[named..].iter().any(|s| s.1 + u64::from(margin) > end))
                {
                    step.rows = (at - step.from) as u32;
                    self.inner_due.truncate(named);
                }
                // The image rows the step stages, of the stage's room.
                if !a.spans.is_empty() {
                    stage_room -= a.rows_within(step.from, step.rows, stage_room).1;
                }
                let completes =
                    !step.replay && a.cells.len() + u64::from(step.rows) == u64::from(a.known);
                step.kind = PartKind::Prefill { completes };
            }
            self.parts.push(step);
        }

        // The pages this step writes. Draft rows that do not fit are
        // dropped; when the rows that must run do not fit, somebody gives
        // way.
        let outcome = self.settle();
        self.stats.record(Phase::Plan, t_plan.elapsed().as_nanos() as u64);

        let t_write = Instant::now();
        if let Some(slot) = outcome.exhausted.and_then(|seq| {
            self.slots.iter().position(|a| a.as_ref().is_some_and(|a| a.seq == seq))
        }) {
            // The pool holds this one sequence and has no page left for it:
            // the request ends where a full context would end it.
            if let Some(a) = &mut self.slots[slot] {
                a.stop.flush(&mut a.emitted);
            }
            self.stats.counters.requests_exhausted += 1;
            self.retire(slot as u32, FinishReason::Length);
        }
        self.seat(&outcome);
        let mut kept = 0;
        for i in 0..self.parts.len() {
            let mut part = self.parts[i];
            // A sequence that gave way or ended has no part in the step, and
            // one that took its slot in this pass has none yet.
            let Some(a) = self.slots[part.slot as usize].as_mut().filter(|a| a.seq == part.seq)
            else {
                continue;
            };
            a.waited_steps = 0;
            if outcome.cut.contains(&part.seq) {
                self.stats.counters.draft_rows_cut += u64::from(part.rows - 1);
                part.rows = 1;
            }
            let writes = if part.replay { 0 } else { u64::from(part.rows) };
            let mut grown = a.cells.append(&mut self.pool, writes);
            if let (Err(short), Some(cache)) = (&grown, &mut self.cache) {
                // The accounting counted on cached pages that turned out to be
                // held by a sequence: give up others.
                let given = cache.evict(&mut self.pool, short.needed.saturating_sub(short.free));
                self.stats.counters.cache_pages_evicted += u64::from(given.pages);
                self.stats.counters.cache_pages_evicted_late += u64::from(given.pages);
                self.stats.counters.checkpoints_dropped += u64::from(given.checkpoints);
                grown = a.cells.append(&mut self.pool, writes);
            }
            match grown {
                Ok(grown) => {
                    if let Some(copy) = grown.copy {
                        // The sequence writes into a page that others read, so it
                        // writes into a copy.
                        let cache = self.cache.as_mut().expect("only a cache shares pages");
                        cache.copy_page(copy.from, copy.to, copy.cells);
                        self.stats.counters.page_copies += 1;
                    }
                    a.draft.truncate(part.rows.saturating_sub(1) as usize);
                    if a.draft.is_empty() {
                        a.model_drafter = None;
                    }
                    self.parts[kept] = part;
                    kept += 1;
                }
                Err(error) => {
                    // The accounting granted these pages, so this is a
                    // bookkeeping bug and not load.
                    let _ = a.sink.send(Event::Error(EngineError::Internal(format!(
                        "the page pool ran out under a granted claim: {error}"
                    ))));
                    self.retire(part.slot, FinishReason::Cancelled);
                }
            }
        }
        self.parts.truncate(kept);
        if self.parts.is_empty() {
            // Nothing else could run: the paused step goes on.
            if self.paused.is_some() {
                self.resume(began, self.config.piece_ms.max(1.0) as u32);
            }
            return;
        }

        // The places of the checkpoints due inside the step's rows: taking them may give
        // other checkpoints up, which leave the devices below.
        self.reserve_inner();
        // What left the devices while the step was planned is copied into
        // RAM before a step writes its pages again: the step that only drafts
        // writes the cells of the MTP block's chains.
        if let Some(cache) = &mut self.cache {
            cache.settle_leaving(&mut self.pool, &mut *self.exec);
        }
        // The model's drafters run first, as a step of their own, and the
        // round verifies what the conductor takes of their drafts.
        let chain_ms = if self.parts.iter().any(|p| self.takes_model_drafts(p)) {
            match self.chains_first() {
                Ok(ms) => ms,
                Err(message) => return self.fail(message),
            }
        } else {
            0.0
        };
        let pause_ms = if pass { self.config.piece_ms.max(1.0) as u32 } else { 0 };
        let used = match self.write_record(RecordKind::Round { pause_ms }) {
            Ok(used) => used,
            Err(message) => return self.fail(message),
        };
        self.stats.record(Phase::Write, t_write.elapsed().as_nanos() as u64);
        self.run_record(began, used, turn, generating, chain_ms, true);
    }

    /// The sequences that generate, besides the one in slot `besides` (the
    /// paused step's, which runs alone).
    fn generating_besides(&self, besides: Option<usize>) -> u32 {
        let slots = self.slots.iter().enumerate();
        slots
            .filter(|(slot, a)| {
                Some(*slot) != besides && a.as_ref().is_some_and(|a| a.generating())
            })
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn characters_come_out_whole_whatever_the_tokens_cut() {
        // "aü€" arriving a byte at a time: nothing comes out in the middle of a
        // character, and nothing is lost.
        let text = "a\u{fc}\u{20ac}";
        let mut pending = Vec::new();
        let mut out = String::new();
        for (i, byte) in text.bytes().enumerate() {
            pending.push(byte);
            assert!(!take_characters(&mut pending, |ch| {
                out.push(ch);
                false
            }));
            assert!(text.starts_with(&out), "after byte {i}: {out:?}");
        }
        assert_eq!(out, text);
        assert!(pending.is_empty());
        // A byte that starts nothing is replaced, and what follows it is kept.
        let mut pending = vec![b'x', 0xff, b'y', 0xe2, 0x82];
        let mut out = String::new();
        take_characters(&mut pending, |ch| {
            out.push(ch);
            false
        });
        assert_eq!(out, "x\u{fffd}y");
        assert_eq!(pending, [0xe2, 0x82], "the unfinished character waits");
        // The stop is reported and the characters before it are consumed.
        let mut pending = b"abc".to_vec();
        assert!(take_characters(&mut pending, |ch| ch == 'b'));
    }

    #[test]
    fn the_load_of_rows_follows_what_the_model_reads() {
        // Dense attention: row i of a chunk at 100 reads 100 + i cells.
        assert_eq!(load_of(u32::MAX, 100, 3), Load::chunk(100, 3));
        // No attention: rows and no cells.
        assert_eq!(load_of(0, 100, 3), Load { rows: 3, cells: 0 });
        // A span of 10: rows at 8, 9 read their context, the rest the span.
        assert_eq!(load_of(10, 8, 5), Load { rows: 5, cells: 8 + 9 + 3 * 10 });
        assert_eq!(load_of(10, 50, 5), Load { rows: 5, cells: 50 });
    }
}
