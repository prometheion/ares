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

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

use ares_api::{EngineError, Event, EventSink, FinishReason, RequestId, Sampling, Usage};
use ares_sampler::{SampleParams, SampleRow, TokenMask};
use ares_sched::{
    Acceptance, Admission, AdmissionConfig, Arrival, Choice, Confidence, Curve, Estimate, Limits,
    Load, NgramCorpus, NgramDrafter, Offer, Outcome, PartKind, Pending, PoolView, PriceTable,
    Prices, Request, SeqId, Share, Turn, Want, Wish, choose, plan,
};
use ares_state::{
    Boundary, Chain, ChainHash, Owner, PagePool, PageState, PrefixIndex, Priority, Resume, Sequence,
};

use crate::cache::Cache;
use crate::config::RuntimeConfig;
use crate::exec::{Clock, ExecError, Executor, StepMeta, StepOutput};
use crate::image::{ImageInput, ImageName, ImageRows, ImageSpan, ImageStore, Spans, cell_name};
use crate::model::{ContextSpec, Drafter, DrafterKind, Model, TowerJob, Vision};
use crate::record::{self, Commit, ImageRun, RecordWriter, Row, TableUpdate};
use crate::sampling::{Context, RequestSampler, StepRows, TokenFilter};
use crate::timing::{Calibration, DRAFT_POSITIONS, Phase, RequestReport, Snapshot, Stats};
use crate::watch::Watch;

/// The drafters, as the choice of a window and the step record number them:
/// the n-gram drafter, which costs nothing to ask, is 0; the model's
/// drafters ([`Model::drafters`]) follow from 1 in the model's order, each
/// with the device time its drafts cost.
const NGRAM: usize = 0;

/// What a round leaves of the measured acceptance of the model's own drafter
/// for a sequence that was offered it and did not take it, and of the
/// engine's: a drafter that nobody asks measures nothing, and what it measured
/// when it was poor would keep it out for ever. The sequence's fades quickly
/// (it is the text that changed, usually), the engine's slowly.
const UNASKED_SEQUENCE: f64 = 0.995;
const UNASKED_ENGINE: f64 = 0.999;

/// What a chain of `links` links of the model's own drafter is expected to
/// take: link `j` costs what the curve says a chain of `j + 1` links takes
/// over one of `j`, and is made as often as the draft before it leaves the
/// chain above its bar (`passes[j]`; a drafter that knows its probabilities
/// ends a chain there). The price of the whole chain for a drafter that never
/// ends one early.
fn expected_chain_ms(chain: &Curve, passes: &[(f64, f64)], links: u32) -> f64 {
    let (mut ms, mut reach, mut before) = (0.0, 1.0, 0.0);
    for j in 0..links {
        let Some(upto) = chain.trial_ms(j + 1) else { return 0.0 };
        reach *= passes.get(j as usize).map_or(1.0, |(made, _)| *made);
        ms += reach * (upto - before);
        before = upto;
    }
    ms
}

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
            self.stop.token();
            if model.ends(token) {
                self.stop.flush(&mut self.emitted);
                self.sampler.reply_ended();
                self.finished = Some(FinishReason::Stop);
                return;
            }
            let previous = self.states.last().copied().unwrap_or_else(|| spec.start());
            self.tokens.push(token);
            self.names.push(u64::from(token));
            self.states.push(spec.advance(previous, &self.tokens));
            self.drafter.push(corpus, token);
            let piece_at = self.pending.len();
            model.piece(token, &mut self.pending);
            if self.sampler.wants_probs() {
                self.sampler.token_taken(model, i, token, &self.pending[piece_at..]);
            }
            let (stop, emitted) = (&mut self.stop, &mut self.emitted);
            if take_characters(&mut self.pending, |ch| stop.push(ch, emitted)) {
                self.finished = Some(self.stop.finish());
                return;
            }
            // A reply caught in a loop ends as a limit ends it (`guard`).
            if self.sampler.cuts(&self.tokens, self.prompt_len as usize) {
                self.stop.flush(&mut self.emitted);
                self.sampler.reply_ended();
                self.finished = Some(FinishReason::Length);
                return;
            }
            if self.max_tokens.is_some_and(|max| self.completion >= max)
                || self.tokens.len() as u32 >= context_length
            {
                self.stop.flush(&mut self.emitted);
                self.sampler.reply_ended();
                self.finished = Some(FinishReason::Length);
                return;
            }
        }
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
        while let Ok(command) = self.rx.try_recv() {
            if let Command::Submit(submission) = command {
                let _ = submission.sink.send(Event::Error(gone.clone()));
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

/// The load of `rows` consecutive rows of a sequence that start at position
/// `context`, for a model whose rows read at most `span` cells each.
/// The fewest rows a piece of prompt is cut to by its time bound
/// ([`RuntimeConfig::piece_ms`]): the smallest step of prompt the warm-up
/// measures.
const MIN_PIECE: u32 = 16;

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
    /// The model's drafters, in its order.
    drafters: Vec<Drafter>,
    /// Per model drafter: what its drafts cost by their number, in
    /// milliseconds of the step that only drafts.
    chains: Vec<Curve>,
    /// Per model drafter: how its drafts have been accepted by every
    /// sequence, per position (where a new sequence's estimate starts), and
    /// by the probability it gave them (what a chain is cut by).
    engines: Vec<Acceptance>,
    confidences: Vec<Confidence>,
    /// Whose turn it is, rounds of generation or pieces of prompt.
    share: Share,
    /// The rows of each prompt that has started and is not read yet, for the
    /// share's decision.
    prompt_rows: Vec<u32>,
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
    /// The milliseconds the next step may run before the model pauses it, and whether the
    /// record being written goes on with a paused step (its rows were counted).
    pause_ms: u32,
    resuming: bool,
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
    /// worth), and what a token costs at the rate the round was chosen for.
    worth: Vec<f32>,
    token_ms: f64,
    /// Per model drafter, the bar of each link after the first, as the last
    /// round set them.
    needs: Vec<Vec<f32>>,
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
        let share = Share::new(config.prefill_slowdown);
        let drafters = model.drafters();
        let engines =
            drafters.iter().map(|_| Acceptance::new(config.window_rows as usize, 0.999)).collect();
        let needs = drafters.iter().map(|_| Vec::new()).collect();
        // A step that waits for what it fetches is dear because of what it
        // fetched, not because the machine was busy.
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
                Cache::new(
                    &model.card().id,
                    &spec,
                    durable.as_ref(),
                    &config,
                    config.cells_per_page,
                )
                .with_reach(model.prompt_rows().min(max_rows))
                .with_ram(exec.ram())
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
            chains: drafters.iter().map(|_| Curve::new()).collect(),
            engines,
            confidences: drafters.iter().map(|_| Confidence::new(0.999)).collect(),
            drafters,
            bars: Vec::with_capacity(max_rows as usize),
            worth: Vec::with_capacity(max_rows as usize),
            token_ms: 0.0,
            needs,
            share,
            prompt_rows: Vec::with_capacity(slots),
            paused: None,
            paused_images: Vec::new(),
            give_up: false,
            pause_ms: 0,
            resuming: false,
            asks,
            asked: 0,
            inner_due: Vec::new(),
            inner: Vec::new(),
            paused_inner: Vec::new(),
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

    /// Runs steps before the first request: a few of one row, so lazily
    /// loaded kernels and first-use costs are paid, and then steps of every
    /// row count a window can have and of the sizes a prompt's piece comes
    /// in, which measure what a step costs on these devices (and, for a model
    /// with an MTP block, what each length of its chain costs). The
    /// measurements are the table that drafts and pieces are decided from;
    /// every step that runs corrects it afterwards. The steps work in slot 0
    /// on the first pages of the pool, which the first request overwrites.
    pub(crate) fn warm_up(&mut self) -> Result<(), ExecError> {
        let (begun, first_step) = (Instant::now(), self.step_no);
        for _ in 0..3 {
            self.probe(1, Probe::Prompt)?;
        }
        let window = if self.drafters.iter().any(|d| d.drafts > 0) {
            self.model.draft_rows().min(self.max_rows)
        } else {
            0
        };
        // The rows of the windows and of the rounds without drafts that the
        // sequences make up together, then the doublings up to a piece of
        // prompt. Prompts are read a piece at a time, so nothing above is run.
        let piece = self.model.prompt_rows().min(self.max_rows);
        let capacity = self.config.pages.saturating_mul(self.config.cells_per_page);
        let top = piece.min(capacity).min(self.context_length);
        let mut sizes: Vec<u32> = (1..=window.max(self.config.max_slots.min(8)).min(top)).collect();
        let mut rows = 16;
        while rows < top {
            if rows > *sizes.last().expect("one row at least") {
                sizes.push(rows);
            }
            rows *= 2;
        }
        if top > *sizes.last().expect("one row at least") {
            sizes.push(top);
        }
        // The fastest of two for the steps the policy chooses between, which
        // are short and many; the long ones are measured once. What other work
        // on the machine adds is not the price of a step, and the steps that
        // run correct the table anyway.
        for rows in sizes {
            let reps = if rows <= 24 { 2 } else { 1 };
            let shape = if rows <= window { Probe::Window } else { Probe::Prompt };
            let mut fastest = f64::INFINITY;
            for _ in 0..reps {
                fastest = fastest.min(self.probe(rows, shape)?.0);
            }
            self.prices.measured(rows, fastest);
        }
        // Each drafter of the model: steps that only draft, for windows of 2
        // rows to the longest one sequence has, whose first row is the newest
        // token and the others the drafts the drafter makes.
        for d in 0..self.drafters.len() {
            let longest =
                self.config.window_rows.min(window).min(top).min(self.drafters[d].drafts + 1);
            for links in 1..longest {
                let mut fastest = f64::INFINITY;
                for _ in 0..2 {
                    fastest = fastest.min(self.probe(links + 1, Probe::Chain(d))?.0);
                }
                self.chains[d].set(links, fastest);
            }
        }
        // The last probe left a window that nothing has committed.
        self.probe(1, Probe::Prompt)?;
        self.calibration = Some(Calibration {
            steps: (self.step_no - first_step) as u32,
            seconds: begun.elapsed().as_secs_f64(),
            step_ms: self.prices.points().collect(),
            chain_ms: self.chains.iter().map(|c| c.points().collect()).collect(),
        });
        self.stats.reset();
        Ok(())
    }

    /// One step of `rows` rows of the sequence in slot 0, from position 0,
    /// shaped as `shape` says (a chain is a step that only drafts, for a
    /// window of `rows` rows). Returns the time of the step in milliseconds
    /// and what the model's own draft chain took of it.
    fn probe(&mut self, rows: u32, shape: Probe) -> Result<(f64, f64), ExecError> {
        let cells = self.config.cells_per_page;
        let used = {
            let pages = rows.div_ceil(cells);
            let entries: Vec<u32> = (0..pages).map(|page| self.model.page_entry(page)).collect();
            let mut writer = RecordWriter::new(self.exec.record(), self.step_no + 1, 1, rows)
                .expect("room for the rows");
            // A window of one row is certain, as it is in a round.
            let deferred = shape != Probe::Prompt && rows > 1;
            writer.segment(0, 0, rows, !deferred).expect("room for a segment");
            if let (Probe::Chain(drafter), true) = (shape, deferred) {
                writer.drafts_only();
                writer.model_drafts(drafter as u32 + 1).expect("room for the flag");
            }
            for position in 0..rows {
                writer
                    .row(Row { slot: 0, position, token: 0, context: 0 })
                    .expect("room for a row");
            }
            if deferred {
                writer.runs(&[(0, rows)]).expect("room for a run");
                writer
                    .samplers((0..rows).map(|position| SampleParams::GREEDY.row(position)))
                    .expect("room for samplers");
            } else {
                writer.runs(&[(rows - 1, 1)]).expect("room for a run");
                writer.samplers([SampleParams::GREEDY.row(rows - 1)]).expect("room for a sampler");
            }
            // The window of the step before is committed by this one.
            let previous: Vec<Commit> = self.pending_probe.take().into_iter().collect();
            writer.commits(&previous).expect("room for a commit");
            writer
                .updates(
                    [TableUpdate { slot: 0, first: 0, count: pages }].into_iter(),
                    entries.into_iter(),
                )
                .expect("room for the page-table entries");
            writer.finish().expect("room for the header")
        };
        // A window leaves its rows for the next record to commit; a step that
        // only drafts leaves none.
        if shape == Probe::Window && rows > 1 {
            self.pending_probe = Some(Commit { row0: 0, rows, slot: 0, accepted: 1 });
        }
        self.step_no += 1;
        let begun = Instant::now();
        self.exec.dispatch(&StepMeta {
            step: self.step_no,
            record_bytes: used,
            after_idle: true,
        })?;
        self.exec.collect(self.config.conductor_spin, &mut self.output)?;
        let ms = begun.elapsed().as_secs_f64() * 1e3;
        // An executor that knows the time of its steps without a clock says so.
        let ms = self.output.step_ms.take().unwrap_or(ms);
        Ok((ms, self.output.draft_ms))
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
                    chain_ms: self.chains.iter().map(|c| c.points().collect()).collect(),
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

    /// A name for the admission accounting that no request the engine holds
    /// has.
    fn fresh_seq(&mut self) -> SeqId {
        loop {
            self.next_seq = self.next_seq.wrapping_add(1);
            let seq = SeqId(self.next_seq);
            let taken = |a: &Active| a.seq == seq;
            if !self.slots.iter().flatten().any(taken) && !self.waiting.iter().any(taken) {
                return seq;
            }
        }
    }

    /// Puts a new request in line. It starts at the next pass of the
    /// admission accounting in which it fits.
    fn enqueue(&mut self, submission: Submission) {
        let Submission {
            id,
            prompt,
            boundaries,
            images,
            max_tokens,
            stop,
            sampling,
            draft,
            filter,
            priority,
            displace,
            sink,
        } = submission;
        let refuse = |error: EngineError| {
            let _ = sink.send(Event::Error(error));
        };
        if let Some(why) = &self.failed {
            return refuse(EngineError::Internal(why.clone()));
        }
        if self.by_id.contains_key(&id) {
            return refuse(EngineError::InvalidRequest(format!("request id {id} is in use")));
        }
        if !images.is_empty() && self.vision.is_none() {
            return refuse(EngineError::InvalidRequest("the model reads no images".into()));
        }
        if !images.is_empty() && self.stage_rows == 0 {
            return refuse(EngineError::Internal(
                "the executor has no image stage for the rows of images".into(),
            ));
        }
        if displace {
            // The line is full: the worst request in it of a worse priority
            // that has not started (the latest of them) is refused in this
            // one's place. A request that ran has sent its client text, and
            // stays.
            let worst = self
                .waiting
                .iter()
                .enumerate()
                .filter(|(_, a)| !a.ran && a.priority > priority)
                .max_by_key(|(_, a)| (a.priority, a.stop.submitted()))
                .map(|(i, _)| i);
            let Some(i) = worst else { return refuse(EngineError::Overloaded) };
            let _ = self.waiting[i].sink.send(Event::Error(EngineError::Overloaded));
            self.dismiss(i, FinishReason::Cancelled);
            self.stats.counters.requests_displaced += 1;
        }
        // The names of the cells: an image's by the image and its row.
        let mut names: Vec<u64> = prompt.iter().map(|t| u64::from(*t)).collect();
        for span in &images {
            for j in 0..span.image.rows() {
                if let Some(name) = names.get_mut((span.first + j) as usize) {
                    *name = cell_name(&span.image.name, j);
                }
            }
        }
        let spans = Spans::new(images);
        // A request without a seed gets one of its own.
        let seed = (self.clock.now_ns() ^ id.0.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let sampler = match RequestSampler::new(
            &sampling,
            &*self.model,
            &mut self.sampling_context,
            seed,
            filter,
        ) {
            Ok(sampler) => sampler,
            Err(error) => return refuse(error),
        };
        let prompt_len = prompt.len() as u32;
        let seq = self.fresh_seq();
        let room = self.context_length.saturating_sub(prompt_len);
        // What the cache has for the prompt now. A request whose first
        // missing page another sequence is computing is held back.
        let (chain, cached, idle, waits_for, brought_back) = match &mut self.cache {
            Some(cache) => {
                let chain = cache.chain(&names);
                // What RAM has of it comes back now, before the step is
                // planned, so that the accounting sees the pages it took.
                let probe = cache.probe(&mut self.pool, Some(&mut *self.exec), &chain, &names);
                let waits = probe.found.computing.map(|(hash, _)| hash);
                (chain, probe.resume.cached_cells as u32, probe.idle, waits, probe.brought > 0)
            }
            None => {
                (Chain::new(ChainHash::root(&[]), self.config.cells_per_page), 0, 0, None, false)
            }
        };
        let arrival = Arrival {
            seq,
            prompt_cells: prompt_len,
            cached_cells: cached,
            idle_cached_pages: idle,
            max_new_tokens: max_tokens.map_or(room, |max| max.min(room)),
            priority,
        };
        if self.admission.enqueue(arrival).is_err() {
            // The one refusal: the whole pool cannot hold the prompt and the
            // first token after it.
            return refuse(EngineError::ContextOverflow {
                prompt_tokens: prompt_len,
                limit: self.pool.capacity().saturating_mul(self.pool.cells_per_page()),
            });
        }
        let mut states = Vec::with_capacity(prompt.len() + 64);
        let mut state = self.spec.start();
        for i in 0..prompt.len() {
            state = self.spec.advance(state, &prompt[..=i]);
            states.push(state);
        }
        let mut tokens = Vec::with_capacity(prompt.len() + 64);
        tokens.extend_from_slice(&prompt);
        if waits_for.is_some() {
            self.admission.hold(seq);
            self.stats.counters.requests_held += 1;
        }
        self.waiting.push(Active {
            id,
            seq,
            sink,
            cells: Sequence::new(),
            prompt_len,
            boundaries,
            known: prompt_len,
            chain,
            owner: Owner(u64::from(seq.0)),
            ran: false,
            replay_end: 0,
            replay_left: 0,
            checkpoint_at: 0,
            waits_for,
            was_held: waits_for.is_some(),
            brought_back,
            looked_at: self.cache.as_ref().map_or(0, |cache| cache.evictions),
            cached_prompt: 0,
            replayed: 0,
            computed: 0,
            tokens,
            names,
            spans,
            states,
            drafter: NgramDrafter::new(),
            model_acceptance: (0..self.drafters.len())
                .map(|_| Acceptance::new(self.config.window_rows as usize, 0.95))
                .collect(),
            draft_allowed: draft,
            sampler,
            stop,
            max_tokens,
            completion: 0,
            waited_steps: 0,
            priority,
            counted_waiting: false,
            draft: Vec::with_capacity(self.config.window_rows as usize),
            model_drafter: None,
            ngram: Vec::with_capacity(self.config.window_rows as usize),
            ngram_chances: Vec::with_capacity(self.config.window_rows as usize),
            offered: (0..self.drafters.len()).map(|_| Vec::new()).collect(),
            model_chances: Vec::with_capacity(self.config.window_rows as usize),
            model_needs: Vec::with_capacity(self.config.window_rows as usize),
            model_probs: Vec::with_capacity(self.config.window_rows as usize),
            pending: Vec::new(),
            emitted: String::new(),
            finished: None,
        });
        self.by_id.insert(id, Place::Waiting);
    }

    /// Gives the request with the name `seq` a slot on the devices: it runs
    /// from the next plan on.
    fn start(&mut self, seq: SeqId) {
        let Some(i) = self.waiting.iter().position(|a| a.seq == seq) else { return };
        // What the cache has for it now, which may be more or less than when
        // it was put in line.
        let probe = match &mut self.cache {
            Some(cache) => {
                let a = &self.waiting[i];
                Some(cache.probe(&mut self.pool, None, &a.chain, &a.names[..a.known as usize]))
            }
            None => None,
        };
        if let Some(probe) = &probe
            && let Some((hash, _)) = probe.found.computing
            && !self.waiting[i].ran
        {
            // A sequence that started a moment ago computes the page this
            // prompt misses first. The accounting let the request in, and
            // only sequences that run announce pages, so it goes back in
            // line and waits for the page.
            let a = &mut self.waiting[i];
            a.waits_for = Some(hash);
            if !a.was_held {
                a.was_held = true;
                self.stats.counters.requests_held += 1;
            }
            let arrival = Arrival {
                seq,
                prompt_cells: a.prompt_len,
                cached_cells: probe.resume.cached_cells as u32,
                idle_cached_pages: probe.idle,
                max_new_tokens: a.max_new_tokens(self.context_length),
                priority: a.priority,
            };
            self.admission.remove(seq);
            // It passed the one refusal when it arrived.
            let _ = self.admission.enqueue(arrival);
            self.admission.hold(seq);
            return;
        }
        let Some(slot) = self.free_slots.pop() else {
            // The accounting never starts more requests than there are
            // slots, so this is a bookkeeping bug and not load.
            let a = self.waiting.swap_remove(i);
            let _ = a.sink.send(Event::Error(EngineError::Internal(
                "a request was started without a free slot".into(),
            )));
            self.by_id.remove(&a.id);
            self.admission.remove(seq);
            return;
        };
        let mut a = self.waiting.swap_remove(i);
        a.stop.started();
        if let (Some(cache), Some(probe)) = (&mut self.cache, probe) {
            let counters = &mut self.stats.counters;
            let Resume { cached_cells: cached, restore, replay, .. } = probe.resume;
            a.cells = Sequence::from_shared(&mut self.pool, &probe.found.table(), cached);
            a.chain = cache.chain_after(&probe.found, &a.names[..a.known as usize]);
            // The pages this sequence will fill are its to announce.
            cache.index.announce(a.owner, a.chain.unpublished());
            a.replay_end = cached as u32;
            a.replay_left = (replay.end - replay.start) as u32;
            a.checkpoint_at = 0;
            if let Some(id) = restore {
                cache.restore(slot, id);
                a.checkpoint_at = replay.start;
                counters.checkpoints_restored += 1;
            }
            if cached > 0 {
                counters.prefix_hits += 1;
                counters.cached_cells += cached;
            }
            if !a.ran {
                a.cached_prompt = cached.min(u64::from(a.prompt_len)) as u32;
            }
        }
        a.ran = true;
        a.waits_for = None;
        self.by_id.insert(a.id, Place::Slot(slot));
        self.slots[slot as usize] = Some(a);
        self.active += 1;
    }

    /// Lets what a sequence computed stay in the prefix cache, as it ends,
    /// gives way or is cancelled: the pages that are full and not published
    /// yet, the partly used last page as a tail, and the live state as the
    /// checkpoint of that tail (or of the last full page, when the cells end
    /// on a page boundary), unless with one place it would push out the
    /// request's own anchor ([`Cache::save_end`]). The copy of the state runs
    /// with the next record, before anything of it writes a slot.
    fn keep(&mut self, a: &mut Active, slot: u32) {
        let Some(cache) = &mut self.cache else { return };
        cache.index.withdraw(a.owner);
        // A sequence that leaves before the record that restores its state
        // never had that state in its slot.
        let restored = cache.cancel_restore(slot);
        let cells = a.cells.len();
        let per_page = u64::from(self.pool.cells_per_page());
        cache.index.publish(
            &mut self.pool,
            &mut a.cells,
            &mut a.chain,
            &a.names,
            cells,
            Priority::NORMAL,
        );
        cache.note(a.owner, &a.chain, cells, true);
        let full = (cells / per_page) as usize;
        let rest = &a.names[full * per_page as usize..cells as usize];
        let target = if rest.is_empty() {
            full.checked_sub(1).map(|page| a.chain.hashes()[page])
        } else {
            let prior = a.chain.before(full);
            let page = a.cells.pages()[full];
            if let Some(evicted) =
                cache.index.insert_tail(&mut self.pool, prior, rest, page, Priority::NORMAL)
            {
                self.stats.counters.checkpoints_dropped += u64::from(cache.forget(evicted));
            }
            Some(PrefixIndex::tail_hash(prior, rest))
        };
        if let Some(hash) = target
            && !restored
            && a.state_pos() == cells
        {
            let saved = cache.save_end(slot, hash, a.owner);
            self.stats.counters.checkpoints_saved += u64::from(saved.taken);
            self.stats.counters.checkpoints_dropped += u64::from(saved.dropped);
        }
    }

    /// Takes the running request with the name `seq` off the devices: it
    /// lets go of its pages and its slot and waits at the front of the line.
    /// What it computed stays in the prefix cache (when the model has one), so
    /// it comes back as a prefix hit; without one it computes everything it has
    /// read and written again, as a prompt made of all its tokens.
    fn suspend(&mut self, seq: SeqId) {
        let Some(slot) = self.slots.iter().position(|a| a.as_ref().is_some_and(|a| a.seq == seq))
        else {
            return;
        };
        let Some(mut a) = self.slots[slot].take() else { return };
        self.keep(&mut a, slot as u32);
        std::mem::take(&mut a.cells).release(&mut self.pool);
        a.known = a.tokens.len() as u32;
        a.replay_left = 0;
        // What it computed may leave the devices while it waits: it may
        // bring it back from RAM once more.
        a.brought_back = false;
        a.draft.clear();
        a.ngram.clear();
        a.model_drafter = None;
        a.waited_steps = 0;
        // What it needs to come back: the rows the cache does not hold, which
        // are claimed again step by step, as a prompt's are.
        let (rows, idle) = match &mut self.cache {
            Some(cache) => {
                a.looked_at = cache.evictions;
                let names = &a.names[..a.known as usize];
                let probe = cache.probe(&mut self.pool, None, &a.chain, names);
                (
                    probe.resume.compute.end.saturating_sub(probe.resume.compute.start) as u32,
                    probe.idle,
                )
            }
            None => (a.known, 0),
        };
        self.admission.set_restore(seq, 0, idle);
        self.admission.set_recompute(seq, rows);
        self.by_id.insert(a.id, Place::Waiting);
        self.waiting.push(a);
        self.free_slots.push(slot as u32);
        self.active -= 1;
        self.stats.counters.requests_suspended += 1;
    }

    /// Ends the request in `slot`: tells its sink (if it still listens),
    /// leaves what it computed in the prefix cache, gives its pages and its
    /// slot back.
    fn retire(&mut self, slot: u32, finish: FinishReason) {
        self.retire_with(slot, finish, true);
    }

    /// [`Conductor::retire`], with or without leaving the request's pages and
    /// state in the cache: not when the engine is going down or a device
    /// failed, and the state is not there.
    fn retire_with(&mut self, slot: u32, finish: FinishReason, keep: bool) {
        // A sequence in the middle of a paused step has a state that is neither
        // before nor after its rows: nothing of it is kept, and the model is told
        // to give the step up.
        let keep = match self.paused {
            Some(p) if p.slot == slot => {
                self.paused = None;
                self.paused_images.clear();
                self.give_up = true;
                let saves = std::mem::take(&mut self.paused_inner);
                self.release_inner(saves);
                false
            }
            _ => keep,
        };
        let Some(mut a) = self.slots[slot as usize].take() else { return };
        self.by_id.remove(&a.id);
        if !a.sink.is_closed() {
            if !a.emitted.is_empty() {
                a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            a.sink.send(Event::Done { finish, usage: a.usage() });
        }
        if keep {
            self.keep(&mut a, slot);
        } else if let Some(cache) = &mut self.cache {
            cache.index.withdraw(a.owner);
        }
        self.stats.report(RequestReport {
            id: a.id,
            prompt_tokens: a.prompt_len,
            cached_tokens: a.cached_prompt,
            replayed_rows: a.replayed,
            computed_rows: a.computed,
        });
        a.cells.release(&mut self.pool);
        self.free_slots.push(slot);
        self.active -= 1;
        match finish {
            FinishReason::Cancelled => {
                self.admission.remove(a.seq);
                self.stats.counters.requests_cancelled += 1;
            }
            _ => {
                // What it produced of what it was allowed teaches the
                // accounting what to expect of the others.
                self.admission.finished(a.seq);
                self.stats.counters.requests_finished += 1;
            }
        }
    }

    /// Ends the waiting request at index `i` of the line.
    fn dismiss(&mut self, i: usize, finish: FinishReason) {
        let mut a = self.waiting.swap_remove(i);
        self.by_id.remove(&a.id);
        self.admission.remove(a.seq);
        if !a.sink.is_closed() {
            if !a.emitted.is_empty() {
                a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            a.sink.send(Event::Done { finish, usage: a.usage() });
        }
        a.cells.release(&mut self.pool);
        self.stats.counters.requests_cancelled += 1;
    }

    fn finish_everything(&mut self, finish: FinishReason) {
        for slot in 0..self.slots.len() as u32 {
            self.retire_with(slot, finish, false);
        }
        while !self.waiting.is_empty() {
            self.dismiss(self.waiting.len() - 1, finish);
        }
    }

    /// Ends every request with an error, after the devices failed. The
    /// engine accepts no more requests.
    fn fail(&mut self, message: String) {
        let error = EngineError::Internal(message.clone());
        for slot in 0..self.slots.len() as u32 {
            if let Some(a) = &self.slots[slot as usize] {
                let _ = a.sink.send(Event::Error(error.clone()));
                self.retire_with(slot, FinishReason::Cancelled, false);
            }
        }
        while let Some(a) = self.waiting.last() {
            let _ = a.sink.send(Event::Error(error.clone()));
            self.dismiss(self.waiting.len() - 1, FinishReason::Cancelled);
        }
        let _ = self.failure.set(message.clone());
        self.failed = Some(message);
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
                self.admission.set_recompute(a.seq, rows as u32);
                self.admission.set_restore(a.seq, 0, probe.idle);
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

    /// Decides how many drafts each generating sequence asks for, and
    /// fills `requests` for the planner. A round of generation asks for the
    /// windows (`prompts` false), a piece of prompt for the prompts and the
    /// newest token of each generating sequence, without drafts.
    ///
    /// The drafts of the windows are chosen together, by the rate of the
    /// whole step ([`ares_sched::choose`]): each sequence offers what the
    /// n-gram drafter proposes (with its confidence in each token) and
    /// what the model's MTP block is likely to get right at each
    /// position (measured for this sequence, from what the engine saw where it
    /// has none yet), and the step takes the rows that pay, from the drafter
    /// that pays, by the price of a step of that many rows.
    fn want(&mut self, turn: Turn) {
        self.requests.clear();
        let round = turn == Turn::Round;
        let span = u64::from(self.span);
        let generating = self.slots.iter().flatten().filter(|a| a.generating()).count() as u32;
        // Every draft row needs its logits, like the one row every sequence
        // may need: drafts share what the model gives beyond those, and the
        // rows of a step that verifies drafts are bounded by the model.
        let spare_rows = self
            .model
            .logit_rows()
            .saturating_sub(self.active as u32)
            .min(self.model.draft_rows().saturating_sub(generating));
        let drafting = round && !self.drafters.is_empty() && self.config.window_rows > 1;
        // How often the draft before a link leaves its chain above the link's
        // bar, by the bars of the round before, per drafter: among all drafts
        // (how often the link is made, which is what asking for it costs) and
        // among the drafts that are accepted (how often it is made when its
        // row counts). The first link has no bar.
        let passes: Vec<Vec<(f64, f64)>> = self
            .needs
            .iter()
            .zip(&self.confidences)
            .map(|(needs, confidence)| {
                std::iter::once((1.0, 1.0))
                    .chain(needs.iter().map(|need| confidence.share_at_least((*need).into())))
                    .collect()
            })
            .collect();
        // The windows that offer nothing are rows of the step all the same:
        // the price of the step counts them in.
        let (mut quiet, mut quiet_cells) = (0u32, 0u64);
        let paused = self.paused.map(|p| p.slot as usize);
        for (slot, a) in self.slots.iter_mut().enumerate() {
            let Some(a) = a else { continue };
            // The paused step's sequence runs only that step, alone.
            if Some(slot) == paused {
                continue;
            }
            if !a.generating() {
                // Prompts wait for their piece, and for the rows of an image
                // the tower has not computed yet.
                let ready = a.ready_rows(a.state_pos(), a.prefill_left(), &self.images);
                if !round && ready > 0 {
                    self.requests.push(Request {
                        seq: SeqId(slot as u32),
                        want: Want::Prefill { tokens: ready },
                        waited_steps: a.waited_steps,
                        priority: a.priority,
                    });
                }
                continue;
            }
            a.draft.clear();
            a.model_drafter = None;
            a.ngram.clear();
            a.ngram_chances.clear();
            a.model_chances.clear();
            a.offered.iter_mut().for_each(Vec::clear);
            if drafting && a.draft_allowed {
                // A window never needs rows past the end of the request: the
                // tokens still wanted, and the room left in the context.
                let wanted = a.max_tokens.map_or(u32::MAX, |m| m - a.completion);
                let room = self.context_length.saturating_sub(a.tokens.len() as u32);
                let cap = self
                    .config
                    .window_rows
                    .saturating_sub(1)
                    .min(wanted.saturating_sub(1))
                    .min(room)
                    // A sequence whose next pick depends on the picks before
                    // it by a state only a pick can advance verifies no drafts.
                    .min(if a.sampler.allows_drafts() { u32::MAX } else { 0 })
                    as usize;
                // The window ends where a checkpoint is due, so that the
                // state is exactly there when all of it is accepted.
                let cap = match &self.cache {
                    Some(cache) => {
                        let from = a.cells.len();
                        let to = from + 1 + cap as u64;
                        match cache.capture(a.prompt_len, &a.boundaries, a.checkpoint_at, from, to)
                        {
                            Some(at) => cap.min((at - from - 1) as usize),
                            None => cap,
                        }
                    }
                    None => cap,
                };
                if cap > 0 {
                    // The n-gram drafter first: its proposal costs nothing, and it
                    // learns from what is right whether it is taken or not.
                    a.drafter.draft(&self.corpus, cap, &mut a.ngram, &mut a.ngram_chances);
                    // A draft that a rule of the request forbids cannot be
                    // picked: it and the ones after it are not offered.
                    a.sampler.cut_drafts(
                        &a.tokens,
                        a.prompt_len as usize,
                        &mut a.ngram,
                        &mut a.ngram_chances,
                    );
                    // The model's drafters know no rule: a request with one
                    // asks for them while the rule leaves its next token
                    // free. The round verifies their drafts with their masks
                    // and corrections, as an n-gram draft's.
                    let free = a.sampler.allows_model_drafts(&a.tokens, a.prompt_len as usize);
                    for (d, drafter) in self.drafters.iter().enumerate() {
                        let own = if free { cap.min(drafter.drafts as usize) } else { 0 };
                        let chances = &mut a.offered[d];
                        a.model_acceptance[d].rates(
                            &self.engines[d],
                            self.config.draft_prior,
                            own,
                            chances,
                        );
                        // A draft counts when its link is made and it is right.
                        for (chance, (_, made)) in chances.iter_mut().zip(&passes[d]) {
                            *chance *= made;
                        }
                    }
                }
            }
            if a.ngram_chances.is_empty() && a.offered.iter().all(Vec::is_empty) {
                quiet += 1;
                quiet_cells += a.cells.len().min(span);
            }
            self.requests.push(Request {
                seq: SeqId(slot as u32),
                want: Want::Window { drafts: 0 },
                waited_steps: a.waited_steps,
                priority: a.priority,
            });
        }
        if !drafting {
            return;
        }
        // The offers of every generating sequence that has any: the n-gram drafter's,
        // then each model drafter's under its id.
        let mut offers: Vec<Vec<Offer<'_>>> = Vec::new();
        let mut asking: Vec<usize> = Vec::new();
        for (slot, a) in self.slots.iter().enumerate() {
            let Some(a) = a.as_ref().filter(|a| a.generating() && a.draft_allowed) else {
                continue;
            };
            if a.ngram_chances.is_empty() && a.offered.iter().all(Vec::is_empty) {
                continue;
            }
            offers.push(
                std::iter::once(Offer { drafter: NGRAM, chances: &a.ngram_chances })
                    .chain(
                        a.offered
                            .iter()
                            .enumerate()
                            .map(|(d, chances)| Offer { drafter: d + 1, chances }),
                    )
                    .collect(),
            );
            asking.push(slot);
        }
        let wishes: Vec<Wish<'_>> = asking
            .iter()
            .zip(&offers)
            .map(|(slot, offers)| {
                let a = self.slots[*slot].as_ref().expect("asked");
                Wish { reads: a.cells.len().min(span), offers }
            })
            .collect();
        // The prices a window is tried at: a size that has not run for long
        // costs no more than the nearest smaller one that has, so it is taken
        // again when it could pay, and the step measures it.
        let step = |load: Load| {
            self.prices.trial_ms(Load { rows: load.rows + quiet, cells: load.cells + quiet_cells })
        };
        let chain = |drafter: usize, links: u32| match drafter {
            NGRAM => 0.0,
            d => expected_chain_ms(&self.chains[d - 1], &passes[d - 1], links),
        };
        let prices = Prices { step: &step, chain: &chain, spare_rows };
        let choices = choose(&wishes, &prices);
        // Where a chain of a drafter that makes its links one after the other
        // ends: one more link is made while the chance that its draft counts
        // (the reach of the chain so far, known to the drafter from its
        // probabilities, times the chance of a draft it has not made yet) is
        // worth the link's time at the rate this step is expected to have.
        let rate = ares_sched::rate(&wishes, &prices, &choices);
        self.token_ms = if rate > 0.0 { 1.0 / rate } else { 0.0 };
        drop(wishes);
        for (d, needs) in self.needs.iter_mut().enumerate() {
            let curve = &self.chains[d];
            let mean = self.confidences[d].mean(self.config.draft_prior);
            let upto = |links: u32| curve.ms(links).unwrap_or(0.0);
            needs.clear();
            needs.extend(
                (1..self.config.window_rows)
                    .map(|link| ares_sched::need(rate, upto(link + 1) - upto(link), mean) as f32),
            );
        }
        let mut asked = vec![false; self.drafters.len()];
        for (slot, choice) in asking.iter().zip(choices) {
            let a = self.slots[*slot].as_mut().expect("asked");
            let Choice { drafter, drafts } = choice;
            let drafts = drafts as usize;
            match drafter {
                Some(NGRAM) => a.draft.extend_from_slice(&a.ngram[..drafts]),
                Some(id) => {
                    let d = id - 1;
                    a.model_drafter = Some(d);
                    asked[d] = true;
                    a.draft.resize(drafts, 0);
                    a.model_chances.clear();
                    a.model_chances.extend_from_slice(&a.offered[d]);
                    // The first link is the window's reason: it has no bar.
                    a.model_needs.clear();
                    a.model_needs.push(0.0);
                    a.model_needs.extend(self.needs[d].iter().take(drafts.saturating_sub(1)));
                }
                None => {}
            }
            for (d, chances) in a.offered.iter().enumerate() {
                if a.model_drafter != Some(d) && !chances.is_empty() {
                    a.model_acceptance[d].fade(UNASKED_SEQUENCE);
                }
            }
            if let Some(request) = self.requests.iter_mut().find(|r| r.seq.0 == *slot as u32) {
                request.want = Want::Window { drafts: drafts as u32 };
            }
        }
        for (engine, asked) in self.engines.iter_mut().zip(asked) {
            if !asked {
                engine.fade(UNASKED_ENGINE);
            }
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
        let generating = self
            .slots
            .iter()
            .enumerate()
            .filter(|(slot, a)| {
                Some(*slot) != paused_slot && a.as_ref().is_some_and(|a| a.generating())
            })
            .count() as u32;
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
                Ok(ms) if ms > 0.0 => self.share.spent(turn, ms, generating, true),
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
        let limits = Limits {
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
        self.reserve_inner();
        let used = match self.write_record(false) {
            Ok(used) => used,
            Err(message) => return self.fail(message),
        };
        self.stats.record(Phase::Write, t_write.elapsed().as_nanos() as u64);
        self.run_record(began, used, turn, generating, chain_ms, true);
    }

    /// Takes places for the checkpoints due inside the rows of the planned parts.
    fn reserve_inner(&mut self) {
        self.inner.clear();
        let Some(cache) = &mut self.cache else { return };
        for &(slot, position) in &self.inner_due {
            if !self.parts.iter().any(|p| p.slot == slot) {
                continue;
            }
            let (place, dropped) = cache.reserve();
            self.stats.counters.checkpoints_dropped += u64::from(dropped);
            if let Some(place) = place {
                self.inner.push(crate::InnerSave { slot, place, position: position as u32 });
            }
        }
    }

    /// Gives back the places taken for checkpoints inside a step that will not run.
    fn release_inner(&mut self, saves: Vec<crate::InnerSave>) {
        if let Some(cache) = &mut self.cache {
            for save in saves {
                cache.unreserve(save.place);
            }
        }
    }

    /// Runs the step whose record (`used` bytes) is written, for `turn`, with
    /// `generating` sequences generating and `chain_ms` milliseconds of the
    /// drafters' step before it; `observe` lets its time teach the prices. A
    /// step the model paused keeps its part for later and commits nothing.
    fn run_record(
        &mut self,
        began: Instant,
        used: usize,
        turn: Turn,
        generating: u32,
        chain_ms: f64,
        observe: bool,
    ) {
        // The devices.
        let t_dispatch = Instant::now();
        self.step_no += 1;
        let meta = StepMeta { step: self.step_no, record_bytes: used, after_idle: self.after_idle };
        if let (false, Some(done)) = (self.after_idle, self.last_done_ns) {
            self.stats.record(Phase::Turnaround, self.clock.now_ns().saturating_sub(done));
        }
        self.after_idle = false;
        // What the step carries: every row and the cells it reads.
        let load = self.parts.iter().fold(Load::default(), |load, part| {
            load.plus(load_of(self.span, part.from, part.rows))
        });
        self.heartbeat.begin();
        if let Err(error) = self.exec.dispatch(&meta) {
            self.heartbeat.end();
            return self.fail(error.to_string());
        }
        let t_wait = Instant::now();
        self.stats.record(Phase::Dispatch, (t_wait - t_dispatch).as_nanos() as u64);
        let collected = self.exec.collect(self.config.conductor_spin, &mut self.output);
        self.heartbeat.end();
        let woke = match collected {
            Ok(woke) => woke,
            Err(error) => return self.fail(error.to_string()),
        };
        let t_back = Instant::now();
        self.stats.record(Phase::Wait, (t_back - t_wait).as_nanos() as u64);
        self.stats.counters.conductor_wakes.record(woke);
        let step_ms = (t_back - t_dispatch).as_secs_f64() * 1e3;
        // An executor that knows the time of its steps without a clock says so.
        let step_ms = self.output.step_ms.take().unwrap_or(step_ms);
        // A step the model paused, or one that went on where it was paused, did
        // part of its rows' work.
        if observe && !self.output.paused {
            self.prices.observe(load, step_ms);
        }
        // The round took its chains and its rows.
        let step_ms = step_ms + chain_ms;
        self.share.spent(turn, step_ms, generating, self.prompt_rows.iter().any(|rows| *rows > 0));
        let last_done = self.output.devices.iter().map(|d| d.finished_at_ns).max().unwrap_or(0);
        self.last_done_ns = Some(last_done);
        for (device, times) in self.output.devices.iter().enumerate() {
            let notice = if times.finished_at_ns == last_done {
                self.clock.now_ns().saturating_sub(last_done)
            } else {
                0
            };
            self.stats.record_device(device, times, notice);
        }
        if self.output.paused {
            // The model stopped between two pieces of the step: its rows run again
            // later, alone, and go on there. What the record carried for the step
            // before (its windows' commits) has been done.
            if self.parts.len() != 1 {
                return self.fail("the model paused a step of several sequences".into());
            }
            let part = self.parts[0];
            self.paused = Some(part);
            self.hold_images(part);
            self.paused_inner = std::mem::take(&mut self.inner);
            self.commits.clear();
            self.stats.counters.steps_paused += 1;
            self.stats.record(Phase::Step, began.elapsed().as_nanos() as u64);
            self.stats.counters.steps += 1;
            return;
        }

        // The paused step has run to its end (or another step ran while it waits).
        if self.paused.is_none() {
            self.paused_images.clear();
        }
        let t_commit = Instant::now();
        self.commit();
        self.stats.record(Phase::Commit, t_commit.elapsed().as_nanos() as u64);

        let t_events = Instant::now();
        self.deliver();
        self.stats.record(Phase::Events, t_events.elapsed().as_nanos() as u64);
        self.stats.record(Phase::Step, began.elapsed().as_nanos() as u64);
        self.stats.counters.steps += 1;
    }

    /// Runs the image tower for the prompts that wait for an image's rows:
    /// the pass under way, or a new one for the first such image of the
    /// shortest of them. While sequences generate it runs as many units as
    /// fit the time of a piece (by what the units took before), otherwise the
    /// rest of the pass. A pass that ends leaves its rows in the store.
    /// Returns what the tower took, in milliseconds (0 when it had nothing to
    /// do).
    fn run_tower(&mut self, generating: u32) -> Result<f64, String> {
        let Some(vision) = self.vision else { return Ok(0.0) };
        if self.pass.is_none() {
            let mut waiting: Vec<&Active> =
                self.slots.iter().flatten().filter(|a| !a.generating()).collect();
            waiting.sort_by_key(|a| a.prefill_left());
            // From where the state is; for the paused step's sequence after its
            // rows, whose images it holds.
            let paused = self.paused;
            let from = |a: &Active| match paused {
                Some(p) if p.seq == a.seq => (p.from + u64::from(p.rows)) as u32,
                _ => a.state_pos() as u32,
            };
            let Some(image) = waiting.iter().find_map(|a| a.missing_image(&self.images, from(a)))
            else {
                return Ok(0.0);
            };
            self.pass = Some(TowerPass { image, next: 0, rows: Vec::new(), ms: 0.0 });
        }
        let pass = self.pass.as_mut().expect("a pass");
        let left = vision.units.saturating_sub(pass.next).max(1);
        let rows = f64::from(pass.image.rows().max(1));
        let units = if generating == 0 {
            left
        } else {
            // What a unit takes: of this pass so far, or by the rows of the
            // last one; four units (the stem and three blocks) to learn it.
            let per_unit = if pass.next > 0 {
                Some(pass.ms / f64::from(pass.next))
            } else {
                self.unit_ms_per_row.map(|per_row| per_row * rows)
            };
            match per_unit.filter(|ms| *ms > 0.0) {
                Some(ms) => ((self.config.piece_ms / ms) as u32).clamp(1, left),
                None => left.min(4),
            }
        };
        let mut job = TowerJob {
            image: pass.image.clone(),
            units: pass.next..pass.next + units,
            rows: std::mem::take(&mut pass.rows),
            ms: 0.0,
        };
        self.exec.tower(&mut job).map_err(|error| error.to_string())?;
        pass.next += units;
        pass.ms += job.ms;
        self.unit_ms_per_row = Some(pass.ms / f64::from(pass.next) / rows);
        let counters = &mut self.stats.counters;
        counters.tower_units += u64::from(units);
        counters.tower_us += (job.ms * 1e3) as u64;
        if pass.next >= vision.units {
            let image = pass.image.clone();
            self.pass = None;
            let rows =
                ImageRows { nx: image.nx, ny: image.ny, width: vision.width, values: job.rows };
            if rows.values.len() != (image.rows() * vision.width) as usize {
                return Err(format!(
                    "the image tower gave {} values for an image of {} rows of {}",
                    rows.values.len(),
                    image.rows(),
                    vision.width
                ));
            }
            self.images.put(image.name, Arc::new(rows));
            self.stats.counters.images_computed += 1;
        } else {
            pass.rows = job.rows;
        }
        Ok(job.ms)
    }

    /// Copies the rows of the step's image runs into the image stage, in the
    /// order of the runs.
    fn fill_stage(&mut self) {
        if self.staged.is_empty() {
            return;
        }
        let width = self.vision.map_or(0, |v| v.width) as usize;
        let stage = self.exec.image_stage();
        let mut at = 0;
        for (rows, first, count) in self.staged.drain(..) {
            let values = &rows.values[first as usize * width..(first + count) as usize * width];
            stage[at..at + values.len()].copy_from_slice(values);
            at += values.len();
        }
    }

    /// Holds the rows of the images among `part`'s rows (the step the model paused)
    /// until the step has run to its end: those it held when it paused before (the store
    /// may have let them go since), else the store's.
    fn hold_images(&mut self, part: StepPart) {
        let held = std::mem::take(&mut self.paused_images);
        let Some(a) = &self.slots[part.slot as usize] else { return };
        let (from, end) = (part.from as u32, part.from as u32 + part.rows);
        let Some(first) = a.spans.from(from) else { return };
        for span in a.spans.spans()[first..].iter().take_while(|s| s.first < end) {
            let name = span.image.name;
            if self.paused_images.iter().any(|(n, _)| *n == name) {
                continue;
            }
            let rows = match held.iter().find(|(n, _)| *n == name) {
                Some((_, rows)) => Some(rows.clone()),
                None => self.images.get(&name),
            };
            if let Some(rows) = rows {
                self.paused_images.push((name, rows));
            }
        }
    }

    /// Runs the step the model paused again, alone, for at most `budget_ms`
    /// milliseconds before the model may pause it again (0: as long as it takes):
    /// the same rows, whose cells are claimed already; the model goes on where it
    /// stopped.
    fn resume(&mut self, began: Instant, budget_ms: u32) {
        let Some(part) = self.paused.take() else { return };
        if !self.slots[part.slot as usize].as_ref().is_some_and(|a| a.seq == part.seq) {
            self.paused_images.clear();
            self.give_up = true;
            let saves = std::mem::take(&mut self.paused_inner);
            self.release_inner(saves);
            return;
        }
        // The checkpoints inside its rows were named when it began; the model keeps them.
        self.inner = std::mem::take(&mut self.paused_inner);
        let t_write = Instant::now();
        self.parts.clear();
        self.parts.push(part);
        if let Some(cache) = &mut self.cache {
            cache.settle_leaving(&mut self.pool, &mut *self.exec);
        }
        self.pause_ms = budget_ms;
        self.resuming = true;
        let used = self.write_record(false);
        self.pause_ms = 0;
        self.resuming = false;
        let used = match used {
            Ok(used) => used,
            Err(message) => return self.fail(message),
        };
        self.stats.record(Phase::Write, t_write.elapsed().as_nanos() as u64);
        let generating = self
            .slots
            .iter()
            .enumerate()
            .filter(|(slot, a)| {
                *slot != part.slot as usize && a.as_ref().is_some_and(|a| a.generating())
            })
            .count() as u32;
        let turn = Turn::Piece { rows: part.rows };
        self.run_record(began, used, turn, generating, 0.0, false);
    }

    /// Whether `part` is a window whose drafts one of the model's drafters
    /// makes.
    fn takes_model_drafts(&self, part: &StepPart) -> bool {
        part.rows > 1
            && matches!(part.kind, PartKind::Window)
            && self.slots[part.slot as usize].as_ref().is_some_and(|a| a.model_drafter.is_some())
    }

    /// The step that only drafts: the model's drafters make the drafts of the
    /// windows that take them, as many as were asked (or as far as their bars
    /// let a chain go), and nothing else runs. The drafts come back with what
    /// the drafter said of each, and the round then verifies as many of each
    /// window as pay ([`Conductor::verify_lengths`]). Returns what the step
    /// took, in milliseconds: the price of the drafts.
    fn chains_first(&mut self) -> Result<f64, String> {
        let used = self.write_record(true)?;
        let t_dispatch = Instant::now();
        self.step_no += 1;
        let meta = StepMeta { step: self.step_no, record_bytes: used, after_idle: self.after_idle };
        if let (false, Some(done)) = (self.after_idle, self.last_done_ns) {
            self.stats.record(Phase::Turnaround, self.clock.now_ns().saturating_sub(done));
        }
        self.after_idle = false;
        self.exec.dispatch(&meta).map_err(|error| error.to_string())?;
        let t_wait = Instant::now();
        self.stats.record(Phase::Dispatch, (t_wait - t_dispatch).as_nanos() as u64);
        let woke = self
            .exec
            .collect(self.config.conductor_spin, &mut self.output)
            .map_err(|error| error.to_string())?;
        let t_back = Instant::now();
        self.stats.record(Phase::Wait, (t_back - t_wait).as_nanos() as u64);
        self.stats.counters.conductor_wakes.record(woke);
        self.last_done_ns = self.output.devices.iter().map(|d| d.finished_at_ns).max();
        let step_ms = (t_back - t_dispatch).as_secs_f64() * 1e3;
        let step_ms = self.output.step_ms.take().unwrap_or(step_ms);

        // The drafts of every chain, in the order of the windows: those of the
        // links that were made, and what the drafter said of them.
        let reports = self.output.draft_probs.len() == self.output.drafts.len();
        let mut at = 0usize;
        let mut longest = vec![0usize; self.drafters.len()];
        for part in &self.parts {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let Some(d) =
                a.model_drafter.filter(|_| part.rows > 1 && matches!(part.kind, PartKind::Window))
            else {
                continue;
            };
            let asked = part.rows as usize - 1;
            let end = (at + asked).min(self.output.drafts.len());
            let drafts = &self.output.drafts[at..end];
            let made = drafts.iter().take_while(|draft| **draft != u32::MAX).count();
            a.draft.clear();
            a.draft.extend_from_slice(&drafts[..made]);
            a.model_probs.clear();
            if reports {
                a.model_probs.extend_from_slice(&self.output.draft_probs[at..at + made]);
            }
            at = end;
            longest[d] = longest[d].max(made);
            let counters = &mut self.stats.counters;
            counters.model_links_asked += asked as u64;
            counters.model_links_cut += (asked - made) as u64;
        }
        if longest.iter().any(|links| *links > 0) {
            let counters = &mut self.stats.counters;
            counters.model_draft_steps += 1;
            counters.model_draft_links += *longest.iter().max().expect("a drafter") as u64;
            counters.model_draft_us += (step_ms * 1e3) as u64;
        }
        // The step's time is each drafter's that ran in it: a step of two
        // drafters is rare (two sequences that chose differently) and prices
        // both a little high.
        for (chain, links) in self.chains.iter_mut().zip(longest) {
            if links > 0 {
                chain.observe(links as u32, step_ms);
            }
        }
        self.verify_lengths();
        Ok(step_ms)
    }

    /// How many drafts of each chain of the model's own drafter the round
    /// verifies, now that they are made: each draft is worth the chance that
    /// it and the ones before it are accepted, read from the probability the
    /// drafter gave it ([`ares_sched::Confidence`]; by its position where the
    /// drafter says nothing), and the round takes as many as raise its rate
    /// ([`ares_sched::choose`], with the chains paid for). The rows of the
    /// drafts it does not take leave the step, and their cells the table.
    fn verify_lengths(&mut self) {
        let span = u64::from(self.span);
        // The rows of the round that are not such windows are its rows all
        // the same.
        let (mut fixed, mut fixed_cells) = (0u32, 0u64);
        let mut asking: Vec<usize> = Vec::new();
        let mut chances: Vec<Vec<f64>> = Vec::new();
        let mut drafters: Vec<usize> = Vec::new();
        for (index, part) in self.parts.iter().enumerate() {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let Some(d) =
                a.model_drafter.filter(|_| part.rows > 1 && matches!(part.kind, PartKind::Window))
            else {
                fixed += part.rows;
                fixed_cells += load_of(self.span, part.from, part.rows).cells;
                continue;
            };
            let mut own = Vec::with_capacity(a.draft.len());
            if a.model_probs.len() == a.draft.len() {
                own.extend(a.model_probs.iter().map(|p| self.confidences[d].chance(*p)));
            } else {
                a.model_acceptance[d].rates(
                    &self.engines[d],
                    self.config.draft_prior,
                    a.draft.len(),
                    &mut own,
                );
            }
            // A draft that a rule of the request forbids cannot be picked: it
            // and the ones after it are not verified.
            a.sampler.cut_drafts(&a.tokens, a.prompt_len as usize, &mut a.draft, &mut own);
            a.model_probs.truncate(a.draft.len());
            asking.push(index);
            chances.push(own);
            drafters.push(d + 1);
        }
        let offers: Vec<[Offer<'_>; 1]> = chances
            .iter()
            .zip(&drafters)
            .map(|(own, drafter)| [Offer { drafter: *drafter, chances: own }])
            .collect();
        let wishes: Vec<Wish<'_>> = asking
            .iter()
            .zip(&offers)
            .map(|(index, offers)| Wish { reads: self.parts[*index].from.min(span), offers })
            .collect();
        let step = |load: Load| {
            self.prices.trial_ms(Load { rows: load.rows + fixed, cells: load.cells + fixed_cells })
        };
        let chain = |_: usize, _: u32| 0.0;
        let spare_rows = self.model.draft_rows().saturating_sub(self.parts.len() as u32);
        let prices = Prices { step: &step, chain: &chain, spare_rows };
        let choices = choose(&wishes, &prices);
        let rate = ares_sched::rate(&wishes, &prices, &choices);
        drop(wishes);
        if rate > 0.0 {
            self.token_ms = 1.0 / rate;
        }
        for ((index, choice), own) in asking.into_iter().zip(choices).zip(chances.iter()) {
            let part = &mut self.parts[index];
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let taken = (choice.drafts as usize).min(a.draft.len());
            // The chance of each draft the round verifies, for what its row
            // is worth.
            a.model_chances.clear();
            a.model_chances.extend(own.iter().take(taken).copied());
            let dropped = part.rows as usize - 1 - taken;
            self.stats.counters.model_drafts_unverified += (a.draft.len() - taken) as u64;
            a.draft.truncate(taken);
            a.model_probs.truncate(taken);
            part.rows = 1 + taken as u32;
            let keep = a.cells.len() - dropped as u64;
            a.cells.truncate(&mut self.pool, keep);
        }
    }

    /// Writes the step record for the parts in `self.parts`. Returns its
    /// length in bytes. With `drafts_only`, the record of the step that only
    /// drafts ([`Conductor::chains_first`]): the windows whose drafts the
    /// model's own drafter makes, as long as their chains are asked, with
    /// their draft bars. Without, the round or the piece itself, every window
    /// with its drafts as tokens. The commit of the step before goes with
    /// whichever comes first.
    fn write_record(&mut self, drafts_only: bool) -> Result<usize, String> {
        let full = |e: record::RecordFull| format!("{e}");
        let (mut segments, mut rows) = (0u32, 0u32);
        for part in self.parts.iter().filter(|p| !drafts_only || self.takes_model_drafts(p)) {
            segments += 1;
            rows += part.rows;
        }
        let buffer = self.exec.record();
        let mut writer =
            RecordWriter::new(buffer, self.step_no + 1, segments, rows).map_err(full)?;
        if drafts_only {
            writer.drafts_only();
        } else {
            writer.pause_after(self.pause_ms);
        }
        if std::mem::take(&mut self.give_up) {
            writer.give_up_paused();
        }
        writer.asked(self.asked);
        if !self.resuming && !drafts_only {
            writer.inner_saves(&self.inner).map_err(full)?;
        }
        self.runs.clear();
        self.samplers.clear();
        self.step_rows.clear();
        self.updates.clear();
        self.bars.clear();
        self.worth.clear();
        self.image_runs.clear();
        self.staged.clear();
        let mut stage_at = 0u32;
        let mut row_index = 0u32;
        let counters = &mut self.stats.counters;
        let window = self.spec.window.max(1) as usize;
        let mut has_prompt = false;
        let mut has_generation = false;
        // The drafter whose table of chances the bars are read with: the
        // first that ends chains by them (a model has one MTP block at most).
        let mut bars_of: Option<usize> = None;
        for part in &self.parts {
            let own = if drafts_only && part.rows > 1 && matches!(part.kind, PartKind::Window) {
                self.slots[part.slot as usize].as_ref().and_then(|a| a.model_drafter)
            } else {
                None
            };
            if drafts_only && own.is_none() {
                continue;
            }
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            // The position of the first row.
            let first_cell = part.from as u32;
            match part.kind {
                PartKind::Prefill { completes } => {
                    has_prompt = true;
                    if self.resuming {
                        // Counted when the step was first written.
                    } else if part.replay {
                        counters.replay_rows += u64::from(part.rows);
                    } else {
                        counters.prefill_rows += u64::from(part.rows);
                    }
                    // The rows of a prompt are certain: recurrent state
                    // advances with them.
                    writer.segment(part.slot, first_cell, part.rows, true).map_err(full)?;
                    if part.replay {
                        writer.replay().map_err(full)?;
                    }
                    for k in 0..part.rows {
                        let position = first_cell + k;
                        writer
                            .row_at(
                                Row {
                                    slot: part.slot,
                                    position,
                                    token: a.tokens[position as usize],
                                    context: self
                                        .spec
                                        .reduce(a.states[position as usize], position),
                                },
                                a.spans.rotary(position),
                            )
                            .map_err(full)?;
                    }
                    // The rows of the images among them come from the store,
                    // through the stage.
                    if let Some(first) = a.spans.from(first_cell) {
                        let end = first_cell + part.rows;
                        for span in a.spans.spans()[first..].iter() {
                            if span.first >= end {
                                break;
                            }
                            let (lo, hi) = (span.first.max(first_cell), span.end().min(end));
                            // A paused step's own rows first: the store may have let
                            // them go meanwhile.
                            let held =
                                self.paused_images.iter().find(|(n, _)| *n == span.image.name);
                            let rows = match held {
                                Some((_, rows)) => Some(rows.clone()),
                                None => self.images.get(&span.image.name),
                            };
                            let Some(rows) = rows else {
                                return Err(format!(
                                    "the rows of an image of request {} are not in the store",
                                    a.id
                                ));
                            };
                            self.image_runs.push(ImageRun {
                                row: row_index + (lo - first_cell),
                                rows: hi - lo,
                                stage: stage_at,
                            });
                            self.staged.push((rows, lo - span.first, hi - lo));
                            stage_at += hi - lo;
                        }
                    }
                    if completes {
                        self.runs.push((row_index + part.rows - 1, 1));
                        let row = a.sampler.row(
                            first_cell + part.rows - 1,
                            &a.tokens,
                            a.prompt_len as usize,
                            &[],
                            0,
                            &mut self.step_rows,
                        );
                        self.samplers.push(row);
                    }
                }
                PartKind::Window | PartKind::Score { .. } => {
                    has_generation = !drafts_only;
                    if !drafts_only {
                        counters.window_rows += 1;
                        counters.draft_rows += u64::from(part.rows - 1);
                    }
                    let newest = a.tokens.len() - 1;
                    // A window with drafts leaves recurrent state alone until
                    // the next step commits the rows that were accepted.
                    writer
                        .segment(part.slot, first_cell, part.rows, part.rows == 1)
                        .map_err(full)?;
                    if let Some(d) = own {
                        // The draft rows' tokens below are placeholders: the
                        // model's drafter makes them.
                        writer.model_drafts(d as u32 + 1).map_err(full)?;
                        // A drafter that makes its links one after the other
                        // ends its chain by the bars; one pass makes them all.
                        let links = part.rows as usize - 1;
                        let needs: &[f32] = match self.drafters[d].kind {
                            DrafterKind::Mtp => &a.model_needs,
                            DrafterKind::Dflash => &[],
                        };
                        self.bars.extend(needs.iter().take(links));
                        self.bars.resize(self.bars.len() + links - links.min(needs.len()), 0.0);
                        if self.drafters[d].kind == DrafterKind::Mtp {
                            bars_of.get_or_insert(d);
                        }
                    }
                    // A link of the chain picks with the sampler row of the
                    // row that verifies it, made as the first of the window:
                    // without a mask for the drafts before it, which are not
                    // known yet. The round verifies them with their masks.
                    let known: &[u32] = if own.is_some() { &[] } else { &a.draft };
                    for k in 0..part.rows as usize {
                        let row = a.sampler.row(
                            first_cell + k as u32,
                            &a.tokens,
                            a.prompt_len as usize,
                            known,
                            k.min(known.len()),
                            &mut self.step_rows,
                        );
                        self.samplers.push(row);
                    }
                    writer
                        .row_at(
                            Row {
                                slot: part.slot,
                                position: first_cell,
                                token: a.tokens[newest],
                                context: self.spec.reduce(a.states[newest], first_cell),
                            },
                            a.spans.rotary(first_cell),
                        )
                        .map_err(full)?;
                    // The contexts of the drafts follow from the drafts
                    // themselves, as they will once they are accepted.
                    self.scratch_tokens.clear();
                    self.scratch_tokens
                        .extend_from_slice(&a.tokens[a.tokens.len().saturating_sub(window)..]);
                    let mut state = a.states[newest];
                    for (k, &draft) in a.draft.iter().enumerate() {
                        self.scratch_tokens.push(draft);
                        state = self.spec.advance(state, &self.scratch_tokens);
                        let position = first_cell + 1 + k as u32;
                        writer
                            .row_at(
                                Row {
                                    slot: part.slot,
                                    position,
                                    token: draft,
                                    context: self.spec.reduce(state, position),
                                },
                                a.spans.rotary(position),
                            )
                            .map_err(full)?;
                    }
                    // Every row of a window is picked, in one run with the
                    // windows before it.
                    match self.runs.last_mut() {
                        Some((first, count)) if *first + *count == row_index => *count += part.rows,
                        _ => self.runs.push((row_index, part.rows)),
                    }
                    // What each draft row is worth: the chance that the
                    // drafts up to its own are right, times what a token
                    // costs.
                    // Before the step has a price nothing is known of it.
                    if !drafts_only && part.rows > 1 && self.token_ms > 0.0 {
                        let chances = if a.model_drafter.is_some() {
                            &a.model_chances
                        } else {
                            &a.ngram_chances
                        };
                        self.worth.resize(row_index as usize + 1, 0.0);
                        let mut reach = 1.0;
                        for k in 0..part.rows as usize - 1 {
                            reach *= chances.get(k).copied().unwrap_or(0.0);
                            self.worth.push((reach * self.token_ms) as f32);
                        }
                    }
                }
            }
            row_index += part.rows;
        }
        if !drafts_only {
            counters.rows += u64::from(row_index);
        }
        if has_prompt && has_generation {
            counters.mixed_steps += 1;
        }
        writer.runs(&self.runs).map_err(full)?;
        writer.corrections(&self.step_rows.entries).map_err(full)?;
        writer.masks(&self.step_rows.masks, self.mask_words).map_err(full)?;
        writer.samplers(self.samplers.iter().copied()).map_err(full)?;
        writer.commits(&self.commits).map_err(full)?;
        if drafts_only {
            // The devices commit these with this record: the round's must not
            // carry them again.
            self.commits.clear();
        }
        // Page-table changes since the devices last saw the tables.
        for part in &self.parts {
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let dirty = a.cells.take_dirty();
            if !dirty.is_empty() {
                self.updates.push((part.slot, dirty));
            }
        }
        let (slots, model) = (&self.slots, &self.model);
        writer
            .updates(
                self.updates.iter().map(|(slot, range)| TableUpdate {
                    slot: *slot,
                    first: range.start as u32,
                    count: range.len() as u32,
                }),
                self.updates.iter().flat_map(|(slot, range)| {
                    let a = slots[*slot as usize].as_ref().expect("running");
                    a.cells.pages()[range.clone()].iter().map(|page| model.page_entry(page.0))
                }),
            )
            .map_err(full)?;
        self.freed.clear();
        self.pool.take_freed(record::FREED_PER_RECORD, &mut self.freed);
        writer.freed(&self.freed).map_err(full)?;
        if let Some(d) = bars_of {
            writer.draft_bars(&self.confidences[d].chances(), &self.bars).map_err(full)?;
        }
        if !self.worth.is_empty() {
            self.worth.resize(row_index as usize, 0.0);
            writer.row_worth(&self.worth).map_err(full)?;
        }
        if !self.image_runs.is_empty() {
            writer.images(&self.image_runs).map_err(full)?;
        }
        if let Some(cache) = &mut self.cache {
            cache.write_copies(&mut writer).map_err(full)?;
        }
        let used = writer.finish().map_err(full)?;
        self.fill_stage();
        Ok(used)
    }

    /// Accepts or rejects what the devices brought back, sequence by
    /// sequence.
    fn commit(&mut self) {
        let mut cursor = 0usize;
        let mut row0 = 0u32;
        self.commits.clear();
        for part in &self.parts {
            let first_row = row0;
            row0 += part.rows;
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let before = a.completion;
            // Where the recurrent state is after the rows.
            let mut reached = part.from + u64::from(part.rows);
            match part.kind {
                PartKind::Prefill { completes } => {
                    if part.replay {
                        a.replay_left -= part.rows;
                        a.replayed += part.rows;
                    } else {
                        a.computed += part.rows;
                    }
                    // The rows that ran are certain: their tokens are text
                    // now, for every sequence to draft from. A request that
                    // came back computes rows whose tokens are there already.
                    for at in a.drafter.len()..a.cells.len() as usize {
                        a.drafter.push(&mut self.corpus, a.tokens[at]);
                    }
                    if completes {
                        let pick = self.output.picks[cursor];
                        let probs = self.output.logprobs.get(cursor..=cursor).unwrap_or(&[]);
                        a.sampler.picked(&[pick], probs);
                        cursor += 1;
                        a.accept(
                            &*self.model,
                            &self.spec,
                            self.context_length,
                            &mut self.corpus,
                            &[pick.token],
                        );
                    }
                }
                PartKind::Window | PartKind::Score { .. } => {
                    let picks = &self.output.picks[cursor..cursor + part.rows as usize];
                    let probs_at = cursor;
                    cursor += part.rows as usize;
                    // Draft `i` is accepted when it equals what the model
                    // picked after the row before it, and only if all the
                    // drafts before it were.
                    // Every draft of the window is known: an n-gram draft's, and of
                    // the model's own those its chain made that the round
                    // took. Their rows carry the masks the drafts before them
                    // make, so every pick is the request's.
                    let drafts = part.rows as usize - 1;
                    // A row the model left without what it would have waited
                    // for, and every row of the window after it, has no pick
                    // that counts: the rows before it test the drafts, and the
                    // last of them gives the token after the accepted ones.
                    let exact = self
                        .output
                        .inexact
                        .iter()
                        .find(|row| (first_row + 1..first_row + part.rows).contains(row))
                        .map_or(part.rows, |row| row - first_row)
                        as usize;
                    if exact < part.rows as usize {
                        self.stats.counters.windows_left += 1;
                    }
                    let made = drafts.min(a.draft.len()).min(exact);
                    let probs: &[f32] =
                        if a.model_drafter.is_some() { &a.model_probs } else { &[] };
                    let right = (0..made).take_while(|&i| picks[i].token == a.draft[i]).count();
                    let accepted = right.min(exact - 1);
                    if drafts > 0 {
                        let source = a.model_drafter.map_or(0, |d| self.drafters[d].kind.source());
                        if let Some(d) = a.model_drafter {
                            // A draft after the first rejected one was not
                            // tested.
                            let tested = made.min(right + 1);
                            a.model_acceptance[d].observe(tested, right.min(tested));
                            self.engines[d].observe(tested, right.min(tested));
                            // What its probability said of each draft that
                            // was tested.
                            let counters = &mut self.stats.counters;
                            for (i, p) in probs.iter().enumerate().take(tested) {
                                if i > right {
                                    break;
                                }
                                self.confidences[d].observe(*p, i < right);
                                let bin = Confidence::bin(*p);
                                counters.model_tested_by_p[bin] += 1;
                                counters.model_accepted_by_p[bin] += u64::from(i < right);
                            }
                        } else {
                            // The corpus learns what matches like this one
                            // are worth.
                            a.drafter.observe(&mut self.corpus, made, right);
                        }
                        let counters = &mut self.stats.counters;
                        counters.draft_lengths[source][made.min(DRAFT_POSITIONS)] += 1;
                        counters.drafts_accepted += accepted as u64;
                        counters.drafts_rejected += (made - accepted) as u64;
                        // Positions up to the first rejected one were tested;
                        // a draft no exact row tested was no draft.
                        for i in 0..made.min(right + 1) {
                            let at = i.min(counters.drafts_tested_at[source].len() - 1);
                            counters.drafts_tested_at[source][at] += 1;
                            counters.drafts_accepted_at[source][at] += u64::from(i < right);
                        }
                    }
                    // A proposal of the n-gram drafter that was not taken (the model's
                    // drafts were, or none) learns from the tokens that are now
                    // known: the first of them is known in every window.
                    if (a.model_drafter.is_some() || drafts == 0) && !a.ngram.is_empty() {
                        let known = a.ngram.len().min(accepted + 1);
                        let right =
                            (0..known).take_while(|&i| picks[i].token == a.ngram[i]).count();
                        a.drafter.observe(&mut self.corpus, known, right);
                    }
                    self.scratch_tokens.clear();
                    self.scratch_tokens.extend(picks[..=accepted].iter().map(|p| p.token));
                    let probs =
                        self.output.logprobs.get(probs_at..probs_at + accepted + 1).unwrap_or(&[]);
                    a.sampler.picked(&picks[..=accepted], probs);
                    a.accept(
                        &*self.model,
                        &self.spec,
                        self.context_length,
                        &mut self.corpus,
                        &self.scratch_tokens,
                    );
                    // A request that ends inside the accepted rows keeps
                    // those that came before its end: its state and its cells
                    // stop there.
                    let consumed = (a.completion - before) as usize;
                    // The devices commit the newest token's row and the
                    // accepted drafts to recurrent state with the next step.
                    if drafts > 0 {
                        self.commits.push(Commit {
                            row0: first_row,
                            rows: part.rows,
                            slot: part.slot,
                            accepted: consumed as u32,
                        });
                    }
                    // The newest token's row and the accepted drafts keep
                    // their cells; the rejected rows are cut from the table.
                    let keep = a.cells.len() - u64::from(part.rows) + consumed as u64;
                    a.cells.truncate(&mut self.pool, keep);
                    reached = part.from + consumed as u64;
                }
            }
            let produced = a.completion - before;
            if produced > 0 {
                self.stats.counters.tokens += u64::from(produced);
                self.admission.generated(a.seq, produced);
            }
            // The cache: the pages that are full now enter the index, and a
            // checkpoint is taken where the rows ended on a position that is
            // due one (the rows were cut to end there).
            if let Some(cache) = &mut self.cache {
                let settled = a.cells.len();
                cache.index.publish(
                    &mut self.pool,
                    &mut a.cells,
                    &mut a.chain,
                    &a.names,
                    settled,
                    Priority::NORMAL,
                );
                cache.note(a.owner, &a.chain, settled, false);
                // The checkpoints the model saved inside the rows, then the one at their
                // end.
                let (mut last, mut from) = (a.checkpoint_at, part.from);
                while let Some(at) = cache.capture(a.prompt_len, &a.boundaries, last, from, reached)
                {
                    let page = (at / u64::from(self.pool.cells_per_page())) as usize;
                    let root = cache.at_root(&a.boundaries, at);
                    let hash = a.chain.hashes()[page - 1];
                    let saved = if at == reached {
                        cache.save(part.slot, hash, a.owner, root)
                    } else {
                        let Some(k) = self
                            .inner
                            .iter()
                            .position(|s| s.slot == part.slot && u64::from(s.position) == at)
                        else {
                            break;
                        };
                        let save = self.inner.swap_remove(k);
                        cache.attach(save.place, hash, a.owner, root)
                    };
                    self.stats.counters.checkpoints_saved += u64::from(saved.taken);
                    self.stats.counters.checkpoints_dropped += u64::from(saved.dropped);
                    a.checkpoint_at = at;
                    if at == reached {
                        break;
                    }
                    (last, from) = (at, at);
                }
            }
        }
        // A place taken for a checkpoint inside the rows that the commit did not name.
        let left = std::mem::take(&mut self.inner);
        self.release_inner(left);
    }

    /// Sends the text of the step to the sinks, ends the requests that are
    /// done and the ones whose sink gave up.
    fn deliver(&mut self) {
        for i in 0..self.parts.len() {
            let slot = self.parts[i].slot;
            let a = self.slots[slot as usize].as_mut().expect("running");
            // A rule that failed in the middle of the reply: the picks of
            // this step were made without it, and the text is not sent.
            if let Some(message) = a.sampler.failure() {
                a.emitted.clear();
                let _ = a.sink.send(Event::Error(EngineError::Internal(message)));
                self.retire(slot, FinishReason::Cancelled);
                continue;
            }
            let mut listening = true;
            // The probabilities of the tokens whose text goes out now go out
            // first (a token that prints nothing goes out with the text
            // before it).
            if a.sampler.wants_probs() {
                let probs = a.sampler.text_sent(a.emitted.len());
                if !probs.is_empty() {
                    listening = a.sink.send(Event::Logprobs(probs));
                }
            }
            if listening && !a.emitted.is_empty() {
                listening = a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            match a.finished {
                Some(finish) => {
                    if listening {
                        a.sink.send(Event::Done { finish, usage: a.usage() });
                    }
                    self.retire(slot, finish);
                }
                // The sink refused: nobody wants the rest.
                None if !listening => self.retire(slot, FinishReason::Cancelled),
                None => {}
            }
        }
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
