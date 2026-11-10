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
//!            starts)  first)   someone gives way)                                   corpus, update)
//! ```
//!
//! # Who runs
//!
//! Nothing is reserved for a request's longest possible length and nothing is
//! refused for load. A request is put in line ([`ares_sched::Admission`]) and
//! starts when the pages of its prompt and the output it is expected to
//! produce fit next to what the running requests are expected to need; a
//! request that is alone always starts. Every step claims the pages its rows
//! write. When they do not fit, draft rows are dropped first; then the request
//! with the fewest generated tokens gives way: it lets go of its pages and
//! its place on the devices and waits at the front of the line. The runtime
//! cannot move a sequence's state to host memory yet, so a request that gave
//! way computes its rows again when it comes back, as a prompt made of
//! everything it has read and written. The one request left in the pool is
//! never made to wait: when the pool cannot hold it any longer it ends, as it
//! would at the end of the context.
//!
//! What bounds the number of requests: the ones that run are bounded by the
//! places the devices have ([`RuntimeConfig::max_slots`]: a recurrent state,
//! a page table and a row of logits each) and by the pool; the ones that wait
//! are not bounded.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

use ares_api::{EngineError, Event, EventSink, FinishReason, RequestId, Sampling, Usage};
use ares_cuda::{SampleParams, SampleRow};
use ares_sched::{
    Acceptance, Admission, AdmissionConfig, Arrival, DraftCost, Limits, Load, LookupCorpus,
    LookupDrafter, Outcome, PartKind, PoolView, Request, SeqId, StepTimeModel, Want,
    best_draft_len, plan, prefill_rows_at,
};
use ares_state::{PagePool, Sequence};

use crate::config::{Drafters, RuntimeConfig};
use crate::exec::{Clock, ExecError, Executor, StepMeta, StepOutput};
use crate::model::{ContextSpec, Model};
use crate::record::{self, Commit, RecordWriter, Row, TableUpdate};
use crate::text::StopMatcher;
use crate::timing::{Phase, Snapshot, Stats};

/// Drafts cost the model nothing to produce: a lookup.
const LOOKUP: DraftCost = DraftCost { fixed_ms: 0.0, per_token_ms: 0.0 };

/// How long the conductor sleeps on its channel when requests wait and none
/// can start. Admission starts the first in line whenever nothing runs, so
/// this is a guard against spinning and not a path a request takes.
const STALLED: Duration = Duration::from_millis(1);

/// What other threads ask of the conductor.
pub(crate) enum Command {
    Submit(Box<Submission>),
    Cancel(RequestId),
    Stats { reset: bool, reply: SyncSender<StatsCopy> },
    Inspect { reply: SyncSender<Vec<SequenceInfo>> },
    Shutdown,
}

/// The conductor's measurements, copied out. The percentiles are computed from
/// the copy by whoever asked, so that sorting the samples never takes the
/// conductor's time.
pub(crate) struct StatsCopy {
    stats: Stats,
    pages_free: u32,
    pages_capacity: u32,
    active: u32,
    waiting: u32,
}

impl StatsCopy {
    pub(crate) fn into_snapshot(self) -> Snapshot {
        Snapshot::new(&self.stats, self.pages_free, self.pages_capacity, self.active, self.waiting)
    }
}

/// A request that has been checked.
pub(crate) struct Submission {
    pub id: RequestId,
    pub prompt: Vec<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    /// How the request's tokens are picked.
    pub sampling: Sampling,
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
    /// token has no cell until a step computes it.
    cells: Sequence,
    /// The prompt and the output.
    tokens: Vec<u32>,
    /// Per token, the host-side context state (see [`ContextSpec`]).
    states: Vec<u32>,
    prompt_len: u32,
    /// How many of `tokens` are computed as certain rows before the sequence
    /// generates: the prompt, and for a request that gave way and came back
    /// everything it had read and written by then.
    known: u32,
    /// The sequence's place in the corpus of all text. Its tokens enter the
    /// corpus when they are certain: prompt tokens when the step that
    /// computed their rows is back, generated tokens when they are accepted.
    drafter: LookupDrafter,
    /// How the model's own drafts have done for this sequence.
    own_acceptance: Acceptance,
    /// How the sequence's tokens are picked.
    sampling: SampleParams,
    stop: StopMatcher,
    max_tokens: Option<u32>,
    completion: u32,
    waited_steps: u32,
    /// Whether the request was counted as one that had to wait.
    counted_waiting: bool,
    /// Drafts for the coming step; cut to the rows the plan grants. When the
    /// model drafts them itself (`own`), placeholders until the step brings
    /// the drafts back.
    draft: Vec<u32>,
    own: bool,
    /// What the lookup proposed for a step in which the model's own drafts
    /// were taken: the lookup still learns from what turns out to be right.
    lookup: Vec<u32>,
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
            cached_prompt_tokens: 0,
        }
    }

    /// Takes the tokens the model produced for this request, in order, up to
    /// and including the one that ends it.
    fn accept(
        &mut self,
        model: &dyn Model,
        spec: &ContextSpec,
        context_length: u32,
        corpus: &mut LookupCorpus,
        picked: &[u32],
    ) {
        for &token in picked {
            self.completion += 1;
            if model.ends(token) {
                self.stop.flush(&mut self.emitted);
                self.finished = Some(FinishReason::Stop);
                return;
            }
            let previous = self.states.last().copied().unwrap_or_else(|| spec.start());
            self.tokens.push(token);
            self.states.push(spec.advance(previous, &self.tokens));
            self.drafter.push(corpus, token);
            model.piece(token, &mut self.pending);
            let (stop, emitted) = (&mut self.stop, &mut self.emitted);
            if take_characters(&mut self.pending, |ch| stop.push(ch, emitted)) {
                self.finished = Some(FinishReason::Stop);
                return;
            }
            if self.max_tokens.is_some_and(|max| self.completion >= max)
                || self.tokens.len() as u32 >= context_length
            {
                self.stop.flush(&mut self.emitted);
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

/// One sequence's rows in the step being run.
#[derive(Clone, Copy, Debug)]
struct StepPart {
    slot: u32,
    /// The sequence that was in the slot when the step was planned.
    seq: SeqId,
    rows: u32,
    kind: PartKind,
}

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
}

impl PoolView for View<'_> {
    fn evictable(&mut self) -> u32 {
        // Nothing is cached yet: every page has a sequence that holds it.
        0
    }

    fn releasable(&mut self, seq: SeqId) -> u32 {
        self.slots
            .iter()
            .flatten()
            .find(|a| a.seq == seq)
            .map_or(0, |a| a.cells.releasable(self.pool))
    }
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
    step_time: StepTimeModel,
    /// The text of all sequences, which drafts are looked up in.
    corpus: LookupCorpus,
    stats: Stats,
    step_no: u64,
    after_idle: bool,
    /// When the last device finished the previous step, on the executor's clock.
    last_done_ns: Option<u64>,
    failed: Option<String>,
    shutting_down: bool,
    /// Requests wait and the last pass could start none of them.
    stalled: bool,
    // Scratch that lives across steps so that a step allocates nothing for it.
    requests: Vec<Request>,
    parts: Vec<StepPart>,
    updates: Vec<(u32, Range<usize>)>,
    runs: Vec<(u32, u32)>,
    /// The outcome of this step's draft windows, for the next step's record.
    commits: Vec<Commit>,
    output: StepOutput,
    scratch_tokens: Vec<u32>,
    /// The lookup's proposal for the sequence being looked at, and the
    /// confidence of each of its tokens.
    proposal: Vec<u32>,
    confidence: Vec<f64>,
    /// How the wanted rows of the step are picked, in their order.
    samplers: Vec<SampleRow>,
    /// What a link of the model's own draft chain costs, in device
    /// milliseconds: measured from the steps that drafted, `None` before the
    /// first.
    own_link_ms: Option<f64>,
}

impl Conductor {
    pub(crate) fn new(
        model: Arc<dyn Model>,
        config: RuntimeConfig,
        exec: Box<dyn Executor>,
        rx: Receiver<Command>,
    ) -> Self {
        let max_rows = match config.max_rows {
            0 => model.max_rows(),
            n => n.min(model.max_rows()),
        };
        let slots = config.max_slots as usize;
        let admission = AdmissionConfig {
            output_clip: config.output_clip,
            initial_ratio: config.output_ratio,
            max_running: config.max_slots,
            ..AdmissionConfig::new(config.pages, config.cells_per_page)
        };
        Self {
            spec: model.context(),
            context_length: model.card().context_length,
            span: model.attention_span(),
            pool: PagePool::new(config.pages, config.cells_per_page),
            admission: Admission::new(admission),
            slots: (0..slots).map(|_| None).collect(),
            free_slots: (0..config.max_slots).rev().collect(),
            waiting: Vec::new(),
            by_id: HashMap::with_capacity(slots),
            next_seq: 0,
            corpus: LookupCorpus::new(config.lookup),
            stats: Stats::new(exec.devices(), config.stats_window),
            output: StepOutput {
                picks: Vec::with_capacity(max_rows as usize),
                drafts: Vec::with_capacity(max_rows as usize),
                draft_ms: 0.0,
                devices: Vec::with_capacity(exec.devices()),
            },
            samplers: Vec::with_capacity(max_rows as usize),
            own_link_ms: None,
            requests: Vec::with_capacity(slots),
            parts: Vec::with_capacity(max_rows as usize),
            updates: Vec::with_capacity(slots),
            runs: Vec::with_capacity(max_rows as usize),
            commits: Vec::with_capacity(slots),
            scratch_tokens: Vec::with_capacity(64),
            proposal: Vec::with_capacity(config.max_draft as usize),
            confidence: Vec::with_capacity(config.max_draft as usize),
            model,
            config,
            max_rows,
            clock: exec.clock(),
            exec,
            rx,
            active: 0,
            step_time: StepTimeModel::default(),
            step_no: 0,
            after_idle: true,
            last_done_ns: None,
            failed: None,
            shutting_down: false,
            stalled: false,
        }
    }

    /// Runs a few steps of one row, so lazily loaded kernels and first-use
    /// costs are paid before the first request, and forgets their timing.
    pub(crate) fn warm_up(&mut self) -> Result<(), ExecError> {
        for _ in 0..3 {
            let used = {
                // One row of a sequence that starts in slot 0, on the pool's
                // first page; the first request overwrites both.
                let entry = self.model.page_entry(0);
                let mut writer = RecordWriter::new(self.exec.record(), self.step_no, 1, 1)
                    .expect("room for a row");
                writer.segment(0, 0, 1, true).expect("room for a segment");
                writer
                    .row(Row { slot: 0, position: 0, token: 0, context: 0 })
                    .expect("room for a row");
                writer.runs(&[(0, 1)]).expect("room for a run");
                writer
                    .updates(
                        [TableUpdate { slot: 0, first: 0, count: 1 }].into_iter(),
                        [entry].into_iter(),
                    )
                    .expect("room for a page-table entry");
                writer.finish().expect("room for the header")
            };
            self.step_no += 1;
            self.exec.dispatch(&StepMeta {
                step: self.step_no,
                record_bytes: used,
                after_idle: true,
            })?;
            self.exec.collect(Duration::from_millis(1), &mut self.output)?;
        }
        self.stats.reset();
        Ok(())
    }

    /// The loop. Returns after a shutdown, when every request has ended.
    pub(crate) fn run(mut self) {
        loop {
            if self.active == 0 && !self.shutting_down {
                // Nothing runs. With nobody in line either, sleep until
                // somebody asks for something.
                let command = if self.waiting.is_empty() {
                    self.rx.recv().map(Some).map_err(|_| RecvTimeoutError::Disconnected)
                } else if self.stalled {
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
            let turn = Instant::now();
            while let Ok(command) = self.rx.try_recv() {
                self.handle(command);
            }
            if self.shutting_down {
                break;
            }
            self.stats.record(Phase::Intake, turn.elapsed().as_nanos() as u64);
            if self.active == 0 && self.waiting.is_empty() {
                continue;
            }
            self.step(turn);
        }
        self.finish_everything(FinishReason::Cancelled);
        // `Runtime::shutdown` stops the front end from sending before it
        // queues the shutdown command, so nothing can arrive after it; whatever
        // was queued behind it is ended here all the same.
        while let Ok(command) = self.rx.try_recv() {
            self.handle(command);
        }
        self.finish_everything(FinishReason::Cancelled);
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
                let copy = StatsCopy {
                    stats: self.stats.clone(),
                    pages_free: self.pool.free_pages(),
                    pages_capacity: self.pool.capacity(),
                    active: self.active as u32,
                    waiting: self.waiting.len() as u32,
                };
                if reset {
                    self.stats.reset();
                }
                let _ = reply.send(copy);
            }
            Command::Inspect { reply } => {
                let _ = reply.send(self.inspect());
            }
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
        let Submission { id, prompt, max_tokens, stop, sampling, sink } = submission;
        let refuse = |error: EngineError| {
            let _ = sink.send(Event::Error(error));
        };
        if let Some(why) = &self.failed {
            return refuse(EngineError::Internal(why.clone()));
        }
        if self.by_id.contains_key(&id) {
            return refuse(EngineError::InvalidRequest(format!("request id {id} is in use")));
        }
        let prompt_len = prompt.len() as u32;
        let seq = self.fresh_seq();
        let room = self.context_length.saturating_sub(prompt_len);
        let arrival = Arrival {
            seq,
            prompt_cells: prompt_len,
            cached_cells: 0,
            idle_cached_pages: 0,
            max_new_tokens: max_tokens.map_or(room, |max| max.min(room)),
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
        self.waiting.push(Active {
            id,
            seq,
            sink,
            cells: Sequence::new(),
            prompt_len,
            known: prompt_len,
            tokens,
            states,
            drafter: LookupDrafter::new(),
            own_acceptance: Acceptance::new(
                self.config.max_draft as usize,
                self.config.draft_prior,
            ),
            sampling: {
                // What the request leaves open is the model's own setting. A
                // request without a seed gets one of its own.
                let model = self.model.sampling();
                SampleParams {
                    temperature: sampling.temperature.unwrap_or(model.temperature),
                    top_k: sampling.top_k.unwrap_or(model.top_k),
                    top_p: sampling.top_p.unwrap_or(model.top_p),
                    min_p: sampling.min_p.unwrap_or(model.min_p),
                    seed: sampling.seed.unwrap_or_else(|| {
                        (self.clock.now_ns() ^ id.0.rotate_left(32))
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    }),
                }
            },
            stop: StopMatcher::new(&stop),
            max_tokens,
            completion: 0,
            waited_steps: 0,
            counted_waiting: false,
            draft: Vec::with_capacity(self.config.max_draft as usize),
            own: false,
            lookup: Vec::with_capacity(self.config.max_draft as usize),
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
        let a = self.waiting.swap_remove(i);
        self.by_id.insert(a.id, Place::Slot(slot));
        self.slots[slot as usize] = Some(a);
        self.active += 1;
    }

    /// Takes the running request with the name `seq` off the devices: it
    /// lets go of its pages and its slot and waits at the front of the line.
    /// Its state is not kept anywhere, so it computes everything it has read
    /// and written again when it comes back.
    fn suspend(&mut self, seq: SeqId) {
        let Some(slot) = self.slots.iter().position(|a| a.as_ref().is_some_and(|a| a.seq == seq))
        else {
            return;
        };
        let Some(mut a) = self.slots[slot].take() else { return };
        std::mem::take(&mut a.cells).release(&mut self.pool);
        a.known = a.tokens.len() as u32;
        a.draft.clear();
        a.lookup.clear();
        a.own = false;
        a.waited_steps = 0;
        // Nothing is brought back when it returns; its rows are claimed
        // again step by step, as a prompt's are.
        self.admission.set_restore(seq, 0, 0);
        self.admission.set_recompute(seq, a.known);
        self.by_id.insert(a.id, Place::Waiting);
        self.waiting.push(a);
        self.free_slots.push(slot as u32);
        self.active -= 1;
        self.stats.counters.requests_suspended += 1;
    }

    /// Ends the request in `slot`: tells its sink (if it still listens),
    /// gives its pages and its slot back.
    fn retire(&mut self, slot: u32, finish: FinishReason) {
        let Some(mut a) = self.slots[slot as usize].take() else { return };
        self.by_id.remove(&a.id);
        if !a.sink.is_closed() {
            if !a.emitted.is_empty() {
                a.sink.send(Event::Text(std::mem::take(&mut a.emitted)));
            }
            a.sink.send(Event::Done { finish, usage: a.usage() });
        }
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
            self.retire(slot, finish);
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
                self.retire(slot, FinishReason::Cancelled);
            }
        }
        while let Some(a) = self.waiting.last() {
            let _ = a.sink.send(Event::Error(error.clone()));
            self.dismiss(self.waiting.len() - 1, FinishReason::Cancelled);
        }
        self.failed = Some(message);
    }

    /// One pass of the admission accounting: the pages the parts in
    /// `self.parts` claim, then who starts.
    fn settle(&mut self) -> Outcome {
        let mut claims = self.admission.begin(self.pool.free_pages());
        for part in &self.parts {
            let Some(a) = &self.slots[part.slot as usize] else { continue };
            // A prompt's rows and a sequence's newest token must run; draft
            // rows are given up before anybody has to wait.
            let (must, prompt_rows) = match part.kind {
                PartKind::Prefill { .. } => (part.rows, part.rows),
                PartKind::Window | PartKind::Score { .. } => (1, 0),
            };
            let required = a.cells.pages_to_append(&self.pool, u64::from(must));
            let all = a.cells.pages_to_append(&self.pool, u64::from(part.rows));
            claims.claim(part.seq, required, all.saturating_sub(required), prompt_rows);
        }
        claims.settle(&mut View { pool: &self.pool, slots: &self.slots })
    }

    /// Carries out who gives way and who starts.
    fn seat(&mut self, outcome: &Outcome) {
        for seq in &outcome.spill {
            self.suspend(*seq);
        }
        for seq in outcome.resume.iter().chain(&outcome.admit) {
            self.start(*seq);
        }
    }

    /// Decides how many drafts each generating sequence asks for, and
    /// fills `requests` for the planner. Returns the load of the step's
    /// rows that are not optional and not a prompt's: the newest token of
    /// every generating sequence.
    fn want(&mut self) -> Load {
        self.requests.clear();
        let span = self.span;
        let reads = |a: &Active| a.cells.len().min(u64::from(span));
        let mut generating = 0u32;
        let mut windows = Load::default();
        for a in self.slots.iter().flatten().filter(|a| a.generating()) {
            generating += 1;
            windows = windows.plus(load_of(span, a.cells.len(), 1));
        }
        let cost = self.step_time.cost();
        // Every draft row needs its logits, like the one row every sequence
        // may need: drafts share what the model gives beyond those.
        let mut spare_logit_rows = self.model.logit_rows().saturating_sub(self.active as u32);
        for (slot, a) in self.slots.iter_mut().enumerate() {
            let Some(a) = a else { continue };
            let want = if a.generating() {
                a.draft.clear();
                // A window never needs rows past the end of the request: the
                // tokens still wanted, and the room left in the context.
                let wanted = a.max_tokens.map_or(u32::MAX, |m| m - a.completion);
                let room = self.context_length.saturating_sub(a.tokens.len() as u32);
                // And the model bounds the rows of a step that verifies drafts.
                let cap = self
                    .config
                    .max_draft
                    .min(wanted.saturating_sub(1))
                    .min(room)
                    .min(self.model.draft_rows().saturating_sub(1))
                    .min(spare_logit_rows) as usize;
                a.own = false;
                a.lookup.clear();
                if self.config.speculation && cap > 0 {
                    // Two drafters: the lookup, whose proposal costs nothing
                    // but is there only where text repeats, and the model's
                    // own draft block, which always has one and pays for
                    // every link. Each gets its best length; the one with
                    // more expected tokens per millisecond drafts.
                    if self.config.drafters == Drafters::Own {
                        self.proposal.clear();
                        self.confidence.clear();
                    } else {
                        a.drafter.draft(
                            &self.corpus,
                            cap,
                            &mut self.proposal,
                            &mut self.confidence,
                        );
                    }
                    let own_cap = match self.config.drafters {
                        Drafters::Lookup => 0,
                        Drafters::Both | Drafters::Own => cap.min(self.model.own_drafts() as usize),
                    };
                    let own_cost =
                        DraftCost { fixed_ms: 0.0, per_token_ms: self.own_link_ms.unwrap_or(0.0) };
                    let (lookup, own) = match cost {
                        // Before any step was measured, try whole drafts.
                        None => (self.proposal.len(), own_cap),
                        // A step time model that has seen steps of one size
                        // only cannot tell a fixed cost per step from a cost
                        // per row: it puts everything on the rows (base 0),
                        // and then no draft ever pays, so the rows never vary
                        // and it never learns better. A steady batch gets
                        // stuck there. So every fourth sequence in turn drafts
                        // regardless, which varies the rows of the steps.
                        Some(cost) if cost.base_ms <= 0.0 => {
                            if (self.step_no + slot as u64).is_multiple_of(4) {
                                (self.proposal.len(), own_cap)
                            } else {
                                (0, 0)
                            }
                        }
                        Some(cost) => {
                            // A draft row of this sequence costs what a row
                            // at its depth costs; the step without drafts is
                            // the newest token of every generating sequence,
                            // each at its own depth.
                            let per_row = cost.row_ms(reads(a));
                            let base = cost.ms(windows) - per_row * f64::from(generating);
                            let best = |rates: &[f64], drafter| {
                                best_draft_len(
                                    rates,
                                    base,
                                    per_row,
                                    generating,
                                    f64::from(generating),
                                    drafter,
                                )
                            };
                            // The lookup says how sure it is of each token:
                            // from how long the match is and how its earlier
                            // occurrences went on.
                            let lookup = best(&self.confidence, LOOKUP);
                            let own = best(&a.own_acceptance.rates()[..own_cap], own_cost);
                            if own.len > 0 && own.tokens_per_ms > lookup.tokens_per_ms {
                                (0, own.len)
                            } else {
                                (lookup.len, 0)
                            }
                        }
                    };
                    // Where nothing decided yet, a proposal of the lookup
                    // goes first: it is rarer and free.
                    if lookup > 0 {
                        a.draft.extend_from_slice(&self.proposal[..lookup]);
                        spare_logit_rows -= lookup as u32;
                    } else if own > 0 {
                        a.own = true;
                        a.draft.resize(own, 0);
                        a.lookup.extend_from_slice(&self.proposal);
                        spare_logit_rows -= own as u32;
                    }
                }
                Want::Window { drafts: a.draft.len() as u32 }
            } else {
                Want::Prefill { tokens: a.known - a.cells.len() as u32 }
            };
            self.requests.push(Request {
                seq: SeqId(slot as u32),
                want,
                waited_steps: a.waited_steps,
            });
        }
        windows
    }

    /// One turn of the loop with something to do.
    fn step(&mut self, turn: Instant) {
        let t_plan = Instant::now();
        // Those who wait come in before the plan, so that a request that
        // fits shares this step.
        self.parts.clear();
        if !self.waiting.is_empty() {
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
        let windows = self.want();
        let drafting =
            self.requests.iter().any(|r| matches!(r.want, Want::Window { drafts } if drafts > 0));
        // The prompt rows next to generation are sized for the deepest
        // prompt, whose rows cost most: the planner picks among the prompts
        // by other rules, and whichever it serves stays inside the slowdown.
        let deepest = self
            .slots
            .iter()
            .flatten()
            .filter(|a| !a.generating())
            .map(|a| a.cells.len().min(u64::from(self.span)))
            .max()
            .unwrap_or(0);
        let limits = Limits {
            max_rows: if drafting {
                self.max_rows.min(self.model.draft_rows())
            } else {
                self.max_rows
            },
            prefill_rows_with_windows: prefill_rows_at(
                &self.step_time,
                windows,
                deepest,
                self.config.prefill_slowdown,
                self.config.prefill_fallback_rows,
            ),
            starvation_steps: self.config.starvation_steps,
        };
        let planned = plan(&self.requests, &limits);
        for request in &self.requests {
            if let Some(a) = self.slots[request.seq.0 as usize].as_mut() {
                a.waited_steps = a.waited_steps.saturating_add(1);
            }
        }
        for part in &planned.parts {
            if let Some(a) = &self.slots[part.seq.0 as usize] {
                let (slot, seq) = (part.seq.0, a.seq);
                self.parts.push(StepPart { slot, seq, rows: part.rows, kind: part.kind });
            }
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
            match a.cells.append(&mut self.pool, u64::from(part.rows)) {
                Ok(grown) => {
                    debug_assert!(grown.copy.is_none(), "nothing forks a sequence in this runtime");
                    a.draft.truncate(part.rows.saturating_sub(1) as usize);
                    a.own &= !a.draft.is_empty();
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
            return;
        }

        let used = match self.write_record() {
            Ok(used) => used,
            Err(message) => return self.fail(message),
        };
        self.stats.record(Phase::Write, t_write.elapsed().as_nanos() as u64);

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
            let cells = self.slots[part.slot as usize].as_ref().map_or(0, |a| a.cells.len());
            load.plus(load_of(self.span, cells - u64::from(part.rows), part.rows))
        });
        if let Err(error) = self.exec.dispatch(&meta) {
            return self.fail(error.to_string());
        }
        let t_wait = Instant::now();
        self.stats.record(Phase::Dispatch, (t_wait - t_dispatch).as_nanos() as u64);
        let woke = match self.exec.collect(self.config.conductor_spin, &mut self.output) {
            Ok(woke) => woke,
            Err(error) => return self.fail(error.to_string()),
        };
        let t_back = Instant::now();
        self.stats.record(Phase::Wait, (t_back - t_wait).as_nanos() as u64);
        self.stats.counters.conductor_wakes.record(woke);
        // What the model's own drafter took is not a cost of the rows: it is
        // learned on its own, per link of the longest chain of the step.
        let own_links = self
            .parts
            .iter()
            .filter(|p| self.slots[p.slot as usize].as_ref().is_some_and(|a| a.own))
            .map(|p| p.rows.saturating_sub(1))
            .max()
            .unwrap_or(0);
        let draft_ms = self.output.draft_ms;
        if own_links > 0 && draft_ms > 0.0 {
            let counters = &mut self.stats.counters;
            counters.own_draft_steps += 1;
            counters.own_draft_links += u64::from(own_links);
            counters.own_draft_us += (draft_ms * 1e3) as u64;
            let per_link = draft_ms / f64::from(own_links);
            self.own_link_ms = Some(match self.own_link_ms {
                Some(known) => known + 0.1 * (per_link - known),
                None => per_link,
            });
        }
        self.step_time.observe_load(load, (t_back - t_dispatch).as_secs_f64() * 1e3 - draft_ms);
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

        let t_commit = Instant::now();
        self.commit();
        self.stats.record(Phase::Commit, t_commit.elapsed().as_nanos() as u64);

        let t_events = Instant::now();
        self.deliver();
        self.stats.record(Phase::Events, t_events.elapsed().as_nanos() as u64);
        self.stats.record(Phase::Step, turn.elapsed().as_nanos() as u64);
        self.stats.counters.steps += 1;
    }

    /// Writes the step record for the parts in `self.parts`. Returns its
    /// length in bytes.
    fn write_record(&mut self) -> Result<usize, String> {
        let full = |e: record::RecordFull| format!("{e}");
        let rows: u32 = self.parts.iter().map(|p| p.rows).sum();
        let buffer = self.exec.record();
        let mut writer = RecordWriter::new(buffer, self.step_no + 1, self.parts.len() as u32, rows)
            .map_err(full)?;
        self.runs.clear();
        self.samplers.clear();
        self.updates.clear();
        let mut row_index = 0u32;
        let counters = &mut self.stats.counters;
        let window = self.spec.window.max(1) as usize;
        let mut has_prompt = false;
        let mut has_generation = false;
        for part in &self.parts {
            let a = self.slots[part.slot as usize].as_ref().expect("running");
            let first_cell = (a.cells.len() - u64::from(part.rows)) as u32;
            match part.kind {
                PartKind::Prefill { completes } => {
                    has_prompt = true;
                    counters.prefill_rows += u64::from(part.rows);
                    // The rows of a prompt are certain: recurrent state
                    // advances with them.
                    writer.segment(part.slot, first_cell, part.rows, true).map_err(full)?;
                    for k in 0..part.rows {
                        let position = first_cell + k;
                        writer
                            .row(Row {
                                slot: part.slot,
                                position,
                                token: a.tokens[position as usize],
                                context: self.spec.reduce(a.states[position as usize], position),
                            })
                            .map_err(full)?;
                    }
                    if completes {
                        self.runs.push((row_index + part.rows - 1, 1));
                        self.samplers.push(a.sampling.row(first_cell + part.rows - 1));
                    }
                }
                PartKind::Window | PartKind::Score { .. } => {
                    has_generation = true;
                    counters.window_rows += 1;
                    counters.draft_rows += u64::from(part.rows - 1);
                    let newest = a.tokens.len() - 1;
                    // A window with drafts leaves recurrent state alone until
                    // the next step commits the rows that were accepted.
                    writer
                        .segment(part.slot, first_cell, part.rows, part.rows == 1)
                        .map_err(full)?;
                    if a.own {
                        // The draft rows' tokens below are placeholders: the
                        // model's draft block writes them on the device.
                        writer.own_drafts().map_err(full)?;
                    }
                    self.samplers.extend((0..part.rows).map(|k| a.sampling.row(first_cell + k)));
                    writer
                        .row(Row {
                            slot: part.slot,
                            position: first_cell,
                            token: a.tokens[newest],
                            context: self.spec.reduce(a.states[newest], first_cell),
                        })
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
                            .row(Row {
                                slot: part.slot,
                                position,
                                token: draft,
                                context: self.spec.reduce(state, position),
                            })
                            .map_err(full)?;
                    }
                    // Every row of a window is picked, in one run with the
                    // windows before it.
                    match self.runs.last_mut() {
                        Some((first, count)) if *first + *count == row_index => *count += part.rows,
                        _ => self.runs.push((row_index, part.rows)),
                    }
                }
            }
            row_index += part.rows;
        }
        counters.rows += u64::from(row_index);
        if has_prompt && has_generation {
            counters.mixed_steps += 1;
        }
        writer.runs(&self.runs).map_err(full)?;
        writer.samplers(self.samplers.iter().copied()).map_err(full)?;
        writer.commits(&self.commits).map_err(full)?;
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
        writer.finish().map_err(full)
    }

    /// Accepts or rejects what the devices brought back, sequence by
    /// sequence.
    fn commit(&mut self) {
        let mut cursor = 0usize;
        let mut draft_cursor = 0usize;
        let mut row0 = 0u32;
        self.commits.clear();
        for part in &self.parts {
            let first_row = row0;
            row0 += part.rows;
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            let before = a.completion;
            match part.kind {
                PartKind::Prefill { completes } => {
                    // The rows that ran are certain: their tokens are text
                    // now, for every sequence to draft from. A request that
                    // came back computes rows whose tokens are there already.
                    for at in a.drafter.len()..a.cells.len() as usize {
                        a.drafter.push(&mut self.corpus, a.tokens[at]);
                    }
                    if completes {
                        let pick = self.output.picks[cursor];
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
                    cursor += part.rows as usize;
                    // Draft `i` is accepted when it equals what the model
                    // picked after the row before it, and only if all the
                    // drafts before it were.
                    let drafts = part.rows as usize - 1;
                    if a.own {
                        // The drafts the model made on the device came back
                        // with the picks. A model that brings fewer than it
                        // was asked for has none of them accepted.
                        let end = (draft_cursor + drafts).min(self.output.drafts.len());
                        a.draft.clear();
                        a.draft.extend_from_slice(&self.output.drafts[draft_cursor..end]);
                        a.draft.resize(drafts, u32::MAX);
                        draft_cursor = end;
                    }
                    let accepted =
                        (0..drafts).take_while(|&i| picks[i].token == a.draft[i]).count();
                    if drafts > 0 {
                        let source = usize::from(a.own);
                        if a.own {
                            a.own_acceptance.observe(drafts, accepted);
                        } else {
                            // The corpus learns what matches like this one
                            // are worth.
                            a.drafter.observe(&mut self.corpus, drafts, accepted);
                        }
                        let counters = &mut self.stats.counters;
                        counters.drafts_accepted += accepted as u64;
                        counters.drafts_rejected += (drafts - accepted) as u64;
                        // Positions up to the first rejected one were tested.
                        for i in 0..drafts.min(accepted + 1) {
                            let at = i.min(counters.drafts_tested_at[source].len() - 1);
                            counters.drafts_tested_at[source][at] += 1;
                            counters.drafts_accepted_at[source][at] += u64::from(i < accepted);
                        }
                    }
                    // A proposal of the lookup that was not taken learns from
                    // the tokens that are now known.
                    if a.own && !a.lookup.is_empty() {
                        let known = a.lookup.len().min(accepted + 1);
                        let right =
                            (0..known).take_while(|&i| picks[i].token == a.lookup[i]).count();
                        a.drafter.observe(&mut self.corpus, known, right);
                    }
                    // The devices commit the newest token's row and the
                    // accepted drafts to recurrent state with the next step.
                    if drafts > 0 {
                        self.commits.push(Commit {
                            row0: first_row,
                            rows: part.rows,
                            slot: part.slot,
                            accepted: 1 + accepted as u32,
                        });
                    }
                    // The newest token's row and the accepted drafts keep
                    // their cells; the rejected rows are cut from the table.
                    let keep = a.cells.len() - u64::from(part.rows) + 1 + accepted as u64;
                    a.cells.truncate(&mut self.pool, keep);
                    self.scratch_tokens.clear();
                    self.scratch_tokens.extend(picks[..=accepted].iter().map(|p| p.token));
                    a.accept(
                        &*self.model,
                        &self.spec,
                        self.context_length,
                        &mut self.corpus,
                        &self.scratch_tokens,
                    );
                }
            }
            let produced = a.completion - before;
            if produced > 0 {
                self.stats.counters.tokens += u64::from(produced);
                self.admission.generated(a.seq, produced);
            }
        }
    }

    /// Sends the text of the step to the sinks, ends the requests that are
    /// done and the ones whose sink gave up.
    fn deliver(&mut self) {
        for i in 0..self.parts.len() {
            let slot = self.parts[i].slot;
            let a = self.slots[slot as usize].as_mut().expect("running");
            let mut listening = true;
            if !a.emitted.is_empty() {
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
