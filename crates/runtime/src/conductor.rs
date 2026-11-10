//! The thread that decides.
//!
//! Everything the engine knows about its requests lives here, as plain data
//! that only this thread touches: sequences and their page tables, the plan
//! of each step, the step time model, the drafters and their acceptance
//! statistics, the state of every request. Other threads reach it through one
//! channel ([`Command`]) and are reached through sinks and the executor, so
//! there is nothing to lock.
//!
//! One turn of the loop is one step:
//!
//! ```text
//! intake -> plan -> write record -> dispatch ... wait ... collect -> commit -> events
//!           (drafts decided first)                (devices work)    (accept, cut,   (sinks)
//!                                                                    update)
//! ```

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::time::{Duration, Instant};

use ares_api::{EngineError, Event, EventSink, FinishReason, RequestId, Usage};
use ares_sched::{
    Acceptance, DraftCost, Limits, NgramDrafter, PartKind, Request, SeqId, StepTimeModel, Want,
    best_draft_len, plan, prefill_rows_next_to,
};
use ares_state::{PagePool, Sequence};

use crate::config::RuntimeConfig;
use crate::exec::{Clock, ExecError, Executor, StepMeta, StepOutput};
use crate::model::{ContextSpec, Model};
use crate::record::{self, Commit, RecordWriter, Row, TableUpdate};
use crate::text::StopMatcher;
use crate::timing::{Phase, Snapshot, Stats};

/// Drafts cost the model nothing to produce: a lookup.
const LOOKUP: DraftCost = DraftCost { fixed_ms: 0.0, per_token_ms: 0.0 };

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
}

impl StatsCopy {
    pub(crate) fn into_snapshot(self) -> Snapshot {
        Snapshot::new(&self.stats, self.pages_free, self.pages_capacity, self.active)
    }
}

/// A request that has been checked and holds a reservation.
pub(crate) struct Submission {
    pub id: RequestId,
    pub prompt: Vec<u32>,
    pub max_tokens: Option<u32>,
    pub stop: Vec<String>,
    pub sink: EventSink,
    /// Pages reserved for the request's longest possible length.
    pub reserved: u32,
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

/// The books that let [`Engine::submit`](ares_api::Engine::submit) refuse a
/// request at once, without asking the conductor: pages and slots are
/// reserved for a request's longest possible length when it is accepted and
/// given back when it ends, so a running request cannot run out of either.
#[derive(Debug)]
pub(crate) struct Admission {
    pages: AtomicI64,
    slots: AtomicI64,
}

impl Admission {
    pub fn new(pages: u32, slots: u32) -> Self {
        Self { pages: AtomicI64::new(i64::from(pages)), slots: AtomicI64::new(i64::from(slots)) }
    }

    /// Takes `pages` and a slot, or takes nothing and says no.
    pub fn try_reserve(&self, pages: u32) -> bool {
        if self.slots.fetch_sub(1, Ordering::AcqRel) <= 0 {
            self.slots.fetch_add(1, Ordering::AcqRel);
            return false;
        }
        let need = i64::from(pages);
        let mut free = self.pages.load(Ordering::Acquire);
        loop {
            if free < need {
                self.slots.fetch_add(1, Ordering::AcqRel);
                return false;
            }
            match self.pages.compare_exchange_weak(
                free,
                free - need,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(now) => free = now,
            }
        }
    }

    pub fn release(&self, pages: u32) {
        self.pages.fetch_add(i64::from(pages), Ordering::AcqRel);
        self.slots.fetch_add(1, Ordering::AcqRel);
    }
}

/// A request that is running.
struct Active {
    id: RequestId,
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
    drafter: NgramDrafter,
    acceptance: Acceptance,
    stop: StopMatcher,
    max_tokens: Option<u32>,
    completion: u32,
    reserved: u32,
    waited_steps: u32,
    /// Drafts for the coming step; cut to the rows the plan grants.
    draft: Vec<u32>,
    /// Bytes of generated text that are not whole characters yet.
    pending: Vec<u8>,
    /// Text produced this step, not yet delivered.
    emitted: String,
    finished: Option<FinishReason>,
}

impl Active {
    fn generating(&self) -> bool {
        self.cells.len() >= u64::from(self.prompt_len)
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
            self.drafter.push(token);
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
    rows: u32,
    kind: PartKind,
}

pub(crate) struct Conductor {
    model: Arc<dyn Model>,
    spec: ContextSpec,
    config: RuntimeConfig,
    context_length: u32,
    max_rows: u32,
    exec: Box<dyn Executor>,
    /// The executor's clock: device threads stamp their events with it.
    clock: Clock,
    rx: Receiver<Command>,
    admission: Arc<Admission>,
    pool: PagePool,
    slots: Vec<Option<Active>>,
    free_slots: Vec<u32>,
    by_id: HashMap<RequestId, u32>,
    active: usize,
    step_time: StepTimeModel,
    stats: Stats,
    step_no: u64,
    after_idle: bool,
    /// When the last device finished the previous step, on the executor's clock.
    last_done_ns: Option<u64>,
    failed: Option<String>,
    shutting_down: bool,
    // Scratch that lives across steps so that a step allocates nothing for it.
    requests: Vec<Request>,
    parts: Vec<StepPart>,
    updates: Vec<(u32, Range<usize>)>,
    runs: Vec<(u32, u32)>,
    /// The outcome of this step's draft windows, for the next step's record.
    commits: Vec<Commit>,
    output: StepOutput,
    scratch_tokens: Vec<u32>,
}

impl Conductor {
    pub(crate) fn new(
        model: Arc<dyn Model>,
        config: RuntimeConfig,
        exec: Box<dyn Executor>,
        rx: Receiver<Command>,
        admission: Arc<Admission>,
    ) -> Self {
        let max_rows = match config.max_rows {
            0 => model.max_rows(),
            n => n.min(model.max_rows()),
        };
        let slots = config.max_slots as usize;
        Self {
            spec: model.context(),
            context_length: model.card().context_length,
            pool: PagePool::new(config.pages, config.cells_per_page),
            slots: (0..slots).map(|_| None).collect(),
            free_slots: (0..config.max_slots).rev().collect(),
            by_id: HashMap::with_capacity(slots),
            stats: Stats::new(exec.devices(), config.stats_window),
            output: StepOutput {
                picks: Vec::with_capacity(max_rows as usize),
                devices: Vec::with_capacity(exec.devices()),
            },
            requests: Vec::with_capacity(slots),
            parts: Vec::with_capacity(max_rows as usize),
            updates: Vec::with_capacity(slots),
            runs: Vec::with_capacity(max_rows as usize),
            commits: Vec::with_capacity(slots),
            scratch_tokens: Vec::with_capacity(64),
            model,
            config,
            max_rows,
            clock: exec.clock(),
            exec,
            rx,
            admission,
            active: 0,
            step_time: StepTimeModel::default(),
            step_no: 0,
            after_idle: true,
            last_done_ns: None,
            failed: None,
            shutting_down: false,
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
                // Nothing to do: sleep until somebody asks for something.
                match self.rx.recv() {
                    Ok(command) => {
                        self.after_idle = true;
                        self.handle(command);
                    }
                    Err(_) => break,
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
            if self.active == 0 {
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
            Command::Submit(submission) => self.admit(*submission),
            Command::Cancel(id) => {
                if let Some(&slot) = self.by_id.get(&id) {
                    self.retire(slot, FinishReason::Cancelled);
                }
            }
            Command::Stats { reset, reply } => {
                self.stats.counters.device_wakes = self.exec.device_wakes();
                let copy = StatsCopy {
                    stats: self.stats.clone(),
                    pages_free: self.pool.free_pages(),
                    pages_capacity: self.pool.capacity(),
                    active: self.active as u32,
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

    fn admit(&mut self, submission: Submission) {
        let Submission { id, prompt, max_tokens, stop, sink, reserved } = submission;
        let refuse = |error: EngineError| {
            let _ = sink.send(Event::Error(error));
            self.admission.release(reserved);
        };
        if let Some(why) = &self.failed {
            return refuse(EngineError::Internal(why.clone()));
        }
        if self.by_id.contains_key(&id) {
            return refuse(EngineError::InvalidRequest(format!("request id {id} is in use")));
        }
        let Some(slot) = self.free_slots.pop() else {
            self.stats.counters.requests_refused += 1;
            return refuse(EngineError::Overloaded);
        };
        let mut states = Vec::with_capacity(prompt.len() + 64);
        let mut state = self.spec.start();
        for i in 0..prompt.len() {
            state = self.spec.advance(state, &prompt[..=i]);
            states.push(state);
        }
        let mut drafter = NgramDrafter::new(self.config.ngram);
        drafter.extend(&prompt);
        let mut tokens = Vec::with_capacity(prompt.len() + 64);
        tokens.extend_from_slice(&prompt);
        self.slots[slot as usize] = Some(Active {
            id,
            sink,
            cells: Sequence::new(),
            prompt_len: prompt.len() as u32,
            tokens,
            states,
            drafter,
            acceptance: Acceptance::new(self.config.max_draft as usize, self.config.draft_prior),
            stop: StopMatcher::new(&stop),
            max_tokens,
            completion: 0,
            reserved,
            waited_steps: 0,
            draft: Vec::with_capacity(self.config.max_draft as usize),
            pending: Vec::new(),
            emitted: String::new(),
            finished: None,
        });
        self.by_id.insert(id, slot);
        self.active += 1;
    }

    /// Ends the request in `slot`: tells its sink (if it still listens),
    /// gives its pages and its reservation back.
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
        self.admission.release(a.reserved);
        self.free_slots.push(slot);
        self.active -= 1;
        match finish {
            FinishReason::Cancelled => self.stats.counters.requests_cancelled += 1,
            _ => self.stats.counters.requests_finished += 1,
        }
    }

    fn finish_everything(&mut self, finish: FinishReason) {
        for slot in 0..self.slots.len() as u32 {
            self.retire(slot, finish);
        }
    }

    /// Ends every request with an error, after the devices failed. The
    /// engine accepts no more requests.
    fn fail(&mut self, message: String) {
        for slot in 0..self.slots.len() as u32 {
            if let Some(a) = &self.slots[slot as usize] {
                let _ = a.sink.send(Event::Error(EngineError::Internal(message.clone())));
                self.retire(slot, FinishReason::Cancelled);
            }
        }
        self.failed = Some(message);
    }

    /// Decides how many drafts each generating sequence asks for, and
    /// fills `requests` for the planner.
    fn want(&mut self) -> u32 {
        self.requests.clear();
        let generating = self.slots.iter().flatten().filter(|a| a.generating()).count() as u32;
        let time = self.step_time.base_and_per_row();
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
                if self.config.speculation && cap > 0 {
                    let proposal = a.drafter.draft(cap);
                    let take = match time {
                        _ if proposal.is_empty() => 0,
                        // Before any step was measured, try the whole draft.
                        None => proposal.len(),
                        // A step time model that has seen steps of one size
                        // only cannot tell a fixed cost per step from a cost
                        // per row: it puts everything on the rows (base 0),
                        // and then no draft ever pays, so the rows never vary
                        // and it never learns better. A steady batch gets
                        // stuck there. So every fourth sequence in turn drafts
                        // regardless, which varies the rows of the steps.
                        Some((base, _)) if base <= 0.0 => {
                            if (self.step_no + slot as u64).is_multiple_of(4) {
                                proposal.len()
                            } else {
                                0
                            }
                        }
                        Some((base, per_row)) => {
                            best_draft_len(
                                &a.acceptance.rates()[..proposal.len()],
                                base,
                                per_row,
                                generating,
                                f64::from(generating),
                                LOOKUP,
                            )
                            .len
                        }
                    };
                    a.draft.extend_from_slice(&proposal[..take]);
                    spare_logit_rows -= take as u32;
                }
                Want::Window { drafts: a.draft.len() as u32 }
            } else {
                Want::Prefill { tokens: a.prompt_len - a.cells.len() as u32 }
            };
            self.requests.push(Request {
                seq: SeqId(slot as u32),
                want,
                waited_steps: a.waited_steps,
            });
        }
        generating
    }

    /// One turn of the loop with something to do.
    fn step(&mut self, turn: Instant) {
        let t_plan = Instant::now();
        let generating = self.want();
        let drafting =
            self.requests.iter().any(|r| matches!(r.want, Want::Window { drafts } if drafts > 0));
        let limits = Limits {
            max_rows: if drafting {
                self.max_rows.min(self.model.draft_rows())
            } else {
                self.max_rows
            },
            prefill_rows_with_windows: prefill_rows_next_to(
                &self.step_time,
                generating,
                self.config.prefill_slowdown,
                self.config.prefill_fallback_rows,
            ),
            starvation_steps: self.config.starvation_steps,
        };
        let planned = plan(&self.requests, &limits);
        self.stats.record(Phase::Plan, t_plan.elapsed().as_nanos() as u64);

        // Pages first: a sequence that cannot grow is dropped from the step
        // before anything is written.
        let t_write = Instant::now();
        self.parts.clear();
        for part in &planned.parts {
            let slot = part.seq.0;
            let a = self.slots[slot as usize].as_mut().expect("planned sequences are running");
            a.waited_steps = 0;
            match a.cells.append(&mut self.pool, u64::from(part.rows)) {
                Ok(grown) => {
                    debug_assert!(grown.copy.is_none(), "nothing forks a sequence in this runtime");
                    a.draft.truncate(part.rows.saturating_sub(1) as usize);
                    self.parts.push(StepPart { slot, rows: part.rows, kind: part.kind });
                }
                Err(error) => {
                    // Admission reserves every sequence's longest length, so
                    // this is a bookkeeping bug and not load.
                    let _ = a.sink.send(Event::Error(EngineError::Internal(format!(
                        "the page pool ran out under a reservation: {error}"
                    ))));
                    self.retire(slot, FinishReason::Cancelled);
                }
            }
        }
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
        let rows: u32 = self.parts.iter().map(|p| p.rows).sum();
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
        self.step_time.observe(rows, (t_back - t_dispatch).as_secs_f64() * 1e3);
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
        let mut row0 = 0u32;
        self.commits.clear();
        for part in &self.parts {
            let first_row = row0;
            row0 += part.rows;
            let a = self.slots[part.slot as usize].as_mut().expect("running");
            match part.kind {
                PartKind::Prefill { completes: false } => {}
                PartKind::Prefill { completes: true } => {
                    let pick = self.output.picks[cursor];
                    cursor += 1;
                    a.accept(&*self.model, &self.spec, self.context_length, &[pick.token]);
                    self.stats.counters.tokens += 1;
                }
                PartKind::Window | PartKind::Score { .. } => {
                    let picks = &self.output.picks[cursor..cursor + part.rows as usize];
                    cursor += part.rows as usize;
                    // Draft `i` is accepted when it equals what the model
                    // picked after the row before it, and only if all the
                    // drafts before it were.
                    let drafts = part.rows as usize - 1;
                    let accepted =
                        (0..drafts).take_while(|&i| picks[i].token == a.draft[i]).count();
                    if drafts > 0 {
                        a.acceptance.observe(drafts, accepted);
                        self.stats.counters.drafts_accepted += accepted as u64;
                        self.stats.counters.drafts_rejected += (drafts - accepted) as u64;
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
                    let before = a.completion;
                    a.accept(&*self.model, &self.spec, self.context_length, &self.scratch_tokens);
                    self.stats.counters.tokens += u64::from(a.completion - before);
                }
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
    fn a_reservation_takes_pages_and_a_slot_and_gives_both_back() {
        let admission = Admission::new(10, 2);
        assert!(admission.try_reserve(6));
        assert!(!admission.try_reserve(5), "only 4 pages are left");
        assert!(admission.try_reserve(4));
        assert!(!admission.try_reserve(0), "no slot is left, however few pages are asked for");
        admission.release(6);
        assert!(admission.try_reserve(6), "the pages and the slot are back");
        assert!(!admission.try_reserve(1));
    }

    #[test]
    fn a_refused_reservation_takes_nothing() {
        let admission = Admission::new(4, 3);
        assert!(!admission.try_reserve(5));
        // All three slots and all four pages must still be there.
        assert!(admission.try_reserve(1));
        assert!(admission.try_reserve(1));
        assert!(admission.try_reserve(2));
    }

    #[test]
    fn threads_racing_for_pages_never_hold_more_than_exist() {
        use std::sync::atomic::AtomicU32;
        let admission = Arc::new(Admission::new(100, 1000));
        let held = Arc::new(AtomicU32::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (admission, held) = (admission.clone(), held.clone());
                std::thread::spawn(move || {
                    let mut most = 0;
                    for _ in 0..20_000 {
                        if admission.try_reserve(7) {
                            most = most.max(held.fetch_add(7, Ordering::SeqCst) + 7);
                            held.fetch_sub(7, Ordering::SeqCst);
                            admission.release(7);
                        }
                    }
                    most
                })
            })
            .collect();
        for thread in threads {
            assert!(thread.join().unwrap() <= 100);
        }
        assert!(admission.try_reserve(100), "everything was given back");
    }
}
