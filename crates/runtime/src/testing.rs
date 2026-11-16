//! Executors without a GPU, for testing the conductor and for measuring it
//! alone.
//!
//! [`HostExecutor`] runs no model. It reads the rows of the step record and
//! answers each with [`SyntheticModel::host_pick`], a fixed function of the
//! row's token and context, so the conductor's planning, page handling,
//! speculation, stopping and event delivery run exactly as with devices.
//! It is not an inference path: its picks mean nothing (a row with a mask picks
//! the first token its mask allows from the pick on), and it exists so that
//! every decision the conductor makes can be checked by the default
//! `cargo test`, and so that the benchmark can show what the conductor costs
//! with devices that take no time.
//!
//! # The oracle
//!
//! Given the pool's shape ([`HostExecutor::with_pool`]) the executor also
//! plays a device that keeps state, so that a mistake in what the conductor
//! does with pages, checkpoints and slots shows in the text and not only in
//! a counter. It keeps what a device keeps:
//!
//! - the **cells** of every page: a cell holds the token of its row and a hash
//!   of every token of its sequence up to and including that one, in order,
//!   and the cell before it;
//! - for a model with recurrent state (`prefix_cache: Some(bytes)` with
//!   `bytes > 0`) the **state of every slot**, which has absorbed the same
//!   tokens in order, and the checkpoint places that state is saved to and
//!   restored from.
//!
//! It carries out the record's copies and its replay segments in the order the
//! record documents, and a row reads what it may read: its slot's state, and
//! every cell of its sequence before it through the page table. A row is
//! consistent when the state is at the row's position, the cells before it
//! form one unbroken history that the state continues, the context the
//! conductor derived from the sequence's tokens (which the oracle derives
//! again from what it holds) is the same, and a replayed row finds the cell it
//! would have written. The pick of a consistent row is `host_pick(token,
//! context)`, as without the oracle, so text behaves as it always did. An
//! inconsistent row, and every row that later reads what it left (its slot's
//! state, the cells it wrote, the checkpoints it was saved in), picks another
//! token than the model would, and the problem is put in the log
//! ([`HostExecutor::problems`]).
//!
//! A restore of the wrong place, a replay that is missing or runs over too
//! many rows, a write into a page that others read, a tail that does not hold
//! what it says, a slot that was taken before its state was saved: each of
//! them changes the generated tokens.
//!
//! Given the books of page-locked RAM as well ([`HostExecutor::with_ram`]),
//! the oracle plays the strictest device those books allow: a checkpoint
//! place holds memory from the first copy into it, booked as a device part
//! books its chunks, and loses it, with what it held, as soon as the
//! conductor has called the place unused. A save into a place that is not in
//! use, and a checkpoint that leaves such a place for the drive, are problems;
//! a restore from one finds nothing.
//!
//! # What a step costs
//!
//! The conductor decides the drafts of a step and the turns of prompts and
//! generation from what steps cost, which on devices it measures with its
//! clock. A test gives the executor that cost ([`HostExecutor::with_delay`],
//! [`HostExecutor::with_step_cost`], [`HostExecutor::with_link_cost`]) and the
//! executor tells the conductor exactly that ([`StepOutput::step_ms`]), so the
//! prices the policy sees are the ones the test wrote down, however busy the
//! machine is and however late a sleeping thread is woken. The executor also
//! takes the time, so that requests arrive and are cancelled while steps run;
//! a test that needs no time to pass asks for
//! [`HostExecutor::without_sleeping`], and one that wants the conductor's own
//! measuring asks for [`HostExecutor::timed_by_the_conductor`]. An executor
//! that was given no cost says nothing, and the conductor measures what the
//! host needs for a step.

mod faulty;

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::exec::{Clock, ExecError, Executor, Pick, StepMeta, StepOutput};
use crate::model::{ContextSpec, Model};
use crate::ram::PinnedRam;
use crate::record::{RecordView, StateCopyKind};
use crate::sync::{WakeCounts, Woke};
use crate::synthetic::SyntheticModel;
use crate::timing::DeviceTimes;

pub use faulty::{Failure, Fault, Faulty};

/// A latch a test closes to hold steps back and opens to let them finish.
#[derive(Clone, Debug)]
pub struct Gate(Arc<(Mutex<bool>, Condvar)>);

impl Gate {
    /// A gate that starts open.
    pub fn new() -> Self {
        Self(Arc::new((Mutex::new(true), Condvar::new())))
    }

    pub fn close(&self) {
        *self.0.0.lock().expect("gate") = false;
    }

    pub fn open(&self) {
        *self.0.0.lock().expect("gate") = true;
        self.0.1.notify_all();
    }

    fn wait(&self, limit: Duration) -> bool {
        let (lock, condvar) = &*self.0;
        let (_open, timeout) = condvar
            .wait_timeout_while(lock.lock().expect("gate"), limit, |open| !*open)
            .expect("gate");
        !timeout.timed_out()
    }
}

impl Default for Gate {
    fn default() -> Self {
        Self::new()
    }
}

/// What the page tables of a step look like on the device, as the test
/// executor mirrors them: `[slot][entry]`.
pub type TableMirror = Arc<Mutex<Vec<Vec<u32>>>>;

type Observer = Box<dyn FnMut(&RecordView<'_>) + Send>;

/// What the oracle found wrong, as text, the first few of them.
pub type Problems = Arc<Mutex<Vec<String>>>;

/// How many problems the oracle writes down.
const MAX_PROBLEMS: usize = 16;

/// An executor that computes picks on the host.
pub struct HostExecutor {
    model: Arc<SyntheticModel>,
    devices: usize,
    record: Vec<u8>,
    picks: Vec<Pick>,
    delay: Duration,
    /// The time of a step by its rows, in place of `delay`.
    step_cost: Option<Box<dyn Fn(u32) -> Duration + Send>>,
    /// How long the step that was dispatched takes.
    step_delay: Duration,
    /// Whether a step takes the time it costs, and whether the executor tells
    /// the conductor what it cost.
    sleeps: bool,
    states: bool,
    /// What a link of the model's draft chain takes on the device, and what
    /// the chain of the step that was dispatched drafted and took.
    link: Duration,
    drafts: Vec<u32>,
    draft_probs: Vec<f32>,
    draft_ms: f64,
    gate: Option<Gate>,
    observer: Option<Observer>,
    dispatched_at: Instant,
    clock: Clock,
    tables: TableMirror,
    fail_at_step: Option<u64>,
    failing: bool,
    oracle: Option<Oracle>,
    problems: Problems,
    /// The books of the page-locked RAM the checkpoint places lie in.
    ram: Option<PinnedRam>,
    /// Which rows of a window (by step number and row) the model leaves
    /// without what it would have waited for, and the rows of the step that
    /// was dispatched that are not exact for it.
    leaving: Option<Box<dyn Fn(u64, u32) -> bool + Send>>,
    inexact: Vec<u32>,
}

impl std::fmt::Debug for HostExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostExecutor").field("delay", &self.delay).finish_non_exhaustive()
    }
}

impl HostExecutor {
    /// A row that carries a mask picks the first token the mask allows from
    /// its own pick on (and on around the vocabulary): still a fixed function
    /// of the row, and a pick the mask forbids is never made.
    fn apply_masks(view: &RecordView<'_>, picks: &mut [Pick], vocab: u32) {
        for (index, pick) in picks.iter_mut().enumerate() {
            let Some(mask) = view.sampler(index as u32).mask() else { continue };
            let allowed = |t: u32| view.mask_word(mask, t / 32) >> (t % 32) & 1 != 0;
            if let Some(token) = (0..vocab).map(|k| (pick.token + k) % vocab).find(|&t| allowed(t))
            {
                pick.token = token;
            }
        }
    }

    /// An executor for `model`, which sizes the record for `max_rows` rows and
    /// `max_slots` sequences.
    pub fn new(model: Arc<SyntheticModel>, max_rows: u32, max_slots: u32) -> Self {
        let record_bytes =
            crate::record::max_bytes(max_rows, max_slots, model.logit_rows(), model.vocab());
        Self {
            model,
            devices: 1,
            record: vec![0; record_bytes],
            picks: Vec::new(),
            delay: Duration::ZERO,
            step_cost: None,
            step_delay: Duration::ZERO,
            sleeps: true,
            states: true,
            link: Duration::ZERO,
            drafts: Vec::new(),
            draft_probs: Vec::new(),
            draft_ms: 0.0,
            gate: None,
            observer: None,
            dispatched_at: Instant::now(),
            clock: Clock::new(),
            tables: Arc::new(Mutex::new(vec![Vec::new(); max_slots as usize])),
            fail_at_step: None,
            failing: false,
            oracle: None,
            problems: Arc::default(),
            ram: None,
            leaving: None,
            inexact: Vec::new(),
        }
    }

    /// Makes the executor an oracle for a runtime whose pool has `pages` pages
    /// of `cells_per_page` cells (see the module documentation). Needed for
    /// a model that takes part in the prefix cache.
    pub fn with_pool(mut self, pages: u32, cells_per_page: u32) -> Self {
        let slots = self.tables.lock().expect("tables").len();
        self.oracle = Some(Oracle::new(&self.model, pages, cells_per_page, slots));
        self
    }

    /// Makes the checkpoint places lie in the page-locked memory of `ram`,
    /// as a device's do: the conductor keeps to the allowance of these books,
    /// and the oracle holds the memory of a place only while the conductor
    /// has the place in use (see the module documentation).
    pub fn with_ram(mut self, ram: PinnedRam) -> Self {
        // The places are booked here as they are first used, as a device
        // part locks them: the books learn which hold memory.
        ram.host_places(self.model.config().prefix_cache.unwrap_or(0));
        self.ram = Some(ram);
        self
    }

    /// What the oracle found wrong so far: empty when everything the
    /// conductor did was consistent.
    pub fn problems(&self) -> Problems {
        self.problems.clone()
    }

    /// Pretends to be `n` devices, for the shape of the timing tables.
    pub fn with_devices(mut self, n: usize) -> Self {
        self.devices = n;
        self
    }

    /// Makes every step cost `delay` whatever it carries, and take at least
    /// that long, as a device would.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Makes a step take `base` and `per_row` for each of its rows, as a device
    /// does (a pass over the weights, and a cost for every row), instead of
    /// the same time whatever it carries.
    pub fn with_row_cost(self, base: Duration, per_row: Duration) -> Self {
        self.with_step_cost(move |rows| base + per_row * rows)
    }

    /// Makes a step take what `cost` says of its rows, as a device does:
    /// not a line, but a table with edges, such as a second launch of the
    /// weight kernels from the ninth row on.
    pub fn with_step_cost(mut self, cost: impl Fn(u32) -> Duration + Send + 'static) -> Self {
        self.step_cost = Some(Box::new(cost));
        self
    }

    /// Makes each link of the model's own draft chain take `link` on the
    /// device, in addition to the step: the chain of a step is as long as the
    /// longest chain its drafter made (a drafter that reports probabilities,
    /// [`crate::SyntheticConfig::mtp_probability`], ends a chain at the
    /// record's draft bars).
    pub fn with_link_cost(mut self, link: Duration) -> Self {
        self.link = link;
        self
    }

    /// Makes the steps cost what they cost without taking the time: the
    /// executor tells the conductor what a step took and does not sleep for
    /// it. For tests of what the conductor decides from the prices, in which
    /// nothing has to happen while a step runs.
    pub fn without_sleeping(mut self) -> Self {
        self.sleeps = false;
        self
    }

    /// Leaves the time of a step to the conductor's clock, as devices do: the
    /// executor takes the time a step costs and says nothing about it. What
    /// the conductor measures then is never less than the cost and otherwise
    /// depends on how busy the machine is, so a test with such an executor
    /// asserts only what holds at any speed.
    pub fn timed_by_the_conductor(mut self) -> Self {
        self.states = false;
        self
    }

    /// Holds every step back while `gate` is closed.
    pub fn with_gate(mut self, gate: Gate) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Calls `observer` with the record of every step, before it runs.
    pub fn with_observer(mut self, observer: impl FnMut(&RecordView<'_>) + Send + 'static) -> Self {
        self.observer = Some(Box::new(observer));
        self
    }

    /// Leaves rows of windows without what they would have waited for, as a
    /// model does whose rows that verify drafts never wait for the file:
    /// `left(step, row)` names them. Such a row (past the first of a window)
    /// and every row of its window after it pick another token than the
    /// model's, and the executor says that they are not exact
    /// ([`StepOutput::inexact`]).
    pub fn leaving(mut self, left: impl Fn(u64, u32) -> bool + Send + 'static) -> Self {
        self.leaving = Some(Box::new(left));
        self
    }

    /// Makes the step with this number fail, as a device error would.
    pub fn failing_at(mut self, step: u64) -> Self {
        self.fail_at_step = Some(step);
        self
    }

    /// The page tables as the "device" holds them.
    pub fn tables(&self) -> TableMirror {
        self.tables.clone()
    }
}

impl Executor for HostExecutor {
    fn devices(&self) -> usize {
        self.devices
    }

    fn clock(&self) -> Clock {
        self.clock
    }

    fn record(&mut self) -> &mut [u8] {
        &mut self.record
    }

    fn dispatch(&mut self, meta: &StepMeta) -> Result<(), ExecError> {
        let broken =
            |e: crate::record::RecordError| ExecError::Device { device: 0, message: e.to_string() };
        // The model's own drafts: a step that only drafts makes the chains of
        // the windows that take them and hands the drafts back.
        self.drafts.clear();
        self.draft_probs.clear();
        let mut longest = 0;
        let mut patches = Vec::new();
        {
            let view = RecordView::new(&self.record[..meta.record_bytes]).map_err(broken)?;
            let reports = self.model.config().mtp_probability.is_some();
            let mut bar = 0;
            for i in (0..view.segments()).filter(|i| view.model_drafts(*i)) {
                let segment = view.segment(i);
                let newest = view.row(segment.row0);
                let links = segment.n_rows - 1;
                let chain =
                    self.model.host_chain(newest.token, newest.context, newest.position, links);
                // A drafter that knows its probabilities ends the chain where
                // the record's bars say one more link does not pay; the rows
                // of the links it did not make keep what the record had.
                let made = if reports && view.draft_bars() >= bar + links {
                    let needs: Vec<f32> = (bar..bar + links).map(|k| view.draft_need(k)).collect();
                    let chances = chain.iter().map(|(_, p)| f64::from(view.draft_chance(*p)));
                    ares_sched::links_made(chances, &needs) as u32
                } else {
                    links
                };
                bar += links;
                longest = longest.max(made);
                for (j, (draft, p)) in chain.iter().enumerate() {
                    if j as u32 >= made {
                        self.drafts.push(u32::MAX);
                        self.draft_probs.push(f32::NAN);
                        continue;
                    }
                    let position = newest.position + 1 + j as u32;
                    patches.push((
                        segment.row0 + 1 + j as u32,
                        *draft,
                        self.model.context_of(*draft, position),
                    ));
                    self.drafts.push(*draft);
                    self.draft_probs.push(*p);
                }
            }
            if !reports {
                self.draft_probs.clear();
            }
        }
        for (row, token, context) in patches {
            RecordView::set_row_inputs(&mut self.record, row, token, context);
        }
        let draft_time = self.link * longest;
        self.draft_ms = draft_time.as_secs_f64() * 1e3;
        let view = RecordView::new(&self.record[..meta.record_bytes]).map_err(broken)?;
        if let Some(observer) = &mut self.observer {
            observer(&view);
        }
        self.failing = self.fail_at_step == Some(meta.step);
        // A step that only drafts runs no row: it takes its chain.
        self.step_delay = draft_time
            + match (&self.step_cost, view.drafts_only()) {
                (_, true) => Duration::ZERO,
                (Some(cost), false) => cost(view.rows()),
                (None, false) => self.delay,
            };
        self.picks.clear();
        let mut tables = self.tables.lock().expect("tables");
        // The page-table updates, as the device copies them.
        let mut taken = 0;
        for i in 0..view.updates() {
            let update = view.update(i);
            let table = &mut tables[update.slot as usize];
            table.resize((update.first + update.count) as usize, 0);
            for k in 0..update.count {
                table[(update.first + k) as usize] = view.entry(taken + k);
            }
            taken += update.count;
        }
        match &mut self.oracle {
            Some(oracle) => {
                oracle.give_back(self.ram.as_ref());
                oracle.step(
                    &self.model,
                    &view,
                    &tables,
                    &mut self.picks,
                    self.ram.as_ref(),
                    &self.problems,
                )
            }
            None if view.drafts_only() => {}
            None => {
                if self.model.config().prefix_cache.is_some() {
                    return Err(ExecError::Device {
                        device: 0,
                        message: "the model takes part in the prefix cache, and the host executor \
                                  was built without the pool's shape (HostExecutor::with_pool)"
                            .into(),
                    });
                }
                for i in 0..view.runs() {
                    let (first, count) = view.run(i);
                    for row in first..first + count {
                        let row = view.row(row);
                        let (token, logit) = self.model.host_pick(row.token, row.context);
                        self.picks.push(Pick { token, logit, aux: 0.0 });
                    }
                }
            }
        }
        drop(tables);
        Self::apply_masks(&view, &mut self.picks, self.model.config().vocab());
        // Rows the model left: from the first in a window on, another token.
        self.inexact.clear();
        if let Some(left) = &self.leaving {
            let mut pick_of = std::collections::HashMap::new();
            for run in 0..view.runs() {
                let (first, count) = view.run(run);
                for row in first..first + count {
                    pick_of.insert(row, pick_of.len());
                }
            }
            let vocab = self.model.config().vocab();
            for i in 0..view.segments() {
                let segment = view.segment(i);
                let rows = segment.row0 + 1..segment.row0 + segment.n_rows;
                if segment.direct || view.drafts_only() {
                    continue;
                }
                let Some(first) = rows.clone().find(|row| left(meta.step, *row)) else { continue };
                for row in first..segment.row0 + segment.n_rows {
                    self.inexact.push(row);
                    if let Some(pick) = pick_of.get(&row).and_then(|at| self.picks.get_mut(*at)) {
                        pick.token = (pick.token + 1) % vocab;
                    }
                }
            }
        }
        self.dispatched_at = Instant::now();
        Ok(())
    }

    fn collect(&mut self, _spin: Duration, out: &mut StepOutput) -> Result<Woke, ExecError> {
        if let Some(gate) = &self.gate
            && !gate.wait(Duration::from_secs(20))
        {
            return Err(ExecError::Device {
                device: 0,
                message: "the test never opened the gate".into(),
            });
        }
        if self.sleeps
            && let Some(left) = self.step_delay.checked_sub(self.dispatched_at.elapsed())
        {
            std::thread::sleep(left);
        }
        if self.failing {
            return Err(ExecError::Device { device: 0, message: "injected failure".into() });
        }
        out.picks.clear();
        out.picks.extend_from_slice(&self.picks);
        out.drafts.clear();
        out.drafts.extend_from_slice(&self.drafts);
        out.draft_probs.clear();
        out.draft_probs.extend_from_slice(&self.draft_probs);
        out.draft_ms = self.draft_ms;
        out.inexact.clear();
        out.inexact.extend_from_slice(&self.inexact);
        // A step that was given a cost took exactly that, whatever the clock
        // says; one that was given none is the conductor's to measure.
        let priced = self.step_cost.is_some() || !self.delay.is_zero();
        out.step_ms = (self.states && priced).then_some(self.step_delay.as_secs_f64() * 1e3);
        out.devices.clear();
        let finished_at_ns = self.clock.now_ns();
        out.devices.resize(self.devices, DeviceTimes { finished_at_ns, ..DeviceTimes::default() });
        Ok(Woke::Spinning)
    }

    fn transfer(&mut self, batch: &mut [crate::Transfer]) -> Result<(), ExecError> {
        if let Some(oracle) = &mut self.oracle {
            oracle.give_back(self.ram.as_ref());
        }
        for transfer in batch {
            transfer.ok = match &mut self.oracle {
                Some(oracle) if self.model.config().durable.is_some() => {
                    oracle.transfer(transfer, self.ram.as_ref(), &self.problems)
                }
                _ => false,
            };
        }
        Ok(())
    }

    fn checkpoint_ram(&self) -> u64 {
        self.ram.as_ref().map_or(0, PinnedRam::checkpoints)
    }

    fn ram(&self) -> Option<PinnedRam> {
        self.ram.clone()
    }

    fn device_wakes(&self) -> WakeCounts {
        WakeCounts::default()
    }

    fn shutdown(&mut self) {}
}

// --- the oracle ------------------------------------------------------------

/// The hash every state and cell starts from: the state of a sequence that has
/// read nothing.
const ROOT: u64 = 0x5eed_0123_4567_89ab;

/// The hash of a history that `token` follows.
fn link(chain: u64, token: u32) -> u64 {
    let mut h = (chain ^ u64::from(token).wrapping_mul(0x9e37_79b9_7f4a_7c15)).rotate_left(23);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// The recurrent state of a sequence as the oracle keeps it: everything the
/// sequence has read, in order.
#[derive(Clone, Debug)]
struct State {
    /// How many tokens it has absorbed.
    pos: u32,
    /// A hash of all of them, in order.
    chain: u64,
    /// The context of the last of them, as the conductor derives it from the
    /// sequence's tokens ([`ContextSpec::advance`]), and the tokens that
    /// derivation reads.
    ctx: u32,
    recent: Vec<u32>,
    /// Something that went into it was wrong.
    poisoned: bool,
}

impl State {
    fn empty(spec: &ContextSpec) -> Self {
        Self { pos: 0, chain: ROOT, ctx: spec.start(), recent: Vec::new(), poisoned: false }
    }

    /// The state after `token`.
    fn absorbed(&self, spec: &ContextSpec, token: u32) -> Self {
        let keep = spec.window.max(1) as usize;
        let mut recent = self.recent.clone();
        recent.push(token);
        if recent.len() > keep {
            recent.remove(0);
        }
        let ctx = spec.advance(self.ctx, &recent);
        Self {
            pos: self.pos + 1,
            chain: link(self.chain, token),
            ctx,
            recent,
            poisoned: self.poisoned,
        }
    }
}

/// The bytes of a cell and of a state without its recent tokens, as the
/// oracle writes them for the level of the prefix cache on SSD.
pub(crate) const CELL_BYTES: usize = 32;
pub(crate) const STATE_BYTES: usize = 32;

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

fn long(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"))
}

/// What one position of a sequence stores: the key and value of a real model.
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    written: bool,
    poisoned: bool,
    token: u32,
    ctx: u32,
    /// The hash of the history before the cell's token, and with it.
    prev: u64,
    chain: u64,
}

#[derive(Debug, Default)]
struct Slot {
    state: Option<State>,
    /// The state after each row of the last step's window that was not
    /// certain: what its commit picks from.
    pending: Vec<State>,
}

/// The device the host executor plays when it knows the pool's shape.
struct Oracle {
    spec: ContextSpec,
    /// The model has recurrent state, which lives in the slots and in
    /// checkpoint places. Without it the state before a row follows from the
    /// cells.
    recurrent: bool,
    cells_per_page: u32,
    cells: Vec<Cell>,
    slots: Vec<Slot>,
    places: HashMap<u32, State>,
    /// The bytes of a checkpoint place, and the places that hold memory in
    /// the books the executor was given.
    place_bytes: u64,
    held: Vec<u32>,
    vocab: u32,
}

fn note(problems: &Problems, what: String) {
    let mut log = problems.lock().expect("problems");
    if log.len() < MAX_PROBLEMS {
        log.push(what);
    }
}

impl Oracle {
    fn new(model: &crate::SyntheticModel, pages: u32, cells_per_page: u32, slots: usize) -> Self {
        let config = model.config();
        Self {
            spec: model.context(),
            recurrent: config.prefix_cache.is_some_and(|bytes| bytes > 0),
            cells_per_page,
            cells: vec![Cell::default(); pages as usize * cells_per_page as usize],
            slots: (0..slots).map(|_| Slot::default()).collect(),
            places: HashMap::new(),
            place_bytes: config.prefix_cache.unwrap_or(0),
            held: Vec::new(),
            vocab: config.vocab(),
        }
    }

    /// The places the conductor has called unused lose their memory, and
    /// what they held with it. Between steps, as a device part gives chunks
    /// back; a device may keep them longer, never shorter.
    fn give_back(&mut self, ram: Option<&PinnedRam>) {
        let Some(ram) = ram else { return };
        let (places, bytes) = (&mut self.places, self.place_bytes);
        self.held.retain(|place| {
            let live = ram.place_is_live(*place);
            if !live {
                places.remove(place);
                ram.release_checkpoints(bytes);
                ram.host_place_held(*place, false);
            }
            live
        });
    }

    /// A copy writes place `place`: it takes memory in the books when it
    /// holds none, and it must be a place the conductor has in use.
    fn take(&mut self, place: u32, what: &str, ram: Option<&PinnedRam>, problems: &Problems) {
        let Some(ram) = ram else { return };
        if !ram.place_is_live(place) {
            note(problems, format!("{what} place {place}, which the conductor called unused"));
        }
        if !self.held.contains(&place) {
            ram.host_place_held(place, true);
            if let Err(error) = ram.book_checkpoints(self.place_bytes) {
                note(problems, format!("{what} place {place}: {error}"));
            }
            self.held.push(place);
        }
    }

    /// Carries out a transfer of the cache's level on SSD: a page's cells or
    /// a place's state as bytes in the first part, and back.
    fn transfer(
        &mut self,
        transfer: &mut crate::Transfer,
        ram: Option<&PinnedRam>,
        problems: &Problems,
    ) -> bool {
        use crate::TransferKind::{PageIn, PageOut, StateIn, StateOut};
        match transfer.kind {
            StateOut if ram.is_some_and(|ram| !ram.place_is_live(transfer.object)) => {
                note(
                    problems,
                    format!(
                        "a checkpoint leaves place {} for the drive, and the conductor called \
                         that place unused",
                        transfer.object
                    ),
                );
                return false;
            }
            StateIn => {
                self.take(transfer.object, "a checkpoint from the drive lands in", ram, problems)
            }
            _ => {}
        }
        let per_page = self.cells_per_page as usize;
        let Some(part) = transfer.parts.first_mut() else { return false };
        let cells = transfer.object as usize * per_page..(transfer.object as usize + 1) * per_page;
        match transfer.kind {
            PageOut | PageIn if cells.end > self.cells.len() => false,
            PageOut => {
                for (cell, out) in self.cells[cells].iter().zip(part.mem.chunks_mut(CELL_BYTES)) {
                    out[0] = u8::from(cell.written) | u8::from(cell.poisoned) << 1;
                    out[4..8].copy_from_slice(&cell.token.to_le_bytes());
                    out[8..12].copy_from_slice(&cell.ctx.to_le_bytes());
                    out[16..24].copy_from_slice(&cell.prev.to_le_bytes());
                    out[24..32].copy_from_slice(&cell.chain.to_le_bytes());
                }
                (part.len, part.tag) = (per_page * CELL_BYTES, 1);
                true
            }
            PageIn if part.len != per_page * CELL_BYTES || part.tag != 1 => false,
            PageIn => {
                for (cell, bytes) in self.cells[cells].iter_mut().zip(part.mem.chunks(CELL_BYTES)) {
                    *cell = Cell {
                        written: bytes[0] & 1 != 0,
                        poisoned: bytes[0] & 2 != 0,
                        token: word(bytes, 4),
                        ctx: word(bytes, 8),
                        prev: long(bytes, 16),
                        chain: long(bytes, 24),
                    };
                }
                true
            }
            StateOut => {
                let Some(state) = self.places.get(&transfer.object) else { return false };
                let out = &mut part.mem;
                out[0..4].copy_from_slice(&state.pos.to_le_bytes());
                out[4..8].copy_from_slice(&state.ctx.to_le_bytes());
                out[8..16].copy_from_slice(&state.chain.to_le_bytes());
                out[16] = u8::from(state.poisoned);
                out[20..24].copy_from_slice(&(state.recent.len() as u32).to_le_bytes());
                for (i, token) in state.recent.iter().enumerate() {
                    let at = STATE_BYTES + 4 * i;
                    out[at..at + 4].copy_from_slice(&token.to_le_bytes());
                }
                (part.len, part.tag) = (STATE_BYTES + 4 * state.recent.len(), 2);
                true
            }
            StateIn if part.len < STATE_BYTES || part.tag != 2 => false,
            StateIn => {
                let bytes = &part.mem[..part.len];
                let recent = word(bytes, 20) as usize;
                if part.len != STATE_BYTES + 4 * recent {
                    return false;
                }
                let state = State {
                    pos: word(bytes, 0),
                    ctx: word(bytes, 4),
                    chain: long(bytes, 8),
                    poisoned: bytes[16] != 0,
                    recent: (0..recent).map(|i| word(bytes, STATE_BYTES + 4 * i)).collect(),
                };
                self.places.insert(transfer.object, state);
                true
            }
        }
    }

    fn cell(&self, table: &[u32], position: u32) -> Option<&Cell> {
        let page = *table.get((position / self.cells_per_page) as usize)?;
        self.cells.get(
            page as usize * self.cells_per_page as usize
                + (position % self.cells_per_page) as usize,
        )
    }

    fn cell_mut(&mut self, table: &[u32], position: u32) -> Option<&mut Cell> {
        let page = *table.get((position / self.cells_per_page) as usize)?;
        let at = page as usize * self.cells_per_page as usize
            + (position % self.cells_per_page) as usize;
        self.cells.get_mut(at)
    }

    /// The state of a sequence that has read the cells before `position`.
    fn state_from_cells(&self, table: &[u32], position: u32) -> State {
        let mut state = State::empty(&self.spec);
        let Some(last) = position.checked_sub(1).and_then(|q| self.cell(table, q)) else {
            return state;
        };
        let keep = self.spec.window.max(1);
        state.pos = position;
        state.chain = last.chain;
        state.ctx = last.ctx;
        state.recent = (position.saturating_sub(keep)..position)
            .map(|q| self.cell(table, q).map_or(0, |cell| cell.token))
            .collect();
        state
    }

    /// Checks everything a row at `position` reads against the state it starts
    /// from: the cells form one history, and the state is the one after it.
    /// Returns whether a cell it read was poisoned, or what is wrong.
    fn check_reads(&self, table: &[u32], position: u32, before: &State) -> Result<bool, String> {
        let mut prev = ROOT;
        let mut poisoned = false;
        for q in 0..position {
            let Some(cell) = self.cell(table, q).filter(|cell| cell.written) else {
                return Err(format!("cell {q} is read and was never written"));
            };
            if cell.prev != prev {
                return Err(format!(
                    "cell {q} was written after another history than the cell before it holds"
                ));
            }
            poisoned |= cell.poisoned;
            prev = cell.chain;
        }
        if self.recurrent && before.chain != prev {
            return Err(format!(
                "the state is not the one after the {position} cells the row reads"
            ));
        }
        Ok(poisoned)
    }

    fn step(
        &mut self,
        model: &crate::SyntheticModel,
        view: &RecordView<'_>,
        tables: &[Vec<u32>],
        picks: &mut Vec<Pick>,
        ram: Option<&PinnedRam>,
        problems: &Problems,
    ) {
        let complain = |what: String| note(problems, what);

        // The commit of the step before.
        for i in 0..view.commits() {
            let commit = view.commit(i);
            let Some(slot) = self.slots.get_mut(commit.slot as usize) else { continue };
            let pending = std::mem::take(&mut slot.pending);
            match commit.accepted.checked_sub(1).and_then(|k| pending.get(k as usize)) {
                Some(state) => slot.state = Some(state.clone()),
                None => {
                    complain(format!(
                        "slot {} commits rows the last step did not keep",
                        commit.slot
                    ));
                    slot.state.get_or_insert_with(|| State::empty(&self.spec)).poisoned = true;
                }
            }
        }
        // The saves, then the page copies, then the restores.
        for i in 0..view.state_copies() {
            let copy = view.state_copy(i);
            if copy.kind == StateCopyKind::Save {
                self.take(copy.place, "a save writes", ram, problems);
                let state = self.slots.get(copy.slot as usize).and_then(|slot| slot.state.clone());
                self.places.insert(
                    copy.place,
                    state.unwrap_or_else(|| {
                        complain(format!("slot {} is saved and holds nothing", copy.slot));
                        State { poisoned: true, ..State::empty(&self.spec) }
                    }),
                );
            }
        }
        let per_page = self.cells_per_page as usize;
        for i in 0..view.page_copies() {
            let copy = view.page_copy(i);
            let (from, to) = (copy.from as usize * per_page, copy.to as usize * per_page);
            if to + per_page > self.cells.len() || from + per_page > self.cells.len() {
                complain(format!("page copy {copy:?} is outside the pool"));
                continue;
            }
            for k in 0..(copy.cells as usize).min(per_page) {
                self.cells[to + k] = self.cells[from + k];
            }
        }
        for i in 0..view.state_copies() {
            let copy = view.state_copy(i);
            if copy.kind == StateCopyKind::Restore {
                let state = self.places.get(&copy.place).cloned().unwrap_or_else(|| {
                    complain(format!("place {} is restored and was never saved", copy.place));
                    State { poisoned: true, ..State::empty(&self.spec) }
                });
                match self.slots.get_mut(copy.slot as usize) {
                    Some(slot) => slot.state = Some(state),
                    None => complain(format!("slot {} does not exist", copy.slot)),
                }
            }
        }

        // A page the record names as freed is not one the step reads: a device
        // that keeps something per page gives it back when it is told.
        let freed: Vec<u32> = (0..view.freed()).map(|i| view.freed_page(i)).collect();
        if !freed.is_empty() {
            for s in 0..view.segments() {
                let seg = view.segment(s);
                let table: &[u32] = tables.get(seg.slot as usize).map_or(&[], Vec::as_slice);
                let reads = (seg.pos0 as usize + seg.n_rows as usize).div_ceil(per_page);
                for page in table.iter().take(reads).filter(|page| freed.contains(page)) {
                    complain(format!(
                        "page {page} is freed in the record of a step whose segment {s} reads it"
                    ));
                }
            }
        }

        // A step that only drafts runs no row and leaves no window.
        if view.drafts_only() {
            return;
        }

        // The rows.
        let mut results: Vec<(u32, f32)> = vec![(0, 0.0); view.rows() as usize];
        for s in 0..view.segments() {
            let seg = view.segment(s);
            let replay = view.replay(s);
            let slot = seg.slot as usize;
            let table: &[u32] = tables.get(slot).map_or(&[], Vec::as_slice);
            let first = view.row(seg.row0);
            let start = first.position;
            let mut state = if !self.recurrent {
                self.state_from_cells(table, start)
            } else if start == 0 {
                State::empty(&self.spec)
            } else {
                match self.slots.get(slot).and_then(|slot| slot.state.clone()) {
                    Some(state) => state,
                    None => {
                        complain(format!("slot {slot} starts at position {start} with no state"));
                        State { poisoned: true, ..State::empty(&self.spec) }
                    }
                }
            };
            if self.recurrent && state.pos != start {
                complain(format!(
                    "slot {slot} has read {} tokens and its row is at position {start}",
                    state.pos
                ));
                state.poisoned = true;
            }
            let mut after_each = Vec::new();
            for k in 0..seg.n_rows {
                let row = view.row(seg.row0 + k);
                if row.position != start + k {
                    complain(format!("row {k} of a segment at {start} is at {}", row.position));
                    state.poisoned = true;
                }
                let mut wrong = None;
                match self.check_reads(table, row.position, &state) {
                    Ok(true) => state.poisoned = true,
                    Ok(false) => {}
                    Err(what) => wrong = Some(what),
                }
                let after = state.absorbed(&self.spec, row.token);
                if self.spec.reduce(after.ctx, row.position) != row.context {
                    wrong.get_or_insert_with(|| {
                        "the context of the row is not the one of the sequence's tokens".into()
                    });
                }
                let own = Cell {
                    written: true,
                    poisoned: false,
                    token: row.token,
                    ctx: after.ctx,
                    prev: state.chain,
                    chain: after.chain,
                };
                if replay {
                    match self.cell(table, row.position) {
                        Some(cell)
                            if cell.written
                                && (cell.token, cell.prev, cell.chain)
                                    == (own.token, own.prev, own.chain) => {}
                        _ => {
                            wrong.get_or_insert_with(|| {
                                "a replayed row finds a cell other than the one it rebuilds".into()
                            });
                        }
                    }
                }
                let mut after = after;
                if let Some(what) = wrong {
                    complain(format!("slot {slot}, position {}: {what}", row.position));
                    after.poisoned = true;
                }
                if !replay {
                    match self.cell_mut(table, row.position) {
                        Some(cell) => *cell = Cell { poisoned: after.poisoned, ..own },
                        None => {
                            complain(format!(
                                "slot {slot}, position {}: the page table has no page to write",
                                row.position
                            ));
                            after.poisoned = true;
                        }
                    }
                }
                let (mut token, logit) = model.host_pick(row.token, row.context);
                if after.poisoned {
                    token = (token + 1) % self.vocab;
                }
                results[(seg.row0 + k) as usize] = (token, logit);
                state = after;
                if !seg.direct {
                    after_each.push(state.clone());
                }
            }
            if let Some(held) = self.slots.get_mut(slot) {
                if seg.direct {
                    if self.recurrent {
                        held.state = Some(state);
                    }
                } else {
                    held.pending = after_each;
                }
            }
        }
        for i in 0..view.runs() {
            let (first, count) = view.run(i);
            for row in first..first + count {
                let (token, logit) = results[row as usize];
                picks.push(Pick { token, logit, aux: 0.0 });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SyntheticConfig;
    use crate::model::Split;
    use crate::record::{Commit, PageCopy, RecordWriter, Row, StateCopy, TableUpdate};

    const PER_PAGE: u32 = 4;

    /// The rows of one sequence in a record: `rows` rows from position `from`
    /// of a text.
    struct Seg<'a> {
        slot: u32,
        text: &'a [u32],
        from: u32,
        rows: u32,
        direct: bool,
        replay: bool,
    }

    fn seg(slot: u32, text: &[u32], from: u32, rows: u32) -> Seg<'_> {
        Seg { slot, text, from, rows, direct: true, replay: false }
    }

    /// A device of the oracle, driven with records written by hand.
    struct Device {
        host: HostExecutor,
        model: Arc<SyntheticModel>,
        step: u64,
    }

    impl Device {
        fn new(prefix_cache: Option<u64>) -> Self {
            let mut config = SyntheticConfig::small(Split::Single, 1);
            config.prefix_cache = prefix_cache;
            config.context.buckets = 512;
            let model = Arc::new(crate::SyntheticModel::new(config).unwrap());
            let host = HostExecutor::new(model.clone(), 64, 8).with_pool(32, PER_PAGE);
            Self { host, model, step: 0 }
        }

        /// The contexts the conductor writes for `text`.
        fn contexts(&self, text: &[u32]) -> Vec<u32> {
            let spec = self.model.context();
            let mut state = spec.start();
            (0..text.len())
                .map(|i| {
                    state = spec.advance(state, &text[..=i]);
                    spec.reduce(state, i as u32)
                })
                .collect()
        }

        /// The pick a consistent row gets.
        fn right(&self, text: &[u32], position: u32) -> u32 {
            let context = self.contexts(text)[position as usize];
            self.model.host_pick(text[position as usize], context).0
        }

        /// Runs one record whose rows are all wanted, and returns the picks.
        fn run(
            &mut self,
            segs: &[Seg<'_>],
            updates: &[(u32, u32, &[u32])],
            commits: &[Commit],
            states: &[StateCopy],
            pages: &[PageCopy],
        ) -> Vec<u32> {
            self.step += 1;
            let rows: u32 = segs.iter().map(|s| s.rows).sum();
            let all_contexts: Vec<Vec<u32>> = segs.iter().map(|s| self.contexts(s.text)).collect();
            let mut writer =
                RecordWriter::new(self.host.record(), self.step, segs.len() as u32, rows).unwrap();
            for (s, contexts) in segs.iter().zip(&all_contexts) {
                writer.segment(s.slot, s.from, s.rows, s.direct).unwrap();
                if s.replay {
                    writer.replay().unwrap();
                }
                for k in 0..s.rows {
                    let position = s.from + k;
                    writer
                        .row(Row {
                            slot: s.slot,
                            position,
                            token: s.text[position as usize],
                            context: contexts[position as usize],
                        })
                        .unwrap();
                }
            }
            writer.runs(&[(0, rows)]).unwrap();
            writer.commits(commits).unwrap();
            let entries: Vec<u32> = updates.iter().flat_map(|u| u.2.iter().copied()).collect();
            writer
                .updates(
                    updates.iter().map(|u| TableUpdate {
                        slot: u.0,
                        first: u.1,
                        count: u.2.len() as u32,
                    }),
                    entries.into_iter(),
                )
                .unwrap();
            writer.copies(states, pages).unwrap();
            let used = writer.finish().unwrap();
            self.host
                .dispatch(&StepMeta { step: self.step, record_bytes: used, after_idle: false })
                .unwrap();
            let mut out = StepOutput::default();
            self.host.collect(Duration::ZERO, &mut out).unwrap();
            out.picks.iter().map(|p| p.token).collect()
        }

        fn problems(&self) -> usize {
            self.host.problems.lock().unwrap().len()
        }
    }

    fn text(len: u32) -> Vec<u32> {
        (0..len).map(|i| (i * 7 + 3) % 251).collect()
    }

    fn save(slot: u32, place: u32) -> StateCopy {
        StateCopy { kind: StateCopyKind::Save, slot, place }
    }

    fn restore(slot: u32, place: u32) -> StateCopy {
        StateCopy { kind: StateCopyKind::Restore, slot, place }
    }

    #[test]
    fn a_consistent_history_picks_what_the_model_picks() {
        let mut device = Device::new(Some(64));
        let t = text(24);
        // A prompt in two steps, then a window of three rows whose first two
        // are accepted, then the row after them.
        device.run(&[seg(0, &t, 0, 10)], &[(0, 0, &[0, 1, 2])], &[], &[], &[]);
        let picks = device.run(&[seg(0, &t, 10, 7)], &[(0, 3, &[3, 4])], &[], &[], &[]);
        assert_eq!(picks[6], device.right(&t, 16));
        let window = Seg { direct: false, ..seg(0, &t, 17, 3) };
        let picks = device.run(&[window], &[(0, 5, &[5])], &[], &[], &[]);
        assert_eq!(picks, (17..20).map(|p| device.right(&t, p)).collect::<Vec<_>>());
        let commit = Commit { row0: 0, rows: 3, slot: 0, accepted: 2 };
        let picks = device.run(&[seg(0, &t, 19, 1)], &[], &[commit], &[], &[]);
        assert_eq!(picks, [device.right(&t, 19)]);
        assert_eq!(device.problems(), 0, "{:?}", device.host.problems);
    }

    #[test]
    fn a_restore_of_the_right_place_and_a_replay_of_the_rows_after_it_continue_a_history() {
        let mut device = Device::new(Some(64));
        let t = text(24);
        device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        // The state at 8 is saved before the rows that go on from there.
        device.run(&[seg(0, &t, 8, 8)], &[(0, 2, &[2, 3])], &[], &[save(0, 5)], &[]);
        // Another sequence takes the pages of the first 16 cells, restores the
        // state at 8 and replays 8..16 reading its cells, then goes on.
        let share = [(1, 0, &[0u32, 1, 2, 3][..])];
        let replay = Seg { replay: true, ..seg(1, &t, 8, 8) };
        let picks = device.run(&[replay], &share, &[], &[restore(1, 5)], &[]);
        assert_eq!(picks.len(), 8);
        let picks = device.run(&[seg(1, &t, 16, 4)], &[(1, 4, &[4])], &[], &[], &[]);
        assert_eq!(picks, (16..20).map(|p| device.right(&t, p)).collect::<Vec<_>>());
        assert_eq!(device.problems(), 0, "{:?}", device.host.problems);
    }

    #[test]
    fn a_restore_of_another_place_a_missing_replay_and_a_wrong_replay_change_what_is_picked() {
        let t = text(24);
        let other = text(30).into_iter().map(|x| x + 1).collect::<Vec<_>>();
        // Places 5 (the state at 8 of `t`) and 6 (the state at 12 of `other`).
        let prepare = |device: &mut Device| {
            device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
            device.run(&[seg(0, &t, 8, 8)], &[(0, 2, &[2, 3])], &[], &[save(0, 5)], &[]);
            device.run(&[seg(2, &other, 0, 12)], &[(2, 0, &[4, 5, 6])], &[], &[], &[]);
            device.run(&[seg(2, &other, 12, 1)], &[(2, 3, &[7])], &[], &[save(2, 6)], &[]);
        };
        let share = [(1, 0, &[0u32, 1, 2, 3][..])];
        let replay = |from, rows| Seg { replay: true, ..seg(1, &t, from, rows) };

        // The right thing, for comparison.
        let mut device = Device::new(Some(64));
        prepare(&mut device);
        device.run(&[replay(8, 8)], &share, &[], &[restore(1, 5)], &[]);
        let right = device.run(&[seg(1, &t, 16, 1)], &[(1, 4, &[8])], &[], &[], &[]);
        assert_eq!((device.problems(), right[0]), (0, device.right(&t, 16)));

        // Restoring the state at 12 of another text: it is at another position.
        let mut device = Device::new(Some(64));
        prepare(&mut device);
        device.run(&[replay(8, 8)], &share, &[], &[restore(1, 6)], &[]);
        let wrong = device.run(&[seg(1, &t, 16, 1)], &[(1, 4, &[8])], &[], &[], &[]);
        assert!(device.problems() > 0);
        assert_ne!(wrong[0], device.right(&t, 16));

        // A replay that stops two rows short.
        let mut device = Device::new(Some(64));
        prepare(&mut device);
        device.run(&[replay(8, 6)], &share, &[], &[restore(1, 5)], &[]);
        let short = device.run(&[seg(1, &t, 16, 1)], &[(1, 4, &[8])], &[], &[], &[]);
        assert!(device.problems() > 0);
        assert_ne!(short[0], device.right(&t, 16));

        // A replay of other tokens than the cells hold.
        let mut device = Device::new(Some(64));
        prepare(&mut device);
        let mut changed = t.clone();
        changed[10] += 1;
        let wrong_rows = Seg { replay: true, ..seg(1, &changed, 8, 8) };
        device.run(&[wrong_rows], &share, &[], &[restore(1, 5)], &[]);
        let after = device.run(&[seg(1, &t, 16, 1)], &[(1, 4, &[8])], &[], &[], &[]);
        assert!(device.problems() > 0);
        assert_ne!(after[0], device.right(&t, 16));

        // No restore at all: the slot holds whatever it held.
        let mut device = Device::new(Some(64));
        prepare(&mut device);
        device.run(&[replay(8, 8)], &share, &[], &[], &[]);
        assert!(device.problems() > 0);
    }

    #[test]
    fn a_write_into_a_page_that_others_read_shows_when_they_read_it_again() {
        let mut device = Device::new(Some(64));
        let t = text(24);
        let u: Vec<u32> = t.iter().map(|x| x + 5).collect();
        device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        // A second sequence whose table holds the first's page 1 as well writes
        // its own tokens at 4..8, as a conductor that skipped the copy would.
        device.run(&[seg(1, &u, 0, 4)], &[(1, 0, &[2])], &[], &[], &[]);
        device.run(&[seg(1, &u, 4, 4)], &[(1, 1, &[1])], &[], &[], &[]);
        assert_eq!(device.problems(), 0, "the second sequence read its own cells");
        // The first goes on and reads what the second left in the page.
        let picks = device.run(&[seg(0, &t, 8, 1)], &[(0, 2, &[3])], &[], &[], &[]);
        assert!(device.problems() > 0);
        assert_ne!(picks[0], device.right(&t, 8));
    }

    #[test]
    fn a_copied_page_serves_the_sequence_that_writes_into_it_and_leaves_the_first_alone() {
        let mut device = Device::new(Some(64));
        let t = text(24);
        // 6 cells: page 0 full, page 1 holds 2. A second sequence continues
        // in a copy of page 1 (page 5), and the first goes on in its own.
        device.run(&[seg(0, &t, 0, 6)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        device.run(&[seg(0, &t, 6, 1)], &[], &[], &[save(0, 2)], &[]);
        let copy = PageCopy { from: 1, to: 5, cells: 2 };
        let share = [(1, 0, &[0u32, 5][..])];
        let rows = Seg { replay: true, ..seg(1, &t, 0, 6) };
        device.run(&[rows], &share, &[], &[], &[copy]);
        let mut more = t.clone();
        more[6] = 200;
        let picks = device.run(&[seg(1, &more, 6, 2)], &[(1, 1, &[5])], &[], &[], &[]);
        assert_eq!(picks.len(), 2);
        // The first sequence's page 1 is as it was.
        let first = device.run(&[seg(0, &t, 7, 1)], &[], &[], &[], &[]);
        assert_eq!(first, [device.right(&t, 7)]);
        assert_eq!(device.problems(), 0, "{:?}", device.host.problems);

        // Without the copy the second sequence's page is empty, and the rows
        // that rebuild the state do not find what they should.
        let mut device = Device::new(Some(64));
        device.run(&[seg(0, &t, 0, 6)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        let rows = Seg { replay: true, ..seg(1, &t, 0, 6) };
        device.run(&[rows], &share, &[], &[], &[]);
        assert!(device.problems() > 0);
    }

    #[test]
    fn a_save_that_runs_after_its_slot_was_taken_keeps_the_wrong_state() {
        let t = text(24);
        let other = text(24).into_iter().map(|x| x + 9).collect::<Vec<_>>();
        let mut device = Device::new(Some(64));
        device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        // The slot is taken by a new sequence, whose rows run in this record
        // and in the one after it, which carries the save.
        device.run(&[seg(0, &other, 0, 3)], &[(0, 0, &[2])], &[], &[], &[]);
        device.run(&[seg(1, &t, 0, 1)], &[(1, 0, &[3])], &[], &[save(0, 5)], &[]);
        // The state in place 5 is the new sequence's, and restoring it for the
        // old one's continuation finds it at another position.
        device.run(&[seg(2, &t, 8, 1)], &[(2, 0, &[0, 1])], &[], &[restore(2, 5)], &[]);
        assert!(device.problems() > 0);
    }

    #[test]
    fn a_record_runs_the_commit_then_the_saves_then_the_restores_then_the_rows() {
        let mut device = Device::new(Some(64));
        let t = text(24);
        device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        // A window of 3 rows from 8, whose rows are not committed yet.
        let window = Seg { direct: false, ..seg(0, &t, 8, 3) };
        device.run(&[window], &[(0, 2, &[2])], &[], &[], &[]);
        // The next record commits 2 of them (the state is at 10), saves that
        // state, restores it into another slot in the same record and runs its
        // next row there. Any other order finds the wrong state.
        let commit = Commit { row0: 0, rows: 3, slot: 0, accepted: 2 };
        let share = [(1, 0, &[0u32, 1, 2][..])];
        let picks =
            device.run(&[seg(1, &t, 10, 1)], &share, &[commit], &[save(0, 7), restore(1, 7)], &[]);
        assert_eq!(picks, [device.right(&t, 10)]);
        assert_eq!(device.problems(), 0, "{:?}", device.host.problems);
    }

    #[test]
    fn a_model_without_recurrent_state_takes_its_history_from_the_cells() {
        let mut device = Device::new(Some(0));
        let t = text(24);
        device.run(&[seg(0, &t, 0, 8)], &[(0, 0, &[0, 1])], &[], &[], &[]);
        // Another sequence shares the 8 cells with no state to restore and
        // goes on at once.
        let picks = device.run(&[seg(1, &t, 8, 2)], &[(1, 0, &[0, 1, 2])], &[], &[], &[]);
        assert_eq!(picks, (8..10).map(|p| device.right(&t, p)).collect::<Vec<_>>());
        assert_eq!(device.problems(), 0, "{:?}", device.host.problems);
        // Reading a cell that was never written is wrong.
        device.run(&[seg(3, &t, 16, 1)], &[(3, 0, &[0, 1, 4, 5])], &[], &[], &[]);
        assert!(device.problems() > 0);
    }
}
