//! The experts a wide step misses in a layer, and how they are brought.
//!
//! The many-row kernels read an expert from the address the host gives them and skip an
//! expert without one, so a layer of a wide step is a short conversation between the host
//! and the device (`Part::experts_round`):
//!
//! 1. The layer is enqueued. The host gives the first round the experts that are in a slot
//!    now; the device computes them while the host waits for the layer's route record.
//! 2. The record says which experts the rows picked. Those without a slot are the layer's
//!    misses. The store brings them into transit slots at once (`ExpertStore::transit_want`:
//!    from their RAM slot, or from the file through staging), next to the first round: no
//!    slot is touched, so nothing has to wait for the kernels that read slots. Whenever a
//!    round has ended and half of the transit slots hold experts that have arrived (or all
//!    that are left), those are the next round's table; its output is added to the first
//!    one's, and its slots are free again when it has ended.
//! 3. When the record of a layer arrives, the reads of the layers ahead are started on a
//!    guess (`Feed::guess_ahead`): what the wide step before missed there.
//!
//! A store without transit brings the misses into the layer's slots instead, as many as
//! the layer has at a time, once the first round has ended (nothing reads a slot any more).
//!
//! In a step of up to `Options::cpu_rows` rows the CPU takes part (`hybrid/wide.rs` of the
//! store): of the misses in the RAM tier, those that the cost model gives it (the fewest
//! rows first) are computed from the rows' mixed input, which the part copies to the host
//! when it enqueues the layer ([`Part::cpu_act`]), and their sums are added to the layer's
//! output behind the rounds ([`Part::add_cpu_sums`]). The layer is done when the rounds and
//! the CPU are.
//!
//! [`Misses::advance`] never waits. The caller serves the store and calls it again.

use std::time::{Duration, Instant};

use std::sync::Arc;

use ares_runtime::expert_store::{WideCpu, WidePick};

use ares_cuda::{Device, Event, PinnedBuffer};

use super::Error;
use super::board::Board;
use super::feed::Feed;
use super::part::{Part, StepEnv};

/// What wide steps found and waited for, added up over layers.
#[derive(Clone, Copy, Debug, Default)]
pub struct WideStats {
    /// Layers of wide steps.
    pub layers: u64,
    /// Experts the rows of a layer picked (of those the part computes), over the layers.
    pub picked: u64,
    /// Of them, the experts that had no slot when the layer was enqueued.
    pub missing: u64,
    /// Rounds after the first one.
    pub extra_rounds: u64,
    /// Time from enqueuing a layer to its route record arriving, after the host had
    /// nothing else to do.
    pub record_wait: Duration,
    /// Time, with misses, from the record to the end of the first round on the device.
    pub compute_wait: Duration,
    /// Time from asking the store for a batch of misses to its arrival.
    pub bring_wait: Duration,
    /// Time from enqueuing a later round to its end.
    pub round_wait: Duration,
    /// Time of the layers from their enqueuing to the end of their last round.
    pub layer_time: Duration,
    /// Of the missing experts, those the CPU computed, and their (expert, row) pairs.
    pub cpu_experts: u64,
    pub cpu_pairs: u64,
    /// Time from the end of the last round to the CPU's sums being added, where the CPU
    /// ended later.
    pub cpu_wait: Duration,
}

impl WideStats {
    /// Adds `other`.
    pub(super) fn add(&mut self, other: &WideStats) {
        self.layers += other.layers;
        self.picked += other.picked;
        self.missing += other.missing;
        self.extra_rounds += other.extra_rounds;
        self.record_wait += other.record_wait;
        self.compute_wait += other.compute_wait;
        self.bring_wait += other.bring_wait;
        self.round_wait += other.round_wait;
        self.layer_time += other.layer_time;
        self.cpu_experts += other.cpu_experts;
        self.cpu_pairs += other.cpu_pairs;
        self.cpu_wait += other.cpu_wait;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The record of the layer is on its way.
    Record,
    /// The first round runs; the misses wait for it to end.
    Compute,
    /// A batch of misses is being brought.
    Arrive,
    /// A later round runs.
    Round,
    /// The misses come through transit: a round runs, or the next one waits for experts.
    Transit,
    /// The record has arrived: the CPU's share of the misses waits for the rows' mixed input,
    /// the rest goes to transit.
    Split,
    /// The rounds have ended; the CPU's sums are still to come.
    Cpu,
    /// Every expert the layer's rows picked has been computed.
    Done,
}

/// The picks of a layer of a prompt read layer by layer, as the host reads them for the
/// rounds of its experts: a header of 16 bytes, then the ids of every row's picks in the order
/// of the rows, then their weights (the form the router's board has them in), which a worker
/// that does not route takes for its own rounds.
pub(super) struct SpanPicks {
    host: PinnedBuffer,
    len: usize,
    copied: Event,
    /// The layer whose copies `copied` was last recorded behind, plus one (0: none yet): a
    /// worker's thread reads the picks the mixer's thread copies, and must not take the
    /// event of the layer before for its layer's.
    layer: std::sync::atomic::AtomicUsize,
}

impl SpanPicks {
    /// Room for the picks of `rows` rows of `top_k` picks each.
    pub(super) fn new(device: &Device, rows: u32, top_k: u32) -> Result<Self, Error> {
        let len = 16 + 8 * rows as usize * top_k as usize;
        let host = PinnedBuffer::new(device, len)?;
        let copied = Event::new(device)?;
        Ok(Self { host, len, copied, layer: std::sync::atomic::AtomicUsize::new(0) })
    }

    /// Whether `rows` rows of `top_k` picks fit.
    pub(super) fn holds(&self, rows: u32, top_k: u32) -> bool {
        16 + 8 * rows as usize * top_k as usize == self.len
    }

    /// Where the copies write, and the event recorded behind them.
    pub(super) fn target(&self) -> (*mut u8, &Event) {
        (self.host.as_ptr().cast_mut(), &self.copied)
    }

    /// The copies of layer `layer`'s picks are enqueued behind the event (after
    /// [`SpanPicks::target`]'s copies were).
    pub(super) fn mark(&self, layer: usize) {
        self.layer.store(layer + 1, std::sync::atomic::Ordering::Release);
    }

    /// A new pass begins: no layer's copies are enqueued yet.
    pub(super) fn reset(&self) {
        self.layer.store(0, std::sync::atomic::Ordering::Release);
    }

    fn arrived(&self, layer: usize) -> Result<bool, Error> {
        Ok(self.layer.load(std::sync::atomic::Ordering::Acquire) == layer + 1
            && self.copied.is_done()?)
    }

    fn record(&self) -> &[u8] {
        // SAFETY: the copies behind `copied` have run, and nothing writes the buffer before
        // the next layer's copies are enqueued, after this layer is done.
        unsafe { std::slice::from_raw_parts(self.host.as_ptr(), self.len) }
    }
}

/// Whether a layer is finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Progress {
    Waiting,
    Done,
}

/// One layer of a wide step on one part, between its first round and the end of its last.
pub(super) struct Misses {
    /// The part and the layer within the part.
    part: usize,
    layer: usize,
    /// The store's layer.
    at: usize,
    rows: u32,
    /// The board that has the layer's picks: the part's own, or the mixer's for a worker,
    /// which does not route. `relay` says the first round waits for them. In a pass, the
    /// picks of all its rows instead (`span`).
    source: Arc<Board>,
    relay: bool,
    span: Option<Arc<SpanPicks>>,
    stage: Stage,
    since: Instant,
    /// The first round's table: the address of every expert that had a slot.
    resident: Vec<u64>,
    /// The experts that did not, most picked first, and the batch being brought.
    missing: Vec<u32>,
    batch: std::ops::Range<usize>,
    /// Through transit: the experts that have arrived and wait for a round (with their
    /// transit slot), the ones the running round reads, whether a round runs, and how many
    /// of the misses have been given to a round.
    arrived: Vec<(u32, u32)>,
    in_round: Vec<u32>,
    running: bool,
    computed: usize,
    begun: Instant,
    stats: WideStats,
    /// The CPU's part: its jobs; their sums with the first row they are for, until the part
    /// takes them; and when the rounds ended.
    cpu: Option<WideCpu>,
    cpu_sums: Option<(u32, Vec<f32>)>,
    rounds_ended: Option<Instant>,
}

impl Misses {
    /// Starts layer `layer` of `part` (`index` among the parts) with the experts that are in
    /// a slot of the part's device now and that `owned` says the part computes: enqueues the
    /// first round, or, for a part that does not route (`source`: the board of the part that
    /// does), leaves it to [`Misses::advance`] once the picks have reached the host.
    pub(super) fn begin(
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &Feed,
        index: usize,
        layer: usize,
        source: Option<Arc<Board>>,
        owned: &dyn Fn(u32) -> bool,
    ) -> Result<Self, Error> {
        Self::begin_with(part, env, feed, index, layer, source, None, owned)
    }

    /// [`Misses::begin`] for a layer of a pass: the picks of all its rows reach the host in
    /// `span` (`env` is the step of all the rows), from this part's router or, with `relay`,
    /// from the part that routes, whose picks the first round waits for.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_pass(
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &Feed,
        index: usize,
        layer: usize,
        span: Arc<SpanPicks>,
        relay: bool,
        owned: &dyn Fn(u32) -> bool,
    ) -> Result<Self, Error> {
        let board = relay.then(|| part.board());
        Self::begin_with(part, env, feed, index, layer, board, Some(span), owned)
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_with(
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &Feed,
        index: usize,
        layer: usize,
        source: Option<Arc<Board>>,
        span: Option<Arc<SpanPicks>>,
        owned: &dyn Fn(u32) -> bool,
    ) -> Result<Self, Error> {
        let at = feed.layout().store_layer(index, part.layer_range().start + layer);
        let n_experts = feed.layout().shapes[at].n_experts;
        let resident: Vec<u64> = (0..n_experts)
            .map(|expert| match feed.store.vram_slot(at, expert).filter(|_| owned(expert)) {
                Some(slot) => part.slot_address(layer, slot),
                None => 0,
            })
            .collect();
        let relay = source.is_some();
        if !relay {
            part.experts_round(env, layer, &resident, None)?;
        }
        Ok(Self {
            source: source.unwrap_or_else(|| part.board()),
            relay,
            span,
            part: index,
            layer,
            at,
            rows: env.record.dims().n_rows,
            stage: Stage::Record,
            since: Instant::now(),
            resident,
            missing: Vec::new(),
            batch: 0..0,
            arrived: Vec::new(),
            in_round: Vec::new(),
            running: false,
            computed: 0,
            begun: Instant::now(),
            stats: WideStats::default(),
            cpu: None,
            cpu_sums: None,
            rounds_ended: None,
        })
    }

    /// Takes the next step of the layer that does not wait: reads the record when it has
    /// arrived, asks for a batch of misses, enqueues a round once its batch is in slots.
    /// `routed` says the part that routes has enqueued this layer (a record can only be
    /// read after that).
    pub(super) fn advance(
        &mut self,
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &mut Feed,
        routed: bool,
        owned: &dyn Fn(u32) -> bool,
    ) -> Result<Progress, Error> {
        loop {
            if !matches!(self.stage, Stage::Record | Stage::Split | Stage::Done) {
                self.serve_cpu(part, env, feed)?;
            }
            match self.stage {
                Stage::Done => return Ok(Progress::Done),
                Stage::Cpu => return self.finish(part, env, feed),
                Stage::Record => {
                    let arrived = match &self.span {
                        Some(span) => span.arrived(self.layer)?,
                        None => routed && self.source.arrived(self.layer)?,
                    };
                    if !arrived {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.record_wait += self.since.elapsed();
                    let source = Arc::clone(&self.source);
                    let span = self.span.clone();
                    let record = match &span {
                        Some(span) => span.record(),
                        None => source.record(self.layer),
                    };
                    if std::mem::take(&mut self.relay) {
                        part.experts_round(env, self.layer, &self.resident, Some(record))?;
                    }
                    let counts = part.picked_counts(self.layer, self.rows, record);
                    // The heat counts an expert once a step, as a decode step
                    // does whatever rows pick it: counted by its rows, a prompt
                    // outweighed every token generated after it, and the tiers
                    // filled from such heat held what long prompts used.
                    let once: Vec<u32> = counts.iter().map(|c| u32::from(*c > 0)).collect();
                    feed.store.record_counts(self.at, &once, 1.0);
                    let picked = |e: &u32| counts[*e as usize] > 0 && owned(*e);
                    let all = 0..counts.len() as u32;
                    self.stats.layers = 1;
                    self.stats.picked = all.clone().filter(picked).count() as u64;
                    self.missing =
                        all.filter(picked).filter(|e| self.resident[*e as usize] == 0).collect();
                    self.missing.sort_by_key(|e| std::cmp::Reverse(counts[*e as usize]));
                    self.stats.missing = self.missing.len() as u64;
                    self.stage = Stage::Split;
                }
                Stage::Split => {
                    // The CPU's share of the misses: decided and held in their RAM slots at
                    // once, when the rows' mixed input is on the host (with the record, but
                    // for a part that does not route).
                    if part.cpu_takes_part() && !self.missing.is_empty() {
                        let Some(act) = part.cpu_act(self.rows)? else {
                            return Ok(Progress::Waiting);
                        };
                        let source = Arc::clone(&self.source);
                        let span = self.span.clone();
                        let record = match &span {
                            Some(span) => span.record(),
                            None => source.record(self.layer),
                        };
                        let counts = part.picked_counts(self.layer, self.rows, record);
                        let picks = self.split_for_cpu(feed, record, &counts);
                        if !picks.is_empty() {
                            let cpu =
                                feed.store.wide_cpu(self.at, self.rows as usize, act, &picks)?;
                            self.cpu = Some(cpu);
                        }
                    }
                    let transit = feed.has_transit(self.at);
                    if transit {
                        feed.named(self.at, &self.missing)?;
                    }
                    if self.missing.is_empty() {
                        self.stage = Stage::Cpu;
                        continue;
                    }
                    self.stage = if transit { Stage::Transit } else { Stage::Compute };
                    self.running = true;
                    self.since = Instant::now();
                }
                Stage::Transit => {
                    if !part.rounds_done()? {
                        return Ok(Progress::Waiting);
                    }
                    if std::mem::take(&mut self.running) {
                        // The round that ran has ended: the first one, or one from transit,
                        // whose slots are free for the next experts.
                        let waited = self.since.elapsed();
                        if self.computed == 0 {
                            self.stats.compute_wait += waited;
                        } else {
                            self.stats.round_wait += waited;
                        }
                        self.since = Instant::now();
                        feed.store.transit_release(self.at, &self.in_round)?;
                        self.in_round.clear();
                    }
                    if self.computed == self.missing.len() {
                        self.stage = Stage::Cpu;
                        continue;
                    }
                    feed.store.transit_take(self.at, &mut self.arrived);
                    let left = self.missing.len() - self.computed;
                    let batch = feed.transit_batch(self.at).min(left);
                    // A round waits for a batch, as long as experts of the layer are on
                    // their way. When none is (the others wait for the slots this round
                    // gives back, or for staging that another layer holds), it takes what
                    // has arrived.
                    if self.arrived.is_empty()
                        || (self.arrived.len() < batch && feed.store.transit_moving(self.at) > 0)
                    {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.bring_wait += self.since.elapsed();
                    let mut table = vec![0u64; self.resident.len()];
                    for (expert, slot) in self.arrived.drain(..) {
                        table[expert as usize] = feed.transit_address(self.at, slot);
                        self.in_round.push(expert);
                    }
                    self.computed += self.in_round.len();
                    part.experts_round(env, self.layer, &table, None)?;
                    self.stats.extra_rounds += 1;
                    self.running = true;
                    self.since = Instant::now();
                }
                Stage::Compute => {
                    if !part.rounds_done()? {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.compute_wait += self.since.elapsed();
                    self.bring(feed)?;
                }
                Stage::Arrive => {
                    let batch = &self.missing[self.batch.clone()];
                    if !batch.iter().all(|e| feed.store.in_vram(self.at, *e)) {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.bring_wait += self.since.elapsed();
                    let mut table = vec![0u64; self.resident.len()];
                    for expert in batch {
                        if let Some(slot) = feed.store.vram_slot(self.at, *expert) {
                            table[*expert as usize] = part.slot_address(self.layer, slot);
                        }
                    }
                    part.experts_round(env, self.layer, &table, None)?;
                    self.stats.extra_rounds += 1;
                    self.stage = Stage::Round;
                    self.since = Instant::now();
                }
                Stage::Round => {
                    if !part.rounds_done()? {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.round_wait += self.since.elapsed();
                    if self.batch.end == self.missing.len() {
                        self.stage = Stage::Cpu;
                        continue;
                    }
                    self.bring(feed)?;
                }
            }
        }
    }

    /// Asks the store for the next batch of misses: as many as the layer has slots, which no
    /// kernel reads any more.
    fn bring(&mut self, feed: &mut Feed) -> Result<(), Error> {
        let room = feed.store.slots(self.at).0.max(1);
        let first = self.batch.end;
        // What the rounds so far computed has ended on the device: it may give way.
        let done: Vec<u32> = (0..self.resident.len() as u32)
            .filter(|e| self.resident[*e as usize] != 0)
            .chain(self.missing[..first].iter().copied())
            .collect();
        feed.store.finished(self.at, &done);
        self.batch = first..(first + room).min(self.missing.len());
        feed.store.routed(self.at, &self.missing[self.batch.clone()])?;
        self.stage = Stage::Arrive;
        self.since = Instant::now();
        Ok(())
    }

    /// Ends the layer once the CPU's sums have been added too.
    fn finish(
        &mut self,
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &mut Feed,
    ) -> Result<Progress, Error> {
        let ended = *self.rounds_ended.get_or_insert_with(Instant::now);
        if !self.serve_cpu(part, env, feed)? {
            return Ok(Progress::Waiting);
        }
        if self.stats.cpu_experts > 0 {
            self.stats.cpu_wait = ended.elapsed();
        }
        self.stats.layer_time = self.begun.elapsed();
        feed.layer_done(self.part, self.layer + 1, &self.stats)?;
        self.stage = Stage::Done;
        Ok(Progress::Done)
    }

    /// Takes the misses the cost model gives the CPU out of `self.missing` and returns their
    /// picks, from the layer's `record` (the ids of the rows' picks after a header of 16 bytes,
    /// then their weights); `counts` are the rows that picked each expert.
    fn split_for_cpu(&mut self, feed: &Feed, record: &[u8], counts: &[u32]) -> Vec<WidePick> {
        let misses: Vec<(u32, u32)> =
            self.missing.iter().map(|e| (*e, counts[*e as usize])).collect();
        let on_cpu = feed.store.wide_split(self.at, &misses);
        let mut picks: Vec<WidePick> = Vec::new();
        if !on_cpu.iter().any(|cpu| *cpu) {
            return picks;
        }
        let mut slot = vec![usize::MAX; counts.len()];
        for (i, (expert, _)) in misses.iter().enumerate() {
            if on_cpu[i] {
                slot[*expert as usize] = picks.len();
                picks.push(WidePick { expert: *expert, rows: Vec::new() });
            }
        }
        let top_k = feed.layout().shapes[self.at].top_k as usize;
        let total = top_k * self.rows as usize;
        let ids = &record[16..16 + 4 * total];
        let weights = &record[16 + 4 * total..16 + 8 * total];
        for (p, (id, weight)) in
            ids.as_chunks::<4>().0.iter().zip(weights.as_chunks::<4>().0).enumerate()
        {
            let expert = u32::from_le_bytes(*id) as usize;
            if let Some(at) = slot.get(expert).filter(|at| **at != usize::MAX) {
                let row = (p / top_k) as u32;
                picks[*at].rows.push((row, f32::from_le_bytes(*weight)));
            }
        }
        let mut kept = misses.iter().zip(&on_cpu).filter(|(_, cpu)| !**cpu);
        self.missing = kept.by_ref().map(|((expert, _), _)| *expert).collect();
        self.stats.cpu_experts = picks.len() as u64;
        self.stats.cpu_pairs = picks.iter().map(|p| p.rows.len() as u64).sum();
        picks
    }

    /// Moves the CPU's part on without waiting: takes the sums of its jobs when they have
    /// run and hands them to the part. Returns whether the CPU's part is done (or there was
    /// none).
    fn serve_cpu(
        &mut self,
        part: &mut Part,
        env: &StepEnv<'_>,
        feed: &mut Feed,
    ) -> Result<bool, Error> {
        if self.cpu.as_ref().is_some_and(WideCpu::is_done) {
            let cpu = self.cpu.take().expect("the CPU's part");
            let rows = cpu.rows();
            let n = feed.layout().shapes[self.at].n_embd as usize;
            let mut sums = vec![0.0f32; rows.len() * n];
            feed.store.wide_cpu_end(cpu, &mut sums)?;
            self.cpu_sums = Some((rows.start, sums));
        }
        if let Some((first, sums)) = &self.cpu_sums {
            if !part.add_cpu_sums(env, *first, sums)? {
                return Ok(false);
            }
            self.cpu_sums = None;
        }
        Ok(self.cpu.is_none() && self.cpu_sums.is_none())
    }
}
