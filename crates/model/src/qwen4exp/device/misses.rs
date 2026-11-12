//! The experts a wide step misses in a layer, and how they are brought.
//!
//! The many-row kernels read an expert from the address the host gives them and skip an
//! expert without one, so a layer of a wide step is a short conversation between the host
//! and the device (`Part::experts_round`):
//!
//! 1. The layer is enqueued. The host gives the first round the experts that are in a slot
//!    now; the device computes them while the host waits for the layer's route record.
//! 2. The record says which experts the rows picked. Those without a slot are the layer's
//!    misses. When the first round has ended on the device (nothing reads a slot any more),
//!    the store brings the misses, as many as the layer has slots at a time, and each batch
//!    that has arrived is the next round's table; its output is added to the first one's.
//!
//! [`Misses::advance`] never waits. The caller serves the store and calls it again.

use std::time::{Duration, Instant};

use std::sync::Arc;

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
    /// Every expert the layer's rows picked has been computed.
    Done,
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
    /// which does not route. `relay` says the first round waits for them.
    source: Arc<Board>,
    relay: bool,
    stage: Stage,
    since: Instant,
    /// The first round's table: the address of every expert that had a slot.
    resident: Vec<u64>,
    /// The experts that did not, most picked first, and the batch being brought.
    missing: Vec<u32>,
    batch: std::ops::Range<usize>,
    stats: WideStats,
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
            part: index,
            layer,
            at,
            rows: env.record.dims().n_rows,
            stage: Stage::Record,
            since: Instant::now(),
            resident,
            missing: Vec::new(),
            batch: 0..0,
            stats: WideStats::default(),
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
            match self.stage {
                Stage::Done => return Ok(Progress::Done),
                Stage::Record => {
                    if !routed || !self.source.arrived(self.layer)? {
                        return Ok(Progress::Waiting);
                    }
                    self.stats.record_wait += self.since.elapsed();
                    let record = self.source.record(self.layer);
                    if std::mem::take(&mut self.relay) {
                        part.experts_round(env, self.layer, &self.resident, Some(record))?;
                    }
                    let counts = part.picked_counts(self.layer, self.rows, record);
                    feed.store.record_counts(self.at, &counts, 1.0);
                    let picked = |e: &u32| counts[*e as usize] > 0 && owned(*e);
                    let all = 0..counts.len() as u32;
                    self.stats.layers = 1;
                    self.stats.picked = all.clone().filter(picked).count() as u64;
                    self.missing =
                        all.filter(picked).filter(|e| self.resident[*e as usize] == 0).collect();
                    self.missing.sort_by_key(|e| std::cmp::Reverse(counts[*e as usize]));
                    self.stats.missing = self.missing.len() as u64;
                    if self.missing.is_empty() {
                        return self.finish(feed);
                    }
                    self.stage = Stage::Compute;
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
                        return self.finish(feed);
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

    fn finish(&mut self, feed: &mut Feed) -> Result<Progress, Error> {
        feed.layer_done(self.part, self.layer + 1, &self.stats);
        self.stage = Stage::Done;
        Ok(Progress::Done)
    }
}
