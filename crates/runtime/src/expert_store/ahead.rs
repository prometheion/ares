//! Reads ahead of a router: what a prediction names for a layer that the running step has not
//! routed yet (the routers of later layers applied on the device to the rows of this one,
//! Pre-gated MoE, FATE, Run My AI) and that no tier holds is read from the file into the RAM
//! tier before the layer's record arrives. A read ahead only moves bytes: the record finds the
//! expert in RAM or on its way, as it would find any other, and the step computes what it
//! would compute anyway.
//!
//! What the drive gives to guesses it takes from the reads a step waits for, so the
//! read-ahead keeps within bounds that come from the drive and the step, never from numbers
//! of a machine:
//!
//! - **Only what is picked more often than not.** When the tiers hold what a text uses, an
//!   expert that no tier holds is one the text has not used, and a prediction that names it
//!   is mostly wrong. So the store counts, per kind of step (one row, or several: the rows
//!   of a prompt or of a window pick far more experts that no tier holds), distance of the
//!   prediction (one layer or two) and place in its list (the best rank among the rows'
//!   picks), how often an expert named there that no tier held was then picked, read or
//!   not, and reads ahead only at the places where that has been at least half the time: a
//!   read for nothing then costs the drive no more than the reads that are used save the
//!   steps. Where the tiers hold little (one card, a small RAM tier, a large file) more
//!   places qualify; where they hold nearly everything few do, and nothing is read.
//! - **In flight**: at most as many experts as the drives take of prefetches at once
//!   ([`ares_io::Reader::prefetch_room`]). The reader keeps every prefetch behind every
//!   demand read and halves the prefetch chunks at a drive while a demand chunk is there, so
//!   a demand read shares the drive with at most a few chunks of 512 KiB.
//! - **Bytes per step**: at most what the drive could read in the time of a step beside
//!   what the step reads on demand (the drive's measured pace, a mean of the step's time and
//!   of its demand bytes over the steps that predicted).
//! - **Dropped**: when the layer's record arrives, the guesses for it that have not started
//!   are forgotten and a read of an expert it did not pick is cancelled; one that came back
//!   for nothing stays in the RAM tier as an unused guess, worth nothing there, the first to
//!   give way.
//!
//! A read ahead takes only room of the RAM tier that nobody uses ([`Want::Spare`]): a free
//! slot, a guess nobody used, a copy of an expert that VRAM holds. It never takes the place
//! of a member a later step may need, so a read for nothing costs the drive and no more.

use std::collections::HashMap;
use std::time::Instant;

use ares_io::Reader;
use ares_weights::Want;

use super::{ExpertStore, Geometry, Phase, SlotDevice};

/// The places of a list that are counted apart: the ranks of a step of one row; later places
/// (a step of several rows names more) count with the last.
const PLACES: usize = 16;

/// The samples a place needs before its count decides, and the most it keeps: the counts are
/// halved beyond that, so that they follow the text and the tiers.
const MIN_SAMPLES: f32 = 16.0;
const MAX_SAMPLES: f32 = 512.0;

/// How often an expert that no tier held and that a prediction named was then picked, per
/// kind of step and distance of the prediction (one row or several, one layer or two: the
/// four rows of the tables) and place in its list.
#[derive(Clone, Debug, Default)]
struct Odds {
    named: [[f32; PLACES]; 4],
    picked: [[f32; PLACES]; 4],
}

impl Odds {
    /// The class of a place `place` of a prediction `distance` layers ahead in a step of
    /// `rows` rows.
    fn class(rows: usize, distance: usize, place: usize) -> (usize, usize) {
        let kind = usize::from(rows > 1);
        (2 * kind + distance.clamp(1, 2) - 1, place.min(PLACES - 1))
    }

    /// Whether a read ahead at this place is used at least as often as not.
    fn worth(&self, (d, p): (usize, usize)) -> bool {
        let named = self.named[d][p];
        named >= MIN_SAMPLES && 2.0 * self.picked[d][p] >= named
    }

    fn record(&mut self, (d, p): (usize, usize), picked: bool) {
        self.named[d][p] += 1.0;
        self.picked[d][p] += f32::from(u8::from(picked));
        if self.named[d][p] > MAX_SAMPLES {
            self.named[d][p] /= 2.0;
            self.picked[d][p] /= 2.0;
        }
    }
}

/// What a prediction has done for an expert of a layer that the running step has not
/// routed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// A tier held it or it was on its way when it was named: nothing to read.
    Held,
    /// In no tier: waits in the guess queue for room.
    Queued,
    /// Its read has started.
    Reading(u64),
    /// Its bytes are in the RAM tier.
    Arrived,
}

/// The read-ahead of the running step and its bounds.
#[derive(Debug)]
pub(super) struct Ahead {
    /// What was named for the layers ahead, with the class (distance, place) of its naming.
    named: HashMap<(usize, u32), (State, (usize, usize))>,
    odds: Odds,
    /// Reads ahead that have started and not come back, and the most there may be.
    in_flight: usize,
    lanes: usize,
    /// Bytes read ahead in the running step, and the most it may read.
    bytes: u64,
    budget: u64,
    /// The bytes of an expert's read, about.
    expert_bytes: u64,
    /// Bytes of the demand reads started in the running step.
    pub(super) demand_bytes: u64,
    /// Whether a prediction arrived in the running step, and when the step began.
    predicted: bool,
    began: Option<Instant>,
    /// Means over the steps that predicted: the milliseconds of a step and the bytes its
    /// demand reads took.
    step_ms: Option<f64>,
    demand_mean: f64,
    /// The reads ahead of experts that their record picked while they were on their way,
    /// with the time of the record.
    picked: HashMap<u64, Instant>,
}

impl Ahead {
    pub(super) fn new(layers: &[Geometry], reader: &Reader) -> Self {
        let expert_bytes =
            layers.iter().map(|g| g.lens.iter().sum::<usize>() as u64).max().unwrap_or(1).max(1);
        let lanes = (reader.prefetch_room() as u64).div_ceil(expert_bytes).max(1) as usize;
        Self {
            named: HashMap::new(),
            odds: Odds::default(),
            in_flight: 0,
            lanes,
            bytes: 0,
            budget: u64::MAX,
            expert_bytes,
            demand_bytes: 0,
            predicted: false,
            began: None,
            step_ms: None,
            demand_mean: 0.0,
            picked: HashMap::new(),
        }
    }

    /// Whether another read ahead may start now.
    pub(super) fn has_room(&self) -> bool {
        self.in_flight < self.lanes && self.bytes < self.budget
    }

    /// How a guess may start now: a read ahead that waits only when there is room and its
    /// place is worth it, and then only into room of the RAM tier that nobody uses
    /// ([`Want::Spare`]); any other guess as a guess.
    pub(super) fn may_start(&self, layer: usize, expert: u32) -> Option<Want> {
        match self.named.get(&(layer, expert)) {
            Some((State::Queued, class)) => {
                (self.has_room() && self.odds.worth(*class)).then_some(Want::Spare)
            }
            _ => Some(Want::Guess),
        }
    }
}

impl<D: SlotDevice> ExpertStore<D> {
    /// A prediction made `distance` layers ahead (1 or 2) in a step of `rows` rows says that
    /// the router of `layer`, which the running step has not reached, will pick `experts`, the
    /// likeliest first. Those
    /// that no tier holds and that are not on their way are read from the file into the RAM
    /// tier, the likeliest first and behind every read a step waits for, as far as the
    /// bounds of the read-ahead let them and only at the places of a list whose reads have
    /// been used at least as often as not (see the module); [`ExpertStore::poll`] starts
    /// them. The layer's record then finds them in RAM or on their way, and drops what it did
    /// not pick.
    ///
    /// Nothing happens between steps, for a layer whose record has arrived, or for a layer
    /// whose experts are all resident.
    pub fn ahead(&mut self, layer: usize, experts: &[u32], distance: usize, rows: usize) {
        if !self.in_step
            || self.phase.get(layer) != Some(&Phase::Ahead)
            || self.layers[layer].resident
        {
            return;
        }
        self.ahead.predicted = true;
        let n_experts = self.heat.n_experts();
        let mut read = Vec::new();
        for (place, &expert) in experts.iter().enumerate() {
            if expert as usize >= n_experts {
                continue;
            }
            let key = (layer, expert);
            let class = Odds::class(rows, distance, place);
            if let Some((state, named)) = self.ahead.named.get_mut(&key) {
                // A nearer prediction counts for the expert from now on.
                if *state == State::Queued && class.0 < named.0 {
                    *named = class;
                }
                continue;
            }
            self.stats.predicted += 1;
            let (vram, ram) = (&self.vram.sets[layer], &self.ram.sets[layer]);
            let held = vram.contains(expert)
                || vram.is_loading(expert)
                || ram.contains(expert)
                || ram.is_loading(expert);
            let state = if held {
                State::Held
            } else {
                read.push(expert);
                State::Queued
            };
            self.ahead.named.insert(key, (state, class));
        }
        if !read.is_empty() {
            self.guesses.guess(&mut self.ram.sets, layer, &read, self.step);
        }
    }

    /// The read with number `read` of `expert` of `layer` has started from the guess queue,
    /// `bytes` long. Returns whether it is a read ahead of a router.
    pub(super) fn ahead_started(
        &mut self,
        layer: usize,
        expert: u32,
        read: u64,
        bytes: u64,
    ) -> bool {
        let Some((state, _)) = self.ahead.named.get_mut(&(layer, expert)) else { return false };
        if *state != State::Queued {
            return false;
        }
        *state = State::Reading(read);
        self.ahead.in_flight += 1;
        self.ahead.bytes += bytes;
        self.stats.ahead_reads += 1;
        self.stats.ahead_read_bytes += bytes;
        true
    }

    /// A read ahead has come back, or `failed` (it was cancelled).
    pub(super) fn ahead_read_done(&mut self, read: u64, layer: usize, expert: u32, failed: bool) {
        self.ahead.in_flight = self.ahead.in_flight.saturating_sub(1);
        if let Some(record) = self.ahead.picked.remove(&read)
            && !failed
        {
            self.stats.ahead_miss_ms.push(record.elapsed().as_secs_f32() * 1e3);
        }
        if let Some((state, _)) = self.ahead.named.get_mut(&(layer, expert))
            && *state == State::Reading(read)
        {
            // A read that failed is counted as never read.
            *state = if failed { State::Queued } else { State::Arrived };
        }
    }

    /// The record of `layer` picked `picks`: what was predicted for the layer is done with.
    /// A read of an expert it did not pick that has not come back is cancelled.
    pub(super) fn ahead_routed(&mut self, layer: usize, picks: &[u32]) {
        if self.ahead.named.is_empty() {
            return;
        }
        let keys: Vec<(usize, u32)> =
            self.ahead.named.keys().filter(|key| key.0 == layer).copied().collect();
        let now = Instant::now();
        for key in keys {
            let Some((state, class)) = self.ahead.named.remove(&key) else { continue };
            let picked = picks.contains(&key.1);
            self.stats.predicted_picked += u64::from(picked);
            if state != State::Held {
                // An expert no tier held when it was named: what its place is worth.
                self.ahead.odds.record(class, picked);
                self.stats.predicted_in_file += 1;
                self.stats.predicted_in_file_picked += u64::from(picked);
            }
            match state {
                State::Held => {}
                State::Queued => self.stats.ahead_left += 1,
                State::Reading(read) if picked => {
                    self.stats.ahead_used += 1;
                    self.ahead.picked.insert(read, now);
                }
                State::Reading(read) => {
                    self.stats.ahead_wasted += 1;
                    for part in 0..3 {
                        // What is at the drive comes back; the rest is not started.
                        let _ = self.reader.cancel(read * 4 + part);
                    }
                }
                State::Arrived if picked => self.stats.ahead_used += 1,
                State::Arrived => self.stats.ahead_wasted += 1,
            }
        }
    }

    pub(super) fn ahead_begin_step(&mut self) {
        let ahead = &mut self.ahead;
        ahead.named.clear();
        ahead.bytes = 0;
        ahead.demand_bytes = 0;
        ahead.predicted = false;
        ahead.began = Some(Instant::now());
    }

    /// The step has ended: what was predicted for layers it never routed is done with (a
    /// read on its way comes back as an unused guess), and the bounds of the next step
    /// follow what this one took.
    pub(super) fn ahead_end_step(&mut self) {
        for (_, (state, _)) in std::mem::take(&mut self.ahead.named) {
            match state {
                State::Held => {}
                State::Queued => self.stats.ahead_left += 1,
                State::Reading(_) | State::Arrived => self.stats.ahead_wasted += 1,
            }
        }
        let (Some(began), true) = (self.ahead.began.take(), self.ahead.predicted) else {
            return;
        };
        let ms = began.elapsed().as_secs_f64() * 1e3;
        let ahead = &mut self.ahead;
        let demand = ahead.demand_bytes as f64;
        match ahead.step_ms {
            Some(mean) => {
                ahead.step_ms = Some(mean + 0.1 * (ms - mean));
                ahead.demand_mean += 0.1 * (demand - ahead.demand_mean);
            }
            None => {
                ahead.step_ms = Some(ms);
                ahead.demand_mean = demand;
            }
        }
        // The drives' throughput as the reader has measured it lately: bytes per
        // millisecond over all of them. Unmeasured, the read-ahead keeps only its bound in
        // flight.
        let per_ms: f64 = self
            .reader
            .drive_stats()
            .iter()
            .filter(|d| d.shared_ns_per_mib > 0)
            .map(|d| (1u64 << 20) as f64 * 1e6 / d.shared_ns_per_mib as f64)
            .sum();
        let ahead = &mut self.ahead;
        ahead.budget = match (per_ms > 0.0, ahead.step_ms) {
            (true, Some(step_ms)) => {
                let idle = (per_ms * step_ms - ahead.demand_mean).max(0.0) as u64;
                idle.max(ahead.lanes as u64 * ahead.expert_bytes)
            }
            _ => u64::MAX,
        };
    }
}
