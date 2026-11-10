//! Item 1: the RAM tier as a cache in front of the SSD.
//!
//! Sixteen layers of 128 experts, 8 picked per token and layer, and in every
//! layer a tier of 32 slots (a quarter of the experts) in front of the SSD.
//! The model waits for an expert that is not in the tier: that is a miss, and
//! the number of reads it has to start itself per token is what the tier is
//! for.
//!
//! Time is counted in layers. A layer computes for 1; a read from the SSD
//! takes 1.5 (give or take a tenth), four can run at once, and a read that
//! starts when a layer's router has run is done before the router of the
//! layer after next. Demand reads start at once and take a lane while they
//! run, so guesses compete with them for the SSD, which is the case when
//! experts are streamed: here a token's misses alone would fill the lanes.
//! The predictor is a noisy oracle: it names, for the layer two ahead, the
//! experts that layer will route to with a recall of 0.8, and plausible wrong
//! experts for the rest (a recall of this size is what the sources report for
//! the next layer; it is an assumption here, not a measurement of anything).

use std::collections::{HashMap, VecDeque};

use ares_weights::{ExpertHeat, GuessQueue, SlotSet, Start, Swap, SwapRule, Want};

use crate::synthetic::{Rng, Shape, Workload};

pub const LAYERS: usize = 16;
pub const EXPERTS: usize = 128;
/// Passes whose routing starts every tier (a profile), before the counting.
const WARM: usize = 300;

/// What the tier, the SSD and the predictor are like.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Slots per layer.
    pub slots: usize,
    /// Reads that can run at once.
    pub lanes: usize,
    /// Layers that one read takes.
    pub read: f64,
    /// The share of a layer's picks the predictor names one layer ahead; it
    /// falls by 0.05 for every layer further.
    pub recall: f64,
    /// How many layers ahead the guesses are read for.
    pub distance: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self { slots: 32, lanes: 4, read: 1.5, recall: 0.85, distance: 2 }
    }
}

impl Params {
    pub fn recall_at(&self, distance: usize) -> f64 {
        (self.recall - 0.05 * (distance as f64 - 1.0)).max(0.0)
    }
}

/// How a tier is fed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Re-elected every 32 tokens from the heat, with the swaps free; what is
    /// not in the tier is read for the token and dropped.
    Elect,
    /// An expert the model waited for is kept, replacing the coldest member;
    /// nothing is protected, nothing is predicted.
    Admit,
    /// Also reads predicted experts, each taking its slot the moment it is
    /// guessed, with nothing protected.
    Eager,
    /// The same, with the experts the step needs protected.
    EagerProtected,
    /// Predicted experts wait for a free lane and take their slot then;
    /// experts the step needs are protected; an unused guess is worth
    /// nothing. This is how the bookkeeping is meant to be called.
    Late,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub tokens: usize,
    /// Experts the model had to wait for.
    pub misses: usize,
    /// Of them, experts that a guess was already reading.
    pub late: usize,
    /// Of them, experts that were in the tier when the layer began and were
    /// evicted by another demand read of the same layer.
    pub evicted_in_use: usize,
    /// Guessed reads that were started.
    pub guess_reads: usize,
    /// What the layers waited for reads, in layers: per layer, the longest
    /// wait of one of its experts, summed.
    pub wait: f64,
    /// From a guess being made to its data being there, for each guess that
    /// arrived, in layers.
    pub guess_latency: Vec<f64>,
    /// What a runtime sees of the predictor at every distance up to 4, though
    /// it reads for one: (distance, experts the layer routed that were not in
    /// its tier and a prediction had named, experts it routed that were not in
    /// its tier). Layers with none of the latter are left out.
    pub predictions: Vec<(usize, u32, u32)>,
}

impl Outcome {
    /// Experts the model waited for, per token.
    pub fn misses_per_token(&self) -> f64 {
        self.misses as f64 / self.tokens as f64
    }

    /// Reads the model had to start itself, per token: the misses that no
    /// guess was reading.
    pub fn demand_reads_per_token(&self) -> f64 {
        (self.misses - self.late) as f64 / self.tokens as f64
    }

    pub fn wait_per_token(&self) -> f64 {
        self.wait / self.tokens as f64
    }
}

struct Flight {
    /// When the guess it came from was made.
    made: f64,
    done: f64,
    layer: usize,
    swap: Swap,
}

struct Sim<'a> {
    work: &'a Workload,
    params: Params,
    arm: Arm,
    heat: ExpertHeat,
    sets: Vec<SlotSet>,
    queue: GuessQueue,
    inflight: Vec<Flight>,
    /// When the demand reads that are running are done; they occupy lanes.
    demand_until: Vec<f64>,
    /// Reads whose slot was taken when they were guessed and that wait for a lane.
    bound: VecDeque<(usize, Swap)>,
    /// When a guess for (layer, expert) was made.
    made_at: HashMap<(usize, u32), f64>,
    rng: Rng,
    /// Draws the predictions that are only observed, not read.
    observer: Rng,
    out: Outcome,
}

/// What a predictor with the given `recall` says layer `layer` will route to
/// at `pass`: the experts it will route to, each named with probability
/// `recall`, and plausible wrong ones for the rest, in a random order.
pub fn predict(work: &Workload, rng: &mut Rng, pass: usize, layer: usize, recall: f64) -> Vec<u32> {
    let truth = work.routed(pass, layer);
    let mut guess: Vec<u32> = truth.iter().copied().filter(|_| rng.unit() < recall).collect();
    while guess.len() < truth.len() {
        let wrong = work.plausible_expert(layer, rng);
        if !guess.contains(&wrong) && !truth.contains(&wrong) {
            guess.push(wrong);
        }
    }
    for i in (1..guess.len()).rev() {
        guess.swap(i, rng.below(i + 1));
    }
    guess
}

impl Sim<'_> {
    fn new(work: &Workload, arm: Arm, params: Params) -> Sim<'_> {
        let mut heat = ExpertHeat::new(LAYERS, EXPERTS, 0.5);
        let mut sets: Vec<SlotSet> =
            (0..LAYERS).map(|_| SlotSet::new(EXPERTS, params.slots)).collect();
        for pass in 0..WARM {
            for layer in 0..LAYERS {
                heat.record(layer, work.routed(pass, layer));
            }
        }
        for (layer, set) in sets.iter_mut().enumerate() {
            for swap in set.elect(heat.layer(layer), &SwapRule::default()) {
                set.commit(&swap);
            }
        }
        Sim {
            work,
            params,
            arm,
            heat,
            sets,
            queue: GuessQueue::new(),
            inflight: Vec::new(),
            demand_until: Vec::new(),
            bound: VecDeque::new(),
            made_at: HashMap::new(),
            rng: Rng::new(99),
            observer: Rng::new(5),
            out: Outcome::default(),
        }
    }

    /// Puts waiting guesses on the free lanes.
    fn pump(&mut self, step: u64, now: f64) {
        self.demand_until.retain(|done| *done > now);
        while self.inflight.len() + self.demand_until.len() < self.params.lanes {
            let started = match self.arm {
                Arm::Late => {
                    self.queue.start(&mut self.sets, &self.heat, step).map(|s| (s.layer, s.swap))
                }
                Arm::Eager | Arm::EagerProtected => self.bound.pop_front(),
                Arm::Elect | Arm::Admit => None,
            };
            let Some((layer, swap)) = started else { break };
            self.out.guess_reads += 1;
            // A read takes its time give or take a tenth.
            let took = self.params.read * (0.9 + 0.2 * self.rng.unit());
            let made = self.made_at.get(&(layer, swap.load)).copied().unwrap_or(now);
            self.inflight.push(Flight { made, done: now + took, layer, swap });
        }
    }

    /// The model waits for `expert` of `layer`. The read is made at once,
    /// ahead of the guesses, and takes a lane while it runs. Returns how long
    /// the model waits.
    fn demand(&mut self, layer: usize, expert: u32, step: u64, now: f64) -> f64 {
        match self.sets[layer].start_read(expert, Want::Demand, step, self.heat.layer(layer)) {
            Start::Slot(swap) => {
                self.demand_until.push(now + self.params.read);
                self.sets[layer].commit(&swap);
                self.params.read
            }
            Start::Loading => {
                // A guess is reading it: promote it and wait for what is left.
                self.out.late += 1;
                let on_lane =
                    self.inflight.iter().position(|f| f.layer == layer && f.swap.load == expert);
                let waiting = self.bound.iter().position(|(l, s)| *l == layer && s.load == expert);
                let (swap, wait) = match (on_lane, waiting) {
                    (Some(i), _) => {
                        let flight = self.inflight.remove(i);
                        self.out.guess_latency.push(flight.done - flight.made);
                        (flight.swap, (flight.done - now).max(0.0))
                    }
                    (None, Some(i)) => (self.bound.remove(i).expect("found").1, self.params.read),
                    (None, None) => unreachable!("a loading slot has a read"),
                };
                self.sets[layer].commit(&swap);
                wait
            }
            Start::NoRoom | Start::Present => self.params.read,
        }
    }

    fn tick(&mut self, pass: usize, layer: usize) {
        let step = pass as u64 + 1;
        let now = (pass * LAYERS + layer) as f64;
        // Reads that have arrived by now.
        let (done, still): (Vec<Flight>, Vec<Flight>) =
            std::mem::take(&mut self.inflight).into_iter().partition(|f| f.done <= now);
        self.inflight = still;
        for flight in done {
            self.sets[flight.layer].commit(&flight.swap);
            self.out.guess_latency.push(flight.done - flight.made);
        }
        self.pump(step, now);

        // The router of this layer has run.
        self.made_at.retain(|(l, _), _| *l != layer);
        let routed = self.work.routed(pass, layer).to_vec();
        self.heat.record(layer, &routed);
        match self.arm {
            Arm::Late => {
                self.sets[layer].routed(&routed, step);
                self.queue.forget_layer(layer);
            }
            Arm::EagerProtected => self.sets[layer].routed(&routed, step),
            // Used, but needed by no step: step 0 is over already.
            Arm::Admit | Arm::Eager => self.sets[layer].routed(&routed, 0),
            Arm::Elect => {}
        }
        // Guesses for this layer that never got a lane give back their slot.
        let mut i = 0;
        while i < self.bound.len() {
            if self.bound[i].0 == layer {
                let (_, swap) = self.bound.remove(i).expect("in range");
                self.sets[layer].abort(&swap);
            } else {
                i += 1;
            }
        }

        let missing: Vec<u32> =
            routed.iter().copied().filter(|e| !self.sets[layer].contains(*e)).collect();
        self.out.misses += missing.len();
        let mut layer_wait = 0.0f64;
        for expert in &missing {
            let wait = if self.arm == Arm::Elect {
                self.params.read
            } else {
                self.demand(layer, *expert, step, now)
            };
            layer_wait = layer_wait.max(wait);
        }
        if self.arm != Arm::Elect {
            // A demand read may have evicted an expert of this layer that was
            // in the tier and is needed as well.
            for expert in &routed {
                if !self.sets[layer].contains(*expert) && !missing.contains(expert) {
                    self.out.misses += 1;
                    self.out.evicted_in_use += 1;
                    let wait = self.demand(layer, *expert, step, now);
                    layer_wait = layer_wait.max(wait);
                }
            }
        }
        self.out.wait += layer_wait;

        // What the predictor would say for each distance, to be compared with
        // what the layer then routes to.
        if !matches!(self.arm, Arm::Elect | Arm::Admit) {
            for distance in 1..=4 {
                let target = layer + distance;
                if target >= LAYERS {
                    break;
                }
                let wanted: Vec<u32> = self
                    .work
                    .routed(pass, target)
                    .iter()
                    .copied()
                    .filter(|e| !self.sets[target].contains(*e))
                    .collect();
                if wanted.is_empty() {
                    continue;
                }
                let recall = self.params.recall_at(distance);
                let guess = predict(self.work, &mut self.observer, pass, target, recall);
                let hits = wanted.iter().filter(|e| guess.contains(e)).count();
                self.out.predictions.push((distance, hits as u32, wanted.len() as u32));
            }
        }

        // The predictions for the layer a distance ahead.
        let target = layer + self.params.distance;
        if !matches!(self.arm, Arm::Elect | Arm::Admit) && target < LAYERS {
            let recall = self.params.recall_at(self.params.distance);
            let guess = predict(self.work, &mut self.rng, pass, target, recall);
            for expert in &guess {
                self.made_at.entry((target, *expert)).or_insert(now);
            }
            match self.arm {
                Arm::Late => self.queue.guess(&mut self.sets, target, &guess, step),
                Arm::Eager | Arm::EagerProtected => {
                    if self.arm == Arm::EagerProtected {
                        self.sets[target].predicted(&guess, step);
                    }
                    for expert in guess {
                        let heat = self.heat.layer(target);
                        if let Start::Slot(swap) =
                            self.sets[target].start_read(expert, Want::Guess, step, heat)
                        {
                            self.bound.push_back((target, swap));
                        }
                    }
                }
                Arm::Elect | Arm::Admit => unreachable!(),
            }
        }
        self.pump(step, now);
    }
}

pub fn run(work: &Workload, arm: Arm, params: Params) -> Outcome {
    let mut sim = Sim::new(work, arm, params);
    for pass in WARM..work.len() {
        for layer in 0..LAYERS {
            sim.tick(pass, layer);
        }
        sim.out.tokens += 1;
        if pass % 32 == 31 {
            sim.heat.decay();
            if arm == Arm::Elect {
                for (layer, set) in sim.sets.iter_mut().enumerate() {
                    for swap in set.elect(sim.heat.layer(layer), &SwapRule::default()) {
                        set.commit(&swap);
                    }
                }
            }
        }
    }
    sim.out
}

/// The workload every premise test of the tiers is run on.
pub fn workload(passes: usize) -> Workload {
    let shape = Shape { layers: LAYERS, early: 4, experts: EXPERTS, k: 8, vocab: 2000, topics: 6 };
    Workload::generate(shape, passes, 11)
}

/// The premise of the cache: on a workload whose routing is local and whose
/// topics change, a tier that keeps what it read, takes slots for guesses only
/// when a lane is free and protects what the step needs makes the model
/// start fewer reads and wait less than a tier that is only re-elected.
///
/// The arms add one thing each, in this order: keeping what was read, reading
/// guesses (their slots taken when they are made, nothing protected),
/// protecting what the step needs, and taking the slot of a guess only when
/// its read starts. What was found on this workload, at the default
/// parameters (see the printed table for the others):
///
/// - Keeping what was read, on its own, serves no more than the free
///   re-election does. It is what lets a guess land in the tier.
/// - Reading guesses does most of the work.
/// - Protecting what the step needs stops a demand read from evicting an
///   expert of its own layer that is in use, which happened for one
///   pick in 160 to 200 without it, and saves about 7 % of the reads the model
///   starts on top of the guesses.
/// - Taking the slot of a guess when its read starts saves another 1 %. There
///   are few guesses that go stale on this workload.
/// - A guess that nobody used counting as worth nothing changes nothing that
///   can be measured here (its wrong guesses are cold anyway); it is tested on
///   its own in the unit tests.
#[test]
fn a_tier_that_admits_what_it_read_and_protects_what_the_step_needs_reads_less() {
    let work = workload(8000);
    let arms = [Arm::Elect, Arm::Admit, Arm::Eager, Arm::EagerProtected, Arm::Late];
    let mut outcomes = Vec::new();
    for params in [
        Params::default(),
        Params { lanes: 8, ..Params::default() },
        Params { recall: 0.65, ..Params::default() },
    ] {
        eprintln!("{params:?}, per token");
        let outs: Vec<Outcome> = arms.iter().map(|arm| run(&work, *arm, params)).collect();
        for (arm, out) in arms.iter().zip(&outs) {
            eprintln!(
                "  {:>14}: {:5.2} experts waited for, {:5.2} reads started by the model,                  {:5.2} layers of waiting; {} evicted in use, {} guessed reads",
                format!("{arm:?}"),
                out.misses_per_token(),
                out.demand_reads_per_token(),
                out.wait_per_token(),
                out.evicted_in_use,
                out.guess_reads,
            );
        }
        outcomes.push(outs);
    }

    // The default machine: a tier that is only re-elected against the full
    // bookkeeping.
    let [elect, admit, eager, protected, late] = &outcomes[0][..] else { unreachable!() };
    assert!(
        late.demand_reads_per_token() < 0.7 * elect.demand_reads_per_token(),
        "reads started by the model: {:.2} against {:.2}",
        late.demand_reads_per_token(),
        elect.demand_reads_per_token()
    );
    assert!(late.misses_per_token() < 0.95 * elect.misses_per_token());
    assert!(late.wait_per_token() < 0.85 * elect.wait_per_token());
    // Nothing a layer uses is evicted by a read of the same layer.
    assert_eq!((protected.evicted_in_use, late.evicted_in_use), (0, 0));
    assert!(admit.evicted_in_use > 0 && eager.evicted_in_use > 0);
    assert!(protected.demand_reads_per_token() < 0.95 * eager.demand_reads_per_token());
    assert!(protected.wait_per_token() < eager.wait_per_token());
    assert!(late.demand_reads_per_token() <= protected.demand_reads_per_token());

    // With room for more reads at once the saving in waiting is larger, and with
    // a predictor that names only two in three the reads still fall.
    let [elect, .., late] = &outcomes[1][..] else { unreachable!() };
    assert!(late.wait_per_token() < 0.6 * elect.wait_per_token());
    let [elect, .., late] = &outcomes[2][..] else { unreachable!() };
    assert!(late.demand_reads_per_token() < 0.85 * elect.demand_reads_per_token());
}
