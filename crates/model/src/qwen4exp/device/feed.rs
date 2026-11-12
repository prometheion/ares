//! The expert store as the parts of a model use it: how it is laid out for a
//! placement, how the slots of a device are made, and how the route records
//! the parts' routers leave are handed to it.
//!
//! The store is owned by one thread at a time. [`Session`](super::Session)
//! drives it from its own thread; the runtime's device threads of
//! [`Qwen4Exp`](super::Qwen4Exp) share it behind a lock that none of them
//! waits for in a step.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_cuda::{Device, ExpertShape, MAX_BATCH, RouteGroup, RouteRecord};
use ares_gguf::Gguf;
use ares_io::{ExpertMap, Reader};
use ares_runtime::expert_store::cpu::{CpuPool, Placement as CpuPlacement, Topology};
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, LayerSpec, Serve, StoreConfig, StorePlan,
};
use ares_weights::{Lookahead, LookaheadRule};

use super::Error;
use super::board::Board;
use super::misses::WideStats;
use super::part::ExpertAt;
use super::placement::Placement;
use super::session::Fill;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Where the experts' tensors are in the file, for a store with `per_layer`
/// entries for each layer of the model.
pub(super) fn expert_map(gguf: &Gguf, c: &Config, per_layer: usize) -> Result<ExpertMap, Error> {
    let patterns = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    Ok(if per_layer > 1 {
        let names: Vec<Vec<String>> = (0..c.n_layer * per_layer)
            .map(|at| {
                patterns.iter().map(|p| format!("blk.{}.{p}.weight", at / per_layer)).collect()
            })
            .collect();
        ExpertMap::new(gguf, &names)?
    } else {
        ExpertMap::discover(gguf, &patterns)?
    })
}

/// How the experts of a model are laid out for the store: where each layer's
/// matrices are in the file and in a slot, and which device runs which layer.
#[derive(Clone, Debug)]
pub(super) struct Layout {
    /// One entry per layer of the store: a layer once per device that owns
    /// experts of it.
    pub specs: Vec<LayerSpec>,
    pub shapes: Vec<ExpertShape>,
    /// Entries of the store per layer of the model: 1, or the number of
    /// devices when they share every layer.
    pub per_layer: usize,
    /// The bytes of the experts a device would hold if it held its share of
    /// them all.
    pub expert_bytes: u64,
}

impl Layout {
    /// The layout of the experts of `placement`, and where their tensors are
    /// in the file.
    pub(super) fn new(
        gguf: &Gguf,
        c: &Config,
        w: &Weights<'_>,
        placement: &Placement,
    ) -> Result<(Self, ExpertMap), Error> {
        let per_layer = if placement.tensor.is_some() { placement.devices.len() } else { 1 };
        let store_layers = c.n_layer * per_layer;
        let map = expert_map(gguf, c, per_layer)?;
        let mut specs = Vec::with_capacity(store_layers);
        let mut shapes = Vec::with_capacity(store_layers);
        let mut expert_bytes = 0u64;
        for at in 0..store_layers {
            let layer = at / per_layer;
            let moe = &w.layers[layer].moe;
            let shape = ExpertShape {
                n_embd: c.n_embd as u32,
                n_ff: c.experts.ff as u32,
                n_experts: c.experts.count as u32,
                top_k: c.experts.used as u32,
                gate_ty: moe.gate.ty(),
                down_ty: moe.down.ty(),
            };
            let geometry = || invalid(format!("layer {layer}: the experts do not fit a slot"));
            let slot_bytes = shape.slot_bytes().ok_or_else(geometry)?;
            specs.push(LayerSpec {
                device: if per_layer > 1 {
                    at % per_layer
                } else {
                    placement
                        .device_of(layer)
                        .ok_or_else(|| invalid(format!("layer {layer} is on no device")))?
                },
                offsets: [
                    0,
                    shape.up_offset().ok_or_else(geometry)?,
                    shape.down_offset().ok_or_else(geometry)?,
                ],
                slot_bytes,
            });
            expert_bytes += slot_bytes * c.experts.count as u64 / per_layer as u64;
            shapes.push(shape);
        }
        Ok((Self { specs, shapes, per_layer, expert_bytes }, map))
    }

    /// The most bytes a tier in host memory can use: one slot per expert that
    /// has none in VRAM is all it ever holds, and a slot is the expert and a
    /// few blocks of the file.
    pub(super) fn ram_wanted(&self, vram_bytes: u64) -> u64 {
        (self.expert_bytes + (self.expert_bytes >> 4)).saturating_sub(vram_bytes).max(64 << 20)
    }

    /// The store layer of layer `layer` on device `part`.
    pub(super) fn store_layer(&self, part: usize, layer: usize) -> usize {
        if self.per_layer > 1 { layer * self.per_layer + part } else { layer }
    }
}

/// Gives `config` the transit slots of every device and the staging slots in host memory
/// for the experts that wide steps miss (`ares_runtime::expert_store`'s transit), from what
/// the budgets of the tiers leave outside them. `slots` sets the transit slots of every
/// device instead (0: none, and the misses go into the tier's slots).
///
/// - A device's transit holds half of the experts of a layer that the device computes and
///   that have no slot in VRAM, so that two rounds bring a layer's misses: one batch is
///   computed while the next is uploaded. At least two rows' picks, at most an eighth of
///   the device's bytes for slots.
/// - Staging holds twice the experts of a layer that are in no tier (the reads of a layer,
///   and those read ahead for the next), at least twice what the drives take at once, at
///   most an eighth of the bytes of the RAM tier.
pub(super) fn plan_transit(
    config: &mut StoreConfig,
    layout: &Layout,
    reader: &Reader,
    slots: Option<u32>,
) {
    let all = layout.expert_bytes.max(1) as f64;
    let vram: u64 = config.vram_bytes.iter().sum();
    let outside_vram = (1.0 - vram as f64 / all).clamp(0.0, 1.0);
    let in_no_tier = (1.0 - (vram as f64 + config.ram_bytes as f64) / all).clamp(0.0, 1.0);
    let n_experts = f64::from(layout.shapes[0].n_experts);
    let stride = layout.specs.iter().map(|s| s.slot_bytes).max().unwrap_or(1).max(1);
    let floor = 2 * config.top_k;
    // What the tiers must keep: the slots a decode step needs in every layer.
    let tier_floor = u64::from(config.max_rows * config.top_k) + config.vram_reserve as u64;
    config.transit_slots = config
        .vram_bytes
        .iter()
        .enumerate()
        .map(|(device, bytes)| {
            let kept: u64 = layout
                .specs
                .iter()
                .filter(|spec| spec.device == device)
                .map(|spec| spec.slot_bytes * tier_floor)
                .sum();
            let most = (bytes / 8).min(bytes.saturating_sub(kept)) / stride;
            let most = most.min(u64::from(u32::MAX)) as u32;
            let wanted = (n_experts / layout.per_layer as f64 * outside_vram / 2.0).ceil() as u32;
            let slots = slots.unwrap_or(wanted.max(floor)).min(most);
            if slots < 2 { 0 } else { slots }
        })
        .collect();
    // A slot of staging is an expert and a few blocks of the file.
    let stage = stride + (stride >> 4);
    let kept = 2 * layout.specs.len() as u64 * stage;
    let most = (config.ram_bytes / 8).min(config.ram_bytes.saturating_sub(kept)) / stage;
    let most = most.min(u64::from(u32::MAX)) as u32;
    let at_once =
        (ares_io::Config::default().queue_depth * reader.drives().len().max(1)).div_ceil(3) as u32;
    let wanted = (2.0 * n_experts * in_no_tier).ceil() as u32;
    let staging = wanted.max(2 * at_once).min(most);
    config.staging_slots = if staging < 2 { 0 } else { staging };
    if config.staging_slots == 0 || slots == Some(0) {
        config.transit_slots.clear();
        config.staging_slots = 0;
    }
}

/// The slots of one device: its pools and tables, and the host shares that
/// let the CPU compute what a layer misses, reserved in the device's ledger
/// and in `pinned`.
pub(super) fn device_slots(
    device: &Device,
    ledger: &mut Ledger,
    pinned: &mut Ledger,
    pools: &[(u64, u32)],
    layout: &Layout,
    serve: Serve,
    rows: u32,
) -> Result<CudaSlots, Error> {
    let mut slots = CudaSlots::new(
        device,
        ledger,
        pinned,
        pools,
        layout.shapes.len(),
        layout.shapes[0].n_experts,
    )?;
    if serve != Serve::Stream {
        slots.add_host_shares(ledger, pinned, layout.shapes[0].host_share_bytes(rows))?;
    }
    Ok(slots)
}

/// Where the experts of the layers of `part` are on its device.
pub(super) fn places(
    store: &ExpertStore<CudaSlots>,
    layers: std::ops::Range<usize>,
    layout: &Layout,
    index: usize,
) -> Vec<ExpertAt> {
    layers
        .map(|layer| {
            let at = layout.store_layer(index, layer);
            let (device, pool) = store.place(at);
            debug_assert_eq!(device, index);
            let slots = store.device(device);
            ExpertAt {
                table: slots.table(at),
                pool: slots.pool(pool).0,
                host: slots.host_share(at),
            }
        })
        .collect()
}

/// The CPU workers of the store: all the cores the system reports but
/// `reserved`, which belong to the threads that drive the step.
pub(super) fn cpu_pool(workers: Option<usize>, reserved: usize) -> Result<CpuPool, Error> {
    let placement = match workers {
        Some(workers) => CpuPlacement::unpinned(workers.max(1)),
        None => CpuPlacement::for_machine(&Topology::read(), reserved),
    };
    CpuPool::new(&placement, Duration::from_millis(20))
        .map_err(|e| invalid(format!("the CPU workers of the expert store: {e}")))
}

/// The expert store and the route records that feed it.
pub(super) struct Feed {
    pub store: ExpertStore<CudaSlots>,
    boards: Vec<Arc<Board>>,
    /// The first layer of every part.
    firsts: Vec<usize>,
    layout: Layout,
    used: u32,
    rows: u32,
    /// Per part: the layers of the running step whose route records were
    /// handed to the store.
    followed: Vec<usize>,
    groups: Vec<RouteGroup>,
    /// What the wide steps found and waited for, since the counters were read.
    wide: WideStats,
    /// Per layer of the store: the experts the wide step before found in no slot there,
    /// which is the guess for the next one, and whether the running step has made it.
    last_missing: Vec<Vec<u32>>,
    guessed: Vec<bool>,
    /// How many layers ahead the guesses are read, from the time a guessed read takes and
    /// the time of a layer.
    lookahead: Lookahead,
    /// The guessed reads the store had counted, and their time, when they were last looked
    /// at.
    ahead_seen: (u64, f64),
}

impl Feed {
    pub(super) fn new(
        store: ExpertStore<CudaSlots>,
        boards: Vec<Arc<Board>>,
        firsts: Vec<usize>,
        layout: Layout,
        used: u32,
    ) -> Self {
        Self {
            followed: vec![0; boards.len()],
            last_missing: vec![Vec::new(); layout.specs.len()],
            guessed: vec![false; layout.specs.len()],
            lookahead: Lookahead::default(),
            ahead_seen: (0, 0.0),
            store,
            boards,
            firsts,
            layout,
            used,
            rows: 0,
            groups: Vec::new(),
            wide: WideStats::default(),
        }
    }

    /// Whether the misses of layer `at` of the store go through transit.
    pub(super) fn has_transit(&self, at: usize) -> bool {
        self.store.transit_pool(at).is_some() && self.store.staging_slots() > 0
    }

    /// How many experts of layer `at` a round from transit waits for: half of the transit
    /// slots of the layer's device, so that the other half fills while the round runs.
    pub(super) fn transit_batch(&self, at: usize) -> usize {
        self.store.transit_pool(at).map_or(1, |(_, _, slots)| (slots as usize / 2).max(1))
    }

    /// The address of transit slot `slot` of the device of layer `at`.
    pub(super) fn transit_address(&self, at: usize, slot: u32) -> u64 {
        let Some((pool, stride, _)) = self.store.transit_pool(at) else { return 0 };
        let (device, _) = self.store.place(at);
        self.store.device(device).pool(pool).0.as_raw() + u64::from(slot) * stride
    }

    /// The record of layer `at` of the store, in a wide step, named `missing`, which have
    /// no slot: they are brought into transit, and the reads of the layers ahead start on a
    /// guess.
    pub(super) fn named(&mut self, at: usize, missing: &[u32]) -> Result<(), Error> {
        self.want(at, missing)?;
        self.last_missing[at] = missing.to_vec();
        self.guess_ahead(at + self.layout.per_layer)
    }

    /// Brings `missing` of layer `at` of the store into transit, and notes how many of
    /// those that have to be read had been guessed and how long the guessed reads took.
    pub(super) fn want(&mut self, at: usize, missing: &[u32]) -> Result<(), Error> {
        let named = self.store.transit_want(at, missing)?;
        // The guess is the same for every distance: what is chosen is how many layers a
        // guessed read needs to arrive.
        let rule = LookaheadRule::default();
        for distance in 1..=rule.max_distance {
            self.lookahead.observe_recall(distance, named.guessed, named.guessed + named.unguessed);
        }
        let stats = self.store.stats();
        let (done, ms) = (stats.transit_ahead_done, stats.transit_ahead_ms);
        if done > self.ahead_seen.0 {
            self.lookahead
                .observe_read((ms - self.ahead_seen.1) / (done - self.ahead_seen.0) as f64);
        }
        self.ahead_seen = (done, ms);
        Ok(())
    }

    /// Starts the guessed reads of the layers from `first` of the store on, every
    /// `per_layer`-th (the layers of one part, or all when the parts hold a range of layers
    /// each), as far ahead as the lookahead says.
    fn guess_ahead(&mut self, first: usize) -> Result<(), Error> {
        let step = self.layout.per_layer;
        for at in (first..self.guessed.len()).step_by(step).take(self.lookahead.distance()) {
            if !std::mem::replace(&mut self.guessed[at], true) && self.has_transit(at) {
                self.store.transit_guess(at, &self.last_missing[at])?;
            }
        }
        Ok(())
    }

    /// The board of part `index`.
    pub(super) fn board(&self, index: usize) -> Arc<Board> {
        self.boards[index].clone()
    }

    pub(super) fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The layers of every part whose route records the store has.
    pub(super) fn followed(&self) -> &[usize] {
        &self.followed
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers, most wanted first, as far as the slots go: from the file into
    /// VRAM, and into RAM what VRAM has no room for. `owner` says which device
    /// owns an expert when the devices share the layers. Returns when nothing
    /// moves any more; no step may run meanwhile.
    pub(super) fn fill(
        &mut self,
        n_layer: usize,
        n_experts: usize,
        owner: &dyn Fn(usize, u32) -> usize,
        experts_of: &mut dyn FnMut(usize) -> Vec<u32>,
    ) -> Result<Fill, Error> {
        let started = Instant::now();
        self.store.take_stats();
        let before = self.store.reader().stats();
        let shared = self.layout.per_layer > 1;
        for layer in 0..n_layer {
            let wanted = experts_of(layer);
            for part in 0..self.layout.per_layer {
                let at = self.layout.store_layer(part, layer);
                let (vram, ram) = self.store.slots(at);
                let mut order: Vec<u32> = wanted
                    .iter()
                    .copied()
                    .filter(|expert| !shared || owner(layer, *expert) == part)
                    .collect();
                order.truncate((vram + ram).min(n_experts));
                self.store.fill(at, &order)?;
            }
        }
        let progress =
            |store: &ExpertStore<CudaSlots>| (store.stats().uploads, store.stats().demand_reads);
        let mut last = (progress(&self.store), Instant::now());
        while !self.store.is_settled() && last.1.elapsed() < Duration::from_secs(2) {
            self.store.poll()?;
            let now = progress(&self.store);
            if now != last.0 {
                last = (now, Instant::now());
            }
        }
        let after = self.store.reader().stats();
        let stats = self.store.take_stats();
        Ok(Fill {
            uploads: stats.uploads,
            read_bytes: after.prefetch.device_bytes + after.demand.device_bytes
                - before.prefetch.device_bytes
                - before.demand.device_bytes,
            time: started.elapsed(),
        })
    }

    /// A step of `rows` rows begins: its routers have not run.
    pub(super) fn begin_step(&mut self, rows: u32) -> Result<(), Error> {
        self.rows = rows;
        self.followed.fill(0);
        self.store.begin_step();
        self.guessed.fill(false);
        if rows > MAX_BATCH {
            // The first layers of every part have no layer before them to start their
            // guesses.
            for index in 0..self.boards.len() {
                self.guess_ahead(self.layout.store_layer(index, self.firsts[index]))?;
            }
        }
        Ok(())
    }

    /// Hands the route records that have arrived to the store, per part in
    /// the order of its layers, of the first `enqueued[part]` layers, and lets
    /// the store work. Never waits.
    pub(super) fn pump(&mut self, enqueued: &[usize]) -> Result<(), Error> {
        if self.rows > MAX_BATCH {
            // The layers of a wide step are followed by `Misses`.
            self.store.poll()?;
            return Ok(());
        }
        for (index, board) in self.boards.iter().enumerate() {
            while self.followed[index] < enqueued[index] {
                let i = self.followed[index];
                if !board.arrived(i)? {
                    break;
                }
                let record = RouteRecord::parse(board.record(i), self.used, self.rows)
                    .ok_or_else(|| invalid("a route record that is none"))?;
                let at = self.layout.store_layer(index, self.firsts[index] + i);
                self.groups.clear();
                self.groups.extend(record.groups());
                // The store uploads what the layer misses or computes it from
                // the layer's rows, by what its measured costs say.
                self.store.routed_rows(
                    at,
                    record.token(),
                    self.rows as usize,
                    &self.groups,
                    board.rows(i, self.rows as usize),
                )?;
                self.followed[index] += 1;
            }
        }
        self.store.poll()?;
        Ok(())
    }

    /// Layers of part `index` up to `followed` have been handled by [`Misses`], which found
    /// `stats`.
    pub(super) fn layer_done(
        &mut self,
        index: usize,
        followed: usize,
        stats: &WideStats,
    ) -> Result<(), Error> {
        self.followed[index] = followed;
        self.wide.add(stats);
        let at = self.layout.store_layer(index, self.firsts[index] + followed - 1);
        if self.has_transit(at) {
            // What a layer takes when it waits for no expert is what a guessed read can
            // hide behind.
            let own = stats.layer_time.saturating_sub(stats.bring_wait);
            self.lookahead.observe_layer(own.as_secs_f64() * 1e3);
            // The guesses for this layer that its record did not name are dropped.
            self.store.transit_forget(at)?;
        }
        Ok(())
    }

    /// What wide steps found and waited for since the last call.
    pub(super) fn take_wide(&mut self) -> WideStats {
        std::mem::take(&mut self.wide)
    }

    /// Whether every route record of the `enqueued` layers is with the store.
    pub(super) fn is_followed(&self, enqueued: &[usize]) -> bool {
        self.followed == enqueued
    }

    /// The step has ended on the devices.
    pub(super) fn end_step(&mut self) -> Result<(), Error> {
        Ok(self.store.end_step()?)
    }

    /// What a step that stands still looks like, for the error that ends it.
    pub(super) fn standing_still(&self, enqueued: &[usize]) -> String {
        let stats = self.store.stats();
        format!(
            "per part {enqueued:?} layers are enqueued and {:?} have their route record handled; \
             the store waits for {} experts and is {}settled; in this step and since the \
             counters were read: {} picks, {} in a slot, {} computed on the CPU, {} uploads, {} \
             reads from the file",
            self.followed,
            self.store.waiting(),
            if self.store.is_settled() { "" } else { "not " },
            stats.picks,
            stats.hits,
            stats.cpu_experts,
            stats.uploads,
            stats.demand_reads,
        )
    }
}

/// Builds the store of a placement from the slots of its devices.
pub(super) fn build_store(
    plan: StorePlan,
    layout: &Layout,
    map: ExpertMap,
    reader: Reader,
    slots: Vec<CudaSlots>,
    region: HostRegion,
    cpu: Option<(CpuPool, Serve)>,
) -> Result<ExpertStore<CudaSlots>, Error> {
    let mut store = ExpertStore::new(plan, map, reader, slots, region, None)?;
    if let Some((pool, serve)) = cpu {
        store.enable_cpu(pool, layout.shapes.clone(), serve)?;
    }
    Ok(store)
}
