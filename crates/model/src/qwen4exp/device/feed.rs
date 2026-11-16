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
use ares_cuda::{Device, ExpertShape, MAX_BATCH, PredictedList, RouteGroup, RouteRecord};
use ares_gguf::Gguf;
use ares_io::{ExpertMap, ExpertSpan, FileId, Reader};
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
use super::prune::{PruneMask, PruneStats};
use super::session::Fill;
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::{DraftWeights, Moe, Weights};

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// Where the experts' tensors are in the file, for a store with `per_layer`
/// entries for each layer of the model. With `draft`, the experts of the
/// model's draft block are the store's layer 0, in a file of their own that
/// the reader opens behind the model's parts ([`reader`]).
pub(super) fn expert_map(
    gguf: &Gguf,
    c: &Config,
    per_layer: usize,
    draft: Option<&DraftLayer<'_>>,
) -> Result<ExpertMap, Error> {
    let patterns = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    let mut map = if per_layer > 1 {
        let names: Vec<Vec<String>> = (0..c.n_layer * per_layer)
            .map(|at| {
                patterns.iter().map(|p| format!("blk.{}.{p}.weight", at / per_layer)).collect()
            })
            .collect();
        ExpertMap::new(gguf, &names)?
    } else {
        ExpertMap::discover(gguf, &patterns)?
    };
    if let Some(draft) = draft {
        let places = draft
            .weights
            .expert_places(draft.gguf)
            .ok_or_else(|| invalid("the draft block's experts are not in its file"))?;
        let spans = places.map(|p| ExpertSpan {
            file: FileId(gguf.n_parts() as u16 + p.part),
            base: p.base,
            stride: p.stride,
            len: p.len,
        });
        map.insert_first(c.experts.count, &spans);
    }
    Ok(map)
}

/// The reader of the model's parts and, behind them, of the parts of the
/// file of its draft block.
pub(super) fn reader(gguf: &Gguf, draft: Option<&Gguf>) -> Result<Reader, Error> {
    let config = ares_io::Config::default();
    let Some(draft) = draft else {
        return Ok(Reader::open_gguf(gguf, config)?);
    };
    let paths: Vec<&std::path::Path> = (0..gguf.n_parts())
        .map(|part| gguf.path(part as u16))
        .chain((0..draft.n_parts()).map(|part| draft.path(part as u16)))
        .collect();
    Ok(Reader::open(&paths, config)?)
}

/// The draft block of a model as the store sees it.
#[derive(Clone, Copy, Debug)]
pub(super) struct DraftLayer<'a> {
    pub gguf: &'a Gguf,
    pub weights: &'a DraftWeights<'a>,
    /// The device that runs the block: the one with the output head.
    pub device: usize,
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
    /// The store's layers in front of the model's: 1 with a draft block, whose
    /// experts are layer 0 (a step routes it before any layer of the model),
    /// 0 without.
    pub base: usize,
    /// The device of the draft block and the bytes of all its experts, which
    /// stay resident there.
    pub draft: Option<(usize, u64)>,
}

impl Layout {
    /// The layout of the experts of `placement`, and where their tensors are
    /// in the file.
    pub(super) fn new(
        gguf: &Gguf,
        c: &Config,
        w: &Weights<'_>,
        placement: &Placement,
        draft: Option<&DraftLayer<'_>>,
    ) -> Result<(Self, ExpertMap), Error> {
        let per_layer = if placement.tensor.is_some() { placement.devices.len() } else { 1 };
        let store_layers = c.n_layer * per_layer;
        let map = expert_map(gguf, c, per_layer, draft)?;
        let base = usize::from(draft.is_some());
        let mut specs = Vec::with_capacity(base + store_layers);
        let mut shapes = Vec::with_capacity(base + store_layers);
        let mut expert_bytes = 0u64;
        let shape_of = |moe: &Moe<'_>| ExpertShape {
            n_embd: c.n_embd as u32,
            n_ff: c.experts.ff as u32,
            n_experts: c.experts.count as u32,
            top_k: c.experts.used as u32,
            gate_ty: moe.gate.ty(),
            down_ty: moe.down.ty(),
        };
        let spec_of = |shape: &ExpertShape, device: usize, what: &str, resident: bool| {
            let geometry = || invalid(format!("{what}: the experts do not fit a slot"));
            Ok::<_, Error>(LayerSpec {
                device,
                offsets: [
                    0,
                    shape.up_offset().ok_or_else(geometry)?,
                    shape.down_offset().ok_or_else(geometry)?,
                ],
                slot_bytes: shape.slot_bytes().ok_or_else(geometry)?,
                resident,
            })
        };
        let mut draft_bytes = None;
        if let Some(draft) = draft {
            // The draft block's experts stay in VRAM, all of them: a chain
            // routes them in every link, and a link that waits for an expert
            // costs what a token costs.
            let shape = shape_of(&draft.weights.layer.moe);
            let spec = spec_of(&shape, draft.device, "the draft block", true)?;
            draft_bytes = Some((draft.device, spec.slot_bytes * c.experts.count as u64));
            specs.push(spec);
            shapes.push(shape);
        }
        for at in 0..store_layers {
            let layer = at / per_layer;
            let shape = shape_of(&w.layers[layer].moe);
            let device = if per_layer > 1 {
                at % per_layer
            } else {
                placement
                    .device_of(layer)
                    .ok_or_else(|| invalid(format!("layer {layer} is on no device")))?
            };
            let spec = spec_of(&shape, device, &format!("layer {layer}"), false)?;
            expert_bytes += spec.slot_bytes * c.experts.count as u64 / per_layer as u64;
            specs.push(spec);
            shapes.push(shape);
        }
        Ok((Self { specs, shapes, per_layer, expert_bytes, base, draft: draft_bytes }, map))
    }

    /// The bytes of the draft block's experts on `device`: all of them on the
    /// device that runs the block, none elsewhere.
    pub(super) fn draft_bytes(&self, device: usize) -> u64 {
        self.draft.filter(|(on, _)| *on == device).map_or(0, |(_, bytes)| bytes)
    }

    /// The most bytes a tier in host memory can use: one slot per expert that
    /// has none in VRAM is all it ever holds, and a slot is the expert and a
    /// few blocks of the file.
    pub(super) fn ram_wanted(&self, vram_bytes: u64) -> u64 {
        let all = self.expert_bytes + self.draft.map_or(0, |(_, bytes)| bytes);
        (all + (all >> 4)).saturating_sub(vram_bytes).max(64 << 20)
    }

    /// The store layer of layer `layer` on device `part`.
    pub(super) fn store_layer(&self, part: usize, layer: usize) -> usize {
        self.base + if self.per_layer > 1 { layer * self.per_layer + part } else { layer }
    }

    /// The store's layers of the model, without the draft block's.
    pub(super) fn model_layers(&self) -> std::ops::Range<usize> {
        self.base..self.specs.len()
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

/// The part that owns each expert of each layer when the devices share every
/// layer, `owners[layer * n_experts + expert]`: what [`Feed::set_owners`] takes.
pub(super) fn owners(placement: &Placement, c: &Config) -> Vec<u8> {
    (0..c.n_layer)
        .flat_map(|layer| {
            (0..c.experts.count as u32).map(move |expert| placement.owner(layer, expert) as u8)
        })
        .collect()
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
    /// Per part: the layers of the running step whose predictions of the picks
    /// of the layers after them were handed to the store, and the token of the
    /// last route record (a prediction of the step carries it).
    ahead_followed: Vec<usize>,
    tokens: Vec<u32>,
    /// With both devices on every layer: per layer and expert, the part that owns
    /// it (what the predictions of the mixer name goes to the owner's layer of
    /// the store).
    owners: Option<Vec<u8>>,
    /// Per layer of the store, in wide steps: whether the step before read experts of the
    /// layer from the file (where the next one is predicted), and whether the running
    /// step's record of the layer has been named to the store.
    file_missed: Vec<bool>,
    named_wide: Vec<bool>,
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
    /// Whether the reads of the layers ahead are started on a guess.
    read_ahead: bool,
    /// The route records of the model's draft block, whose experts are layer
    /// 0 of the store.
    draft: Option<DraftRecords>,
    /// The pruning of the routers, when the model prunes (`prune.rs`).
    prune: Option<FeedPrune>,
}

/// What the store's side of pruning keeps: the mask it writes at the start
/// of a step, which experts each part owns, and the counts.
struct FeedPrune {
    mask: Arc<PruneMask>,
    /// With two devices on every layer: the mixer's index and, per layer and
    /// expert, 1 where the other device owns the expert.
    owner: Option<(usize, Vec<u8>)>,
    n_experts: usize,
    /// Whether the steps prune nothing for now (a measurement that reads a
    /// prompt the same way for every rule).
    paused: bool,
    stats: PruneStats,
    /// Per expert of a layer: picks that were pruned, as heat.
    counts: Vec<u32>,
}

/// The route records of a draft block. Its steps run one after the other
/// inside a step of the model, each with a record of its own in the same
/// place of the block's board.
struct DraftRecords {
    board: Arc<Board>,
    /// The rows of the block's step whose record the store has not seen yet.
    waiting: Option<u32>,
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
            ahead_followed: vec![0; boards.len()],
            tokens: vec![0; boards.len()],
            owners: None,
            file_missed: vec![true; layout.specs.len()],
            named_wide: vec![false; layout.specs.len()],
            last_missing: vec![Vec::new(); layout.specs.len()],
            guessed: vec![false; layout.specs.len()],
            lookahead: Lookahead::default(),
            ahead_seen: (0, 0.0),
            read_ahead: false,
            store,
            boards,
            firsts,
            layout,
            used,
            rows: 0,
            groups: Vec::new(),
            wide: WideStats::default(),
            draft: None,
            prune: None,
        }
    }

    /// The model prunes: `mask` is the one its parts copy at the start of a
    /// step, which the store writes from here on. With two devices on every
    /// layer, `owner` is the mixer's index and, per layer and expert, 1 where
    /// the other device owns the expert.
    pub(super) fn set_prune(
        &mut self,
        mask: Arc<PruneMask>,
        owner: Option<(usize, Vec<u8>)>,
        n_experts: usize,
    ) {
        self.prune = Some(FeedPrune {
            mask,
            owner,
            n_experts,
            paused: false,
            stats: PruneStats::default(),
            counts: vec![0; n_experts],
        });
    }

    /// Whether the steps from the next on prune (they do unless told not to).
    pub(super) fn set_pruning(&mut self, on: bool) {
        if let Some(prune) = &mut self.prune {
            prune.paused = !on;
        }
    }

    /// What pruning did since the last call.
    pub(super) fn take_prune(&mut self) -> PruneStats {
        self.prune.as_mut().map(|p| std::mem::take(&mut p.stats)).unwrap_or_default()
    }

    /// The board of the model's draft block: its records are those of layer 0
    /// of the store.
    pub(super) fn set_draft(&mut self, board: Arc<Board>) {
        self.draft = Some(DraftRecords { board, waiting: None });
    }

    /// A step of the draft block with `rows` rows was enqueued: its route
    /// record is on its way to the block's board.
    pub(super) fn draft_routed(&mut self, rows: u32) {
        if let Some(draft) = &mut self.draft {
            draft.waiting = Some(rows);
        }
    }

    /// Whether the store has the route record of the draft block's last step.
    pub(super) fn draft_followed(&self) -> bool {
        self.draft.as_ref().is_none_or(|draft| draft.waiting.is_none())
    }

    /// With both devices on every layer: the part that owns each expert of each
    /// layer, `owners[layer * n_experts + expert]`.
    pub(super) fn set_owners(&mut self, owners: Vec<u8>) {
        self.owners = Some(owners);
    }

    /// Sets whether wide steps read ahead on a guess (they do not unless told to).
    pub(super) fn set_read_ahead(&mut self, read_ahead: bool) {
        self.read_ahead = read_ahead;
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
        self.named_wide[at] = true;
        self.want(at, missing)?;
        self.last_missing[at] = missing.to_vec();
        self.guess_ahead(at + self.layout.per_layer)
    }

    /// Brings `missing` of layer `at` of the store into transit, and notes how many of
    /// those that have to be read had been guessed and how long the guessed reads took.
    pub(super) fn want(&mut self, at: usize, missing: &[u32]) -> Result<(), Error> {
        let named = self.store.transit_want(at, missing)?;
        self.file_missed[at] = named.guessed + named.unguessed > 0;
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
        if !self.read_ahead {
            return Ok(());
        }
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
        if self.draft.is_some() {
            // The draft block is asked as the layer behind the model's last.
            let mut order = experts_of(n_layer);
            let (vram, ram) = self.store.slots(0);
            order.truncate((vram + ram).min(n_experts));
            self.store.fill(0, &order)?;
        }
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
        self.settle()?;
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

    /// Lets the store work until nothing moves any more.
    fn settle(&mut self) -> Result<(), Error> {
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
        Ok(())
    }

    /// Brings the experts of the draft block into the tiers as far as its
    /// slots go, before the first step: the block's router runs for every
    /// link of a chain, and the first chains of a server decide what the
    /// conductor takes a link to cost. No step may run meanwhile.
    pub(super) fn fill_draft(&mut self, n_experts: usize) -> Result<(), Error> {
        if self.draft.is_none() {
            return Ok(());
        }
        let (vram, ram) = self.store.slots(0);
        let order: Vec<u32> = (0..(vram + ram).min(n_experts) as u32).collect();
        self.store.fill(0, &order)?;
        self.settle()?;
        self.store.take_stats();
        Ok(())
    }

    /// A step of `rows` rows begins: its routers have not run.
    pub(super) fn begin_step(&mut self, rows: u32) -> Result<(), Error> {
        self.rows = rows;
        self.followed.fill(0);
        self.ahead_followed.fill(0);
        self.store.begin_step();
        if let Some(prune) = self.prune.as_ref().filter(|_| rows <= MAX_BATCH) {
            prune.mask.build(&self.store, &self.layout, prune.paused);
        }
        self.guessed.fill(false);
        self.named_wide.fill(false);
        if rows > MAX_BATCH {
            // The next layer's picks are predicted where the step before read from the file.
            for index in 0..self.boards.len() {
                let board = &self.boards[index];
                for i in 0..board.layers().saturating_sub(1) {
                    let layer = self.firsts[index] + i + 1;
                    let wanted = (0..self.layout.per_layer).any(|part| {
                        let part = if self.layout.per_layer > 1 { part } else { index };
                        self.file_missed[self.layout.store_layer(part, layer)]
                    });
                    board.set_wide_wanted(i, wanted);
                }
            }
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
        if let Some(draft) = &mut self.draft
            && let Some(rows) = draft.waiting
            && draft.board.arrived(0)?
        {
            let record = RouteRecord::parse(draft.board.record(0), self.used, rows)
                .ok_or_else(|| invalid("a route record of the draft block that is none"))?;
            self.groups.clear();
            self.groups.extend(record.groups());
            self.store.routed_rows(
                0,
                record.token(),
                rows as usize,
                &self.groups,
                draft.board.rows(0, rows as usize),
            )?;
            draft.waiting = None;
        }
        if self.rows > MAX_BATCH {
            // The layers of a wide step are followed by `Misses`.
            self.follow_wide_predictions()?;
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
                if let Some(prune) = &mut self.prune {
                    prune.follow(
                        &mut self.store,
                        at,
                        index,
                        self.firsts[index] + i,
                        &record,
                        &self.groups,
                    );
                }
                // The store uploads what the layer misses or computes it from
                // the layer's rows, by what its measured costs say.
                self.store.routed_rows(
                    at,
                    record.token(),
                    self.rows as usize,
                    &self.groups,
                    board.rows(i, self.rows as usize),
                )?;
                self.tokens[index] = record.token();
                self.followed[index] += 1;
            }
        }
        // The predictions of the picks of later layers, behind the records: of the layers
        // whose record the store has, so that what a step waits for goes first.
        for index in 0..self.boards.len() {
            self.follow_predictions(index)?;
        }
        self.store.poll()?;
        Ok(())
    }

    /// Hands the store the picks the routers' parts predicted in a wide step for the layers
    /// whose records have not been named yet: per layer of the store the experts some row
    /// will pick, the most picked first, as the guesses of the layer
    /// ([`ExpertStore::transit_guess`]: what no tier holds is read ahead into staging).
    fn follow_wide_predictions(&mut self) -> Result<(), Error> {
        let top_k = self.used as usize;
        let n_experts = self.layout.shapes[0].n_experts as usize;
        let per_layer = self.layout.per_layer;
        for index in 0..self.boards.len() {
            let board = self.boards[index].clone();
            if !board.predicts_wide() {
                continue;
            }
            for i in 0..board.layers().saturating_sub(1) {
                let Some(ids) = board.take_wide(i, top_k)? else { continue };
                let layer = self.firsts[index] + i + 1;
                let mut counts = vec![0u32; n_experts];
                for id in ids.as_chunks::<4>().0 {
                    if let Some(count) = counts.get_mut(u32::from_le_bytes(*id) as usize) {
                        *count += 1;
                    }
                }
                for part in 0..per_layer {
                    let owner = if per_layer > 1 { part } else { index };
                    let at = self.layout.store_layer(owner, layer);
                    if self.named_wide[at] || !self.has_transit(at) {
                        continue;
                    }
                    let owners = self.owners.as_ref().filter(|_| per_layer > 1);
                    let mut experts: Vec<u32> = (0..n_experts as u32)
                        .filter(|e| counts[*e as usize] > 0)
                        .filter(|e| {
                            owners.is_none_or(|o| {
                                o.get(layer * n_experts + *e as usize).copied() == Some(part as u8)
                            })
                        })
                        .collect();
                    experts.sort_by(|a, b| {
                        counts[*b as usize].cmp(&counts[*a as usize]).then(a.cmp(b))
                    });
                    self.store.transit_guess(at, &experts)?;
                }
            }
        }
        Ok(())
    }

    /// Hands the store what the router part `index` predicted at the layers whose records
    /// it has, for the layers after them ([`ExpertStore::ahead`]).
    fn follow_predictions(&mut self, index: usize) -> Result<(), Error> {
        let board = &self.boards[index];
        if board.ahead() == 0 {
            return Ok(());
        }
        let n_experts = self.layout.shapes[0].n_experts as usize;
        while self.ahead_followed[index] < self.followed[index] {
            let i = self.ahead_followed[index];
            if !board.ahead_arrived(i)? {
                break;
            }
            self.ahead_followed[index] += 1;
            for d in 0..board.ahead() {
                let target = i + 1 + d;
                if target >= board.layers() {
                    break;
                }
                // A list of another step: no prediction was made at this layer.
                let Some(list) = PredictedList::parse(board.ahead_list(i, d), self.tokens[index])
                else {
                    continue;
                };
                let layer = self.firsts[index] + target;
                match &self.owners {
                    Some(owners) if self.layout.per_layer > 1 => {
                        for part in 0..self.layout.per_layer {
                            let experts: Vec<u32> = list
                                .experts()
                                .filter(|e| {
                                    owners.get(layer * n_experts + *e as usize).copied()
                                        == Some(part as u8)
                                })
                                .collect();
                            if !experts.is_empty() {
                                let at = self.layout.store_layer(part, layer);
                                self.store.ahead(at, &experts, d + 1);
                            }
                        }
                    }
                    _ => {
                        let experts: Vec<u32> = list.experts().collect();
                        self.store.ahead(self.layout.store_layer(index, layer), &experts, d + 1);
                    }
                }
            }
        }
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

impl FeedPrune {
    /// Counts what the router of `layer` (store layer `at`, on part `index`)
    /// pruned in the record, and gives the picks it left out or stood an
    /// expert in for their heat: the router wanted them, and the tiers are to
    /// learn that, so that a text's experts come in between steps.
    fn follow(
        &mut self,
        store: &mut ExpertStore<CudaSlots>,
        at: usize,
        index: usize,
        layer: usize,
        record: &RouteRecord<'_>,
        groups: &[RouteGroup],
    ) {
        let n = self.n_experts;
        let owned = |e: u32| match &self.owner {
            None => true,
            Some((mixer, owner)) => (owner[layer * n + e as usize] != 0) == (index != *mixer),
        };
        let mut pruned = 0u64;
        let mut stood_in = 0u64;
        let mut any = false;
        for row in 0..record.rows() {
            let bit = 1u32 << row;
            let ids: Vec<u32> = record.ids(row).collect();
            for id in &ids {
                if !owned(*id) {
                    continue;
                }
                self.stats.picks += 1;
                if !groups.iter().any(|g| g.expert == *id && g.rows & bit != 0) {
                    pruned += 1;
                    self.counts[*id as usize] += 1;
                    any = true;
                }
            }
            stood_in +=
                groups.iter().filter(|g| g.rows & bit != 0 && !ids.contains(&g.expert)).count()
                    as u64;
        }
        self.stats.stood_in += stood_in;
        self.stats.touched += pruned;
        if any {
            store.record_counts(at, &self.counts, 1.0);
            self.counts.fill(0);
        }
    }
}
