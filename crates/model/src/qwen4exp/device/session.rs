//! A model on its devices, driven one step at a time from one host thread:
//! how the tests and the tools step it by hand and look at every layer.
//!
//! The parts open through the stages every opening goes through (`open.rs`)
//! and a step drives them through the runtime's own [`Flow`]: the lead over
//! the route records, the rows of the n-gram table, the rounds of a wide
//! step's missed experts and the waits all serve the expert store as the
//! runtime's device threads do. What the session does itself is what a hand
//! needs: one host thread for every part (by layers one part after the other,
//! the residual handed over in its own lanes; on two devices that share every
//! layer each layer on both, the mixer first), every sequence owning its pages
//! from the start, the tier in RAM locked whole, the experts of two devices
//! divided by a recorded routing when asked ([`SessionOptions::routing`]), a
//! hook that sees every layer ([`Probe`]) and the logits on the host.
//!
//! It is not a face over the runtime's executor, as the 27B's session is: the
//! probe stops both devices at every layer and reads both parts, which the
//! runtime's device threads could only do with a barrier between them inside
//! the production step, and the tests compare the logits, which the runtime
//! samples on the device.
//!
//! The rows of the n-gram table that a token needs depend on the token and
//! the two before it, so they are asked for the moment a step has picked its
//! token ([`Session::expect`] for a caller that knows better), not when the
//! next step reaches the layer that reads them.

use std::ops::Deref;
use std::sync::{Arc, MutexGuard};
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, Lane, MAX_BATCH, Mesh, Parcel, RouteRecord, Segment,
    StepRecord, Stream,
};
use ares_gguf::Gguf;
use ares_io::Reader;
use ares_runtime::PinnedRam;
use ares_runtime::expert_store::{
    CudaSlots, ExpertProfile, ExpertStore, FillStats, HostRegion, StorePlan, StoreStats,
};

use super::Error;
use super::feed::{Feed, Fill, Layout, build_store, cpu_pool, device_slots, places};
use super::flow::{Flow, Routing, Turn, wide_layer};
use super::misses::WideStats;
use super::ngram::NgramFeed;
use super::open::{self, PartReport};
use super::options::ModelOptions;
use super::part::{Part, StepEnv};
use super::placement::{Chosen, Placement, choose, ram_tier_bytes};
use super::prune::{PruneMask, PruneStats};
use super::usage;
use crate::blocks::far::FarPool;
use crate::blocks::rows::{HostTable, TOKEN_EMBD};
use crate::qwen4exp::Config;
use crate::qwen4exp::weights::Weights;

/// The choices of a session: the model's, and how the session itself sets
/// up the tier in RAM and divides the experts of two devices, which the
/// runtime decides by its own rules.
#[derive(Clone, Debug, Default)]
pub struct SessionOptions {
    /// The choices of the model.
    pub model: ModelOptions,
    /// Bytes of page-locked RAM for the experts' tier in host memory, locked
    /// whole; `None` takes half of what the system reports available.
    pub ram_tier: Option<u64>,
    /// How often every expert of every layer was picked in a recorded routing
    /// (`routing[layer][expert]`: a profile that ships with a model, or the
    /// heat of an earlier run). Two devices that share the layers divide the
    /// experts by it ([`Placement::assign`]); without it by a fixed rule
    /// that gives each device the share of the experts its slots are of all.
    ///
    /// A real choice, not an improvement: with a routing the slower device
    /// computes fewer experts and never misses one, but its slots hold what
    /// the recording used, whatever the text at hand uses. Measured on
    /// Flash-Next with a recording that included the text: a step that
    /// misses nothing 15.9 instead of 16.7 ms, and 10.7 instead of 6.5
    /// experts outside their slots per step.
    pub routing: Option<Vec<Vec<u32>>>,
    /// The share of the picks that the worker's experts are to get when the
    /// experts are divided by a routing; `None` takes the worker's share of
    /// both devices' speed, from the read times measured at load.
    pub worker_picks: Option<f64>,
}

/// The rows of one sequence in a step: they follow the sequence's cells.
#[derive(Clone, Copy, Debug)]
pub struct Rows<'a> {
    /// The sequence's slot.
    pub slot: u32,
    /// The tokens of the rows.
    pub tokens: &'a [u32],
    /// Whether every row is certain to be accepted; otherwise the rows are a
    /// draft window, and [`Session::accept`] says how many count.
    pub direct: bool,
    /// Whose logits are wanted.
    pub wanted: Wanted,
}

/// Which rows of a sequence's rows get their logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wanted {
    /// None.
    None,
    /// The last row.
    Last,
    /// Every row.
    All,
}

/// What a step took on the host's clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepTime {
    /// From the start of the step to its logits.
    pub wall: Duration,
    /// Building the record, decoding the embeddings and hashing.
    pub prepare: Duration,
    /// Waiting for the rows of the n-gram table.
    pub rows: Duration,
    /// Enqueuing the kernels (the device runs meanwhile).
    pub enqueue: Duration,
    /// Waiting for the devices after everything was enqueued.
    pub wait: Duration,
}

/// What one layer left on its device, for [`Probe::after`].
#[derive(Debug)]
pub struct LayerOut<'a> {
    /// The residual after the layer, `rows * D` values.
    pub residual: &'a [f32],
    /// The output of the recurrent or attention part, `rows * N` values.
    pub core: &'a [f32],
    /// The output of the mixture of experts, `rows * N` values.
    pub ffn: &'a [f32],
    /// Per row the experts the router picked, in rank order.
    pub picks: Vec<Vec<u32>>,
}

/// A hook that sees every layer of a step: the tests compare each layer
/// with the reference through it. A probed step waits for the device after
/// every layer.
pub trait Probe {
    /// Called before `layer` is enqueued. The residual the layer reads is
    /// replaced by what is returned, when something is.
    fn before(&mut self, layer: usize) -> Option<Vec<f32>>;
    /// Called when `layer` has run.
    fn after(&mut self, layer: usize, out: LayerOut<'_>);
}

/// What every part sees of a step.
struct Step<'a> {
    record: &'a StepRecord,
    segments: &'a [Segment],
    tokens: &'a [u32],
    token: u32,
    xbase: u32,
}

/// The step as the part of `$device` sees it. Built where it is used: between
/// the uses the session follows the route records.
macro_rules! step_env {
    ($self:ident, $step:expr, $device:expr, $fetched:expr, $inbound:expr) => {
        StepEnv {
            stream: &$self.streams[$device],
            record: $step.record,
            segments: $step.segments,
            token: $step.token,
            tokens: $step.tokens,
            ngram: $fetched.as_deref(),
            inbound: $inbound,
            xbase: $step.xbase,
            pools: $self.pools[$device].ptr(),
            replay: &[],
        }
    };
}

/// The experts each of `rows` rows picked, from a route record: that of the decode kernels
/// for up to [`MAX_BATCH`] rows, else the ids behind the header.
fn picks_of(record: &[u8], used: usize, rows: u32) -> Result<Vec<Vec<u32>>, Error> {
    if rows > MAX_BATCH {
        let ids = record
            .get(16..16 + 4 * used * rows as usize)
            .ok_or_else(|| Error::invalid("a route record that is none"))?;
        return Ok(ids
            .as_chunks::<4>()
            .0
            .chunks(used)
            .map(|row| row.iter().map(|id| u32::from_le_bytes(*id)).collect())
            .collect());
    }
    let route = RouteRecord::parse(record, used as u32, rows)
        .ok_or_else(|| Error::invalid("a route record that is none"))?;
    Ok((0..rows as usize).map(|row| route.ids(row).collect()).collect())
}

/// How long the upload of `bytes` bytes from page-locked memory to `device`
/// takes: what an expert that is in host memory costs when a step misses it.
fn upload_time(device: &Device, bytes: u64) -> Result<Duration, Error> {
    let len = bytes.max(4096) as usize;
    let host = ares_cuda::PinnedBuffer::new(device, len)?;
    let target = DeviceBuffer::new(device, len as u64)?;
    let stream = Stream::new(device, 0)?;
    let run = |count: u32| -> Result<(), Error> {
        for _ in 0..count {
            // SAFETY: both buffers hold `len` bytes and outlive the copies,
            // which have run when `synchronize` returns.
            unsafe { stream.copy_from_host(target.ptr(), host.as_ptr(), len)? };
        }
        stream.synchronize()?;
        Ok(())
    };
    run(2)?;
    let started = Instant::now();
    run(8)?;
    Ok(started.elapsed() / 8)
}

/// The expert store of a session, held while it is looked at: no step runs
/// meanwhile.
pub struct StoreRef<'a>(MutexGuard<'a, Feed>);

impl std::fmt::Debug for StoreRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreRef").finish_non_exhaustive()
    }
}

impl Deref for StoreRef<'_> {
    type Target = ExpertStore<CudaSlots>;

    fn deref(&self) -> &Self::Target {
        &self.0.store
    }
}

/// A model on its devices, stepped by hand.
pub struct Session<'g> {
    c: Config,
    /// The weights the session was opened from, which it does not outlive.
    _weights: &'g Weights<'g>,
    opts: SessionOptions,
    placement: Placement,
    devices: Vec<Device>,
    streams: Vec<Stream>,
    done: Vec<Event>,
    probe_done: Vec<Event>,
    parts: Vec<Part>,
    /// Per part: who routes for it and which experts it computes.
    routing: Vec<Routing>,
    /// The pools of K/V and keys of every part, and the K and V of the pages
    /// beyond VRAM.
    pools: Vec<DeviceBuffer>,
    _far: Vec<FarPool>,
    ledgers: Vec<Ledger>,
    flow: Flow,
    lanes: Vec<Lane>,
    _mesh: Option<Mesh>,
    step: u32,
    picks: Vec<u32>,
    n_out: usize,
    last: StepTime,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("placement", &self.placement).finish_non_exhaustive()
    }
}

impl<'g> Session<'g> {
    /// Opens the model of `gguf` on `devices`: places the layers, uploads
    /// the dense weights, and builds the expert store with empty tiers
    /// ([`Session::fill`] fills them).
    pub fn open(
        gguf: &'g Gguf,
        c: &Config,
        w: &'g Weights<'g>,
        devices: Vec<Device>,
        opts: SessionOptions,
    ) -> Result<Self, Error> {
        if devices.is_empty() {
            return Err(Error::invalid("a session needs a device"));
        }
        let model = &opts.model;
        let Chosen { mut placement, speeds, memory } = choose(gguf, c, w, &devices, model)?;
        let mut ledgers: Vec<Ledger> = devices
            .iter()
            .zip(&memory)
            .map(|(device, m)| Ledger::new(format!("CUDA{}", device.ordinal()), m.budget))
            .collect();
        // The embedding table where the file is mapped: the part that embeds
        // stages a step's rows from it.
        let mapped = Gguf::open(gguf.path(0)).map_err(|e| Error::invalid(e.to_string()))?;
        let table = Arc::new(HostTable::mapped(Arc::new(mapped), TOKEN_EMBD)?);
        let mut parts = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            let ledger = &mut ledgers[index];
            parts.push(open::open_part(
                c,
                w,
                &model.part,
                &placement,
                index,
                device,
                ledger,
                &table,
            )?);
        }

        // The pools of K/V and keys, whose pages every sequence slot owns, and
        // the K and V of the pages beyond VRAM, locked whole: a session has no
        // RAM limit to keep.
        let mut pools = Vec::with_capacity(parts.len());
        let mut far = Vec::new();
        let ram = PinnedRam::new(u64::MAX / 4);
        for (index, part) in parts.iter_mut().enumerate() {
            ledgers[index].reserve("qwen4exp K/V and key pools", part.pool_bytes())?;
            pools.push(DeviceBuffer::new(&devices[index], part.pool_bytes())?);
            if let Some(mut pool) = part.open_far(&ram)? {
                pool.lock_all().map_err(|short| match short {
                    Ok(short) => {
                        Error::invalid(format!("{} bytes of K and V beyond VRAM", short.0))
                    }
                    Err(error) => Error::invalid(error.to_string()),
                })?;
                far.push(pool);
            }
            part.own_pages()?;
        }

        // The expert store: slots from what the placement left, a tier in
        // page-locked RAM, reads from the file. When the devices share the
        // layers, the store sees every layer once per device: the experts of
        // the layer that the device owns.
        let (layout, map) = Layout::new(gguf, c, w, &placement, None)?;
        let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
        let vram: Vec<u64> = placement
            .devices
            .iter()
            .map(|d| d.slots.min(model.slot_bytes.unwrap_or(u64::MAX)))
            .collect();
        let ram_bytes = ram_tier_bytes(opts.ram_tier, layout.ram_wanted(vram.iter().sum()));
        let room: Vec<u64> =
            ledgers.iter().map(|l| open::slot_room(l.free(), &layout, c, model)).collect();
        let config = open::store_config(c, model, &layout, &reader, vram, ram_bytes, &room);
        let reserve = config.vram_reserve;
        let plan = StorePlan::new(config, &map, &reader, &layout.specs, None)?;
        if let Some((times, mixer)) = speeds
            && let Some(routing) = opts.routing.as_ref().filter(|r| r.len() == c.n_layer)
        {
            // With a recorded routing the experts are divided by it: each
            // device gets the share of the picks that its speed is of both,
            // and what a step can miss is owned where an upload is faster.
            let slots: Vec<Vec<usize>> = (0..layout.per_layer)
                .map(|device| {
                    (0..c.n_layer)
                        .map(|layer| {
                            plan.slots(layout.store_layer(device, layer)).0.saturating_sub(reserve)
                        })
                        .collect()
                })
                .collect();
            let worker = 1 - mixer;
            let worker_picks = opts
                .worker_picks
                .unwrap_or(times[mixer] / (times[0] + times[1]).max(1e-9))
                .clamp(0.0, 1.0);
            let mut picks = [1.0 - worker_picks; 2];
            picks[worker] = worker_picks;
            let expert = layout.shapes[0].slot_bytes().unwrap_or(1 << 20);
            let uploads = [upload_time(&devices[0], expert)?, upload_time(&devices[1], expert)?];
            placement.assign(routing, &slots, picks, usize::from(uploads[1] < uploads[0]));
        }
        open::connect_all(&mut parts, &placement, c)?;
        let mut pinned = Ledger::new("pinned RAM", ram_bytes + (1 << 30));
        let mut slots = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            slots.push(device_slots(
                device,
                &devices,
                &mut ledgers[index],
                &mut pinned,
                &plan.device_pools(index),
                &layout,
                model.serve,
                model.part.max_rows.min(MAX_BATCH),
            )?);
        }
        let region = HostRegion::pinned(&devices[0], &mut pinned, plan.ram_bytes())?;
        // One core stays with the thread that drives the step.
        let cpu = (model.serve != ares_runtime::expert_store::Serve::Stream)
            .then(|| cpu_pool(model.cpu_workers, 1).map(|pool| (pool, model.serve)))
            .transpose()?;
        let store = build_store(plan, &layout, map, reader, slots, region, cpu)?;
        for (index, part) in parts.iter_mut().enumerate() {
            let layers = part.layer_range();
            part.set_experts(places(&store, layers, &layout, index))?;
        }
        if model.part.prune.is_some() {
            pinned
                .reserve("qwen4exp pruning mask", PruneMask::bytes(c.n_layer, c.experts.count))?;
        }
        let boards = parts.iter().map(Part::board).collect();
        let (feed, mask) =
            open::open_feed(store, boards, &placement, &layout, c, model, &devices[0])?;
        if let Some(mask) = mask {
            for part in &mut parts {
                part.set_prune_mask(mask.clone());
            }
        }
        let io_cpus = feed.store.io_cpus().to_vec();
        let ngram = NgramFeed::open(gguf, c, model.row_cache, model.part.max_slots, &io_cpus)?;
        let used = c.experts.used as u32;
        let own = model.part.own_rows();
        let flow = Flow::new(feed, ngram, parts.len(), model.stall, own, used, ram);

        let wide = c.wide() as u32;
        let (mesh, lanes) = if devices.len() > 1 && placement.tensor.is_none() {
            let values = model.part.max_rows * wide;
            let mesh = Mesh::new(&devices, values)?;
            let lanes = (1..devices.len())
                .map(|to| Lane::new(&mesh, to - 1, to, 2, values))
                .collect::<Result<_, _>>()?;
            (Some(mesh), lanes)
        } else {
            (None, Vec::new())
        };
        let events = || -> Result<Vec<Event>, Error> {
            devices.iter().map(|d| Event::new(d).map_err(Error::from)).collect()
        };
        let routing = (0..parts.len()).map(|index| Routing::of(&placement, c, index)).collect();
        Ok(Self {
            c: c.clone(),
            _weights: w,
            streams: devices.iter().map(|d| Stream::new(d, 0)).collect::<Result<_, _>>()?,
            done: events()?,
            probe_done: events()?,
            opts,
            placement,
            devices,
            parts,
            routing,
            pools,
            _far: far,
            ledgers,
            flow,
            lanes,
            _mesh: mesh,
            step: 0,
            picks: Vec::new(),
            n_out: 0,
            last: StepTime::default(),
        })
    }

    /// The placement of the model.
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The parts, in the order of the layers.
    pub fn parts(&self) -> &[Part] {
        &self.parts
    }

    /// The expert store. No step runs while it is held.
    pub fn store(&self) -> StoreRef<'_> {
        StoreRef(self.flow.lock_feed())
    }

    /// What wide steps found missing and waited for since the last call.
    pub fn take_wide_stats(&mut self) -> WideStats {
        self.flow.lock_feed().take_wide()
    }

    /// What the store counted since the last call.
    pub fn take_store_stats(&mut self) -> StoreStats {
        self.flow.lock_feed().store.take_stats()
    }

    /// What the routers pruned since the last call (the `prune` of the
    /// parts' options).
    pub fn take_prune_stats(&mut self) -> PruneStats {
        self.flow.lock_feed().take_prune()
    }

    /// Whether the steps from the next on prune, for a session opened with
    /// pruning (they do until told not to): a measurement reads the prompt the
    /// same way for every rule.
    pub fn set_pruning(&mut self, on: bool) {
        self.flow.lock_feed().set_pruning(on);
    }

    /// The memory of every part.
    pub fn report(&self) -> Vec<PartReport> {
        let feed = self.flow.lock_feed();
        let layout = feed.layout();
        self.parts
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let n_slots = part
                    .layer_range()
                    .map(|l| feed.store.slots(layout.store_layer(index, l)).0)
                    .sum();
                let slots = self.placement.devices[index].slots;
                PartReport::of(part, index, slots, n_slots, self.ledgers[index].reserved())
            })
            .collect()
    }

    /// The slots of the experts' tier in host memory, over all layers.
    pub fn ram_slots(&self) -> usize {
        let per_layer = if self.placement.tensor.is_some() { self.parts.len() } else { 1 };
        let feed = self.flow.lock_feed();
        (0..self.c.n_layer * per_layer).map(|at| feed.store.slots(at).1).sum()
    }

    /// The cell the next row of the sequence in `slot` takes.
    pub fn position(&self, slot: u32) -> u32 {
        self.flow.lock_ngram().position(slot)
    }

    /// Starts the sequence in `slot` over: its next row is cell 0.
    pub fn restart(&mut self, slot: u32) -> Result<(), Error> {
        self.flow.lock_ngram().restart(slot)
    }

    /// Seeds the store's heat with a usage `profile` of the model and fills
    /// the tiers from it in one sweep through the file, as the runtime does
    /// when it begins to serve: VRAM with the hottest experts of every layer,
    /// the RAM tier with the next. Returns what was brought and how long it
    /// took.
    pub fn fill_from_profile(
        &mut self,
        profile: &ExpertProfile,
    ) -> Result<(FillStats, Duration), Error> {
        let mut feed = self.flow.lock_feed();
        let layout = feed.layout().clone();
        Ok(usage::seed_and_fill(&mut feed.store, &layout, &self.placement, profile)?)
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers, most wanted first, as far as the slots go: from the file into
    /// VRAM, and into RAM what VRAM has no room for. Returns when nothing
    /// moves any more.
    pub fn fill(&mut self, experts_of: &mut dyn FnMut(usize) -> Vec<u32>) -> Result<Fill, Error> {
        let placement = &self.placement;
        self.flow.lock_feed().fill(
            self.c.n_layer,
            self.c.experts.count,
            &|layer, expert| placement.owner(layer, expert),
            experts_of,
        )
    }

    /// Runs one step with the rows of `rows` and returns the most likely
    /// token of every wanted row, in order. The draft windows of the step
    /// before must have been settled with [`Session::accept`].
    pub fn step(&mut self, rows: &[Rows<'_>]) -> Result<&[u32], Error> {
        self.run(rows, None)?;
        Ok(&self.picks)
    }

    /// [`Session::step`] with a hook that sees every layer. Slow: the host
    /// waits for the device after every layer.
    pub fn step_probed(
        &mut self,
        rows: &[Rows<'_>],
        probe: &mut dyn Probe,
    ) -> Result<&[u32], Error> {
        self.run(rows, Some(probe))?;
        Ok(&self.picks)
    }

    fn run(&mut self, rows: &[Rows<'_>], probe: Option<&mut dyn Probe>) -> Result<(), Error> {
        if self.flow.lock_ngram().has_windows() {
            return Err(Error::invalid("the draft windows of the last step were not settled"));
        }
        let started = Instant::now();
        // The record and the rows of the n-gram table; the part that embeds
        // stages the rows of the embedding table itself.
        let mut segments = Vec::with_capacity(rows.len());
        let mut tokens = Vec::new();
        let mut out_rows = Vec::new();
        for part in rows {
            if part.slot >= self.opts.model.part.max_slots {
                return Err(Error::invalid(format!("slot {} of the session", part.slot)));
            }
            if part.tokens.is_empty() {
                return Err(Error::invalid("a sequence without rows in a step"));
            }
            if segments.iter().any(|s: &Segment| s.slot == part.slot) {
                return Err(Error::invalid(format!("slot {} twice in a step", part.slot)));
            }
            let row0 = tokens.len() as u32;
            let count = part.tokens.len() as u32;
            segments.push(Segment {
                row0,
                n_rows: count,
                slot: part.slot,
                pos0: self.position(part.slot) as i32,
                direct: part.direct,
            });
            match part.wanted {
                Wanted::None => {}
                Wanted::Last => out_rows.push(row0 + count - 1),
                Wanted::All => out_rows.extend(row0..row0 + count),
            }
            tokens.extend_from_slice(part.tokens);
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= self.c.n_vocab) {
            return Err(Error::invalid(format!(
                "token {token} of a vocabulary of {}",
                self.c.n_vocab
            )));
        }
        let record = StepRecord::new(&segments, &tokens, None, &out_rows)?;
        let dims = record.dims();
        // The rows of the table: those asked for when the tokens were first
        // known, when these are the tokens; otherwise they are read now.
        let asked: Vec<(u32, &[u32])> = rows.iter().map(|p| (p.slot, p.tokens)).collect();
        self.flow.lock_ngram().ask(&asked)?;
        let prepared = Instant::now();
        let mut waited_rows = Duration::ZERO;

        // The tier of experts takes a piece more when it may hold more, before
        // the step, as the runtime's conductor does: page-locked memory does
        // not change inside a step, which runs from here to the end of this
        // call (`ares_cuda::gate`).
        self.flow.lock_feed().store.grow_ram(1).map_err(|e| Error::invalid(e.to_string()))?;
        let _steps = ares_cuda::gate::Steps::begin();

        // The step begins for the store, as the runtime's does: its windows
        // verify drafts (no row is worth less than its expert here).
        self.step += 1;
        let windows: Vec<(u32, u32)> =
            segments.iter().filter(|s| !s.direct).map(|s| (s.row0, s.n_rows)).collect();
        for (index, stream) in self.streams.iter().enumerate() {
            let windows = Some((windows.as_slice(), None));
            self.flow.begin(u64::from(self.step), dims.n_rows, windows, index, stream, None)?;
        }
        // The exchanges between a mixer and its worker are numbered through
        // the steps: eight per layer.
        let per_step = 8 * (self.c.n_layer as u32 + 1);
        let step = Step {
            record: &record,
            segments: &segments,
            tokens: &tokens,
            token: self.step,
            xbase: self.step.wrapping_mul(per_step).wrapping_add(1),
        };
        let n_out = match self.placement.tensor.as_ref().map(|tensor| tensor.mixer) {
            Some(mixer) => self.enqueue_shared(&step, mixer, probe, &mut waited_rows)?,
            None => self.enqueue_by_layers(&step, probe, &mut waited_rows)?,
        };
        let enqueued = Instant::now();
        // The device with the head ends the step with the logits.
        let head = self.parts.iter().position(|part| part.has_head());
        if let Some(head) = head {
            self.flow.wait_for(&self.done[head], "a device ran the step")?;
        }

        // The picks, and where the sequences are now.
        self.n_out = n_out as usize;
        self.picks.clear();
        if let Some(head) = head {
            let n_vocab = self.c.n_vocab;
            for row in self.parts[head].logits(self.n_out).chunks_exact(n_vocab) {
                let best = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| {
                    if *v > b.1 { (i, *v) } else { b }
                });
                self.picks.push(best.0 as u32);
            }
        }
        {
            let mut ngram = self.flow.lock_ngram();
            for part in rows {
                if part.direct {
                    ngram.advance(part.slot, part.tokens);
                } else {
                    ngram.hold(part.slot, part.tokens);
                }
            }
        }
        // Every sequence most likely goes on with the token it just picked:
        // the rows of the table for that step are asked for now, before the
        // step's own bookkeeping.
        if rows.len() == self.picks.len()
            && rows.iter().all(|part| part.direct && part.wanted == Wanted::Last)
        {
            let next: Vec<(u32, [u32; 1])> =
                rows.iter().zip(&self.picks).map(|(part, pick)| (part.slot, [*pick])).collect();
            let next: Vec<(u32, &[u32])> =
                next.iter().map(|(slot, token)| (*slot, token.as_slice())).collect();
            self.expect(&next)?;
        }

        for (index, done) in self.done.iter().enumerate() {
            self.flow.wait_for(done, "a device ran the step")?;
            self.flow.follow(index)?;
        }
        for part in &mut self.parts {
            self.flow.finish(true)?;
            part.flush_profile()?;
        }
        let ended = Instant::now();
        self.last = StepTime {
            wall: ended - started,
            prepare: prepared - started,
            rows: waited_rows,
            enqueue: (enqueued - prepared).saturating_sub(waited_rows),
            wait: ended - enqueued,
        };
        Ok(())
    }

    /// Waits for the rows of the n-gram table of the step the first time a
    /// layer needs them, and for how long into `waited`.
    fn fetch_rows(
        flow: &Flow,
        fetched: &mut Option<Vec<u8>>,
        waited: &mut Duration,
    ) -> Result<(), Error> {
        if fetched.is_none() {
            let waiting = Instant::now();
            *fetched = Some(flow.await_rows()?);
            *waited = waiting.elapsed();
        }
        Ok(())
    }

    /// The step by layers: one part after the other, the residual handed
    /// from device to device.
    fn enqueue_by_layers(
        &mut self,
        step: &Step<'_>,
        mut probe: Option<&mut dyn Probe>,
        waited_rows: &mut Duration,
    ) -> Result<u32, Error> {
        let dims = step.record.dims();
        let mut fetched: Option<Vec<u8>> = None;
        let mut inbound: Option<DevicePtr> = None;
        let mut parcel: Option<Parcel> = None;
        let mut n_out = 0;
        for device in 0..self.parts.len() {
            if let Some(parcel) = &parcel {
                // SAFETY: the parcel is read on this stream between here and
                // its release below, by the copy `Part::begin` enqueues.
                inbound =
                    Some(unsafe { self.lanes[device - 1].receive(&self.streams[device], parcel)? });
            }
            self.parts[device].begin(&step_env!(self, step, device, fetched, inbound))?;
            if let Some(parcel) = parcel.take() {
                self.lanes[device - 1].release(&self.streams[device], parcel)?;
            }
            inbound = None;
            let first_layer = self.parts[device].layer_range().start;
            for i in 0..self.parts[device].n_layers() {
                if self.parts[device].ngram_layer() == Some(i) {
                    Self::fetch_rows(&self.flow, &mut fetched, waited_rows)?;
                }
                if let Some(probe) = probe.as_deref_mut() {
                    self.settle(step, &fetched, &[device])?;
                    if let Some(r) = probe.before(first_layer + i) {
                        self.parts[device].write_residual(&r)?;
                    }
                }
                let env = step_env!(self, step, device, fetched, inbound);
                let part = &mut self.parts[device];
                self.flow.enqueue_layer(part, device, &env, i)?;
                if dims.n_rows > MAX_BATCH {
                    let routing = &self.routing[device];
                    let turn = Turn { part, index: device, env: &env, routing, pass: false };
                    wide_layer(&self.flow, i, &mut [turn])?;
                }
                self.flow.pump()?;
                if let Some(probe) = probe.as_deref_mut() {
                    self.settle(step, &fetched, &[device])?;
                    self.flow.follow(device)?;
                    let part = &self.parts[device];
                    let rows = dims.n_rows as usize;
                    let out = LayerOut {
                        residual: &part.read_residual(rows)?,
                        core: &part.read_core_out(rows)?,
                        ffn: &part.read_ffn_out(rows)?,
                        picks: picks_of(part.route_record(i), self.c.experts.used, dims.n_rows)?,
                    };
                    probe.after(first_layer + i, out);
                }
            }
            n_out = self.parts[device].end(&step_env!(self, step, device, fetched, inbound))?;
            if device + 1 < self.parts.len() {
                let values = dims.n_rows * self.c.wide() as u32;
                // SAFETY: the residual of the part holds the step's rows and
                // is not written again before the next step.
                parcel = Some(unsafe {
                    self.lanes[device].send_f32(
                        &self.streams[device],
                        self.parts[device].residual(),
                        values,
                    )?
                });
            }
            self.done[device].record(&self.streams[device])?;
        }
        Ok(n_out)
    }

    /// Brings the parts of `devices` to rest for a probe: each enqueues what
    /// it has deferred (a worker sends what it has, a mixer writes it to the
    /// streams), and the host waits until every one has run it.
    fn settle(
        &mut self,
        step: &Step<'_>,
        fetched: &Option<Vec<u8>>,
        devices: &[usize],
    ) -> Result<(), Error> {
        for &device in devices {
            self.parts[device].flush(&step_env!(self, step, device, fetched, None))?;
            self.probe_done[device].record(&self.streams[device])?;
        }
        for &device in devices {
            self.flow.wait_for(&self.probe_done[device], "a device ran a probed layer")?;
        }
        Ok(())
    }

    /// The step of two devices that share every layer: each layer is
    /// enqueued on the mixer and on the worker, which exchange what they
    /// need between themselves.
    fn enqueue_shared(
        &mut self,
        step: &Step<'_>,
        mixer: usize,
        mut probe: Option<&mut dyn Probe>,
        waited_rows: &mut Duration,
    ) -> Result<u32, Error> {
        let worker = 1 - mixer;
        let dims = step.record.dims();
        let rows = dims.n_rows as usize;
        let mut fetched: Option<Vec<u8>> = None;
        for device in [mixer, worker] {
            self.parts[device].begin(&step_env!(self, step, device, fetched, None))?;
        }
        // In front of and behind a probed layer both devices come to rest.
        let both = [worker, mixer];
        for i in 0..self.c.n_layer {
            if self.parts[mixer].ngram_layer() == Some(i) {
                Self::fetch_rows(&self.flow, &mut fetched, waited_rows)?;
            }
            if let Some(probe) = probe.as_deref_mut() {
                self.settle(step, &fetched, &both)?;
                if let Some(r) = probe.before(i) {
                    self.parts[mixer].write_residual(&r)?;
                }
            }
            let (m_env, w_env) = (
                step_env!(self, step, mixer, fetched, None),
                step_env!(self, step, worker, fetched, None),
            );
            let (first, second) = self.parts.split_at_mut(1);
            let (m_part, w_part) = if mixer == 0 {
                (&mut first[0], &mut second[0])
            } else {
                (&mut second[0], &mut first[0])
            };
            self.flow.enqueue_layer(m_part, mixer, &m_env, i)?;
            self.flow.enqueue_layer(w_part, worker, &w_env, i)?;
            if dims.n_rows > MAX_BATCH {
                let (m_routing, w_routing) = (&self.routing[mixer], &self.routing[worker]);
                let mut turns = [
                    Turn {
                        part: m_part,
                        index: mixer,
                        env: &m_env,
                        routing: m_routing,
                        pass: false,
                    },
                    Turn {
                        part: w_part,
                        index: worker,
                        env: &w_env,
                        routing: w_routing,
                        pass: false,
                    },
                ];
                wide_layer(&self.flow, i, &mut turns)?;
            }
            self.flow.pump()?;
            if let Some(probe) = probe.as_deref_mut() {
                self.settle(step, &fetched, &both)?;
                self.flow.follow(mixer)?;
                self.flow.follow(worker)?;
                let (m, w) = (&self.parts[mixer], &self.parts[worker]);
                let sum = |mut a: Vec<f32>, b: Vec<f32>| {
                    a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                    a
                };
                // Each half is the sum of the two devices' shares; the
                // worker's share of the experts is what it sent last.
                let none = || vec![0.0f32; rows * self.c.n_embd];
                let core = sum(
                    if m.has_core(i) { m.read_core_out(rows)? } else { none() },
                    if w.has_core(i) { w.read_core_out(rows)? } else { none() },
                );
                let ffn = sum(m.read_ffn_out(rows)?, m.read_peer_partial(rows)?);
                // Both records keep every row's picks.
                let out = LayerOut {
                    residual: &m.read_residual(rows)?,
                    core: &core,
                    ffn: &ffn,
                    picks: picks_of(m.route_record(i), self.c.experts.used, dims.n_rows)?,
                };
                probe.after(i, out);
            }
        }
        self.parts[worker].end(&step_env!(self, step, worker, fetched, None))?;
        let n_out = self.parts[mixer].end(&step_env!(self, step, mixer, fetched, None))?;
        for device in [mixer, worker] {
            self.done[device].record(&self.streams[device])?;
        }
        Ok(n_out)
    }

    /// Says what the next step will most likely be: for each sequence its
    /// slot and its next tokens, in the order the step will have them. The
    /// rows of the n-gram table those tokens need are read now, so the step
    /// finds them there; a step with other tokens reads its own. A step
    /// whose every sequence asks for the pick of its one row does this itself
    /// with the picks; a caller that samples otherwise, or drafts, says so
    /// here once it knows.
    pub fn expect(&mut self, next: &[(u32, &[u32])]) -> Result<(), Error> {
        self.flow.lock_ngram().expect(next)
    }

    /// Settles the draft windows of the last step: `accepted[i]` rows of the
    /// `i`-th of them count. The commit is enqueued at once and runs before
    /// the next step.
    pub fn accept(&mut self, accepted: &[u32]) -> Result<(), Error> {
        let windows = {
            let mut ngram = self.flow.lock_ngram();
            let windows = ngram.has_windows();
            ngram.settle(accepted)?;
            windows
        };
        if windows {
            for (part, stream) in self.parts.iter_mut().zip(&self.streams) {
                part.enqueue_commit(stream, accepted)?;
            }
        }
        Ok(())
    }

    /// The logits of the `i`-th wanted row of the last step.
    pub fn logits(&self, i: usize) -> Option<&[f32]> {
        let n_vocab = self.c.n_vocab;
        let all = self.parts.iter().find(|part| part.has_head())?.logits(self.n_out);
        all.get(i * n_vocab..(i + 1) * n_vocab)
    }

    /// What the last step took.
    pub fn last_time(&self) -> StepTime {
        self.last
    }

    /// The devices of the session, in the order of the layers.
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// The choices the session was opened with.
    pub fn options(&self) -> &SessionOptions {
        &self.opts
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // Nothing of the session may be freed under a kernel that still
        // runs, and nothing may synchronise with a device while a kernel
        // there waits for the host: let the store serve what a broken step
        // left waiting, for a while, before the buffers go.
        let idle = |streams: &[Stream]| streams.iter().all(|s| s.is_idle().unwrap_or(true));
        self.flow.drain(|| idle(&self.streams));
        if idle(&self.streams) {
            for stream in &self.streams {
                let _ = stream.synchronize();
            }
        }
    }
}
