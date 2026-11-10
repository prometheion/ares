//! A model on its devices, driven one step at a time from one host thread.
//!
//! The session owns what a step needs besides the parts: the expert store
//! with its tiers, the reader of the n-gram table, the hand-over between
//! devices, and the least bookkeeping a few sequences need (every sequence
//! owns its pages from the start). A step is enqueued layer by layer; between
//! layers and while the devices run, the session follows the route records
//! that arrive on the parts' side streams and lets the store bring the
//! experts a layer picked that are in no slot, for which the layer's kernels
//! wait on the device. The host itself waits for the device in one place per
//! step: its end.

use std::cell::OnceCell;
use std::time::{Duration, Instant};

use ares_core::Ledger;
use ares_cuda::{
    Device, DevicePtr, Event, Lane, Mesh, Parcel, RouteGroup, RouteRecord, Segment, StepRecord,
    Stream,
};
use ares_gguf::Gguf;
use ares_io::{Class, Completion, ExpertMap, Reader, RowGather, RowReader, RowTable};
use ares_runtime::expert_store::cpu::{CpuPool, Placement as CpuPlacement, Topology};
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, LayerSpec, Serve, StoreConfig, StorePlan, StoreStats,
};

use super::Error;
use super::part::{ExpertAt, Options, Part, PartSetup, StepEnv};
use super::placement::{DeviceMemory, Placement, ram_tier_bytes};
use crate::qwen4exp::Config;
use crate::qwen4exp::reference::pieces;
use crate::qwen4exp::weights::Weights;

/// How many layers the host enqueues ahead of the last layer whose route
/// record it has handled.
///
/// A step is not enqueued blind to its end: a layer that misses an expert
/// makes the device wait for the host, and the driver's queue of commands for
/// a device is finite. A host that goes on enqueuing the rest of the step (a
/// layer is about 30 launches, a step of 48 layers 1 300) fills that queue,
/// blocks inside a launch, and can then no longer bring the expert the device
/// waits for. Seen as a step that never ends, on one device from the first
/// prompt step on, on two devices with a few hundred timed events more per
/// step. The host enqueues much faster than a device computes, so staying
/// this far ahead never leaves a device without work.
const LEAD_LAYERS: usize = 6;

/// The most page-locked RAM the experts' tier takes whatever the system has.
const RAM_TIER_LIMIT: u64 = 8 << 30;

/// The choices of a session.
#[derive(Clone, Debug)]
pub struct SessionOptions {
    /// The choices of every part.
    pub part: Options,
    /// Bytes of every device that stay free.
    pub headroom: u64,
    /// Bytes of page-locked RAM for the experts' tier in host memory; `None`
    /// takes a third of what the system reports available, at most 8 GiB.
    pub ram_tier: Option<u64>,
    /// The first layer of every device and the number of layers, to place
    /// the layers by hand; `None` lets [`Placement::plan`] decide.
    pub bounds: Option<Vec<usize>>,
    /// Free slots the store keeps per layer in VRAM; `None` for its default.
    pub reserve: Option<usize>,
    /// The most bytes of expert slots on a device, whatever it has left;
    /// `None` for all it has left.
    pub slot_bytes: Option<u64>,
    /// How an expert that is in no slot is served: the engine lets the
    /// store's measured costs decide between an upload and the CPU.
    pub serve: Serve,
    /// Workers that compute experts on the CPU; `None` takes the cores the
    /// system reports, less one for the thread that drives the step.
    pub cpu_workers: Option<usize>,
    /// Bytes of the cache of n-gram table rows.
    pub row_cache: usize,
    /// How long a step may stand still before the session gives up on it
    /// with an error that says what it waited for: a device that waits for
    /// an expert nobody brings would otherwise wait for ever.
    pub stall: Duration,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            part: Options::default(),
            headroom: 256 << 20,
            ram_tier: None,
            bounds: None,
            reserve: None,
            slot_bytes: None,
            serve: Serve::Measured,
            cpu_workers: None,
            row_cache: 16 << 20,
            stall: Duration::from_secs(30),
        }
    }
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

/// What filling the tiers took.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fill {
    /// Experts uploaded to VRAM slots.
    pub uploads: u64,
    /// Bytes read from the file.
    pub read_bytes: u64,
    /// How long it took.
    pub time: Duration,
}

/// The memory of one part, for reports.
#[derive(Clone, Debug)]
pub struct PartReport {
    /// The device's name.
    pub device: String,
    /// The layers on it.
    pub layers: std::ops::Range<usize>,
    /// Bytes of dense weights.
    pub weights: u64,
    /// Bytes of expert slots, and how many slots that is.
    pub slots: u64,
    pub n_slots: usize,
    /// Bytes of recurrent state per sequence.
    pub state_per_slot: u64,
    /// Bytes of K, V and pooled keys per 1000 cells, and of the keys alone.
    pub kv_per_1000: u64,
    pub keys_per_1000: u64,
    /// Bytes of scratch, route records and step inputs.
    pub scratch: u64,
    /// Everything the session reserved on the device.
    pub reserved: u64,
    /// How long the upload of the dense weights took.
    pub upload: Duration,
}

struct Sequence {
    position: u32,
    /// The last tokens of the sequence, as many as the n-gram hash reads.
    recent: Vec<u32>,
}

struct Window {
    slot: u32,
    tokens: Vec<u32>,
}

/// A model on its devices, stepped by hand.
pub struct Session<'g> {
    c: Config,
    w: &'g Weights<'g>,
    opts: SessionOptions,
    placement: Placement,
    devices: Vec<Device>,
    streams: Vec<Stream>,
    done: Vec<Event>,
    probe_done: Vec<Event>,
    parts: Vec<Part>,
    ledgers: Vec<Ledger>,
    store: ExpertStore<CudaSlots>,
    lanes: Vec<Lane>,
    _mesh: Option<Mesh>,
    rows: Option<(Reader, RowReader)>,
    gather: RowGather,
    completions: Vec<Completion>,
    sequences: Vec<Sequence>,
    /// The draft windows of the last step.
    windows: Vec<Window>,
    step: u32,
    step_rows: u32,
    /// Layers of the running step that were enqueued, and whose route
    /// records were handed to the store, both counted over all devices.
    enqueued: usize,
    followed: usize,
    groups: Vec<RouteGroup>,
    picks: Vec<u32>,
    n_out: usize,
    last: StepTime,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("placement", &self.placement).finish_non_exhaustive()
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
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
            return Err(invalid("a session needs a device"));
        }
        let mut memory = Vec::with_capacity(devices.len());
        for device in &devices {
            let free = device.mem_info()?.free;
            memory.push(DeviceMemory {
                budget: free.saturating_sub(opts.headroom),
                multiprocessors: device.info().multiprocessors,
            });
        }
        let placement = match &opts.bounds {
            Some(bounds) => Placement::at(c, w, &opts.part, &memory, bounds)?,
            None => Placement::plan(c, w, &opts.part, &memory)?,
        };

        let mut ledgers: Vec<Ledger> = devices
            .iter()
            .zip(&memory)
            .map(|(device, m)| Ledger::new(format!("CUDA{}", device.ordinal()), m.budget))
            .collect();
        let mut parts = Vec::with_capacity(devices.len());
        for (index, (device, plan)) in devices.iter().zip(&placement.devices).enumerate() {
            parts.push(Part::open(
                c,
                w,
                &opts.part,
                PartSetup {
                    index,
                    layers: plan.layers.clone(),
                    head: plan.head,
                    device,
                    ledger: &mut ledgers[index],
                },
            )?);
        }

        // The expert store: slots from what the placement left, a tier in
        // page-locked RAM, reads from the file.
        let map = ExpertMap::discover(gguf, &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"])?;
        let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
        let mut specs = Vec::with_capacity(c.n_layer);
        let mut shapes = Vec::with_capacity(c.n_layer);
        let mut expert_bytes = 0u64;
        for layer in 0..c.n_layer {
            let moe = &w.layers[layer].moe;
            let shape = ares_cuda::ExpertShape {
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
                device: placement
                    .device_of(layer)
                    .ok_or_else(|| invalid(format!("layer {layer} is on no device")))?,
                offsets: [
                    0,
                    shape.up_offset().ok_or_else(geometry)?,
                    shape.down_offset().ok_or_else(geometry)?,
                ],
                slot_bytes,
            });
            expert_bytes += slot_bytes * c.experts.count as u64;
            shapes.push(shape);
        }
        // No more page-locked RAM than a tier that holds every expert takes
        // (a slot there is the expert and a few blocks of the file).
        let ram_limit = RAM_TIER_LIMIT.min((2 * expert_bytes).max(64 << 20));
        let ram_bytes = ram_tier_bytes(opts.ram_tier, ram_limit);
        let mut config = StoreConfig::new(
            opts.part.max_rows,
            c.experts.used as u32,
            placement
                .devices
                .iter()
                .map(|d| d.slots.min(opts.slot_bytes.unwrap_or(u64::MAX)))
                .collect(),
            ram_bytes,
        );
        if let Some(reserve) = opts.reserve {
            config.vram_reserve = reserve;
        }
        let plan = StorePlan::new(config, &map, &reader, &specs, None)?;
        let mut pinned = Ledger::new("pinned RAM", ram_bytes + (1 << 30));
        let mut slots = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            let mut device_slots = CudaSlots::new(
                device,
                &mut ledgers[index],
                &mut pinned,
                &plan.device_pools(index),
                c.n_layer,
                c.experts.count as u32,
            )?;
            if opts.serve != Serve::Stream {
                device_slots.add_host_shares(
                    &mut ledgers[index],
                    &mut pinned,
                    shapes[0].host_share_bytes(opts.part.max_rows),
                )?;
            }
            slots.push(device_slots);
        }
        let region = HostRegion::pinned(&devices[0], &mut pinned, plan.ram_bytes())?;
        let mut store = ExpertStore::new(plan, map, reader, slots, region, None)?;
        if opts.serve != Serve::Stream {
            // One core stays with the thread that drives the step.
            let workers = match opts.cpu_workers {
                Some(workers) => CpuPlacement::unpinned(workers.max(1)),
                None => CpuPlacement::for_machine(&Topology::read(), 1),
            };
            let pool = CpuPool::new(&workers, Duration::from_millis(20))
                .map_err(|e| invalid(format!("the CPU workers of the expert store: {e}")))?;
            store.enable_cpu(pool, shapes, opts.serve)?;
        }
        for (index, part) in parts.iter_mut().enumerate() {
            let at = part
                .layer_range()
                .map(|layer| {
                    let (device, pool) = store.place(layer);
                    debug_assert_eq!(device, index);
                    let slots = store.device(device);
                    ExpertAt {
                        table: slots.table(layer),
                        pool: slots.pool(pool).0,
                        host: slots.host_share(layer),
                    }
                })
                .collect();
            part.set_experts(at)?;
        }

        // The rows of the n-gram table come through a reader of their own:
        // the store takes every completion of its reader for itself.
        let rows = match &c.ngram {
            Some(_) => {
                let info = gguf
                    .tensor("per_layer_token_embd.weight")
                    .ok_or_else(|| invalid("the model has no n-gram table"))?;
                let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
                let table = RowTable::for_reader(&reader, info)?;
                let row_reader = RowReader::new(&reader, table, opts.row_cache);
                Some((reader, row_reader))
            }
            None => None,
        };

        let wide = c.wide() as u32;
        let (mesh, lanes) = if devices.len() > 1 {
            let values = opts.part.max_rows * wide;
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
        Ok(Self {
            c: c.clone(),
            w,
            streams: devices.iter().map(|d| Stream::new(d, 0)).collect::<Result<_, _>>()?,
            done: events()?,
            probe_done: events()?,
            sequences: (0..opts.part.max_slots)
                .map(|_| Sequence { position: 0, recent: Vec::new() })
                .collect(),
            opts,
            placement,
            devices,
            parts,
            ledgers,
            store,
            lanes,
            _mesh: mesh,
            rows,
            gather: RowGather::new(),
            completions: Vec::new(),
            windows: Vec::new(),
            step: 0,
            step_rows: 0,
            enqueued: 0,
            followed: 0,
            groups: Vec::new(),
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

    /// The expert store.
    pub fn store(&self) -> &ExpertStore<CudaSlots> {
        &self.store
    }

    /// What the store counted since the last call.
    pub fn take_store_stats(&mut self) -> StoreStats {
        self.store.take_stats()
    }

    /// The memory of every part.
    pub fn report(&self) -> Vec<PartReport> {
        self.parts
            .iter()
            .enumerate()
            .map(|(index, part)| {
                let layers = part.layer_range();
                let per_1000 = |bytes: u64| bytes * 1000 / u64::from(ares_cuda::ENTRY_CELLS);
                PartReport {
                    device: part.device().info().name.clone(),
                    n_slots: layers.clone().map(|l| self.store.slots(l).0).sum(),
                    layers,
                    weights: part.weight_bytes(),
                    slots: self.placement.devices[index].slots,
                    state_per_slot: part.state_bytes_per_slot(),
                    kv_per_1000: per_1000(part.page_bytes()),
                    keys_per_1000: per_1000(part.key_page_bytes()),
                    scratch: part.scratch_bytes(),
                    reserved: self.ledgers[index].reserved(),
                    upload: part.upload_time(),
                }
            })
            .collect()
    }

    /// The cell the next row of the sequence in `slot` takes.
    pub fn position(&self, slot: u32) -> u32 {
        self.sequences.get(slot as usize).map_or(0, |s| s.position)
    }

    /// Starts the sequence in `slot` over: its next row is cell 0.
    pub fn restart(&mut self, slot: u32) -> Result<(), Error> {
        let sequence = self
            .sequences
            .get_mut(slot as usize)
            .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
        sequence.position = 0;
        sequence.recent.clear();
        Ok(())
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers, most wanted first, as far as the slots go: from the file into
    /// VRAM, and into RAM what VRAM has no room for. Returns when nothing
    /// moves any more.
    pub fn fill(&mut self, experts_of: &mut dyn FnMut(usize) -> Vec<u32>) -> Result<Fill, Error> {
        let started = Instant::now();
        self.store.take_stats();
        let before = self.store.reader().stats();
        for layer in 0..self.c.n_layer {
            let (vram, ram) = self.store.slots(layer);
            let mut order = experts_of(layer);
            order.truncate((vram + ram).min(self.c.experts.count));
            self.store.fill(layer, &order)?;
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

    /// The device and the layer of that device's part for the `index`-th
    /// layer of the model.
    fn locate(&self, index: usize) -> (usize, usize) {
        let mut first = 0;
        for (device, part) in self.parts.iter().enumerate() {
            if index < first + part.n_layers() {
                return (device, index - first);
            }
            first += part.n_layers();
        }
        (self.parts.len() - 1, 0)
    }

    /// Hands the route records that have arrived to the store, in the order
    /// of the layers, and lets the store work. Never waits.
    fn pump(&mut self) -> Result<(), Error> {
        while self.followed < self.enqueued {
            let (device, i) = self.locate(self.followed);
            let part = &self.parts[device];
            if !part.record_arrived(i)? {
                break;
            }
            let record = RouteRecord::parse(
                part.route_record(i),
                self.c.experts.used as u32,
                self.step_rows,
            )
            .ok_or_else(|| invalid("a route record that is none"))?;
            self.groups.clear();
            self.groups.extend(record.groups());
            // The store uploads what the layer misses or computes it from
            // the layer's rows, by what its measured costs say.
            let rows = part.route_rows(i, self.step_rows as usize);
            self.store.routed_rows(
                self.followed,
                record.token(),
                self.step_rows as usize,
                &self.groups,
                rows,
            )?;
            self.followed += 1;
        }
        self.store.poll()?;
        Ok(())
    }

    /// Spins until `event` has happened on its device, following the route
    /// records meanwhile.
    fn wait(&mut self, device: usize, probe: bool) -> Result<(), Error> {
        let started = Instant::now();
        loop {
            let event = if probe { &self.probe_done[device] } else { &self.done[device] };
            if event.is_done()? {
                return Ok(());
            }
            self.pump()?;
            self.watch(started, "a device ran the step")?;
            std::hint::spin_loop();
        }
    }

    /// Hands every route record of the layers enqueued so far to the store,
    /// waiting for those that have not arrived.
    fn follow(&mut self) -> Result<(), Error> {
        let started = Instant::now();
        while self.followed < self.enqueued {
            self.pump()?;
            self.watch(started, "route records were on their way")?;
        }
        Ok(())
    }

    /// An error when the session has waited since `started` for longer than
    /// a step may stand still, with what it knows of the step.
    fn watch(&self, started: Instant, what: &str) -> Result<(), Error> {
        if started.elapsed() < self.opts.stall {
            return Ok(());
        }
        let (device, i) = self.locate(self.followed.min(self.enqueued.saturating_sub(1)));
        let arrived = self.parts[device].record_arrived(i).unwrap_or(false);
        let done: Vec<bool> = self.done.iter().map(|e| e.is_done().unwrap_or(false)).collect();
        let stats = self.store.stats();
        Err(invalid(format!(
            "step {} of {} rows stood still for {:.0?} while {what}: {} of {} enqueued layers have \
             their route record handled (the record of layer {} has {}arrived), devices done: \
             {done:?}; the store waits for {} experts and is {}settled; in this step and since the \
             counters were read: {} picks, {} in a slot, {} computed on the CPU, {} uploads, {} \
             reads from the file",
            self.step,
            self.step_rows,
            self.opts.stall,
            self.followed,
            self.enqueued,
            self.followed,
            if arrived { "" } else { "not " },
            self.store.waiting(),
            if self.store.is_settled() { "" } else { "not " },
            stats.picks,
            stats.hits,
            stats.cpu_experts,
            stats.uploads,
            stats.demand_reads,
        )))
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

    fn run(&mut self, rows: &[Rows<'_>], mut probe: Option<&mut dyn Probe>) -> Result<(), Error> {
        if !self.windows.is_empty() {
            return Err(invalid("the draft windows of the last step were not settled"));
        }
        let started = Instant::now();
        let n = self.c.n_embd;
        // The record, the embeddings, the rows of the n-gram table.
        let mut segments = Vec::with_capacity(rows.len());
        let mut tokens = Vec::new();
        let mut out_rows = Vec::new();
        let mut table_rows = Vec::new();
        for part in rows {
            let sequence = self
                .sequences
                .get(part.slot as usize)
                .ok_or_else(|| invalid(format!("slot {} of the session", part.slot)))?;
            if part.tokens.is_empty() {
                return Err(invalid("a sequence without rows in a step"));
            }
            if segments.iter().any(|s: &Segment| s.slot == part.slot) {
                return Err(invalid(format!("slot {} twice in a step", part.slot)));
            }
            let row0 = tokens.len() as u32;
            let count = part.tokens.len() as u32;
            segments.push(Segment {
                row0,
                n_rows: count,
                slot: part.slot,
                pos0: sequence.position as i32,
                direct: part.direct,
            });
            match part.wanted {
                Wanted::None => {}
                Wanted::Last => out_rows.push(row0 + count - 1),
                Wanted::All => out_rows.extend(row0..row0 + count),
            }
            if let Some(ngram) = &self.c.ngram {
                let mut before = sequence.recent.clone();
                for &token in part.tokens {
                    table_rows.extend(pieces::ngram_rows(ngram, token, &before));
                    before.push(token);
                }
            }
            tokens.extend_from_slice(part.tokens);
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= self.c.n_vocab) {
            return Err(invalid(format!("token {token} of a vocabulary of {}", self.c.n_vocab)));
        }
        let record = StepRecord::new(&segments, &tokens, None, &out_rows)?;
        let dims = record.dims();
        let mut x = vec![0.0f32; tokens.len() * n];
        for (row, &token) in tokens.iter().enumerate() {
            self.w.token_embd.decode_row(token as usize, &mut x[row * n..(row + 1) * n]);
        }
        if let Some((reader, row_reader)) = &mut self.rows {
            row_reader.start(
                reader,
                &table_rows,
                u64::from(self.step) << 20,
                Class::Demand,
                &mut self.gather,
            )?;
        }
        let prepared = Instant::now();
        let mut waited_rows = Duration::ZERO;

        self.step += 1;
        self.step_rows = dims.n_rows;
        self.enqueued = 0;
        self.followed = 0;
        self.store.begin_step();
        let token = self.step;
        let fetched: OnceCell<Vec<u8>> = OnceCell::new();
        let mut inbound: Option<DevicePtr> = None;
        // The step as the part of `$device` sees it. Built where it is used:
        // between the uses the session follows the route records.
        macro_rules! env {
            ($device:expr) => {
                StepEnv {
                    stream: &self.streams[$device],
                    record: &record,
                    segments: &segments,
                    token,
                    x: &x,
                    ngram: fetched.get().map(Vec::as_slice),
                    inbound,
                }
            };
        }
        let mut parcel: Option<Parcel> = None;
        let mut n_out = 0;
        for device in 0..self.parts.len() {
            if let Some(parcel) = &parcel {
                // SAFETY: the parcel is read on this stream between here and
                // its release below, by the copy `Part::begin` enqueues.
                inbound =
                    Some(unsafe { self.lanes[device - 1].receive(&self.streams[device], parcel)? });
            }
            self.store.device_mut(device).fence(&self.streams[device])?;
            self.parts[device].begin(&env!(device))?;
            if let Some(parcel) = parcel.take() {
                self.lanes[device - 1].release(&self.streams[device], parcel)?;
            }
            inbound = None;
            let first_layer = self.parts[device].layer_range().start;
            for i in 0..self.parts[device].n_layers() {
                if self.parts[device].ngram_layer() == Some(i) && fetched.get().is_none() {
                    let waiting = Instant::now();
                    self.await_rows()?;
                    waited_rows = waiting.elapsed();
                    let _ = fetched.set(self.gather.rows().to_vec());
                }
                if let Some(probe) = probe.as_deref_mut() {
                    self.parts[device].flush(&env!(device))?;
                    self.probe_done[device].record(&self.streams[device])?;
                    self.wait(device, true)?;
                    if let Some(r) = probe.before(first_layer + i) {
                        self.parts[device].write_residual(&r)?;
                    }
                }
                let held = Instant::now();
                while self.enqueued - self.followed >= LEAD_LAYERS {
                    self.pump()?;
                    self.watch(held, "the host held back the next layer")?;
                }
                self.parts[device].enqueue_layer(&env!(device), i)?;
                self.enqueued += 1;
                self.pump()?;
                if let Some(probe) = probe.as_deref_mut() {
                    self.parts[device].flush(&env!(device))?;
                    self.probe_done[device].record(&self.streams[device])?;
                    self.wait(device, true)?;
                    self.follow()?;
                    let part = &self.parts[device];
                    let rows = dims.n_rows as usize;
                    let route = RouteRecord::parse(
                        part.route_record(i),
                        self.c.experts.used as u32,
                        dims.n_rows,
                    )
                    .ok_or_else(|| invalid("a route record that is none"))?;
                    let out = LayerOut {
                        residual: &part.read_residual(rows)?,
                        core: &part.read_core_out(rows)?,
                        ffn: &part.read_ffn_out(rows)?,
                        picks: (0..rows).map(|row| route.ids(row).collect()).collect(),
                    };
                    probe.after(first_layer + i, out);
                }
            }
            n_out = self.parts[device].end(&env!(device))?;
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
        let enqueued = Instant::now();
        for device in 0..self.parts.len() {
            self.wait(device, false)?;
        }
        self.follow()?;
        self.store.end_step()?;
        for part in &mut self.parts {
            part.flush_profile()?;
        }

        // The picks, and where the sequences are now.
        self.n_out = n_out as usize;
        self.picks.clear();
        if let Some(head) = self.parts.last() {
            let n_vocab = self.c.n_vocab;
            for row in head.logits(self.n_out).chunks_exact(n_vocab) {
                let best = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, v)| {
                    if *v > b.1 { (i, *v) } else { b }
                });
                self.picks.push(best.0 as u32);
            }
        }
        for part in rows {
            if part.direct {
                self.advance(part.slot, part.tokens);
            } else {
                self.windows.push(Window { slot: part.slot, tokens: part.tokens.to_vec() });
            }
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

    /// Waits for the rows of the n-gram table of the step, following the
    /// route records meanwhile.
    fn await_rows(&mut self) -> Result<(), Error> {
        let started = Instant::now();
        while !self.gather.is_done() {
            self.watch(started, "rows of the n-gram table were read")?;
            let Some((reader, row_reader)) = &mut self.rows else { break };
            reader.poll(&mut self.completions);
            for completion in self.completions.drain(..) {
                row_reader.accept(&mut self.gather, completion);
            }
            self.pump()?;
        }
        match self.gather.result() {
            Ok(()) => Ok(()),
            Err(error) => Err(invalid(format!("reading rows of the n-gram table: {error}"))),
        }
    }

    fn advance(&mut self, slot: u32, tokens: &[u32]) {
        let keep = self.c.ngram.as_ref().map_or(0, |g| g.ngram_size.saturating_sub(1));
        let sequence = &mut self.sequences[slot as usize];
        sequence.position += tokens.len() as u32;
        sequence.recent.extend_from_slice(tokens);
        let surplus = sequence.recent.len().saturating_sub(keep);
        sequence.recent.drain(..surplus);
    }

    /// Settles the draft windows of the last step: `accepted[i]` rows of the
    /// `i`-th of them count. The commit is enqueued at once and runs before
    /// the next step.
    pub fn accept(&mut self, accepted: &[u32]) -> Result<(), Error> {
        if accepted.len() != self.windows.len() {
            return Err(invalid(format!(
                "{} outcomes for {} draft windows",
                accepted.len(),
                self.windows.len()
            )));
        }
        if self.windows.is_empty() {
            return Ok(());
        }
        for (window, count) in self.windows.iter().zip(accepted) {
            if *count as usize > window.tokens.len() {
                return Err(invalid(format!(
                    "{count} accepted rows of a window of {}",
                    window.tokens.len()
                )));
            }
        }
        for (part, stream) in self.parts.iter_mut().zip(&self.streams) {
            part.enqueue_commit(stream, accepted)?;
        }
        let windows = std::mem::take(&mut self.windows);
        for (window, count) in windows.iter().zip(accepted) {
            self.advance(window.slot, &window.tokens[..*count as usize]);
        }
        Ok(())
    }

    /// The logits of the `i`-th wanted row of the last step.
    pub fn logits(&self, i: usize) -> Option<&[f32]> {
        let n_vocab = self.c.n_vocab;
        let all = self.parts.last()?.logits(self.n_out);
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
        let started = Instant::now();
        let idle = |streams: &[Stream]| streams.iter().all(|s| s.is_idle().unwrap_or(true));
        while !idle(&self.streams) && started.elapsed() < Duration::from_secs(5) {
            if self.store.poll().is_err() {
                break;
            }
        }
        if idle(&self.streams) {
            for stream in &self.streams {
                let _ = stream.synchronize();
            }
        }
    }
}
