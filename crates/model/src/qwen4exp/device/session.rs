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
//!
//! Two devices that can write each other's memory work on every layer
//! together ([`Spread`]): the session then enqueues each layer on both, the
//! mixer first, and the devices exchange what they need on their own. The
//! expert store sees every layer once per device, with the experts that
//! device owns.
//!
//! The rows of the n-gram table that a token needs depend on the token and
//! the two before it, so they are asked for the moment a step has picked its
//! token ([`Session::expect`] for a caller that knows better), not when the
//! next step reaches the layer that reads them.

use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, Kernels, Lane, Mesh, MixerMatrix, Parcel, Q4Kernels,
    RouteRecord, Segment, StepRecord, Stream, peer_access,
};
use ares_gguf::Gguf;
use ares_io::Reader;
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, Serve, StoreConfig, StorePlan, StoreStats,
};

use super::Error;
use super::feed::{Feed, Layout, build_store, cpu_pool, device_slots, places};
use super::load::Role;
use super::ngram::NgramFeed;
use super::part::{Options, Part, PartSetup, StepEnv};
use super::placement::{DeviceMemory, PartPlan, Placement, TensorChoice, ram_tier_bytes};
use crate::qwen4exp::Config;
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

/// How a model is spread over two devices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spread {
    /// Both devices on every layer when they can (two devices, each able to
    /// write the other's memory, mixers the fused read takes) and no layer
    /// bounds are given; otherwise by layers.
    Auto,
    /// A range of layers per device: one sequence uses one device at a time.
    Layers,
    /// Both devices on every layer. What is `None` is taken from a
    /// measurement at load: the mixer is the device whose read of a stream
    /// mixer is faster, and its share of the recurrent heads is the worker's
    /// read time over the sum of both. The shared expert is the worker's
    /// unless said otherwise.
    Tensor {
        /// The device that holds the streams and runs the mixers.
        mixer: Option<usize>,
        /// Its share of the key heads of the recurrent layers.
        core_share: Option<f64>,
        /// Whether it computes the shared expert.
        shared_on_mixer: Option<bool>,
    },
}

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
    /// How two devices share the model.
    pub spread: Spread,
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
            spread: Spread::Auto,
            routing: None,
            worker_picks: None,
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
    /// The part's index in the placement.
    pub index: usize,
    /// The device's name.
    pub device: String,
    /// What the part is to the residual streams.
    pub role: Role,
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

/// What every part sees of a step.
struct Step<'a> {
    record: &'a StepRecord,
    segments: &'a [Segment],
    x: &'a [f32],
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
            x: $step.x,
            ngram: $fetched.as_deref(),
            inbound: $inbound,
            xbase: $step.xbase,
            pools: $self.pools[$device].ptr(),
        }
    };
}

/// How long a read of a stream mixer of the model's sizes takes on `device`:
/// the two fused launches over weights of the first layer's types, which are
/// bound by the device's memory as the model's dense products are.
pub fn read_time(device: &Device, c: &Config, w: &Weights<'_>) -> Result<Duration, Error> {
    let (hc, n, rank) = (c.n_stream as u32, c.n_embd as u32, c.mixer_rank as u32);
    let mixer = &w.layers[0].attn_mixer;
    let wide = u64::from(hc) * u64::from(n);
    let width = |ty: GgmlType| if ty == GgmlType::F32 { 4 } else { 2 };
    let zeroed = |bytes: u64| -> Result<DeviceBuffer, Error> {
        let buffer = DeviceBuffer::new(device, bytes.max(64))?;
        buffer.write(0, &vec![0u8; bytes.max(64) as usize])?;
        Ok(buffer)
    };
    let kernels = Q4Kernels::load(device)?;
    let stream = Stream::new(device, 0)?;
    let (r, norm, xn) = (zeroed(wide * 4)?, zeroed(wide * 4)?, zeroed(wide * 4)?);
    let down = zeroed(u64::from(rank) * wide * width(mixer.down.ty()))?;
    let up = zeroed(wide * u64::from(rank) * width(mixer.up.ty()))?;
    let partials = zeroed(Q4Kernels::read_partials(1, hc, rank) * 4)?;
    let act = zeroed(Kernels::quantized_activation_bytes(n, 1))?;
    let run = |count: u32| -> Result<(), Error> {
        for _ in 0..count {
            // SAFETY: every buffer is of this device and of the sizes the
            // two launches read and write for one row.
            unsafe {
                kernels.read_down(
                    &stream,
                    r.ptr(),
                    r.ptr(),
                    None,
                    Some((
                        norm.ptr(),
                        xn.ptr(),
                        MixerMatrix { ptr: down.ptr(), ty: mixer.down.ty() },
                        None,
                    )),
                    partials.ptr(),
                    1,
                    hc,
                    n,
                    rank,
                    c.rms_eps,
                )?;
                kernels.read_up(
                    &stream,
                    xn.ptr(),
                    partials.ptr(),
                    MixerMatrix { ptr: up.ptr(), ty: mixer.up.ty() },
                    act.ptr(),
                    None,
                    None,
                    1,
                    hc,
                    n,
                    rank,
                )?;
            }
        }
        stream.synchronize()?;
        Ok(())
    };
    run(8)?;
    let started = Instant::now();
    run(64)?;
    Ok(started.elapsed() / 64)
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

/// What [`choose`] decided and measured.
pub(super) struct Chosen {
    pub placement: Placement,
    /// For two devices that share every layer: the time of a read of a stream
    /// mixer on each, and the mixer among them.
    pub speeds: Option<([f64; 2], usize)>,
    /// What each device had to give the model when it was looked at.
    pub memory: Vec<DeviceMemory>,
}

/// Where the layers of the model go on `devices`: a range each, or both
/// devices on every layer when they can write each other's memory and the
/// mixers run as fused reads. Measures a read of a stream mixer on each of
/// two devices that share the layers.
pub(super) fn choose(
    c: &Config,
    w: &Weights<'_>,
    devices: &[Device],
    part: &Options,
    headroom: u64,
    spread: &Spread,
    bounds: Option<&[usize]>,
) -> Result<Chosen, Error> {
    let mut memory = Vec::with_capacity(devices.len());
    for device in devices {
        let free = device.mem_info()?.free;
        memory.push(DeviceMemory {
            budget: free.saturating_sub(headroom),
            multiprocessors: device.info().multiprocessors,
        });
    }
    // Both devices on every layer when each can write the other's memory
    // and the mixers run as fused reads.
    let peers = devices.len() == 2
        && peer_access(devices[0].ordinal(), devices[1].ordinal())?.0
        && peer_access(devices[1].ordinal(), devices[0].ordinal())?.0;
    let mixer = &w.layers[0].attn_mixer;
    let fused = Q4Kernels::reads_fused(
        c.n_stream as u32,
        c.n_embd as u32,
        c.mixer_rank as u32,
        &[mixer.down.ty(), mixer.up.ty()],
    );
    let shared = match *spread {
        Spread::Layers => None,
        Spread::Auto if bounds.is_some() || !peers || !fused => None,
        Spread::Auto => Some((None, None, None)),
        Spread::Tensor { .. } if !peers => {
            return Err(invalid(
                "two devices share every layer only when each can write the other's memory \
                 (peer access between exactly two devices)",
            ));
        }
        Spread::Tensor { mixer, core_share, shared_on_mixer } => {
            Some((mixer, core_share, shared_on_mixer))
        }
    };
    // What is measured of two devices that share the layers: the time
    // of a read of a stream mixer on each, and the mixer among them.
    let mut speeds: Option<([f64; 2], usize)> = None;
    let placement = match (shared, bounds) {
        (Some((mixer, core_share, shared_on_mixer)), _) => {
            devices[0].enable_peer_access(&devices[1])?;
            devices[1].enable_peer_access(&devices[0])?;
            let times = [
                read_time(&devices[0], c, w)?.as_secs_f64(),
                read_time(&devices[1], c, w)?.as_secs_f64(),
            ];
            let mixer = mixer.unwrap_or(usize::from(times[1] < times[0])).min(1);
            speeds = Some((times, mixer));
            let (fast, slow) = (times[mixer], times[1 - mixer]);
            let choice = TensorChoice {
                mixer,
                core_share: core_share.unwrap_or(slow / (fast + slow).max(1e-9)),
                shared_on_mixer: shared_on_mixer.unwrap_or(false),
            };
            Placement::tensor(c, w, part, &memory, &choice)?
        }
        (None, Some(bounds)) => Placement::at(c, w, part, &memory, bounds)?,
        (None, None) => Placement::plan(c, w, part, &memory)?,
    };
    Ok(Chosen { placement, speeds, memory })
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
    /// The pools of K/V and keys of every part.
    pools: Vec<DeviceBuffer>,
    ledgers: Vec<Ledger>,
    feed: Feed,
    lanes: Vec<Lane>,
    _mesh: Option<Mesh>,
    ngram: NgramFeed,
    step: u32,
    step_rows: u32,
    /// Per part: its layers of the running step that were enqueued.
    enqueued: Vec<usize>,
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
        let Chosen { mut placement, speeds, memory } = choose(
            c,
            w,
            &devices,
            &opts.part,
            opts.headroom,
            &opts.spread,
            opts.bounds.as_deref(),
        )?;

        let mut ledgers: Vec<Ledger> = devices
            .iter()
            .zip(&memory)
            .map(|(device, m)| Ledger::new(format!("CUDA{}", device.ordinal()), m.budget))
            .collect();
        let mut parts = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            let PartPlan { role, first, shares, head } = placement.part(c, index);
            parts.push(Part::open(
                c,
                w,
                &opts.part,
                PartSetup { index, role, first, shares, head, device, ledger: &mut ledgers[index] },
            )?);
        }
        if let Some(tensor) = &placement.tensor {
            // Each part learns where the other receives, the mixer which
            // experts are the worker's.
            let (mixer, worker) = (tensor.mixer, 1 - tensor.mixer);
            let ports = |part: &Part| {
                part.ports().ok_or_else(|| invalid("a part that shares layers has no ports"))
            };
            let (to_mixer, to_worker) = (ports(&parts[mixer])?, ports(&parts[worker])?);
            let cores = |part: &Part| (0..c.n_layer).map(|i| part.has_core(i)).collect::<Vec<_>>();
            let (mixer_cores, worker_cores) = (cores(&parts[mixer]), cores(&parts[worker]));
            parts[mixer].set_peer(to_worker, worker_cores)?;
            parts[worker].set_peer(to_mixer, mixer_cores)?;
        }

        // The pools of K/V and keys, whose pages every sequence slot owns.
        let mut pools = Vec::with_capacity(parts.len());
        for (index, part) in parts.iter_mut().enumerate() {
            ledgers[index].reserve("qwen4exp K/V and key pools", part.pool_bytes())?;
            pools.push(DeviceBuffer::new(&devices[index], part.pool_bytes())?);
            part.own_pages()?;
        }

        // The expert store: slots from what the placement left, a tier in
        // page-locked RAM, reads from the file.
        // When the devices share the layers, the store sees every layer once
        // per device: the experts of the layer that the device owns.
        let (layout, map) = Layout::new(gguf, c, w, &placement)?;
        let store_layers = c.n_layer * layout.per_layer;
        let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
        let vram: Vec<u64> = placement
            .devices
            .iter()
            .map(|d| d.slots.min(opts.slot_bytes.unwrap_or(u64::MAX)))
            .collect();
        let ram_bytes = ram_tier_bytes(opts.ram_tier, layout.ram_wanted(vram.iter().sum()));
        let mut config =
            StoreConfig::new(opts.part.max_rows, c.experts.used as u32, vram, ram_bytes);
        if let Some(reserve) = opts.reserve {
            config.vram_reserve = reserve;
        }
        let reserve = config.vram_reserve;
        let plan = StorePlan::new(config, &map, &reader, &layout.specs, None)?;
        if let Some((times, mixer)) = speeds {
            // With a recorded routing the experts are divided by it: each
            // device gets the share of the picks that its speed is of both,
            // and what a step can miss is owned where an upload is faster.
            if let Some(routing) = opts.routing.as_ref().filter(|r| r.len() == c.n_layer) {
                let slots: Vec<Vec<usize>> = (0..layout.per_layer)
                    .map(|device| {
                        (0..c.n_layer)
                            .map(|layer| {
                                plan.slots(layout.store_layer(device, layer))
                                    .0
                                    .saturating_sub(reserve)
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
                let uploads =
                    [upload_time(&devices[0], expert)?, upload_time(&devices[1], expert)?];
                placement.assign(routing, &slots, picks, usize::from(uploads[1] < uploads[0]));
            }
            let owner: Vec<u8> = (0..c.n_layer)
                .flat_map(|layer| {
                    let placement = &placement;
                    (0..c.experts.count as u32)
                        .map(move |expert| u8::from(placement.owner(layer, expert) != mixer))
                })
                .collect();
            parts[mixer].set_owner(&owner)?;
        }
        let mut pinned = Ledger::new("pinned RAM", ram_bytes + (1 << 30));
        let mut slots = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            slots.push(device_slots(
                device,
                &mut ledgers[index],
                &mut pinned,
                &plan.device_pools(index),
                &layout,
                opts.serve,
                opts.part.max_rows,
            )?);
        }
        debug_assert_eq!(plan.n_layers(), store_layers);
        let region = HostRegion::pinned(&devices[0], &mut pinned, plan.ram_bytes())?;
        // One core stays with the thread that drives the step.
        let cpu = (opts.serve != Serve::Stream)
            .then(|| cpu_pool(opts.cpu_workers, 1).map(|pool| (pool, opts.serve)))
            .transpose()?;
        let store = build_store(plan, &layout, map, reader, slots, region, cpu)?;
        for (index, part) in parts.iter_mut().enumerate() {
            let layers = part.layer_range();
            part.set_experts(places(&store, layers, &layout, index))?;
        }
        let boards = parts.iter().map(Part::board).collect();
        let firsts = parts.iter().map(|part| part.layer_range().start).collect();
        let feed = Feed::new(store, boards, firsts, layout, c.experts.used as u32);

        let ngram = NgramFeed::open(gguf, c, opts.row_cache, opts.part.max_slots)?;

        let wide = c.wide() as u32;
        let (mesh, lanes) = if devices.len() > 1 && placement.tensor.is_none() {
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
        let n_parts = parts.len();
        Ok(Self {
            c: c.clone(),
            w,
            streams: devices.iter().map(|d| Stream::new(d, 0)).collect::<Result<_, _>>()?,
            done: events()?,
            probe_done: events()?,
            opts,
            placement,
            devices,
            parts,
            pools,
            ledgers,
            feed,
            lanes,
            _mesh: mesh,
            ngram,
            step: 0,
            step_rows: 0,
            enqueued: vec![0; n_parts],
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
        &self.feed.store
    }

    /// What the store counted since the last call.
    pub fn take_store_stats(&mut self) -> StoreStats {
        self.feed.store.take_stats()
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
                    index,
                    device: part.device().info().name.clone(),
                    role: part.role(),
                    n_slots: layers
                        .clone()
                        .map(|l| self.feed.store.slots(self.store_layer(index, l)).0)
                        .sum(),
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

    /// The slots of the experts' tier in host memory, over all layers.
    pub fn ram_slots(&self) -> usize {
        let per_layer = if self.placement.tensor.is_some() { self.parts.len() } else { 1 };
        (0..self.c.n_layer * per_layer).map(|at| self.feed.store.slots(at).1).sum()
    }

    /// The store's number for the experts of `layer` that the part `part`
    /// owns: the layer itself when the model is split by layers.
    fn store_layer(&self, part: usize, layer: usize) -> usize {
        self.feed.layout().store_layer(part, layer)
    }

    /// The cell the next row of the sequence in `slot` takes.
    pub fn position(&self, slot: u32) -> u32 {
        self.ngram.position(slot)
    }

    /// Starts the sequence in `slot` over: its next row is cell 0.
    pub fn restart(&mut self, slot: u32) -> Result<(), Error> {
        self.ngram.restart(slot)
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers, most wanted first, as far as the slots go: from the file into
    /// VRAM, and into RAM what VRAM has no room for. Returns when nothing
    /// moves any more.
    pub fn fill(&mut self, experts_of: &mut dyn FnMut(usize) -> Vec<u32>) -> Result<Fill, Error> {
        let placement = &self.placement;
        self.feed.fill(
            self.c.n_layer,
            self.c.experts.count,
            &|layer, expert| placement.owner(layer, expert),
            experts_of,
        )
    }

    /// Hands the route records that have arrived to the store and lets the
    /// store and the reads of table rows work. Never waits.
    fn pump(&mut self) -> Result<(), Error> {
        self.feed.pump(&self.enqueued)?;
        self.ngram.poll();
        Ok(())
    }

    /// The most layers any part is enqueued ahead of its route records.
    fn lead(&self) -> usize {
        self.enqueued.iter().zip(self.feed.followed()).map(|(e, f)| e - f).max().unwrap_or(0)
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
        while !self.feed.is_followed(&self.enqueued) {
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
        let done: Vec<bool> = self.done.iter().map(|e| e.is_done().unwrap_or(false)).collect();
        Err(invalid(format!(
            "step {} of {} rows stood still for {:.0?} while {what}: {}, devices done: {done:?}",
            self.step,
            self.step_rows,
            self.opts.stall,
            self.feed.standing_still(&self.enqueued),
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

    fn run(&mut self, rows: &[Rows<'_>], probe: Option<&mut dyn Probe>) -> Result<(), Error> {
        if self.ngram.has_windows() {
            return Err(invalid("the draft windows of the last step were not settled"));
        }
        let started = Instant::now();
        let n = self.c.n_embd;
        // The record, the embeddings, the rows of the n-gram table.
        let mut segments = Vec::with_capacity(rows.len());
        let mut tokens = Vec::new();
        let mut out_rows = Vec::new();
        for part in rows {
            if part.slot >= self.opts.part.max_slots {
                return Err(invalid(format!("slot {} of the session", part.slot)));
            }
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
                pos0: self.ngram.position(part.slot) as i32,
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
            return Err(invalid(format!("token {token} of a vocabulary of {}", self.c.n_vocab)));
        }
        let record = StepRecord::new(&segments, &tokens, None, &out_rows)?;
        let dims = record.dims();
        let mut x = vec![0.0f32; tokens.len() * n];
        for (row, &token) in tokens.iter().enumerate() {
            self.w.token_embd.decode_row(token as usize, &mut x[row * n..(row + 1) * n]);
        }
        // The rows of the table: those asked for when the tokens were first
        // known, when these are the tokens; otherwise they are read now.
        let asked: Vec<(u32, &[u32])> = rows.iter().map(|p| (p.slot, p.tokens)).collect();
        self.ngram.ask(&asked)?;
        let prepared = Instant::now();
        let mut waited_rows = Duration::ZERO;

        self.step += 1;
        self.step_rows = dims.n_rows;
        self.enqueued.fill(0);
        self.feed.begin_step(dims.n_rows);
        // The exchanges between a mixer and its worker are numbered through
        // the steps: eight per layer.
        let per_step = 8 * (self.c.n_layer as u32 + 1);
        let step = Step {
            record: &record,
            segments: &segments,
            x: &x,
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
            self.wait(head, false)?;
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
        for part in rows {
            if part.direct {
                self.ngram.advance(part.slot, part.tokens);
            } else {
                self.ngram.hold(part.slot, part.tokens);
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

        for device in 0..self.parts.len() {
            self.wait(device, false)?;
        }
        self.follow()?;
        self.feed.end_step()?;
        for part in &mut self.parts {
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
            self.feed.store.device_mut(device).fence(&self.streams[device])?;
            self.parts[device].begin(&step_env!(self, step, device, fetched, inbound))?;
            if let Some(parcel) = parcel.take() {
                self.lanes[device - 1].release(&self.streams[device], parcel)?;
            }
            inbound = None;
            let first_layer = self.parts[device].layer_range().start;
            for i in 0..self.parts[device].n_layers() {
                if self.parts[device].ngram_layer() == Some(i) && fetched.is_none() {
                    let waiting = Instant::now();
                    self.await_rows()?;
                    *waited_rows = waiting.elapsed();
                    fetched = Some(self.ngram.rows()?.to_vec());
                }
                if let Some(probe) = probe.as_deref_mut() {
                    self.parts[device].flush(&step_env!(self, step, device, fetched, inbound))?;
                    self.probe_done[device].record(&self.streams[device])?;
                    self.wait(device, true)?;
                    if let Some(r) = probe.before(first_layer + i) {
                        self.parts[device].write_residual(&r)?;
                    }
                }
                let held = Instant::now();
                while self.lead() >= LEAD_LAYERS {
                    self.pump()?;
                    self.watch(held, "the host held back the next layer")?;
                }
                self.parts[device]
                    .enqueue_layer(&step_env!(self, step, device, fetched, inbound), i)?;
                self.enqueued[device] += 1;
                self.pump()?;
                if let Some(probe) = probe.as_deref_mut() {
                    self.parts[device].flush(&step_env!(self, step, device, fetched, inbound))?;
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
            self.feed.store.device_mut(device).fence(&self.streams[device])?;
            self.parts[device].begin(&step_env!(self, step, device, fetched, None))?;
        }
        // In front of and behind a probed layer both devices come to rest:
        // the worker sends what it has, the mixer writes it to the streams.
        macro_rules! settle {
            () => {
                for device in [worker, mixer] {
                    self.parts[device].flush(&step_env!(self, step, device, fetched, None))?;
                    self.probe_done[device].record(&self.streams[device])?;
                }
                for device in [worker, mixer] {
                    self.wait(device, true)?;
                }
            };
        }
        for i in 0..self.c.n_layer {
            if self.parts[mixer].ngram_layer() == Some(i) && fetched.is_none() {
                let waiting = Instant::now();
                self.await_rows()?;
                *waited_rows = waiting.elapsed();
                fetched = Some(self.ngram.rows()?.to_vec());
            }
            if let Some(probe) = probe.as_deref_mut() {
                settle!();
                if let Some(r) = probe.before(i) {
                    self.parts[mixer].write_residual(&r)?;
                }
            }
            let held = Instant::now();
            while self.lead() >= LEAD_LAYERS {
                self.pump()?;
                self.watch(held, "the host held back the next layer")?;
            }
            for device in [mixer, worker] {
                self.parts[device]
                    .enqueue_layer(&step_env!(self, step, device, fetched, None), i)?;
                self.enqueued[device] += 1;
            }
            self.pump()?;
            if let Some(probe) = probe.as_deref_mut() {
                settle!();
                self.follow()?;
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
                let route =
                    RouteRecord::parse(m.route_record(i), self.c.experts.used as u32, dims.n_rows)
                        .ok_or_else(|| invalid("a route record that is none"))?;
                let out = LayerOut {
                    residual: &m.read_residual(rows)?,
                    core: &core,
                    ffn: &ffn,
                    picks: (0..rows).map(|row| route.ids(row).collect()).collect(),
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
        self.ngram.expect(next)
    }

    /// Waits for the rows of the n-gram table of the step, following the
    /// route records meanwhile.
    fn await_rows(&mut self) -> Result<(), Error> {
        let started = Instant::now();
        while !self.ngram.is_done() {
            self.watch(started, "rows of the n-gram table were read")?;
            self.pump()?;
        }
        self.ngram.rows().map(|_| ())
    }

    /// Settles the draft windows of the last step: `accepted[i]` rows of the
    /// `i`-th of them count. The commit is enqueued at once and runs before
    /// the next step.
    pub fn accept(&mut self, accepted: &[u32]) -> Result<(), Error> {
        if !self.ngram.has_windows() {
            return self.ngram.settle(accepted);
        }
        self.ngram.settle(accepted)?;
        for (part, stream) in self.parts.iter_mut().zip(&self.streams) {
            part.enqueue_commit(stream, accepted)?;
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
        let started = Instant::now();
        let idle = |streams: &[Stream]| streams.iter().all(|s| s.is_idle().unwrap_or(true));
        while !idle(&self.streams) && started.elapsed() < Duration::from_secs(5) {
            if self.feed.store.poll().is_err() {
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
