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
    RouteGroup, RouteRecord, Segment, StepRecord, Stream, peer_access,
};
use ares_gguf::Gguf;
use ares_io::{Class, Completion, ExpertMap, Reader, RowGather, RowReader, RowTable};
use ares_runtime::expert_store::cpu::{CpuPool, Placement as CpuPlacement, Topology};
use ares_runtime::expert_store::{
    CudaSlots, ExpertStore, HostRegion, LayerSpec, Serve, StoreConfig, StorePlan, StoreStats,
};

use super::Error;
use super::load::{LayerShare, Role};
use super::part::{ExpertAt, Options, Part, PartSetup, StepEnv};
use super::placement::{DeviceMemory, Placement, TensorChoice, ram_tier_bytes};
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

struct Sequence {
    position: u32,
    /// The last tokens of the sequence, as many as the n-gram hash reads.
    recent: Vec<u32>,
}

struct Window {
    slot: u32,
    tokens: Vec<u32>,
}

/// Rows of the n-gram table that were asked for before the step that needs
/// them.
struct Ahead {
    rows: Vec<u64>,
    gather: RowGather,
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
    /// The rows asked for ahead of the next step, and gathers that were
    /// given up with reads still in flight.
    ahead: Option<Ahead>,
    stale: Vec<RowGather>,
    reads: u64,
    completions: Vec<Completion>,
    sequences: Vec<Sequence>,
    /// The draft windows of the last step.
    windows: Vec<Window>,
    step: u32,
    step_rows: u32,
    /// Per part: its layers of the running step that were enqueued, and
    /// those whose route records were handed to the store.
    enqueued: Vec<usize>,
    followed: Vec<usize>,
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
        let shared = match opts.spread {
            Spread::Layers => None,
            Spread::Auto if opts.bounds.is_some() || !peers || !fused => None,
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
        let mut placement = match (shared, &opts.bounds) {
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
                Placement::tensor(c, w, &opts.part, &memory, &choice)?
            }
            (None, Some(bounds)) => Placement::at(c, w, &opts.part, &memory, bounds)?,
            (None, None) => Placement::plan(c, w, &opts.part, &memory)?,
        };

        let mut ledgers: Vec<Ledger> = devices
            .iter()
            .zip(&memory)
            .map(|(device, m)| Ledger::new(format!("CUDA{}", device.ordinal()), m.budget))
            .collect();
        let mut parts = Vec::with_capacity(devices.len());
        for (index, (device, plan)) in devices.iter().zip(&placement.devices).enumerate() {
            let (role, shares) = match &placement.tensor {
                Some(tensor) => (
                    if index == tensor.mixer { Role::Mixer } else { Role::Worker },
                    tensor.shares[index].clone(),
                ),
                None => (
                    Role::Whole,
                    plan.layers.clone().map(|layer| LayerShare::whole(c, layer)).collect(),
                ),
            };
            parts.push(Part::open(
                c,
                w,
                &opts.part,
                PartSetup {
                    index,
                    role,
                    first: plan.layers.start,
                    shares,
                    head: plan.head,
                    device,
                    ledger: &mut ledgers[index],
                },
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

        // The expert store: slots from what the placement left, a tier in
        // page-locked RAM, reads from the file.
        // When the devices share the layers, the store sees every layer once
        // per device: the experts of the layer that the device owns.
        let patterns = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
        let per_layer = if placement.tensor.is_some() { devices.len() } else { 1 };
        let store_layers = c.n_layer * per_layer;
        let map = if per_layer > 1 {
            let names: Vec<Vec<String>> = (0..store_layers)
                .map(|at| {
                    patterns.iter().map(|p| format!("blk.{}.{p}.weight", at / per_layer)).collect()
                })
                .collect();
            ExpertMap::new(gguf, &names)?
        } else {
            ExpertMap::discover(gguf, &patterns)?
        };
        let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
        let mut specs = Vec::with_capacity(store_layers);
        let mut shapes = Vec::with_capacity(store_layers);
        let mut expert_bytes = 0u64;
        for at in 0..store_layers {
            let layer = at / per_layer;
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
        let reserve = config.vram_reserve;
        let plan = StorePlan::new(config, &map, &reader, &specs, None)?;
        if let Some((times, mixer)) = speeds {
            // With a recorded routing the experts are divided by it: each
            // device gets the share of the picks that its speed is of both,
            // and what a step can miss is owned where an upload is faster.
            if let Some(routing) = opts.routing.as_ref().filter(|r| r.len() == c.n_layer) {
                let slots: Vec<Vec<usize>> = (0..per_layer)
                    .map(|device| {
                        (0..c.n_layer)
                            .map(|layer| {
                                plan.slots(layer * per_layer + device).0.saturating_sub(reserve)
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
                let expert = shapes[0].slot_bytes().unwrap_or(1 << 20);
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
            let mut device_slots = CudaSlots::new(
                device,
                &mut ledgers[index],
                &mut pinned,
                &plan.device_pools(index),
                store_layers,
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
                    let at = if per_layer > 1 { layer * per_layer + index } else { layer };
                    let (device, pool) = store.place(at);
                    debug_assert_eq!(device, index);
                    let slots = store.device(device);
                    ExpertAt {
                        table: slots.table(at),
                        pool: slots.pool(pool).0,
                        host: slots.host_share(at),
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
            ahead: None,
            stale: Vec::new(),
            reads: 0,
            completions: Vec::new(),
            windows: Vec::new(),
            step: 0,
            step_rows: 0,
            enqueued: vec![0; n_parts],
            followed: vec![0; n_parts],
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
                    role: part.role(),
                    n_slots: layers
                        .clone()
                        .map(|l| self.store.slots(self.store_layer(index, l)).0)
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
        (0..self.c.n_layer * per_layer).map(|at| self.store.slots(at).1).sum()
    }

    /// The store's number for the experts of `layer` that the part `part`
    /// owns: the layer itself when the model is split by layers.
    fn store_layer(&self, part: usize, layer: usize) -> usize {
        if self.placement.tensor.is_some() { layer * self.parts.len() + part } else { layer }
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
        let shared = self.placement.tensor.is_some();
        for layer in 0..self.c.n_layer {
            let wanted = experts_of(layer);
            for part in 0..if shared { self.parts.len() } else { 1 } {
                let at = self.store_layer(part, layer);
                let (vram, ram) = self.store.slots(at);
                let mut order: Vec<u32> = wanted
                    .iter()
                    .copied()
                    .filter(|expert| !shared || self.placement.owner(layer, *expert) == part)
                    .collect();
                order.truncate((vram + ram).min(self.c.experts.count));
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

    /// Hands the route records that have arrived to the store, per part in
    /// the order of its layers, and lets the store and the reads of table
    /// rows work. Never waits.
    fn pump(&mut self) -> Result<(), Error> {
        for index in 0..self.parts.len() {
            while self.followed[index] < self.enqueued[index] {
                let i = self.followed[index];
                let part = &self.parts[index];
                if !part.record_arrived(i)? {
                    break;
                }
                let record = RouteRecord::parse(
                    part.route_record(i),
                    self.c.experts.used as u32,
                    self.step_rows,
                )
                .ok_or_else(|| invalid("a route record that is none"))?;
                let at = self.store_layer(index, part.layer_range().start + i);
                self.groups.clear();
                self.groups.extend(record.groups());
                // The store uploads what the layer misses or computes it from
                // the layer's rows, by what its measured costs say.
                let rows = part.route_rows(i, self.step_rows as usize);
                self.store.routed_rows(
                    at,
                    record.token(),
                    self.step_rows as usize,
                    &self.groups,
                    rows,
                )?;
                self.followed[index] += 1;
            }
        }
        self.poll_rows();
        self.store.poll()?;
        Ok(())
    }

    /// The most layers any part is enqueued ahead of its route records.
    fn lead(&self) -> usize {
        self.enqueued.iter().zip(&self.followed).map(|(e, f)| e - f).max().unwrap_or(0)
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
        while self.followed != self.enqueued {
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
        let stats = self.store.stats();
        Err(invalid(format!(
            "step {} of {} rows stood still for {:.0?} while {what}: per part {:?} layers are \
             enqueued and {:?} have their route record handled, devices done: {done:?}; the store \
             waits for {} experts and is {}settled; in this step and since the counters were \
             read: {} picks, {} in a slot, {} computed on the CPU, {} uploads, {} reads from the \
             file",
            self.step,
            self.step_rows,
            self.opts.stall,
            self.enqueued,
            self.followed,
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

    fn run(&mut self, rows: &[Rows<'_>], probe: Option<&mut dyn Probe>) -> Result<(), Error> {
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
        // The rows of the table: those asked for when the tokens were first
        // known, when these are the tokens; otherwise they are read now.
        match self.ahead.take() {
            Some(ahead) if ahead.rows == table_rows => {
                let last = std::mem::replace(&mut self.gather, ahead.gather);
                self.retire(last);
            }
            ahead => {
                if let Some(ahead) = ahead {
                    self.retire(ahead.gather);
                }
                let last = std::mem::replace(&mut self.gather, RowGather::new());
                self.retire(last);
                if let Some((reader, row_reader)) = &mut self.rows {
                    self.reads += 1;
                    row_reader.start(
                        reader,
                        &table_rows,
                        self.reads << 20,
                        Class::Demand,
                        &mut self.gather,
                    )?;
                }
            }
        }
        let prepared = Instant::now();
        let mut waited_rows = Duration::ZERO;

        self.step += 1;
        self.step_rows = dims.n_rows;
        self.enqueued.fill(0);
        self.followed.fill(0);
        self.store.begin_step();
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
                self.advance(part.slot, part.tokens);
            } else {
                self.windows.push(Window { slot: part.slot, tokens: part.tokens.to_vec() });
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
        self.store.end_step()?;
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
            self.store.device_mut(device).fence(&self.streams[device])?;
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
                    fetched = Some(self.gather.rows().to_vec());
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
            self.store.device_mut(device).fence(&self.streams[device])?;
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
                fetched = Some(self.gather.rows().to_vec());
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
        let Some(ngram) = &self.c.ngram else { return Ok(()) };
        let mut table_rows = Vec::new();
        for (slot, tokens) in next {
            let sequence = self
                .sequences
                .get(*slot as usize)
                .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
            let mut before = sequence.recent.clone();
            for &token in *tokens {
                table_rows.extend(pieces::ngram_rows(ngram, token, &before));
                before.push(token);
            }
        }
        if self.ahead.as_ref().is_some_and(|ahead| ahead.rows == table_rows) {
            return Ok(());
        }
        if let Some(ahead) = self.ahead.take() {
            self.retire(ahead.gather);
        }
        let Some((reader, row_reader)) = &mut self.rows else { return Ok(()) };
        let mut gather = RowGather::new();
        self.reads += 1;
        row_reader.start(reader, &table_rows, self.reads << 20, Class::Demand, &mut gather)?;
        self.ahead = Some(Ahead { rows: table_rows, gather });
        Ok(())
    }

    /// Keeps a gather that is given up until its reads have come back: their
    /// buffers return to the reader's pool and their rows go to the cache.
    fn retire(&mut self, gather: RowGather) {
        if gather.pending() > 0 {
            self.stale.push(gather);
        }
    }

    /// Hands the reads of table rows that have completed to their gathers.
    /// Never waits.
    fn poll_rows(&mut self) {
        let Some((reader, row_reader)) = &mut self.rows else { return };
        reader.poll(&mut self.completions);
        for completion in self.completions.drain(..) {
            let mut left = row_reader.accept(&mut self.gather, completion);
            if let (Some(completion), Some(ahead)) = (left.take(), &mut self.ahead) {
                left = row_reader.accept(&mut ahead.gather, completion);
            }
            for stale in &mut self.stale {
                let Some(completion) = left.take() else { break };
                left = row_reader.accept(stale, completion);
            }
        }
        self.stale.retain(|gather| gather.pending() > 0);
    }

    /// Waits for the rows of the n-gram table of the step, following the
    /// route records meanwhile.
    fn await_rows(&mut self) -> Result<(), Error> {
        let started = Instant::now();
        while !self.gather.is_done() {
            self.watch(started, "rows of the n-gram table were read")?;
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
