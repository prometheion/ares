//! Loading a model: from a GGUF file and what the operator asked for to a
//! running [`Runtime`] on the devices, with an account of every decision and
//! of where the memory went.
//!
//! The order is: read the file, look at the devices, decide (which devices,
//! which split, which shares, how many sequences, how much context), log the
//! decision with its reasons, load, log what it took. What the operator named
//! is taken as it is and judged by the devices' ledgers at load; what was
//! left open comes from numbers the devices and the model give (`plan`).
//!
//! # What is estimated
//!
//! The runtime reserves the pages of sequence state before the model's parts
//! are built, and the model has no plan of its sizes that can be asked
//! without a device. So the context that fits, when the operator names none,
//! comes from an estimate of what the model takes (`footprint`): weights
//! from the placement and the tensor table, recurrent state and the scratch
//! of a step from the model's dimensions. After the load the log states what
//! each part really took next to what was estimated.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ares_api::Engine;
use ares_core::units::{Bytes, MIB};
use ares_cuda::{Device, DeviceInfo, Wait};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{
    EmbeddingReport, Options, Placement, Qwen35, Spread, exact_bytes, page_bytes,
};
use ares_model::qwen35::weights::{Matrix, Mixer};
use ares_model::qwen35::{Config, Weights};
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig, Split, required_ram};
use ares_tokenizer::ChatOptions;

use crate::models::{LoadFailure, Loader, Site, Unit};
use crate::plan::{
    self, CELLS_PER_PAGE, DEFAULT_PARALLEL, DEFAULT_WINDOW, Drafter, LoadOptions, Room,
    SINGLE_DEVICE_MIN_CONTEXT, SplitKind,
};

/// Memory of every device that the engine leaves unreserved: for what the
/// driver takes besides the engine's own buffers (its context, the kernels'
/// code) and for whatever else lives on the card.
pub(crate) const VRAM_HEADROOM: u64 = 256 * MIB;

/// The architectures that have an implementation on the devices.
pub const IMPLEMENTED: &[&str] =
    &[ares_model::qwen35::config::ARCHITECTURE, ares_model::qwen4exp::config::ARCHITECTURE];

/// Why a model could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("cannot open the model: {0}")]
    Open(#[from] ares_gguf::Error),
    #[error(
        "the model file declares no architecture (general.architecture), so there is nothing \
         to run it with"
    )]
    NoArchitecture,
    /// The file is a model this build has no device implementation for.
    #[error(
        "the model architecture '{0}' has no implementation on the devices: this build runs \
         {implemented}",
        implemented = IMPLEMENTED.join(", ")
    )]
    Architecture(String),
    /// What the operator asked for cannot be done; the message says why.
    #[error("{0}")]
    Options(String),
    #[error("devices: {0}")]
    Cuda(#[from] ares_cuda::Error),
    #[error("the model cannot be set up: {0}")]
    Model(String),
    #[error("the model could not be loaded onto the devices: {0}")]
    Start(String),
    /// The devices or the RAM limit cannot hold the model next to the other
    /// models of this server, which could be unloaded for it.
    #[error("{0}")]
    NoRoom(String),
}

/// `error`, a lack of memory, as a load next to other models sees it: room
/// that unloading one of them can make.
pub(crate) fn short(site: &Site, error: LoadError) -> LoadError {
    if site.beside { LoadError::NoRoom(error.to_string()) } else { error }
}

/// The site of a model that is loaded alone, with the RAM its options name.
fn alone(options: &LoadOptions) -> Site {
    Site { beside: false, ram: options.ram, ram_is_rest: false }
}

pub(crate) fn refuse(message: impl Into<String>) -> LoadError {
    LoadError::Options(message.into())
}

/// Whether the model's own draft block takes part in speculation: when
/// `--drafters` names `mtp`, or names nothing and the model has one
/// (`has_block`). `--drafters mtp` for a model with none is refused, saying
/// `no_block`, and so is a drafter from a file, which is not built yet.
pub(crate) fn draft_block_takes_part(
    options: &LoadOptions,
    has_block: bool,
    no_block: &str,
) -> Result<bool, LoadError> {
    let Some(list) = &options.drafters else {
        return Ok(has_block);
    };
    for drafter in list {
        if let Drafter::Block(path) = drafter {
            return Err(refuse(format!(
                "--drafters {}: block drafters from a file of their own (DFlash2) are not \
                 built yet",
                path.display()
            )));
        }
    }
    let asked = list.contains(&Drafter::Mtp);
    if asked && !has_block {
        return Err(refuse(format!("--drafters mtp: {no_block}")));
    }
    Ok(asked)
}

/// A device that was opened, and what it had free then.
pub(crate) struct Held {
    pub(crate) device: Device,
    pub(crate) free: u64,
}

/// A model on the devices, being served.
pub struct Loaded {
    runtime: Arc<Runtime>,
    name: String,
    held: Vec<Held>,
    /// The most RAM the engine may use.
    ram: u64,
    /// What every device had free when the model was on it, after the load.
    free_loaded: Vec<u64>,
}

/// What an unload gave back.
#[derive(Debug, Clone, PartialEq)]
pub struct Released {
    /// For every device the model was on: its number, what the model had
    /// taken of it by the device's own report (free before the load less
    /// free after it), and what the device reported as free before and
    /// after the unload.
    pub devices: Vec<DeviceRelease>,
    /// The memory of the process that is in RAM (`VmRSS`), before and after.
    pub rss: Option<(u64, u64)>,
    pub seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRelease {
    pub ordinal: u32,
    pub taken: u64,
    pub free_before: u64,
    pub free_after: u64,
}

/// The bytes of the process that are in RAM, as Linux reports them.
fn resident() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find_map(|line| line.strip_prefix("VmRSS:"))?;
    line.trim().strip_suffix("kB")?.trim().parse::<u64>().ok()?.checked_mul(1024)
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded").field("name", &self.name).finish_non_exhaustive()
    }
}

impl Loaded {
    pub(crate) fn new(runtime: Runtime, name: String, held: Vec<Held>, ram: u64) -> Self {
        let free_loaded =
            held.iter().map(|h| h.device.mem_info().map_or(h.free, |memory| memory.free)).collect();
        Self { runtime: Arc::new(runtime), name, held, ram, free_loaded }
    }

    /// The engine to serve.
    pub fn engine(&self) -> Arc<dyn Engine> {
        self.runtime.clone()
    }

    /// The name the model is served under.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Ends every request that still runs, joins the conductor and the device
    /// threads, and gives the devices back. Returns when all of that is done,
    /// with what the devices report: the log says for every device what came
    /// back, and warns when it is less than the model took.
    pub fn unload(self) -> Released {
        let begun = Instant::now();
        let rss_before = resident();
        let free_before: Vec<_> = self.held.iter().map(|h| h.device.mem_info().ok()).collect();
        self.runtime.shutdown();
        let running = self.runtime.is_running();
        // The device threads have ended and dropped what they held. A model
        // that holds device memory of its own (the slots of experts) frees it
        // with the runtime, whose last handle this is once the requests that
        // ended have let go of theirs.
        let Self { mut runtime, held, name, free_loaded, .. } = self;
        let patience = Instant::now();
        let runtime = loop {
            match Arc::try_unwrap(runtime) {
                Ok(runtime) => break Some(runtime),
                Err(shared) if patience.elapsed() > std::time::Duration::from_secs(5) => {
                    tracing::warn!(
                        model = %name,
                        handles = Arc::strong_count(&shared),
                        "requests still hold the engine: what the model itself holds is freed \
                         when they let go"
                    );
                    drop(shared);
                    break None;
                }
                Err(shared) => {
                    runtime = shared;
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        };
        drop(runtime);
        let mut devices = Vec::new();
        for ((held, before), loaded) in held.iter().zip(free_before).zip(free_loaded) {
            let info = held.device.info();
            match (before, held.device.mem_info()) {
                (Some(before), Ok(after)) => {
                    let release = DeviceRelease {
                        ordinal: info.ordinal,
                        taken: held.free.saturating_sub(loaded),
                        free_before: before.free,
                        free_after: after.free,
                    };
                    let back = release.free_after.saturating_sub(release.free_before);
                    tracing::info!(
                        "device {} ({}): {} released of the {} the model took, {} free",
                        info.ordinal,
                        info.name,
                        Bytes(back),
                        Bytes(release.taken),
                        Bytes(after.free),
                    );
                    // Allocation granularity aside, all of it comes back.
                    if back + 16 * MIB < release.taken {
                        tracing::warn!(
                            "device {}: {} of what the model took did not come back",
                            info.ordinal,
                            Bytes(release.taken - back)
                        );
                    }
                    devices.push(release);
                }
                (_, Err(error)) => tracing::warn!("device {}: {error}", info.ordinal),
                (None, _) => {}
            }
        }
        // These handles go last; a device no other model is on closes with
        // its driver context.
        drop(held);
        let rss = rss_before.zip(resident());
        let seconds = begun.elapsed().as_secs_f64();
        tracing::info!(
            model = %name,
            running,
            "the engine has stopped in {:.0} ms: threads joined, devices released{}",
            seconds * 1e3,
            match rss {
                Some((before, after)) =>
                    format!(", {} of the process in RAM ({} before)", Bytes(after), Bytes(before)),
                None => String::new(),
            }
        );
        Released { devices, rss, seconds }
    }
}

impl Unit for Loaded {
    fn engine(&self) -> Arc<dyn Engine> {
        self.runtime.clone()
    }

    fn ram(&self) -> u64 {
        self.ram
    }

    fn unload(self: Box<Self>) {
        Loaded::unload(*self);
    }
}

/// Loads models onto the devices: the server's [`Loader`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DeviceLoader;

impl Loader for DeviceLoader {
    fn load(
        &self,
        preset: &crate::preset::Preset,
        site: &Site,
    ) -> Result<Box<dyn Unit>, LoadFailure> {
        let mut options = preset.options.load.clone();
        options.ram = site.ram;
        match load_at(&options, site) {
            Ok(loaded) => Ok(Box::new(loaded)),
            Err(LoadError::NoRoom(why)) => Err(LoadFailure::NoRoom(why)),
            Err(error) => Err(LoadFailure::Failed(error.to_string())),
        }
    }
}

/// What the parts of a model take on one device, by estimate.
#[derive(Debug, Clone, Copy, Default)]
struct Footprint {
    weights: u64,
    /// The scratch of a step and the small buffers.
    step: u64,
    per_sequence: u64,
    per_page: u64,
    /// The exact K/V entries of the formats with a tile mean, for the
    /// sequences that run at once: a pool of its own, whatever the context.
    exact: u64,
}

fn matrix_bytes(m: &Matrix<'_>) -> u64 {
    m.bytes().len() as u64
}

fn part_of(bytes: u64, have: usize, of: usize) -> u64 {
    (bytes * have as u64).div_ceil(of.max(1) as u64)
}

/// The stored bytes of block `layer`: its mixer and its feed-forward part.
fn block_bytes(w: &Weights<'_>, layer: usize) -> (u64, u64) {
    let l = &w.layers[layer];
    let ffn = matrix_bytes(&l.block.ffn_gate)
        + matrix_bytes(&l.block.ffn_up)
        + matrix_bytes(&l.block.ffn_down);
    let mixer = match &l.mixer {
        Mixer::Recurrent(r) => {
            matrix_bytes(&r.qkv)
                + matrix_bytes(&r.gate)
                + matrix_bytes(&r.beta)
                + matrix_bytes(&r.alpha)
                + matrix_bytes(&r.out)
        }
        Mixer::Attention(a) => {
            matrix_bytes(&a.q) + matrix_bytes(&a.k) + matrix_bytes(&a.v) + matrix_bytes(&a.out)
        }
    };
    (mixer, ffn)
}

/// What `placement` puts on each device, estimated without a device.
///
/// The weights follow the loader (`qwen35::device::load`): a device's slice of
/// every block, the embedding rows the draft chain picks, its rows of the head,
/// the draft block.
/// The scratch of a step follows `Part::open` with the widths of the whole
/// model, which a slice does not exceed. Both are estimates: the loader
/// aligns and rearranges, and the part has more small buffers than are
/// counted here, for which a margin is added.
fn footprint(
    c: &Config,
    w: &Weights<'_>,
    placement: &Placement,
    options: &Options,
    parallel: u32,
) -> Vec<Footprint> {
    let r = &c.recurrent;
    let a = &c.attention;
    let (d, f) = (c.n_embd as u64, c.n_ff as u64);
    let (kd, vd, cd) = (r.key_dim() as u64, r.value_dim() as u64, r.conv_dim() as u64);
    let (hq, hkv) = ((a.n_head * a.head_dim) as u64, (a.n_head_kv * a.head_dim) as u64);
    let rows = u64::from(options.max_rows);
    let widest = d.max(vd).max(hq).max(f);
    let per_row = 4
        * (3 * d
            + cd.max(2 * hq).max(f)
            + vd.max(hkv).max(f)
            + 2 * r.n_v_heads as u64
            + kd.max(hq)
            + (kd + vd).max(hq)
            + vd.max(hq)
            + 2 * hkv)
        + 4 * a.rope.dim as u64
        + 3 * widest;
    let values_per_key = (r.n_v_heads / r.n_k_heads.max(1)) as u64;

    placement
        .devices
        .iter()
        .enumerate()
        .map(|(index, plan)| {
            let mut weights = 0u64;
            let mut per_sequence = 0u64;
            for (layer, slice) in plan.blocks.iter().enumerate() {
                let Some(slice) = slice else { continue };
                let (mixer, ffn) = block_bytes(w, layer);
                let heads = slice.heads.len();
                weights += part_of(ffn, slice.ffn.len(), c.n_ff);
                if c.is_attention(layer) {
                    weights += part_of(mixer, heads, a.n_head_kv);
                } else {
                    weights += part_of(mixer, heads, r.n_k_heads);
                    // The convolution's window and the delta-rule matrices of
                    // the heads this device computes.
                    let keys = heads as u64 * r.head_k_dim as u64;
                    let value_heads = heads as u64 * values_per_key;
                    let values = value_heads * r.head_v_dim as u64;
                    let conv = (2 * keys + values) * (r.conv_kernel as u64).saturating_sub(1);
                    let delta = value_heads * (r.head_k_dim * r.head_v_dim) as u64;
                    per_sequence += (conv + delta) * options.state.bytes();
                }
            }
            // The embedding table lies in host memory. The device that embeds
            // keeps the rows of the tokens the draft chain picks, as a matrix
            // of their own unless the head is the table, whose rows they
            // are then.
            if let Some(draft) =
                placement.draft.as_ref().filter(|draft| draft.device == index && w.output.is_some())
            {
                weights +=
                    part_of(matrix_bytes(&w.token_embd), draft.head_rows as usize, c.n_vocab);
            }
            weights += part_of(matrix_bytes(w.head()), plan.head.len(), c.n_vocab);
            if let (Some(_), Some(nextn)) =
                (placement.draft.as_ref().filter(|draft| draft.device == index), &w.nextn)
            {
                let at = &nextn.attention;
                weights += matrix_bytes(&nextn.block.ffn_gate)
                    + matrix_bytes(&nextn.block.ffn_up)
                    + matrix_bytes(&nextn.block.ffn_down)
                    + matrix_bytes(&at.q)
                    + matrix_bytes(&at.k)
                    + matrix_bytes(&at.v)
                    + matrix_bytes(&at.out)
                    + matrix_bytes(&nextn.eh_proj);
            }
            let logits = u64::from(options.logit_rows) * 4 * (d + plan.head.len() as u64);
            Footprint {
                weights,
                // The margin: alignment of the arena, the sampler, records,
                // attention plans, links, the draft block's buffers.
                step: rows * per_row + logits + weights / 64 + 64 * MIB,
                per_sequence,
                per_page: page_bytes(c, placement, index, options),
                exact: exact_bytes(c, placement, index, options, parallel),
            }
        })
        .collect()
}

/// Where the token embedding of a device lies, for the report at start.
fn embedding_note(e: &EmbeddingReport) -> String {
    if !e.embeds {
        return String::new();
    }
    let on_device = match (e.kept_rows, e.shared_with_head) {
        (0, _) => "no rows on the device".to_owned(),
        (rows, true) => format!("{rows} rows on the device, the first rows of the head"),
        (rows, false) => format!("{rows} rows on the device ({})", Bytes(e.kept_bytes)),
    };
    match e.host_bytes {
        0 => format!(", token embedding {} on the device ({on_device})", Bytes(e.table_bytes)),
        host => format!(
            ", token embedding {} in page-locked RAM ({} locked, with the rows of a step; \
             {on_device}; {} of the step scratch for the rows the host stages)",
            Bytes(e.table_bytes),
            Bytes(host),
            Bytes(e.staged_bytes)
        ),
    }
}

/// The link of a device to its slot, as Linux reports it.
fn pci_link(info: &DeviceInfo) -> Option<String> {
    let base = format!("/sys/bus/pci/devices/{}", info.pci);
    let read = |name: &str| {
        std::fs::read_to_string(format!("{base}/{name}")).ok().map(|s| s.trim().to_owned())
    };
    Some(format!("{} x{}", read("current_link_speed")?, read("current_link_width")?))
}

/// The bound on a device's memory bandwidth that it reports, in bits per
/// second: only its ratio between devices is used.
pub(crate) fn bandwidth(info: &DeviceInfo) -> f64 {
    f64::from(info.memory_bus_width_bits) * f64::from(info.memory_clock_khz) * 1e3
}

pub(crate) fn describe(held: &Held) {
    let info = held.device.info();
    tracing::info!(
        "device {}: {}, compute capability {}.{}, {} of {} free, memory interface {} bit at {} \
         MHz, PCI {}{}",
        info.ordinal,
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        Bytes(held.free),
        Bytes(info.total_memory),
        info.memory_bus_width_bits,
        info.memory_clock_khz / 1000,
        info.pci,
        pci_link(info).map(|link| format!(" ({link})")).unwrap_or_default(),
    );
}

/// Says how the devices of a split reach each other. This is the rule of
/// `ares_cuda::Mesh::new`: the direct route when every pair has peer access.
pub(crate) fn describe_links(held: &[Held]) -> Result<(), LoadError> {
    if held.len() < 2 {
        return Ok(());
    }
    let mut all = true;
    for a in held {
        for b in held {
            let (from, to) = (a.device.ordinal(), b.device.ordinal());
            if from != to {
                let (access, rank) = ares_cuda::peer_access(from, to)?;
                all &= access;
                tracing::info!(
                    "link {from} -> {to}: {}",
                    if access {
                        format!("peer access (the driver's rank of the link: {rank})")
                    } else {
                        "no peer access".to_owned()
                    }
                );
            }
        }
    }
    if all {
        tracing::info!("the devices exchange directly, from device to device");
    } else {
        tracing::warn!(
            "the devices exchange through page-locked host memory, which is slower for a tensor \
             split: {} names a driver library with peer access, if the system has one",
            ares_cuda::LIBRARY_ENV
        );
    }
    Ok(())
}

pub(crate) fn join<T: ToString>(items: impl IntoIterator<Item = T>, with: &str) -> String {
    items.into_iter().map(|item| item.to_string()).collect::<Vec<_>>().join(with)
}

fn shares_text(shares: &[f64]) -> String {
    join(shares.iter().map(|share| format!("{share:.3}")), " / ")
}

/// The token ids of a draft vocabulary file: one per line, `#` for comments.
fn read_vocabulary(path: &Path) -> Result<Arc<[u32]>, LoadError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| refuse(format!("cannot read the draft vocabulary {}: {e}", path.display())))?;
    let mut tokens = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        tokens.push(line.parse::<u32>().map_err(|_| {
            refuse(format!(
                "draft vocabulary {}, line {}: '{line}' is not a token id",
                path.display(),
                number + 1
            ))
        })?);
    }
    if tokens.is_empty() {
        return Err(refuse(format!("the draft vocabulary {} is empty", path.display())));
    }
    Ok(tokens.into())
}

/// The placement of `spread`, with the draft block where `Qwen35::new` puts
/// it, to estimate with.
fn place(
    c: &Config,
    w: &Weights<'_>,
    spread: &Spread,
    options: &Options,
) -> Result<Placement, LoadError> {
    let model = |e: ares_model::qwen35::device::PlacementError| LoadError::Model(e.to_string());
    let placement = match spread {
        Spread::Single => Placement::single(c),
        Spread::Tensor(shares) => Placement::tensor(c, shares).map_err(model)?,
        Spread::TensorByHalf(shares, halves) => {
            Placement::tensor_by_half(c, shares, halves).map_err(model)?
        }
        Spread::Layer(shares) => Placement::layer(c, w, shares).map_err(model)?,
    };
    if c.n_nextn == 1 && options.window_rows > 1 && placement.split != Split::Layer {
        placement.with_draft(c, options.draft_vocabulary.as_deref()).map_err(model)
    } else {
        Ok(placement)
    }
}

fn rooms(held: &[Held], footprint: &[Footprint], headroom: u64) -> Vec<Room> {
    held.iter()
        .zip(footprint)
        .map(|(held, f)| Room {
            usable: held.free.saturating_sub(headroom),
            fixed: f.weights + f.step + f.exact,
            per_sequence: f.per_sequence,
            per_page: f.per_page,
        })
        .collect()
}

/// Opens the devices the operator named, or all the driver reports, and says
/// in the log what each is.
pub(crate) fn hold_devices(options: &LoadOptions) -> Result<(Vec<u32>, Vec<Held>), LoadError> {
    let known = ares_cuda::devices()?;
    if known.is_empty() {
        return Err(refuse("the driver reports no CUDA device"));
    }
    if let Ok((major, minor)) = ares_cuda::driver_version() {
        tracing::info!(
            "CUDA driver {major}.{minor}, {} device(s){}",
            known.len(),
            match std::env::var_os(ares_cuda::LIBRARY_ENV) {
                Some(path) =>
                    format!(", library {} ({})", path.to_string_lossy(), ares_cuda::LIBRARY_ENV),
                None => String::new(),
            }
        );
    }
    let ordinals: Vec<u32> = match &options.devices {
        Some(list) => {
            for (i, ordinal) in list.iter().enumerate() {
                if !known.iter().any(|info| info.ordinal == *ordinal) {
                    return Err(refuse(format!(
                        "--devices: there is no device {ordinal}; the driver reports {}",
                        join(known.iter().map(|info| info.ordinal), ", ")
                    )));
                }
                if list[..i].contains(ordinal) {
                    return Err(refuse(format!("--devices names device {ordinal} twice")));
                }
            }
            list.clone()
        }
        None => known.iter().map(|info| info.ordinal).collect(),
    };
    if ordinals.is_empty() {
        return Err(refuse("--devices names no device"));
    }
    let mut held = Vec::new();
    for ordinal in &ordinals {
        let device = Device::open(*ordinal, Wait::Block)?;
        let free = device.mem_info()?.free;
        held.push(Held { device, free });
        describe(&held[held.len() - 1]);
    }
    Ok((ordinals, held))
}

/// The most host RAM the engine may use: what the operator named, or half of
/// what the machine has available now. `required` is what the engine needs to
/// run at all, its step buffers; `nothing` says what is not spared when the
/// machine reports no available memory.
pub(crate) fn ram_limit(
    options: &LoadOptions,
    site: &Site,
    required: u64,
    nothing: &str,
) -> Result<(u64, String), LoadError> {
    ram_limit_of(options, site, &[("its step buffers", required)], nothing)
}

/// [`ram_limit`] for an engine that must page-lock more than its step
/// buffers to run: `needs` is all of it, by what it is for. A limit below
/// their sum is refused, and the refusal names every share.
fn ram_limit_of(
    options: &LoadOptions,
    site: &Site,
    needs: &[(&str, u64)],
    nothing: &str,
) -> Result<(u64, String), LoadError> {
    let required: u64 = needs.iter().map(|(_, bytes)| bytes).sum();
    let available = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| plan::parse_available_ram(&text));
    let (ram, why_ram) = match options.ram {
        Some(ram) if site.ram_is_rest => {
            (ram, "what the server's limit has left next to the loaded models".to_owned())
        }
        Some(ram) => (ram, "as asked".to_owned()),
        None => match available {
            Some(available) => (
                plan::default_ram(Some(available)).max(required),
                format!(
                    "half of the {} the machine has available now; --ram changes it",
                    Bytes(available)
                ),
            ),
            None => (
                required,
                format!("the machine reports no available memory, so {nothing}; --ram sets it"),
            ),
        },
    };
    if ram < required {
        let error = refuse(plan::too_little_ram(ram, needs));
        // What is left of the server's limit grows when a model is unloaded;
        // a number the operator named for this model does not.
        return Err(if site.ram_is_rest { short(site, error) } else { error });
    }
    Ok((ram, why_ram))
}

/// Loads the model of `options` and starts the runtime over it. Takes as
/// long as the weights take to reach the devices; call it from a thread that
/// may block.
pub fn load(options: &LoadOptions) -> Result<Loaded, LoadError> {
    load_at(options, &alone(options))
}

/// [`load`] at `site`: next to other models of this server, a model that
/// finds too little room is not loaded with what there is but refused with
/// [`LoadError::NoRoom`], before its weights are read, so that one of the
/// others can be unloaded for it.
pub fn load_at(options: &LoadOptions, site: &Site) -> Result<Loaded, LoadError> {
    let begun = Instant::now();
    let gguf = Gguf::open(&options.model)?;
    let architecture = gguf.architecture().map(str::to_owned);
    tracing::info!(
        path = %options.model.display(),
        architecture = architecture.as_deref().unwrap_or("(none declared)"),
        size = %Bytes(gguf.file_size()),
        parts = gguf.n_parts(),
        tensors = gguf.tensors().len(),
        "opened the model file"
    );
    let architecture = architecture.ok_or(LoadError::NoArchitecture)?;
    match architecture.as_str() {
        ares_model::qwen35::config::ARCHITECTURE => load_qwen35(gguf, options, site, begun),
        ares_model::qwen4exp::config::ARCHITECTURE => {
            crate::qwen4exp::load(gguf, options, site, begun)
        }
        _ => Err(LoadError::Architecture(architecture)),
    }
}

/// Loads a `qwen35` model: dense, all of it on the devices.
fn load_qwen35(
    gguf: Gguf,
    options: &LoadOptions,
    site: &Site,
    begun: Instant,
) -> Result<Loaded, LoadError> {
    let config = Config::from_gguf(&gguf).map_err(|e| LoadError::Model(e.to_string()))?;
    let weights = Weights::load(&gguf, &config).map_err(|e| LoadError::Model(e.to_string()))?;
    tracing::info!(
        "model: {} blocks of width {}, {} tokens, trained for a context of {}, {}",
        config.n_layer,
        config.n_embd,
        config.n_vocab,
        config.n_ctx,
        if config.n_nextn == 1 { "with a draft block" } else { "no draft block" },
    );

    // What can be refused before a device is touched.
    let kv = options.kv.unwrap_or(Options::default().kv);
    let parallel = options.parallel.unwrap_or(DEFAULT_PARALLEL);
    if parallel == 0 {
        return Err(refuse("--parallel must be at least 1"));
    }
    // The model drafters that take part: all the model has unless the operator
    // names them. Nothing speculates without one, the lookup in earlier text
    // included.
    let mtp = draft_block_takes_part(
        options,
        config.n_nextn == 1,
        "this model file has no draft block (multi-token prediction)",
    )?;
    // The model's draft block is not built for a layer split (the block, the
    // device that embeds and the first rows of the head must be one device).
    let layer = options.split == Some(SplitKind::Layer);
    if mtp && layer && options.drafters.is_some() {
        return Err(refuse(
            "--drafters mtp: the model's draft block is not built for a layer split; use a \
             single device or a tensor split",
        ));
    }
    let blocked_by_layer = mtp && layer;
    let mtp = mtp && !layer;
    let window = options.window.unwrap_or(DEFAULT_WINDOW);
    if options.window.is_some() && !mtp {
        return Err(refuse(if blocked_by_layer {
            "--window: the model's draft block is not built for a layer split, so no window is \
             reserved"
        } else {
            "--window without a model drafter does nothing"
        }));
    }
    let vocabulary = match &options.draft_vocabulary {
        Some(_) if !mtp => {
            return Err(refuse(
                "a draft vocabulary is for the model's draft block, and no drafter of that kind \
                 takes part",
            ));
        }
        Some(path) => Some(read_vocabulary(path)?),
        None => None,
    };
    if options.context == Some(0) {
        return Err(refuse("--context must be at least 1"));
    }

    // The devices.
    let (ordinals, mut held) = hold_devices(options)?;

    // The model's settings that do not depend on the placement.
    let window_rows = if mtp { parallel.saturating_mul(window) } else { 0 };
    let part_options = Options {
        precision: options.precision,
        kv,
        logit_rows: window_rows.max(parallel),
        window_rows,
        draft_vocabulary: vocabulary.clone(),
        ..Options::default()
    };
    if part_options.max_rows < window_rows {
        return Err(refuse(format!(
            "{parallel} sequences with windows of {window} rows are {window_rows} rows in a \
             step, more than the {} a step has: lower --parallel or --window",
            part_options.max_rows
        )));
    }
    let headroom = VRAM_HEADROOM;
    let smallest = options.context.unwrap_or(SINGLE_DEVICE_MIN_CONTEXT.min(config.n_ctx as u32));

    // Which devices, and how the model is spread over them.
    let holds = |held: &[Held], spread: &Spread| -> Result<bool, LoadError> {
        let placement = place(&config, &weights, spread, &part_options)?;
        let footprint = footprint(&config, &weights, &placement, &part_options, parallel);
        let pages = plan::pages_that_fit(&rooms(held, &footprint, headroom), parallel);
        Ok(pages >= plan::pool_pages(smallest, parallel))
    };
    let split = match options.split {
        Some(SplitKind::Single) if options.devices.as_ref().is_some_and(|list| list.len() > 1) => {
            return Err(refuse(
                "--split single takes one device and --devices names several: name one, or \
                 choose tensor or layer",
            ));
        }
        Some(kind @ (SplitKind::Tensor | SplitKind::Layer)) if held.len() < 2 => {
            return Err(refuse(format!(
                "--split {} needs at least two devices and there is one",
                kind.name()
            )));
        }
        Some(kind) => kind,
        None if held.len() == 1 => SplitKind::Single,
        None => {
            // One device, if one holds the model with a useful context.
            let mut alone = false;
            for candidate in &held {
                alone |= holds(std::slice::from_ref(candidate), &Spread::Single)?;
            }
            if alone { SplitKind::Single } else { SplitKind::Tensor }
        }
    };
    let why_split = match (options.split, split) {
        (Some(_), _) => "as asked",
        (None, SplitKind::Single) if ordinals.len() == 1 => "one device to place it on",
        (None, SplitKind::Single) => "one device holds the model and its context",
        (None, _) => "no single device holds the model and its context",
    };
    if split == SplitKind::Single {
        if options.shares.is_some() {
            return Err(refuse("--shares is for a split over several devices, and there is one"));
        }
        if held.len() > 1 {
            // The first device, in the order given, that holds the model; the
            // first one if none does, so that its ledger says what is missing.
            let mut chosen = 0;
            for (i, candidate) in held.iter().enumerate() {
                if holds(std::slice::from_ref(candidate), &Spread::Single)? {
                    chosen = i;
                    break;
                }
            }
            let keep = held.swap_remove(chosen);
            held = vec![keep];
        }
    }
    let (spread, why_shares) = match split {
        SplitKind::Single => (Spread::Single, ""),
        SplitKind::Tensor | SplitKind::Layer => {
            let (shares, why) = match &options.shares {
                Some(shares) if shares.len() != held.len() => {
                    return Err(refuse(format!(
                        "--shares has {} values for {} devices",
                        shares.len(),
                        held.len()
                    )));
                }
                Some(shares) => {
                    let total: f64 = shares.iter().sum();
                    if shares.iter().any(|s| !s.is_finite() || *s <= 0.0) || total <= 0.0 {
                        return Err(refuse("--shares must be positive numbers"));
                    }
                    (shares.iter().map(|s| s / total).collect::<Vec<_>>(), "as asked")
                }
                None => {
                    let wanted = plan::shares_by_bandwidth(
                        &held.iter().map(|h| bandwidth(h.device.info())).collect::<Vec<_>>(),
                    );
                    // What a device can take of a model that is shared out
                    // evenly in everything: weights, state and context.
                    let whole = whole(&config, &weights, &part_options, parallel, smallest);
                    let most: Vec<f64> = held
                        .iter()
                        .map(|h| {
                            let usable = h.free.saturating_sub(headroom + whole.per_device);
                            usable as f64 / whole.shared.max(1) as f64
                        })
                        .collect();
                    match plan::fit_shares(&wanted, &most) {
                        Some(shares) if shares == wanted => {
                            (shares, "by the memory bandwidth the devices report")
                        }
                        Some(shares) => (
                            shares,
                            "by the memory bandwidth the devices report, lowered where a \
                             device's memory is too small for its share",
                        ),
                        None => {
                            let error = refuse(format!(
                                "the model does not fit on the devices: it needs about {} with \
                                 a context of {smallest} tokens, and they have {} free",
                                Bytes(whole.shared + whole.per_device * held.len() as u64),
                                Bytes(held.iter().map(|h| h.free.saturating_sub(headroom)).sum()),
                            ));
                            return Err(short(site, error));
                        }
                    }
                }
            };
            match split {
                SplitKind::Tensor => (Spread::Tensor(shares), why),
                _ => (Spread::Layer(shares), why),
            }
        }
    };
    describe_links(&held)?;
    let placement = place(&config, &weights, &spread, &part_options)?;
    if vocabulary.is_some() && placement.draft.is_none() {
        return Err(refuse(
            "a draft vocabulary is for the model's draft block, which is not built for a layer \
             split",
        ));
    }
    let footprint = footprint(&config, &weights, &placement, &part_options, parallel);
    let fit = plan::pages_that_fit(&rooms(&held, &footprint, headroom), parallel);

    // Context: tokens of state for all sequences together.
    let model_context = config.n_ctx as u32;
    let taken_text = || {
        join(
            held.iter().zip(&footprint).map(|(h, f)| {
                format!(
                    "device {} has {} free and the model takes {} without K and V",
                    h.device.ordinal(),
                    Bytes(h.free),
                    Bytes(f.weights + f.step + f.exact + f.per_sequence * u64::from(parallel))
                )
            }),
            ", ",
        )
    };
    let (context, why_context) = match options.context {
        // Next to other models the estimate decides before the weights are
        // read; alone the devices' ledgers do at load, as ever.
        Some(context) if site.beside && fit < plan::pool_pages(context, parallel) => {
            return Err(LoadError::NoRoom(format!(
                "the devices cannot hold the model with its context of {context} tokens next to \
                 what is loaded (by estimate: {})",
                taken_text()
            )));
        }
        Some(context) => (context, "as asked"),
        None => {
            let context = plan::default_context(fit, parallel, model_context);
            // Next to other models a context below the least a device is
            // chosen for is no room: one of them makes way.
            if site.beside && context < smallest {
                return Err(LoadError::NoRoom(format!(
                    "the devices have room for a context of {context} tokens next to what is \
                     loaded, less than the {smallest} a model is loaded for (by estimate: {})",
                    taken_text()
                )));
            }
            if context == 0 {
                return Err(refuse(format!(
                    "the devices have no room for sequence state next to the model (by \
                     estimate: {}); name fewer sequences with --parallel, another split, or a \
                     context with --context to try it anyway",
                    join(
                        held.iter().zip(&footprint).map(|(h, f)| format!(
                            "device {} has {} free and the model takes {}",
                            h.device.ordinal(),
                            Bytes(h.free),
                            Bytes(
                                f.weights + f.step + f.exact + f.per_sequence * u64::from(parallel)
                            )
                        )),
                        ", "
                    )
                )));
            }
            if context < model_context.saturating_mul(parallel) {
                (context, "what the devices' free memory holds, by estimate")
            } else {
                (context, "the model's own context for every sequence")
            }
        }
    };
    let longest = context.min(model_context);
    let pages = plan::pool_pages(context, parallel);

    tracing::info!(
        "plan: {} on device{} {} ({why_split}){}",
        match split {
            SplitKind::Single => "everything",
            SplitKind::Tensor => "tensor split",
            SplitKind::Layer => "layer split",
        },
        if held.len() == 1 { "" } else { "s" },
        join(held.iter().map(|h| h.device.ordinal()), ", "),
        match &spread {
            Spread::Tensor(shares) | Spread::Layer(shares) =>
                format!(", shares {} ({why_shares})", shares_text(shares)),
            _ => String::new(),
        }
    );
    tracing::info!(
        "plan: context of {context} tokens for all sequences together ({why_context}), {pages} \
         pages; a request holds at most {longest} tokens; {parallel} sequence{} at once ({})",
        if parallel == 1 { "" } else { "s" },
        if options.parallel.is_some() { "as asked" } else { "the default" },
    );
    tracing::info!(
        "plan: K in {}, V in {}; products {}; {}",
        plan::format_name(kv.0),
        plan::format_name(kv.1),
        match options.precision {
            ares_cuda::Precision::Fast => "fast (8-bit activations for a few rows)",
            ares_cuda::Precision::Exact => "exact (f16 on the tensor cores for every product)",
        },
        match (mtp, &placement.draft) {
            (false, _) if blocked_by_layer => "no speculation: the model's draft block is not \
                 built for a layer split, and earlier text alone is not a drafter"
                .to_owned(),
            (false, _) => match options.drafters {
                Some(_) => "no drafters (--drafters none): no speculation".to_owned(),
                None => "no model drafter (the file has no draft block): no speculation".to_owned(),
            },
            (true, Some(draft)) => format!(
                "drafts from the model's draft block on device {} ({}) and from earlier text, \
                 windows of up to {window} rows; whose drafts and how many are chosen for every \
                 step from the price of a step, measured at load, and the acceptance, measured \
                 while running",
                held[draft.device].device.ordinal(),
                match &vocabulary {
                    Some(list) => format!("a vocabulary of {} tokens", list.len()),
                    None => format!("the {} tokens whose head rows are there", draft.head_rows),
                }
            ),
            (true, None) => {
                "no speculation: the model has no draft block on a device of this placement"
                    .to_owned()
            }
        }
    );
    for (h, f) in held.iter().zip(&footprint) {
        tracing::info!(
            "estimate, device {}: weights {}, step scratch and small buffers {}, recurrent \
             state {} per sequence, K and V {} per 1000 tokens; with this plan {} of {} free",
            h.device.ordinal(),
            Bytes(f.weights),
            Bytes(f.step),
            Bytes(f.per_sequence),
            Bytes(f.per_page * 1000 / u64::from(CELLS_PER_PAGE)),
            Bytes(
                f.weights
                    + f.step
                    + f.exact
                    + f.per_sequence * u64::from(parallel)
                    + f.per_page * u64::from(pages)
            ),
            Bytes(h.free),
        );
    }

    // Load.
    let name = options.name.clone().unwrap_or_else(|| crate::preset::id_of_file(&options.model));
    let chat = ChatOptions { variables: options.chat_variables.clone(), ..ChatOptions::default() };
    drop(weights);
    let model = Arc::new(
        Qwen35::new(gguf, name.clone(), &spread, part_options, longest)
            .map_err(|e| LoadError::Model(e.to_string()))?
            .with_chat_options(chat),
    );
    if !model.tokenizer().has_chat_template() {
        tracing::warn!(
            "the model file has no chat template: /v1/chat/completions will refuse requests, \
             /v1/completions works"
        );
    }
    // Host RAM: one limit for the engine. What it needs to run is taken from
    // it first: its step buffers and, where the token embedding table lies in
    // host memory, the table and the rows of a step. A limit below that is
    // refused: the table does not go back to a device, whose memory is what
    // bounds the context. The rest is the budget of the checkpoints of
    // recurrent state (the only other user today), which are locked as they
    // come into use.
    let required = required_ram(&*model, parallel);
    let table = model.pinned_ram();
    let mut needs = vec![("its step buffers", required - table)];
    if table > 0 {
        needs.push((
            "the token embedding table and the rows of a step, which lie in page-locked RAM \
             and not on a device",
            table,
        ));
    }
    let (ram, why_ram) = ram_limit_of(options, site, &needs, "none is spared for checkpoints")?;
    let checkpoint = model.prefix_cache().map_or(0, |cache| cache.bytes());
    let runtime_config = RuntimeConfig {
        pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: parallel,
        window_rows: if mtp { window } else { 1 },
        checkpoint_budget: ram - required,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: headroom,
        ram_limit: ram,
        ..GpuConfig::new(&runtime_config, &*model)
    };
    tracing::info!(
        "plan: RAM, at most {} for the engine ({why_ram}): {} for its step buffers{}, the rest \
         for checkpoints of recurrent state, {} each over all devices, {}; the memory is \
         page-locked as checkpoints are taken",
        Bytes(ram),
        Bytes(required - table),
        match table {
            0 => String::new(),
            bytes => format!(
                ", {} for the token embedding table and the rows of a step (locked at load)",
                Bytes(bytes)
            ),
        },
        Bytes(checkpoint),
        match checkpoint {
            0 => "none for this model".to_owned(),
            bytes => format!("room for {}", (ram - required) / bytes),
        }
    );
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let devices: Vec<Device> = held.iter().map(|h| h.device.clone()).collect();
    let runtime = Runtime::start_gpu(model.clone(), devices, runtime_config, gpu).map_err(|e| {
        let error = e.to_string();
        // A context that was estimated to fit and did not is the server's
        // mistake, and the operator should know the way around it.
        let short_of_memory = error.to_lowercase().contains("memory");
        let error = LoadError::Start(if options.context.is_none() && short_of_memory {
            format!(
                "{error}. The context of {context} tokens was chosen from an estimate of \
                     what the model takes; name a smaller one with --context"
            )
        } else {
            error
        });
        // The estimate was too low: next to other models that is no room.
        if short_of_memory { short(site, error) } else { error }
    })?;
    tracing::info!(
        "loaded in {:.1} s ({:.1} s since the start)",
        loading.elapsed().as_secs_f64(),
        begun.elapsed().as_secs_f64()
    );
    if let Some(calibration) = runtime.snapshot().and_then(|snapshot| snapshot.calibration) {
        let table = |points: &[(u32, f64)]| {
            points.iter().map(|(n, ms)| format!("{n}: {ms:.1}")).collect::<Vec<_>>().join(", ")
        };
        tracing::info!(
            "the price of a step was measured in {:.1} s ({} steps), in ms by rows: {}{}",
            calibration.seconds,
            calibration.steps,
            table(&calibration.step_ms),
            if calibration.chain_ms.is_empty() {
                String::new()
            } else {
                format!("; the draft block's chain, by links: {}", table(&calibration.chain_ms))
            }
        );
    }

    // What it took.
    for ((h, report), f) in held.iter().zip(model.reports()).zip(&footprint) {
        let state = report.state_bytes_per_slot * u64::from(parallel);
        let kv_bytes = report.page_bytes * u64::from(pages);
        let draft = report.draft.as_ref().map_or(0, |d| d.carry + d.scratch + d.records);
        let exact = report.exact_bytes + report.draft.as_ref().map_or(0, |d| d.exact);
        let known = report.weight_bytes
            + state
            + kv_bytes
            + exact
            + report.scratch_bytes
            + report.kept_bytes
            + draft;
        let free = h.device.mem_info()?.free;
        let taken = h.free.saturating_sub(free);
        tracing::info!(
            "device {} ({}): weights {} in {:.1} s{}{}, recurrent state {} ({} per sequence), K \
             and V {} ({} per 1000 tokens), exact K/V entries {}, step scratch {}, inputs kept \
             for draft windows {}, draft block buffers {}, everything else {}; taken {}, free {}",
            h.device.ordinal(),
            report.device,
            Bytes(report.weight_bytes),
            report.upload.as_secs_f64(),
            report
                .draft
                .as_ref()
                .map(|d| format!(" (of which the draft block {})", Bytes(d.weights)))
                .unwrap_or_default(),
            embedding_note(&report.embedding),
            Bytes(state),
            Bytes(report.state_bytes_per_slot),
            Bytes(kv_bytes),
            Bytes(report.page_bytes * 1000 / u64::from(CELLS_PER_PAGE)),
            Bytes(exact),
            Bytes(report.scratch_bytes),
            Bytes(report.kept_bytes),
            Bytes(draft),
            Bytes(taken.saturating_sub(known)),
            Bytes(taken),
            Bytes(free),
        );
        tracing::info!(
            "device {}: the ledger holds {} in {} reservations: {}; {} of what the device took \
             are in no reservation (the sampler's buffers, allocation granularity)",
            h.device.ordinal(),
            Bytes(report.ledger.iter().map(|(_, bytes)| bytes).sum()),
            report.ledger.len(),
            join(
                report.ledger.iter().map(|(name, bytes)| format!("{name} {}", Bytes(*bytes))),
                "; "
            ),
            Bytes(taken.saturating_sub(report.ledger.iter().map(|(_, bytes)| bytes).sum())),
        );
        let fixed = f.weights + f.step + f.exact + f.per_sequence * u64::from(parallel);
        let really = taken.saturating_sub(kv_bytes);
        tracing::info!(
            "device {}: the model without K and V took {} (estimated {}); {} are free beyond \
             the {} left unreserved, room for about {} more tokens of context",
            h.device.ordinal(),
            Bytes(really),
            Bytes(fixed),
            Bytes(free.saturating_sub(headroom)),
            Bytes(headroom),
            (free.saturating_sub(headroom) / report.page_bytes.max(1))
                .saturating_mul(u64::from(CELLS_PER_PAGE)),
        );
    }
    tracing::info!(
        "context: {context} tokens of sequence state, all of it in VRAM (state is not moved to \
         RAM yet, so a context that does not fit is refused at load)"
    );
    Ok(Loaded::new(runtime, name, held, ram))
}

/// What the whole model takes when it is shared out: the part that is
/// divided among the devices by their shares, and what every device takes
/// besides.
struct Whole {
    shared: u64,
    per_device: u64,
}

fn whole(c: &Config, w: &Weights<'_>, options: &Options, parallel: u32, context: u32) -> Whole {
    // One device with everything gives the totals. A split divides weights,
    // recurrent state and K/V by the shares, near enough, and repeats the
    // scratch of a step on every device.
    let all = footprint(c, w, &Placement::single(c), options, parallel)[0];
    let margin = all.weights / 64;
    Whole {
        shared: all.weights
            + all.exact
            + margin
            + all.per_sequence * u64::from(parallel)
            + all.per_page * u64::from(plan::pool_pages(context, parallel)),
        per_device: all.step - margin,
    }
}
