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
use std::sync::{Arc, OnceLock};
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
use ares_runtime::{
    GpuConfig, Machine, Meminfo, Model, Runtime, RuntimeConfig, Split, SsdConfig, required_ram,
};
use ares_tokenizer::ChatOptions;

use crate::models::{LoadFailure, Loader, Site, Unit};
use crate::plan::{
    self, AttentionShares, CELLS_PER_PAGE, DEFAULT_PARALLEL, DEFAULT_WINDOW, Drafter, LoadOptions,
    ProjectorWeights, Room, SINGLE_DEVICE_MIN_CONTEXT, SplitKind,
};
use ares_model::qwen3vl::{self, Projector};

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

/// Whether the model's MTP block takes part in speculation: when
/// `--drafters` names `mtp`, or names nothing and the model has one
/// (`has_block`). `--drafters mtp` for a model with none is refused, saying
/// `no_block`.
pub(crate) fn mtp_takes_part(
    options: &LoadOptions,
    has_block: bool,
    no_block: &str,
) -> Result<bool, LoadError> {
    let Some(list) = &options.drafters else {
        return Ok(has_block);
    };
    let asked = list.contains(&Drafter::Mtp);
    if asked && !has_block {
        return Err(refuse(format!("--drafters mtp: {no_block}")));
    }
    Ok(asked)
}

/// The drafters from files of their own that `--drafters` names, each
/// recognised by the architecture its file names: an MTP block of the model
/// (`mtp_architecture`, the block of a model whose file carries none or
/// another one) and a DFlash drafter (architecture `dflash`), one of each at
/// most. A file that is no drafter of this model is refused by what it is.
pub(crate) struct DrafterFiles {
    pub mtp: Option<(std::path::PathBuf, Gguf)>,
    pub dflash: Option<std::sync::Arc<ares_model::dflash::File>>,
}

pub(crate) fn drafter_files(
    options: &LoadOptions,
    mtp_architecture: &str,
) -> Result<DrafterFiles, LoadError> {
    let mut files = DrafterFiles { mtp: None, dflash: None };
    for drafter in options.drafters.iter().flatten() {
        let Drafter::File(path) = drafter else { continue };
        let named = |e: String| refuse(format!("--drafters {}: {e}", path.display()));
        let gguf = Gguf::open(path).map_err(|e| named(e.to_string()))?;
        match gguf.architecture() {
            Some(ares_model::dflash::ARCHITECTURE) if files.dflash.is_none() => {
                let file =
                    ares_model::dflash::File::open(path).map_err(|e| named(e.to_string()))?;
                files.dflash = Some(std::sync::Arc::new(file));
            }
            Some(arch) if arch == mtp_architecture && files.mtp.is_none() => {
                files.mtp = Some((path.clone(), gguf));
            }
            Some(arch) if arch == ares_model::dflash::ARCHITECTURE || arch == mtp_architecture => {
                return Err(refuse("--drafters: one MTP block and one DFlash drafter at most"));
            }
            other => {
                return Err(named(format!(
                    "a file of architecture '{}' is no drafter of this model (an MTP block is \
                     '{mtp_architecture}', a DFlash drafter 'dflash')",
                    other.unwrap_or("none")
                )));
            }
        }
    }
    if files.mtp.is_some() && options.drafters.iter().flatten().any(|d| *d == Drafter::Mtp) {
        return Err(refuse(
            "--drafters: mtp names the MTP block of the model file and a file names another; \
             the model drafts with one",
        ));
    }
    Ok(files)
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
/// the MTP block.
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
                placement.mtp().filter(|draft| draft.device == index && w.output.is_some())
            {
                weights +=
                    part_of(matrix_bytes(&w.token_embd), draft.head_rows as usize, c.n_vocab);
            }
            weights += part_of(matrix_bytes(w.head()), plan.head.len(), c.n_vocab);
            if let (Some(_), Some(nextn)) =
                (placement.mtp().filter(|draft| draft.device == index), &w.nextn)
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
            // A DFlash drafter on the device of the draft plan: its weights,
            // its ring per sequence, its scratch.
            let mut step_extra = 0u64;
            if let (Some(file), Some(draft)) =
                (&options.dflash, placement.draft.as_ref().filter(|d| d.device == index))
                && let Ok(fw) = file.weights()
            {
                let setup = |slots: u32| ares_model::dflash::DrafterSetup {
                    max_slots: slots,
                    max_rows: options.max_rows,
                    window_rows: options.window_rows,
                    head_rows: draft.head_rows,
                };
                let one = ares_model::dflash::bytes_of(&file.config, &fw, &setup(1));
                let two = ares_model::dflash::bytes_of(&file.config, &fw, &setup(2));
                weights += one.weights;
                per_sequence += two.total() - one.total();
                step_extra += one.scratch + one.records;
            }
            let logits = u64::from(options.logit_rows) * 4 * (d + plan.head.len() as u64);
            Footprint {
                weights,
                // The margin: alignment of the arena, the sampler, records,
                // attention plans, links, the MTP block's buffers.
                step: rows * per_row + logits + weights / 64 + 64 * MIB + step_extra,
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

/// Opens the projector of `--mmproj` for a model of width `n_embd`, with the
/// limits and the home of the options; read and checked before a device is
/// touched.
pub(crate) fn open_projector(
    options: &LoadOptions,
    n_embd: usize,
) -> Result<Option<Projector>, LoadError> {
    let Some(path) = &options.mmproj else {
        if options.mmproj_weights != ProjectorWeights::Ram
            || options.image_max_tokens != plan::IMAGE_MAX_TOKENS
        {
            return Err(refuse(
                "--mmproj-weights and --image-max-tokens are for a projector, and --mmproj names \
                 none",
            ));
        }
        return Ok(None);
    };
    let failed = |what: String| refuse(format!("--mmproj {}: {what}", path.display()));
    let gguf = Gguf::open(path).map_err(|e| failed(e.to_string()))?;
    if gguf.architecture() != Some("clip") {
        return Err(failed(format!(
            "the file is not a projector (its architecture is {})",
            gguf.architecture().unwrap_or("not declared")
        )));
    }
    let limits = qwen3vl::Limits {
        min_rows: plan::IMAGE_MIN_TOKENS as usize,
        max_rows: options.image_max_tokens as usize,
    };
    let home = match options.mmproj_weights {
        ProjectorWeights::Ram => qwen3vl::Home::Ram,
        ProjectorWeights::Vram => qwen3vl::Home::Vram,
    };
    let projector = Projector::open(gguf, limits, home).map_err(|e| failed(e.to_string()))?;
    if projector.n_out() as usize != n_embd {
        return Err(failed(format!(
            "its rows have {} values and the model's {n_embd}: it is another model's projector",
            projector.n_out()
        )));
    }
    tracing::info!(
        path = %path.display(),
        "images: the projector's tower ({} blocks), images of {} to {} rows, its weights in {} \
         ({} of VRAM and {} of page-locked RAM)",
        projector.config().n_layer,
        limits.min_rows,
        limits.max_rows,
        match home {
            qwen3vl::Home::Ram => "RAM, uploaded block by block for every new image",
            qwen3vl::Home::Vram => "VRAM",
        },
        Bytes(projector.vram_bytes()),
        Bytes(projector.pinned_bytes()),
    );
    Ok(Some(projector))
}

/// The device the image tower runs on: in the RAM home, where every new image
/// uploads its weights, the one with the fastest link to the host, as Linux
/// reports it, and then the most multiprocessors; in the VRAM home the most
/// multiprocessors, then the link. Facts the devices report, no measurement:
/// which device computes a block faster is not measured at load.
pub(crate) fn tower_device(infos: &[&DeviceInfo], home: qwen3vl::Home) -> usize {
    let link = |info: &DeviceInfo| link_rate(info).unwrap_or(0.0);
    let key = |info: &DeviceInfo| match home {
        qwen3vl::Home::Ram => (link(info), f64::from(info.multiprocessors)),
        qwen3vl::Home::Vram => (f64::from(info.multiprocessors), link(info)),
    };
    (0..infos.len())
        .max_by(|&a, &b| {
            key(infos[a])
                .partial_cmp(&key(infos[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.cmp(&a))
        })
        .unwrap_or(0)
}

/// The rate of a device's link to its slot, as Linux reports it: transfers per
/// second times lanes (only its ratio between devices is used).
fn link_rate(info: &DeviceInfo) -> Option<f64> {
    let base = format!("/sys/bus/pci/devices/{}", info.pci);
    let read = |name: &str| std::fs::read_to_string(format!("{base}/{name}")).ok();
    let speed: f64 = read("current_link_speed")?.split_whitespace().next()?.parse().ok()?;
    let width: f64 = read("current_link_width")?.trim().parse().ok()?;
    Some(speed * width)
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
pub(crate) fn read_vocabulary(path: &Path) -> Result<Arc<[u32]>, LoadError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| refuse(format!("--draft-vocab {}: cannot read it: {e}", path.display())))?;
    let mut tokens = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        tokens.push(line.parse::<u32>().map_err(|_| {
            refuse(format!(
                "--draft-vocab {}, line {}: '{line}' is not a token id",
                path.display(),
                number + 1
            ))
        })?);
    }
    if tokens.is_empty() {
        return Err(refuse(format!("--draft-vocab {}: the list is empty", path.display())));
    }
    Ok(tokens.into())
}

/// The shares of the attention heads of a tensor split by `shares` for a
/// context of `pages` pages ([`plan::attention_for_context`]), or, when the
/// devices cannot hold that many wherever the heads are, the shares with which
/// they hold the most ([`plan::attention_by_room`]); by estimate.
///
/// What a device takes besides the heads is its footprint with every head on
/// another device: its share of the weights, the scratch, the recurrent state,
/// and the K and V of the MTP block, which has heads of its own on the device
/// that holds it (296 MiB for 262 144 tokens of Qwen3.8-27B with K and V in
/// `q5`). What the heads take is the difference with all of them on the
/// device, the scratch that grows with the weights included. A device holds
/// the heads of each block whole, so it can hold up to a head more than its
/// share (about 100 MiB of the 27B's at that context): the shares are checked
/// against the footprint of the placement they make, and a device that is
/// over gives heads to the others until none is (when a round leaves a device
/// over, the context that fits is smaller, which the plan says or refuses).
#[allow(clippy::too_many_arguments)]
fn attention_for_context(
    c: &Config,
    w: &Weights<'_>,
    options: &Options,
    parallel: u32,
    free: &[u64],
    shares: &[f64],
    headroom: u64,
    pages: u32,
) -> Result<Vec<f64>, LoadError> {
    /// Rounds of checking the shares against the footprint they make: one
    /// moves the heads a device is over by, and rounding to whole heads may
    /// leave it over by one.
    const ROUNDS: usize = 8;
    let n = free.len();
    let footprint_of = |attention: Vec<f64>| -> Result<Vec<Footprint>, LoadError> {
        let spread = Spread::TensorWithAttention(shares.to_vec(), attention);
        Ok(footprint(c, w, &place(c, w, &spread, options)?, options, parallel))
    };
    let alone_on =
        |on: usize| -> Vec<f64> { (0..n).map(|i| f64::from(u8::from(i == on))).collect() };
    let usable: Vec<u64> = free.iter().map(|f| f.saturating_sub(headroom)).collect();
    // What a device takes without K and V, and with K and V of the pages.
    let fixed = |f: &Footprint| f.weights + f.step + f.exact + f.per_sequence * u64::from(parallel);
    let total = |f: &Footprint| fixed(f) + f.per_page * u64::from(pages);
    let (mut room, mut room_for_pages) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut heads, mut page) = (0, 0);
    for (i, usable) in usable.iter().enumerate() {
        let all = footprint_of(alone_on(i))?[i];
        let none = footprint_of(alone_on((i + 1) % n))?[i];
        room.push(usable.saturating_sub(fixed(&none)));
        room_for_pages.push(usable.saturating_sub(total(&none)));
        heads = heads.max(fixed(&all).saturating_sub(fixed(&none)));
        page = page.max(all.per_page.saturating_sub(none.per_page));
    }
    let by_room = || {
        plan::attention_by_room(&room).ok_or_else(|| {
            refuse("no device has room for the attention heads next to its share of the weights")
        })
    };
    let Some(mut attention) =
        plan::attention_for_context(shares, &room_for_pages, heads, page, pages)
    else {
        return by_room();
    };
    let per_share = (heads as f64 + page as f64 * f64::from(pages)).max(1.0);
    let mut most: Vec<f64> =
        room_for_pages.iter().map(|r| (*r as f64 / per_share).min(1.0)).collect();
    for _ in 0..ROUNDS {
        let planned = footprint_of(attention.clone())?;
        let over: Vec<u64> = planned
            .iter()
            .zip(&usable)
            .map(|(f, usable)| total(f).saturating_sub(*usable))
            .collect();
        if over.iter().all(|o| *o == 0) {
            break;
        }
        for ((most, over), share) in most.iter_mut().zip(&over).zip(&attention) {
            if *over > 0 {
                *most = most.min(share - *over as f64 / per_share);
            }
        }
        match plan::fit_shares(shares, &most) {
            Some(fitted) => attention = fitted,
            None => return by_room(),
        }
    }
    Ok(attention)
}

/// The placement of `spread`, with the MTP block where `Qwen35::new` puts
/// it, to estimate with.
fn place(
    c: &Config,
    w: &Weights<'_>,
    spread: &Spread,
    options: &Options,
) -> Result<Placement, LoadError> {
    let model = |e: ares_model::qwen35::device::PlacementError| LoadError::Model(e.to_string());
    let placement = spread.place(c, w).map_err(model)?;
    let drafts = options.window_rows > 1;
    let vocabulary = options.draft_vocabulary.as_deref();
    if drafts && (c.n_nextn == 1 || options.mtp_file.is_some()) && options.mtp {
        placement.with_mtp(c, vocabulary).map_err(model)
    } else if drafts && options.dflash.is_some() && placement.split != Split::Layer {
        placement.with_head_view(c, vocabulary, false).map_err(model)
    } else {
        Ok(placement)
    }
}

fn rooms(free: &[u64], footprint: &[Footprint], headroom: u64) -> Vec<Room> {
    free.iter()
        .zip(footprint)
        .map(|(free, f)| Room {
            usable: free.saturating_sub(headroom),
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

static WATCH: OnceLock<Arc<Machine>> = OnceLock::new();

/// The watch over the machine's memory that the engines of this process
/// follow: the machine itself ([`Machine::system`]), unless [`follow`] named
/// another before the first model was loaded.
pub fn watch() -> Arc<Machine> {
    WATCH.get_or_init(Machine::system).clone()
}

/// Makes the engines of this process follow `machine`. For a test or a tool
/// that gives the engine a memory of its own to follow (a machine with less
/// than this one has, say); the server never calls it. Returns false, and
/// changes nothing, once a watch is in use.
pub fn follow(machine: Arc<Machine>) -> bool {
    WATCH.set(machine).is_ok()
}

/// The most host RAM the engine may use: what the operator named, or all the
/// machine has. Below that limit the engine follows the machine
/// ([`Machine`]): its elastic memory is what is available less a headroom.
/// `needs` is what the engine must page-lock to run at all (its step buffers,
/// and what else a model or the prompt cache cannot do without), by what it
/// is for: a limit below their sum is refused, and the refusal names every
/// share. `nothing` says what is not spared when the machine reports no
/// memory.
pub(crate) fn ram_limit_of(
    options: &LoadOptions,
    site: &Site,
    needs: &[(&str, u64)],
    nothing: &str,
) -> Result<(u64, String), LoadError> {
    let required: u64 = needs.iter().map(|(_, bytes)| bytes).sum();
    let machine = watch();
    // What the engine really holds below its limit, as it stands now.
    let follows = match machine.available() {
        Some(available) => format!(
            "below it the engine holds what the machine has available, {} now, less a headroom \
             of {}, and gives memory back when other programs need it",
            Bytes(available),
            Bytes(machine.headroom())
        ),
        None => "the machine does not report what it has available, so the engine holds no \
                 more than it needs to run"
            .to_owned(),
    };
    let (ram, why_ram) = match options.ram {
        Some(ram) if site.ram_is_rest => {
            (ram, format!("what the server's limit has left next to the loaded models; {follows}"))
        }
        Some(ram) => (ram, format!("as asked; {follows}")),
        None => match Meminfo::total() {
            Some(total) => (
                plan::default_ram(Some(total)).max(required),
                format!("all the machine has; {follows}; --ram sets a lower limit"),
            ),
            None => (
                required,
                format!("the machine reports no memory, so {nothing}; --ram sets the limit"),
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

/// What a model is, read from its file and its options without loading it:
/// for `/props` of a model that is not loaded.
#[derive(Debug, Clone)]
pub(crate) struct Described {
    /// The chat template in effect (the file's, or `--chat-template-file`).
    pub chat_template: Option<String>,
    /// What the template declares, with the model's chat variables.
    pub format: ares_api::ChatFormat,
    /// The context its options name (`--context`), else the longest the
    /// model file declares; a load decides what fits the devices.
    pub context_length: Option<u32>,
}

/// Reads [`Described`] for the model of `options`: the file's metadata and
/// its tokenizer, no tensor. Call it where waiting is allowed.
pub(crate) fn read_described(options: &LoadOptions) -> Result<Described, LoadError> {
    let gguf = Gguf::open(&options.model)?;
    let architecture = gguf.architecture().ok_or(LoadError::NoArchitecture)?.to_owned();
    let tokenizer =
        ares_tokenizer::Tokenizer::from_gguf(&gguf).map_err(|e| LoadError::Model(e.to_string()))?;
    let (chat, chat_template) = chat_options(options, &tokenizer)?;
    let format = ares_model::chat::chat_format(&tokenizer, &chat);
    let file =
        gguf.u64(&format!("{architecture}.context_length")).and_then(|n| u32::try_from(n).ok());
    Ok(Described { chat_template, format, context_length: options.context.or(file) })
}

/// The settings a model renders conversations with: the variables of the
/// chat template, and the template of `--chat-template-file` in place of the
/// file's, compiled with the model's tokenizer. Also the text of the template
/// that is in effect, if there is one.
pub(crate) fn chat_options(
    options: &LoadOptions,
    tokenizer: &ares_tokenizer::Tokenizer,
) -> Result<(ChatOptions, Option<String>), LoadError> {
    let mut chat =
        ChatOptions { variables: options.chat_variables.clone(), ..ChatOptions::default() };
    let Some(path) = &options.chat_template else {
        return Ok((chat, tokenizer.chat_template_source().map(str::to_owned)));
    };
    let failed = |what: String| {
        LoadError::Options(format!("--chat-template-file '{}': {what}", path.display()))
    };
    let source = std::fs::read_to_string(path).map_err(|e| failed(e.to_string()))?;
    let template = tokenizer.compile_template(&source).map_err(|e| failed(e.to_string()))?;
    tracing::info!(file = %path.display(), "the chat template of the file takes the place of the model's");
    chat.template = Some(Arc::new(template));
    Ok((chat, Some(source)))
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
        if config.n_nextn == 1 { "with an MTP block" } else { "no MTP block" },
    );

    // What can be refused before a device is touched.
    if options.expert_prune.is_some() {
        return Err(refuse(
            "--expert-prune does nothing for this model: it has no mixture of experts",
        ));
    }
    let kv = options.kv.unwrap_or(Options::default().kv);
    let parallel = options.parallel.unwrap_or(DEFAULT_PARALLEL);
    if parallel == 0 {
        return Err(refuse("--parallel must be at least 1"));
    }
    // The model drafters that take part: all the model has unless the operator
    // names them, the MTP block of the model file or of a file of its own.
    // Nothing speculates without one, the n-gram drafter included.
    let files = drafter_files(options, ares_model::qwen35::MTP_ARCHITECTURE)?;
    let mtp_file = match files.mtp {
        Some((path, file)) => {
            // Read before a device is touched: a block of another model is
            // refused by name.
            Weights::load(&gguf, &config)
                .and_then(|w| w.with_mtp_file(&file, &config))
                .map_err(|e| refuse(format!("--drafters {}: {e}", path.display())))?;
            tracing::info!("drafter: the MTP block of {}", path.display());
            Some(std::sync::Arc::new(file))
        }
        None => None,
    };
    let mtp = mtp_file.is_some()
        || mtp_takes_part(
            options,
            config.n_nextn == 1,
            "this model file has no MTP block (multi-token prediction); a file of one is named \
             by its path",
        )?;
    let layer = options.split == Some(SplitKind::Layer);
    // A DFlash drafter from a file: it reads taps of the model's layers on
    // the device that embeds and holds the first rows of the head, which a
    // layer split does not give.
    let dflash = files.dflash;
    if let Some(file) = &dflash {
        if layer {
            return Err(refuse(
                "--drafters: the DFlash drafter is not built for a layer split; use a single \
                 device or a tensor split",
            ));
        }
        file.config
            .fits(config.n_embd, config.n_layer, config.n_vocab)
            .map_err(|e| refuse(format!("--drafters: {e}")))?;
        tracing::info!(
            "drafter: DFlash from {}, {} layers, blocks of {} tokens, taps of layers {:?}, {} \
             of weights",
            file.gguf.path(0).display(),
            file.config.n_layer,
            file.config.block,
            file.config.target_layers,
            Bytes(file.bytes()),
        );
    }
    let window = options.window.unwrap_or(DEFAULT_WINDOW);
    let drafts = mtp || dflash.is_some();
    if options.window.is_some() && !drafts {
        return Err(refuse("--window without a model drafter does nothing"));
    }
    let vocabulary = match &options.draft_vocabulary {
        Some(_) if !drafts => {
            return Err(refuse(
                "a draft vocabulary is for the model's MTP block, and no drafter of that kind \
                 takes part",
            ));
        }
        Some(path) => Some(read_vocabulary(path)?),
        None => None,
    };
    if options.context == Some(0) {
        return Err(refuse("--context must be at least 1"));
    }
    let projector = open_projector(options, config.n_embd)?;

    // The devices.
    let (ordinals, mut held) = hold_devices(options)?;
    // The image tower's VRAM is held back on the device that runs it before
    // anything is planned there.
    let tower_bytes = projector.as_ref().map_or(0, Projector::vram_bytes);
    let mut tower_at = projector
        .as_ref()
        .map(|p| tower_device(&held.iter().map(|h| h.device.info()).collect::<Vec<_>>(), p.home()));
    if let Some(t) = tower_at {
        held[t].free = held[t].free.saturating_sub(tower_bytes);
    }

    // The model's settings that do not depend on the placement.
    let window_rows = if drafts { parallel.saturating_mul(window) } else { 0 };
    let defaults = Options::default();
    let part_options = Options {
        precision: options.precision,
        kv,
        max_rows: options.ubatch.unwrap_or(defaults.max_rows),
        logit_rows: window_rows.max(parallel),
        window_rows,
        draft_vocabulary: vocabulary.clone(),
        mtp,
        mtp_file,
        dflash: dflash.clone(),
        ..defaults
    };
    // A dense model runs every step in its own memory: a step of prompt has
    // at most the rows of its scratch.
    if let Some(batch) = options.batch.filter(|b| *b > part_options.max_rows) {
        return Err(refuse(format!(
            "--batch-size {batch}: this model runs no step wider than its scratch of {} rows \
             (--ubatch-size); only a mixture of experts lends VRAM for wider ones",
            part_options.max_rows
        )));
    }
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
        let free: Vec<u64> = held.iter().map(|h| h.free).collect();
        let pages = plan::pages_that_fit(&rooms(&free, &footprint, headroom), parallel);
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
            let mut keep = held.swap_remove(chosen);
            // The tower runs on the one device left.
            if let Some(t) = tower_at
                && t != chosen
            {
                keep.free = keep.free.saturating_sub(tower_bytes);
            }
            tower_at = tower_at.map(|_| 0);
            held = vec![keep];
        }
    }
    if options.attention != AttentionShares::ByContext && split != SplitKind::Tensor {
        return Err(refuse(
            "shares of the attention heads are for a tensor split, where every device holds \
             heads of every attention block",
        ));
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
            match (split, &options.attention) {
                (SplitKind::Tensor, AttentionShares::Given(attention))
                    if attention.len() != held.len() =>
                {
                    return Err(refuse(format!(
                        "{} shares of the attention heads for {} devices",
                        attention.len(),
                        held.len()
                    )));
                }
                (SplitKind::Tensor, AttentionShares::Given(attention)) => {
                    (Spread::TensorWithAttention(shares, attention.clone()), why)
                }
                (SplitKind::Tensor, AttentionShares::ByContext) => {
                    let wanted = options.context.unwrap_or(config.n_ctx as u32);
                    let attention = attention_for_context(
                        &config,
                        &weights,
                        &part_options,
                        parallel,
                        &held.iter().map(|h| h.free).collect::<Vec<_>>(),
                        &shares,
                        headroom,
                        plan::pool_pages(wanted, parallel),
                    )?;
                    (Spread::TensorWithAttention(shares, attention), why)
                }
                _ => (Spread::Layer(shares), why),
            }
        }
    };
    describe_links(&held)?;
    let placement = place(&config, &weights, &spread, &part_options)?;
    if vocabulary.is_some() && placement.draft.is_none() {
        return Err(refuse(
            "a draft vocabulary is for the model's MTP block, which is not built for a layer \
             split",
        ));
    }
    let footprint = footprint(&config, &weights, &placement, &part_options, parallel);
    let free: Vec<u64> = held.iter().map(|h| h.free).collect();
    let fit = plan::pages_that_fit(&rooms(&free, &footprint, headroom), parallel);

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
        // The estimate decides before the weights are read: a plan is never
        // above what the devices have free. Next to other models that is no
        // room, which one of them makes; alone the context is refused.
        Some(context) if fit < plan::pool_pages(context, parallel) => {
            let held_tokens = plan::pool_tokens(fit, parallel);
            if site.beside {
                return Err(LoadError::NoRoom(format!(
                    "the devices cannot hold the model with its context of {context} tokens \
                     next to what is loaded (by estimate: {})",
                    taken_text()
                )));
            }
            return Err(refuse(format!(
                "--context {context}: the devices hold about {held_tokens} tokens of context \
                 for {parallel} sequence{} with this plan (by estimate: {}; K and V take {} per \
                 1000 tokens); name a smaller context, fewer sequences with --parallel, K and \
                 V in fewer bits with --kv{}",
                if parallel == 1 { "" } else { "s" },
                taken_text(),
                join(
                    held.iter().zip(&footprint).map(|(h, f)| format!(
                        "{} on device {}",
                        Bytes(f.per_page * 1000 / u64::from(CELLS_PER_PAGE)),
                        h.device.ordinal()
                    )),
                    " and "
                ),
                if options.shares.is_some() {
                    " or other shares of the weights with --shares (the heads of the attention \
                     blocks are placed for the context already)"
                } else {
                    ""
                },
            )));
        }
        Some(context) => (context, "as asked".to_owned()),
        None => {
            let plan = plan::ContextPlan::new(None, model_context, fit, parallel);
            let context = plan.context();
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
            (context, plan.why())
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
            Spread::TensorWithAttention(shares, attention) => format!(
                ", shares {} ({why_shares}), of the attention heads with their K and V {}",
                shares_text(shares),
                shares_text(attention)
            ),
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
        match (drafts, &placement.draft) {
            (false, _) => match options.drafters {
                Some(_) => "no drafters (--drafters none): no speculation".to_owned(),
                None => "no model drafter (the file has no MTP block): no speculation".to_owned(),
            },
            (true, Some(draft)) => format!(
                "drafts from {} on device {} ({}) and from earlier text, windows of up to \
                 {window} rows; whose drafts and how many are chosen for every step from the \
                 price of a step, measured at load, and the acceptance, measured while running",
                match (mtp, &dflash) {
                    (true, Some(_)) => "the model's MTP block and the DFlash drafter",
                    (true, None) => "the model's MTP block",
                    (false, _) => "the DFlash drafter",
                },
                held[draft.device].device.ordinal(),
                match &vocabulary {
                    Some(list) => format!("a vocabulary of {} tokens", list.len()),
                    None => format!("the {} tokens whose head rows are there", draft.head_rows),
                }
            ),
            (true, None) => {
                "no speculation: the model has no MTP block on a device of this placement"
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
    drop(weights);
    let model = Qwen35::new(gguf, name.clone(), &spread, part_options, longest)
        .map_err(|e| LoadError::Model(e.to_string()))?;
    let model = match (projector, tower_at) {
        (Some(projector), Some(t)) => {
            tracing::info!(
                "plan: the image tower on device {}, {} of its VRAM held back for it",
                held[t].device.ordinal(),
                Bytes(tower_bytes)
            );
            model
                .with_projector(Arc::new(projector), t)
                .map_err(|e| LoadError::Model(e.to_string()))?
        }
        _ => model,
    };
    let (chat, template) = chat_options(options, model.tokenizer())?;
    let model = Arc::new(model.with_chat_options(chat));
    if template.is_none() {
        tracing::warn!(
            "the model file has no chat template: /v1/chat/completions will refuse requests, \
             /v1/completions works"
        );
    }
    // Host RAM: one limit for the engine. What it needs to run is taken from
    // it first: its step buffers and, where the token embedding table lies in
    // host memory, the table and the rows of a step. A limit below that is
    // refused: the table does not go back to a device, whose memory is what
    // bounds the context. The rest is the prompt cache's in RAM (the only
    // other user in a model without experts): checkpoints of recurrent state
    // and the pages that left the devices, locked as they come into use, as
    // far as the machine has the memory, and given back when it gets short.
    // The prompt cache on SSD adds two needs of its own, below.
    let required = required_ram(&*model, parallel);
    let tower_ram = model.projector().map_or(0, |p| p.pinned_bytes());
    let table = model.pinned_ram() - tower_ram;
    let mut needs = vec![("its step buffers", required - table - tower_ram)];
    if tower_ram > 0 {
        needs.push((
            "the image tower's weights, which lie in page-locked RAM (--mmproj-weights ram)",
            tower_ram,
        ));
    }
    if table > 0 {
        needs.push((
            "the token embedding table and the rows of a step, which lie in page-locked RAM \
             and not on a device",
            table,
        ));
    }
    let checkpoint = model.prefix_cache().map_or(0, |cache| cache.bytes());
    // One checkpoint place is what the prompt cache keeps whatever the
    // machine has (the cache's floor in the engine's books): without a
    // checkpoint a hit replays every row it found. The prompt cache on SSD
    // moves state through RAM on its way to the drive and back: with a
    // directory for it, a few pages join that place in the floor. And it
    // copies checkpoints out of their places into buffers of ordinary memory,
    // which the engine holds once they are in use: they are not page-locked,
    // so they are not in those books, and they count against the limit here.
    let ssd = options.cache.as_ref().map(|dir| {
        SsdConfig::new(dir, options.cache_size.unwrap_or(crate::plan::DEFAULT_CACHE_SIZE))
    });
    let durable = ssd.as_ref().and_then(|_| model.durable(CELLS_PER_PAGE));
    let transit = durable.as_ref().map_or(0, SsdConfig::transit_ram);
    let floor =
        durable.as_ref().map_or(checkpoint, |durable| SsdConfig::floor_ram(durable, checkpoint));
    let through = durable.is_some() && floor > 0;
    match (floor, through) {
        (0, _) => {}
        (_, true) => needs.push((
            "one checkpoint of recurrent state and the pages on their way to the prompt cache \
             on SSD or back (page-locked)",
            floor,
        )),
        (_, false) => needs.push((
            "one checkpoint of recurrent state, which the prompt cache in RAM keeps whatever \
             the machine has (page-locked)",
            floor,
        )),
    }
    if transit > 0 {
        needs.push((
            "the checkpoints in transit to the prompt cache on SSD and back, at their most \
             (ordinary memory, taken as it is first used)",
            transit,
        ));
    }
    let (ram, why_ram) =
        ram_limit_of(options, site, &needs, "none is spared for the prompt cache in RAM")?;
    // What the engine may page-lock: the limit less the buffers in transit.
    let lockable = ram - transit;
    let runtime_config = RuntimeConfig {
        pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: parallel,
        queue: options.queue,
        window_rows: if drafts { window } else { 1 },
        cache_ram: lockable - required,
        ssd: ssd.clone(),
        max_rows: options.batch.unwrap_or(0),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: headroom,
        ram_limit: lockable,
        ..GpuConfig::new(&runtime_config, &*model)
    };
    tracing::info!(
        "plan: RAM, at most {} for the engine ({why_ram}): {} for its step buffers{}{}, the \
         rest for the prompt cache in RAM, the pages that left the devices and checkpoints of \
         recurrent state, {} each over all devices, {}; the memory is page-locked as pages \
         leave the devices and checkpoints are taken, and the coldest are given up when the \
         machine needs it{}",
        Bytes(ram),
        Bytes(required - table - tower_ram),
        match table {
            0 => String::new(),
            bytes => format!(
                ", {} for the token embedding table and the rows of a step (locked at load)",
                Bytes(bytes)
            ),
        },
        match transit {
            0 => String::new(),
            bytes => format!(
                ", at most {} for checkpoints in transit between RAM and the prompt cache's \
                 store on SSD (ordinary memory, taken as it is first used)",
                Bytes(bytes)
            ),
        },
        Bytes(checkpoint),
        match checkpoint {
            0 => "none for this model".to_owned(),
            bytes => format!("room for {}", (lockable - required) / bytes),
        },
        match (floor, through) {
            (0, _) => String::new(),
            (_, true) => format!(
                ", down to {} that state passes through on its way to the prompt cache's store \
                 on SSD and back, a checkpoint place and pages, which are kept whatever the \
                 machine has; only what leaves RAM is written there, and what is worth keeping \
                 when the model is unloaded",
                Bytes(floor)
            ),
            (_, false) => format!(
                ", down to {}, one checkpoint place, which is kept whatever the machine has",
                Bytes(floor)
            ),
        }
    );
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let devices: Vec<Device> = held.iter().map(|h| h.device.clone()).collect();
    let machine = Some(watch());
    let started =
        Runtime::start_gpu_following(model.clone(), devices, runtime_config, gpu, machine);
    let runtime = started.map_err(|e| {
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
    let runtime = runtime.with_chat_template(template);
    tracing::info!(
        "loaded in {:.1} s ({:.1} s since the start)",
        loading.elapsed().as_secs_f64(),
        begun.elapsed().as_secs_f64()
    );
    if let (Some(ssd), Some(snapshot)) = (&ssd, runtime.snapshot()) {
        let held = snapshot.counters.ssd;
        match &snapshot.ssd_problem {
            Some(why) => tracing::warn!(
                "the prompt cache on SSD in {} is not in use, the cache lives in the devices and \
                 in RAM alone: {why}",
                ssd.dir.display()
            ),
            None => tracing::info!(
                "prompt cache on SSD: {}, at most {} for everything in it; this model's store \
                 holds {} pages and {} checkpoints in {}; texts of {} tokens and more are \
                 written when they leave RAM, and when the model is unloaded",
                ssd.dir.display(),
                Bytes(ssd.max_bytes),
                held.store_pages,
                held.store_states,
                Bytes(held.store_bytes),
                ssd.min_cells
            ),
        }
    }
    if let Some(calibration) = runtime.snapshot().and_then(|snapshot| snapshot.calibration) {
        let table = |points: &[(u32, f64)]| {
            points.iter().map(|(n, ms)| format!("{n}: {ms:.1}")).collect::<Vec<_>>().join(", ")
        };
        tracing::info!(
            "the price of a step was measured in {:.1} s ({} steps), in ms by rows: {}{}",
            calibration.seconds,
            calibration.steps,
            table(&calibration.step_ms),
            calibration
                .chain_ms
                .iter()
                .zip(model.drafters())
                .map(|(chain, drafter)| {
                    format!("; {} by drafts: {}", drafter.kind.name(), table(chain))
                })
                .collect::<String>()
        );
    }

    // What it took.
    for ((h, report), f) in held.iter().zip(model.reports()).zip(&footprint) {
        let state = report.state_bytes_per_slot * u64::from(parallel);
        let kv_bytes = report.page_bytes * u64::from(pages);
        let draft = report.mtp.as_ref().map_or(0, |d| d.carry + d.scratch + d.records);
        let exact = report.exact_bytes + report.mtp.as_ref().map_or(0, |d| d.exact);
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
             for draft windows {}, MTP block buffers {}, everything else {}; taken {}, free {}",
            h.device.ordinal(),
            report.device,
            Bytes(report.weight_bytes),
            report.upload.as_secs_f64(),
            report
                .mtp
                .as_ref()
                .map(|d| format!(" (of which the MTP block {})", Bytes(d.weights)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ares_cuda::CacheFormat;

    /// The owner's production settings on the two development cards, as the
    /// owner's fork serves them (2026-11-22): Qwen3.8-27B UD-Q4_K_S
    /// (`ARES_TEST_QWEN35_27B`), a tensor split of the weights 0.625 / 0.375,
    /// 262 144 tokens of context for two sequences, K and V in `q5`, the MTP
    /// block drafting with a draft vocabulary (`ARES_TEST_DRAFT_VOCAB`, else the
    /// head's rows), and the free memory the cards had: 11.52 and 11.57 GiB.
    /// The heads by the context once left the RTX 4070 planned above its free
    /// memory (the MTP block's K and V were not counted), and the load died; the
    /// plan now holds the context on both devices within what they have. A
    /// context they cannot hold is refused by estimate, before anything is
    /// loaded (here with K and V in `f16`).
    #[test]
    fn the_27b_at_the_owners_settings_plans_within_free_memory() {
        let Some(path) = std::env::var_os("ARES_TEST_QWEN35_27B") else {
            eprintln!("skipped: set ARES_TEST_QWEN35_27B");
            return;
        };
        let gguf = Gguf::open(&path).unwrap();
        let config = Config::from_gguf(&gguf).unwrap();
        let weights = Weights::load(&gguf, &config).unwrap();
        let vocabulary = std::env::var_os("ARES_TEST_DRAFT_VOCAB")
            .map(|p| read_vocabulary(Path::new(&p)).unwrap());
        let parallel = 2;
        let window_rows = parallel * DEFAULT_WINDOW;
        let options = |kv: CacheFormat| Options {
            kv: (kv, kv),
            logit_rows: window_rows.max(parallel),
            window_rows,
            draft_vocabulary: vocabulary.clone(),
            mtp: true,
            ..Options::default()
        };
        let free = [11_796 * MIB, 11_847 * MIB];
        let shares = [0.625, 0.375];
        let context = 262_144;
        let pages = plan::pool_pages(context, parallel);
        for (kv, fits) in [(CacheFormat::Q5, true), (CacheFormat::F16, false)] {
            let options = options(kv);
            let attention = attention_for_context(
                &config,
                &weights,
                &options,
                parallel,
                &free,
                &shares,
                VRAM_HEADROOM,
                pages,
            )
            .unwrap();
            let spread = Spread::TensorWithAttention(shares.to_vec(), attention.clone());
            let placement = place(&config, &weights, &spread, &options).unwrap();
            let footprint = footprint(&config, &weights, &placement, &options, parallel);
            let fit = plan::pages_that_fit(&rooms(&free, &footprint, VRAM_HEADROOM), parallel);
            for (device, f) in footprint.iter().enumerate() {
                let planned = f.weights
                    + f.step
                    + f.exact
                    + f.per_sequence * u64::from(parallel)
                    + f.per_page * u64::from(pages);
                eprintln!(
                    "{kv:?}: device {device}: heads {:.3}, {} MiB planned for {pages} pages of \
                     {} MiB free; {fit} pages fit",
                    attention[device],
                    planned / MIB,
                    free[device] / MIB
                );
                if fits {
                    assert!(planned + VRAM_HEADROOM <= free[device], "{kv:?}: device {device}");
                }
            }
            assert_eq!(fit >= pages, fits, "{kv:?}: {fit} pages fit, {pages} asked");
        }
    }
}
