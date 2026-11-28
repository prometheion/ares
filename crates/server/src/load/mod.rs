//! Loading a model: from a GGUF file and what the operator asked for to a
//! running [`Runtime`] on the devices, with an account of every decision and
//! of where the memory went.
//!
//! The order is: read the file, look at the devices, decide (which devices,
//! which split, which shares, how many sequences, how much context), log the
//! decision with its reasons, load, log what it took. What the operator named
//! is taken as it is and judged by the devices' ledgers at load; what was
//! left open comes from numbers the devices and the model give (`plan`, and
//! the model's own estimate in `ares-model`).
//!
//! - `qwen35.rs` loads a dense model, all of it on the devices; `qwen4exp.rs`
//!   a mixture of experts whose dense parts are on the devices and whose
//!   experts follow the router through slots in VRAM, a tier in RAM and the
//!   file. Each is cut into its phases: what can be refused before a device
//!   is touched, the devices, the split, the context, the plan in the log,
//!   the model built, the engine started, what it took.
//! - `files.rs` reads what a load needs besides the model file: the
//!   drafters, the draft vocabulary, the projector, the chat template.
//! - `devices.rs` opens and describes the devices and their links.
//! - `ram.rs` is the host RAM of both: one limit, what the engine needs to
//!   run, and the machine it follows (`Budget`).
//! - `start.rs` is the tail both share: the runtime's settings, the start,
//!   what the engine found when it started.

mod devices;
mod files;
mod host;
mod products;
mod qwen35;
mod qwen4exp;
mod ram;
mod start;

use std::sync::Arc;
use std::time::Instant;

use ares_api::Engine;
use ares_core::units::{Bytes, MIB};
use ares_gguf::Gguf;
use ares_runtime::Runtime;

pub(crate) use devices::Held;
pub(crate) use files::chat_options;
pub use ram::{follow, machine};

use crate::models::{LoadFailure, Loader, Site, Unit};
use crate::plan::LoadOptions;

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

/// An error of the model as the load reports it.
fn model_error(e: impl ToString) -> LoadError {
    LoadError::Model(e.to_string())
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
        ares_model::qwen35::config::ARCHITECTURE => qwen35::load(gguf, options, site, begun),
        ares_model::qwen4exp::config::ARCHITECTURE => qwen4exp::load(gguf, options, site, begun),
        host::ARCHITECTURE => host::load(options, begun),
        _ => Err(LoadError::Architecture(architecture)),
    }
}

pub(crate) fn join<T: ToString>(items: impl IntoIterator<Item = T>, with: &str) -> String {
    items.into_iter().map(|item| item.to_string()).collect::<Vec<_>>().join(with)
}

fn shares_text(shares: &[f64]) -> String {
    join(shares.iter().map(|share| format!("{share:.3}")), " / ")
}

/// The plan's line of the context: `context` tokens (`why`) in `pages` pages
/// for `parallel` sequences, of which a request holds at most `longest`.
fn log_context(
    options: &LoadOptions,
    context: u32,
    why: &str,
    pages: u32,
    longest: u32,
    parallel: u32,
) {
    tracing::info!(
        "plan: context of {context} tokens for all sequences together ({why}), {pages} pages; a \
         request holds at most {longest} tokens; {parallel} sequence{} at once ({})",
        if parallel == 1 { "" } else { "s" },
        if options.parallel.is_some() { "as asked" } else { "the default" },
    );
}
