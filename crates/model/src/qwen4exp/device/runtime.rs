//! `qwen4exp` as the runtime drives it: the [`Model`] that tells the runtime
//! what it is, and on each device's thread the part it opens (`parts.rs`)
//! behind the [`DeviceModel`] that turns the runtime's step into the part's
//! (`device.rs`). `plan.rs` answers what the server asks before it opens the
//! model (the rows of a step, the placement, the pages); `pass.rs` reads a
//! long prompt layer by layer.
//!
//! # How a step waits for experts
//!
//! The experts of a layer are known when its router has run, and an expert
//! that is in no slot has to be brought (uploaded, or computed by the CPU)
//! while the device waits for it. So a step is not enqueued blind to its end,
//! and the host has work in it besides enqueuing: following the route records
//! the routers leave and serving the expert store, which the device threads
//! share through the [`Flow`] (`flow.rs`). A thread serves the store
//! whenever it would otherwise wait: for the lead over the route records, for
//! the rows of the n-gram table, for a wide step's missed experts, and for the
//! end of its step ([`DeviceModel::wait`], which the runtime calls where it
//! would otherwise block in the driver).
//!
//! The conductor sees nothing of this. It dispatches a step and collects it
//! when every device has ended; a step that misses experts is a longer step.
//! Cancellation and new requests are handled between steps, as for any model.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use ares_api::{ChatFormat, EngineError, ExpertStats, ModelCard, Prompt};
use ares_cuda::{Device, ENTRY_CELLS, MAX_BATCH};
use ares_gguf::Gguf;
use ares_runtime::expert_store::{ExpertTypes, FillStats, Serve, StoreStats};
use ares_runtime::{
    ContextSpec, DeviceContext, DeviceModel, Drafter, Durable, FarPages, LinkSpec, Model,
    ModelError, PrefixCache, RecordView, Split, Vision,
};
use ares_state::PageLayout;
use ares_tokenizer::{ChatOptions, Tokenizer};

use super::feed::{Fill, Layout, MtpExperts};
use super::flow::Flow;
use super::misses::WideStats;
use super::mtp::{MtpDrafted, MtpReport, MtpTimes, Watch};
use super::ngram;
use super::open::PartReport;
use super::options::ModelOptions;
use super::part::{Frame, MTP_KV, Options, PAGE_ENTRY, PartShape, Pieces};
use super::placement::{PartPlan, Placement};
use super::prune::PruneMask;
use super::usage::{Log, ProfileReport, Usage, UsageStats};
use crate::blocks::head::HeadView;
use crate::blocks::rows::{HostTable, TOKEN_EMBD};
use crate::blocks::slots::{self, Declared, TAILS_PER_SLOT};
use crate::blocks::step;
use crate::chat::ModelText;
use crate::qwen3vl::Projector;
use crate::qwen4exp::reference::mtp_config;
use crate::qwen4exp::weights::{MtpWeights, Weights};
use crate::qwen4exp::{Config, Error as FileError};

mod device;
mod meeting;
mod parts;
mod pass;
mod plan;

use meeting::Meeting;
use parts::Build;
pub(super) use pass::{PassKey, PassLoan};
pub use plan::{
    batch_rows, drafter_bytes, page_parts, place, reads_layer_by_layer, slot_floors, step_rows,
};
use plan::{place_in, widest_rows};

/// What the expert store measured of serving a miss on this machine
/// ([`Qwen4Exp::expert_costs`]).
#[derive(Clone, Debug, Default)]
pub struct ExpertCosts {
    /// Per pair of types of the experts (gate and up, down): the seconds a
    /// CPU job takes per (expert, row) pair with one row and with eight.
    pub cpu: Vec<(ExpertTypes, Option<f64>, Option<f64>)>,
    /// Per device: the seconds of one upload, once one was timed.
    pub upload: Vec<Option<f64>>,
}

/// Why a model could not be set up.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error(transparent)]
    File(#[from] FileError),
    #[error(transparent)]
    Device(#[from] super::Error),
    #[error(transparent)]
    Model(#[from] ModelError),
    /// A choice that a model on the runtime cannot carry out.
    #[error("{0}")]
    Choice(String),
}

impl OpenError {
    /// Whether the model cannot be opened for lack of memory (a placement
    /// that does not fit, an allocation the driver refused): next to other
    /// models, unloading one may make room.
    pub fn is_no_room(&self) -> bool {
        match self {
            Self::Device(e) => e.is_no_room(),
            Self::Model(e) => e.is_no_room(),
            Self::File(_) | Self::Choice(_) => false,
        }
    }
}

/// A `qwen4exp` model file, placed.
pub struct Qwen4Exp {
    gguf: Arc<Gguf>,
    config: Config,
    opts: ModelOptions,
    placement: Placement,
    layout: Layout,
    /// The model's name, tokenizer and chat, and the projector images are
    /// read through.
    text: ModelText,
    meeting: Meeting<Build>,
    flow: OnceLock<Arc<Flow>>,
    reports: Mutex<Vec<PartReport>>,
    /// The page-locked memory of the parts, besides the tier of experts.
    pinned: u64,
    /// The bytes one page of the pools takes on each device.
    page_bytes: Vec<u64>,
    /// The pages whose K and V lie in host memory, when some do.
    far: Option<FarPages>,
    /// What each device keeps of a sequence and of a page, for the prefix
    /// cache (`resume.rs`).
    declared: Vec<Declared>,
    /// The file of the model's MTP block, when it drafts.
    mtp: Option<Arc<Gguf>>,
    /// A DFlash drafter, when it drafts.
    dflash: Option<Arc<crate::dflash::File>>,
    mtp_report: Mutex<Option<MtpReport>>,
    mtp_watch: Arc<Mutex<Watch>>,
    /// The usage profile of the experts, for a model that keeps one.
    usage: Usage,
    /// The token embedding table where the file is mapped (`blocks/rows.rs`).
    rows: Arc<HostTable>,
    /// The rows of the widest step, once the parts are open: what the expert
    /// slots can lend the scratch for, at most `opts.part.max_rows`.
    lone_rows: OnceLock<u32>,
    /// The most rows of a prompt read layer by layer, once the parts are open: what the
    /// expert slots can lend for its kept rows, at most `opts.pass_rows`; 0 for none.
    pass_span: OnceLock<u32>,
    /// The order of the output head's rows and the draft head's rows, which
    /// every drafter reads (`blocks/head.rs`).
    view: HeadView,
}

impl std::fmt::Debug for Qwen4Exp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4Exp")
            .field("name", &self.text.name)
            .field("devices", &self.placement.n_devices())
            .field("shared", &self.placement.tensor.is_some())
            .finish_non_exhaustive()
    }
}

impl Qwen4Exp {
    /// Reads the configuration of `gguf` and decides the placement on
    /// `devices`. `tokenizer` is the model's (`Tokenizer::from_gguf`);
    /// `context` is the longest sequence the model is served with
    /// (`opts.part.cells`); the pools hold `opts.part.pool_pages` pages in
    /// all. Nothing is uploaded until the runtime opens the parts.
    pub fn new(
        gguf: Gguf,
        tokenizer: Tokenizer,
        name: impl Into<String>,
        devices: &[Device],
        mut opts: ModelOptions,
        context: u32,
    ) -> Result<Self, OpenError> {
        let choice = |why: &str| Err(OpenError::Choice(why.to_owned()));
        if opts.part.pool_pages == 0 {
            return choice("the pools of K/V and keys need a number of pages");
        }
        let config = Config::from_gguf(&gguf)?;
        let context = context.clamp(1, config.n_ctx as u32);
        if opts.part.cells != context {
            return choice("the cells of a sequence are the context of the model");
        }
        // The sampler reads the logits on the device.
        opts.part.host_logits = false;
        // The parts read prompts layer by layer up to this many rows.
        opts.part.pass_rows = opts.pass_rows;
        let weights = Weights::load(&gguf, &config)?;
        let (placement, page_bytes) = place_in(&gguf, &config, &weights, devices, &opts)?;
        let (layout, _map) = Layout::new(&gguf, &config, &weights, &placement, None)?;
        let mut pinned = 0;
        let mut far_bytes = 0;
        let mut declared = Vec::with_capacity(devices.len());
        for (index, device) in devices.iter().enumerate() {
            let PartPlan { role, first, shares, head } = placement.part(&config, index);
            let layers = shares.len();
            // The part that holds the n-gram block drives the feed of its rows,
            // whose checkpoint keeps what the feed knows of a sequence.
            let feeds = role != super::Role::Worker
                && config
                    .ngram
                    .as_ref()
                    .is_some_and(|g| (first..first + layers).contains(&g.layer));
            let shape = PartShape { role, first, shares, head };
            let frame = Frame::new(&config, &opts.part, &shape, device.info().multiprocessors)?;
            far_bytes += frame.page_parts().0;
            let mut part = frame.declared(&opts.part);
            if feeds {
                part.checkpoint += ngram::saved_bytes(config.ngram.as_ref());
            }
            declared.push(part);
            let shares = if opts.serve == Serve::Stream {
                0
            } else {
                (layers * layout.per_layer) as u64
                    * layout.shapes[0].host_share_bytes(opts.part.max_rows.min(MAX_BATCH))
            };
            // The entries the store copies through are a ring of 256 KiB.
            pinned += frame.pinned_bytes(&opts.part, layers) + shares + (256 << 10);
        }
        if opts.part.prune.is_some() {
            pinned += PruneMask::bytes(config.n_layer, config.experts.count);
        }
        let n = placement.n_devices();
        let text = ModelText::new(name.into(), context, tokenizer, config.n_embd as u32);
        let gguf = Arc::new(gguf);
        let rows = Arc::new(HostTable::mapped(gguf.clone(), TOKEN_EMBD)?);
        let config_vocab = config.n_vocab as u32;
        let near = opts.part.near_pages();
        let far =
            (near < opts.part.pages()).then_some(FarPages { first: near, page_bytes: far_bytes });
        Ok(Self {
            gguf,
            config,
            opts,
            placement,
            layout,
            text,
            meeting: parts::meeting(n),
            flow: OnceLock::new(),
            reports: Mutex::new(Vec::new()),
            pinned,
            page_bytes,
            far,
            declared,
            mtp: None,
            dflash: None,
            mtp_report: Mutex::new(None),
            mtp_watch: Arc::default(),
            usage: Usage::default(),
            rows,
            lone_rows: OnceLock::new(),
            pass_span: OnceLock::new(),
            view: HeadView::new(config_vocab, config_vocab, None).map_err(OpenError::Choice)?,
        })
    }

    /// Reads images through `projector`, whose tower runs on device part
    /// `device`. Refused for a projector whose rows are not the model's width,
    /// and for a model whose n-gram embedding hashes another token for an image
    /// cell than the pad token the chat's markers write. Before the runtime
    /// opens the parts.
    pub fn with_projector(
        mut self,
        projector: Arc<Projector>,
        device: usize,
    ) -> Result<Self, OpenError> {
        // The n-gram embedding hashes its image token for an image cell; the
        // record carries the pad token the chat's markers write there.
        let pad = self.text.image_pad();
        if let Some(ngram) = &self.config.ngram
            && pad != Some(ngram.image_token)
        {
            return Err(OpenError::Choice(format!(
                "the n-gram embedding hashes token {} for an image cell, and the chat writes \
                 {pad:?} there",
                ngram.image_token
            )));
        }
        let devices = self.placement.n_devices();
        self.text
            .set_projector(projector, device, devices)
            .map_err(|why| OpenError::Choice(format!("the projector: {why}")))?;
        Ok(self)
    }

    /// The projector images are read through, if there is one.
    pub fn projector(&self) -> Option<&Arc<Projector>> {
        self.text.projector()
    }

    /// Gives every drafter of the model a draft vocabulary (`blocks/head.rs`):
    /// the output head's rows of its tokens come first, the drafters read only
    /// them, and the device of the head keeps the rows of the embedding table
    /// of only them. Before the runtime opens the parts.
    pub fn with_draft_vocabulary(mut self, tokens: &[u32]) -> Result<Self, OpenError> {
        let n = self.config.n_vocab as u32;
        self.view = HeadView::new(n, n, Some(tokens)).map_err(OpenError::Choice)?;
        self.opts.part.head_order = self.view.order.clone();
        Ok(self)
    }

    /// Keeps the usage profile of the model's experts in `dir` (see
    /// `usage.rs`): a profile that is there for this model file fills the
    /// tiers when the engine begins to serve ([`Qwen4Exp::serving`]), and the
    /// profile is written while the engine serves and when the model is
    /// dropped. `log` gets what the owner should hear of (a profile that
    /// cannot be written). Before the runtime opens the parts.
    pub fn with_usage_profile(mut self, dir: &std::path::Path, log: Log) -> Self {
        self.usage = Usage::open(dir, &self.gguf, &self.config, log);
        self
    }

    /// The engine serves from now on: the heat of the experts becomes the
    /// usage profile's (the steps that measured the price of a step are no
    /// text), the tiers are filled from it, and the profile is kept, the
    /// empty slots filled whenever nothing runs. Call it once the runtime
    /// runs and before the first request; it returns when the tiers are
    /// filled, with what was found and filled; `None` without a profile, or
    /// before the runtime has opened the parts.
    pub fn serving(&self) -> Option<ProfileReport> {
        self.usage.serving()
    }

    /// Writes the usage profile now if steps ran since it was last written:
    /// its path, or why it could not be written; `None` when there was
    /// nothing to write.
    pub fn save_profile(&self) -> Option<Result<std::path::PathBuf, String>> {
        self.usage.save()
    }

    /// Fills the tiers from the heat the steps so far left, in one sweep
    /// through the file at the pace of a start, as a start from a profile
    /// does: after a warm-up, so that VRAM holds the hottest experts of every
    /// layer before the first request. Returns what it brought and how long
    /// it took; `None` without a usage profile or before the engine serves.
    /// Call it while no request runs.
    pub fn fill_from_heat(&self) -> Option<(FillStats, Duration)> {
        self.usage.fill_from_heat()
    }

    /// What the fill of the empty slots did while nothing ran, and what the
    /// tiers of the model's layers hold; `None` without a usage profile.
    pub fn usage_stats(&self) -> Option<UsageStats> {
        self.usage.stats()
    }

    /// Gives the model its MTP block from `draft`, a file of the block as
    /// Strata packs it ([`MtpWeights`]): windows then take the block's
    /// drafts, as many rows as `opts.part.window_rows` allows. The block runs
    /// on the device that holds the output head, in every placement, and its
    /// routed experts go through the expert store with the model's. Before the
    /// runtime opens the parts.
    /// Takes a DFlash drafter: windows then take its drafts as well, as many
    /// rows as `opts.part.window_rows` allows. It runs on the device that
    /// holds the streams of every layer it reads and the output head: one
    /// device, or the mixer of two devices on every layer. Before the runtime
    /// opens the parts.
    pub fn with_dflash(mut self, file: Arc<crate::dflash::File>) -> Result<Self, OpenError> {
        let choice = |why: String| Err(OpenError::Choice(why));
        if self.opts.part.window_rows < 2 {
            return choice("the DFlash drafter needs windows of at least two rows".into());
        }
        if self.opts.part.window_rows > MAX_BATCH {
            return choice(
                "a step that verifies drafts runs the decode kernels, which take eight rows".into(),
            );
        }
        let c = &self.config;
        if let Err(e) = file.config.fits(c.n_embd, c.n_layer, c.n_vocab) {
            return choice(e.to_string());
        }
        if self.placement.tensor.is_none() && self.placement.n_devices() > 1 {
            return choice(
                "the DFlash drafter is not built for a split by layers: its taps would have to \
                 cross to the device of the head"
                    .into(),
            );
        }
        self.dflash = Some(file);
        Ok(self)
    }

    pub fn with_mtp(mut self, file: Gguf) -> Result<Self, OpenError> {
        let choice = |why: &str| Err(OpenError::Choice(why.to_owned()));
        if self.opts.part.window_rows < 2 {
            return choice("an MTP block needs windows of at least two rows");
        }
        if self.opts.part.window_rows > MAX_BATCH {
            return choice(
                "a step that verifies drafts runs the decode kernels, which take eight rows",
            );
        }
        let (c, n) = (&self.config, self.placement.n_devices());
        let head = MtpWeights::load(&file, c)?;
        let weights = Weights::load(&self.gguf, c)?;
        let device = self.head_device();
        let layer = MtpExperts { gguf: &file, weights: &head, device };
        let (layout, _map) = Layout::new(&self.gguf, c, &weights, &self.placement, Some(&layer))?;
        // What the block keeps of a sequence (its slot and the carry) and of
        // a page (its K/V group and pooled keys), next to the head's part.
        let block = mtp_config(c);
        let block_opts = Options {
            kv: MTP_KV,
            max_rows: MAX_BATCH,
            logit_rows: MAX_BATCH,
            window_rows: 0,
            host_logits: false,
            profile: None,
            ..self.opts.part.clone()
        };
        let frame = Frame::new(&block, &block_opts, &PartShape::whole(&block, 0..1, true), 1)?;
        let mut declared = frame.declared(&block_opts);
        let part = &mut self.declared[device];
        part.checkpoint += declared.checkpoint + c.wide() as u64 * 4;
        part.groups.append(&mut declared.groups);
        part.arrays += declared.arrays;
        // The block's host share and its staged rows and picks.
        if self.opts.serve != Serve::Stream {
            self.pinned += layout.shapes[0].host_share_bytes(MAX_BATCH);
        }
        self.pinned += u64::from(MAX_BATCH) * c.wide() as u64 * 4 + (64 << 10);
        self.layout = layout;
        debug_assert!(device < n);
        drop((head, weights));
        self.mtp = Some(Arc::new(file));
        Ok(self)
    }

    /// The device part that holds the final mixer and the output head.
    fn head_device(&self) -> usize {
        (0..self.placement.n_devices())
            .find(|d| self.placement.part(&self.config, *d).head)
            .unwrap_or(0)
    }

    /// What the MTP block took on its device, once the runtime has opened
    /// the parts; `None` for a model without one.
    pub fn mtp_report(&self) -> Option<MtpReport> {
        *self.mtp_report.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the MTP block's steps took and how its drafts fared since the
    /// engine started.
    pub fn mtp_times(&self) -> MtpTimes {
        self.mtp_watch.lock().unwrap_or_else(|e| e.into_inner()).times
    }

    /// The newest chains the MTP block made, oldest first (the last 1024),
    /// and forgets them: for tests and measurements that compare its drafts
    /// with the reference's.
    pub fn take_mtp_chains(&self) -> Vec<MtpDrafted> {
        self.mtp_watch.lock().unwrap_or_else(|e| e.into_inner()).recent.drain(..).collect()
    }

    /// Renders chat prompts with `options`: whether the model thinks, and the
    /// variables a server passes to the chat template.
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.text.set_chat_options(options);
        self
    }

    /// The model's configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// What the parts took, by device, once the runtime has opened them.
    pub fn reports(&self) -> Vec<PartReport> {
        let mut reports = self.reports.lock().unwrap_or_else(|e| e.into_inner()).clone();
        reports.sort_by_key(|r| r.index);
        reports
    }

    /// What the wide steps found missing and waited for since the engine started; `None`
    /// before the parts are open.
    pub fn wide_stats(&self) -> Option<WideStats> {
        let flow = self.flow.get()?;
        flow.fold_wide();
        Some(*flow.wide.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// The counters of the reader of the model file (reads by class, bytes, latencies); `None`
    /// before the parts are open.
    pub fn read_stats(&self) -> Option<ares_io::Stats> {
        Some(self.flow.get()?.lock_feed().store.reader_stats())
    }

    /// How often the predictions of steps of up to eight rows named an expert that no tier
    /// held and that was then picked, by kind of step, distance and place (see
    /// `ExpertStore::ahead_odds`); `None` before the parts are open.
    pub fn ahead_odds(&self) -> Option<[[(f32, f32); 16]; 4]> {
        Some(self.flow.get()?.lock_feed().store.ahead_odds())
    }

    /// How often the guesses of wide steps were then named, by class (see
    /// `ExpertStore::transit_odds`); `None` before the parts are open.
    pub fn guess_odds(&self) -> Option<[(f32, f32); 9]> {
        Some(self.flow.get()?.lock_feed().store.transit_odds())
    }

    /// What the expert store counted since the last call; `None` before the
    /// parts are open.
    pub fn take_store_stats(&self) -> Option<StoreStats> {
        let flow = self.flow.get()?;
        flow.fold_wide();
        let (stats, ram_bytes, held) = {
            let mut feed = flow.lock_feed();
            let held = self.layout.model_layers().fold((0, 0), |(vram, ram), at| {
                let (in_vram, in_ram) = feed.store.held(at);
                (vram + in_vram as u64, ram + in_ram as u64)
            });
            (feed.store.take_stats(), feed.store.ram_bytes().0, held)
        };
        let mut totals = flow.totals.lock().unwrap_or_else(|e| e.into_inner());
        totals.ram_tier_bytes = ram_bytes;
        (totals.vram_experts, totals.ram_experts) = held;
        totals.picks += stats.picks;
        totals.hits += stats.hits;
        totals.from_ram += stats.miss_from_ram;
        totals.from_disk += stats.miss_from_disk;
        totals.on_their_way += stats.miss_on_its_way;
        totals.cpu_experts += stats.cpu_experts;
        totals.uploads += stats.uploads + stats.transit_uploads;
        totals.read_bytes +=
            stats.demand_read_bytes + stats.guess_read_bytes + stats.transit_read_bytes;
        totals.reads += stats.demand_reads
            + stats.guess_reads
            + stats.transit_reads
            + stats.transit_ahead_reads;
        totals.demand_reads += stats.demand_reads;
        totals.demand_read_us += (stats.demand_read_ms * 1e3) as u64;
        totals.wait_us += (stats.wait_ms * 1e3) as u64;
        totals.predicted += stats.predicted;
        totals.predicted_picked += stats.predicted_picked;
        // Wide steps read their predicted picks ahead through transit's guesses.
        totals.ahead_reads += stats.ahead_reads + stats.transit_ahead_reads;
        totals.ahead_read_bytes += stats.ahead_read_bytes;
        totals.ahead_used += stats.ahead_used + stats.transit_ahead_used;
        totals.ahead_wasted += stats.ahead_wasted + stats.transit_ahead_unused;
        totals.transit_admitted += stats.transit_admitted;
        totals.transit_adopted += stats.transit_adopted;
        totals.transit_guess_uploads += stats.transit_guess_uploads;
        totals.transit_guess_uploads_used += stats.transit_guess_uploads_used;
        Some(stats)
    }

    /// The slots of the experts in VRAM and in the RAM tier, over all layers,
    /// and the bytes of the RAM tier, as they are now (the RAM tier follows
    /// the memory the machine has); `None` before the parts are open.
    pub fn expert_slots(&self) -> Option<(usize, usize, u64)> {
        let flow = self.flow.get()?;
        let feed = flow.lock_feed();
        let (vram, ram) = self.layout.model_layers().fold((0, 0), |(v, r), at| {
            let (vram, ram) = feed.store.slots(at);
            (v + vram, r + ram)
        });
        let ram_bytes = feed.store.ram_bytes().0;
        Some((vram, ram, ram_bytes))
    }

    /// What the expert store measured of serving a miss on this machine: per
    /// pair of types of the experts (gate and up, down), the seconds a CPU job
    /// takes per (expert, row) pair for one row and for eight, and per device
    /// the seconds of one upload (`None` before one was timed). `None` before
    /// the parts are open or without a CPU half.
    pub fn expert_costs(&self) -> Option<ExpertCosts> {
        let flow = self.flow.get()?;
        let feed = flow.lock_feed();
        let model = feed.store.cost_model()?;
        let cpu = model
            .timed_types()
            .into_iter()
            .map(|types| {
                let per = |pairs: usize| model.job_time(types, pairs).map(|t| t / pairs as f64);
                (types, per(1), per(8))
            })
            .collect();
        let devices = self.placement.n_devices();
        let upload = (0..devices).map(|d| model.upload_time(d)).collect();
        Some(ExpertCosts { cpu, upload })
    }

    /// The slots of the MTP block's experts in VRAM and in the RAM tier;
    /// `None` without an MTP block or before the parts are open.
    pub fn mtp_slots(&self) -> Option<(usize, usize)> {
        self.mtp.as_ref()?;
        Some(self.flow.get()?.lock_feed().store.slots(0))
    }

    /// Brings the experts `experts_of(layer)` names for every layer into the
    /// tiers before anything is served, most wanted first, as far as the slots
    /// go (a recorded routing, or every expert for a model that fits); the
    /// MTP block is asked for as the layer behind the model's last. Call it
    /// while no request runs; `None` before the parts are open.
    pub fn fill(
        &self,
        experts_of: &mut dyn FnMut(usize) -> Vec<u32>,
    ) -> Option<Result<Fill, ModelError>> {
        let flow = self.flow.get()?;
        let placement = &self.placement;
        let c = &self.config;
        Some(
            flow.lock_feed()
                .fill(
                    c.n_layer,
                    c.experts.count,
                    &|layer, expert| placement.owner(layer, expert),
                    experts_of,
                )
                .map_err(ModelError::from),
        )
    }
}

impl Qwen4Exp {
    /// Whether the model on `devices` devices reads a prompt wider than the parts' own
    /// scratch layer by layer: asked for, with micro-batches the many-row kernels take, K and
    /// V without an exact pool (whose tiles convert per step for every layer at once), no
    /// DFlash drafter (which reads the residual of a step's rows at some layers), and a
    /// placement the pass is built for.
    fn reads_layer_by_layer(&self) -> bool {
        self.opts.pass_rows > self.opts.part.own_rows()
            && reads_layer_by_layer(&self.opts.part, self.dflash.is_some())
    }

    /// Everything besides the tokens that changes the bytes of a page or of a
    /// checkpoint as the device parts hand them out
    /// ([`ares_runtime::Durable::fingerprint`]):
    ///
    /// - the model's files and the MTP block's (size, header and samples
    ///   of the weights);
    /// - the formats of K and V, of the pooled keys and of the recurrent
    ///   state, and how the products of a prompt round;
    /// - what each device holds of every layer (its role, its heads, the
    ///   shared expert), since the bytes are laid out per device.
    ///
    /// How a missed expert is served is not in it: the CPU's sums of a miss
    /// differ from a device's by rounding, as two runs differ in their misses.
    fn state_fingerprint(&self) -> Vec<u8> {
        let mut out = b"qwen4exp state 1\0".to_vec();
        for part in 0..self.gguf.n_parts() {
            crate::files::identity(self.gguf.path(part as u16), &mut out);
        }
        if let Some(draft) = &self.mtp {
            out.extend_from_slice(b"mtp block\0");
            for part in 0..draft.n_parts() {
                crate::files::identity(draft.path(part as u16), &mut out);
            }
        }
        let o = &self.opts.part;
        let settings = format!(
            "kv {:?} keys {:?} state {:?} precision {:?} mtp {:?}",
            o.kv,
            o.keys,
            o.state,
            o.precision,
            self.mtp.as_ref().map(|_| MTP_KV)
        );
        out.extend_from_slice(settings.as_bytes());
        for device in 0..self.placement.n_devices() {
            let PartPlan { role, first, shares, head } = self.placement.part(&self.config, device);
            out.push(0xff);
            out.extend_from_slice(format!("{role:?} {first} {head}").as_bytes());
            for share in shares {
                out.extend_from_slice(&share.heads.start.to_le_bytes());
                out.extend_from_slice(&share.heads.end.to_le_bytes());
                out.push(u8::from(share.shared));
            }
        }
        out
    }
}

impl Model for Qwen4Exp {
    fn card(&self) -> ModelCard {
        self.text.card()
    }

    fn vocab(&self) -> u32 {
        self.config.n_vocab as u32
    }

    fn ends(&self, token: u32) -> bool {
        self.text.ends(token)
    }

    fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>, EngineError> {
        self.text.encode(prompt)
    }

    fn encode_with_boundaries(
        &self,
        prompt: &Prompt,
    ) -> Result<ares_runtime::EncodedPrompt, EngineError> {
        self.text.encode_with_boundaries(prompt)
    }

    fn vision(&self) -> Option<Vision> {
        self.text.vision()
    }

    fn chat_format(&self) -> Option<ChatFormat> {
        Some(self.text.format.clone())
    }

    fn piece(&self, token: u32, text: &mut Vec<u8>) {
        self.text.piece(token, text);
    }

    fn context(&self) -> ContextSpec {
        ContextSpec::NONE
    }

    fn split(&self) -> Split {
        match (&self.placement.tensor, self.placement.n_devices()) {
            (Some(_), _) => Split::Shared,
            (None, 1) => Split::Single,
            (None, _) => Split::Layer,
        }
    }

    fn devices(&self) -> usize {
        self.placement.n_devices()
    }

    fn links(&self) -> LinkSpec {
        // A split by layers hands the residual of all rows of a step over at
        // once, in a ring of two parcels; the devices that share every layer
        // write to each other themselves.
        match self.split() {
            Split::Layer => LinkSpec {
                sum_values: 0,
                lane_slots: pass::PARCELS as u32,
                lane_values: self.opts.part.max_rows * self.config.wide() as u32,
            },
            _ => LinkSpec::default(),
        }
    }

    fn saves_inside_steps(&self) -> u32 {
        // A pass saves each layer's state as it goes past a checkpoint, between two
        // micro-batches, each wider than the decode kernels take.
        match self.pass_span.get() {
            Some(&span) if span > 0 => 2 * MAX_BATCH,
            _ => 0,
        }
    }

    fn steps_wait_for_fetches(&self) -> bool {
        // An expert that no tier in VRAM holds is brought while the step
        // waits for it.
        true
    }

    fn pauses_wide_steps(&self) -> bool {
        // A pass pauses between two micro-batches once the record's time has
        // passed.
        self.pass_span.get().is_some_and(|&span| span > 0)
    }

    fn pinned_ram(&self) -> u64 {
        self.pinned + self.text.pinned_ram()
    }

    fn expert_stats(&self) -> Option<ExpertStats> {
        self.take_store_stats()?;
        let totals = self.flow.get()?.totals.lock().unwrap_or_else(|e| e.into_inner());
        Some(*totals)
    }

    fn max_rows(&self) -> u32 {
        // Once the parts are open: the rows the slots can lend for, a prompt read layer by
        // layer's or a lent step's.
        match (self.pass_span.get(), self.lone_rows.get()) {
            (Some(&pass), _) if pass > 0 => pass,
            (_, Some(&lone)) => lone,
            _ => self.opts.part.max_rows.max(self.opts.pass_rows),
        }
    }

    fn prompt_rows(&self) -> u32 {
        self.opts.part.own_rows()
    }

    fn own_rows(&self) -> u32 {
        self.opts.part.own_rows()
    }

    fn logit_rows(&self) -> u32 {
        self.opts.part.logit_rows
    }

    fn draft_rows(&self) -> u32 {
        self.opts.part.window_rows
    }

    fn drafters(&self) -> Vec<Drafter> {
        let dflash = self.dflash.as_ref().map(|file| file.config.drafts() as u32);
        step::drafters(self.mtp.is_some(), dflash, self.opts.part.window_rows)
    }

    fn attention_span(&self) -> u32 {
        self.config.indexer.dense_limit() as u32
    }

    fn prefix_cache(&self) -> Option<PrefixCache> {
        Some(slots::prefix_cache(&self.declared, Some(TAILS_PER_SLOT)))
    }

    fn durable(&self, cells_per_page: u32) -> Option<Durable> {
        (cells_per_page == ENTRY_CELLS)
            .then(|| slots::durable(&self.declared, self.state_fingerprint()))
    }

    fn page_entry(&self, page: u32) -> u32 {
        page | PAGE_ENTRY
    }

    fn far_pages(&self) -> Option<FarPages> {
        self.far
    }

    fn before_step(&self, record: &RecordView<'_>) -> Result<(), ModelError> {
        match self.flow.get() {
            Some(flow) => flow.before_step(record),
            None => Ok(()),
        }
    }

    fn page_layout(&self, cells_per_page: u32) -> PageLayout {
        // The attention kernels store 128 cells per entry; `open` refuses any
        // other page size.
        (0..self.placement.n_devices())
            .fold(PageLayout::new(self.placement.n_devices(), cells_per_page), |layout, device| {
                layout.per_page(device, "K, V and keys", self.page_bytes[device])
            })
    }

    fn open(&self, cx: DeviceContext<'_>) -> Result<Box<dyn DeviceModel>, ModelError> {
        self.open_part(cx).inspect_err(|error| self.meeting.fail(error.to_string()))
    }
}
