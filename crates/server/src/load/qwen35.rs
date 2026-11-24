//! Loading a `qwen35` model (Qwen3.5, Qwen3.8 27B): dense, all of it on the
//! devices.
//!
//! The phases: what can be refused before a device is touched ([`Asked`]),
//! the devices, the split, the shares of a split, the context, the plan in
//! the log, the model built, the engine started ([`super::start`]), and what
//! each device took. Which devices, which split, which shares and how much
//! context are decided by an estimate of what each device takes
//! ([`Estimate`], in `ares-model`): the runtime reserves the pages of sequence
//! state before the model's parts are built, and a part's sizes cannot be
//! asked without a device. After the load the log states what each part
//! really took next to the estimate.

use std::sync::Arc;
use std::time::Instant;

use ares_core::units::Bytes;
use ares_cuda::CacheFormat;
use ares_gguf::Gguf;
use ares_model::qwen3vl::Projector;
use ares_model::qwen35::Config;
use ares_model::qwen35::device::{
    EmbeddingReport, Estimate, Footprint, Options, Placement, Qwen35, Spread, fit_shares,
};

use super::devices::{
    Held, bandwidth, describe_links, free_of, hold_devices, multiprocessors_of, ordinals, tower_of,
};
use super::files::{chat_options, drafter_files, mtp_takes_part, open_projector, read_vocabulary};
use super::ram::{Budget, Rest};
use super::start::{self, Engine, log_started};
use super::{LoadError, Loaded, VRAM_HEADROOM, join, log_context, refuse, shares_text, short};
use crate::models::Site;
use crate::plan::{
    self, AttentionShares, CELLS_PER_PAGE, DEFAULT_PARALLEL, DEFAULT_WINDOW, LoadOptions, Room,
    SINGLE_DEVICE_MIN_CONTEXT, SplitKind,
};

fn model_error(e: impl ToString) -> LoadError {
    LoadError::Model(e.to_string())
}

/// Loads a `qwen35` model from `gguf`, which is open, with `options` at
/// `site`; `begun` is when the load began.
pub(super) fn load(
    gguf: Gguf,
    options: &LoadOptions,
    site: &Site,
    begun: Instant,
) -> Result<Loaded, LoadError> {
    let config = Estimate::new(&gguf).map_err(model_error)?.config().clone();
    tracing::info!(
        "model: {} blocks of width {}, {} tokens, trained for a context of {}, {}",
        config.n_layer,
        config.n_embd,
        config.n_vocab,
        config.n_ctx,
        if config.n_nextn == 1 { "with an MTP block" } else { "no MTP block" },
    );
    let asked = Asked::read(&gguf, &config, options)?;
    let parallel = asked.parallel;

    // The devices. The image tower's VRAM is held back on the device that
    // runs it before anything is planned there.
    let (ordinals, mut held) = hold_devices(options)?;
    let tower_bytes = asked.projector.as_ref().map_or(0, Projector::vram_bytes);
    let mut tower_at = asked.projector.as_ref().map(|p| tower_of(&held, p.home()));
    if let Some(t) = tower_at {
        held[t].free = held[t].free.saturating_sub(tower_bytes);
    }
    let part_options = asked.part_options(options)?;
    let plan = Planning {
        estimate: Estimate::for_options(&gguf, &part_options).map_err(model_error)?,
        part: &part_options,
        parallel,
        smallest: options.context.unwrap_or(SINGLE_DEVICE_MIN_CONTEXT.min(config.n_ctx as u32)),
    };

    // Which devices, and how the model is spread over them.
    let (split, why_split) = plan.split(options, &held, ordinals.len())?;
    if split == SplitKind::Single {
        if options.shares.is_some() {
            return Err(refuse("--shares is for a split over several devices, and there is one"));
        }
        if held.len() > 1 {
            let chosen = plan.single_device(&held)?;
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
    let (spread, why_shares) = plan.spread(options, &config, site, &held, split)?;
    describe_links(&held)?;
    let placement = plan.estimate.place(&spread, &part_options).map_err(model_error)?;
    if asked.vocabulary.is_some() && placement.draft.is_none() {
        return Err(refuse(
            "a draft vocabulary is for the model's MTP block, which is not built for a layer \
             split",
        ));
    }
    let footprint = plan.footprint_of(&placement, &held);
    let fit = plan.pages_that_fit(&held, &footprint);

    // Context: tokens of state for all sequences together.
    let (context, why_context) = plan.context(options, &config, site, &held, &footprint, fit)?;
    let longest = context.min(config.n_ctx as u32);
    let pages = plan::pool_pages(context, parallel);
    log_split(&held, split, why_split, &spread, why_shares);
    log_context(options, context, &why_context, pages, longest, parallel);
    log_drafting(options, &asked, &held, &placement);
    for (h, f) in held.iter().zip(&footprint) {
        tracing::info!(
            "estimate, device {}: weights {}, step scratch and small buffers {}, recurrent \
             state {} per sequence, K and V {} per 1000 tokens; with this plan {} of {} free",
            h.device.ordinal(),
            Bytes(f.weights),
            Bytes(f.step),
            Bytes(f.per_sequence),
            Bytes(f.per_page * 1000 / u64::from(CELLS_PER_PAGE)),
            Bytes(f.total(parallel, pages)),
            Bytes(h.free),
        );
    }
    drop(plan);

    // The model, built for the plan.
    let name = options.name.clone().unwrap_or_else(|| crate::preset::id_of_file(&options.model));
    let (window, drafts) = (asked.window, asked.drafts());
    let model =
        Qwen35::new(gguf, name.clone(), &spread, part_options, longest).map_err(model_error)?;
    let model = match (asked.projector, tower_at) {
        (Some(projector), Some(t)) => {
            tracing::info!(
                "plan: the image tower on device {}, {} of its VRAM held back for it",
                held[t].device.ordinal(),
                Bytes(tower_bytes)
            );
            model.with_projector(Arc::new(projector), t).map_err(model_error)?
        }
        _ => model,
    };
    let (chat, template) = chat_options(options, model.tokenizer())?;
    let model = Arc::new(model.with_chat_options(chat));

    // Host RAM: the step buffers, and where the token embedding table lies in
    // host memory, the table and the rows of a step. A limit below that is
    // refused: the table does not go back to a device, whose memory is what
    // bounds the context. The rest is the prompt cache's in RAM (the only
    // other user in a model without experts).
    let tower_ram = model.projector().map_or(0, |p| p.pinned_bytes());
    let table = ares_runtime::Model::pinned_ram(&*model) - tower_ram;
    let budget = Budget::of(&*model, options, site, parallel, (table, tower_ram), Rest::Cache)?;
    let engine = Engine {
        pages,
        parallel,
        window: if drafts { window } else { 1 },
        max_rows: options.batch.unwrap_or(0),
        context,
    };
    let devices = held.iter().map(|h| h.device.clone()).collect();
    let runtime = start::start(&model, devices, options, site, &budget, engine, template, begun)?;
    log_started(&runtime, budget.ssd.as_ref(), &ares_runtime::Model::drafters(&*model));
    log_taken(&model, &held, &footprint, parallel, pages)?;
    tracing::info!(
        "context: {context} tokens of sequence state, all of it in VRAM (state is not moved to \
         RAM yet, so a context that does not fit is refused at load)"
    );
    Ok(Loaded::new(runtime, name, held, budget.ram))
}

/// What the operator asked for, read and checked before a device is touched.
struct Asked {
    kv: (CacheFormat, CacheFormat),
    parallel: u32,
    /// The most rows of a sequence's window.
    window: u32,
    /// Whether the model's MTP block drafts.
    mtp: bool,
    /// The MTP block's own file, when `--drafters` names one.
    mtp_file: Option<Arc<Gguf>>,
    dflash: Option<Arc<ares_model::dflash::File>>,
    vocabulary: Option<Arc<[u32]>>,
    projector: Option<Projector>,
}

impl Asked {
    /// Reads and checks what `options` ask of the model of `gguf`.
    fn read(gguf: &Gguf, config: &Config, options: &LoadOptions) -> Result<Self, LoadError> {
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
        // The model drafters that take part: all the model has unless the
        // operator names them, the MTP block of the model file or of a file of
        // its own. Nothing speculates without one, the n-gram drafter included.
        let files = drafter_files(options, ares_model::qwen35::MTP_ARCHITECTURE)?;
        let mtp_file = match files.mtp {
            Some((path, file)) => {
                // Read before a device is touched: a block of another model
                // is refused by name.
                Estimate::new(gguf)
                    .and_then(|estimate| estimate.with_mtp_file(&file).map(drop))
                    .map_err(|e| refuse(format!("--drafters {}: {e}", path.display())))?;
                tracing::info!("drafter: the MTP block of {}", path.display());
                Some(Arc::new(file))
            }
            None => None,
        };
        let mtp = mtp_file.is_some()
            || mtp_takes_part(
                options,
                config.n_nextn == 1,
                "this model file has no MTP block (multi-token prediction); a file of one is \
                 named by its path",
            )?;
        // A DFlash drafter from a file: it reads taps of the model's layers on
        // the device that embeds and holds the first rows of the head, which a
        // layer split does not give.
        let dflash = files.dflash;
        if let Some(file) = &dflash {
            if options.split == Some(SplitKind::Layer) {
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
                    "a draft vocabulary is for the model's MTP block, and no drafter of that \
                     kind takes part",
                ));
            }
            Some(path) => Some(read_vocabulary(path)?),
            None => None,
        };
        if options.context == Some(0) {
            return Err(refuse("--context must be at least 1"));
        }
        let projector = open_projector(options, config.n_embd)?;
        Ok(Self { kv, parallel, window, mtp, mtp_file, dflash, vocabulary, projector })
    }

    /// Whether a model drafter takes part.
    fn drafts(&self) -> bool {
        self.mtp || self.dflash.is_some()
    }

    /// The model's settings, which do not depend on the placement. A dense
    /// model runs every step in its own memory: a step of prompt has at most
    /// the rows of its scratch, which must hold the windows of every sequence.
    fn part_options(&self, options: &LoadOptions) -> Result<Options, LoadError> {
        let parallel = self.parallel;
        let window_rows = if self.drafts() { parallel.saturating_mul(self.window) } else { 0 };
        let defaults = Options::default();
        let part = Options {
            precision: options.precision,
            kv: self.kv,
            max_rows: options.ubatch.unwrap_or(defaults.max_rows),
            logit_rows: window_rows.max(parallel),
            window_rows,
            draft_vocabulary: self.vocabulary.clone(),
            mtp: self.mtp,
            mtp_file: self.mtp_file.clone(),
            dflash: self.dflash.clone(),
            ..defaults
        };
        if let Some(batch) = options.batch.filter(|b| *b > part.max_rows) {
            return Err(refuse(format!(
                "--batch-size {batch}: this model runs no step wider than its scratch of {} rows \
                 (--ubatch-size); only a mixture of experts lends VRAM for wider ones",
                part.max_rows
            )));
        }
        if part.max_rows < window_rows {
            return Err(refuse(format!(
                "{parallel} sequences with windows of {} rows are {window_rows} rows in a step, \
                 more than the {} a step has: lower --parallel or --window",
                self.window, part.max_rows
            )));
        }
        Ok(part)
    }
}

/// What a load plans with: the model's [`Estimate`], the options its parts
/// are opened with, the sequences that run at once, and the least context a
/// device is chosen for (the one asked for, else
/// [`SINGLE_DEVICE_MIN_CONTEXT`] or the model's own if that is less).
struct Planning<'a> {
    estimate: Estimate<'a>,
    part: &'a Options,
    parallel: u32,
    smallest: u32,
}

impl Planning<'_> {
    /// What each of `held` takes with the model placed as `placement`.
    fn footprint_of(&self, placement: &Placement, held: &[Held]) -> Vec<Footprint> {
        let sms = multiprocessors_of(held);
        self.estimate.footprint(placement, self.part, self.parallel, &sms)
    }

    /// The pages of sequence state that fit on `held` next to `footprint`.
    fn pages_that_fit(&self, held: &[Held], footprint: &[Footprint]) -> u32 {
        let rooms: Vec<Room> = free_of(held)
            .iter()
            .zip(footprint)
            .map(|(free, f)| Room {
                usable: free.saturating_sub(VRAM_HEADROOM),
                fixed: f.weights + f.step + f.exact,
                per_sequence: f.per_sequence,
                per_page: f.per_page,
            })
            .collect();
        plan::pages_that_fit(&rooms, self.parallel)
    }

    /// Whether `held` hold the model spread as `spread` with the least
    /// context a device is chosen for.
    fn holds(&self, held: &[Held], spread: &Spread) -> Result<bool, LoadError> {
        let placement = self.estimate.place(spread, self.part).map_err(model_error)?;
        let fit = self.pages_that_fit(held, &self.footprint_of(&placement, held));
        Ok(fit >= plan::pool_pages(self.smallest, self.parallel))
    }

    /// The split, as asked, or one device if one holds the model with a
    /// useful context, else a tensor split; with why, for the log.
    fn split(
        &self,
        options: &LoadOptions,
        held: &[Held],
        named: usize,
    ) -> Result<(SplitKind, &'static str), LoadError> {
        let split = match options.split {
            Some(SplitKind::Single)
                if options.devices.as_ref().is_some_and(|list| list.len() > 1) =>
            {
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
                let mut alone = false;
                for candidate in held {
                    alone |= self.holds(std::slice::from_ref(candidate), &Spread::Single)?;
                }
                if alone { SplitKind::Single } else { SplitKind::Tensor }
            }
        };
        let why = match (options.split, split) {
            (Some(_), _) => "as asked",
            (None, SplitKind::Single) if named == 1 => "one device to place it on",
            (None, SplitKind::Single) => "one device holds the model and its context",
            (None, _) => "no single device holds the model and its context",
        };
        Ok((split, why))
    }

    /// The device of `held` the model goes on alone: the first, in the order
    /// given, that holds it; the first one if none does, so that its ledger
    /// says what is missing.
    fn single_device(&self, held: &[Held]) -> Result<usize, LoadError> {
        for (i, candidate) in held.iter().enumerate() {
            if self.holds(std::slice::from_ref(candidate), &Spread::Single)? {
                return Ok(i);
            }
        }
        Ok(0)
    }

    /// How `split` spreads the model over `held`: the shares of the weights
    /// as asked or by the devices' memory bandwidth, lowered where a device's
    /// memory is too small for its share, and under a tensor split the shares
    /// of the attention heads, given or placed for the context; with why the
    /// shares are what they are, for the log.
    fn spread(
        &self,
        options: &LoadOptions,
        config: &Config,
        site: &Site,
        held: &[Held],
        split: SplitKind,
    ) -> Result<(Spread, &'static str), LoadError> {
        if split == SplitKind::Single {
            return Ok((Spread::Single, ""));
        }
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
            None => self.shares_by_bandwidth(site, held)?,
        };
        match (split, &options.attention) {
            (SplitKind::Tensor, AttentionShares::Given(attention))
                if attention.len() != held.len() =>
            {
                Err(refuse(format!(
                    "{} shares of the attention heads for {} devices",
                    attention.len(),
                    held.len()
                )))
            }
            (SplitKind::Tensor, AttentionShares::Given(attention)) => {
                Ok((Spread::TensorWithAttention(shares, attention.clone()), why))
            }
            (SplitKind::Tensor, AttentionShares::ByContext) => {
                let wanted = options.context.unwrap_or(config.n_ctx as u32);
                let attention = self
                    .estimate
                    .attention_for_context(
                        self.part,
                        self.parallel,
                        &free_of(held),
                        &multiprocessors_of(held),
                        &shares,
                        VRAM_HEADROOM,
                        plan::pool_pages(wanted, self.parallel),
                    )
                    .map_err(model_error)?
                    .ok_or_else(|| {
                        refuse(
                            "no device has room for the attention heads next to its share of \
                             the weights",
                        )
                    })?;
                Ok((Spread::TensorWithAttention(shares, attention), why))
            }
            _ => Ok((Spread::Layer(shares), why)),
        }
    }

    /// The shares of the weights by the memory bandwidth the devices report,
    /// lowered where a device's memory is too small for its share of a model
    /// that is shared out evenly in everything: weights, state and context.
    fn shares_by_bandwidth(
        &self,
        site: &Site,
        held: &[Held],
    ) -> Result<(Vec<f64>, &'static str), LoadError> {
        let wanted = plan::shares_by_bandwidth(
            &held.iter().map(|h| bandwidth(h.device.info())).collect::<Vec<_>>(),
        );
        let most_sms = held.iter().map(|h| h.device.info().multiprocessors).max();
        let whole = self.estimate.whole(
            self.part,
            self.parallel,
            plan::pool_pages(self.smallest, self.parallel),
            most_sms.unwrap_or(1),
        );
        let most: Vec<f64> = held
            .iter()
            .map(|h| {
                let usable = h.free.saturating_sub(VRAM_HEADROOM + whole.per_device);
                usable as f64 / whole.shared.max(1) as f64
            })
            .collect();
        match fit_shares(&wanted, &most) {
            Some(shares) if shares == wanted => {
                Ok((shares, "by the memory bandwidth the devices report"))
            }
            Some(shares) => Ok((
                shares,
                "by the memory bandwidth the devices report, lowered where a device's memory is \
                 too small for its share",
            )),
            None => {
                let error = refuse(format!(
                    "the model does not fit on the devices: it needs about {} with a context of \
                     {} tokens, and they have {} free",
                    Bytes(whole.shared + whole.per_device * held.len() as u64),
                    self.smallest,
                    Bytes(held.iter().map(|h| h.free.saturating_sub(VRAM_HEADROOM)).sum()),
                ));
                Err(short(site, error))
            }
        }
    }

    /// The context, as asked when `fit` pages hold it, else what fits of the
    /// model's own; with why, for the log. Next to other models (`site`) a
    /// context that does not fit is no room, which one of them makes.
    fn context(
        &self,
        options: &LoadOptions,
        config: &Config,
        site: &Site,
        held: &[Held],
        footprint: &[Footprint],
        fit: u32,
    ) -> Result<(u32, String), LoadError> {
        let parallel = self.parallel;
        let taken = |without_k_and_v: &str| {
            join(
                held.iter().zip(footprint).map(|(h, f)| {
                    format!(
                        "device {} has {} free and the model takes {}{without_k_and_v}",
                        h.device.ordinal(),
                        Bytes(h.free),
                        Bytes(f.fixed(parallel))
                    )
                }),
                ", ",
            )
        };
        match options.context {
            // The estimate decides before the weights are read: a plan is
            // never above what the devices have free. Next to other models
            // that is no room, which one of them makes; alone the context is
            // refused.
            Some(context) if fit < plan::pool_pages(context, parallel) => {
                let held_tokens = plan::pool_tokens(fit, parallel);
                if site.beside {
                    return Err(LoadError::NoRoom(format!(
                        "the devices cannot hold the model with its context of {context} tokens \
                         next to what is loaded (by estimate: {})",
                        taken(" without K and V")
                    )));
                }
                Err(refuse(format!(
                    "--context {context}: the devices hold about {held_tokens} tokens of context \
                     for {parallel} sequence{} with this plan (by estimate: {}; K and V take {} \
                     per 1000 tokens); name a smaller context, fewer sequences with --parallel, K \
                     and V in fewer bits with --kv{}",
                    if parallel == 1 { "" } else { "s" },
                    taken(" without K and V"),
                    join(
                        held.iter().zip(footprint).map(|(h, f)| format!(
                            "{} on device {}",
                            Bytes(f.per_page * 1000 / u64::from(CELLS_PER_PAGE)),
                            h.device.ordinal()
                        )),
                        " and "
                    ),
                    if options.shares.is_some() {
                        " or other shares of the weights with --shares (the heads of the \
                         attention blocks are placed for the context already)"
                    } else {
                        ""
                    },
                )))
            }
            Some(context) => Ok((context, "as asked".to_owned())),
            None => {
                let plan = plan::ContextPlan::new(None, config.n_ctx as u32, fit, parallel);
                let context = plan.context();
                // Next to other models a context below the least a device is
                // chosen for is no room: one of them makes way.
                if site.beside && context < self.smallest {
                    return Err(LoadError::NoRoom(format!(
                        "the devices have room for a context of {context} tokens next to what is \
                         loaded, less than the {} a model is loaded for (by estimate: {})",
                        self.smallest,
                        taken(" without K and V")
                    )));
                }
                if context == 0 {
                    return Err(refuse(format!(
                        "the devices have no room for sequence state next to the model (by \
                         estimate: {}); name fewer sequences with --parallel, another split, or \
                         a context with --context to try it anyway",
                        taken("")
                    )));
                }
                Ok((context, plan.why()))
            }
        }
    }
}

/// The plan's line of the split and its shares.
fn log_split(held: &[Held], split: SplitKind, why_split: &str, spread: &Spread, why_shares: &str) {
    tracing::info!(
        "plan: {} on device{} {} ({why_split}){}",
        match split {
            SplitKind::Single => "everything",
            SplitKind::Tensor => "tensor split",
            SplitKind::Layer => "layer split",
        },
        if held.len() == 1 { "" } else { "s" },
        ordinals(held),
        match spread {
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
}

/// The plan's line of the formats of K and V, the products and the drafters.
fn log_drafting(options: &LoadOptions, asked: &Asked, held: &[Held], placement: &Placement) {
    tracing::info!(
        "plan: K in {}, V in {}; products {}; {}",
        plan::format_name(asked.kv.0),
        plan::format_name(asked.kv.1),
        match options.precision {
            ares_cuda::Precision::Fast => "fast (8-bit activations for a few rows)",
            ares_cuda::Precision::Exact => "exact (f16 on the tensor cores for every product)",
        },
        match (asked.drafts(), &placement.draft) {
            (false, _) => match options.drafters {
                Some(_) => "no drafters (--drafters none): no speculation".to_owned(),
                None => "no model drafter (the file has no MTP block): no speculation".to_owned(),
            },
            (true, Some(draft)) => format!(
                "drafts from {} on device {} ({}) and from earlier text, windows of up to {} \
                 rows; whose drafts and how many are chosen for every step from the price of a \
                 step, measured at load, and the acceptance, measured while running",
                match (asked.mtp, &asked.dflash) {
                    (true, Some(_)) => "the model's MTP block and the DFlash drafter",
                    (true, None) => "the model's MTP block",
                    (false, _) => "the DFlash drafter",
                },
                held[draft.device].device.ordinal(),
                match &asked.vocabulary {
                    Some(list) => format!("a vocabulary of {} tokens", list.len()),
                    None => format!("the {} tokens whose head rows are there", draft.head_rows),
                },
                asked.window,
            ),
            (true, None) => {
                "no speculation: the model has no MTP block on a device of this placement"
                    .to_owned()
            }
        }
    );
}

/// What each device took, by the parts' reports and the device's own, next to
/// the estimate (`footprint`).
fn log_taken(
    model: &Qwen35,
    held: &[Held],
    footprint: &[Footprint],
    parallel: u32,
    pages: u32,
) -> Result<(), LoadError> {
    for ((h, report), f) in held.iter().zip(model.reports()).zip(footprint) {
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
        let really = taken.saturating_sub(kv_bytes);
        tracing::info!(
            "device {}: the model without K and V took {} (estimated {}); {} are free beyond \
             the {} left unreserved, room for about {} more tokens of context",
            h.device.ordinal(),
            Bytes(really),
            Bytes(f.fixed(parallel)),
            Bytes(free.saturating_sub(VRAM_HEADROOM)),
            Bytes(VRAM_HEADROOM),
            (free.saturating_sub(VRAM_HEADROOM) / report.page_bytes.max(1))
                .saturating_mul(u64::from(CELLS_PER_PAGE)),
        );
    }
    Ok(())
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
