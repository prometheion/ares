//! Loading a `qwen4exp` model (Qwen3.8-Flash-Next): a mixture of experts of
//! which only the dense parts fit in VRAM.
//!
//! The phases are those of the dense model (`qwen35.rs`): what can be refused
//! before a device is touched ([`Asked`]), the devices and the split, the
//! context, the rows of a step, the plan in the log, the model built, the
//! engine started ([`super::start`]), the tiers filled (`tiers.rs`), and what
//! each device took. What differs is what there is to decide: the dense parts
//! are small and sit on the devices; the rest of the VRAM holds slots for
//! experts, which follow the router, with the RAM the engine may use as a
//! tier in front of the file and the CPU computing what no tier brings in
//! time. So the plan is where the layers go (a range per device, or both
//! devices on every layer), how much context costs slots, and how much of the
//! RAM limit the tier takes. The model places itself on the devices and
//! measures what it must (`ares_model::qwen4exp::device::place`).

mod tiers;
mod watch;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use ares_core::units::Bytes;
use ares_cuda::{CacheFormat, Device};
use ares_gguf::Gguf;
use ares_model::qwen3vl::Projector;
use ares_model::qwen4exp::device::{
    ModelOptions, Options, Placement, PruneSetup, Qwen4Exp, RoleMemory, Spread, TensorChoice,
    batch_rows, drafter_bytes, page_parts, place, profile_key, reads_layer_by_layer, slot_floors,
    step_rows,
};
use ares_model::qwen4exp::weights::MTP_ARCHITECTURE;
use ares_model::qwen4exp::{Config, MtpWeights};
use ares_runtime::Model;
use ares_runtime::expert_store::ExpertNeighbours;
use ares_tokenizer::Tokenizer;

use super::devices::{Held, describe_links, hold_devices, ordinals, tower_of};
use super::files::{chat_options, drafter_files, open_projector, read_vocabulary};
use super::products::products_line;
use super::ram::{Budget, Rest};
use super::start::{self, Engine, log_started};
use super::{LoadError, Loaded, VRAM_HEADROOM, join, log_context, model_error, refuse, short};
use crate::models::Site;
use crate::plan::{self, CELLS_PER_PAGE, DEFAULT_PARALLEL, Drafter, LoadOptions, SplitKind};
use tiers::{fill_tiers, log_expert_costs, profile_dir};
use watch::{watch_drafts, watch_reads};

/// The most rows of a step that generates (the newest token of every sequence and the drafts
/// they verify): what the decode kernels of the experts take. Only the slices of a prompt are
/// wider.
const MAX_ROWS: u32 = ares_cuda::MAX_BATCH;

/// The rows of the step a prompt read alone takes unless `--batch-size` says
/// otherwise, as far as the expert slots can lend its scratch: every expert a
/// prompt picks is read once a step, so a long prompt reads the experts of
/// the file once per this many rows (chosen by measurement, see "Large prompt
/// steps on lent VRAM" in `docs/dev/architecture/flash-next-prompts.md`).
const DEFAULT_BATCH: u32 = 8192;

/// The same for a prompt read layer by layer, which keeps about 56 KiB of a row (Flash-Next)
/// where a step keeps 0.5 to 0.7 MiB: a long prompt reads each expert once per this many
/// rows (Strata's chunks of prompt reach the same).
const DEFAULT_PASS: u32 = 32768;

/// The rows of a sequence's window unless `--window` says otherwise: the
/// newest token and three drafts. A row of this model routes to experts of
/// its own, so a draft that is rejected has cost what a token costs; on
/// German and English prose and code the block's fourth link was accepted 7
/// times of 18 where the first three were accepted two times of three
/// (2026-11-15, both cards on every layer).
const DRAFT_WINDOW: u32 = 4;

/// The tokens of K/V a first plan counts with to learn what a page costs on
/// each device, before the context is decided.
const TRIAL_CONTEXT: u32 = 8192;

/// Loads the model of `options`, whose file `gguf` is open, at `site`, and
/// starts the runtime over it; `begun` is when the load began.
pub(super) fn load(
    gguf: Gguf,
    options: &LoadOptions,
    site: &Site,
    begun: Instant,
) -> Result<Loaded, LoadError> {
    let config = Config::from_gguf(&gguf).map_err(model_error)?;
    tracing::info!(
        "model: {} layers of width {} in {} streams, {} experts per layer of which {} are used, \
         {} tokens, trained for a context of {}",
        config.n_layer,
        config.n_embd,
        config.n_stream,
        config.experts.count,
        config.experts.used,
        config.n_vocab,
        config.n_ctx,
    );
    let asked = Asked::read(&gguf, &config, options)?;
    let parallel = asked.parallel;

    // The devices, and how the model is spread over them.
    let OnDevices { held, spread, why_split, tower_at } = hold_and_spread(options, &asked)?;
    let devices: Vec<Device> = held.iter().map(|h| h.device.clone()).collect();
    // How two devices share every layer is measured at load, and the state of
    // a sequence is laid out by it: the prompt cache on SSD keeps state that
    // another load reads only with the same split. With `--cache` the split
    // measured once for this model on these devices is kept next to the
    // stores and taken again; every placement of one load takes one split.
    let saved = options.cache.as_ref().and_then(|dir| {
        let saved = SavedSpread::read(dir, &options.model, &devices)?;
        tracing::info!(
            "plan: the split of the two devices is the one measured before for the prompt \
             cache's state on SSD ({})",
            saved.file.display()
        );
        Some(saved)
    });
    let spread = match (spread, &saved) {
        (Spread::Auto | Spread::Tensor { .. }, Some(saved)) => saved.spread(),
        (spread, _) => spread,
    };
    // The rows of a step in the memory the parts keep: the devices decide, by
    // the memory a step's scratch may take, unless --ubatch-size says.
    let own_rows = match options.ubatch {
        Some(rows) => rows,
        None => step_rows(
            &Gguf::open(&options.model)?,
            &devices,
            &Options {
                kv: asked.kv,
                logit_rows: asked.logit_rows(),
                window_rows: asked.window_rows(),
                max_slots: parallel,
                ..Options::default()
            },
        )
        .map_err(model_error)?,
    };
    let planning = Planning {
        options,
        site,
        asked: &asked,
        held: &held,
        devices: &devices,
        own_rows,
        model_context: config.n_ctx as u32,
    };

    // Context: tokens of state for all sequences together, then the rows of
    // the steps, then the placement for both.
    let context = planning.context(&config, spread)?;
    let (pages, near) = (context.pages, context.near);
    let longest = context.tokens.min(config.n_ctx as u32);
    let (spread, max_rows, pass_rows) = planning.steps(longest, pages, near, context.spread)?;
    let (placement, page_bytes) =
        planning.place(&planning.session(longest, pages, near, spread, max_rows))?;
    let opts =
        ModelOptions { pass_rows, ..planning.session(longest, pages, near, spread, max_rows) };
    if let (Some(dir), None, Some(tensor)) = (&options.cache, &saved, &placement.tensor) {
        SavedSpread::write(dir, &options.model, &devices, &tensor.choice);
    }
    log_split(&held, &placement, why_split);
    log_context(options, context.tokens, &context.why, pages, longest, parallel);
    log_drafting(&asked);
    for ((plan, h), per_page) in placement.devices.iter().zip(&held).zip(&page_bytes) {
        tracing::info!(
            "estimate, device {}: dense weights {}, recurrent state {}, K and V {} per 1000 \
             tokens, step buffers {}; room for expert slots {}, {:.1} % of the {} of experts it \
             owns",
            h.device.ordinal(),
            Bytes(plan.weights),
            Bytes(plan.state),
            Bytes(per_page * 1000 / u64::from(CELLS_PER_PAGE)),
            Bytes(plan.step),
            Bytes(plan.slots),
            100.0 * plan.slot_share(),
            Bytes(plan.experts),
        );
    }

    // The model, built for the plan.
    let name = options.name.clone().unwrap_or_else(|| crate::preset::id_of_file(&options.model));
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(model_error)?;
    let (chat, template) = chat_options(options, &tokenizer)?;
    let model = Qwen4Exp::new(gguf, tokenizer, name.clone(), &devices, opts, longest)
        .map_err(model_error)?
        .with_chat_options(chat);
    let (window, drafts) = (asked.window, asked.drafts());
    let model = Arc::new(asked.attach(model, options, &held, tower_at)?);

    // Host RAM: the step buffers of the devices and the parts' first; the
    // tier of experts takes the rest, as far as the experts that VRAM has no
    // slot for can use it, and the prompt cache in RAM what the tier leaves:
    // the tier ranks first. One checkpoint place the prompt cache keeps
    // whatever the tier holds, booked before the tier: without a checkpoint a
    // hit replays every row it found, and a conversation resumes from its
    // last checkpoint at least.
    let tower_ram = model.projector().map_or(0, |p| p.pinned_bytes());
    let budget = Budget::of(&*model, options, site, parallel, (0, tower_ram), Rest::Experts)?;
    if near.is_some() && budget.required + budget.floor + context.far_most > budget.lockable() {
        tracing::warn!(
            "--ram {}: the K and V of the whole context beyond VRAM take up to {}, more than the \
             limit leaves beside what the engine needs to run; a sequence that grows into the \
             rest fails its step",
            Bytes(budget.ram),
            Bytes(context.far_most)
        );
    }
    let engine = Engine {
        pages,
        parallel,
        window: if drafts { window } else { 1 },
        max_rows: 0,
        context: context.tokens,
    };
    let runtime = start::start(&model, devices, options, site, &budget, engine, template, begun)?;
    log_steps_taken(&model, own_rows, max_rows, pass_rows);
    log_started(&runtime, budget.ssd.as_ref(), &model.drafters());
    fill_tiers(&runtime, &model, &name)?;
    log_expert_costs(&model);
    log_taken(&model, &config, &held)?;
    match near {
        Some(near) => tracing::info!(
            "context: {} tokens of sequence state; K and V of {} tokens in VRAM, of the rest in \
             page-locked RAM as the sequences grow into it (booked in --ram above the tier of \
             experts, which gives the memory up), the indexer's keys all in VRAM",
            context.tokens,
            plan::pool_tokens(near, parallel)
        ),
        None => tracing::info!(
            "context: {} tokens of sequence state, all of it in VRAM",
            context.tokens
        ),
    }
    Ok(Loaded::new(runtime, name, held, budget.ram))
}

/// What the operator asked for, read and checked before a device is touched.
struct Asked {
    kv: (CacheFormat, CacheFormat),
    parallel: u32,
    /// The most rows of a sequence's window.
    window: u32,
    /// The MTP block and the file it is read from: the model file
    /// (`--drafters mtp`) or a file of its own.
    mtp: Option<(PathBuf, Gguf)>,
    dflash: Option<Arc<ares_model::dflash::File>>,
    /// What the drafters take on the device with the head ([`drafter_bytes`]).
    drafter_bytes: u64,
    vocabulary: Option<Arc<[u32]>>,
    prune: Option<PruneSetup>,
    projector: Option<Projector>,
}

impl Asked {
    /// Reads and checks what `options` ask of the model of `gguf`.
    fn read(gguf: &Gguf, config: &Config, options: &LoadOptions) -> Result<Self, LoadError> {
        let kv = options.kv.unwrap_or((CacheFormat::Q8, CacheFormat::Q8));
        if options.shares.is_some() {
            return Err(refuse(
                "--shares does nothing for this model: two devices that share every layer split \
                 its recurrent heads by what each computes faster, measured at load, and a range \
                 of layers per device follows what fits; --split names which of the two",
            ));
        }
        if options.attention != plan::AttentionShares::ByContext {
            return Err(refuse(
                "shares of the attention heads do nothing for this model: its attention layers, \
                 their keys and values are on the device that mixes the streams",
            ));
        }
        // The model's MTP block: in the model file (`--drafters mtp`, a file
        // that carries the `mtp.*` tensors) or a file of its own, which
        // --drafters names by its path (architecture `qwen4exp-mtp`). A DFlash
        // drafter (a file of architecture `dflash`) may come with it. Without a
        // model drafter nothing speculates (the n-gram drafter takes part only
        // next to one), no window is reserved, and the options of drafting mean
        // nothing.
        let files = drafter_files(options, MTP_ARCHITECTURE)?;
        let dflash = files.dflash;
        let in_model = options.drafters.iter().flatten().any(|d| *d == Drafter::Mtp);
        let mtp = match files.mtp {
            Some(found) => Some(found),
            None if in_model => {
                let path = gguf.path(0).to_owned();
                if !MtpWeights::in_model_file(gguf) {
                    return Err(refuse(
                        "--drafters mtp: this model file carries no MTP block; a file of the \
                         block is named by its path",
                    ));
                }
                let file = Gguf::open(&path).map_err(|e| refuse(format!("--drafters mtp: {e}")))?;
                Some((path, file))
            }
            None => None,
        };
        // The block is read before a device is touched: one that is no MTP
        // block of this model is refused by name.
        if let Some((path, file)) = &mtp {
            MtpWeights::load(file, config)
                .map_err(|e| refuse(format!("--drafters {}: {e}", path.display())))?;
        }
        let drafter_bytes = drafter_bytes(mtp.as_ref().map(|(_, file)| file), dflash.as_deref());
        let window = options.window.unwrap_or(DRAFT_WINDOW);
        let drafts = mtp.is_some() || dflash.is_some();
        if options.window.is_some() && !drafts {
            return Err(refuse("--window without a model drafter does nothing"));
        }
        if drafts && !(2..=MAX_ROWS).contains(&window) {
            return Err(refuse(format!(
                "--window {window}: a window is the newest token and its drafts, 2 to {MAX_ROWS} \
                 rows (a step that verifies drafts has {MAX_ROWS} rows for all sequences together)"
            )));
        }
        // A draft vocabulary is the head view of every drafter of the model.
        let vocabulary = match &options.draft_vocabulary {
            Some(_) if !drafts => {
                return Err(refuse("--draft-vocab without a model drafter does nothing"));
            }
            Some(path) => Some(read_vocabulary(path)?),
            None => None,
        };
        let parallel = options.parallel.unwrap_or(DEFAULT_PARALLEL);
        if parallel > MAX_ROWS {
            return Err(refuse(format!(
                "--parallel {parallel}: a step has {MAX_ROWS} rows at most, one for each sequence"
            )));
        }
        if options.context == Some(0) {
            return Err(refuse("--context must be at least 1"));
        }
        let prune = expert_prune(options, gguf, config)?;
        let projector = open_projector(options, config.n_embd)?;
        Ok(Self { kv, parallel, window, mtp, dflash, drafter_bytes, vocabulary, prune, projector })
    }

    /// Whether a model drafter takes part.
    fn drafts(&self) -> bool {
        self.mtp.is_some() || self.dflash.is_some()
    }

    /// The rows of a step that verifies drafts, for all sequences together.
    fn window_rows(&self) -> u32 {
        if self.drafts() { MAX_ROWS } else { 0 }
    }

    /// The rows whose logits a step can want.
    fn logit_rows(&self) -> u32 {
        self.window_rows().max(self.parallel)
    }

    /// `model` with what was asked of it besides its options: the usage
    /// profile of its experts, the draft vocabulary, the drafters and the
    /// projector, whose tower runs on device `tower_at` of `held`.
    fn attach(
        self,
        mut model: Qwen4Exp,
        options: &LoadOptions,
        held: &[Held],
        tower_at: Option<usize>,
    ) -> Result<Qwen4Exp, LoadError> {
        if let Some(dir) = profile_dir(options) {
            model = model.with_usage_profile(&dir, Arc::new(|line| tracing::warn!("{line}")));
        }
        // The tier of experts in RAM giving memory back to the machine.
        model = model.with_log(Arc::new(|line| tracing::info!("{line}")));
        if let Some(tokens) = &self.vocabulary {
            model = model
                .with_draft_vocabulary(tokens)
                .map_err(|e| LoadError::Model(format!("--draft-vocab: {e}")))?;
        }
        if let Some((path, file)) = self.mtp {
            model = model
                .with_mtp(file)
                .map_err(|e| LoadError::Model(format!("--drafters {}: {e}", path.display())))?;
        }
        if let Some(file) = self.dflash {
            let path = file.gguf.path(0).to_owned();
            model = model
                .with_dflash(file)
                .map_err(|e| LoadError::Model(format!("--drafters {}: {e}", path.display())))?;
        }
        if let (Some(projector), Some(t)) = (self.projector, tower_at) {
            tracing::info!(
                "plan: the image tower on device {}, {} of its VRAM held back for it",
                held[t].device.ordinal(),
                Bytes(projector.vram_bytes())
            );
            model = model
                .with_projector(Arc::new(projector), t)
                .map_err(|e| LoadError::Model(format!("--mmproj: {e}")))?;
        }
        Ok(model)
    }
}

/// The devices of a load and how the model is spread over them.
struct OnDevices {
    held: Vec<Held>,
    spread: Spread,
    /// Why the spread is what it is, for the log.
    why_split: &'static str,
    /// The device the image tower runs on.
    tower_at: Option<usize>,
}

/// Opens the devices and decides how the model is spread over them, and the
/// device the image tower runs on, whose VRAM is held back before anything
/// is planned (the expert slots take less).
fn hold_and_spread(options: &LoadOptions, asked: &Asked) -> Result<OnDevices, LoadError> {
    let (_, mut held) = hold_devices(options)?;
    match options.split {
        Some(SplitKind::Single) if options.devices.as_ref().is_some_and(|d| d.len() > 1) => {
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
        Some(SplitKind::Tensor) if held.len() != 2 => {
            return Err(refuse(format!(
                "--split tensor puts two devices on every layer, and {} are named",
                held.len()
            )));
        }
        _ => {}
    }
    if options.split == Some(SplitKind::Single) && held.len() > 1 {
        // The device with the most room: that is what holds experts.
        let most = (0..held.len()).max_by_key(|i| held[*i].free).unwrap_or(0);
        let keep = held.swap_remove(most);
        held = vec![keep];
    }
    let (spread, why_split) = match options.split {
        Some(SplitKind::Tensor) => {
            (Spread::Tensor { mixer: None, core_share: None, shared_on_mixer: None }, "as asked")
        }
        Some(SplitKind::Layer) => (Spread::Layers, "as asked"),
        Some(SplitKind::Single) => (Spread::Layers, "as asked"),
        None if held.len() == 1 => (Spread::Layers, "one device to place it on"),
        None => (
            Spread::Auto,
            "both on every layer when each can write the other's memory, otherwise a range of \
             layers each",
        ),
    };
    let tower_at = asked.projector.as_ref().map(|p| tower_of(&held, p.home()));
    if let (Some(t), Some(projector)) = (tower_at, &asked.projector) {
        held[t].free = held[t].free.saturating_sub(projector.vram_bytes());
    }
    describe_links(&held)?;
    Ok(OnDevices { held, spread, why_split, tower_at })
}

/// The context of a load: its tokens, its pages, and where they lie.
struct Context {
    /// Tokens of state for all sequences together.
    tokens: u32,
    pages: u32,
    /// The pages whose K and V lie in VRAM when the rest overflows into RAM.
    near: Option<u32>,
    /// What the K and V of the pages beyond VRAM take at their most.
    far_most: u64,
    /// Why, for the log.
    why: String,
    /// The spread with the split a trial placement measured pinned.
    spread: Spread,
}

/// What a load plans with: what was asked, the devices, and the rows of a
/// step in the parts' own memory.
struct Planning<'a> {
    options: &'a LoadOptions,
    site: &'a Site,
    asked: &'a Asked,
    held: &'a [Held],
    devices: &'a [Device],
    own_rows: u32,
    model_context: u32,
}

impl Planning<'_> {
    /// The model's options for `context` tokens in `pages` pages (`near` of
    /// them in VRAM), spread as `spread`, with steps of up to `max_rows` rows.
    fn session(
        &self,
        context: u32,
        pages: u32,
        near: Option<u32>,
        spread: Spread,
        max_rows: u32,
    ) -> ModelOptions {
        let asked = self.asked;
        ModelOptions {
            part: Options {
                kv: asked.kv,
                max_rows,
                own_rows: self.own_rows,
                logit_rows: asked.logit_rows(),
                window_rows: asked.window_rows(),
                max_slots: asked.parallel,
                cells: context.min(self.model_context),
                pool_pages: pages,
                near_pages: near,
                prune: asked.prune.clone(),
                ..Options::default()
            },
            headroom: VRAM_HEADROOM,
            spread,
            roles_dir: profile_dir(self.options),
            ..ModelOptions::default()
        }
    }

    /// A placement's error, named by the split that was asked; one that does
    /// not fit is, next to other models, no room.
    fn named_split(&self, e: impl ToString) -> LoadError {
        let e = e.to_string();
        let fits = !e.contains("does not fit");
        let error = LoadError::Model(match self.options.split {
            Some(kind) => format!("--split {}: {e}", kind.name()),
            None => e,
        });
        if fits { error } else { short(self.site, error) }
    }

    /// Where the model opened with `opts` goes, and what a page takes on each
    /// device ([`place`]).
    fn place(&self, opts: &ModelOptions) -> Result<(Placement, Vec<u64>), LoadError> {
        let gguf = Gguf::open(&self.options.model)?;
        place(&gguf, self.devices, opts).map_err(|e| self.named_split(e))
    }

    /// The context: the operator's (`--context`, for all sequences together;
    /// a request holds at most the model's own), or without a number from the
    /// operator the model's own (the owner's rule of 2026-11-22). Every page of
    /// K and V in VRAM is expert slots fewer, and a step reads only the cells
    /// the indexer picks, so the indexer's keys of every page stay in VRAM,
    /// the K and V of the first pages up to a share of the room for slots, and
    /// those of the pages beyond in RAM, as far as the sequences grow into them
    /// (`ares_model::blocks::far`); a named context is placed the same way.
    fn context(&self, config: &Config, spread: Spread) -> Result<Context, LoadError> {
        let parallel = self.asked.parallel;
        let model_context = self.model_context;
        let trial = TRIAL_CONTEXT.min(model_context);
        let gguf = Gguf::open(&self.options.model)?;
        let trial_options =
            self.session(trial, plan::pool_pages(trial, parallel), None, spread, self.own_rows);
        let (placement, page_bytes) =
            place(&gguf, self.devices, &trial_options).map_err(|e| self.named_split(e))?;
        log_trial(&placement, self.held);
        let spread = pinned(spread, &placement);
        // The pools of the trial are slots again.
        let pages_of_trial = u64::from(plan::pool_pages(trial, parallel));
        let slots: Vec<u64> = placement
            .devices
            .iter()
            .zip(&page_bytes)
            .map(|(d, page)| d.slots + page * pages_of_trial)
            .collect();
        let floors = slot_floors(config, &placement, self.asked.drafter_bytes);
        let parts =
            page_parts(&gguf, self.devices, &trial_options, &placement).map_err(model_error)?;
        let asked = self.options.context;
        let all = plan::pool_pages(asked.unwrap_or(model_context), parallel);
        let Some(near) = plan::near_pages(&slots, &floors, &parts, all) else {
            let error = refuse(match asked {
                Some(context) => format!(
                    "--context {context}: the devices have no room for the indexer's keys of so \
                     many tokens next to the experts' slots; name a smaller context"
                ),
                None => "the devices have no room for the indexer's keys of the model's context \
                         next to the experts' slots; name a context with --context"
                    .to_owned(),
            });
            return Err(short(self.site, error));
        };
        let plan = plan::ContextPlan::overflowing(asked, model_context, all, near, parallel);
        let far_most = parts.iter().map(|(kv, _)| kv).sum::<u64>() * u64::from(all - near);
        tracing::info!(
            "plan: the context takes {} of the bytes for expert slots (the indexer's keys of \
             every page and K and V of {} pages; at least {} stay for the fewest slots each layer \
             needs and the drafters){}",
            join(
                parts.iter().zip(&slots).map(|((kv, keys), room)| format!(
                    "{} of {}",
                    Bytes(keys * u64::from(all) + kv * u64::from(near)),
                    Bytes(*room)
                )),
                " and "
            ),
            near,
            join(floors.iter().map(|b| Bytes(*b).to_string()), " and "),
            if plan.overflow {
                format!(
                    "; K and V of the other {} pages in RAM as sequences grow into them, {} at \
                     most",
                    all - near,
                    Bytes(far_most)
                )
            } else {
                String::new()
            },
        );
        Ok(Context {
            tokens: plan.context(),
            pages: all,
            near: plan.overflow.then_some(near),
            far_most,
            why: plan.why(),
            spread,
        })
    }

    /// The rows of the steps for `longest` tokens in `pages` pages: a prompt
    /// read alone runs layer by layer where the model can
    /// (`device/runtime/pass.rs`): its steps are passes of up to as many rows
    /// as asked or the default, as far as the expert slots lend for the rows
    /// they keep, and the parts' steps never take more than their own
    /// scratch. Elsewhere it takes steps as many rows wide as asked or the
    /// default, as far as the expert slots can lend the scratch for them.
    /// Returns the spread with the split pinned, the most rows of a step and
    /// the most rows of a pass (0 for none).
    fn steps(
        &self,
        longest: u32,
        pages: u32,
        near: Option<u32>,
        spread: Spread,
    ) -> Result<(Spread, u32, u32), LoadError> {
        let (own_rows, batch) = (self.own_rows, self.options.batch);
        let (placement, page_bytes) =
            self.place(&self.session(longest, pages, near, spread, own_rows))?;
        log_trial(&placement, self.held);
        let spread = pinned(spread, &placement);
        let own = self.session(longest, pages, near, spread, own_rows);
        let passes = reads_layer_by_layer(&own.part, self.asked.dflash.is_some());
        let (max_rows, pass_rows) = if passes {
            (own_rows, batch.unwrap_or(DEFAULT_PASS).max(own_rows))
        } else {
            let asked = batch.unwrap_or(DEFAULT_BATCH).max(own_rows);
            let max_rows = batch_rows(
                &Gguf::open(&self.options.model)?,
                self.devices,
                &own,
                &placement,
                pages,
                &page_bytes,
                asked,
                self.asked.drafter_bytes,
            )
            .map_err(model_error)?;
            if batch.is_some_and(|b| max_rows < b) {
                return Err(refuse(format!(
                    "--batch-size {asked}: the expert slots can lend the scratch of a step of \
                     {max_rows} rows at most, beside steps of {own_rows} rows in the parts' own \
                     memory"
                )));
            }
            (max_rows, 0)
        };
        tracing::info!(
            "steps of up to {own_rows} rows in the parts' own memory{}",
            if pass_rows > own_rows {
                format!(
                    ", and a prompt read alone layer by layer in steps of up to {pass_rows} rows, \
                     its rows kept on VRAM the expert slots lend for the step"
                )
            } else if max_rows > own_rows {
                format!(
                    ", and of up to {max_rows} rows for a prompt read alone, on VRAM the expert \
                     slots lend for the step"
                )
            } else {
                String::new()
            }
        );
        Ok((spread, max_rows, pass_rows))
    }
}

/// The plan's line of the split: where the layers go.
fn log_split(held: &[Held], placement: &Placement, why_split: &str) {
    tracing::info!(
        "plan: {} ({why_split})",
        match &placement.tensor {
            Some(tensor) => format!(
                "both devices on every layer: device {} holds the streams and runs the mixers, \
                 the other {}",
                held[tensor.mixer].device.ordinal(),
                match (tensor.choice.core_share >= 1.0, tensor.choice.shared_on_mixer) {
                    (true, true) => "holds and computes the experts it owns, nothing else",
                    (true, false) => "computes the shared experts and the experts it owns",
                    (false, true) => {
                        "computes its share of every layer's recurrent heads and the experts it \
                         owns"
                    }
                    (false, false) => {
                        "computes its share of every layer's recurrent heads, the shared experts \
                         and the experts it owns"
                    }
                }
            ),
            None if held.len() == 1 => format!("all layers on device {}", ordinals(held)),
            None => join(
                placement.devices.iter().zip(held).map(|(plan, h)| {
                    format!(
                        "layers {}..{} on device {}",
                        plan.layers.start,
                        plan.layers.end,
                        h.device.ordinal()
                    )
                }),
                ", ",
            ),
        }
    );
}

/// The plan's line of the formats of K and V and of the drafters.
fn log_drafting(asked: &Asked) {
    let window = asked.window;
    tracing::info!(
        "plan: K in {}, V in {}; {}",
        plan::format_name(asked.kv.0),
        plan::format_name(asked.kv.1),
        match (&asked.mtp, &asked.dflash) {
            (None, None) => "no speculation: the model drafts with an MTP block from a file \
                             of its own (--drafters <file>), and earlier text alone is not a \
                             drafter"
                .to_owned(),
            (block, dflash) => format!(
                "{}{}{} draft{}, on the device that holds the output head: windows of up to \
                 {window} rows a sequence, {MAX_ROWS} rows a step, as long as they pay; the \
                 n-gram drafter joins",
                block
                    .as_ref()
                    .map(|(path, _)| format!("the model's MTP block ({})", path.display()))
                    .unwrap_or_default(),
                if block.is_some() && dflash.is_some() { " and " } else { "" },
                dflash
                    .as_ref()
                    .map(|f| format!("the DFlash drafter ({})", f.gguf.path(0).display()))
                    .unwrap_or_default(),
                if block.is_some() && dflash.is_some() { "" } else { "s" },
            ),
        },
    );
}

/// Says what a prompt read alone takes once the parts are open: the rows the
/// expert slots lend for, which the model knows then, against the rows
/// planned (`own_rows` in the parts' own memory, `max_rows` on lent VRAM,
/// `pass_rows` of a pass, 0 for none).
fn log_steps_taken(model: &Qwen4Exp, own_rows: u32, max_rows: u32, pass_rows: u32) {
    if model.max_rows() > own_rows && pass_rows > own_rows {
        tracing::info!(
            "a prompt read alone runs layer by layer in steps of up to {} rows, kept on VRAM \
             the expert slots lend",
            model.max_rows()
        );
    } else if pass_rows > own_rows {
        tracing::warn!(
            "the expert slots cannot lend for a prompt read layer by layer: a prompt read \
             alone takes steps of {own_rows} rows"
        );
    } else if model.max_rows() > own_rows {
        tracing::info!(
            "a prompt read alone takes steps of up to {} rows on VRAM the expert slots lend",
            model.max_rows()
        );
    } else if max_rows > own_rows {
        tracing::warn!(
            "the expert slots cannot lend the scratch of a step wider than {own_rows} rows: a \
             prompt read alone takes steps of {own_rows} rows"
        );
    }
}

/// What each device took, by the parts' reports and the device's own, the
/// slots of experts and the MTP block; starts the threads that log, while
/// the model is served, what its reads and its MTP block do.
fn log_taken(model: &Arc<Qwen4Exp>, config: &Config, held: &[Held]) -> Result<(), LoadError> {
    for (h, report) in held.iter().zip(model.reports()) {
        let free = h.device.mem_info()?.free;
        tracing::info!(
            "device {} ({}): {:?} of layers {:?}, dense weights {} in {:.1} s, {} slots for \
             experts in {}, recurrent state {} per sequence, K and V {} per 1000 tokens, step \
             buffers {}; taken {}, free {}",
            h.device.ordinal(),
            report.device,
            report.role,
            report.layers,
            Bytes(report.weights),
            report.upload.as_secs_f64(),
            report.n_slots,
            Bytes(report.slots),
            Bytes(report.state_per_slot),
            Bytes(report.kv_per_1000),
            Bytes(report.scratch),
            Bytes(h.free.saturating_sub(free)),
            Bytes(free),
        );
        tracing::info!(
            "device {}: {}",
            h.device.ordinal(),
            products_line(report.full_rate, &report.paths, &[])
        );
    }
    if let Some((vram, ram_slots, ram_bytes)) = model.expert_slots() {
        let all = config.n_layer * config.experts.count;
        tracing::info!(
            "experts: {all} in the file; {vram} slots in VRAM ({:.1} %), {ram_slots} in the RAM \
             tier of {} now ({:.1} %); the slots fill from what the routers pick",
            100.0 * vram as f64 / all as f64,
            Bytes(ram_bytes),
            100.0 * ram_slots as f64 / all as f64,
        );
    }
    watch_reads(Arc::downgrade(model));
    if let Some(report) = model.mtp_report() {
        watch_drafts(Arc::downgrade(model));
        let (vram, ram_slots) = model.mtp_slots().unwrap_or((0, 0));
        tracing::info!(
            "MTP block, on device {}: dense weights {}, K, V and keys {}, buffers {}; its {} \
             experts share the tiers: {vram} slots in VRAM, {ram_slots} in RAM",
            held[report.index].device.ordinal(),
            Bytes(report.weights),
            Bytes(report.pools),
            Bytes(report.buffers),
            config.experts.count,
        );
    }
    Ok(())
}

/// What `--expert-prune` asks of the routers, with the table of similar
/// experts for the rule `similar`, which is kept beside the usage profile.
fn expert_prune(
    options: &LoadOptions,
    gguf: &Gguf,
    config: &Config,
) -> Result<Option<PruneSetup>, LoadError> {
    let Some(asked) = options.expert_prune else { return Ok(None) };
    if asked.from_rank > config.experts.used as u32 {
        return Err(refuse(format!(
            "--expert-prune: rank {} of the {} picks of a row",
            asked.from_rank, config.experts.used
        )));
    }
    let prune = ares_cuda::Prune::new(asked.rule, asked.from_rank - 1);
    let near = if asked.rule == ares_cuda::PruneRule::Similar {
        let dir = profile_dir(options).ok_or_else(|| {
            refuse("--expert-prune similar: no directory for the table of similar experts")
        })?;
        let key = profile_key(gguf, config);
        let path = ExpertNeighbours::path(&key, &dir);
        let table = ExpertNeighbours::read(&dir, &key)
            .map_err(|e| refuse(format!("--expert-prune similar: {}: {e}", path.display())))?
            .ok_or_else(|| {
                refuse(format!(
                    "--expert-prune similar: there is no table of similar experts for this \
                     model file ({})",
                    path.display()
                ))
            })?;
        Some(Arc::new(table))
    } else {
        None
    };
    tracing::info!(
        "expert pruning: {:?} from the {}. pick of a row on, for experts only the SSD holds; \
         this changes the output",
        asked.rule,
        asked.from_rank
    );
    Ok(Some(PruneSetup { prune, near }))
}

/// Says what the trial of the second device's role measured, when `placement`
/// made one (both devices on every layer, the role not remembered).
fn log_trial(placement: &Placement, held: &[Held]) {
    let Some((tensor, trial)) =
        placement.tensor.as_ref().and_then(|t| Some((t, t.trial.as_ref()?)))
    else {
        return;
    };
    let worker = held[1 - tensor.mixer].device.ordinal();
    let role = |choice: &TensorChoice| {
        if choice.core_share >= 1.0 && choice.shared_on_mixer {
            "experts only".to_owned()
        } else {
            format!(
                "{:.0} % of the recurrent heads{}",
                100.0 * (1.0 - choice.core_share),
                if choice.shared_on_mixer { "" } else { " and the shared experts" }
            )
        }
    };
    let (when, kept_in) = match &trial.memory {
        RoleMemory::Remembered(path) => {
            (format!("remembered from {} (timed in", path.display()), String::new())
        }
        RoleMemory::Written(path) => {
            ("timed at load (in".to_owned(), format!("; remembered in {}", path.display()))
        }
        RoleMemory::NotWritten(why) => {
            ("timed at load (in".to_owned(), format!("; not remembered: {why}"))
        }
        RoleMemory::None => ("timed at load (in".to_owned(), String::new()),
    };
    tracing::info!(
        "plan: the role of device {worker}, {when} {:.1} s: {}): {}; kept: {}{kept_in}",
        trial.took.as_secs_f64(),
        if trial.wide_rows > 0 {
            format!("a decode step and a step of {} rows each", trial.wide_rows)
        } else {
            "a decode step each".to_owned()
        },
        join(
            trial
                .times
                .iter()
                .map(|t| format!(
                    "{} {:.2} ms{}",
                    role(&t.choice),
                    t.decode.as_secs_f64() * 1e3,
                    t.wide
                        .map_or(String::new(), |w| format!(" and {:.0} ms", w.as_secs_f64() * 1e3))
                ))
                .chain(trial.refused.iter().map(|(c, why)| format!("{} refused: {why}", role(c)))),
            ", "
        ),
        role(&trial.kept),
    );
}

/// The spread that places the model again as `placement` does: its split of
/// two devices that share every layer, measured or given.
fn pinned(spread: Spread, placement: &Placement) -> Spread {
    match &placement.tensor {
        Some(tensor) => Spread::Tensor {
            mixer: Some(tensor.choice.mixer),
            core_share: Some(tensor.choice.core_share),
            shared_on_mixer: Some(tensor.choice.shared_on_mixer),
        },
        None => spread,
    }
}

/// The split of two devices that share every layer as a load measured it,
/// kept in a file next to the prompt cache's stores, by model file and for
/// the devices it was measured on.
struct SavedSpread {
    choice: TensorChoice,
    file: PathBuf,
}

impl SavedSpread {
    fn file(dir: &Path, model: &Path) -> PathBuf {
        let stem = model.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy());
        dir.join(format!("{stem}.split"))
    }

    fn devices(devices: &[Device]) -> String {
        join(devices.iter().map(|d| d.info().name.clone()), ", ")
    }

    /// The split kept for `model` on `devices`, if there is one for them.
    fn read(dir: &Path, model: &Path, devices: &[Device]) -> Option<Self> {
        let file = Self::file(dir, model);
        let text = std::fs::read_to_string(&file).ok()?;
        let value = |key: &str| {
            text.lines().find_map(|line| line.strip_prefix(key)?.strip_prefix(' ').map(str::trim))
        };
        if devices.len() != 2 || value("devices")? != Self::devices(devices) {
            return None;
        }
        let choice = TensorChoice {
            mixer: value("mixer")?.parse().ok()?,
            core_share: value("core_share")?.parse().ok()?,
            shared_on_mixer: value("shared_on_mixer")?.parse().ok()?,
        };
        (choice.mixer < 2 && (0.0..=1.0).contains(&choice.core_share))
            .then_some(Self { choice, file })
    }

    /// Keeps `choice` for `model` on `devices`; a file that cannot be written
    /// costs the next load the measurement, which the log says.
    fn write(dir: &Path, model: &Path, devices: &[Device], choice: &TensorChoice) {
        let file = Self::file(dir, model);
        let text = format!(
            "devices {}\nmixer {}\ncore_share {}\nshared_on_mixer {}\n",
            Self::devices(devices),
            choice.mixer,
            choice.core_share,
            choice.shared_on_mixer
        );
        if let Err(error) = std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&file, text))
        {
            tracing::warn!(
                "the split of the two devices could not be kept in {} ({error}): the next load \
                 measures its own, and the prompt cache's state on SSD of this one is not read \
                 there unless the measurement is the same",
                file.display()
            );
        }
    }

    fn spread(&self) -> Spread {
        Spread::Tensor {
            mixer: Some(self.choice.mixer),
            core_share: Some(self.choice.core_share),
            shared_on_mixer: Some(self.choice.shared_on_mixer),
        }
    }
}
