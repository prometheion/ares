//! Loading a `qwen4exp` model (Qwen3.8-Flash-Next): a mixture of experts of
//! which only the dense parts fit in VRAM.
//!
//! The order is the one of `load`: read the file, look at the devices, decide,
//! log the decision with its reasons, load, log what it took. What differs from
//! a dense model is what there is to decide: the dense parts are small and sit
//! on the devices; the rest of the VRAM holds slots for experts, which follow
//! the router, with the RAM the engine may use as a tier in front of the file
//! and the CPU computing what no tier brings in time. So the plan is where the
//! layers go (a range per device, or both devices on every layer), how much
//! context costs slots, and how much of the RAM limit the tier takes.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_core::units::Bytes;
use ares_gguf::Gguf;
use ares_model::qwen4exp::device::{
    Options, ProfileFound, ProfileReport, PruneSetup, Qwen4Exp, SessionOptions, Spread,
    TensorChoice, batch_rows, page_parts, place, profile_key, reads_layer_by_layer, step_rows,
};
use ares_model::qwen4exp::{Config, MtpWeights};
use ares_runtime::expert_store::ExpertNeighbours;
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig, SsdConfig, required_ram};
use ares_tokenizer::Tokenizer;

use crate::load::{
    LoadError, Loaded, VRAM_HEADROOM, describe_links, drafter_files, hold_devices, join,
    log_started, ram_limit_of, read_vocabulary, refuse, short,
};
use crate::models::Site;
use crate::plan::{self, CELLS_PER_PAGE, DEFAULT_PARALLEL, Drafter, LoadOptions, SplitKind};

/// The most rows of a step that generates (the newest token of every sequence and the drafts
/// they verify): what the decode kernels of the experts take. Only the slices of a prompt are
/// wider.
const MAX_ROWS: u32 = ares_cuda::MAX_BATCH;

/// The rows of the step a prompt read alone takes unless `--batch-size` says
/// otherwise, as far as the expert slots can lend its scratch: every expert a
/// prompt picks is read once a step, so a long prompt reads the experts of
/// the file once per this many rows (chosen by measurement, see "Large prompt
/// steps on lent VRAM" in `docs/ARCHITECTURE.md`).
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

/// A text of German, English and code, about 700 tokens, that the model reads
/// once at the start (see [`warm_up`]).
const WARM_UP: &str = include_str!("warm_up.txt");

/// How long the warm-up may take: the experts of a text come from the drive,
/// and the drive may be busy with something else.
const WARM_UP_LIMIT: Duration = Duration::from_secs(1800);

/// The tokens of K/V a first plan counts with to learn what a page costs on
/// each device, before the context is decided.
const TRIAL_CONTEXT: u32 = 8192;

/// Free slots the expert store keeps per layer in VRAM (its default).
const VRAM_RESERVE: u64 = 8;

/// The bytes of expert slots each device of `placement` cannot do without:
/// a row's picks and the free slots of every layer it holds slots for (what
/// the store gives a layer when the CPU serves what VRAM cannot take), a
/// tenth more for the rounding of slots, and on the device with the head
/// `drafter` bytes for the drafters.
fn slot_floors(
    c: &Config,
    placement: &ares_model::qwen4exp::device::Placement,
    drafter: u64,
) -> Vec<u64> {
    let all: u64 = placement.devices.iter().map(|d| d.experts).sum();
    let per_slot = (c.experts.used as u64 + VRAM_RESERVE) * 11 / 10;
    placement
        .devices
        .iter()
        .map(|d| {
            // One expert of every layer the device holds slots for.
            let round = if placement.tensor.is_some() { all } else { d.experts };
            let round = round / c.experts.count.max(1) as u64;
            per_slot * round + if d.head { drafter } else { 0 }
        })
        .collect()
}

/// Loads the model of `options`, whose file `gguf` is open, and starts the
/// runtime over it.
pub(crate) fn load(
    gguf: Gguf,
    options: &LoadOptions,
    site: &Site,
    begun: Instant,
) -> Result<Loaded, LoadError> {
    let config = Config::from_gguf(&gguf).map_err(|e| LoadError::Model(e.to_string()))?;
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

    // What can be refused before a device is touched.
    let kv = options.kv.unwrap_or((ares_cuda::CacheFormat::Q8, ares_cuda::CacheFormat::Q8));
    if options.shares.is_some() {
        return Err(refuse(
            "--shares does nothing for this model: two devices that share every layer split its \
             recurrent heads by what each computes faster, measured at load, and a range of \
             layers per device follows what fits; --split names which of the two",
        ));
    }
    if options.attention != plan::AttentionShares::ByContext {
        return Err(refuse(
            "shares of the attention heads do nothing for this model: its attention layers, \
             their keys and values are on the device that mixes the streams",
        ));
    }
    if options.precision == ares_cuda::Precision::Exact {
        return Err(refuse(
            "--precision exact is not a choice for this model: its products have one kind of \
             arithmetic",
        ));
    }
    // The model's MTP block: in the model file (`--drafters mtp`, a file that
    // carries the `mtp.*` tensors) or a file of its own, which --drafters
    // names by its path (architecture `qwen4exp-mtp`). A DFlash drafter (a
    // file of architecture `dflash`) may come with it. Without a model
    // drafter nothing speculates (the n-gram drafter takes part only next to
    // one), no window is reserved, and the options of drafting mean nothing.
    let files = drafter_files(options, "qwen4exp-mtp")?;
    let dflash = files.dflash;
    let in_model = options.drafters.iter().flatten().any(|d| *d == Drafter::Mtp);
    let drafter = match files.mtp {
        Some(found) => Some(found),
        None if in_model => {
            let path = gguf.path(0).to_owned();
            if gguf.tensor("mtp.fc_embedding.weight").is_none() {
                return Err(refuse(
                    "--drafters mtp: this model file carries no MTP block; a file of the block \
                     is named by its path",
                ));
            }
            let file = Gguf::open(&path).map_err(|e| refuse(format!("--drafters mtp: {e}")))?;
            Some((path, file))
        }
        None => None,
    };
    // The block is read before a device is touched: one that is no MTP block
    // of this model is refused by name.
    if let Some((path, file)) = &drafter {
        MtpWeights::load(file, &config)
            .map_err(|e| refuse(format!("--drafters {}: {e}", path.display())))?;
    }
    // What the drafters take on the device with the head, for the plan of the
    // context: their weights and, for the MTP block, its experts, which stay
    // resident; a margin for their pools and buffers.
    let drafter_bytes = drafter.as_ref().map_or(0, |(path, file)| {
        let in_model = path.as_path() == gguf.path(0);
        let tensors = file.tensors().iter().filter(|t| !in_model || t.name.starts_with("mtp."));
        tensors.map(|t| t.size).sum::<u64>() + (128 << 20)
    }) + dflash.as_ref().map_or(0, |file| file.bytes() + (128 << 20));
    let window = options.window.unwrap_or(DRAFT_WINDOW);
    let drafts = drafter.is_some() || dflash.is_some();
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
    // Rows of a step that verifies drafts, for all sequences together, and
    // the rows whose logits a step can want.
    let window_rows = if drafts { MAX_ROWS } else { 0 };
    let logit_rows = window_rows.max(parallel);
    if options.context == Some(0) {
        return Err(refuse("--context must be at least 1"));
    }
    let prune = expert_prune(options, &gguf, &config)?;
    let projector = crate::load::open_projector(options, config.n_embd)?;

    // The devices.
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
    // The image tower runs on the device the rule chooses; its VRAM is held
    // back there before anything is planned (the expert slots take less).
    let tower_bytes = projector.as_ref().map_or(0, ares_model::qwen3vl::Projector::vram_bytes);
    let tower_at = projector.as_ref().map(|p| {
        crate::load::tower_device(
            &held.iter().map(|h| h.device.info()).collect::<Vec<_>>(),
            p.home(),
        )
    });
    if let Some(t) = tower_at {
        held[t].free = held[t].free.saturating_sub(tower_bytes);
    }
    describe_links(&held)?;
    let devices: Vec<_> = held.iter().map(|h| h.device.clone()).collect();
    // How two devices share every layer is measured at load, and the state of
    // a sequence is laid out by it: the prompt cache on SSD keeps state that
    // another load reads only with the same split. With `--cache` the split
    // measured once for this model on these devices is kept next to the
    // stores and taken again; every placement of one load takes one split.
    let remembered = options.cache.as_ref().and_then(|dir| {
        let remembered = Split::read(dir, &options.model, &devices)?;
        tracing::info!(
            "plan: the split of the two devices is the one measured before for the prompt \
             cache's state on SSD ({})",
            remembered.file.display()
        );
        Some(remembered)
    });
    let mut spread = match (spread, &remembered) {
        (Spread::Auto | Spread::Tensor { .. }, Some(split)) => split.spread(),
        (spread, _) => spread,
    };

    // The model's settings that do not depend on the context.
    let model_context = config.n_ctx as u32;
    // The rows of a step in the memory the parts keep: the devices decide, by
    // the memory a step's scratch may take, unless --ubatch-size says.
    let own_rows = match options.ubatch {
        Some(rows) => rows,
        None => step_rows(
            &Gguf::open(&options.model)?,
            &devices,
            &Options { kv, logit_rows, window_rows, max_slots: parallel, ..Options::default() },
        )
        .map_err(|e| LoadError::Model(e.to_string()))?,
    };
    let session = |context: u32, pages: u32, near: Option<u32>, spread: Spread, max_rows: u32| {
        SessionOptions {
            part: Options {
                kv,
                max_rows,
                own_rows,
                logit_rows,
                window_rows,
                max_slots: parallel,
                cells: context.min(model_context),
                pool_pages: pages,
                near_pages: near,
                prune: prune.clone(),
                ..Options::default()
            },
            headroom: VRAM_HEADROOM,
            spread,
            ..SessionOptions::default()
        }
    };
    let named_split = |e: String| {
        // A placement that does not fit is, next to other models, no room.
        let fits = !e.contains("does not fit");
        let error = LoadError::Model(match options.split {
            Some(kind) => format!("--split {}: {e}", kind.name()),
            None => e,
        });
        if fits { error } else { short(site, error) }
    };

    // Context: tokens of state for all sequences together. Without a number
    // from the operator, the model's own (the owner's rule of 2026-11-22).
    // Every page of K and V in VRAM is expert slots fewer, and a step reads
    // only the cells the indexer picks, so the indexer's keys of every page
    // stay in VRAM, the K and V of the first pages up to a share of the room
    // for slots, and those of the pages beyond in RAM, as far as the
    // sequences grow into them (`ares_model::blocks::far`). A context that
    // was asked for is all in VRAM.
    let (context, pages, near, far_most, why_context) = match options.context {
        Some(context) => {
            (context, plan::pool_pages(context, parallel), None, 0, "as asked".to_owned())
        }
        None => {
            let trial = TRIAL_CONTEXT.min(model_context);
            let gguf = Gguf::open(&options.model)?;
            let trial_options =
                session(trial, plan::pool_pages(trial, parallel), None, spread, own_rows);
            let (placement, page_bytes) =
                place(&gguf, &devices, &trial_options).map_err(|e| named_split(e.to_string()))?;
            log_trial(&placement, &held);
            spread = pinned(spread, &placement);
            // The pools of the trial are slots again.
            let pages_of_trial = u64::from(plan::pool_pages(trial, parallel));
            let slots: Vec<u64> = placement
                .devices
                .iter()
                .zip(&page_bytes)
                .map(|(d, page)| d.slots + page * pages_of_trial)
                .collect();
            let floors = slot_floors(&config, &placement, drafter_bytes);
            let parts = page_parts(&gguf, &devices, &trial_options, &placement)
                .map_err(|e| LoadError::Model(e.to_string()))?;
            let all = plan::pool_pages(model_context, parallel);
            let Some(near) = plan::near_pages(&slots, &floors, &parts, all) else {
                let error = refuse(
                    "the devices have no room for the indexer's keys of the model's context next \
                     to the experts' slots; name a context with --context",
                );
                return Err(short(site, error));
            };
            let plan = plan::ContextPlan::overflowing(model_context, all, near, parallel);
            tracing::info!(
                "plan: the context takes {} of the bytes for expert slots (the indexer's keys of \
                 every page and K and V of {} pages; at least {} stay for the fewest slots each \
                 layer needs and the drafters){}",
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
                        "; K and V of the other {} pages in RAM as sequences grow into them, {} \
                         at most",
                        all - near,
                        Bytes(parts.iter().map(|(kv, _)| kv).sum::<u64>() * u64::from(all - near))
                    )
                } else {
                    String::new()
                },
            );
            let far_most = parts.iter().map(|(kv, _)| kv).sum::<u64>() * u64::from(all - near);
            (plan.context(), all, plan.overflow.then_some(near), far_most, plan.why())
        }
    };
    let longest = context.min(model_context);
    let (placement, page_bytes) = place(
        &Gguf::open(&options.model)?,
        &devices,
        &session(longest, pages, near, spread, own_rows),
    )
    .map_err(|e| named_split(e.to_string()))?;
    log_trial(&placement, &held);
    let spread = pinned(spread, &placement);
    // A prompt read alone runs layer by layer where the model can (`device/runtime/pass.rs`):
    // its steps are passes of up to as many rows as asked or the default, as far as the
    // expert slots lend for the rows they keep, and the parts' steps never take more than
    // their own scratch. Elsewhere it takes steps as many rows wide as asked or the default,
    // as far as the expert slots can lend the scratch for them.
    let passes = reads_layer_by_layer(
        &session(longest, pages, near, spread, own_rows).part,
        dflash.is_some(),
    );
    let (max_rows, pass_rows) = if passes {
        (own_rows, options.batch.unwrap_or(DEFAULT_PASS).max(own_rows))
    } else {
        let asked = options.batch.unwrap_or(DEFAULT_BATCH).max(own_rows);
        let max_rows = batch_rows(
            &Gguf::open(&options.model)?,
            &devices,
            &session(longest, pages, near, spread, own_rows),
            &placement,
            pages,
            &page_bytes,
            asked,
            drafter_bytes,
        )
        .map_err(|e| LoadError::Model(e.to_string()))?;
        if options.batch.is_some_and(|b| max_rows < b) {
            return Err(refuse(format!(
                "--batch-size {asked}: the expert slots can lend the scratch of a step of \
                 {max_rows} rows at most, beside steps of {own_rows} rows in the parts' own memory"
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
    let (placement, page_bytes) = place(
        &Gguf::open(&options.model)?,
        &devices,
        &session(longest, pages, near, spread, max_rows),
    )
    .map_err(|e| named_split(e.to_string()))?;
    let opts = SessionOptions { pass_rows, ..session(longest, pages, near, spread, max_rows) };
    if let (Some(dir), None, Some(tensor)) = (&options.cache, &remembered, &placement.tensor) {
        Split::write(dir, &options.model, &devices, &tensor.choice);
    }

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
            None if held.len() == 1 => format!("all layers on device {}", ordinals_of(&held)),
            None => join(
                placement.devices.iter().zip(&held).map(|(plan, h)| {
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
    tracing::info!(
        "plan: context of {context} tokens for all sequences together ({why_context}), {pages} \
         pages; a request holds at most {longest} tokens; {parallel} sequence{} at once ({})",
        if parallel == 1 { "" } else { "s" },
        if options.parallel.is_some() { "as asked" } else { "the default" },
    );
    tracing::info!(
        "plan: K in {}, V in {}; {}",
        plan::format_name(kv.0),
        plan::format_name(kv.1),
        match (&drafter, &dflash) {
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

    // Load.
    let name = options.name.clone().unwrap_or_else(|| crate::preset::id_of_file(&options.model));
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| LoadError::Model(e.to_string()))?;
    let (chat, template) = crate::load::chat_options(options, &tokenizer)?;
    if template.is_none() {
        tracing::warn!(
            "the model file has no chat template: /v1/chat/completions will refuse requests, \
             /v1/completions works"
        );
    }
    let mut model = Qwen4Exp::new(gguf, tokenizer, name.clone(), &devices, opts, longest)
        .map_err(|e| LoadError::Model(e.to_string()))?
        .with_chat_options(chat);
    if let Some(dir) = profile_dir(options) {
        model =
            model.with_usage_profile(&dir, std::sync::Arc::new(|line| tracing::warn!("{line}")));
    }
    let drafting = drafts;
    if let Some(tokens) = &vocabulary {
        model = model
            .with_draft_vocabulary(tokens)
            .map_err(|e| LoadError::Model(format!("--draft-vocab: {e}")))?;
    }
    if let Some((path, file)) = drafter {
        model = model
            .with_mtp(file)
            .map_err(|e| LoadError::Model(format!("--drafters {}: {e}", path.display())))?;
    }
    if let Some(file) = dflash.clone() {
        let path = file.gguf.path(0).to_owned();
        model = model
            .with_dflash(file)
            .map_err(|e| LoadError::Model(format!("--drafters {}: {e}", path.display())))?;
    }
    if let (Some(projector), Some(t)) = (projector, tower_at) {
        tracing::info!(
            "plan: the image tower on device {}, {} of its VRAM held back for it",
            held[t].device.ordinal(),
            Bytes(tower_bytes)
        );
        model = model
            .with_projector(std::sync::Arc::new(projector), t)
            .map_err(|e| LoadError::Model(format!("--mmproj: {e}")))?;
    }
    let model = std::sync::Arc::new(model);

    // Host RAM: one limit for the engine. What it needs to run is taken from
    // it first (the step buffers of the devices and the parts'); the tier of
    // experts takes the rest, as far as the experts that VRAM has no slot for
    // can use it, and the prompt cache in RAM (checkpoints of the recurrent
    // state and pages that left the devices) what the tier leaves: the tier
    // ranks first. One checkpoint place the prompt cache keeps whatever the
    // tier holds, booked before the tier: without a checkpoint a hit replays
    // every row it found, and a conversation resumes from its last checkpoint
    // at least. The prompt cache on SSD adds two needs of its own, as for a
    // dense model: a few pages that state passes through on its way to the
    // drive and back, next to that place, and buffers of ordinary memory for
    // checkpoints in transit.
    let required = required_ram(&*model, parallel);
    let tower_ram = model.projector().map_or(0, |p| p.pinned_bytes());
    let checkpoint = model.prefix_cache().map_or(0, |cache| cache.bytes());
    let ssd = options.cache.as_ref().map(|dir| {
        SsdConfig::new(dir, options.cache_size.unwrap_or(crate::plan::DEFAULT_CACHE_SIZE))
    });
    let durable = ssd.as_ref().and_then(|_| model.durable(CELLS_PER_PAGE));
    let transit = durable.as_ref().map_or(0, SsdConfig::transit_ram);
    let floor = checkpoint + durable.as_ref().map_or(0, SsdConfig::floor_pages);
    let mut needs = vec![("its step buffers", required - tower_ram)];
    if tower_ram > 0 {
        needs.push((
            "the image tower's weights, which lie in page-locked RAM (--mmproj-weights ram)",
            tower_ram,
        ));
    }
    match (floor, &durable) {
        (0, _) => {}
        (_, Some(_)) => needs.push((
            "one checkpoint of recurrent state and the pages on their way to the prompt cache \
             on SSD or back (page-locked)",
            floor,
        )),
        (_, None) => needs.push((
            "one checkpoint of recurrent state, which the prompt cache in RAM keeps whatever \
             the tier of experts holds (page-locked)",
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
    let (ram, why_ram) = ram_limit_of(
        options,
        site,
        &needs,
        "the tier of experts gets nothing beyond the step buffers",
    )?;
    // What the engine may page-lock: the limit less the buffers in transit.
    let lockable = ram - transit;
    if near.is_some() && required + floor + far_most > lockable {
        tracing::warn!(
            "--ram {}: the K and V of the whole context beyond VRAM take up to {}, more than the \
             limit leaves beside what the engine needs to run; a sequence that grows into the \
             rest fails its step",
            Bytes(ram),
            Bytes(far_most)
        );
    }
    let runtime_config = RuntimeConfig {
        pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: parallel,
        queue: options.queue,
        window_rows: if drafting { window } else { 1 },
        cache_ram: lockable - required,
        ssd: ssd.clone(),
        generation_floor: options
            .generation_floor
            .unwrap_or(RuntimeConfig::default().generation_floor),
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: VRAM_HEADROOM,
        ram_limit: lockable,
        ..GpuConfig::new(&runtime_config, &*model)
    };
    tracing::info!(
        "plan: RAM, at most {} for the engine ({why_ram}): {} for its step buffers{}, the rest \
         for the tier of experts first, which holds the experts that have no slot in VRAM as \
         far as the memory goes, gives pieces back when the machine needs them, the coldest \
         experts of every layer with them, and takes them again when the memory is free; the \
         prompt cache in RAM has what the tier leaves, for pages that left the devices and \
         checkpoints of recurrent state of {} each over all devices{}",
        Bytes(ram),
        Bytes(required),
        match transit {
            0 => String::new(),
            bytes => format!(
                ", at most {} for checkpoints in transit between RAM and the prompt cache's \
                 store on SSD (ordinary memory, taken as it is first used)",
                Bytes(bytes)
            ),
        },
        Bytes(checkpoint),
        match (floor, &durable) {
            (0, _) => String::new(),
            (bytes, Some(_)) => format!(
                ", and {} that state passes through on its way to the prompt cache's store on SSD \
                 and back, a checkpoint place and pages, which the cache keeps whatever the tier \
                 holds",
                Bytes(bytes)
            ),
            (bytes, None) => format!(
                ", and {} for one checkpoint place, which the cache keeps whatever the tier holds",
                Bytes(bytes)
            ),
        },
    );
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let machine = Some(crate::load::watch());
    let started =
        Runtime::start_gpu_following(model.clone(), devices, runtime_config, gpu, machine);
    let runtime = started.map_err(|e| {
        let error = e.to_string();
        let short_of_memory = error.to_lowercase().contains("memory");
        // The RAM tier's own refusal (not the VRAM budget's, whose words contain these).
        let error = LoadError::Start(if error.contains("the RAM budget") {
            format!(
                "{error}. The tier of experts is what --ram leaves after the {} the engine \
                 needs for its buffers: name a larger --ram",
                Bytes(required)
            )
        } else if options.context.is_none() && short_of_memory {
            format!(
                "{error}. The context of {context} tokens was chosen from an estimate of \
                 what the model takes; name a smaller one with --context"
            )
        } else {
            error
        });
        if short_of_memory { short(site, error) } else { error }
    })?;
    let runtime = runtime.with_chat_template(template);
    tracing::info!(
        "loaded in {:.1} s ({:.1} s since the start)",
        loading.elapsed().as_secs_f64(),
        begun.elapsed().as_secs_f64()
    );
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
    log_started(&runtime, ssd.as_ref(), &model.drafters());
    fill_tiers(&runtime, &model, &name)?;
    log_expert_costs(&model);

    // What it took.
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
    watch_reads(std::sync::Arc::downgrade(&model));
    if let Some(report) = model.mtp_report() {
        watch_drafts(std::sync::Arc::downgrade(&model));
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
    match near {
        Some(near) => tracing::info!(
            "context: {context} tokens of sequence state; K and V of {} tokens in VRAM, of the \
             rest in page-locked RAM as the sequences grow into it (booked in --ram above the \
             tier of experts, which gives the memory up), the indexer's keys all in VRAM",
            plan::pool_tokens(near, parallel)
        ),
        None => tracing::info!("context: {context} tokens of sequence state, all of it in VRAM"),
    }
    Ok(Loaded::new(runtime, name, held, ram))
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
        Some(std::sync::Arc::new(table))
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
fn log_trial(placement: &ares_model::qwen4exp::device::Placement, held: &[crate::load::Held]) {
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
    tracing::info!(
        "plan: the role of device {worker}, timed at load in {:.1} s ({}): {}; kept: {}",
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
fn pinned(spread: Spread, placement: &ares_model::qwen4exp::device::Placement) -> Spread {
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
struct Split {
    choice: TensorChoice,
    file: PathBuf,
}

impl Split {
    fn file(dir: &std::path::Path, model: &std::path::Path) -> PathBuf {
        let stem = model.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy());
        dir.join(format!("{stem}.split"))
    }

    fn devices(devices: &[ares_cuda::Device]) -> String {
        join(devices.iter().map(|d| d.info().name.clone()), ", ")
    }

    /// The split kept for `model` on `devices`, if there is one for them.
    fn read(
        dir: &std::path::Path,
        model: &std::path::Path,
        devices: &[ares_cuda::Device],
    ) -> Option<Self> {
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
    fn write(
        dir: &std::path::Path,
        model: &std::path::Path,
        devices: &[ares_cuda::Device],
        choice: &TensorChoice,
    ) {
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

/// Where the usage profiles of the experts are kept: in the directory of the
/// prompt cache when `--cache` names one (a file of its own next to the
/// stores, which take only directories), otherwise in the user's cache
/// directory (`$XDG_CACHE_HOME/ares`, `~/.cache/ares`). `None` when the user
/// has neither.
fn profile_dir(options: &LoadOptions) -> Option<PathBuf> {
    if let Some(dir) = &options.cache {
        return Some(dir.clone());
    }
    let xdg = std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).filter(|p| p.is_absolute());
    let base = xdg.or_else(|| {
        std::env::var_os("HOME")
            .filter(|home| !home.is_empty())
            .map(|home| PathBuf::from(home).join(".cache"))
    })?;
    Some(base.join("ares"))
}

/// Fills the tiers of experts before the first request: from the usage profile
/// when there is one for this model file ([`Qwen4Exp::serving`]); otherwise
/// the model reads [`WARM_UP`], and the profile is written at once from what
/// that used. One line in the log says which.
fn fill_tiers(runtime: &Runtime, model: &Qwen4Exp, name: &str) -> Result<(), LoadError> {
    let Some(ProfileReport { path, found, filled }) = model.serving() else {
        let took = warm_up(runtime, name)?;
        tracing::info!(
            "warm-up: read a text of German, English and code to fill the tiers of experts, \
             {took:.1} s (no directory for a usage profile of the experts)"
        );
        return Ok(());
    };
    let path = path.display();
    if let (ProfileFound::Found { known }, Some((stats, time))) = (&found, filled)
        && stats.vram + stats.ram > 0
    {
        tracing::info!(
            "expert profile {path}: found ({known} experts with heat); the tiers were filled \
             from it in {:.1} s, {} experts into VRAM and {} into the RAM tier, {} read; no \
             warm-up",
            time.as_secs_f64(),
            stats.vram,
            stats.ram,
            Bytes(stats.read_bytes),
        );
        return Ok(());
    }
    let why = match &found {
        ProfileFound::None => "none for this model file".to_owned(),
        ProfileFound::Ignored(why) => format!("ignored: {why}"),
        ProfileFound::Found { .. } => "found, and it brought nothing into the tiers".to_owned(),
    };
    let took = warm_up(runtime, name)?;
    let written = match model.save_profile() {
        Some(Ok(_)) => "written now".to_owned(),
        Some(Err(error)) => format!("not written: {error}"),
        None => "not written: the warm-up ran no step".to_owned(),
    };
    // The warm-up's steps brought what they missed into the slots that were
    // free; the fill makes every layer's VRAM hold the hottest of what it used.
    let filled = match model.fill_from_heat() {
        Some((stats, time)) => format!(
            "; the tiers were filled from its heat in {:.1} s, {} experts into VRAM and {} into \
             the RAM tier, {} read",
            time.as_secs_f64(),
            stats.vram,
            stats.ram,
            Bytes(stats.read_bytes),
        ),
        None => String::new(),
    };
    tracing::info!(
        "expert profile {path}: {why}; the warm-up read a text of German, English and code to \
         fill the tiers, {took:.1} s; the profile is {written}{filled}"
    );
    Ok(())
}

/// Logs what serving a miss costs on this machine, as the expert store
/// measured it: a CPU job by the types of the experts, an upload by device.
fn log_expert_costs(model: &Qwen4Exp) {
    let Some(costs) = model.expert_costs() else { return };
    let us = |t: Option<f64>| t.map_or("-".to_owned(), |t| format!("{:.0}", t * 1e6));
    let cpu = costs
        .cpu
        .iter()
        .map(|((gate, down), one, eight)| {
            format!("{gate:?}/{down:?} {} and {}", us(*one), us(*eight))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let upload = costs
        .upload
        .iter()
        .enumerate()
        .map(|(device, t)| format!("device {device} {}", us(*t)))
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!(
        "experts outside VRAM, measured: a CPU job per expert and row in us (one row, and each \
         of eight) by the types of gate and up and of down: {cpu}; an upload in us: {upload}"
    );
}

/// Reads [`WARM_UP`] and writes a few tokens after it, so that the tiers of
/// experts hold what a text of German, English and code needs before the first
/// request comes: the slots start empty, and the first text a model reads
/// otherwise waits for the drive at nearly every layer (a minute for a prompt
/// of a thousand tokens, against ten seconds once the slots are filled).
/// Returns the seconds it took.
fn warm_up(runtime: &Runtime, model: &str) -> Result<f64, LoadError> {
    let begun = Instant::now();
    let (done, wait) = mpsc::channel();
    let sink = EventSink::new(move |event| {
        match event {
            Event::Done { .. } => {
                let _ = done.send(Ok(()));
            }
            Event::Error(error) => {
                let _ = done.send(Err(error.to_string()));
            }
            Event::Text(_) | Event::Logprobs(_) => {}
        }
        true
    });
    let request = GenerateRequest {
        model: model.to_owned(),
        prompt: Prompt::Text(WARM_UP.to_owned()),
        sampling: Sampling::default(),
        max_tokens: Some(32),
        stop: Vec::new(),
        draft: true,
        structure: None,
        priority: 0,
        id: RequestId(0),
    };
    runtime.submit(request, sink).map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    wait.recv_timeout(WARM_UP_LIMIT)
        .map_err(|_| LoadError::Start("the warm-up did not end in time".into()))?
        .map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    Ok(begun.elapsed().as_secs_f64())
}

/// How often the log says what the MTP block did, while it does something.
const DRAFT_WATCH: Duration = Duration::from_secs(20);

/// Logs what the model's MTP block did since the engine started, whenever
/// that has changed: the links it made and what one cost, and how the drafts
/// of each link fared with the rows that verified them. Ends with the model.
fn watch_drafts(model: std::sync::Weak<Qwen4Exp>) {
    let watch = move || {
        let mut links = 0;
        loop {
            std::thread::sleep(DRAFT_WATCH);
            let Some(model) = model.upgrade() else { return };
            let times = model.mtp_times();
            if times.links == links {
                continue;
            }
            links = times.links;
            let longest = times.offered.iter().rposition(|n| *n > 0).map_or(0, |i| i + 1);
            let rates = join(
                times.offered[..longest]
                    .iter()
                    .zip(&times.accepted)
                    .map(|(offered, accepted)| format!("{accepted} of {offered}")),
                ", ",
            );
            tracing::info!(
                "MTP block: {} links in {} steps, {:.2} ms a link ({:.2} ms a step that \
                 drafts); accepted per link, of the drafts that were verified: {rates}; {} rows \
                 only wrote keys",
                times.links,
                times.steps,
                times.chain_ms / times.links.max(1) as f64,
                times.chain_ms / times.steps.max(1) as f64,
                times.key_rows,
            );
        }
    };
    // Without the thread the engine runs all the same.
    let _ = std::thread::Builder::new().name("ares-draft-watch".into()).spawn(watch);
}

/// Logs, whenever it has changed, what the reads of experts from the model file took since
/// the engine started (the reader's counters, the latencies of the newest reads) and what the
/// layers of wide steps waited for. Ends with the model.
fn watch_reads(model: std::sync::Weak<Qwen4Exp>) {
    let watch = move || {
        let (mut requests, mut layers) = (0, 0);
        loop {
            std::thread::sleep(DRAFT_WATCH);
            let Some(model) = model.upgrade() else { return };
            let mut read = false;
            if let Some(stats) = model.read_stats() {
                let now = stats.demand.requests + stats.prefetch.requests;
                read = now != requests;
                if read {
                    requests = now;
                    let ms = |d: Duration| d.as_secs_f64() * 1e3;
                    let (d, p) = (&stats.demand_latency, &stats.prefetch_latency);
                    tracing::info!(
                        "reads of the model file: demand {} reads, {}; prefetch {} reads, {}, \
                         {} cancelled, {} promoted; demand latency median {:.2} ms, p95 {:.2}, \
                         p99 {:.2}; prefetch median {:.2} ms, p99 {:.2}; at most {} chunks at \
                         the drive",
                        stats.demand.requests,
                        Bytes(stats.demand.device_bytes),
                        stats.prefetch.requests,
                        Bytes(stats.prefetch.device_bytes),
                        stats.cancelled,
                        stats.promoted,
                        ms(d.p50),
                        ms(d.p95),
                        ms(d.p99),
                        ms(p.p50),
                        ms(p.p99),
                        stats.peak_in_flight,
                    );
                }
            }
            if let Some(odds) = model.ahead_odds().filter(|_| read) {
                let rate = |(n, p): (f32, f32)| {
                    if n < 1.0 { "-".to_owned() } else { format!("{:.0}%", 100.0 * p / n) }
                };
                let kinds =
                    ["one row, 1 ahead", "one row, 2 ahead", "rows, 1 ahead", "rows, 2 ahead"];
                let lines = kinds.iter().zip(&odds).filter(|(_, places)| places[0].0 >= 1.0);
                for (kind, places) in lines {
                    tracing::info!(
                        "predictions of experts in no tier then picked, {kind}, by place: {}",
                        join(places.iter().map(|o| rate(*o)), " "),
                    );
                }
            }
            if let Some(wide) = model.wide_stats()
                && wide.layers != layers
            {
                layers = wide.layers;
                let s = |d: Duration| d.as_secs_f64();
                tracing::info!(
                    "wide steps: {} layers, {} experts picked, {} without a slot, {} rounds \
                     after the first; waited {:.1} s for route records, {:.1} s for first \
                     rounds, {:.1} s for misses to arrive, {:.1} s for later rounds; the layers \
                     took {:.1} s",
                    wide.layers,
                    wide.picked,
                    wide.missing,
                    wide.extra_rounds,
                    s(wide.record_wait),
                    s(wide.compute_wait),
                    s(wide.bring_wait),
                    s(wide.round_wait),
                    s(wide.layer_time),
                );
                if let Some(odds) = model.guess_odds() {
                    let named = |(n, p): (f32, f32)| format!("{:.0} of {:.0}", p, n);
                    tracing::info!(
                        "guesses of wide steps named by their layer's record, newer counts \
                         weighing more: the step before's misses {}; predicted for 1, 2, 3-4, \
                         5-8, 9-16, 17-32, 33-64, more rows: {}",
                        named(odds[0]),
                        join(odds[1..].iter().map(|o| named(*o)), ", "),
                    );
                }
            }
        }
    };
    // Without the thread the engine runs all the same.
    let _ = std::thread::Builder::new().name("ares-read-watch".into()).spawn(watch);
}

fn ordinals_of(held: &[crate::load::Held]) -> String {
    join(held.iter().map(|h| h.device.ordinal()), ", ")
}
