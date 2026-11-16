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
    Options, ProfileFound, ProfileReport, PruneSetup, Qwen4Exp, SessionOptions, Spread, place,
    profile_key, step_rows,
};
use ares_model::qwen4exp::{Config, DraftWeights};
use ares_runtime::expert_store::ExpertNeighbours;
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, required_ram};
use ares_tokenizer::Tokenizer;

use crate::load::{
    LoadError, Loaded, VRAM_HEADROOM, describe_links, hold_devices, join, ram_limit, refuse, short,
};
use crate::models::Site;
use crate::plan::{self, CELLS_PER_PAGE, DEFAULT_PARALLEL, Drafter, LoadOptions, SplitKind};

/// The most rows of a step that generates (the newest token of every sequence and the drafts
/// they verify): what the decode kernels of the experts take. Only the slices of a prompt are
/// wider.
const MAX_ROWS: u32 = ares_cuda::MAX_BATCH;

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
    if options.precision == ares_cuda::Precision::Exact {
        return Err(refuse(
            "--precision exact is not a choice for this model: its products have one kind of \
             arithmetic",
        ));
    }
    // The model's draft block is a file of its own, which --drafters names:
    // the model file carries none. Without it nothing speculates (the lookup
    // in earlier text takes part only next to a model drafter), no window is
    // reserved, and the options of drafting mean nothing.
    let drafter = match options.drafters.as_deref() {
        None | Some([]) => None,
        Some([Drafter::Block(path)]) => Some(path.clone()),
        Some([Drafter::Mtp]) => {
            return Err(refuse(
                "--drafters mtp: this model file carries no draft block; the block is a file of \
                 its own, which --drafters names by its path",
            ));
        }
        Some(_) => {
            return Err(refuse(
                "--drafters: this model drafts with one draft block; name its file alone",
            ));
        }
    };
    // The file is read before a device is touched: one that is no draft block
    // of this model is refused by name.
    let drafter = drafter
        .map(|path| {
            let named = |e: String| refuse(format!("--drafters {}: {e}", path.display()));
            let file = Gguf::open(&path).map_err(|e| named(e.to_string()))?;
            DraftWeights::load(&file, &config).map_err(|e| named(e.to_string()))?;
            Ok::<_, LoadError>((path, file))
        })
        .transpose()?;
    let window = options.window.unwrap_or(DRAFT_WINDOW);
    if options.window.is_some() && drafter.is_none() {
        return Err(refuse("--window without a model drafter does nothing"));
    }
    if drafter.is_some() && !(2..=MAX_ROWS).contains(&window) {
        return Err(refuse(format!(
            "--window {window}: a window is the newest token and its drafts, 2 to {MAX_ROWS} \
             rows (a step that verifies drafts has {MAX_ROWS} rows for all sequences together)"
        )));
    }
    if options.draft_vocabulary.is_some() {
        return Err(refuse(
            "--draft-vocab does nothing for this model: its draft block picks from the whole \
             output head",
        ));
    }
    let parallel = options.parallel.unwrap_or(DEFAULT_PARALLEL);
    if parallel > MAX_ROWS {
        return Err(refuse(format!(
            "--parallel {parallel}: a step has {MAX_ROWS} rows at most, one for each sequence"
        )));
    }
    // Rows of a step that verifies drafts, for all sequences together, and
    // the rows whose logits a step can want.
    let window_rows = if drafter.is_some() { MAX_ROWS } else { 0 };
    let logit_rows = window_rows.max(parallel);
    if options.context == Some(0) {
        return Err(refuse("--context must be at least 1"));
    }
    let prune = expert_prune(options, &gguf, &config)?;

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
    describe_links(&held)?;
    let devices: Vec<_> = held.iter().map(|h| h.device.clone()).collect();

    // The model's settings that do not depend on the context.
    let model_context = config.n_ctx as u32;
    // The rows of a step: the devices decide, by the memory a step's scratch may take.
    let max_rows = step_rows(
        &Gguf::open(&options.model)?,
        &devices,
        &Options { kv, logit_rows, window_rows, max_slots: parallel, ..Options::default() },
    )
    .map_err(|e| LoadError::Model(e.to_string()))?;
    tracing::info!("steps of up to {max_rows} rows");
    let session = |context: u32, pages: u32| SessionOptions {
        part: Options {
            kv,
            max_rows,
            logit_rows,
            window_rows,
            max_slots: parallel,
            cells: context.min(model_context),
            pool_pages: pages,
            prune: prune.clone(),
            ..Options::default()
        },
        headroom: VRAM_HEADROOM,
        spread,
        ..SessionOptions::default()
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
    // from the operator the K/V takes a share of what the experts could use.
    let (context, pages, why_context) = match options.context {
        Some(context) => (context, plan::pool_pages(context, parallel), "as asked"),
        None => {
            let trial = TRIAL_CONTEXT.min(model_context);
            let (placement, page_bytes) = place(
                &Gguf::open(&options.model)?,
                &devices,
                &session(trial, plan::pool_pages(trial, parallel)),
            )
            .map_err(|e| named_split(e.to_string()))?;
            let slots: Vec<u64> = placement.devices.iter().map(|d| d.slots).collect();
            let fit = plan::pages_beside_experts(&slots, &page_bytes);
            let context = plan::default_context(fit, parallel, model_context);
            if context == 0 {
                let error = refuse(
                    "the devices have no room for context next to the experts' slots; name one \
                     with --context to try it anyway",
                );
                return Err(short(site, error));
            }
            (
                context,
                plan::pool_pages(context, parallel),
                "a thirty-second of the room for experts' slots, which every page of K and V \
                 takes from",
            )
        }
    };
    let longest = context.min(model_context);
    let opts = session(longest, pages);
    let (placement, page_bytes) = place(&Gguf::open(&options.model)?, &devices, &opts)
        .map_err(|e| named_split(e.to_string()))?;

    tracing::info!(
        "plan: {} ({why_split})",
        match &placement.tensor {
            Some(tensor) => format!(
                "both devices on every layer: device {} holds the streams and runs the mixers, \
                 the other computes its share of every layer's recurrent heads and the experts \
                 it owns",
                held[tensor.mixer].device.ordinal()
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
        match &drafter {
            Some((path, _)) => format!(
                "the model's draft block ({}) drafts, on the device that holds the output \
                 head: windows of up to {window} rows a sequence, {MAX_ROWS} rows a step, as \
                 long as they pay; the lookup in earlier text joins",
                path.display()
            ),
            None => "no speculation: the model drafts with a draft block from a file of its \
                     own (--drafters <file>), and earlier text alone is not a drafter"
                .to_owned(),
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
    let drafting = drafter.is_some();
    if let Some((path, file)) = drafter {
        model = model
            .with_draft(file)
            .map_err(|e| LoadError::Model(format!("--drafters {}: {e}", path.display())))?;
    }
    let model = std::sync::Arc::new(model);

    // Host RAM: one limit for the engine. What it needs to run is taken from
    // it first (the step buffers of the devices and the parts'); the tier of
    // experts takes the rest, as far as the experts that VRAM has no slot for
    // can use it.
    let required = required_ram(&*model, parallel);
    let (ram, why_ram) = ram_limit(
        options,
        site,
        required,
        "the tier of experts gets nothing beyond the step buffers",
    )?;
    let runtime_config = RuntimeConfig {
        pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: parallel,
        queue: options.queue,
        window_rows: if drafting { window } else { 1 },
        cache_ram: 0,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: VRAM_HEADROOM,
        ram_limit: ram,
        ..GpuConfig::new(&runtime_config, &*model)
    };
    tracing::info!(
        "plan: RAM, at most {} for the engine ({why_ram}): {} for its step buffers, the rest for \
         the tier of experts, which holds the experts that have no slot in VRAM as far as the \
         memory goes, gives pieces back when the machine needs them, the coldest experts of \
         every layer with them, and takes them again when the memory is free",
        Bytes(ram),
        Bytes(required),
    );
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let machine = Some(crate::load::watch());
    let started =
        Runtime::start_gpu_following(model.clone(), devices, runtime_config, gpu, machine);
    let runtime = started.map_err(|e| {
        let error = e.to_string();
        let short_of_memory = error.to_lowercase().contains("memory");
        let error = LoadError::Start(if error.contains("RAM budget") {
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
    if let Some(calibration) = runtime.snapshot().and_then(|snapshot| snapshot.calibration) {
        let table = |points: &[(u32, f64)]| {
            join(points.iter().map(|(n, ms)| format!("{n}: {ms:.1}")), ", ")
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
    fill_tiers(&runtime, &model, &name)?;

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
    if let Some(report) = model.draft_report() {
        watch_drafts(std::sync::Arc::downgrade(&model));
        let (vram, ram_slots) = model.draft_slots().unwrap_or((0, 0));
        tracing::info!(
            "draft block, on device {}: dense weights {}, K, V and keys {}, buffers {}; its {} \
             experts share the tiers: {vram} slots in VRAM, {ram_slots} in RAM",
            held[report.index].device.ordinal(),
            Bytes(report.weights),
            Bytes(report.pools),
            Bytes(report.buffers),
            config.experts.count,
        );
    }
    tracing::info!(
        "context: {context} tokens of sequence state, all of it in VRAM (state is not moved to \
         RAM yet, so a context that does not fit is refused at load)"
    );
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
    tracing::info!(
        "expert profile {path}: {why}; the warm-up read a text of German, English and code to \
         fill the tiers, {took:.1} s; the profile is {written}"
    );
    Ok(())
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
        id: RequestId(0),
    };
    runtime.submit(request, sink).map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    wait.recv_timeout(WARM_UP_LIMIT)
        .map_err(|_| LoadError::Start("the warm-up did not end in time".into()))?
        .map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    Ok(begun.elapsed().as_secs_f64())
}

/// How often the log says what the draft block did, while it does something.
const DRAFT_WATCH: Duration = Duration::from_secs(20);

/// Logs what the model's draft block did since the engine started, whenever
/// that has changed: the links it made and what one cost, and how the drafts
/// of each link fared with the rows that verified them. Ends with the model.
fn watch_drafts(model: std::sync::Weak<Qwen4Exp>) {
    let watch = move || {
        let mut links = 0;
        loop {
            std::thread::sleep(DRAFT_WATCH);
            let Some(model) = model.upgrade() else { return };
            let times = model.draft_times();
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
                "draft block: {} links in {} steps, {:.2} ms a link ({:.2} ms a step that \
                 drafts); accepted per link, of the drafts that were verified: {rates}; {} rows \
                 only wrote keys, {:.3} ms of the host each",
                times.links,
                times.chain_steps,
                times.chain_ms / times.links.max(1) as f64,
                times.chain_ms / times.chain_steps.max(1) as f64,
                times.key_rows,
                times.key_ms / times.key_rows.max(1) as f64,
            );
        }
    };
    // Without the thread the engine runs all the same.
    let _ = std::thread::Builder::new().name("ares-draft-watch".into()).spawn(watch);
}

fn ordinals_of(held: &[crate::load::Held]) -> String {
    join(held.iter().map(|h| h.device.ordinal()), ", ")
}
