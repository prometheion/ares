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

use std::sync::mpsc;
use std::time::{Duration, Instant};

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_core::units::Bytes;
use ares_gguf::Gguf;
use ares_model::qwen4exp::Config;
use ares_model::qwen4exp::device::{Options, Qwen4Exp, SessionOptions, Spread, place};
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, required_ram};
use ares_tokenizer::{ChatOptions, Tokenizer};

use crate::load::{
    LoadError, Loaded, VRAM_HEADROOM, describe_links, draft_block_takes_part, hold_devices, join,
    ram_limit, refuse,
};
use crate::plan::{self, CELLS_PER_PAGE, DEFAULT_PARALLEL, LoadOptions, SplitKind};

/// The most rows of a step: what the expert kernels take.
const MAX_ROWS: u32 = ares_cuda::MAX_BATCH;

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
pub(crate) fn load(gguf: Gguf, options: &LoadOptions, begun: Instant) -> Result<Loaded, LoadError> {
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
    if kv.0.has_mean() || kv.1.has_mean() {
        return Err(refuse(format!(
            "K/V in {}, {}: the formats q7 to q4 need the pool of exact cells and its \
             conversion, which this model does not have; use f16 or q8",
            plan::format_name(kv.0),
            plan::format_name(kv.1)
        )));
    }
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
    // The model has no draft head built yet, and the lookup in earlier text
    // takes part only next to a model drafter: nothing speculates, so no
    // window is reserved, and the options of drafting mean nothing.
    draft_block_takes_part(
        options,
        false,
        "the draft head of this model is not built yet, so no model drafter takes part",
    )?;
    if options.window.is_some() {
        return Err(refuse(
            "--window does nothing for this model: it has no drafter of its own yet, so no \
             window is reserved",
        ));
    }
    if options.draft_vocabulary.is_some() {
        return Err(refuse(
            "--draft-vocab is for a model's own draft block, which this model does not use yet",
        ));
    }
    let parallel = options.parallel.unwrap_or(DEFAULT_PARALLEL);
    if parallel > MAX_ROWS {
        return Err(refuse(format!(
            "--parallel {parallel}: a step has {MAX_ROWS} rows at most, one for each sequence"
        )));
    }
    if options.context == Some(0) {
        return Err(refuse("--context must be at least 1"));
    }

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
    let session = |context: u32, pages: u32| SessionOptions {
        part: Options {
            kv,
            max_rows: MAX_ROWS,
            logit_rows: parallel,
            window_rows: 0,
            max_slots: parallel,
            cells: context.min(model_context),
            pool_pages: pages,
            ..Options::default()
        },
        headroom: VRAM_HEADROOM,
        spread,
        ..SessionOptions::default()
    };
    let named_split = |e: String| {
        LoadError::Model(match options.split {
            Some(kind) => format!("--split {}: {e}", kind.name()),
            None => e,
        })
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
                return Err(refuse(
                    "the devices have no room for context next to the experts' slots; name one \
                     with --context to try it anyway",
                ));
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
        "plan: K in {}, V in {}; no speculation: the model's draft head is not built yet, and \
         earlier text alone is not a drafter",
        plan::format_name(kv.0),
        plan::format_name(kv.1),
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
    let name = options
        .model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "model".to_owned());
    let chat = ChatOptions { variables: options.chat_variables.clone(), ..ChatOptions::default() };
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| LoadError::Model(e.to_string()))?;
    if !tokenizer.has_chat_template() {
        tracing::warn!(
            "the model file has no chat template: /v1/chat/completions will refuse requests, \
             /v1/completions works"
        );
    }
    let model = std::sync::Arc::new(
        Qwen4Exp::new(gguf, tokenizer, name.clone(), &devices, opts, longest)
            .map_err(|e| LoadError::Model(e.to_string()))?
            .with_chat_options(chat),
    );

    // Host RAM: one limit for the engine. What it needs to run is taken from
    // it first (the step buffers of the devices and the parts'); the tier of
    // experts takes the rest, as far as the experts that VRAM has no slot for
    // can use it.
    let required = required_ram(&*model, parallel);
    let (ram, why_ram) =
        ram_limit(options, required, "the tier of experts gets nothing beyond the step buffers")?;
    let runtime_config = RuntimeConfig {
        pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: parallel,
        window_rows: 1,
        checkpoint_budget: 0,
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: VRAM_HEADROOM,
        ram_limit: ram,
        ..GpuConfig::new(&runtime_config, &*model)
    };
    tracing::info!(
        "plan: RAM, at most {} for the engine ({why_ram}): {} for its step buffers, the rest for \
         the tier of experts, which holds the experts that have no slot in VRAM as far as it \
         goes and shrinks for nothing",
        Bytes(ram),
        Bytes(required),
    );
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let runtime = Runtime::start_gpu(model.clone(), devices, runtime_config, gpu).map_err(|e| {
        let error = e.to_string();
        let short_of_memory = error.to_lowercase().contains("memory");
        LoadError::Start(if error.contains("RAM budget") {
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
        })
    })?;
    tracing::info!(
        "loaded in {:.1} s ({:.1} s since the start)",
        loading.elapsed().as_secs_f64(),
        begun.elapsed().as_secs_f64()
    );
    warm_up(&runtime, &name)?;

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
             tier of {} ({:.1} %); the slots fill from what the routers pick",
            100.0 * vram as f64 / all as f64,
            Bytes(ram_bytes),
            100.0 * ram_slots as f64 / all as f64,
        );
    }
    tracing::info!(
        "context: {context} tokens of sequence state, all of it in VRAM (state is not moved to \
         RAM yet, so a context that does not fit is refused at load)"
    );
    Ok(Loaded::new(runtime, name, held))
}

/// Reads [`WARM_UP`] and writes a few tokens after it, so that the tiers of
/// experts hold what a text of German, English and code needs before the first
/// request comes: the slots start empty, and the first text a model reads
/// otherwise waits for the drive at nearly every layer (a minute for a prompt
/// of a thousand tokens, against ten seconds once the slots are filled).
fn warm_up(runtime: &Runtime, model: &str) -> Result<(), LoadError> {
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
        id: RequestId(0),
    };
    runtime.submit(request, sink).map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    wait.recv_timeout(WARM_UP_LIMIT)
        .map_err(|_| LoadError::Start("the warm-up did not end in time".into()))?
        .map_err(|e| LoadError::Start(format!("warm-up: {e}")))?;
    tracing::info!(
        "warm-up: read a text of German, English and code to fill the tiers of experts, {:.1} s",
        begun.elapsed().as_secs_f64()
    );
    Ok(())
}

fn ordinals_of(held: &[crate::load::Held]) -> String {
    join(held.iter().map(|h| h.device.ordinal()), ", ")
}
