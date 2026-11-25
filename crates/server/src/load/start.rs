//! The tail both loaders share: from a model built for its devices and the
//! host RAM it may use to a running engine, with the plan's line of the RAM,
//! the runtime's settings, the start, and what the engine found when it
//! started.

use std::sync::Arc;
use std::time::Instant;

use ares_core::units::Bytes;
use ares_cuda::Device;
use ares_runtime::{GpuConfig, Model, Runtime, RuntimeConfig, SsdConfig};

use super::ram::{self, Budget, Rest};
use super::{LoadError, VRAM_HEADROOM, join, short};
use crate::models::Site;
use crate::plan::{CELLS_PER_PAGE, LoadOptions};

/// What a loader decided that the runtime starts with.
pub(crate) struct Engine {
    /// The pages of the pool of sequence state.
    pub pages: u32,
    /// The sequences that run at once.
    pub parallel: u32,
    /// The most rows of a sequence's window: 1 when the model does not draft.
    pub window: u32,
    /// The most rows of a step; 0 for the model's own.
    pub max_rows: u32,
    /// The tokens of context, for the advice when an estimate was too low.
    pub context: u32,
}

/// Starts the runtime over `model` on `devices` with `budget`: logs the
/// plan's line of the RAM, loads the weights and starts the engine, and
/// gives it `template` (the chat template in effect). A failure that is a
/// lack of memory says the way around it, and is no room next to other
/// models at `site`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn start<M: Model + 'static>(
    model: &Arc<M>,
    devices: Vec<Device>,
    options: &LoadOptions,
    site: &Site,
    budget: &Budget,
    engine: Engine,
    template: Option<String>,
    begun: Instant,
) -> Result<Runtime, LoadError> {
    if template.is_none() {
        tracing::warn!(
            "the model file has no chat template: /v1/chat/completions will refuse requests, \
             /v1/completions works"
        );
    }
    budget.log();
    let runtime_config = RuntimeConfig {
        pages: engine.pages,
        cells_per_page: CELLS_PER_PAGE,
        max_slots: engine.parallel,
        queue: options.queue,
        window_rows: engine.window,
        cache_ram: budget.cache_ram(),
        ssd: budget.ssd.clone(),
        max_rows: engine.max_rows,
        generation_floor: options
            .generation_floor
            .unwrap_or(RuntimeConfig::default().generation_floor),
        watchdog: match options.watchdog {
            None => RuntimeConfig::default().watchdog,
            Some(0) => None,
            Some(seconds) => Some(std::time::Duration::from_secs(seconds)),
        },
        ..RuntimeConfig::default()
    };
    let gpu = GpuConfig {
        vram_headroom: VRAM_HEADROOM,
        ram_limit: budget.lockable(),
        ..GpuConfig::new(&runtime_config, &**model)
    };
    tracing::info!("loading the weights");
    let loading = Instant::now();
    let machine = Some(ram::machine());
    let started =
        Runtime::start_gpu_following(model.clone(), devices, runtime_config, gpu, machine);
    let runtime = started.map_err(|e| {
        let error = e.to_string();
        let short_of_memory = error.to_lowercase().contains("memory");
        // The RAM tier's own refusal (not the VRAM budget's, whose words
        // contain these).
        let error = LoadError::Start(
            if budget.rest() == Rest::Experts && error.contains("the RAM budget") {
                format!(
                    "{error}. The tier of experts is what --ram leaves after the {} the engine \
                     needs for its buffers: name a larger --ram",
                    Bytes(budget.required)
                )
            } else if options.context.is_none() && short_of_memory {
                // A context that was estimated to fit and did not is the
                // server's mistake, and the operator should know the way
                // around it.
                format!(
                    "{error}. The context of {} tokens was chosen from an estimate of what the \
                     model takes; name a smaller one with --context",
                    engine.context
                )
            } else {
                error
            },
        );
        // The estimate was too low: next to other models that is no room.
        if short_of_memory { short(site, error) } else { error }
    })?;
    let runtime = runtime.with_chat_template(template);
    tracing::info!(
        "loaded in {:.1} s ({:.1} s since the start)",
        loading.elapsed().as_secs_f64(),
        begun.elapsed().as_secs_f64()
    );
    Ok(runtime)
}

/// What an engine found when it started: whether the prompt cache on SSD
/// (`ssd`, when one was asked for) is in use and what its store holds, and
/// the price of a step as the calibration measured it, by rows and for each
/// of the model's `drafters` by drafts.
pub(crate) fn log_started(
    runtime: &Runtime,
    ssd: Option<&SsdConfig>,
    drafters: &[ares_runtime::Drafter],
) {
    let Some(snapshot) = runtime.snapshot() else { return };
    if let Some(ssd) = ssd {
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
    if let Some(calibration) = snapshot.calibration {
        let table = |points: &[(u32, f64)]| {
            join(points.iter().map(|(n, ms)| format!("{n}: {ms:.1}")), ", ")
        };
        tracing::info!(
            "the price of a step was measured in {:.1} s ({} steps), in ms by rows: {}{}",
            calibration.seconds,
            calibration.steps,
            table(&calibration.step_ms),
            calibration
                .chain_ms
                .iter()
                .zip(drafters)
                .map(|(chain, drafter)| {
                    format!("; {} by drafts: {}", drafter.kind.name(), table(chain))
                })
                .collect::<String>()
        );
    }
}
