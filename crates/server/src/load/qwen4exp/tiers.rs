//! The tiers of experts before the first request: filled from the usage
//! profile of the model file when there is one, otherwise by a warm-up text,
//! whose use then writes the profile; and what serving a miss costs on this
//! machine.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ares_api::{Engine, Event, EventSink, GenerateRequest, Prompt, RequestId, Sampling};
use ares_core::units::Bytes;
use ares_model::qwen4exp::device::{ProfileFound, ProfileReport, Qwen4Exp};
use ares_runtime::Runtime;

use crate::load::LoadError;
use crate::plan::LoadOptions;

/// A text of German, English and code, about 700 tokens, that the model reads
/// once at the start (see [`warm_up`]).
const WARM_UP: &str = include_str!("warm_up.txt");

/// How long the warm-up may take: the experts of a text come from the drive,
/// and the drive may be busy with something else.
const WARM_UP_LIMIT: Duration = Duration::from_secs(1800);

/// Where the usage profiles of the experts are kept: in the directory of the
/// prompt cache when `--cache` names one (a file of its own next to the
/// stores, which take only directories), otherwise in the user's cache
/// directory (`$XDG_CACHE_HOME/ares`, `~/.cache/ares`). `None` when the user
/// has neither.
pub(super) fn profile_dir(options: &LoadOptions) -> Option<PathBuf> {
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
pub(super) fn fill_tiers(runtime: &Runtime, model: &Qwen4Exp, name: &str) -> Result<(), LoadError> {
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
pub(super) fn log_expert_costs(model: &Qwen4Exp) {
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
        cache: true,
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
