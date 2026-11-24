//! Threads that log, while the model is served, what its MTP block did and
//! what the reads of experts from the model file took. Each ends with the
//! model.

use std::sync::Weak;
use std::time::Duration;

use ares_core::units::Bytes;
use ares_model::qwen4exp::device::Qwen4Exp;

use crate::load::join;

/// How often the log says what the MTP block did, while it does something.
const DRAFT_WATCH: Duration = Duration::from_secs(20);

/// Logs what the model's MTP block did since the engine started, whenever
/// that has changed: the links it made and what one cost, and how the drafts
/// of each link fared with the rows that verified them. Ends with the model.
pub(super) fn watch_drafts(model: Weak<Qwen4Exp>) {
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
pub(super) fn watch_reads(model: Weak<Qwen4Exp>) {
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
