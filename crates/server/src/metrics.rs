//! `GET /metrics`: what an engine has done and holds, in the text format
//! monitoring systems scrape: the counters and gauges of its statistics
//! ([`ares_api::EngineStats`]) and the histograms of how long its requests
//! took ([`latency`]).

pub(crate) mod latency;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};

use crate::error::ApiError;
use crate::models::AcquireError;
use crate::routes::{App, model_in_query, wanted};

/// The gauge that says whether the model is loaded: the one metric of a
/// model that is not, which has no engine to count.
fn loaded_gauge(loaded: bool) -> String {
    format!(
        "# HELP ares_model_loaded Whether the model is loaded (its counters start again at a \
         load)\n# TYPE ares_model_loaded gauge\nares_model_loaded {}\n",
        u8::from(loaded)
    )
}

fn text(body: String) -> Response {
    ([(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"))], body)
        .into_response()
}

/// `GET /metrics`: what an engine has done and holds, in the text format
/// monitoring systems scrape. The engine is the one of `?model=<id>`, or of
/// the only model there is. A model that is not loaded has none: it answers
/// with `ares_model_loaded 0` alone, so that a scraper does not take the
/// server for down.
pub(crate) async fn metrics(State(app): State<App>, request: Request) -> Response {
    let name = match wanted(&app, model_in_query(request.uri().query())) {
        Ok(name) => name,
        Err(error) => return error.into_response(),
    };
    let lease = match app.models.loaded(&name).await {
        Ok(lease) => lease,
        Err(AcquireError::NotLoaded { .. }) => return text(loaded_gauge(false)),
        Err(error) => return ApiError::from(error).into_response(),
    };
    // The engine may wait for the end of its step before it answers.
    let latency = lease.flight().clone();
    let stats = tokio::task::spawn_blocking(move || lease.engine().stats()).await.ok().flatten();
    let latency = &latency.latency;
    match stats {
        Some(stats) => text(loaded_gauge(true) + &stats_text(&stats) + &latency.render()),
        None => ApiError::unsupported("the engine keeps no statistics").into_response(),
    }
}

/// The counters and gauges of `GET /metrics`.
fn stats_text(stats: &ares_api::EngineStats) -> String {
    let (ram, ssd) = (&stats.ram, &stats.ssd);
    let rows: [(&str, &str, &str, u64); 68] = [
        (
            "cache_transfer_nanoseconds_total",
            "counter",
            "Time the step loop waited while the prompt cache moved state between the devices \
             and host memory, for RAM and the SSD alike",
            stats.cache_transfer_ns,
        ),
        (
            "cache_ram_pages",
            "gauge",
            "Pages of sequence state the prompt cache holds in RAM",
            ram.pages,
        ),
        ("cache_ram_bytes", "gauge", "Page-locked RAM the prompt cache's pages hold", ram.bytes),
        (
            "cache_ram_pages_in_total",
            "counter",
            "Pages that left the devices and were copied into the prompt cache in RAM",
            ram.pages_in,
        ),
        (
            "cache_ram_pages_read_total",
            "counter",
            "Pages a prompt took back to the devices from the prompt cache in RAM",
            ram.pages_read,
        ),
        (
            "cache_ram_pages_dropped_total",
            "counter",
            "Pages worth keeping that the prompt cache in RAM gave up unwritten",
            ram.pages_dropped,
        ),
        (
            "cache_ssd_pages_written_total",
            "counter",
            "Pages of sequence state the prompt cache wrote to SSD",
            ssd.pages_written,
        ),
        (
            "cache_ssd_checkpoints_written_total",
            "counter",
            "Checkpoints of recurrent state the prompt cache wrote to SSD",
            ssd.checkpoints_written,
        ),
        (
            "cache_ssd_written_bytes_total",
            "counter",
            "Bytes the prompt cache wrote to SSD",
            ssd.bytes_written,
        ),
        (
            "cache_ssd_write_nanoseconds_total",
            "counter",
            "Time the prompt cache's writer spent writing and syncing",
            ssd.write_ns,
        ),
        (
            "cache_ssd_pages_read_total",
            "counter",
            "Pages of sequence state the prompt cache read back from SSD",
            ssd.pages_read,
        ),
        (
            "cache_ssd_checkpoints_read_total",
            "counter",
            "Checkpoints of recurrent state the prompt cache read back from SSD",
            ssd.checkpoints_read,
        ),
        (
            "cache_ssd_read_bytes_total",
            "counter",
            "Bytes the prompt cache read back from SSD",
            ssd.bytes_read,
        ),
        (
            "cache_ssd_read_nanoseconds_total",
            "counter",
            "Time the prompt cache's readers spent reading and checking, summed",
            ssd.read_ns,
        ),
        ("cache_ssd_bytes", "gauge", "Bytes the model's store takes on SSD", ssd.store_bytes),
        (
            "cache_ssd_pending",
            "gauge",
            "Pages and checkpoints that wait to be written to SSD",
            ssd.pending,
        ),
        (
            "cache_ssd_checkpoints_skipped_total",
            "counter",
            "Checkpoints of recurrent state that were due on SSD and were given up unwritten",
            ssd.checkpoints_skipped,
        ),
        (
            "requests_finished_total",
            "counter",
            "Requests that ran to their end",
            stats.requests_finished,
        ),
        (
            "requests_cancelled_total",
            "counter",
            "Requests that were cancelled",
            stats.requests_cancelled,
        ),
        (
            "requests_refused_total",
            "counter",
            "Requests refused for lack of room",
            stats.requests_refused,
        ),
        (
            "requests_outranked_total",
            "counter",
            "Times a running request gave way to a request of a better priority",
            stats.requests_outranked,
        ),
        (
            "requests_displaced_total",
            "counter",
            "Requests in a full line refused because a request of a better priority came",
            stats.requests_displaced,
        ),
        ("steps_total", "counter", "Steps of the engine", stats.steps),
        ("prompt_tokens_total", "counter", "Prompt tokens computed", stats.prompt_tokens),
        ("generated_tokens_total", "counter", "Tokens generated", stats.generated_tokens),
        ("drafts_proposed_total", "counter", "Draft tokens put to the test", stats.drafts_proposed),
        ("drafts_accepted_total", "counter", "Draft tokens accepted", stats.drafts_accepted),
        (
            "prefix_hits_total",
            "counter",
            "Starts of a request from cells the prefix cache held",
            stats.prefix_hits,
        ),
        (
            "cached_cells_total",
            "counter",
            "Cells taken from the prefix cache instead of computed",
            stats.cached_cells,
        ),
        (
            "replayed_rows_total",
            "counter",
            "Rows run again to rebuild recurrent state from cached cells",
            stats.replayed_rows,
        ),
        (
            "checkpoints_saved_total",
            "counter",
            "Checkpoints of recurrent state taken",
            stats.checkpoints_saved,
        ),
        (
            "checkpoints_restored_total",
            "counter",
            "Checkpoints restored by the start of a request",
            stats.checkpoints_restored,
        ),
        (
            "checkpoints_dropped_total",
            "counter",
            "Checkpoints given up for the budget or with their page",
            stats.checkpoints_dropped,
        ),
        (
            "page_copies_total",
            "counter",
            "Pages copied before a sequence wrote into a page that others read",
            stats.page_copies,
        ),
        ("sequences", "gauge", "Sequences running", u64::from(stats.sequences)),
        ("pages_free", "gauge", "Free pages of sequence state", u64::from(stats.pages_free)),
        ("pages", "gauge", "Pages of sequence state", u64::from(stats.pages)),
        (
            "cached_pages",
            "gauge",
            "Pages and tails the prefix cache holds",
            u64::from(stats.cached_pages),
        ),
        (
            "checkpoint_ram_bytes",
            "gauge",
            "Page-locked RAM held for checkpoints of recurrent state",
            stats.checkpoint_ram,
        ),
        ("expert_picks_total", "counter", "Experts the routers picked", stats.experts.picks),
        (
            "expert_hits_total",
            "counter",
            "Picked experts that were in a VRAM slot",
            stats.experts.hits,
        ),
        (
            "expert_misses_ram_total",
            "counter",
            "Picked experts that were not in a slot and were in the RAM tier",
            stats.experts.from_ram,
        ),
        (
            "expert_misses_disk_total",
            "counter",
            "Picked experts that were not in a slot and only in the model file",
            stats.experts.from_disk,
        ),
        (
            "expert_misses_on_their_way_total",
            "counter",
            "Picked experts that were not in a slot and already on their way",
            stats.experts.on_their_way,
        ),
        (
            "expert_cpu_total",
            "counter",
            "Experts the CPU computed instead of the device",
            stats.experts.cpu_experts,
        ),
        (
            "expert_uploads_total",
            "counter",
            "Experts uploaded to a VRAM slot",
            stats.experts.uploads,
        ),
        (
            "expert_read_bytes_total",
            "counter",
            "Bytes of experts read from the model file",
            stats.experts.read_bytes,
        ),
        (
            "expert_reads_total",
            "counter",
            "Experts read from the model file, every kind of read",
            stats.experts.reads,
        ),
        (
            "expert_demand_reads_total",
            "counter",
            "Experts read from the model file into the RAM tier that a step waited for",
            stats.experts.demand_reads,
        ),
        (
            "expert_demand_read_microseconds_total",
            "counter",
            "Time the reads a step waited for took from their start to the bytes in RAM, summed",
            stats.experts.demand_read_us,
        ),
        (
            "expert_wait_microseconds_total",
            "counter",
            "Time layers waited for the experts they missed, summed over layers",
            stats.experts.wait_us,
        ),
        (
            "expert_wait_cpu_microseconds_total",
            "counter",
            "Wait of decode steps whose layers waited last for the CPU's sums",
            stats.experts.wait_last_us[0],
        ),
        (
            "expert_wait_ram_microseconds_total",
            "counter",
            "Wait of decode steps whose layers waited last for an upload from RAM",
            stats.experts.wait_last_us[1],
        ),
        (
            "expert_wait_file_microseconds_total",
            "counter",
            "Wait of decode steps whose layers waited last for a read of the file",
            stats.experts.wait_last_us[2],
        ),
        (
            "expert_wait_on_its_way_microseconds_total",
            "counter",
            "Wait of decode steps whose layers waited last for an expert on its way",
            stats.experts.wait_last_us[3],
        ),
        (
            "expert_predicted_total",
            "counter",
            "Experts the routers of later layers, applied early, named for a layer",
            stats.experts.predicted,
        ),
        (
            "expert_predicted_picked_total",
            "counter",
            "Predicted experts the layer's own router then picked",
            stats.experts.predicted_picked,
        ),
        (
            "expert_ahead_reads_total",
            "counter",
            "Predicted experts in no tier read from the model file ahead of their router",
            stats.experts.ahead_reads,
        ),
        (
            "expert_ahead_read_bytes_total",
            "counter",
            "Bytes of experts read ahead of their router",
            stats.experts.ahead_read_bytes,
        ),
        (
            "expert_ahead_used_total",
            "counter",
            "Experts read ahead that their router then picked",
            stats.experts.ahead_used,
        ),
        (
            "expert_ahead_wasted_total",
            "counter",
            "Experts read ahead that their router did not pick",
            stats.experts.ahead_wasted,
        ),
        (
            "expert_ram_tier_bytes",
            "gauge",
            "Page-locked RAM the tier of experts holds now",
            stats.experts.ram_tier_bytes,
        ),
        (
            "expert_transit_admitted_total",
            "counter",
            "Experts a wide step read from the model file that the RAM tier kept",
            stats.experts.transit_admitted,
        ),
        (
            "expert_transit_adopted_total",
            "counter",
            "Experts that took a VRAM slot from their transit slot",
            stats.experts.transit_adopted,
        ),
        (
            "expert_transit_guess_uploads_total",
            "counter",
            "Guesses of wide steps uploaded from the RAM tier before their layer's record",
            stats.experts.transit_guess_uploads,
        ),
        (
            "expert_transit_guess_uploads_used_total",
            "counter",
            "Guesses of wide steps uploaded before their record that the record named",
            stats.experts.transit_guess_uploads_used,
        ),
        ("expert_vram_experts", "gauge", "Experts in VRAM slots now", stats.experts.vram_experts),
        (
            "expert_ram_experts",
            "gauge",
            "Experts in the RAM tier now, copies of those in VRAM among them",
            stats.experts.ram_experts,
        ),
    ];
    let mut text = String::new();
    for (name, kind, help, value) in rows {
        text.push_str(&format!(
            "# HELP ares_{name} {help}\n# TYPE ares_{name} {kind}\nares_{name} {value}\n"
        ));
    }
    text
}
