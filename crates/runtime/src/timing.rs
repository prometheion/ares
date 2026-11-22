//! Where the time of a step goes.
//!
//! The question this answers is not how fast the GPUs are but how long they
//! wait for the host, so every step is cut into the pieces the conductor and
//! the device threads spend it on, and each piece is kept as a running total
//! and as the last few thousand samples (for percentiles).
//!
//! Recording a sample never allocates: a [`Series`] is a ring of fixed size,
//! made once at start. Percentiles are computed from a copy of the ring when
//! somebody asks for a [`Snapshot`], never on the conductor's path.
//!
//! All times are wall-clock time of the host, except those named `device`,
//! which come from timed events on the device's stream.

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use ares_api::RequestId;

/// The most recent samples kept per series for percentiles.
pub const DEFAULT_WINDOW: usize = 8192;

/// A series of durations: totals over everything recorded, and the last
/// `window` samples for percentiles.
#[derive(Clone, Debug)]
pub struct Series {
    /// Nanoseconds, saturating at about 4.3 s, which no phase of a step
    /// reaches.
    samples: Vec<u32>,
    next: usize,
    count: u64,
    total_ns: u64,
    max_ns: u64,
}

impl Series {
    pub fn new(window: usize) -> Self {
        Self { samples: vec![0; window.max(1)], next: 0, count: 0, total_ns: 0, max_ns: 0 }
    }

    /// Records one sample of `ns` nanoseconds.
    pub fn push(&mut self, ns: u64) {
        self.samples[self.next] = u32::try_from(ns).unwrap_or(u32::MAX);
        self.next += 1;
        if self.next == self.samples.len() {
            self.next = 0;
        }
        self.count += 1;
        self.total_ns += ns;
        self.max_ns = self.max_ns.max(ns);
    }

    pub fn clear(&mut self) {
        self.samples.fill(0);
        self.next = 0;
        self.count = 0;
        self.total_ns = 0;
        self.max_ns = 0;
    }

    /// Total of everything recorded since the last [`Series::clear`].
    pub fn total(&self) -> Duration {
        Duration::from_nanos(self.total_ns)
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Totals and percentiles. The percentiles are over the last
    /// `window` samples.
    pub fn summary(&self) -> Summary {
        let filled = (self.count as usize).min(self.samples.len());
        let mut recent: Vec<u32> = self.samples[..filled].to_vec();
        recent.sort_unstable();
        let at = |p: f64| {
            if recent.is_empty() {
                0.0
            } else {
                f64::from(recent[((recent.len() - 1) as f64 * p).round() as usize]) / 1e3
            }
        };
        Summary {
            count: self.count,
            total_ms: self.total_ns as f64 / 1e6,
            mean_us: if self.count == 0 {
                0.0
            } else {
                self.total_ns as f64 / self.count as f64 / 1e3
            },
            p50_us: at(0.5),
            p90_us: at(0.9),
            p99_us: at(0.99),
            max_us: self.max_ns as f64 / 1e3,
        }
    }
}

/// What a [`Series`] came to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub count: u64,
    pub total_ms: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

/// The pieces of a step that the conductor spends time on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Taking requests and cancellations off the channel.
    Intake,
    /// `ares_sched::plan`, including the draft decisions before it.
    Plan,
    /// Growing page tables and writing the step record.
    Write,
    /// Handing the step to the device threads.
    Dispatch,
    /// From the last device thread finishing a step to the conductor handing
    /// over the next one: the time every device sat idle for the host, whatever
    /// the placement. Not measured for a step that followed an idle engine.
    Turnaround,
    /// Waiting for the devices' picks (the conductor has nothing else to do).
    Wait,
    /// Accepting or rejecting drafts, cutting rejected rows, updating
    /// drafters and statistics.
    Commit,
    /// Delivering events to sinks.
    Events,
    /// The whole turn of the loop, from the first command to the last event.
    Step,
}

impl Phase {
    pub const ALL: [Phase; 9] = [
        Phase::Intake,
        Phase::Plan,
        Phase::Write,
        Phase::Dispatch,
        Phase::Turnaround,
        Phase::Wait,
        Phase::Commit,
        Phase::Events,
        Phase::Step,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Phase::Intake => "intake",
            Phase::Plan => "plan",
            Phase::Write => "write record",
            Phase::Dispatch => "dispatch",
            Phase::Turnaround => "devices idle (turnaround)",
            Phase::Wait => "wait devices",
            Phase::Commit => "accept/commit",
            Phase::Events => "events",
            Phase::Step => "step (total)",
        }
    }
}

/// The pieces of a step that one device thread spends time on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePhase {
    /// From the conductor handing over the step to the device thread starting
    /// on it: the wake-up latency.
    Wake,
    /// Host time to enqueue the upload of the step record (and page tables).
    Upload,
    /// Host time to enqueue the model's kernels for all rows and the
    /// read-back of the logits.
    Enqueue,
    /// Device time from the start of the step on the stream to its end, from
    /// timed events. Includes any time the stream waited for the host to
    /// enqueue more.
    Device,
    /// Host time from the end of the enqueue until the device thread saw the
    /// step finished.
    Readback,
    /// Host time to take the arg-max of the logits that came back.
    Pick,
    /// Device idle time between the end of the previous step and the start of
    /// this one, from timed events: the time the device waited for the host.
    Gap,
    /// From the device thread finishing to the conductor noticing.
    Notice,
}

impl DevicePhase {
    pub const ALL: [DevicePhase; 8] = [
        DevicePhase::Wake,
        DevicePhase::Upload,
        DevicePhase::Enqueue,
        DevicePhase::Device,
        DevicePhase::Readback,
        DevicePhase::Pick,
        DevicePhase::Gap,
        DevicePhase::Notice,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DevicePhase::Wake => "wake-up latency",
            DevicePhase::Upload => "upload enqueue",
            DevicePhase::Enqueue => "kernel enqueue",
            DevicePhase::Device => "device time",
            DevicePhase::Readback => "wait for device",
            DevicePhase::Pick => "sampler",
            DevicePhase::Gap => "device idle gap",
            DevicePhase::Notice => "conductor notice",
        }
    }
}

/// What one device thread measured in one step, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceTimes {
    pub wake_ns: u64,
    pub upload_ns: u64,
    pub enqueue_ns: u64,
    pub device_ns: u64,
    pub readback_ns: u64,
    pub pick_ns: u64,
    /// `None` for the first step after the engine was idle, which waited for a
    /// request and not for the host.
    pub gap_ns: Option<u64>,
    /// When the device thread finished, on the engine's clock.
    pub finished_at_ns: u64,
}

/// The positions of a draft that [`Counters`] tells apart; later ones count as
/// the last.
pub const DRAFT_POSITIONS: usize = 16;

/// Where drafts come from, for the statistics: the n-gram drafter, an MTP block, a
/// DFlash drafter.
pub const DRAFT_SOURCES: usize = 3;

/// Counts that go with the times.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub steps: u64,
    pub rows: u64,
    /// Rows of prompt tokens.
    pub prefill_rows: u64,
    /// Rows of newest tokens of generating sequences.
    pub window_rows: u64,
    /// Rows of drafts, on top of the newest token of the sequence.
    pub draft_rows: u64,
    pub drafts_accepted: u64,
    /// Draft rows that were cut again after the picks came back.
    pub drafts_rejected: u64,
    /// Drafts by where they came from and their position in the draft,
    /// `[source][position]`: how many were tested (every draft before them
    /// was accepted) and how many of those were accepted. Source 0 is the
    /// n-gram drafter, 1 an MTP block of the model, 2 a DFlash
    /// drafter ([`DrafterKind::source`](crate::DrafterKind::source)); the last
    /// position counts the ones past it too.
    pub drafts_tested_at: [[u64; DRAFT_POSITIONS]; DRAFT_SOURCES],
    pub drafts_accepted_at: [[u64; DRAFT_POSITIONS]; DRAFT_SOURCES],
    /// Windows by where their drafts came from and how many drafts they
    /// verified, `[source][drafts]`: the chain lengths the policy chose.
    pub draft_lengths: [[u64; DRAFT_POSITIONS + 1]; DRAFT_SOURCES],
    /// Steps in which the model's MTP block drafted, the links of its
    /// chains (the longest chain of each step) and their device time in
    /// microseconds.
    pub model_draft_steps: u64,
    pub model_draft_links: u64,
    pub model_draft_us: u64,
    /// The image tower: the units of its passes that ran, their device time
    /// in microseconds, and the images whose rows it computed.
    pub tower_units: u64,
    pub tower_us: u64,
    pub images_computed: u64,
    /// Links the windows asked the model's MTP block for, the links it
    /// did not make of them because it ended a chain at its bar, and the
    /// drafts it made that the round did not verify (the conductor took fewer
    /// by what the drafter said of them).
    pub model_links_asked: u64,
    pub model_links_cut: u64,
    pub model_drafts_unverified: u64,
    /// Windows a row of which the model left without what it would have
    /// waited for (an expert only the file held): their drafts from that row
    /// on were not accepted.
    pub windows_left: u64,
    /// The model's own drafts that were verified, by the probability the
    /// drafter gave them (the ranges of [`ares_sched::Confidence`]), and how
    /// many of them were accepted.
    pub model_tested_by_p: [u64; ares_sched::CONFIDENCE_BINS],
    pub model_accepted_by_p: [u64; ares_sched::CONFIDENCE_BINS],
    /// Steps that carried prompt rows and generating rows together.
    pub mixed_steps: u64,
    /// Tokens delivered to requests (accepted picks).
    pub tokens: u64,
    pub requests_finished: u64,
    pub requests_cancelled: u64,
    /// Requests that did not start in the first pass after they arrived.
    pub requests_waited: u64,
    /// Times a running request gave way to the others: it let go of its
    /// pages and its slot and came back later.
    pub requests_suspended: u64,
    /// Requests that ended because the pool had no page left for them and
    /// nobody else was left to give way.
    pub requests_exhausted: u64,
    /// Draft rows that were dropped from a step because their pages did not
    /// fit.
    pub draft_rows_cut: u64,
    /// Starts of a request that took cells from the prefix cache, and the
    /// cells they took (a request that gave way and came back counts again).
    pub prefix_hits: u64,
    pub cached_cells: u64,
    /// Rows run again to rebuild recurrent state from cached cells.
    pub replay_rows: u64,
    /// Checkpoints of recurrent state taken, restored by a start, and dropped
    /// again (for the budget, or with the page they were on).
    pub checkpoints_saved: u64,
    pub checkpoints_restored: u64,
    pub checkpoints_dropped: u64,
    /// Cached pages given up because a step needed the room, and of them the
    /// pages given up late: a claim the accounting had granted found the pool
    /// short, because it had counted pages as evictable that a sequence held.
    /// That is a miscount, which the pass makes good; it stays 0 when the
    /// accounting is right.
    pub cache_pages_evicted: u64,
    pub cache_pages_evicted_late: u64,
    /// Pages copied before a sequence wrote into a page that others read.
    pub page_copies: u64,
    /// Requests that were held back, once or more, because another sequence
    /// was computing the first page of their prompt that was not cached.
    pub requests_held: u64,
    /// What the levels of the prefix cache in RAM and on SSD have done since
    /// the engine started (a reset leaves them), and what they hold.
    pub ram: crate::RamCounters,
    pub ssd: crate::SsdCounters,
    /// How the conductor's waits for the devices ended.
    pub conductor_wakes: crate::sync::WakeCounts,
    /// How the device threads' waits for work ended.
    pub device_wakes: crate::sync::WakeCounts,
}

/// How many requests that ended a [`Snapshot`] reports.
pub const REPORTED_REQUESTS: usize = 256;

/// What one request that ended took from the prefix cache and what it
/// computed itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestReport {
    pub id: RequestId,
    pub prompt_tokens: u32,
    /// The prompt tokens whose cells came from the prefix cache when the
    /// request first started: what the request's usage calls cached.
    pub cached_tokens: u32,
    /// Rows run again to rebuild recurrent state from cached cells, over all
    /// the times the request started.
    pub replayed_rows: u32,
    /// Rows computed and written for the prompt, and for what a request that
    /// gave way had written, over all the times the request started.
    pub computed_rows: u32,
}

/// The conductor's measurements: owned by the conductor thread, copied out on
/// request.
#[derive(Clone, Debug)]
pub struct Stats {
    pub(crate) phases: Vec<Series>,
    pub(crate) devices: Vec<Vec<Series>>,
    pub counters: Counters,
    /// The last [`REPORTED_REQUESTS`] requests that ended, the oldest first.
    pub(crate) requests: VecDeque<RequestReport>,
}

impl Stats {
    pub fn new(devices: usize, window: usize) -> Self {
        Self {
            phases: Phase::ALL.iter().map(|_| Series::new(window)).collect(),
            devices: (0..devices)
                .map(|_| DevicePhase::ALL.iter().map(|_| Series::new(window)).collect())
                .collect(),
            counters: Counters::default(),
            requests: VecDeque::with_capacity(REPORTED_REQUESTS),
        }
    }

    /// Notes a request that ended.
    pub(crate) fn report(&mut self, report: RequestReport) {
        if self.requests.len() == REPORTED_REQUESTS {
            self.requests.pop_front();
        }
        self.requests.push_back(report);
    }

    pub(crate) fn record(&mut self, phase: Phase, ns: u64) {
        self.phases[phase as usize].push(ns);
    }

    pub(crate) fn record_device(&mut self, device: usize, times: &DeviceTimes, notice_ns: u64) {
        let series = &mut self.devices[device];
        series[DevicePhase::Wake as usize].push(times.wake_ns);
        series[DevicePhase::Upload as usize].push(times.upload_ns);
        series[DevicePhase::Enqueue as usize].push(times.enqueue_ns);
        series[DevicePhase::Device as usize].push(times.device_ns);
        series[DevicePhase::Readback as usize].push(times.readback_ns);
        series[DevicePhase::Pick as usize].push(times.pick_ns);
        if let Some(gap) = times.gap_ns {
            series[DevicePhase::Gap as usize].push(gap);
        }
        series[DevicePhase::Notice as usize].push(notice_ns);
    }

    /// Forgets everything recorded, to measure a run on its own.
    pub fn reset(&mut self) {
        self.phases.iter_mut().for_each(Series::clear);
        self.devices.iter_mut().flatten().for_each(Series::clear);
        self.counters = Counters::default();
        self.requests.clear();
    }
}

/// What the warm-up measured before the first request: the price of a step by
/// its rows and the time of the model's own draft chain by its links, with
/// no context to speak of, and what the measuring cost. The tables the
/// conductor decides drafts and pieces of prompt from start as these.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    /// Steps run, the warm-up's first three included.
    pub steps: u32,
    pub seconds: f64,
    /// `(rows, milliseconds)` of the steps measured.
    pub step_ms: Vec<(u32, f64)>,
    /// Per drafter of the model ([`Model::drafters`](crate::Model::drafters)),
    /// `(drafts, milliseconds)` of the steps that only draft measured: empty
    /// for a model without drafters.
    pub chain_ms: Vec<Vec<(u32, f64)>>,
}

/// A copy of the measurements with percentiles computed, printable as a
/// table.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub counters: Counters,
    /// The conductor's phases, in [`Phase::ALL`] order.
    pub phases: Vec<(Phase, Summary)>,
    /// Per device, its phases in [`DevicePhase::ALL`] order.
    pub devices: Vec<Vec<(DevicePhase, Summary)>>,
    /// Pages free and in all, the pages and tails the prefix index holds (they
    /// are not free; a request that needs room evicts them), sequences running
    /// and requests in line, when the copy was made.
    pub pages_free: u32,
    pub pages_capacity: u32,
    pub cached_pages: u32,
    /// Bytes of page-locked RAM the devices hold for checkpoints of recurrent
    /// state.
    pub checkpoint_ram: u64,
    pub active: u32,
    pub waiting: u32,
    /// The last requests that ended, the oldest first: at most
    /// [`REPORTED_REQUESTS`].
    pub requests: Vec<RequestReport>,
    /// What the warm-up measured; `None` for a runtime that ran none.
    pub calibration: Option<Calibration>,
    /// What a step costs by its rows and each of the model's drafters by its
    /// drafts now, `(count, milliseconds)`: the warm-up's tables as the steps
    /// that ran have corrected them.
    pub step_ms: Vec<(u32, f64)>,
    pub chain_ms: Vec<Vec<(u32, f64)>>,
    /// Why the level of the prefix cache on SSD is not in use although the
    /// engine was given a directory for it; `None` when it is in use or was
    /// not asked for.
    pub ssd_problem: Option<String>,
}

impl Snapshot {
    pub(crate) fn new(
        stats: &Stats,
        pages_free: u32,
        pages_capacity: u32,
        cached_pages: u32,
        checkpoint_ram: u64,
        active: u32,
        waiting: u32,
    ) -> Self {
        Self {
            counters: stats.counters,
            phases: Phase::ALL.iter().map(|p| (*p, stats.phases[*p as usize].summary())).collect(),
            devices: stats
                .devices
                .iter()
                .map(|series| {
                    DevicePhase::ALL.iter().map(|p| (*p, series[*p as usize].summary())).collect()
                })
                .collect(),
            pages_free,
            pages_capacity,
            cached_pages,
            checkpoint_ram,
            active,
            waiting,
            requests: stats.requests.iter().copied().collect(),
            calibration: None,
            step_ms: Vec::new(),
            chain_ms: Vec::new(),
            ssd_problem: None,
        }
    }

    /// The report of the request `id`, if it ended lately.
    pub fn request(&self, id: RequestId) -> Option<&RequestReport> {
        self.requests.iter().rev().find(|report| report.id == id)
    }

    /// The summary of a conductor phase.
    pub fn phase(&self, phase: Phase) -> &Summary {
        &self.phases[phase as usize].1
    }

    /// The summary of a device phase.
    pub fn device_phase(&self, device: usize, phase: DevicePhase) -> &Summary {
        &self.devices[device][phase as usize].1
    }

    /// The share of the time of its steps in which device `device` sat idle
    /// waiting for the host: gap / (gap + device time). Steps that followed an
    /// idle engine are not counted. 0 when nothing was measured.
    pub fn gpu_wait_share(&self, device: usize) -> f64 {
        let gap = self.device_phase(device, DevicePhase::Gap);
        let busy = self.device_phase(device, DevicePhase::Device);
        // Device time of the steps that have a gap, which is all but the first.
        let busy_ms = if busy.count == 0 {
            0.0
        } else {
            busy.total_ms / busy.count as f64 * gap.count as f64
        };
        if gap.total_ms + busy_ms <= 0.0 { 0.0 } else { gap.total_ms / (gap.total_ms + busy_ms) }
    }

    /// The share of the step cycle in which all devices were idle because the
    /// host had not handed them the next step: turnaround / (turnaround +
    /// the step itself). Unlike [`Snapshot::gpu_wait_share`] it is also right
    /// for a layer split, where a device's own idle time includes waiting for
    /// the devices before and after it.
    pub fn host_wait_share(&self) -> f64 {
        let turnaround = self.phase(Phase::Turnaround);
        let (dispatch, wait) = (self.phase(Phase::Dispatch), self.phase(Phase::Wait));
        if turnaround.count == 0 || wait.count == 0 {
            return 0.0;
        }
        let step_ms =
            (dispatch.total_ms + wait.total_ms) / wait.count as f64 * turnaround.count as f64;
        turnaround.total_ms / (turnaround.total_ms + step_ms)
    }

    /// The share of the conductor's loop time spent waiting for the devices.
    pub fn conductor_wait_share(&self) -> f64 {
        let step = self.phase(Phase::Step).total_ms;
        if step <= 0.0 { 0.0 } else { self.phase(Phase::Wait).total_ms / step }
    }
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.counters;
        writeln!(
            f,
            "steps {}  rows {} (prompt {}, newest {}, drafts {}; {} accepted, {} cut)  mixed steps {}  tokens {}",
            c.steps,
            c.rows,
            c.prefill_rows,
            c.window_rows,
            c.draft_rows,
            c.drafts_accepted,
            c.drafts_rejected,
            c.mixed_steps,
            c.tokens
        )?;
        writeln!(
            f,
            "pages {}/{} free  active {}  in line {} ({} waited, {} gave way, {} ended by a full pool, \
             {} draft rows cut)  conductor waits for devices: {} spun, {} parked  \
             device threads' waits for work: {} spun, {} parked",
            self.pages_free,
            self.pages_capacity,
            self.active,
            self.waiting,
            c.requests_waited,
            c.requests_suspended,
            c.requests_exhausted,
            c.draft_rows_cut,
            c.conductor_wakes.spun,
            c.conductor_wakes.parked,
            c.device_wakes.spun,
            c.device_wakes.parked
        )?;
        writeln!(
            f,
            "prefix cache: {} starts took {} cells, {} rows replayed, {} requests held back, \
             {} pages copied, {} pages evicted ({} late); checkpoints: {} saved, {} restored, \
             {} dropped",
            c.prefix_hits,
            c.cached_cells,
            c.replay_rows,
            c.requests_held,
            c.page_copies,
            c.cache_pages_evicted,
            c.cache_pages_evicted_late,
            c.checkpoints_saved,
            c.checkpoints_restored,
            c.checkpoints_dropped
        )?;
        if c.ram.enabled {
            let r = &c.ram;
            writeln!(
                f,
                "prefix cache in RAM: {} pages and {} checkpoints in {} bytes; {} pages came \
                 from the devices, {} went back, {} were dropped unwritten; {} transfers took \
                 {:.1} ms",
                r.pages,
                r.checkpoints,
                r.bytes,
                r.pages_in,
                r.pages_read,
                r.pages_dropped,
                r.transfers,
                r.transfer_ns as f64 / 1e6
            )?;
        }
        if let Some(why) = &self.ssd_problem {
            writeln!(f, "prefix cache on SSD: not in use ({why})")?;
        }
        if c.ssd.enabled {
            let s = &c.ssd;
            writeln!(
                f,
                "prefix cache on SSD: {} pages and {} checkpoints written ({} bytes, {} failed, \
                 {} checkpoints skipped, {} waiting), {} pages and {} checkpoints read ({} \
                 bytes) in {} runs ({} cut); the store holds {} pages and {} checkpoints in {} \
                 bytes; {} transfers took {:.1} ms",
                s.pages_written,
                s.states_written,
                s.bytes_written,
                s.write_errors,
                s.states_skipped,
                s.pending,
                s.pages_read,
                s.states_read,
                s.bytes_read,
                s.imports,
                s.imports_cut,
                s.store_pages,
                s.store_states,
                s.store_bytes,
                s.transfers,
                s.transfer_ns as f64 / 1e6
            )?;
        }
        writeln!(
            f,
            "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "conductor", "mean us", "p50", "p90", "p99", "max"
        )?;
        for (phase, s) in &self.phases {
            writeln!(
                f,
                "  {:<26} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                phase.name(),
                s.mean_us,
                s.p50_us,
                s.p90_us,
                s.p99_us,
                s.max_us
            )?;
        }
        for (device, phases) in self.devices.iter().enumerate() {
            writeln!(
                f,
                "{:<28} {:>9} {:>9} {:>9} {:>9} {:>9}",
                format!("device {device}"),
                "mean us",
                "p50",
                "p90",
                "p99",
                "max"
            )?;
            for (phase, s) in phases {
                writeln!(
                    f,
                    "  {:<26} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                    phase.name(),
                    s.mean_us,
                    s.p50_us,
                    s.p90_us,
                    s.p99_us,
                    s.max_us
                )?;
            }
            writeln!(
                f,
                "  device {device} was idle between steps for {:.2} % of its step cycle (in a layer split this \
                 includes waiting for the other devices; see the turnaround for the host's share)",
                100.0 * self.gpu_wait_share(device)
            )?;
        }
        writeln!(
            f,
            "all devices waited for the host in {:.2} % of the step cycle; the conductor waited for the \
             devices for {:.1} % of its loop time",
            100.0 * self.host_wait_share(),
            100.0 * self.conductor_wait_share()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_come_from_the_recent_window_and_totals_from_everything() {
        let mut series = Series::new(100);
        // 1000 samples of 1 ms, then 100 of 2 ms: the window holds only the latter.
        for _ in 0..1000 {
            series.push(1_000_000);
        }
        for _ in 0..100 {
            series.push(2_000_000);
        }
        let s = series.summary();
        assert_eq!(s.count, 1100);
        assert!((s.total_ms - 1200.0).abs() < 1e-6, "{s:?}");
        assert!((s.mean_us - 1090.9).abs() < 0.1, "{s:?}");
        assert_eq!(s.p50_us, 2000.0);
        assert_eq!(s.max_us, 2000.0);
    }

    #[test]
    fn a_ring_that_is_not_full_yet_ignores_its_empty_slots() {
        let mut series = Series::new(1000);
        for ns in [10_000, 20_000, 30_000, 40_000, 50_000] {
            series.push(ns);
        }
        let s = series.summary();
        assert_eq!((s.p50_us, s.p90_us, s.max_us), (30.0, 50.0, 50.0));
        assert_eq!(Series::new(4).summary(), Summary::default());
    }

    #[test]
    fn recording_does_not_grow_the_ring() {
        let mut series = Series::new(16);
        let capacity = series.samples.capacity();
        for i in 0..10_000 {
            series.push(i);
        }
        assert_eq!(series.samples.capacity(), capacity);
        assert_eq!(series.samples.len(), 16);
    }

    #[test]
    fn the_wait_share_is_the_gap_over_gap_plus_device_time() {
        let mut stats = Stats::new(1, 64);
        // First step: no gap (the engine was idle). Then 3 steps of 9 ms with
        // a 1 ms gap each: the device waited for 10 % of the time.
        let first = DeviceTimes { device_ns: 9_000_000, gap_ns: None, ..Default::default() };
        stats.record_device(0, &first, 0);
        for _ in 0..3 {
            let t =
                DeviceTimes { device_ns: 9_000_000, gap_ns: Some(1_000_000), ..Default::default() };
            stats.record_device(0, &t, 0);
        }
        let snapshot = Snapshot::new(&stats, 0, 0, 0, 0, 0, 0);
        assert!((snapshot.gpu_wait_share(0) - 0.1).abs() < 1e-9, "{}", snapshot.gpu_wait_share(0));
        stats.reset();
        assert_eq!(Snapshot::new(&stats, 0, 0, 0, 0, 0, 0).gpu_wait_share(0), 0.0);
    }

    #[test]
    fn the_table_names_every_phase() {
        let stats = Stats::new(2, 8);
        let text = Snapshot::new(&stats, 3, 4, 0, 0, 1, 0).to_string();
        for phase in Phase::ALL {
            assert!(text.contains(phase.name()), "{text}");
        }
        for phase in DevicePhase::ALL {
            assert!(text.contains(phase.name()), "{text}");
        }
        assert!(text.contains("device 1"), "{text}");
    }
}
