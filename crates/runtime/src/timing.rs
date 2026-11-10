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

use std::fmt;
use std::time::Duration;

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
    /// lookup of earlier text, 1 the model's own draft block; the last
    /// position counts the ones past it too.
    pub drafts_tested_at: [[u64; 8]; 2],
    pub drafts_accepted_at: [[u64; 8]; 2],
    /// Steps in which the model's own draft block drafted, the links of its
    /// chains (the longest chain of each step) and their device time in
    /// microseconds.
    pub own_draft_steps: u64,
    pub own_draft_links: u64,
    pub own_draft_us: u64,
    /// Steps that carried prompt rows and generating rows together.
    pub mixed_steps: u64,
    /// Tokens delivered to requests (accepted picks).
    pub tokens: u64,
    pub requests_finished: u64,
    pub requests_cancelled: u64,
    /// Requests refused for lack of pages or slots.
    pub requests_refused: u64,
    /// How the conductor's waits for the devices ended.
    pub conductor_wakes: crate::sync::WakeCounts,
    /// How the device threads' waits for work ended.
    pub device_wakes: crate::sync::WakeCounts,
}

/// The conductor's measurements: owned by the conductor thread, copied out on
/// request.
#[derive(Clone, Debug)]
pub struct Stats {
    pub(crate) phases: Vec<Series>,
    pub(crate) devices: Vec<Vec<Series>>,
    pub counters: Counters,
}

impl Stats {
    pub fn new(devices: usize, window: usize) -> Self {
        Self {
            phases: Phase::ALL.iter().map(|_| Series::new(window)).collect(),
            devices: (0..devices)
                .map(|_| DevicePhase::ALL.iter().map(|_| Series::new(window)).collect())
                .collect(),
            counters: Counters::default(),
        }
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
    }
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
    /// Pages free and in all, and sequences running, when the copy was made.
    pub pages_free: u32,
    pub pages_capacity: u32,
    pub active: u32,
}

impl Snapshot {
    pub(crate) fn new(stats: &Stats, pages_free: u32, pages_capacity: u32, active: u32) -> Self {
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
            active,
        }
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
            "pages {}/{} free  active {}  refused {}  conductor waits for devices: {} spun, {} parked  \
             device threads' waits for work: {} spun, {} parked",
            self.pages_free,
            self.pages_capacity,
            self.active,
            c.requests_refused,
            c.conductor_wakes.spun,
            c.conductor_wakes.parked,
            c.device_wakes.spun,
            c.device_wakes.parked
        )?;
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
        let snapshot = Snapshot::new(&stats, 0, 0, 0);
        assert!((snapshot.gpu_wait_share(0) - 0.1).abs() < 1e-9, "{}", snapshot.gpu_wait_share(0));
        stats.reset();
        assert_eq!(Snapshot::new(&stats, 0, 0, 0).gpu_wait_share(0), 0.0);
    }

    #[test]
    fn the_table_names_every_phase() {
        let stats = Stats::new(2, 8);
        let text = Snapshot::new(&stats, 3, 4, 1).to_string();
        for phase in Phase::ALL {
            assert!(text.contains(phase.name()), "{text}");
        }
        for phase in DevicePhase::ALL {
            assert!(text.contains(phase.name()), "{text}");
        }
        assert!(text.contains("device 1"), "{text}");
    }
}
