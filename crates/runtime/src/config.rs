//! How the conductor is set up.

use std::time::Duration;

/// The settings of the conductor and its bookkeeping.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Pages in the pool and cells per page: together the most context all
    /// sequences can hold. Requests are admitted against this, for their
    /// longest possible length, so no running request can run out of pages.
    pub pages: u32,
    pub cells_per_page: u32,
    /// The most sequences at once.
    pub max_slots: u32,
    /// The most rows of a step; 0 for the model's own maximum.
    pub max_rows: u32,
    /// Whether n-gram drafts are proposed and verified.
    pub speculation: bool,
    /// The most draft tokens verified for one sequence in a step.
    pub max_draft: u32,
    /// The n-gram length drafts are matched on.
    pub ngram: usize,
    /// The acceptance a new sequence's drafter is assumed to have.
    pub draft_prior: f64,
    /// How much slower generating sequences may get while a prompt is read
    /// (2.0: half speed), which bounds the prompt rows of a step.
    pub prefill_slowdown: f64,
    /// Prompt rows next to generation before any step has been measured.
    pub prefill_fallback_rows: u32,
    /// Steps a prompt may wait before it goes ahead of shorter ones.
    pub starvation_steps: u32,
    /// How long the conductor spins for the devices to finish a step before
    /// it parks. A step takes milliseconds, and a conductor that parks is
    /// woken through the kernel, which takes 20 to 50 microseconds to get it
    /// back on a core while every device sits idle (measured: 40 us of a
    /// 5 ms step, against 1 to 2 us when it spins). So by default it spins
    /// through any step a decode step can be, and parks only for a step so
    /// long that the wake-up costs nothing worth a core.
    pub conductor_spin: Duration,
    /// Samples kept per timing series for percentiles.
    pub stats_window: usize,
    /// Run a few steps before taking requests, so lazy loading of kernels
    /// and the first use of each path are not in the first request's time.
    pub warm_up: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pages: 4096,
            cells_per_page: 16,
            max_slots: 256,
            max_rows: 0,
            speculation: true,
            max_draft: 8,
            ngram: 3,
            draft_prior: 0.6,
            prefill_slowdown: 2.0,
            prefill_fallback_rows: 32,
            starvation_steps: 8,
            conductor_spin: Duration::from_millis(100),
            stats_window: crate::timing::DEFAULT_WINDOW,
            warm_up: true,
        }
    }
}
