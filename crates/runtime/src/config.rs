//! How the conductor is set up.

use std::time::Duration;

use ares_sched::LookupConfig;

/// Which drafters propose tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Drafters {
    /// The lookup in earlier text and the model's own draft block: for every
    /// sequence and step, the one that promises more tokens per millisecond.
    #[default]
    Both,
    /// Only the lookup in earlier text, which costs the devices nothing.
    Lookup,
    /// Only the model's own draft block, for a model that has one.
    Own,
}

/// The settings of the conductor and its bookkeeping.
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Pages in the pool and cells per page: together the most context all
    /// running sequences can hold. Pages are claimed step by step; a request
    /// starts when its prompt and the output it is expected to produce fit,
    /// and waits in line until then.
    pub pages: u32,
    pub cells_per_page: u32,
    /// The most sequences that run at once: the devices keep a recurrent
    /// state, a page table and a row of logits for each. Requests beyond
    /// that wait in line; the line has no limit.
    pub max_slots: u32,
    /// The most output tokens a request is counted with when it is decided
    /// whether another fits next to it. A request that names no limit may
    /// write until the context is full, and hardly any does.
    pub output_clip: u32,
    /// The share of that allowance a running request is expected to use,
    /// before any request has finished. The accounting corrects it from
    /// what finished requests really used.
    pub output_ratio: f64,
    /// The most bytes all checkpoints of recurrent state may take together, on
    /// all devices. A checkpoint is where a prompt that starts from the prefix
    /// cache restores the state of the model's recurrent layers; without one
    /// it replays every cached row. 0 keeps none. Only a model whose
    /// [`crate::Model::prefix_cache`] says it has recurrent state uses it. On
    /// devices the places this holds are locked in RAM as they are used, within
    /// [`crate::GpuConfig::ram_limit`], which [`crate::GpuConfig::new`] sets to
    /// what the executor needs and this budget together.
    pub checkpoint_budget: u64,
    /// The least distance in cells between checkpoints inside a prompt, and
    /// while generating (`ares_state::CheckpointPolicy`).
    pub checkpoint_prompt_interval: u32,
    pub checkpoint_generation_interval: u32,
    /// The most rows of a step; 0 for the model's own maximum.
    pub max_rows: u32,
    /// Whether drafts are proposed and verified.
    pub speculation: bool,
    /// Which drafters propose them.
    pub drafters: Drafters,
    /// The most draft tokens verified for one sequence in a step.
    pub max_draft: u32,
    /// The corpus that drafts are looked up in: its memory and the shortest
    /// match it drafts from.
    pub lookup: LookupConfig,
    /// The acceptance the model's own drafts are assumed to have for a new
    /// sequence.
    pub draft_prior: f64,
    /// How much slower generating sequences may get while a prompt is read,
    /// above 1 (2.0: they get half of the time, the prompt the other half):
    /// the one setting of how prompts and generation take turns
    /// ([`ares_sched::Share`]). A prompt that is read faster is read at the
    /// cost of the sequences that generate.
    pub prefill_slowdown: f64,
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
            output_clip: 4096,
            output_ratio: 0.7,
            checkpoint_budget: 0,
            checkpoint_prompt_interval: 32_768,
            checkpoint_generation_interval: 2048,
            max_rows: 0,
            speculation: true,
            drafters: Drafters::Both,
            max_draft: 8,
            lookup: LookupConfig::default(),
            draft_prior: 0.6,
            prefill_slowdown: 2.0,
            starvation_steps: 8,
            conductor_spin: Duration::from_millis(100),
            stats_window: crate::timing::DEFAULT_WINDOW,
            warm_up: true,
        }
    }
}
