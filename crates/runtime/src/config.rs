//! How the conductor is set up.

use std::time::Duration;

use ares_sched::NgramConfig;
use ares_state::CheckpointPolicy;

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
    /// The most requests that may wait in line while every slot is taken:
    /// one more is refused as [`ares_api::EngineError::Overloaded`]. `None`:
    /// the line has no limit, and a request waits until it runs.
    pub queue: Option<u32>,
    /// The most output tokens a request is counted with when it is decided
    /// whether another fits next to it. A request that names no limit may
    /// write until the context is full, and hardly any does.
    pub output_clip: u32,
    /// The share of that allowance a running request is expected to use,
    /// before any request has finished. The accounting corrects it from
    /// what finished requests really used.
    pub output_ratio: f64,
    /// The most bytes the prefix cache keeps in RAM: the checkpoints of
    /// recurrent state, on all devices together, and the pages that left the
    /// devices. A checkpoint is where a prompt that starts from the prefix
    /// cache restores the state of the model's recurrent layers; without one
    /// it replays every cached row. Only a model whose
    /// [`crate::Model::prefix_cache`] says it has recurrent state takes
    /// checkpoints, and only a model that is [`crate::Model::durable`] keeps
    /// pages here. 0 keeps none. On devices this memory is locked as it is
    /// used, within [`crate::GpuConfig::ram_limit`], which
    /// [`crate::GpuConfig::new`] sets to what the executor needs and this
    /// together; the checkpoints and the pages share it, and it follows the
    /// machine (see "Host memory" in `docs/dev/architecture/host-memory.md`).
    pub cache_ram: u64,
    /// The most bytes the rows of images take that the image tower has
    /// computed (ordinary memory, by image, in the conductor's image store): a
    /// prompt that sends an image again, and a replay of cached cells of an
    /// image, take them from there instead of running the tower again. The
    /// least recently used image goes first.
    pub image_rows: u64,
    /// The least distance in cells between checkpoints inside a prompt, and
    /// while generating, and from the last checkpoint to a message boundary
    /// that gets one (`ares_state::CheckpointPolicy`).
    pub checkpoint_prompt_interval: u32,
    pub checkpoint_generation_interval: u32,
    pub checkpoint_boundary_interval: u32,
    /// The level of the prefix cache that is on SSD: where it lies and how
    /// large it may get. `None` keeps the cache in the devices and in RAM
    /// alone. Only a model that says it is [`crate::Model::durable`] uses it.
    pub ssd: Option<crate::SsdConfig>,
    /// The most rows of a step; 0 for the model's own maximum.
    pub max_rows: u32,
    /// The most rows of one sequence's window in a step: its newest token and
    /// up to `window_rows - 1` drafts. How many of them a step really has
    /// is chosen per step from the price of the step and the acceptance
    /// measured so far; this is the room, which the model reserves memory
    /// for (a row of a window keeps its inputs, its logits and a link of the
    /// MTP block's chain). A model with no drafter of its own
    /// ([`crate::Model::drafters`]) does not draft at all, whatever this is.
    pub window_rows: u32,
    /// The corpus that drafts are looked up in: its memory and the shortest
    /// match it drafts from.
    pub ngram: NgramConfig,
    /// The chance that a draft is accepted before the engine has seen any of
    /// that drafter at that position: where every estimate starts from.
    pub draft_prior: f64,
    /// The rate in tokens per second that every generating sequence keeps
    /// while prompts are read: the one setting of how prompts and generation
    /// take turns ([`ares_sched::Share`]). The rounds of generation get the
    /// time that keeps it, measured from what the last rounds gave, and the
    /// prompts the rest; at most half of the time goes to generation (what a
    /// floor the rounds cannot reach leaves). 0: generation keeps only the
    /// token every sequence gets in each step of prompt. A prompt that is read
    /// faster is read at the cost of the sequences that generate.
    pub generation_floor: f64,
    /// The longest a step for prompts takes, in milliseconds, while another
    /// request waits (a prompt next to it, a sequence that generates): the
    /// piece of prompt is halved until its measured price fits. A prompt read
    /// alone takes the widest step whatever it costs; a request that comes
    /// then waits for that step, and then for pieces of at most this long.
    pub piece_ms: f64,
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
    /// The longest one step may take before the engine is taken for stuck
    /// (SGLang's `--watchdog-timeout`, 300 s there): a device that never
    /// finishes its step leaves every request waiting for ever. The engine
    /// then fails as when a device reports an error: its requests end with
    /// the reason, it accepts no more, and the threads that hang are left
    /// behind when it is dropped. `None`: no watch. Longer than any step that
    /// works: the widest step of a prompt of a mixture of experts whose
    /// experts come from the SSD takes tens of seconds.
    pub watchdog: Option<Duration>,
    /// Run steps before taking requests: the first ones pay for the lazy
    /// loading of kernels and the first use of each path, so it is not in the
    /// first request's time; and then steps of every row count a window can
    /// have and of a few above it, which measure the price of a step on the
    /// devices (and, for a model with an MTP block, of every length of its
    /// chain), where drafts and the pieces of a prompt are decided from. The
    /// measurements are kept up to date by every step that runs.
    pub warm_up: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            pages: 4096,
            cells_per_page: 16,
            max_slots: 256,
            queue: None,
            output_clip: 4096,
            output_ratio: 0.7,
            cache_ram: 0,
            image_rows: 256 << 20,
            checkpoint_prompt_interval: CheckpointPolicy::PROMPT_INTERVAL,
            checkpoint_generation_interval: CheckpointPolicy::GENERATION_INTERVAL,
            checkpoint_boundary_interval: CheckpointPolicy::BOUNDARY_INTERVAL,
            ssd: None,
            max_rows: 0,
            window_rows: 8,
            ngram: NgramConfig::default(),
            draft_prior: 0.6,
            generation_floor: 20.0,
            piece_ms: 500.0,
            starvation_steps: 8,
            conductor_spin: Duration::from_millis(100),
            stats_window: crate::timing::DEFAULT_WINDOW,
            watchdog: Some(Duration::from_secs(600)),
            warm_up: true,
        }
    }
}
