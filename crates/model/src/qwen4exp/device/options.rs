//! The choices a model is opened with: what the server sets for the model it
//! serves, and what the tests and tools set for the models they open.

use std::time::Duration;

use ares_runtime::expert_store::Serve;

use super::part::Options;

/// How a model is spread over two devices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spread {
    /// Both devices on every layer when they can (two devices, each able to
    /// write the other's memory, mixers the fused read takes) and no layer
    /// bounds are given; otherwise by layers.
    Auto,
    /// A range of layers per device: one sequence uses one device at a time.
    Layers,
    /// Both devices on every layer. What is `None` is taken from a
    /// measurement at load: the mixer is the device whose read of a stream
    /// mixer is faster. With neither the share nor the shared expert given, a
    /// trial at load times the worker computing a share of the recurrent
    /// heads and the shared expert (the mixer's share the worker's read time
    /// over the sum of both) against the worker holding experts only (share
    /// 1, the shared expert on the mixer), and keeps the faster (`roles.rs`).
    /// With one of them given, the share is that of the read times and the
    /// shared expert the worker's, unless said otherwise.
    Tensor {
        /// The device that holds the streams and runs the mixers.
        mixer: Option<usize>,
        /// Its share of the key heads of the recurrent layers.
        core_share: Option<f64>,
        /// Whether it computes the shared expert.
        shared_on_mixer: Option<bool>,
    },
}

/// The choices of a model on its devices.
#[derive(Clone, Debug)]
pub struct ModelOptions {
    /// The choices of every part.
    pub part: Options,
    /// Bytes of every device that stay free.
    pub headroom: u64,
    /// How two devices share the model.
    pub spread: Spread,
    /// The first layer of every device and the number of layers, to place
    /// the layers by hand; `None` lets [`Placement::plan`] decide.
    ///
    /// [`Placement::plan`]: super::Placement::plan
    pub bounds: Option<Vec<usize>>,
    /// Free slots the store keeps per layer in VRAM; `None` for its default.
    pub reserve: Option<usize>,
    /// The most bytes of expert slots on a device, whatever it has left;
    /// `None` for all it has left.
    pub slot_bytes: Option<u64>,
    /// How an expert that is in no slot is served: the engine lets the
    /// store's measured costs decide between an upload and the CPU.
    pub serve: Serve,
    /// Workers that compute experts on the CPU; `None` takes the cores the
    /// system reports, less one for each thread that drives the step.
    pub cpu_workers: Option<usize>,
    /// Bytes of the cache of n-gram table rows.
    pub row_cache: usize,
    /// How long a step may stand still before it is given up with an error
    /// that says what it waited for: a device that waits for an expert
    /// nobody brings would otherwise wait for ever.
    pub stall: Duration,
    /// Transit slots of every device for the experts a wide step misses, to
    /// measure what they are worth; `None` lets the plan of the store decide
    /// (`plan_transit`), 0 brings the misses into the tier's slots.
    pub transit: Option<u32>,
    /// Whether wide steps start the reads of the layers ahead on a guess: what the wide step
    /// before missed there and no tier holds. Off unless a measurement asks for it: that
    /// guess is mostly wrong (what a step read and the heat keeps is in RAM afterwards, so
    /// what is left to guess is what one row used), and with it a prompt read 1.6 to 2.8 GiB
    /// more from the file and 5 to 30 % slower (measured 2026-11-12, see
    /// `docs/dev/experiments/flash-next-prompts.md`).
    pub read_ahead: bool,
    /// The bytes of a piece of the RAM tier under the runtime: what the tier
    /// locks or gives back at a time. `None` for the store's own size; tests
    /// of a small model name a smaller one.
    pub ram_piece: Option<usize>,
    /// The most rows of a prompt read layer by layer through the runtime: a step of one
    /// prompt wider than the parts' own scratch ([`Options::own_rows`]) runs as a pass, every
    /// row through a layer before the next layer, each expert brought once for the step, its
    /// rows kept on VRAM the expert slots lend (as many as they can lend for: the model's
    /// `max_rows` says). 0: none; a prompt read alone takes lent steps of up to
    /// [`Options::max_rows`] rows.
    pub pass_rows: u32,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            part: Options::default(),
            headroom: 256 << 20,
            spread: Spread::Auto,
            bounds: None,
            reserve: None,
            slot_bytes: None,
            serve: Serve::Measured,
            cpu_workers: None,
            row_cache: 16 << 20,
            stall: Duration::from_secs(30),
            transit: None,
            read_ahead: false,
            ram_piece: None,
            pass_rows: 0,
        }
    }
}
