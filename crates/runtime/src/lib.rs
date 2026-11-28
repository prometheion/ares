//! The step loop of the engine, on real threads, with a synthetic model.
//!
//! Bad multithreading is what makes an inference engine slow in ways no
//! kernel fixes: threads that contend for shared state, hand-overs that go
//! through the kernel's scheduler, a host that waits between layers. This crate
//! is the engine's answer, built and measured before any real architecture
//! exists.
//!
//! # Threads
//!
//! ```text
//!  HTTP / callers          conductor                     device threads
//!  --------------          ---------                     --------------
//!  submit, cancel ──────►  owns all bookkeeping          one per device:
//!  (never block)  channel  (sequences, page tables,      stream, model part,
//!                          plan, drafters, statistics,   kernels, read-back
//!  EventSink ◄──────────── request state): no locks
//!  (never blocks)          │
//!                          │ plan ─► write record ─► dispatch ───►│ upload, enqueue
//!                          │                                      │ all kernels,
//!                          │ commit ◄─ picks ◄─────────────── done│ read back once
//!                          ▼
//!                          events
//! ```
//!
//! What crosses a thread boundary, and how often:
//!
//! | From → to | What | Per step |
//! | --- | --- | --- |
//! | callers → conductor | `Submit`, `Cancel`, `Stats`, `Shutdown` in a channel | per request, not per step |
//! | conductor → each device thread | a small message (step number, record length) and a ring of a bell the thread spins or parks on | 1 |
//! | conductor → device threads | the step record, in page-locked memory both can see | 1 write, 1 upload per device |
//! | each device thread → conductor | picks and timings in the thread's result slot, one ring when the last is done | 1 |
//! | device thread ↔ device thread | a rendezvous per sum (tensor split), a queue of parcels (layer split) | 2 per layer / 1 per row group |
//! | conductor → sinks | one `Event::Text` per sequence | 1 per sequence |
//!
//! The host waits on a device in exactly one place, once per step: the device
//! thread waits for the end of its step to read the picks. Between devices the
//! order is kept on the devices.
//!
//! # Contents
//!
//! - [`Runtime`] is the [`Engine`](ares_api::Engine).
//! - [`Model`] is what an architecture implements; [`SyntheticModel`] is built
//!   from the kernels that exist.
//! - [`GpuExecutor`] runs steps on devices; [`testing::HostExecutor`] runs
//!   them without, for tests and for measuring the conductor alone. Given the
//!   pool's shape it also plays a device that keeps state, to check what the
//!   conductor does with the prefix cache.
//! - The prefix cache (cached pages, tails, checkpoints of recurrent state,
//!   replayed rows) is the conductor's, for a model whose device side carries
//!   out the step record's copies ([`Model::prefix_cache`]).
//! - [`Snapshot`] is the timing: where every step went.

mod cache;
mod conductor;
mod config;
mod engine;
mod error;
mod exec;
pub mod expert_store;
mod gpu;
mod guard;
mod image;
mod links;
mod model;
mod ram;
mod record;
mod sampling;
mod structured;
mod sync;
mod synthetic;
pub mod testing;
mod text;
mod timing;
pub mod vram;
mod watch;

pub use ares_state::Boundary;
pub use cache::{RamCounters, SsdConfig, SsdCounters, TransferCounters};
pub use conductor::{HOLD_STEPS_WHILE, SequenceInfo, token_hash};
pub use config::RuntimeConfig;
pub use engine::Runtime;
pub use error::RuntimeError;
pub use exec::{
    Clock, ExecError, Executor, HostBytes, Pick, StepMeta, StepOutput, Transfer, TransferKind,
    TransitPart,
};
pub use gpu::{GpuConfig, GpuExecutor, required_ram};
pub use image::{ImageInput, ImageName, ImageSpan, Spans, cell_name};
pub use links::{Links, MAX_DEVICES};
pub use model::{
    ContextSpec, DeviceContext, DeviceModel, DeviceTransfer, Drafter, DrafterKind, Durable,
    EncodedPrompt, FarPages, LinkSpec, Logits, Model, ModelDrafts, ModelError, PrefixCache, Split,
    StepContext, TowerJob, TransferContext, Vision, share_of,
};
pub use ram::{
    CacheBuffer, ContextLock, ContextRegion, ContextShort, LockedChunk, Look, Machine, Meminfo,
    MemorySource, PinnedRam, PlaceChunks, Rule, Verdict,
};
pub use record::{
    Commit, DraftChances, FREED_PER_RECORD, ImageRun, InnerSave, PageCopy, RecordFull, RecordView,
    RecordWriter, Row, StateCopy, StateCopyKind, TableUpdate, draft_chance,
};
pub use sampling::TokenFilter;
pub use sync::{WakeCounts, Woke};
pub use synthetic::{SyntheticConfig, SyntheticModel};
pub use text::{printable, unprintable};
pub use timing::{
    Calibration, Counters, DEFAULT_WINDOW, DRAFT_POSITIONS, DRAFT_SOURCES, DevicePhase,
    DeviceTimes, Phase, RequestReport, Snapshot, Stats, Summary,
};
pub use watch::Thinking;
