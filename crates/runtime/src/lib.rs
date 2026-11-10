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
//!   them without, for tests and for measuring the conductor alone.
//! - [`Snapshot`] is the timing: where every step went.

mod conductor;
mod config;
mod engine;
mod error;
mod exec;
pub mod expert_store;
mod gpu;
mod links;
mod model;
mod record;
mod sync;
mod synthetic;
pub mod testing;
mod text;
mod timing;

pub use conductor::SequenceInfo;
pub use config::RuntimeConfig;
pub use engine::Runtime;
pub use error::RuntimeError;
pub use exec::{Clock, ExecError, Executor, Pick, StepMeta, StepOutput};
pub use gpu::{GpuConfig, GpuExecutor};
pub use links::{Links, MAX_DEVICES};
pub use model::{
    ContextSpec, DeviceContext, DeviceModel, LinkSpec, Logits, Model, ModelError, OwnDrafts,
    Split, StepContext,
};
pub use record::{Commit, RecordFull, RecordView, RecordWriter, Row, TableUpdate};
pub use sync::{WakeCounts, Woke};
pub use synthetic::{SyntheticConfig, SyntheticModel};
pub use text::{StopMatcher, printable, unprintable};
pub use timing::{
    Counters, DEFAULT_WINDOW, DevicePhase, DeviceTimes, Phase, Series, Snapshot, Stats, Summary,
};
