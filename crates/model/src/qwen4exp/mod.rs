//! The `qwen4exp` architecture: Qwen3.8-Flash-Next, the preview of Qwen4. A
//! mixture of experts over several residual streams, with recurrent layers (a
//! gated delta rule), every fourth layer an attention layer that reads the
//! cells an indexer selects, and a hashed n-gram embedding. The specification
//! is `docs/reference/qwen4exp-architecture.md`.

pub mod config;
pub mod device;
mod error;
pub mod reference;
pub mod synthetic;
pub mod weights;

pub use config::{Config, SelectionRule};
pub use error::{Error, Result};
pub use reference::{
    DraftState, DraftWindow, LogitRows, Observer, Recorder, Reference, Routing, SequenceState,
    Window,
};
pub use weights::{DraftWeights, Weights};
