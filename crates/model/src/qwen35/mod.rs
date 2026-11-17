//! The `qwen35` architecture: Qwen3.5 and Qwen3.8 dense models, which mix
//! recurrent blocks (a gated delta rule) with every fourth block of full
//! attention. The specification is `docs/reference/qwen35-architecture.md`.

pub mod config;
pub mod device;
mod error;
pub mod reference;
pub mod weights;

pub use config::Config;
pub use error::{Error, Result};
pub use reference::{
    LogitRows, MtpDraft, MtpRows, MtpState, Reference, SequenceState, StateStorage, Window,
};
pub use weights::{MTP_ARCHITECTURE, Weights};
