//! Types shared by every crate of the engine.
//!
//! This crate has no knowledge of devices, files or models. It holds the
//! vocabulary the other crates agree on: the element types a tensor can have
//! ([`GgmlType`]), byte quantities ([`units`]) and the memory accounting that
//! makes "does it fit" a question answered before anything is allocated
//! ([`ledger`]).

pub mod ggml_type;
pub mod ledger;
pub mod units;

pub use ggml_type::GgmlType;
pub use ledger::{Ledger, LedgerError};
