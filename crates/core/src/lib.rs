//! Types shared by every crate of the engine.
//!
//! This crate has no knowledge of devices, files or models. It holds the
//! vocabulary the other crates agree on: the element types a tensor can have
//! ([`GgmlType`]), byte quantities ([`units`]), the memory accounting that
//! makes "does it fit" a question answered before anything is allocated
//! ([`ledger`]) and the broker that lets the users of host RAM, which changes
//! hands while the engine runs, share one limit by making room for each other
//! ([`broker`]).

pub mod broker;
pub mod ggml_type;
pub mod ledger;
pub mod units;

pub use broker::{
    Broker, BrokerError, Class, Consumer, ConsumerId, ConsumerInfo, Growth, Plan, Release,
};
pub use ggml_type::GgmlType;
pub use ledger::{Ledger, LedgerError};
