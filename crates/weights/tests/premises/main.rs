//! The premises of the tier bookkeeping, shown on a synthetic workload.
//!
//! Every test here builds a routing workload ([`synthetic`]) with the
//! properties real routing has, runs the bookkeeping the way a runtime would
//! call it, and states what the technique buys. The numbers are those of the
//! synthetic workload, not of any model.

mod cache;
mod lookahead;
mod split;
mod synthetic;
mod table;
