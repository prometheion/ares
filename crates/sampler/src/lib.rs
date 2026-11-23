//! The definitions of the sampler.
//!
//! A sampler picks the next token of each row of a step from the row's logits.
//! This crate holds what every backend of it shares:
//!
//! - [`SampleParams`] (the settings of a sequence) and [`SampleRow`] (what the
//!   kernels read for one row), the corrections and the masks that rows carry,
//!   and the records the kernels write back;
//! - the [arithmetic](arith) that a pick is made of, which a backend repeats
//!   operation by operation so that its records equal the host's bit for bit;
//! - [`sample_reference`], the pick of a row written from its definition;
//! - the penalties and the DRY penalty ([`penalties`]), which the host turns
//!   into corrections of the logits before the device picks.
//!
//! The kernels themselves are in `ares-cuda`.

pub mod arith;
pub mod mask;
pub mod penalties;
mod reference;
mod row;

pub use mask::{TokenMask, mask_allows};
pub use reference::{Kept, Reference, sample_reference};
pub use row::*;
