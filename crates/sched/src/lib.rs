//! Deciding what each step of the engine computes.
//!
//! The engine runs in steps. A step pushes a batch of token rows through
//! every layer once. The rows of a step need not have anything in common:
//! a step can carry the next token of three conversations, the drafts of one
//! of them for verification, and a slice of somebody's long prompt. Batch
//! shape is data, not structure, so no step is more expensive to set up than
//! another.
//!
//! This crate holds the decisions, as plain functions over numbers, with no
//! device and no model in sight:
//!
//! - [`plan`] composes a step from what the sequences want ([`Want`]), giving
//!   generating sequences their newest token first, then the option rows of
//!   decisions, then drafts, and prompts the rest.
//! - [`Share`] decides whether the next step is a round of generation or a
//!   piece of prompt, so that a long prompt slows generation by a chosen
//!   factor and no more, and a short one is read at once.
//! - [`Pass`] is the other order for long prompts of a model whose experts
//!   are fetched: all rows up to a layer's router, then that layer's experts
//!   for all rows, then on, so that a prompt fetches each expert once.
//! - [`PriceTable`] is what a step costs by its rows, as a table (the weight
//!   kernels take their rows in launches, so the price has edges), measured on
//!   the devices at load and kept true by the steps that run (a late step
//!   moves nothing, and a size that has not run for long is tried again), with
//!   a term for the cells the rows read ([`StepTimeModel`] fits that term).
//! - [`Acceptance`] is how often a drafter's drafts are accepted, per
//!   position, for a sequence, starting from what the engine has seen.
//!   [`choose`] picks, for the whole step, whose drafts go into each
//!   sequence's window and how many, by the expected tokens per millisecond.
//!   [`Confidence`] is how often a draft is accepted by the probability its
//!   drafter gave it, and [`need`] the bar a chain is cut by as it is made:
//!   the same rule, with the chances of the drafts it has.
//! - [`LookupCorpus`] and [`LookupDrafter`] propose drafts by continuing the
//!   longest earlier occurrence of the current text, in any sequence: one
//!   suffix automaton over everything the engine has read and written, with
//!   a confidence per proposed token. This is what makes copying (file
//!   edits, quoted code, a document sent again) several times faster than
//!   generating.
//! - [`Admission`] decides who runs, who waits and who gives way: pages are
//!   claimed per step, a reservation covers the output that is expected, a
//!   request that does not fit waits, and a step's claims are settled against
//!   a copy of the counters.

mod admission;
mod confidence;
mod draft;
mod lookup;
mod pass;
mod plan;
mod price;
mod share;
mod timing;
mod window;

pub use admission::{Admission, AdmissionConfig, Arrival, Claims, ExceedsPool, Outcome, PoolView};
pub use confidence::{BINS as CONFIDENCE_BINS, Confidence, links_made, need};
pub use draft::{Acceptance, BELIEF};
pub use lookup::{LookupConfig, LookupCorpus, LookupDrafter, LookupMatch};
pub use pass::{Pass, PassStep, Point, pass_rows_between, runs_as_pass};
pub use plan::{Limits, Part, PartKind, Plan, Request, SeqId, Want, plan};
pub use price::{Curve, Estimate, OLD as PRICE_OLD, PriceTable};
pub use share::{Pending, Share, Turn};
pub use timing::{Load, StepCost, StepTimeModel};
pub use window::{Choice, Offer, Prices, Wish, choose, rate};
