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
//!   decisions, then drafts, and prompts the rest, bounded so that a long
//!   prompt slows generation and decisions by a chosen factor and no more.
//! - [`Pass`] is the other order for long prompts of a model whose experts
//!   are fetched: all rows up to a layer's router, then that layer's experts
//!   for all rows, then on, so that a prompt fetches each expert once.
//! - [`StepTimeModel`] learns what a step costs from the steps that ran.
//! - [`best_draft_len`] picks how many draft tokens to verify, from measured
//!   acceptance and cost, and with that also which drafter to use.
//! - [`NgramDrafter`] proposes drafts by continuing an earlier occurrence of
//!   the current text, which is what makes copying (file edits, quoted code)
//!   several times faster than generating.

mod draft;
mod ngram;
mod pass;
mod plan;
mod timing;

pub use draft::{Acceptance, DraftChoice, DraftCost, best_draft_len};
pub use ngram::NgramDrafter;
pub use pass::{Pass, PassStep, Point, pass_rows_between, runs_as_pass};
pub use plan::{Limits, Part, PartKind, Plan, Request, SeqId, Want, plan, prefill_rows_next_to};
pub use timing::StepTimeModel;
