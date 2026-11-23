//! Model architectures of the engine: for each, the configuration read from a
//! GGUF file, the table of its weights, a CPU reference of the forward pass,
//! and the implementation on devices that is held to it.
//!
//! `qwen35` (dense) and `qwen4exp` (a mixture of experts) exist so far, with
//! `qwen3vl`, the image tower and merger that both read images through, and
//! `dflash`, the drafter both can draft with. `chat` is the way from a
//! request's prompt to a model's tokens; the blocks (`blocks/`) are the parts
//! of a step every model is built from.

mod blocks;
pub mod chat;
pub mod dflash;
mod files;
pub mod qwen35;
pub mod qwen3vl;
pub mod qwen4exp;
