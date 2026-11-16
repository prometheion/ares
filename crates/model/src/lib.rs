//! Model architectures of the engine: for each, the configuration read from a
//! GGUF file, the table of its weights, and a CPU reference of the forward pass
//! that the GPU implementations are held to.
//!
//! `qwen35` (dense) and `qwen4exp` (a mixture of experts) exist so far, and
//! `qwen3vl`, the image tower and merger that both read images through.

pub mod chat;
mod files;
pub mod qwen35;
pub mod qwen3vl;
pub mod qwen4exp;
