//! Model architectures of the engine: for each, the configuration read from a
//! GGUF file, the table of its weights, and a CPU reference of the forward pass
//! that the GPU implementations are held to.
//!
//! Only `qwen35` exists so far.

pub mod chat;
pub mod qwen35;
pub mod qwen4exp;
