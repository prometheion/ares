//! The `qwen3vl_merger` projector: the image tower of Qwen3.8 (a vision
//! transformer over patches of the image) and the merger that turns groups of
//! patches into rows of the language model's width. Its file is a GGUF of
//! `general.architecture = "clip"` beside the language model's.
//!
//! What is here is the configuration, the weight table, the pixel work (size
//! rule, resize, normalisation) and a CPU reference of the pass. The reference
//! is a test oracle for a device implementation, not a serving path. It takes
//! pixels; decoding image files is not its business.
//!
//! The pass, with `pw` x `ph` patches of `patch` x `patch` pixels and
//! `nx = pw / 2`, `ny = ph / 2`:
//!
//! 1. each patch, as `3 * patch * patch` values (channel, row, column), times
//!    the sum of the two patch kernels, plus the patch bias;
//! 2. patches are kept in the merger's order: the `2 x 2` groups row by row,
//!    inside a group `(dy, dx)` = (0,0), (0,1), (1,0), (1,1);
//! 3. plus the learned position table, resized bilinearly to `pw` x `ph`;
//! 4. `n_layer` blocks: layer norm, one matrix for query, key and value, a
//!    rotary rotation of query and key by the patch's row and column, full
//!    softmax attention, the output matrix, then layer norm and an MLP with
//!    the tanh GELU, both added to the residual;
//! 5. a last layer norm; the four patches of a group side by side through the
//!    merger's two matrices with the GELU between: `nx * ny` rows.

pub mod config;
mod error;
pub mod image;
pub mod reference;
pub mod weights;

pub use config::Config;
pub use error::{Error, Result};
pub use image::{Limits, normalize, resize_bicubic, target_size};
pub use reference::{Arithmetic, Reference, Rows, Stage};
pub use weights::Weights;
