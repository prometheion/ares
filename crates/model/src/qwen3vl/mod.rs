//! The `qwen3vl_merger` projector: the image tower of Qwen3.8 (a vision
//! transformer over patches of the image) and the merger that turns groups of
//! patches into rows of the language model's width. Its file is a GGUF of
//! `general.architecture = "clip"` beside the language model's.
//!
//! What is here is the configuration, the weight table, the pixel work (size
//! rule, resize, normalisation, patch rows), a CPU reference of the pass and the
//! pass on a device ([`DeviceTower`], with its weights in VRAM or in page-locked
//! RAM). The reference is the test oracle of the device, not a serving path. Both
//! take pixels; decoding image files is not their business.
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
pub mod device;
mod error;
pub mod image;
pub mod projector;
pub mod reference;
pub mod weights;

pub use config::Config;
pub use device::{DeviceTower, Home, Scratch};
pub use error::{Error, Result};
pub use image::{Limits, normalize, patch_position, patch_rows, resize_bicubic, target_size};
pub use projector::{Projector, TowerAt};
pub use reference::{Arithmetic, Reference, Rows, Stage};
pub use weights::Weights;
