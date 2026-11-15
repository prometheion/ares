//! The configuration of a `qwen3vl_merger` projector, read from the metadata
//! of its GGUF file.
//!
//! Parsing is strict: every key the pass needs must be present, a key under
//! `clip.` that this implementation does not know is an error, and a file
//! that asks for something the reference does not compute (another projector
//! type, deepstack layers, an activation other than GELU, a merge other than
//! `2 x 2`) is refused.

use ares_gguf::{Gguf, Value};

use crate::qwen3vl::error::{Error, Result};

/// `general.architecture` of a projector file.
pub const ARCHITECTURE: &str = "clip";
/// `clip.projector_type` of the projector implemented here.
pub const PROJECTOR_TYPE: &str = "qwen3vl_merger";

/// Every key under `clip.` that the implementation knows.
const KNOWN_KEYS: &[&str] = &[
    "has_vision_encoder",
    "projector_type",
    "use_gelu",
    "vision.projection_dim",
    "vision.image_size",
    "vision.patch_size",
    "vision.embedding_length",
    "vision.feed_forward_length",
    "vision.block_count",
    "vision.attention.head_count",
    "vision.attention.layer_norm_epsilon",
    "vision.image_mean",
    "vision.image_std",
    "vision.spatial_merge_size",
    "vision.is_deepstack_layers",
];

/// The shape of the tower and the merger.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The side of a patch in pixels (`vision.patch_size`).
    pub patch: usize,
    /// How many patches per side the merger joins (`vision.spatial_merge_size`);
    /// always 2.
    pub merge: usize,
    /// The width of the tower (`vision.embedding_length`).
    pub n_embd: usize,
    /// The hidden width of a block's MLP (`vision.feed_forward_length`).
    pub n_ff: usize,
    /// The number of blocks (`vision.block_count`).
    pub n_layer: usize,
    /// The attention heads of a block (`vision.attention.head_count`).
    pub n_head: usize,
    /// The epsilon of every layer norm (`vision.attention.layer_norm_epsilon`).
    pub eps: f32,
    /// What is subtracted from a pixel value in `0..=1`, per channel
    /// (`vision.image_mean`).
    pub mean: [f32; 3],
    /// What the difference is divided by (`vision.image_std`).
    pub std: [f32; 3],
    /// The width of an output row, which is the language model's embedding
    /// width (`vision.projection_dim`).
    pub n_out: usize,
    /// The side of the learned position table, in patches
    /// (`vision.image_size / vision.patch_size`).
    pub table_side: usize,
}

impl Config {
    /// The width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// What both sides of an image are multiples of, in pixels.
    pub fn align(&self) -> usize {
        self.patch * self.merge
    }

    /// The values of one patch: three channels of `patch * patch` pixels.
    pub fn patch_len(&self) -> usize {
        3 * self.patch * self.patch
    }

    /// Reads and checks the configuration.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let found = gguf.architecture().unwrap_or("(none)");
        if found != ARCHITECTURE {
            return Err(kind("general.architecture", found, ARCHITECTURE));
        }
        for (key, _) in gguf.metadata() {
            if let Some(rest) = key.strip_prefix("clip.")
                && !KNOWN_KEYS.contains(&rest)
            {
                return Err(Error::UnknownKey { key: key.to_owned() });
            }
        }
        let ty = get(gguf, "projector_type")?.as_str().unwrap_or("(not a string)");
        if ty != PROJECTOR_TYPE {
            return Err(kind("clip.projector_type", ty, PROJECTOR_TYPE));
        }
        if get(gguf, "use_gelu")?.as_bool() != Some(true) {
            return Err(bad("use_gelu", "must be true: the reference computes the tanh GELU"));
        }
        if let Some(value) = gguf.get("clip.has_vision_encoder")
            && value.as_bool() != Some(true)
        {
            return Err(bad("has_vision_encoder", "must be true"));
        }
        let n_layer = size(gguf, "vision.block_count")?;
        if let Some(Value::Array(layers)) = gguf.get("clip.vision.is_deepstack_layers") {
            let flags: Vec<Option<bool>> = layers.iter().map(|v| v.as_bool()).collect();
            if flags.len() != n_layer || flags.iter().any(|f| *f != Some(false)) {
                return Err(bad(
                    "vision.is_deepstack_layers",
                    "must be false for every block: deepstack layers are not implemented",
                ));
            }
        }
        let config = Self {
            patch: size(gguf, "vision.patch_size")?,
            merge: size(gguf, "vision.spatial_merge_size")?,
            n_embd: size(gguf, "vision.embedding_length")?,
            n_ff: size(gguf, "vision.feed_forward_length")?,
            n_layer,
            n_head: size(gguf, "vision.attention.head_count")?,
            eps: get(gguf, "vision.attention.layer_norm_epsilon")?
                .as_f64()
                .filter(|e| *e > 0.0 && e.is_finite())
                .ok_or_else(|| bad("vision.attention.layer_norm_epsilon", "must be positive"))?
                as f32,
            mean: triple(gguf, "vision.image_mean")?,
            std: triple(gguf, "vision.image_std")?,
            n_out: size(gguf, "vision.projection_dim")?,
            table_side: 0,
        };
        let image_size = size(gguf, "vision.image_size")?;
        let inconsistent = |what: String| Err(Error::Inconsistent { what });
        if config.merge != 2 {
            return inconsistent(format!(
                "vision.spatial_merge_size is {}, only a 2 x 2 merge is implemented",
                config.merge
            ));
        }
        if !config.n_embd.is_multiple_of(config.n_head) || !config.head_dim().is_multiple_of(4) {
            return inconsistent(format!(
                "a head of {} / {} values cannot be rotated in two halves of pairs",
                config.n_embd, config.n_head
            ));
        }
        if !image_size.is_multiple_of(config.patch) {
            return inconsistent(format!(
                "vision.image_size {image_size} is not a multiple of the patch size {}",
                config.patch
            ));
        }
        if config.std.contains(&0.0) {
            return inconsistent("vision.image_std has a zero".to_owned());
        }
        Ok(Self { table_side: image_size / config.patch, ..config })
    }
}

fn kind(key: &'static str, found: &str, expected: &'static str) -> Error {
    Error::Kind { key, found: found.to_owned(), expected }
}

fn bad(key: &str, problem: &str) -> Error {
    Error::BadKey { key: format!("clip.{key}"), problem: problem.to_owned() }
}

fn get<'g>(gguf: &'g Gguf, key: &str) -> Result<Value<'g>> {
    let key = format!("clip.{key}");
    gguf.get(&key).ok_or(Error::MissingKey { key })
}

/// A count or size: a positive integer that fits the address space.
fn size(gguf: &Gguf, key: &str) -> Result<usize> {
    get(gguf, key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| bad(key, "must be a positive integer"))
}

/// Three finite numbers, one per colour channel.
fn triple(gguf: &Gguf, key: &str) -> Result<[f32; 3]> {
    let problem = || bad(key, "must be an array of three finite numbers");
    let Value::Array(array) = get(gguf, key)? else { return Err(problem()) };
    let values: Vec<f32> = array.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
    if array.len() != 3 || values.len() != 3 || values.iter().any(|v| !v.is_finite()) {
        return Err(problem());
    }
    Ok([values[0], values[1], values[2]])
}
