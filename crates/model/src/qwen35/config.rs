//! The configuration of a `qwen35` model, read from the metadata of its GGUF
//! file.
//!
//! Parsing is strict. Every key the architecture needs must be present, a key
//! under `qwen35.` that this implementation does not know is an error (it could
//! change the computation and would be ignored silently otherwise), and the
//! values must describe a model that the reference can run. Nothing has a
//! default except the number of next-token-prediction blocks, which is zero when
//! the key is absent.

use ares_gguf::Gguf;

use crate::files::{Keys, vocabulary};
use crate::qwen35::error::{Error, Result};

/// The architecture name in `general.architecture` and the prefix of its keys.
pub const ARCHITECTURE: &str = "qwen35";

/// Every key under `qwen35.` that the implementation knows.
const KNOWN_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "rope.dimension_sections",
    "rope.freq_base",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.time_step_rank",
    "ssm.inner_size",
    "full_attention_interval",
    "nextn_predict_layers",
];

/// Rotary position embedding of the attention blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    /// The number of leading values of a head that are rotated, in pairs
    /// `(i, i + dim / 2)` (`rope.dimension_count`). The rest of the head is
    /// not touched.
    pub dim: usize,
    /// How many of the `dim / 2` rotated pairs belong to each kind of position
    /// (`rope.dimension_sections`: time, height, width, extra). Text uses the
    /// token position for the first three kinds and zero for the extra one.
    pub sections: [usize; 4],
    /// The base of the geometric series of rotation frequencies
    /// (`rope.freq_base`).
    pub freq_base: f32,
}

/// The attention blocks (every `full_attention_interval`-th block).
#[derive(Clone, Debug, PartialEq)]
pub struct Attention {
    /// Query heads (`attention.head_count`).
    pub n_head: usize,
    /// Key and value heads (`attention.head_count_kv`); each serves
    /// `n_head / n_head_kv` consecutive query heads.
    pub n_head_kv: usize,
    /// The width of a query, key and value head (`attention.key_length`, which
    /// must equal `attention.value_length`).
    pub head_dim: usize,
    /// Rotary embedding.
    pub rope: Rope,
}

/// The recurrent blocks: a gated delta rule with a short convolution in front.
#[derive(Clone, Debug, PartialEq)]
pub struct Recurrent {
    /// The width of the convolution window (`ssm.conv_kernel`).
    pub conv_kernel: usize,
    /// The number of query and key heads (`ssm.group_count`).
    pub n_k_heads: usize,
    /// The number of value heads, which is also the number of state matrices
    /// (`ssm.time_step_rank`). A multiple of `n_k_heads`.
    pub n_v_heads: usize,
    /// The width of a query and key head (`ssm.state_size`).
    pub head_k_dim: usize,
    /// The width of a value head: `ssm.inner_size / ssm.time_step_rank`, which
    /// must equal `ssm.state_size`.
    pub head_v_dim: usize,
}

impl Recurrent {
    /// The width of the query part, and of the key part, of the projection:
    /// `n_k_heads * head_k_dim`.
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_k_dim
    }

    /// The width of the value part, `ssm.inner_size`: `n_v_heads * head_v_dim`.
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_v_dim
    }

    /// The channels of the convolution, query, key and value together.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// Everything a `qwen35` model is made of that is not a weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The number of blocks that make the model: `block_count` without the
    /// next-token-prediction block.
    pub n_layer: usize,
    /// The width of the hidden state (`embedding_length`).
    pub n_embd: usize,
    /// The width of the feed-forward part (`feed_forward_length`).
    pub n_ff: usize,
    /// The number of tokens: the rows of `token_embd.weight`.
    pub n_vocab: usize,
    /// The longest context the model was trained for (`context_length`).
    pub n_ctx: usize,
    /// The epsilon of every RMS norm and of the normalisation of queries and
    /// keys in the recurrent blocks (`attention.layer_norm_rms_epsilon`).
    pub rms_eps: f32,
    /// Block `i` is an attention block when `(i + 1) % full_attention_interval
    /// == 0`, and a recurrent block otherwise.
    pub full_attention_interval: usize,
    /// The number of next-token-prediction blocks stored after the model's
    /// blocks (`nextn_predict_layers`; 0 or 1).
    pub n_nextn: usize,
    /// The attention blocks.
    pub attention: Attention,
    /// The recurrent blocks.
    pub recurrent: Recurrent,
}

impl Config {
    /// Reads and checks the configuration of the model in `gguf`.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let keys = Keys::of(gguf, ARCHITECTURE)?;
        keys.only_known(KNOWN_KEYS, "the qwen35 architecture")?;

        let block_count = keys.uint("block_count")?;
        let n_nextn = match keys.get("nextn_predict_layers") {
            None => 0,
            Some(value) => match value.as_u64() {
                Some(n @ 0..=1) => n as usize,
                _ => {
                    return Err(keys
                        .bad(
                            "nextn_predict_layers",
                            format!("is {value}, the implementation knows 0 or 1"),
                        )
                        .into());
                }
            },
        };
        if n_nextn >= block_count {
            return Err(Error::Inconsistent {
                what: format!(
                    "{block_count} blocks cannot hold {n_nextn} next-token block(s) and a model"
                ),
            });
        }
        let n_embd = keys.uint("embedding_length")?;

        let sections = keys
            .required_uints("rope.dimension_sections")?
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>();
        let sections = <[usize; 4]>::try_from(sections)
            .map_err(|_| keys.bad("rope.dimension_sections", "is not an array of 4 counts"))?;

        let head_dim = keys.uint("attention.key_length")?;
        let value_length = keys.uint("attention.value_length")?;
        let rope_dim = keys.uint("rope.dimension_count")?;
        let state_size = keys.uint("ssm.state_size")?;
        let inner_size = keys.uint("ssm.inner_size")?;
        let n_k_heads = keys.uint("ssm.group_count")?;
        let n_v_heads = keys.uint("ssm.time_step_rank")?;
        let n_head = keys.uint("attention.head_count")?;
        let n_head_kv = keys.uint("attention.head_count_kv")?;
        let full_attention_interval = keys.uint("full_attention_interval")?;
        let conv_kernel = keys.uint("ssm.conv_kernel")?;

        let mut config = Config {
            n_layer: block_count - n_nextn,
            n_embd,
            n_ff: keys.uint("feed_forward_length")?,
            n_vocab: 0,
            n_ctx: keys.uint("context_length")?,
            rms_eps: keys.float("attention.layer_norm_rms_epsilon")?,
            full_attention_interval,
            n_nextn,
            attention: Attention {
                n_head,
                n_head_kv,
                head_dim,
                rope: Rope { dim: rope_dim, sections, freq_base: keys.float("rope.freq_base")? },
            },
            recurrent: Recurrent {
                conv_kernel,
                n_k_heads,
                n_v_heads,
                head_k_dim: state_size,
                head_v_dim: inner_size / n_v_heads,
            },
        };

        let inconsistent = |what: String| Err(Error::Inconsistent { what });
        if value_length != head_dim {
            return inconsistent(format!(
                "{} is {head_dim} but value_length is {value_length}",
                keys.name("attention.key_length")
            ));
        }
        if n_head % n_head_kv != 0 {
            return inconsistent(format!(
                "{n_head} query heads cannot be grouped over {n_head_kv} key/value heads"
            ));
        }
        if rope_dim % 2 != 0 || rope_dim > head_dim {
            return inconsistent(format!(
                "rope.dimension_count {rope_dim} must be even and at most the head width {head_dim}"
            ));
        }
        if sections.iter().sum::<usize>() != rope_dim / 2 {
            return inconsistent(format!(
                "rope.dimension_sections {sections:?} must add up to half of rope.dimension_count \
                 ({})",
                rope_dim / 2
            ));
        }
        if inner_size != state_size * n_v_heads {
            return inconsistent(format!(
                "ssm.inner_size {inner_size} is not ssm.state_size {state_size} times \
                 ssm.time_step_rank {n_v_heads}"
            ));
        }
        if n_v_heads % n_k_heads != 0 {
            return inconsistent(format!(
                "ssm.time_step_rank {n_v_heads} is not a multiple of ssm.group_count {n_k_heads}"
            ));
        }

        config.n_vocab = vocabulary(gguf, n_embd)?;
        Ok(config)
    }

    /// Whether block `layer` is an attention block (otherwise it is recurrent).
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.full_attention_interval)
    }

    /// The number of attention blocks of the model.
    pub fn n_attention_layers(&self) -> usize {
        (0..self.n_layer).filter(|&l| self.is_attention(l)).count()
    }

    /// The width of the query, key and value rows of an attention block:
    /// `(n_head * head_dim, n_head_kv * head_dim)`.
    pub fn attention_dims(&self) -> (usize, usize) {
        let a = &self.attention;
        (a.n_head * a.head_dim, a.n_head_kv * a.head_dim)
    }
}
