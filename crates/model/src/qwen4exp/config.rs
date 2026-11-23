//! The configuration of a `qwen4exp` model, read from the metadata of its GGUF
//! file (`docs/reference/qwen4exp-architecture.md`, section 2.2).
//!
//! Parsing is strict. Every key the architecture needs must be present, a key
//! under `qwen4exp.` that this implementation does not know is an error (it
//! could change the computation and would be ignored silently otherwise), and
//! the values must describe a model that the reference can run.

use ares_core::GgmlType;
use ares_gguf::Gguf;

use crate::files::{FileError, Keys, vocabulary};
use crate::qwen4exp::error::{Error, Result};

/// The architecture name in `general.architecture` and the prefix of its keys.
pub const ARCHITECTURE: &str = "qwen4exp";

/// Every key under `qwen4exp.` that the implementation knows.
const KNOWN_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.recurrent_layers",
    "attention.compress_ratios",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "rope.dimension_count",
    "rope.dimension_sections",
    "rope.freq_base",
    "full_attention_interval",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "ssm.time_step_rank",
    "ssm.inner_size",
    "hyper_connection.count",
    "hyper_connection.low_rank",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "ple.layers",
    "ple.ngram_size",
    "ple.heads_per_ngram",
    "ple.conv_kernel",
    "ple.eos_token_id",
    "ple.image_token_id",
    "ple.layer_multipliers",
    "ple.head_vocab_sizes",
    "ple.head_offsets",
    "embedding_length_per_layer_input",
];

/// Rotary position embedding of the attention layers and of the indexer.
#[derive(Clone, Debug, PartialEq)]
pub struct Rope {
    /// The number of leading values of a head that are rotated, in pairs
    /// `(i, i + dim / 2)` (`rope.dimension_count`). The rest is not touched.
    pub dim: usize,
    /// How many of the `dim / 2` rotated pairs read each position stream
    /// (`rope.dimension_sections`: time, height, width, extra), interleaved
    /// as `t, h, w, t, h, w, ...`. Text has `t = h = w =` the cell index and
    /// zero for the extra kind.
    pub sections: [usize; 4],
    /// The base of the geometric series of rotation frequencies
    /// (`rope.freq_base`).
    pub freq_base: f32,
}

/// The attention layers.
#[derive(Clone, Debug, PartialEq)]
pub struct Attention {
    /// Query heads, `H` (`attention.head_count`).
    pub n_head: usize,
    /// Key and value heads, `HKV` (`attention.head_count_kv`); each serves
    /// `n_head / n_head_kv` consecutive query heads.
    pub n_head_kv: usize,
    /// The width of a query, key and value head, `HD` (`attention.key_length`,
    /// which must equal `attention.value_length`).
    pub head_dim: usize,
    /// Rotary embedding.
    pub rope: Rope,
}

/// The indexer that picks the cells an attention layer reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Indexer {
    /// Indexer query heads, `IH` (`attention.indexer.head_count`). The key has
    /// one head.
    pub n_head: usize,
    /// The width of an indexer query head and of the key, `ID`
    /// (`attention.indexer.key_length`).
    pub key_dim: usize,
    /// Cells per block, `B`: the one non-zero value of
    /// `attention.compress_ratios`.
    pub block: usize,
    /// The selection budget in cells, `TOPK` (`attention.indexer.top_k`), a
    /// multiple of `block`.
    pub top_k: usize,
}

impl Indexer {
    /// The largest number of cells a row attends to without selection, and the
    /// cell budget of [`SelectionRule::CellBudget`]: `top_k + block - 1`.
    pub fn dense_limit(&self) -> usize {
        self.top_k + self.block - 1
    }
}

/// How an attention row past [`Indexer::dense_limit`] cells picks what it
/// reads (difference 1 of the specification).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SelectionRule {
    /// Strata's main engine, the rule of the specification: the tail, then
    /// cells of the best blocks until `top_k + block - 1` cells are selected;
    /// the last block taken may give only its first cells.
    #[default]
    CellBudget,
    /// The fork and Strata's split engine: the best `top_k / block` whole
    /// blocks and the tail, `top_k + n % block` cells.
    WholeBlocks,
}

/// The recurrent layers: a gated delta rule with a short convolution in front.
#[derive(Clone, Debug, PartialEq)]
pub struct Recurrent {
    /// Taps of the convolution (`ssm.conv_kernel`).
    pub conv_kernel: usize,
    /// Query and key heads, `GK` (`ssm.group_count`).
    pub n_k_heads: usize,
    /// Value heads, `GV`, which is also the number of state matrices
    /// (`ssm.time_step_rank`). A multiple of `n_k_heads`; value head `n` uses
    /// key head `n % n_k_heads`.
    pub n_v_heads: usize,
    /// The width of a query, key and value head, `S` (`ssm.state_size`).
    pub head_dim: usize,
}

impl Recurrent {
    /// The width of the query part, and of the key part, of the projection.
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_dim
    }

    /// The width of the value part, `ssm.inner_size`.
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_dim
    }

    /// The channels of the convolution, `CG`: query, key and value together.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// The mixture of experts every layer has.
#[derive(Clone, Debug, PartialEq)]
pub struct Experts {
    /// Routed experts per layer, `E` (`expert_count`).
    pub count: usize,
    /// Experts a token uses, `K` (`expert_used_count`).
    pub used: usize,
    /// The inner width of a routed expert, `FF` (`expert_feed_forward_length`).
    pub ff: usize,
    /// The inner width of the shared expert
    /// (`expert_shared_feed_forward_length`).
    pub shared_ff: usize,
}

/// The hashed n-gram embedding ("PLE") of one layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Ngram {
    /// The layer that applies it, 0-based (`ple.layers`, exactly one).
    pub layer: usize,
    /// Tokens hashed together at most (`ple.ngram_size`), which is also the
    /// dilation of the convolution.
    pub ngram_size: usize,
    /// Table rows per n-gram order (`ple.heads_per_ngram`).
    pub heads_per_ngram: usize,
    /// Taps of the convolution (`ple.conv_kernel`).
    pub conv_kernel: usize,
    /// The token that cuts the hash window (`ple.eos_token_id`).
    pub eos_token: u32,
    /// The token an image cell hashes as (`ple.image_token_id`; the cut token
    /// when the file has no such key).
    pub image_token: u32,
    /// Values per table row (`embedding_length_per_layer_input`).
    pub row_dim: usize,
    /// The hash multiplier of each window slot (`ple.layer_multipliers`), slot
    /// 0 being the token itself; `ngram_size` values.
    pub multipliers: Vec<u64>,
    /// The number of rows each head addresses (`ple.head_vocab_sizes`).
    pub head_vocab: Vec<u64>,
    /// The first table row of each head (`ple.head_offsets`).
    pub head_offset: Vec<u64>,
    /// The rows of `per_layer_token_embd.weight`.
    pub table_rows: u64,
    /// The type the file stores the table's rows in (IQ4_NL in the real file).
    pub table_type: GgmlType,
}

impl Ngram {
    /// Table rows gathered per token: `(ngram_size - 1) * heads_per_ngram`.
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// The cells of history the convolution reads besides the cell itself:
    /// `(conv_kernel - 1) * ngram_size`.
    pub fn history(&self) -> usize {
        (self.conv_kernel - 1) * self.ngram_size
    }
}

/// Everything a `qwen4exp` model is made of that is not a weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Layers, `L` (`block_count`).
    pub n_layer: usize,
    /// The width of one residual stream, `N` (`embedding_length`).
    pub n_embd: usize,
    /// Residual streams, `hc` (`hyper_connection.count`).
    pub n_stream: usize,
    /// The rank of the stream mixer, `LR` (`hyper_connection.low_rank`).
    pub mixer_rank: usize,
    /// The number of tokens: the rows of `token_embd.weight`.
    pub n_vocab: usize,
    /// The longest context the model was trained for (`context_length`).
    pub n_ctx: usize,
    /// The epsilon of every RMS norm and of the normalisation of queries and
    /// keys in the recurrent layers (`attention.layer_norm_rms_epsilon`).
    pub rms_eps: f32,
    /// Per layer, whether it is an attention layer (otherwise recurrent):
    /// from `attention.recurrent_layers`, or from `full_attention_interval`
    /// when that array is absent.
    pub attention_layers: Vec<bool>,
    /// The attention layers.
    pub attention: Attention,
    /// The indexer of the attention layers.
    pub indexer: Indexer,
    /// The recurrent layers.
    pub recurrent: Recurrent,
    /// The mixture of experts.
    pub experts: Experts,
    /// The hashed n-gram embedding, when the model has one.
    pub ngram: Option<Ngram>,
}

impl Config {
    /// Reads and checks the configuration of the model in `gguf`.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let keys = Keys::of(gguf, ARCHITECTURE)?;
        keys.only_known(KNOWN_KEYS, "the qwen4exp architecture")?;
        let inconsistent = |what: String| Err(Error::Inconsistent { what });

        let n_layer = keys.uint("block_count")?;
        let n_embd = keys.uint("embedding_length")?;
        let n_stream = keys.uint("hyper_connection.count")?;
        if n_stream < 2 {
            return Err(keys
                .bad(
                    "hyper_connection.count",
                    format!("is {n_stream}, the architecture needs at least 2 streams"),
                )
                .into());
        }

        // Which layers are attention layers.
        let from_array = keys.bools("attention.recurrent_layers")?;
        let interval = match keys.get("full_attention_interval") {
            Some(_) => Some(keys.uint("full_attention_interval")?),
            None => None,
        };
        let from_interval =
            interval.map(|i| (0..n_layer).map(|l| (l + 1).is_multiple_of(i)).collect::<Vec<_>>());
        let attention_layers: Vec<bool> = match (from_array, from_interval) {
            (Some(recurrent), other) => {
                if recurrent.len() != n_layer {
                    return Err(keys
                        .bad(
                            "attention.recurrent_layers",
                            format!("has {} entries for {n_layer} layers", recurrent.len()),
                        )
                        .into());
                }
                let attention: Vec<bool> = recurrent.iter().map(|r| !r).collect();
                if let Some(other) = other
                    && other != attention
                {
                    return inconsistent(
                        "attention.recurrent_layers and full_attention_interval name different \
                         attention layers"
                            .into(),
                    );
                }
                attention
            }
            (None, Some(attention)) => attention,
            (None, None) => {
                let key = keys.name("attention.recurrent_layers");
                return Err(FileError::MissingKey { key }.into());
            }
        };

        // The block size of the indexer: one ratio, on exactly the attention layers.
        let ratios = keys.required_uints("attention.compress_ratios")?;
        if ratios.len() != n_layer {
            return Err(keys
                .bad(
                    "attention.compress_ratios",
                    format!("has {} entries for {n_layer} layers", ratios.len()),
                )
                .into());
        }
        let mut block = 0usize;
        for (layer, (&ratio, &attention)) in ratios.iter().zip(&attention_layers).enumerate() {
            if (ratio != 0) != attention {
                return inconsistent(format!(
                    "layer {layer} has compress ratio {ratio} but is {} attention layer",
                    if attention { "an" } else { "not an" }
                ));
            }
            if ratio != 0 {
                if block != 0 && block as u64 != ratio {
                    return inconsistent(format!(
                        "the attention layers must share one compress ratio, found {block} and \
                         {ratio}"
                    ));
                }
                block = ratio as usize;
            }
        }
        let top_k = keys.uint("attention.indexer.top_k")?;
        if block == 1 || (block > 0 && !top_k.is_multiple_of(block)) {
            return inconsistent(format!(
                "the compress ratio must be above 1 and divide attention.indexer.top_k, found \
                 {block} and {top_k}"
            ));
        }

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
        let n_head = keys.uint("attention.head_count")?;
        let n_head_kv = keys.uint("attention.head_count_kv")?;
        let state_size = keys.uint("ssm.state_size")?;
        let inner_size = keys.uint("ssm.inner_size")?;
        let n_k_heads = keys.uint("ssm.group_count")?;
        let n_v_heads = keys.uint("ssm.time_step_rank")?;

        let mut config = Config {
            n_layer,
            n_embd,
            n_stream,
            mixer_rank: keys.uint("hyper_connection.low_rank")?,
            n_vocab: 0,
            n_ctx: keys.uint("context_length")?,
            rms_eps: keys.float("attention.layer_norm_rms_epsilon")?,
            attention_layers,
            attention: Attention {
                n_head,
                n_head_kv,
                head_dim,
                rope: Rope { dim: rope_dim, sections, freq_base: keys.float("rope.freq_base")? },
            },
            indexer: Indexer {
                n_head: keys.uint("attention.indexer.head_count")?,
                key_dim: keys.uint("attention.indexer.key_length")?,
                block,
                top_k,
            },
            recurrent: Recurrent {
                conv_kernel: keys.uint("ssm.conv_kernel")?,
                n_k_heads,
                n_v_heads,
                head_dim: state_size,
            },
            experts: Experts {
                count: keys.uint("expert_count")?,
                used: keys.uint("expert_used_count")?,
                ff: keys.uint("expert_feed_forward_length")?,
                shared_ff: keys.uint("expert_shared_feed_forward_length")?,
            },
            ngram: None,
        };

        if value_length != head_dim {
            return inconsistent(format!(
                "attention.key_length is {head_dim} but value_length is {value_length}"
            ));
        }
        if !n_head.is_multiple_of(n_head_kv) {
            return inconsistent(format!(
                "{n_head} query heads cannot be grouped over {n_head_kv} key/value heads"
            ));
        }
        if !rope_dim.is_multiple_of(2) || rope_dim > head_dim || rope_dim > config.indexer.key_dim {
            return inconsistent(format!(
                "rope.dimension_count {rope_dim} must be even and at most the head width \
                 {head_dim} and the indexer key width {}",
                config.indexer.key_dim
            ));
        }
        if sections.iter().sum::<usize>() != rope_dim / 2 {
            return inconsistent(format!(
                "rope.dimension_sections {sections:?} must add up to half of \
                 rope.dimension_count ({})",
                rope_dim / 2
            ));
        }
        if inner_size != state_size * n_v_heads {
            return inconsistent(format!(
                "ssm.inner_size {inner_size} is not ssm.state_size {state_size} times \
                 ssm.time_step_rank {n_v_heads}"
            ));
        }
        if !n_v_heads.is_multiple_of(n_k_heads) {
            return inconsistent(format!(
                "ssm.time_step_rank {n_v_heads} is not a multiple of ssm.group_count {n_k_heads}"
            ));
        }
        if config.experts.used > config.experts.count {
            return inconsistent(format!(
                "expert_used_count {} is more than expert_count {}",
                config.experts.used, config.experts.count
            ));
        }
        if config.attention_layers.iter().any(|&a| a) && block == 0 {
            return inconsistent("attention layers without a compress ratio".into());
        }

        config.n_vocab = vocabulary(gguf, n_embd)?;
        config.ngram = Self::ngram(gguf, &keys, &config)?;
        Ok(config)
    }

    fn ngram(gguf: &Gguf, keys: &Keys<'_>, config: &Config) -> Result<Option<Ngram>> {
        let inconsistent = |what: String| Err(Error::Inconsistent { what });
        let layers = keys.uints("ple.layers")?.unwrap_or_default();
        let layer = match layers.as_slice() {
            [] => return Ok(None),
            [layer] => *layer as usize,
            more => {
                return Err(keys
                    .bad(
                        "ple.layers",
                        format!("lists {} layers, the architecture has one such layer", more.len()),
                    )
                    .into());
            }
        };
        if layer >= config.n_layer {
            return Err(keys
                .bad("ple.layers", format!("names layer {layer}, past the last"))
                .into());
        }
        let ngram_size = keys.uint("ple.ngram_size")?;
        if ngram_size < 2 {
            return Err(keys
                .bad("ple.ngram_size", format!("is {ngram_size}, at least 2 needed"))
                .into());
        }
        let token = |suffix: &str| -> Result<u32> {
            let id = keys.uint0(suffix)?;
            if id >= config.n_vocab {
                return Err(keys
                    .bad(suffix, format!("is {id}, outside the vocabulary of {}", config.n_vocab))
                    .into());
            }
            Ok(id as u32)
        };
        let eos_token = token("ple.eos_token_id")?;
        let image_token = match keys.get("ple.image_token_id") {
            Some(_) => token("ple.image_token_id")?,
            None => eos_token,
        };
        let name = "per_layer_token_embd.weight";
        let table =
            gguf.tensor(name).ok_or_else(|| FileError::MissingTensor { name: name.into() })?;
        let mut ngram = Ngram {
            layer,
            ngram_size,
            heads_per_ngram: keys.uint("ple.heads_per_ngram")?,
            conv_kernel: keys.uint("ple.conv_kernel")?,
            eos_token,
            image_token,
            row_dim: keys.uint("embedding_length_per_layer_input")?,
            multipliers: keys.required_uints("ple.layer_multipliers")?,
            head_vocab: keys.required_uints("ple.head_vocab_sizes")?,
            head_offset: keys.required_uints("ple.head_offsets")?,
            table_rows: table.dims.get(1),
            table_type: table.ty,
        };
        let heads = ngram.n_heads();
        if heads * ngram.row_dim != config.n_embd {
            return inconsistent(format!(
                "{heads} n-gram heads of {} values do not make one stream of {}",
                ngram.row_dim, config.n_embd
            ));
        }
        if ngram.multipliers.len() < ngram_size {
            return Err(keys
                .bad(
                    "ple.layer_multipliers",
                    format!("has {} values for a window of {ngram_size}", ngram.multipliers.len()),
                )
                .into());
        }
        ngram.multipliers.truncate(ngram_size);
        for (suffix, array) in
            [("ple.head_vocab_sizes", &ngram.head_vocab), ("ple.head_offsets", &ngram.head_offset)]
        {
            if array.len() != heads {
                let problem = format!("has {} values for {heads} heads", array.len());
                return Err(keys.bad(suffix, problem).into());
            }
        }
        for head in 0..heads {
            let (size, offset) = (ngram.head_vocab[head], ngram.head_offset[head]);
            let fits = size > 0 && offset.checked_add(size).is_some_and(|e| e <= ngram.table_rows);
            if !fits {
                return inconsistent(format!(
                    "n-gram head {head} addresses rows {offset} + {size}, the table has {}",
                    ngram.table_rows
                ));
            }
        }
        Ok(Some(ngram))
    }

    /// Whether layer `layer` is an attention layer (otherwise it is recurrent).
    pub fn is_attention(&self, layer: usize) -> bool {
        self.attention_layers[layer]
    }

    /// The number of attention layers of the model.
    pub fn n_attention_layers(&self) -> usize {
        self.attention_layers.iter().filter(|&&a| a).count()
    }

    /// The width of all residual streams together, `D = hc * N`.
    pub fn wide(&self) -> usize {
        self.n_stream * self.n_embd
    }

    /// The width of the query rows and of the key (or value) rows of an
    /// attention layer: `(n_head * head_dim, n_head_kv * head_dim)`.
    pub fn attention_dims(&self) -> (usize, usize) {
        let a = &self.attention;
        (a.n_head * a.head_dim, a.n_head_kv * a.head_dim)
    }
}
