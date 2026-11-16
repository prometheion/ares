//! The shape of a DFlash drafter, read from its file, and the variants of the
//! architecture that this implementation refuses.

use ares_core::GgmlType;
use ares_gguf::{Gguf, TensorInfo};

/// The `general.architecture` of a drafter file.
pub const ARCHITECTURE: &str = "dflash";

/// Why a drafter file could not be taken.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The file is a model of another architecture.
    #[error("general.architecture is {found}, a DFlash drafter is {ARCHITECTURE}")]
    Architecture { found: String },
    /// A key the drafter needs is missing or has an impossible value.
    #[error("metadata key '{key}' {problem}")]
    Key { key: String, problem: String },
    /// A variant of the architecture that the DFlash kind does not run, named.
    #[error("{what}: the DFlash drafter of Ares does not run this variant")]
    Variant { what: String },
    /// A tensor is missing or has another shape or type than the configuration gives it.
    #[error("tensor '{name}' {problem}")]
    Tensor { name: String, problem: String },
    /// The drafter does not fit the model it is to draft for.
    #[error("the drafter does not fit the model: {what}")]
    Target { what: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The keys of the architecture that this implementation reads; any other
/// `dflash.*` key could change the computation and is refused.
const KNOWN_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.causal",
    "rope.freq_base",
    "rope.dimension_sections",
    "attention.layer_norm_rms_epsilon",
    "attention.key_length",
    "attention.value_length",
    "block_size",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
    "target_layers",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
];

/// Keys of other variants, refused by the name of the variant they belong to.
const VARIANT_KEYS: &[(&str, &str)] = &[
    ("rope.scaling.type", "rotary scaling (a DSpark drafter's yarn)"),
    ("sample_from_anchor", "drafts sampled from the anchor (a DSpark drafter)"),
    ("hyper_connection.count", "a DeepSeek-V4 DSpark drafter (hyper-connections)"),
    ("attention.kv_lora_rank", "a Mistral Large 4 DSpark drafter (compressed attention)"),
    ("lilicorr.hidden_size", "a LiLiCorr lattice scorer"),
    ("hidden_activation", "an activation other than SiLU"),
    ("logit_scale", "a scale of the logits"),
    ("final_logit_softcapping", "a soft cap of the logits"),
    ("embedding_scale", "a scale of the embedding"),
    ("attention.scale", "a scale of attention other than 1/sqrt(head dim)"),
    ("attention.value_scale", "a scale of attention's values"),
];

/// Tensors of other variants, refused by the name of the variant.
const VARIANT_TENSORS: &[(&str, &str)] = &[
    ("markov_w1.weight", "a DSpark drafter (Markov head)"),
    ("d2t", "a drafter with a vocabulary of its own (d2t)"),
    ("fc_norm.weight", "a norm per tap before the fusion (fc_norm)"),
    ("token_embd.weight", "a drafter with an embedding table of its own"),
    ("output.weight", "a drafter with an output head of its own"),
    ("blk.0.attn_gate.weight", "gated attention (a Spark drafter)"),
    ("blk.0.attn_post_norm.weight", "norms after attention"),
    ("blk.0.ffn_post_norm.weight", "norms after the feed-forward part"),
];

/// A DFlash2 drafter's shape.
///
/// The drafter has no embedding table and no output head: both are the
/// model's it drafts for, and so is the vocabulary. It reads the residual
/// stream of that model before `target_layers` (its *taps*), and drafts a
/// block of `block` tokens in one pass: the newest token and `block - 1` mask
/// tokens, of which positions 1 to `block - 1` give 16 candidates each and a
/// table of scores between the candidates of neighbouring positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub n_layer: usize,
    /// The width of the drafter, which is the model's.
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    /// The base of the rotary embedding, which turns all `head_dim` values of
    /// a head in pairs `(i, i + head_dim / 2)`.
    pub rope_base: f32,
    /// Tokens of a block: the newest token and `block - 1` drafts.
    pub block: usize,
    /// The taps of the dynamic convolution and the channels per group.
    pub conv_kernel: usize,
    pub conv_group: usize,
    /// The selector: the rank of its tables and the candidates per position.
    pub selector_rank: usize,
    pub top_k: usize,
    /// The model's layers whose input the drafter reads, in order.
    pub target_layers: Vec<u32>,
    /// The most cells before a row it attends: a row at `p` sees the cells
    /// of positions above `p - window`.
    pub window: usize,
    /// The token of the rows after the newest one.
    pub mask_token: u32,
    /// The tokens of the selector's tables, which are the model's.
    pub n_vocab: usize,
}

impl Config {
    /// Reads the shape of the drafter in `gguf`, and refuses a variant the
    /// DFlash kind cannot run by its name.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let found = gguf.architecture().unwrap_or("(none)");
        if found != ARCHITECTURE {
            return Err(Error::Architecture { found: found.to_owned() });
        }
        let prefix = format!("{ARCHITECTURE}.");
        for (key, _) in gguf.metadata() {
            let Some(rest) = key.strip_prefix(&prefix) else { continue };
            if let Some((_, what)) = VARIANT_KEYS.iter().find(|(k, _)| rest.starts_with(k)) {
                return Err(Error::Variant { what: format!("{what} ({key})") });
            }
            if !KNOWN_KEYS.contains(&rest) {
                return Err(Error::Variant {
                    what: format!("a key this implementation does not know ({key})"),
                });
            }
        }
        for (name, what) in VARIANT_TENSORS {
            if gguf.tensor(name).is_some() {
                return Err(Error::Variant { what: format!("{what} ({name})") });
            }
        }
        if gguf.tensor("selector_hidden.weight").is_none() {
            return Err(Error::Variant {
                what: "a drafter without the selector (DFlash 1)".into(),
            });
        }
        if gguf.tensor("blk.0.attn_conv_proj.weight").is_none() {
            return Err(Error::Variant { what: "a drafter without dynamic convolutions".into() });
        }

        let key = |suffix: &str| format!("{prefix}{suffix}");
        let missing = |suffix: &str| Error::Key { key: key(suffix), problem: "is missing".into() };
        let uint = |suffix: &str| -> Result<usize> {
            let n = gguf.u64(&key(suffix)).ok_or_else(|| missing(suffix))?;
            match usize::try_from(n) {
                Ok(n) if n > 0 => Ok(n),
                _ => Err(Error::Key { key: key(suffix), problem: format!("is {n}") }),
            }
        };
        let float = |suffix: &str| -> Result<f32> {
            let x = gguf.f64(&key(suffix)).ok_or_else(|| missing(suffix))?;
            if x.is_finite() && x > 0.0 {
                Ok(x as f32)
            } else {
                Err(Error::Key { key: key(suffix), problem: format!("is {x}") })
            }
        };
        if gguf.bool(&key("attention.causal")) != Some(false) {
            return Err(Error::Variant {
                what: "causal attention inside the block (attention.causal is not false)".into(),
            });
        }
        let head_dim = uint("attention.key_length")?;
        if uint("attention.value_length")? != head_dim {
            return Err(Error::Variant { what: "keys and values of different lengths".into() });
        }
        let n_layer = uint("block_count")?;
        if let Some(pattern) = gguf.array(&key("attention.sliding_window_pattern")) {
            let all = pattern.iter().all(|v| v.as_bool() == Some(true));
            if !all || pattern.len() != n_layer as u64 {
                return Err(Error::Variant {
                    what: "layers without the sliding window (sliding_window_pattern)".into(),
                });
            }
        }
        // A drafter of a model with sections of its rotary embedding carries
        // degenerate ones: every pair turns with the position.
        if let Some(sections) = gguf.array(&key("rope.dimension_sections")) {
            let sections = sections.to_u32_vec().unwrap_or_default();
            let degenerate = sections.first().copied() == Some(head_dim as u32 / 2)
                && sections.iter().skip(1).all(|s| *s == 0);
            if !degenerate {
                return Err(Error::Variant {
                    what: format!("rotary sections {sections:?} (rope.dimension_sections)"),
                });
            }
        }
        let target_layers = gguf
            .array(&key("target_layers"))
            .and_then(|a| a.to_u32_vec())
            .filter(|layers| !layers.is_empty())
            .ok_or_else(|| missing("target_layers"))?;
        if target_layers.windows(2).any(|w| w[0] >= w[1]) {
            return Err(Error::Key {
                key: key("target_layers"),
                problem: format!("is {target_layers:?}, not ascending"),
            });
        }
        let mask_token = gguf
            .u64("tokenizer.ggml.mask_token_id")
            .and_then(|t| u32::try_from(t).ok())
            .ok_or_else(|| Error::Key {
                key: "tokenizer.ggml.mask_token_id".into(),
                problem: "is missing".into(),
            })?;
        let n_vocab =
            gguf.array("tokenizer.ggml.tokens").map(|a| a.len() as usize).ok_or_else(|| {
                Error::Key { key: "tokenizer.ggml.tokens".into(), problem: "is missing".into() }
            })?;
        let c = Config {
            n_layer,
            n_embd: uint("embedding_length")?,
            n_ff: uint("feed_forward_length")?,
            n_head: uint("attention.head_count")?,
            n_head_kv: uint("attention.head_count_kv")?,
            head_dim,
            eps: float("attention.layer_norm_rms_epsilon")?,
            rope_base: float("rope.freq_base")?,
            block: uint("block_size")?,
            conv_kernel: uint("conv_kernel_size")?,
            conv_group: uint("conv_group_size")?,
            selector_rank: uint("selector_rank")?,
            top_k: uint("selector_top_k")?,
            target_layers,
            window: uint("attention.sliding_window")?,
            mask_token,
            n_vocab,
        };
        c.check()?;
        Ok(c)
    }

    fn check(&self) -> Result<()> {
        let fail = |what: String| Err(Error::Variant { what });
        if self.block < 2 || self.block > MAX_BLOCK {
            return fail(format!("a block of {} tokens (2 to {MAX_BLOCK} run)", self.block));
        }
        if self.top_k == 0 || self.top_k > MAX_TOP_K {
            return fail(format!("{} candidates per position (1 to {MAX_TOP_K} run)", self.top_k));
        }
        if self.conv_kernel == 0 || self.conv_kernel > self.block {
            return fail(format!("a convolution of {} taps", self.conv_kernel));
        }
        if !self.n_embd.is_multiple_of(self.conv_group) {
            return fail(format!("groups of {} in a width of {}", self.conv_group, self.n_embd));
        }
        if self.head_dim % 2 != 0 || self.head_dim > 256 {
            return fail(format!("heads of {} values", self.head_dim));
        }
        if self.n_head_kv == 0 || !self.n_head.is_multiple_of(self.n_head_kv) {
            return fail(format!("{} heads over {} K/V heads", self.n_head, self.n_head_kv));
        }
        if self.mask_token as usize >= self.n_vocab {
            return fail(format!("mask token {} of {} tokens", self.mask_token, self.n_vocab));
        }
        Ok(())
    }

    /// The groups of the dynamic convolution.
    pub fn conv_groups(&self) -> usize {
        self.n_embd / self.conv_group
    }

    /// Values of the coefficients of the dynamic convolution per row: a value
    /// per group, tap and side (before and after the part it surrounds).
    pub fn conv_values(&self) -> usize {
        self.conv_groups() * self.conv_kernel * 2
    }

    /// The values a row's taps hold: the model's residual stream before each
    /// of the target layers.
    pub fn tap_values(&self) -> usize {
        self.target_layers.len() * self.n_embd
    }

    /// Drafts of one pass.
    pub fn drafts(&self) -> usize {
        self.block - 1
    }

    /// Checks that the drafter fits a model of width `n_embd`, `n_layer`
    /// layers and `n_vocab` tokens.
    pub fn fits(&self, n_embd: usize, n_layer: usize, n_vocab: usize) -> Result<()> {
        let fail = |what: String| Err(Error::Target { what });
        if self.n_embd != n_embd {
            return fail(format!("its width is {}, the model's {n_embd}", self.n_embd));
        }
        if let Some(layer) = self.target_layers.iter().find(|l| **l as usize >= n_layer) {
            return fail(format!("it reads layer {layer} of a model of {n_layer} layers"));
        }
        if self.n_vocab != n_vocab {
            return fail(format!("it has {} tokens, the model {n_vocab}", self.n_vocab));
        }
        Ok(())
    }
}

/// The longest block the kernels take.
pub const MAX_BLOCK: usize = 16;
/// The most candidates per position the kernels take.
pub const MAX_TOP_K: usize = 32;

/// One matrix of the drafter as the file stores it.
#[derive(Clone, Debug)]
pub struct Matrix<'g> {
    pub name: String,
    pub ty: GgmlType,
    pub n_in: usize,
    pub n_out: usize,
    pub data: &'g [u8],
}

/// A vector of f32 values of the drafter.
#[derive(Clone, Debug)]
pub struct Vector<'g> {
    pub name: String,
    pub data: &'g [u8],
}

impl Vector<'_> {
    pub fn values(&self) -> Vec<f32> {
        self.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
}

/// The tensors of one layer.
#[derive(Clone, Debug)]
pub struct LayerWeights<'g> {
    pub attn_norm: Vector<'g>,
    pub attn_conv_proj: Matrix<'g>,
    /// `[n_embd, conv_kernel, 2]` f32: the fixed part of the convolution's
    /// weights, per channel, tap and side.
    pub attn_conv_base: Vector<'g>,
    pub q: Matrix<'g>,
    pub k: Matrix<'g>,
    pub v: Matrix<'g>,
    pub q_norm: Vector<'g>,
    pub k_norm: Vector<'g>,
    pub out: Matrix<'g>,
    pub ffn_norm: Vector<'g>,
    pub ffn_conv_proj: Matrix<'g>,
    pub ffn_conv_base: Vector<'g>,
    pub gate: Matrix<'g>,
    pub up: Matrix<'g>,
    pub down: Matrix<'g>,
}

/// The drafter's tensors, borrowed from the mapped file.
#[derive(Clone, Debug)]
pub struct Weights<'g> {
    /// The fusion of the taps: `n_layers_tapped * n_embd -> n_embd`.
    pub fc: Matrix<'g>,
    /// The norm after the fusion.
    pub fc_out_norm: Vector<'g>,
    pub layers: Vec<LayerWeights<'g>>,
    pub output_norm: Vector<'g>,
    /// `n_embd -> selector_rank`: the gate of the selector per position.
    pub selector_hidden: Matrix<'g>,
    /// A row of `selector_rank` values per token: of a candidate, and of the
    /// token before it.
    pub selector_successor: Matrix<'g>,
    pub selector_predecessor: Matrix<'g>,
}

impl<'g> Weights<'g> {
    /// Finds every tensor of `c` in `gguf` and checks its shape; a tensor the
    /// drafter has no use for is refused by name.
    pub fn load(gguf: &'g Gguf, c: &Config) -> Result<Self> {
        let mut used = Vec::new();
        let mut info = |name: &str| -> Result<&'g TensorInfo> {
            used.push(name.to_owned());
            gguf.tensor(name)
                .ok_or_else(|| Error::Tensor { name: name.into(), problem: "is missing".into() })
        };
        let mut matrix = |name: &str, n_in: usize, n_out: usize| -> Result<Matrix<'g>> {
            let t = info(name)?;
            let dims = t.dims.as_slice().to_vec();
            if dims != [n_in as u64, n_out as u64] {
                return Err(Error::Tensor {
                    name: name.into(),
                    problem: format!(
                        "has dims {dims:?}, the configuration needs [{n_in}, {n_out}]"
                    ),
                });
            }
            if !n_in.is_multiple_of(t.ty.block_values() as usize) {
                return Err(Error::Tensor {
                    name: name.into(),
                    problem: format!("is {} in rows of {n_in} values", t.ty),
                });
            }
            Ok(Matrix { name: name.into(), ty: t.ty, n_in, n_out, data: gguf.data(t) })
        };
        let (d, hd) = (c.n_embd, c.head_dim);
        let fc = matrix("fc.weight", c.tap_values(), d)?;
        let selector_hidden = matrix("selector_hidden.weight", d, c.selector_rank)?;
        let selector_successor = matrix("selector_successor.weight", c.selector_rank, c.n_vocab)?;
        let selector_predecessor =
            matrix("selector_predecessor.weight", c.selector_rank, c.n_vocab)?;
        let mut mats = Vec::new();
        for l in 0..c.n_layer {
            let name = |s: &str| format!("blk.{l}.{s}");
            mats.push([
                matrix(&name("attn_conv_proj.weight"), d, c.conv_values())?,
                matrix(&name("attn_q.weight"), d, c.n_head * hd)?,
                matrix(&name("attn_k.weight"), d, c.n_head_kv * hd)?,
                matrix(&name("attn_v.weight"), d, c.n_head_kv * hd)?,
                matrix(&name("attn_output.weight"), c.n_head * hd, d)?,
                matrix(&name("ffn_conv_proj.weight"), d, c.conv_values())?,
                matrix(&name("ffn_gate.weight"), d, c.n_ff)?,
                matrix(&name("ffn_up.weight"), d, c.n_ff)?,
                matrix(&name("ffn_down.weight"), c.n_ff, d)?,
            ]);
        }
        drop(matrix);
        let mut vector = |name: &str, dims: &[u64]| -> Result<Vector<'g>> {
            let t = info(name)?;
            let found = t.dims.as_slice().to_vec();
            if t.ty != GgmlType::F32 || found != dims {
                return Err(Error::Tensor {
                    name: name.into(),
                    problem: format!("is {} {found:?}, the configuration needs F32 {dims:?}", t.ty),
                });
            }
            Ok(Vector { name: name.into(), data: gguf.data(t) })
        };
        let (d64, hd64) = (d as u64, hd as u64);
        let conv = [d64, c.conv_kernel as u64, 2];
        let fc_out_norm = vector("enc.output_norm.weight", &[d64])?;
        let output_norm = vector("output_norm.weight", &[d64])?;
        let mut layers = Vec::with_capacity(c.n_layer);
        for (l, [attn_conv_proj, q, k, v, out, ffn_conv_proj, gate, up, down]) in
            mats.into_iter().enumerate()
        {
            let name = |s: &str| format!("blk.{l}.{s}");
            layers.push(LayerWeights {
                attn_norm: vector(&name("attn_norm.weight"), &[d64])?,
                attn_conv_proj,
                attn_conv_base: vector(&name("attn_conv_base"), &conv)?,
                q,
                k,
                v,
                q_norm: vector(&name("attn_q_norm.weight"), &[hd64])?,
                k_norm: vector(&name("attn_k_norm.weight"), &[hd64])?,
                out,
                ffn_norm: vector(&name("ffn_norm.weight"), &[d64])?,
                ffn_conv_proj,
                ffn_conv_base: vector(&name("ffn_conv_base"), &conv)?,
                gate,
                up,
                down,
            });
        }
        drop(vector);
        if let Some(extra) = gguf.tensors().iter().find(|t| !used.contains(&t.name)) {
            return Err(Error::Variant {
                what: format!("a tensor this implementation does not know ({})", extra.name),
            });
        }
        Ok(Weights {
            fc,
            fc_out_norm,
            layers,
            output_norm,
            selector_hidden,
            selector_successor,
            selector_predecessor,
        })
    }

    /// Every matrix, for sizes and uploads.
    pub fn matrices(&self) -> Vec<&Matrix<'g>> {
        let mut all = vec![
            &self.fc,
            &self.selector_hidden,
            &self.selector_successor,
            &self.selector_predecessor,
        ];
        for l in &self.layers {
            all.extend([
                &l.attn_conv_proj,
                &l.q,
                &l.k,
                &l.v,
                &l.out,
                &l.ffn_conv_proj,
                &l.gate,
                &l.up,
                &l.down,
            ]);
        }
        all
    }

    /// Bytes of every tensor as the file stores it.
    pub fn bytes(&self) -> u64 {
        let vectors = self.layers.len() as u64
            * (self.layers[0].attn_norm.data.len()
                + self.layers[0].attn_conv_base.data.len()
                + self.layers[0].q_norm.data.len()
                + self.layers[0].k_norm.data.len()
                + self.layers[0].ffn_norm.data.len()
                + self.layers[0].ffn_conv_base.data.len()) as u64
            + (self.fc_out_norm.data.len() + self.output_norm.data.len()) as u64;
        self.matrices().iter().map(|m| m.data.len() as u64).sum::<u64>() + vectors
    }
}
