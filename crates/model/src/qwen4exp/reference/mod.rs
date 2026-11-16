//! A CPU implementation of the forward pass of `qwen4exp`, written to be read
//! next to `docs/reference/qwen4exp-architecture.md` (section 3) and to be
//! right.
//!
//! It is an oracle, not an engine: clarity and exactness come before speed.
//! The arithmetic is the architecture's own, in `f32` storage with `f64` sums
//! and transcendental functions (see `ops.rs`), as in the `qwen35` reference;
//! it does not imitate the rounding of any other implementation. Weights are
//! decoded with [`ares_quant::dequantize`] row by row when they are
//! multiplied. Where Strata and the fork differ the reference follows Strata
//! (section 5); the one difference that changes numbers, the selection rule
//! past the dense limit, is a parameter ([`Reference::selection_rule`]).
//!
//! # Windows and state
//!
//! The state of a sequence is explicit ([`SequenceState`]): per recurrent
//! layer the convolution window and the delta-rule matrices; per attention
//! layer the keys and values of every cell, the pooled indexer key of every
//! complete block and the raw indexer keys of the incomplete one; the history
//! of the n-gram block's convolution and the newest tokens for its hash.
//! Rows are run as a [`Window`] with [`Reference::window`], which starts from
//! the committed state and does not change it. [`Reference::commit`] then
//! applies the first `accepted` rows of a window to the state, which is what
//! the engine does after verifying a draft. Rejected rows leave no trace.
//! [`Reference::forward`] is a window that is committed whole.
//!
//! A row gives the same bits whether it is run alone, in a window with
//! others, or after a commit of the rows before it, because every sum has one
//! fixed order.
//!
//! Only text is implemented: the rotary position of a cell is its index, and
//! every cell hashes its own token. The MTP MTP block (section 4) is in
//! `draft.rs`: the mixer, attention and expert functions of this module with
//! the block's own weights, on what [`Window::trunk_out`] holds.

mod blocks;
mod mtp;
mod ops;

pub(crate) use mtp::mtp_config;
pub use mtp::{MtpState, MtpWindow};

/// The operations of the reference that GPU kernels are held to one by one.
/// Each is a pure function over slices with one documented order of
/// summation.
pub mod pieces {
    pub use super::blocks::{
        attend, block_score, decay, expert_act, gated_head_norm, mixer_mix, mixer_squash,
        mixer_write, moe_combine, ngram_gate, ngram_gated, ngram_rows, ngram_update, norm_rope,
        pool_key, rope, rope_table, route, select_cells, shared_gate, stream_norm,
    };
    pub use super::ops::{
        conv_dilated, dot, l2_norm, matmul, rms_norm, rms_scale, sigmoid, silu, softplus, widen,
    };
    pub use crate::qwen35::reference::pieces::{conv_silu, delta_step};
}

use std::collections::BTreeMap;

use crate::qwen4exp::config::{Config, SelectionRule};
use crate::qwen4exp::error::{Error, Result};
use crate::qwen4exp::weights::{
    AttentionWeights, Mixer, Moe, NgramWeights, RecurrentWeights, StreamMixer, Weights,
};

use ops::{matmul, par_ranges};
use pieces::{conv_silu, delta_step, l2_norm};

/// What a sequence carries from token to token (section 6 of the
/// specification).
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceState {
    position: usize,
    /// The tokens of the newest cells, oldest first, at most `ngram_size - 1`.
    recent: Vec<u32>,
    /// `[history][D]`, the normed gated values of the newest cells, oldest
    /// first; zero before the start of the sequence.
    ngram_history: Vec<f32>,
    layers: Vec<LayerState>,
}

#[derive(Clone, Debug, PartialEq)]
enum LayerState {
    /// `conv`: `CG` channels of `conv_kernel - 1` inputs, oldest first.
    /// `delta`: per value head a matrix `[value index][key index]`.
    Recurrent { conv: Vec<f32>, delta: Vec<f32> },
    /// `k`, `v`: per cell `HKV * HD` values. `pooled`: per complete block `ID`
    /// values. `raw_tail`: the raw indexer keys of the cells of the incomplete
    /// block, `ID` values each.
    Attention { k: Vec<f32>, v: Vec<f32>, pooled: Vec<f32>, raw_tail: Vec<f32> },
}

impl SequenceState {
    /// The state of a sequence that has not seen a token.
    pub fn new(config: &Config) -> Self {
        let r = &config.recurrent;
        let layers = (0..config.n_layer)
            .map(|layer| {
                if config.is_attention(layer) {
                    LayerState::Attention {
                        k: Vec::new(),
                        v: Vec::new(),
                        pooled: Vec::new(),
                        raw_tail: Vec::new(),
                    }
                } else {
                    LayerState::Recurrent {
                        conv: vec![0.0; r.conv_dim() * (r.conv_kernel - 1)],
                        delta: vec![0.0; r.n_v_heads * r.head_dim * r.head_dim],
                    }
                }
            })
            .collect();
        let history = config.ngram.as_ref().map_or(0, |g| g.history() * config.wide());
        Self { position: 0, recent: Vec::new(), ngram_history: vec![0.0; history], layers }
    }

    /// The number of cells the state holds: the cell of the next token.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The tokens of the newest cells that the n-gram hash of the next cell
    /// reads, oldest first (at most `ngram_size - 1`).
    pub fn recent_tokens(&self) -> &[u32] {
        &self.recent
    }

    /// The history of the n-gram block's convolution: `[history][D]`, the
    /// normed gated values of the newest cells, oldest first.
    pub fn ngram_history(&self) -> &[f32] {
        &self.ngram_history
    }

    /// The convolution window of a recurrent layer: per channel the last
    /// `conv_kernel - 1` inputs of the convolution, oldest first. `None` for
    /// an attention layer.
    pub fn conv(&self, layer: usize) -> Option<&[f32]> {
        match &self.layers[layer] {
            LayerState::Recurrent { conv, .. } => Some(conv),
            LayerState::Attention { .. } => None,
        }
    }

    /// The delta-rule matrices of a recurrent layer, `[value head][value
    /// index][key index]`. `None` for an attention layer.
    pub fn delta(&self, layer: usize) -> Option<&[f32]> {
        match &self.layers[layer] {
            LayerState::Recurrent { delta, .. } => Some(delta),
            LayerState::Attention { .. } => None,
        }
    }

    /// The keys (after norm and rotation) and the values of an attention
    /// layer, `[cell][kv head][dim]`. `None` for a recurrent layer.
    pub fn kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match &self.layers[layer] {
            LayerState::Attention { k, v, .. } => Some((k, v)),
            LayerState::Recurrent { .. } => None,
        }
    }

    /// The indexer state of an attention layer: the pooled keys of the
    /// complete blocks (`[block][ID]`) and the raw keys of the cells of the
    /// incomplete block (`[cell][ID]`, empty when the cell count is a multiple
    /// of the block size). `None` for a recurrent layer.
    pub fn indexer(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match &self.layers[layer] {
            LayerState::Attention { pooled, raw_tail, .. } => Some((pooled, raw_tail)),
            LayerState::Recurrent { .. } => None,
        }
    }
}

/// Which rows of a window get logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitRows {
    /// None: only the residual is wanted.
    None,
    /// The last row only.
    Last,
    /// Every row.
    All,
}

/// A hook that sees what the reference computes while it computes it. Every
/// method does nothing by default; `()` is the hook that records nothing and
/// [`Recorder`] one that keeps everything.
pub trait Observer {
    /// An intermediate result of `rows` rows, `data.len() / rows` values
    /// each. `layer` is `None` for what belongs to no layer. The names:
    ///
    /// | Name | Width | What |
    /// | --- | --- | --- |
    /// | `embedding` | `N` | the token's row of `token_embd` |
    /// | `ngram_embedding` | `N` | the gathered table rows, side by side |
    /// | `ngram_out` | `D` | the residual after the n-gram block |
    /// | `attn_in` | `N` | the read in front of the recurrent or attention part |
    /// | `attn_out` | `N` | the output of the recurrent or attention part |
    /// | `ffn_in` | `N` | the read in front of the mixture of experts |
    /// | `ffn_out` | `N` | the output of the mixture of experts |
    /// | `layer_out` | `D` | the residual after the layer |
    /// | `final_mix` | `N` | the read in front of the output head |
    fn tensor(&mut self, _layer: Option<usize>, _name: &'static str, _rows: usize, _data: &[f32]) {}

    /// The experts the router picked for `row` in `layer`, in rank order, and
    /// their weights.
    fn routing(&mut self, _layer: usize, _row: usize, _ids: &[u32], _weights: &[f32]) {}

    /// The cells `row` attends to in attention layer `layer`, ascending, and
    /// the scores of the blocks the row sees (empty while the row is within
    /// the dense limit and nothing is scored).
    fn selection(&mut self, _layer: usize, _row: usize, _cells: &[u32], _scores: &[f32]) {}
}

impl Observer for () {}

/// The layer (`None` for what belongs to no layer) and the name of an
/// intermediate result.
pub type TensorKey = (Option<usize>, &'static str);

/// An [`Observer`] that keeps everything it is shown. Use one per window: a
/// second window's results replace those of the first under the same keys.
#[derive(Clone, Debug, Default)]
pub struct Recorder {
    /// Every intermediate result by `(layer, name)`, with its row count.
    pub tensors: BTreeMap<TensorKey, (usize, Vec<f32>)>,
    /// `(layer, row)` to the routed expert ids and weights.
    pub routing: BTreeMap<(usize, usize), (Vec<u32>, Vec<f32>)>,
    /// `(layer, row)` to the selected cells and the block scores.
    pub selection: BTreeMap<(usize, usize), (Vec<u32>, Vec<f32>)>,
}

impl Recorder {
    /// The rows of the intermediate result `name` of `layer`.
    pub fn tensor_of(&self, layer: Option<usize>, name: &'static str) -> Option<&[f32]> {
        self.tensors.get(&(layer, name)).map(|(_, data)| data.as_slice())
    }
}

impl Observer for Recorder {
    fn tensor(&mut self, layer: Option<usize>, name: &'static str, rows: usize, data: &[f32]) {
        self.tensors.insert((layer, name), (rows, data.to_vec()));
    }

    fn routing(&mut self, layer: usize, row: usize, ids: &[u32], weights: &[f32]) {
        self.routing.insert((layer, row), (ids.to_vec(), weights.to_vec()));
    }

    fn selection(&mut self, layer: usize, row: usize, cells: &[u32], scores: &[f32]) {
        self.selection.insert((layer, row), (cells.to_vec(), scores.to_vec()));
    }
}

/// The router's picks of one layer for the rows of a window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Routing {
    /// Per row the `K` expert ids in rank order.
    pub ids: Vec<u32>,
    /// Per row their `K` weights.
    pub weights: Vec<f32>,
}

/// What a window keeps of a layer so that a prefix of its rows can be
/// committed without running the matrices again.
#[derive(Clone, Debug)]
enum LayerRows {
    /// Per row: the projection before the convolution (`CG`), the normalised
    /// query and key, the value, and the decay exponent `g` and `beta` of each
    /// value head.
    Recurrent { qkv: Vec<f32>, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, g: Vec<f32>, beta: Vec<f32> },
    /// Per row: the key and the value as they enter the cache and the raw
    /// indexer key; `pooled`: the keys of the blocks the window completes, in
    /// block order.
    Attention { k: Vec<f32>, v: Vec<f32>, raw: Vec<f32>, pooled: Vec<f32> },
}

/// The result of running rows from a committed state.
#[derive(Clone, Debug)]
pub struct Window {
    /// The tokens of the rows.
    pub tokens: Vec<u32>,
    /// The cell of row 0.
    pub start: usize,
    /// The residual after every layer, when the reference keeps it
    /// ([`Reference::keep_residual`]): `residual[layer]` has `rows * D`
    /// values, stream `c` of a row at `[c * N, (c + 1) * N)`.
    pub residual: Vec<Vec<f32>>,
    /// The residual after the last layer, `rows * D` values: what the final
    /// mixer reads and what the MTP MTP block starts from.
    pub trunk_out: Vec<f32>,
    /// The output of the final mixer, `rows * N` values.
    pub final_mix: Vec<f32>,
    /// The rows that have logits, in order.
    pub logit_rows: Vec<usize>,
    /// The logits of those rows, `V` values each.
    pub logits: Vec<f32>,
    /// The router's picks, per layer.
    pub routing: Vec<Routing>,
    layers: Vec<LayerRows>,
    /// Per row the normed gated value of the n-gram block (`D`).
    ngram: Vec<f32>,
    wide: usize,
    n_vocab: usize,
}

impl Window {
    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// The residual of `row` after layer `layer` (`D` values), when it was
    /// kept.
    pub fn residual_row(&self, layer: usize, row: usize) -> Option<&[f32]> {
        self.residual.get(layer).map(|r| &r[row * self.wide..(row + 1) * self.wide])
    }

    /// The logits of `row`, when it has some.
    pub fn logits_of(&self, row: usize) -> Option<&[f32]> {
        let i = self.logit_rows.iter().position(|&r| r == row)?;
        Some(&self.logits[i * self.n_vocab..(i + 1) * self.n_vocab])
    }
}

/// The forward pass of a model.
#[derive(Debug)]
pub struct Reference<'a, 'g> {
    config: &'a Config,
    weights: &'a Weights<'g>,
    threads: usize,
    rule: SelectionRule,
    keep_residual: bool,
    depth: usize,
}

impl<'a, 'g> Reference<'a, 'g> {
    /// A reference for `weights`, which were loaded for `config`. It uses as
    /// many threads as the machine has, the selection rule of the
    /// specification, and keeps the residual after every layer.
    pub fn new(config: &'a Config, weights: &'a Weights<'g>) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            config,
            weights,
            threads,
            rule: SelectionRule::default(),
            keep_residual: true,
            depth: usize::MAX,
        }
    }

    /// Uses `threads` threads for the large multiplications, the experts and
    /// the heads. The result does not depend on the number.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// The rule by which a row past the dense limit selects cells (difference
    /// 1 of the specification).
    pub fn selection_rule(mut self, rule: SelectionRule) -> Self {
        self.rule = rule;
        self
    }

    /// Whether windows keep the residual after every layer
    /// ([`Window::residual`]); a row of a large model is megabytes.
    pub fn keep_residual(mut self, keep: bool) -> Self {
        self.keep_residual = keep;
        self
    }

    /// Runs only the first `layers` layers of every window (all of them by
    /// default; at least through the layer of the n-gram block). A commit of
    /// such a window advances the state of those layers only, so what the
    /// first layers compute and select over a long text can be read without
    /// the cost of the others. Residual, final mix and logits of such a
    /// window are not the model's.
    pub fn first_layers(mut self, layers: usize) -> Self {
        self.depth = layers;
        self
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        self.config
    }

    /// Runs `tokens` as consecutive rows from the committed `state` and
    /// returns what they compute. The state is not changed.
    pub fn window(
        &self,
        state: &SequenceState,
        tokens: &[u32],
        logits: LogitRows,
    ) -> Result<Window> {
        self.window_observed(state, tokens, logits, &mut ())
    }

    /// [`Reference::window`] with a hook that is shown the intermediate
    /// results, the routing and the selected cells.
    pub fn window_observed(
        &self,
        state: &SequenceState,
        tokens: &[u32],
        logits: LogitRows,
        observer: &mut dyn Observer,
    ) -> Result<Window> {
        let c = self.config;
        let n = tokens.len();
        if n == 0 {
            return Err(Error::NoTokens);
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= c.n_vocab) {
            return Err(Error::Token { token, n_vocab: c.n_vocab });
        }
        let end = state.position + n;
        if end > c.n_ctx {
            return Err(Error::Context { end, limit: c.n_ctx });
        }
        let (n_embd, wide) = (c.n_embd, c.wide());

        // 3.1: the embedding, copied to every stream.
        let mut x = vec![0.0f32; n * n_embd];
        for (row, &token) in tokens.iter().enumerate() {
            self.weights
                .token_embd
                .decode_row(token as usize, &mut x[row * n_embd..(row + 1) * n_embd]);
        }
        observer.tensor(None, "embedding", n, &x);
        let mut r = vec![0.0f32; n * wide];
        for (row, streams) in r.chunks_exact_mut(wide).enumerate() {
            for stream in streams.chunks_exact_mut(n_embd) {
                stream.copy_from_slice(&x[row * n_embd..(row + 1) * n_embd]);
            }
        }

        let mut residual = Vec::new();
        let mut layers = Vec::with_capacity(c.n_layer);
        let mut routing = Vec::with_capacity(c.n_layer);
        let mut ngram = Vec::new();
        for layer in 0..c.n_layer.min(self.depth) {
            let kept = self.run_layer(layer, state, tokens, &mut r, observer);
            if self.keep_residual {
                residual.push(r.clone());
            }
            layers.push(kept.rows);
            routing.push(kept.routing);
            if let Some(nrm) = kept.ngram {
                ngram = nrm;
            }
        }

        // 3.8: the final mixer and the head.
        let (final_mix, _) = self.read(&self.weights.output_mixer, &r, n);
        observer.tensor(None, "final_mix", n, &final_mix);
        let logit_rows: Vec<usize> = match logits {
            LogitRows::None => Vec::new(),
            LogitRows::Last => vec![n - 1],
            LogitRows::All => (0..n).collect(),
        };
        let mut head_in = Vec::with_capacity(logit_rows.len() * n_embd);
        for &row in &logit_rows {
            head_in.extend_from_slice(&final_mix[row * n_embd..(row + 1) * n_embd]);
        }
        let logits = if logit_rows.is_empty() {
            Vec::new()
        } else {
            matmul(self.threads, &self.weights.output, &head_in, logit_rows.len())
        };

        Ok(Window {
            tokens: tokens.to_vec(),
            start: state.position,
            residual,
            trunk_out: r,
            final_mix,
            logit_rows,
            logits,
            routing,
            layers,
            ngram,
            wide,
            n_vocab: c.n_vocab,
        })
    }

    /// Runs layer `layer` alone on `r`, the residual of consecutive rows that
    /// enters it (`tokens.len() * D` values), as rows that follow the
    /// committed `state`, and returns the residual that leaves it. `tokens`
    /// are the rows' tokens, which the layer with the n-gram block hashes.
    /// This is the unit a GPU implementation is compared with layer by layer:
    /// give both the same input and compare the outputs. The state is not
    /// changed.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or `r` is not one row per token.
    pub fn block(
        &self,
        layer: usize,
        state: &SequenceState,
        tokens: &[u32],
        r: &[f32],
    ) -> Vec<f32> {
        self.block_observed(layer, state, tokens, r, &mut ())
    }

    /// [`Reference::block`] with a hook.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or `r` is not one row per token.
    pub fn block_observed(
        &self,
        layer: usize,
        state: &SequenceState,
        tokens: &[u32],
        r: &[f32],
        observer: &mut dyn Observer,
    ) -> Vec<f32> {
        assert!(layer < self.config.n_layer, "the model has {} layers", self.config.n_layer);
        assert_eq!(r.len(), tokens.len() * self.config.wide(), "r must be one row per token");
        let mut out = r.to_vec();
        self.run_layer(layer, state, tokens, &mut out, observer);
        out
    }

    /// What the mixture of experts of `layer` would read if `r` (`n` rows of
    /// `D` values) were the residual in front of it: the read of the layer's
    /// second mixer, `n` rows of `N` values. With the residual of an earlier
    /// point of the model this is the input of a router applied early, which
    /// is what the measurements of read-ahead predictors compare.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or `r` is not `n` rows.
    pub fn expert_input(&self, layer: usize, r: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(r.len(), n * self.config.wide(), "r must be {n} rows of the residual");
        self.read(&self.weights.layers[layer].ffn_mixer, r, n).0
    }

    /// The logits of the router of `layer` for the rows `m` (`n` rows of `N`
    /// values), `n` rows of one logit per expert.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or `m` is not `n` rows.
    pub fn router_logits(&self, layer: usize, m: &[f32], n: usize) -> Vec<f32> {
        matmul(self.threads, &self.weights.layers[layer].moe.router, m, n)
    }

    /// The residual behind the first half of `layer`: `r`, the residual that
    /// enters the layer (behind the n-gram block, in the layer that has one),
    /// with `core`, the output of the layer's recurrent or attention part
    /// (`n` rows of `N` values), written back through the layer's first mixer.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or the rows are not `n`.
    pub fn after_core(&self, layer: usize, r: &[f32], core: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(r.len(), n * self.config.wide(), "r must be {n} rows of the residual");
        let (_, inj) = self.read(&self.weights.layers[layer].attn_mixer, r, n);
        let mut out = r.to_vec();
        self.write(&mut out, core, &inj, n);
        out
    }

    /// One layer over the rows of `r`, in place (section 1 of the
    /// specification).
    fn run_layer(
        &self,
        layer: usize,
        state: &SequenceState,
        tokens: &[u32],
        r: &mut [f32],
        observer: &mut dyn Observer,
    ) -> LayerKept {
        let c = self.config;
        let (n, n_embd, wide) = (tokens.len(), c.n_embd, c.wide());
        let weights = &self.weights.layers[layer];

        let mut ngram = None;
        if let Some(w) = &weights.ngram {
            ngram = Some(self.ngram_block(w, state, tokens, r, observer));
            observer.tensor(Some(layer), "ngram_out", n, r);
        }

        let (m, inj) = self.read(&weights.attn_mixer, r, n);
        observer.tensor(Some(layer), "attn_in", n, &m);
        let (out, rows) = match (&weights.mixer, &state.layers[layer]) {
            (Mixer::Recurrent(w), LayerState::Recurrent { conv, delta }) => {
                self.recurrent(w, conv, delta, &m, n)
            }
            (Mixer::Attention(w), layer_state @ LayerState::Attention { .. }) => {
                self.attention(w, layer_state, state.position, &m, n, layer, observer)
            }
            _ => unreachable!("the state was built for this configuration"),
        };
        observer.tensor(Some(layer), "attn_out", n, &out);
        self.write(r, &out, &inj, n);

        let (m, inj) = self.read(&weights.ffn_mixer, r, n);
        observer.tensor(Some(layer), "ffn_in", n, &m);
        let (out, routing) = self.moe(&weights.moe, &m, n, layer, observer);
        observer.tensor(Some(layer), "ffn_out", n, &out);
        self.write(r, &out, &inj, n);
        observer.tensor(Some(layer), "layer_out", n, r);
        debug_assert_eq!(r.len(), n * wide);
        debug_assert_eq!(out.len(), n * n_embd);
        LayerKept { rows, routing, ngram }
    }

    /// Applies the first `accepted` rows of `window` to `state`, which must be
    /// where the window started. Rows past `accepted` are dropped.
    pub fn commit(
        &self,
        state: &mut SequenceState,
        window: &Window,
        accepted: usize,
    ) -> Result<()> {
        if accepted > window.rows() {
            return Err(Error::Accepted { accepted, rows: window.rows() });
        }
        if window.start != state.position {
            return Err(Error::WindowPosition { window: window.start, state: state.position });
        }
        let c = self.config;
        let r = &c.recurrent;
        let kk = r.conv_kernel - 1;
        let (start, end) = (state.position, state.position + accepted);
        for (layer_state, rows) in state.layers.iter_mut().zip(&window.layers) {
            match (layer_state, rows) {
                (state @ LayerState::Attention { .. }, rows @ LayerRows::Attention { .. }) => {
                    commit_attention(c, state, rows, start, accepted);
                }
                (
                    LayerState::Recurrent { conv, delta },
                    LayerRows::Recurrent { qkv, q, k, v, g, beta },
                ) => {
                    let cd = r.conv_dim();
                    // The new window is the last `kk` columns of the old window
                    // followed by the accepted inputs.
                    let old = conv.clone();
                    for ch in 0..cd {
                        for i in 0..kk {
                            let at = accepted + i;
                            conv[ch * kk + i] =
                                if at < kk { old[ch * kk + at] } else { qkv[(at - kk) * cd + ch] };
                        }
                    }
                    let (s, kd, vd, nv) = (r.head_dim, r.key_dim(), r.value_dim(), r.n_v_heads);
                    for t in 0..accepted {
                        for h in 0..nv {
                            let kh = h % r.n_k_heads;
                            delta_step(
                                &mut delta[h * s * s..(h + 1) * s * s],
                                &q[t * kd + kh * s..][..s],
                                &k[t * kd + kh * s..][..s],
                                &v[t * vd + h * s..][..s],
                                g[t * nv + h],
                                beta[t * nv + h],
                                None,
                            );
                        }
                    }
                }
                _ => unreachable!("the window was run on a state of this configuration"),
            }
        }
        if let Some(ngram) = &c.ngram {
            let (wide, history) = (c.wide(), ngram.history());
            let mut all = std::mem::take(&mut state.ngram_history);
            all.extend_from_slice(&window.ngram[..accepted * wide]);
            state.ngram_history = all[all.len() - history * wide..].to_vec();
            state.recent.extend_from_slice(&window.tokens[..accepted]);
            let keep = state.recent.len().min(ngram.ngram_size - 1);
            state.recent.drain(..state.recent.len() - keep);
        }
        state.position = end;
        Ok(())
    }

    /// Runs `tokens` from `state` and commits all of them.
    pub fn forward(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        logits: LogitRows,
    ) -> Result<Window> {
        self.forward_observed(state, tokens, logits, &mut ())
    }

    /// [`Reference::forward`] with a hook.
    pub fn forward_observed(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        logits: LogitRows,
        observer: &mut dyn Observer,
    ) -> Result<Window> {
        let window = self.window_observed(state, tokens, logits, observer)?;
        self.commit(state, &window, window.rows())?;
        Ok(window)
    }

    /// 3.3, the read of a stream mixer over `n` rows of the residual: the
    /// mixed input of a block (`[n, N]`) and the inject values (`[n, hc]`,
    /// empty for a mixer without inject).
    fn read(&self, mixer: &StreamMixer<'_>, r: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
        let c = self.config;
        let (n_embd, wide, hc) = (c.n_embd, c.wide(), c.n_stream);
        let mut xn = vec![0.0f32; n * wide];
        for (src, dst) in r.chunks_exact(wide).zip(xn.chunks_exact_mut(wide)) {
            pieces::stream_norm(src, &mixer.norm, n_embd, c.rms_eps, dst);
        }
        let down = matmul(self.threads, &mixer.down, &xn, n);
        let mut lo = vec![0.0f32; down.len()];
        pieces::mixer_squash(&down, hc, &mut lo);
        let gate = matmul(self.threads, &mixer.up, &lo, n);
        let mut m = vec![0.0f32; n * n_embd];
        for ((xn, gate), m) in
            xn.chunks_exact(wide).zip(gate.chunks_exact(wide)).zip(m.chunks_exact_mut(n_embd))
        {
            pieces::mixer_mix(xn, gate, m);
        }
        let inj = match &mixer.inject {
            Some(inject) => matmul(self.threads, inject, &xn, n),
            None => Vec::new(),
        };
        (m, inj)
    }

    /// 3.3, the write of a block's output (`[n, N]`) back to the residual.
    fn write(&self, r: &mut [f32], out: &[f32], inj: &[f32], n: usize) {
        let c = self.config;
        let (n_embd, wide, hc) = (c.n_embd, c.wide(), c.n_stream);
        for t in 0..n {
            pieces::mixer_write(
                &mut r[t * wide..(t + 1) * wide],
                &out[t * n_embd..(t + 1) * n_embd],
                &inj[t * hc..(t + 1) * hc],
            );
        }
    }

    /// 3.4, the n-gram block: updates the residual in place and returns the
    /// rows' normed gated values (`[n, D]`), which a commit appends to the
    /// history.
    fn ngram_block(
        &self,
        w: &NgramWeights<'_>,
        state: &SequenceState,
        tokens: &[u32],
        r: &mut [f32],
        observer: &mut dyn Observer,
    ) -> Vec<f32> {
        let c = self.config;
        let (Some(ngram), Some(table)) = (&c.ngram, &self.weights.ngram_table) else {
            unreachable!("a layer has n-gram weights only when the model has the table");
        };
        let (n, n_embd, wide, hc) = (tokens.len(), c.n_embd, c.wide(), c.n_stream);

        // The hash on the host and the gather of the rows.
        let mut before = state.recent.clone();
        let mut emb = vec![0.0f32; n * n_embd];
        for (t, &token) in tokens.iter().enumerate() {
            let rows = pieces::ngram_rows(ngram, token, &before);
            for (h, &row) in rows.iter().enumerate() {
                let at = t * n_embd + h * ngram.row_dim;
                table.decode_row(row, &mut emb[at..at + ngram.row_dim]);
            }
            before.push(token);
        }
        observer.tensor(Some(ngram.layer), "ngram_embedding", n, &emb);

        let key = matmul(self.threads, &w.key, &emb, n);
        let value = matmul(self.threads, &w.value, &emb, n);
        let mut gated = vec![0.0f32; n * wide];
        let mut nrm = vec![0.0f32; n * wide];
        let mut gate = vec![0.0f32; hc];
        for t in 0..n {
            let row = t * wide..(t + 1) * wide;
            pieces::ngram_gate(
                &key[row.clone()],
                &r[row.clone()],
                &w.norm_key,
                &w.norm_query,
                c.rms_eps,
                &mut gate,
            );
            pieces::ngram_gated(
                &value[t * n_embd..(t + 1) * n_embd],
                &gate,
                &mut gated[row.clone()],
            );
            pieces::stream_norm(
                &gated[row.clone()],
                &w.norm_conv,
                n_embd,
                c.rms_eps,
                &mut nrm[row],
            );
        }
        let conv = pieces::conv_dilated(
            &w.conv,
            ngram.conv_kernel,
            ngram.ngram_size,
            &state.ngram_history,
            &nrm,
            wide,
            n,
        );
        pieces::ngram_update(r, &gated, &conv);
        nrm
    }

    /// 3.5, a recurrent layer over `n` rows `x` (`[n, N]`), from the committed
    /// convolution window `conv` and matrices `delta`. Returns the layer's
    /// output (`[n, N]`) and what a commit needs.
    fn recurrent(
        &self,
        w: &RecurrentWeights<'_>,
        conv: &[f32],
        delta: &[f32],
        x: &[f32],
        n: usize,
    ) -> (Vec<f32>, LayerRows) {
        let c = self.config;
        let r = &c.recurrent;
        let (s, nv) = (r.head_dim, r.n_v_heads);
        let (kd, vd, cd, kk) = (r.key_dim(), r.value_dim(), r.conv_dim(), r.conv_kernel - 1);

        let qkv = matmul(self.threads, &w.qkv, x, n); // [n, CG]
        let z = matmul(self.threads, &w.gate, x, n); // [n, GV * S]
        let beta_in = matmul(self.threads, &w.beta, x, n); // [n, GV]
        let alpha = matmul(self.threads, &w.alpha, x, n); // [n, GV]

        let beta: Vec<f32> =
            beta_in.iter().map(|&b| pieces::sigmoid(f64::from(b)) as f32).collect();
        let g: Vec<f32> = alpha
            .iter()
            .enumerate()
            .map(|(i, &a)| pieces::decay(a, w.dt_bias[i % nv], w.a[i % nv]))
            .collect();

        // The causal convolution over the committed inputs and the window's,
        // then silu over the whole vector, before the split.
        let mixed = conv_silu(&w.conv, kk, conv, &qkv, cd, n);
        let mut q = vec![0.0f32; n * kd];
        let mut k = vec![0.0f32; n * kd];
        let mut v = vec![0.0f32; n * vd];
        for t in 0..n {
            let row = &mixed[t * cd..(t + 1) * cd];
            for h in 0..r.n_k_heads {
                let at = t * kd + h * s;
                l2_norm(&row[h * s..(h + 1) * s], c.rms_eps, &mut q[at..at + s]);
                l2_norm(&row[kd + h * s..kd + (h + 1) * s], c.rms_eps, &mut k[at..at + s]);
            }
            v[t * vd..(t + 1) * vd].copy_from_slice(&row[2 * kd..]);
        }

        // The delta rule, one value head per work item, from the committed
        // matrices. Value head h reads key head h % GK.
        let per_head = par_ranges(self.threads, nv, n * s * s * 3, |heads| {
            let mut out = Vec::with_capacity(heads.len() * n * s);
            for h in heads {
                let mut m = delta[h * s * s..(h + 1) * s * s].to_vec();
                let kh = h % r.n_k_heads;
                for t in 0..n {
                    let mut o = vec![0.0f32; s];
                    delta_step(
                        &mut m,
                        &q[t * kd + kh * s..][..s],
                        &k[t * kd + kh * s..][..s],
                        &v[t * vd + h * s..][..s],
                        g[t * nv + h],
                        beta[t * nv + h],
                        Some(&mut o),
                    );
                    out.extend_from_slice(&o);
                }
            }
            out
        });

        let mut gated = vec![0.0f32; n * vd];
        for h in 0..nv {
            for t in 0..n {
                let at = t * vd + h * s;
                pieces::gated_head_norm(
                    &per_head[(h * n + t) * s..(h * n + t + 1) * s],
                    &w.norm,
                    &z[at..at + s],
                    c.rms_eps,
                    &mut gated[at..at + s],
                );
            }
        }
        let out = matmul(self.threads, &w.out, &gated, n);
        (out, LayerRows::Recurrent { qkv, q, k, v, g, beta })
    }

    /// 3.6, an attention layer over `n` rows `x` (`[n, N]`) at cells
    /// `start .. start + n`, after `start` committed cells.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        w: &AttentionWeights<'_>,
        state: &LayerState,
        start: usize,
        x: &[f32],
        n: usize,
        layer: usize,
        observer: &mut dyn Observer,
    ) -> (Vec<f32>, LayerRows) {
        let LayerState::Attention { k: k_cache, v: v_cache, pooled: pooled_cache, raw_tail } =
            state
        else {
            unreachable!("called for an attention layer");
        };
        let c = self.config;
        let (a, ix) = (&c.attention, &c.indexer);
        let (hd, id, block) = (a.head_dim, ix.key_dim, ix.block);
        let (q_dim, kv_dim) = c.attention_dims();
        let u_dim = ix.n_head * id;

        let qg = matmul(self.threads, &w.q, x, n); // [n, H * 2 * HD]
        let k_in = matmul(self.threads, &w.k, x, n); // [n, HKV * HD]
        let v = matmul(self.threads, &w.v, x, n);
        let iq = matmul(self.threads, &w.indexer_q, x, n); // [n, IH * ID]
        let raw = matmul(self.threads, &w.indexer_k, x, n); // [n, ID]

        // Per head the projection holds the query, then the gate. Queries,
        // keys and indexer queries are normed and rotated to the row's cell.
        let mut q = vec![0.0f32; n * q_dim];
        let mut gate = vec![0.0f32; n * q_dim];
        let mut k = vec![0.0f32; n * kv_dim];
        let mut u = vec![0.0f32; n * u_dim];
        for t in 0..n {
            let table = pieces::rope_table(&a.rope, text_position(start + t));
            for h in 0..a.n_head {
                let src = &qg[t * 2 * q_dim + h * 2 * hd..][..2 * hd];
                let at = t * q_dim + h * hd;
                pieces::norm_rope(&src[..hd], &w.q_norm, c.rms_eps, &table, &mut q[at..at + hd]);
                gate[at..at + hd].copy_from_slice(&src[hd..]);
            }
            for h in 0..a.n_head_kv {
                let at = t * kv_dim + h * hd;
                pieces::norm_rope(
                    &k_in[at..at + hd],
                    &w.k_norm,
                    c.rms_eps,
                    &table,
                    &mut k[at..at + hd],
                );
            }
            for e in 0..ix.n_head {
                let at = t * u_dim + e * id;
                pieces::norm_rope(
                    &iq[at..at + id],
                    &w.indexer_q_norm,
                    c.rms_eps,
                    &table,
                    &mut u[at..at + id],
                );
            }
        }

        // The pooled keys of the blocks this window completes, from the
        // carried raw keys and the window's, rotated to the block's first cell.
        let known = start / block;
        let complete = (start + n) / block;
        let mut pooled = vec![0.0f32; (complete - known) * id];
        let mut members = vec![0.0f32; block * id];
        for j in known..complete {
            for (i, member) in members.chunks_exact_mut(id).enumerate() {
                member.copy_from_slice(raw_key(raw_tail, &raw, start, block, id, j * block + i));
            }
            let table = pieces::rope_table(&a.rope, text_position(j * block));
            let at = (j - known) * id;
            pieces::pool_key(
                &members,
                &w.indexer_k_norm,
                c.rms_eps,
                &table,
                &mut pooled[at..at + id],
            );
        }
        let pooled_key = |j: usize| -> &[f32] {
            if j < known {
                &pooled_cache[j * id..(j + 1) * id]
            } else {
                &pooled[(j - known) * id..(j - known + 1) * id]
            }
        };

        // Which cells each row reads: all it sees up to the dense limit, a
        // selection by block scores past it.
        let mut cells = Vec::with_capacity(n);
        for t in 0..n {
            let visible = start + t + 1;
            let scores: Vec<f32> = if visible > ix.dense_limit() {
                let row = &u[t * u_dim..(t + 1) * u_dim];
                (0..visible / block).map(|j| pieces::block_score(row, pooled_key(j))).collect()
            } else {
                Vec::new()
            };
            let selected = pieces::select_cells(self.rule, ix, visible, &scores);
            observer.selection(layer, t, &selected, &scores);
            cells.push(selected);
        }

        // Attention over the listed cells: query head h reads KV head
        // h / (H / HKV).
        let group = a.n_head / a.n_head_kv;
        let (k_all, v_all, cells) = (&k, &v, &cells);
        let longest = cells.iter().map(Vec::len).max().unwrap_or(0);
        let heads = par_ranges(self.threads, n * a.n_head, longest * hd * 2, |items| {
            let mut out = vec![0.0f32; items.len() * hd];
            for (item, o) in items.zip(out.chunks_exact_mut(hd)) {
                let (t, h) = (item / a.n_head, item % a.n_head);
                let kv_head = h / group;
                let key = |cell: u32| -> &[f32] {
                    let cell = cell as usize;
                    if cell < start {
                        &k_cache[cell * kv_dim + kv_head * hd..][..hd]
                    } else {
                        &k_all[(cell - start) * kv_dim + kv_head * hd..][..hd]
                    }
                };
                let value = |cell: u32| -> &[f32] {
                    let cell = cell as usize;
                    if cell < start {
                        &v_cache[cell * kv_dim + kv_head * hd..][..hd]
                    } else {
                        &v_all[(cell - start) * kv_dim + kv_head * hd..][..hd]
                    }
                };
                let at = t * q_dim + h * hd;
                pieces::attend(&q[at..at + hd], &gate[at..at + hd], &cells[t], key, value, o);
            }
            out
        });
        let out = matmul(self.threads, &w.out, &heads, n);
        (out, LayerRows::Attention { k, v, raw, pooled })
    }

    /// 3.7, the mixture of experts over `n` rows `m` (`[n, N]`). An expert's
    /// matrices are fetched from the file for the rows that picked it.
    fn moe(
        &self,
        w: &Moe<'_>,
        m: &[f32],
        n: usize,
        layer: usize,
        observer: &mut dyn Observer,
    ) -> (Vec<f32>, Routing) {
        let c = self.config;
        let (n_embd, e) = (c.n_embd, &c.experts);
        let k = e.used;

        let logits = matmul(self.threads, &w.router, m, n); // [n, E]
        let mut routing = Routing { ids: vec![0; n * k], weights: vec![0.0; n * k] };
        // Expert to the (row, rank) pairs that picked it.
        let mut users: BTreeMap<u32, Vec<(usize, usize)>> = BTreeMap::new();
        for t in 0..n {
            let (ids, weights) =
                (&mut routing.ids[t * k..(t + 1) * k], &mut routing.weights[t * k..(t + 1) * k]);
            pieces::route(&logits[t * e.count..(t + 1) * e.count], ids, weights);
            observer.routing(layer, t, ids, weights);
            for (rank, &id) in ids.iter().enumerate() {
                users.entry(id).or_default().push((t, rank));
            }
        }

        // f(e) = down_e (silu(gate_e m) * (up_e m)) for every (row, expert)
        // pair, one expert at a time.
        let users: Vec<(u32, Vec<(usize, usize)>)> = users.into_iter().collect();
        let cost = 3 * n_embd * e.ff;
        let outputs = par_ranges(self.threads, users.len(), cost, |range| {
            let mut out = Vec::new();
            for (id, pairs) in &users[range] {
                let mut x = Vec::with_capacity(pairs.len() * n_embd);
                for &(t, _) in pairs {
                    x.extend_from_slice(&m[t * n_embd..(t + 1) * n_embd]);
                }
                let id = *id as usize;
                out.extend(expert(
                    &w.gate.expert(id),
                    &w.up.expert(id),
                    &w.down.expert(id),
                    &x,
                    pairs.len(),
                ));
            }
            out
        });
        let mut routed = vec![0.0f32; n * k * n_embd];
        let mut at = 0;
        for (_, pairs) in &users {
            for &(t, rank) in pairs {
                routed[(t * k + rank) * n_embd..][..n_embd]
                    .copy_from_slice(&outputs[at..at + n_embd]);
                at += n_embd;
            }
        }

        // The shared expert on every row, added with its scalar gate.
        let gate = matmul(self.threads, &w.shared_gate, m, n);
        let up = matmul(self.threads, &w.shared_up, m, n);
        let mut act = vec![0.0f32; gate.len()];
        pieces::expert_act(&gate, &up, &mut act);
        let shared = matmul(self.threads, &w.shared_down, &act, n);

        let mut out = vec![0.0f32; n * n_embd];
        for t in 0..n {
            let row = t * n_embd..(t + 1) * n_embd;
            let sh = pieces::shared_gate(&w.shared_gate_inp, &m[row.clone()]);
            pieces::moe_combine(
                &routed[t * k * n_embd..(t + 1) * k * n_embd],
                &routing.weights[t * k..(t + 1) * k],
                &shared[row.clone()],
                sh,
                &mut out[row],
            );
        }
        (out, routing)
    }
}

/// Applies the first `accepted` rows of an attention layer's window, which
/// started after `start` cells, to the layer's state.
fn commit_attention(
    c: &Config,
    state: &mut LayerState,
    rows: &LayerRows,
    start: usize,
    accepted: usize,
) {
    let (
        LayerState::Attention { k, v, pooled, raw_tail },
        LayerRows::Attention { k: wk, v: wv, raw, pooled: w_pooled },
    ) = (state, rows)
    else {
        unreachable!("called for an attention layer and its window");
    };
    let kv_dim = c.attention_dims().1;
    let (block, id) = (c.indexer.block, c.indexer.key_dim);
    let end = start + accepted;
    k.extend_from_slice(&wk[..accepted * kv_dim]);
    v.extend_from_slice(&wv[..accepted * kv_dim]);
    // The pooled keys of the blocks the accepted rows complete.
    let new_blocks = end / block - start / block;
    pooled.extend_from_slice(&w_pooled[..new_blocks * id]);
    // The raw keys of the cells of the block that is now incomplete.
    let first = end / block * block;
    let mut tail = Vec::with_capacity((end - first) * id);
    for cell in first..end {
        tail.extend_from_slice(raw_key(raw_tail, raw, start, block, id, cell));
    }
    *raw_tail = tail;
}

/// What `run_layer` hands back besides the residual.
struct LayerKept {
    rows: LayerRows,
    routing: Routing,
    ngram: Option<Vec<f32>>,
}

/// The rotary position triple `(t, h, w)` of a text cell: its index, three
/// times.
fn text_position(cell: usize) -> [f64; 3] {
    [cell as f64; 3]
}

/// The raw indexer key of `cell`, which lies in the incomplete block carried
/// by the state (`carried`, the keys of cells `start / block * block ..
/// start`) or in the window (`rows`, the keys of cells `start ..`).
fn raw_key<'s>(
    carried: &'s [f32],
    rows: &'s [f32],
    start: usize,
    block: usize,
    id: usize,
    cell: usize,
) -> &'s [f32] {
    if cell < start {
        let at = cell - start / block * block;
        &carried[at * id..(at + 1) * id]
    } else {
        &rows[(cell - start) * id..(cell - start + 1) * id]
    }
}

/// One routed expert over `n` rows `x` (`[n, N]`): `down (silu(gate x) * (up
/// x))`, `[n, N]`, on the calling thread.
fn expert(
    gate: &crate::qwen4exp::weights::Matrix<'_>,
    up: &crate::qwen4exp::weights::Matrix<'_>,
    down: &crate::qwen4exp::weights::Matrix<'_>,
    x: &[f32],
    n: usize,
) -> Vec<f32> {
    let g = matmul(1, gate, x, n);
    let u = matmul(1, up, x, n);
    let mut act = vec![0.0f32; g.len()];
    pieces::expert_act(&g, &u, &mut act);
    matmul(1, down, &act, n)
}
