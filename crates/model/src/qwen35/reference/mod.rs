//! A CPU implementation of the forward pass of `qwen35`, written to be read
//! next to `docs/reference/qwen35-architecture.md` and to be right.
//!
//! It is an oracle, not an engine: clarity and exactness come before speed. The
//! arithmetic is the architecture's own, in `f32` storage with `f64` sums and
//! transcendental functions (see `ops.rs`); it does not imitate the rounding of
//! any other implementation. Weights are decoded with
//! [`ares_quant::dequantize`] row by row when they are multiplied.
//!
//! # Windows and state
//!
//! The state of a sequence is explicit ([`SequenceState`]): per recurrent block
//! the convolution window and the delta-rule matrices, per attention block the
//! keys and values of every position. Rows are run as a [`Window`] with
//! [`Reference::window`], which starts from the committed state and does not
//! change it. [`Reference::commit`] then applies the first `accepted` rows of a
//! window to the state: their keys and values are appended and the recurrent
//! state is advanced over exactly those rows, which is what the engine does
//! after verifying a draft ("A step" in `docs/ARCHITECTURE.md`). Rejected rows
//! leave no trace. [`Reference::forward`] is a window that is committed whole.
//!
//! A row gives the same bits whether it is run alone, in a window with others,
//! or after a commit of the rows before it, because every sum has one fixed
//! order.

mod blocks;
mod mtp;
mod ops;

pub use mtp::{MtpDraft, MtpRows, MtpState};

/// Pieces of the reference that GPU kernels are held to one by one: the
/// functions of the recurrent block (`delta_step`, `conv_silu`) and the scalar
/// operations with their `f64` arithmetic.
pub mod pieces {
    pub use super::blocks::{conv_silu, delta_step};
    pub use super::ops::{l2_norm, rms_norm, rms_scale, sigmoid, silu, softplus};
}

use crate::qwen35::config::Config;
use crate::qwen35::error::{Error, Result};
use crate::qwen35::weights::{Matrix, Mixer, Weights};

use ops::{dot_f32, par_ranges, rms_norm, widen};

/// What a sequence carries from token to token.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceState {
    position: usize,
    layers: Vec<LayerState>,
}

#[derive(Clone, Debug, PartialEq)]
enum LayerState {
    /// `conv`: `conv_dim` channels of `conv_kernel - 1` inputs, oldest first.
    /// `delta`: per value head a matrix of `head_v_dim` rows of `head_k_dim`.
    Recurrent { conv: Vec<f32>, delta: Vec<f32> },
    /// Keys and values of every position, `n_head_kv * head_dim` values each.
    Attention { k: Vec<f32>, v: Vec<f32> },
}

impl SequenceState {
    /// The state of a sequence that has not seen a token.
    pub fn new(config: &Config) -> Self {
        let r = &config.recurrent;
        let layers = (0..config.n_layer)
            .map(|layer| {
                if config.is_attention(layer) {
                    LayerState::Attention { k: Vec::new(), v: Vec::new() }
                } else {
                    LayerState::Recurrent {
                        conv: vec![0.0; r.conv_dim() * (r.conv_kernel - 1)],
                        delta: vec![0.0; r.n_v_heads * r.head_v_dim * r.head_k_dim],
                    }
                }
            })
            .collect();
        Self { position: 0, layers }
    }

    /// The number of tokens the state has seen: the position of the next one.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Rounds the delta-rule matrices of every recurrent block to `storage` and
    /// back, once: what saving the state as a checkpoint in that format and
    /// resuming from it does.
    pub fn round_delta(&mut self, config: &Config, storage: StateStorage) {
        let r = &config.recurrent;
        for layer in &mut self.layers {
            if let LayerState::Recurrent { delta, .. } = layer {
                storage.round(delta, r.head_v_dim, r.head_k_dim);
            }
        }
    }

    /// The convolution window of a recurrent block: per channel the last
    /// `conv_kernel - 1` inputs of the convolution, oldest first. `None` for an
    /// attention block.
    pub fn conv(&self, layer: usize) -> Option<&[f32]> {
        match &self.layers[layer] {
            LayerState::Recurrent { conv, .. } => Some(conv),
            LayerState::Attention { .. } => None,
        }
    }

    /// The delta-rule matrices of a recurrent block, `[v head][v index][k index]`.
    /// `None` for an attention block.
    pub fn delta(&self, layer: usize) -> Option<&[f32]> {
        match &self.layers[layer] {
            LayerState::Recurrent { delta, .. } => Some(delta),
            LayerState::Attention { .. } => None,
        }
    }

    /// The keys and the values of an attention block, `[position][kv head][dim]`.
    /// `None` for a recurrent block.
    pub fn kv(&self, layer: usize) -> Option<(&[f32], &[f32])> {
        match &self.layers[layer] {
            LayerState::Attention { k, v } => Some((k, v)),
            LayerState::Recurrent { .. } => None,
        }
    }
}

/// Which rows of a window get logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitRows {
    /// None: only the hidden states are wanted.
    None,
    /// The last row only.
    Last,
    /// Every row.
    All,
}

/// What a window keeps of a block so that a prefix of its rows can be committed
/// without running the matrices again.
#[derive(Clone, Debug)]
enum LayerRows {
    /// Per row: the projection before the convolution (`conv_dim`), the
    /// normalised query and key (`key_dim` each), the value (`value_dim`), and
    /// the decay input `g` and `beta` of each value head.
    Recurrent { qkv: Vec<f32>, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, g: Vec<f32>, beta: Vec<f32> },
    /// Per row: the key and the value as they enter the cache.
    Attention { k: Vec<f32>, v: Vec<f32> },
}

/// The result of running rows from a committed state.
#[derive(Clone, Debug)]
pub struct Window {
    /// The tokens of the rows.
    pub tokens: Vec<u32>,
    /// The position of row 0.
    pub start: usize,
    /// The hidden state after every block: `hidden[block]` has `rows * n_embd`
    /// values. The input of the first block is the embedding of the token.
    pub hidden: Vec<Vec<f32>>,
    /// The output of the final norm, `rows * n_embd` values.
    pub final_norm: Vec<f32>,
    /// The rows that have logits, in order.
    pub logit_rows: Vec<usize>,
    /// The logits of those rows, `n_vocab` values each.
    pub logits: Vec<f32>,
    layers: Vec<LayerRows>,
    n_embd: usize,
    n_vocab: usize,
}

impl Window {
    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// The hidden state of `row` after block `block`.
    pub fn hidden_row(&self, block: usize, row: usize) -> &[f32] {
        &self.hidden[block][row * self.n_embd..(row + 1) * self.n_embd]
    }

    /// The logits of `row`, when it has some.
    pub fn logits_of(&self, row: usize) -> Option<&[f32]> {
        let i = self.logit_rows.iter().position(|&r| r == row)?;
        Some(&self.logits[i * self.n_vocab..(i + 1) * self.n_vocab])
    }
}

/// How the delta-rule matrices are held between steps. The arithmetic of a step
/// is the same in every case; the format only rounds the matrices after every
/// row that advances them (in a window and in a commit alike, so that a row
/// does not depend on the step that computes it), which is what storing the
/// state in a smaller type costs.
/// Used to measure that cost against `F32`, the exact state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateStorage {
    /// `f32`, as the reference and llama.cpp hold it.
    F32,
    /// IEEE half precision, rounded to nearest even.
    F16,
    /// `bfloat16`, rounded to nearest even.
    Bf16,
    /// `int8` with one `f32` scale per value head (`max |S| / 127`).
    Int8PerHead,
    /// `int8` with one `f32` scale per row of a matrix (one value index of one head).
    Int8PerRow,
}

impl StateStorage {
    /// Rounds `matrices` (`[head][value index][key index]`, `head_dim` per
    /// row, `rows_per_head` rows) to the format and back.
    pub fn round(self, matrices: &mut [f32], rows_per_head: usize, head_dim: usize) {
        match self {
            StateStorage::F32 => {}
            StateStorage::F16 => {
                for v in matrices.iter_mut() {
                    *v = ares_quant::f16_to_f32(ares_quant::f32_to_f16(*v));
                }
            }
            StateStorage::Bf16 => {
                for v in matrices.iter_mut() {
                    *v = ares_quant::bf16_to_f32(ares_quant::f32_to_bf16(*v));
                }
            }
            StateStorage::Int8PerHead | StateStorage::Int8PerRow => {
                let group = if self == StateStorage::Int8PerHead {
                    rows_per_head * head_dim
                } else {
                    head_dim
                };
                for chunk in matrices.chunks_mut(group) {
                    let max = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                    if max == 0.0 {
                        continue;
                    }
                    let scale = max / 127.0;
                    for v in chunk.iter_mut() {
                        *v = (*v / scale).round().clamp(-127.0, 127.0) * scale;
                    }
                }
            }
        }
    }
}

/// What a window takes besides its tokens ([`Reference::window_with`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inputs {
    /// Rows of the window whose input is these `n_embd` values in place of
    /// their token's embedding (an image's rows).
    pub rows: Vec<(usize, Vec<f32>)>,
    /// Per row of the window its rotary position `(time, height, width)`;
    /// `None`: the row's cell for all three.
    pub rotary: Option<Vec<[usize; 3]>>,
}

/// The forward pass of a model.
#[derive(Debug)]
pub struct Reference<'a, 'g> {
    config: &'a Config,
    weights: &'a Weights<'g>,
    threads: usize,
    storage: StateStorage,
}

impl<'a, 'g> Reference<'a, 'g> {
    /// A reference for `weights`, which were loaded for `config`. It uses as
    /// many threads as the machine has.
    pub fn new(config: &'a Config, weights: &'a Weights<'g>) -> Self {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self { config, weights, threads, storage: StateStorage::F32 }
    }

    /// Uses `threads` threads for the large multiplications and the heads. The
    /// result does not depend on the number.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Holds the delta-rule matrices in `storage` between commits (default
    /// `F32`): after a commit advanced them they are rounded to that format.
    pub fn state_storage(mut self, storage: StateStorage) -> Self {
        self.storage = storage;
        self
    }

    /// The configuration.
    pub fn config(&self) -> &Config {
        self.config
    }

    /// Runs `tokens` as consecutive rows from the committed `state` and returns
    /// what they compute. The state is not changed.
    pub fn window(
        &self,
        state: &SequenceState,
        tokens: &[u32],
        logits: LogitRows,
    ) -> Result<Window> {
        self.window_with(state, tokens, &Inputs::default(), logits)
    }

    /// [`Reference::window`] with rows that are not token embeddings and
    /// rotary positions that are not the rows' cells (an image's rows, and
    /// the text after an image): see [`Inputs`].
    pub fn window_with(
        &self,
        state: &SequenceState,
        tokens: &[u32],
        inputs: &Inputs,
        logits: LogitRows,
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

        let mut x = vec![0.0f32; n * c.n_embd];
        for (row, &token) in tokens.iter().enumerate() {
            self.weights
                .token_embd
                .decode_row(token as usize, &mut x[row * c.n_embd..(row + 1) * c.n_embd]);
        }
        for (row, values) in &inputs.rows {
            assert!(*row < n && values.len() == c.n_embd, "an input row of the window");
            x[row * c.n_embd..(row + 1) * c.n_embd].copy_from_slice(values);
        }
        if let Some(rotary) = &inputs.rotary {
            assert_eq!(rotary.len(), n, "a rotary position per row");
        }

        let mut hidden = Vec::with_capacity(c.n_layer);
        let mut rows = Vec::with_capacity(c.n_layer);
        for layer in 0..c.n_layer {
            let (out, kept) = self.run_block(layer, state, &x, n, inputs.rotary.as_deref());
            x = out;
            hidden.push(x.clone());
            rows.push(kept);
        }

        let final_norm = self.norm_rows(&x, &self.weights.output_norm);
        let logit_rows: Vec<usize> = match logits {
            LogitRows::None => Vec::new(),
            LogitRows::Last => vec![n - 1],
            LogitRows::All => (0..n).collect(),
        };
        let mut head_in = Vec::with_capacity(logit_rows.len() * c.n_embd);
        for &row in &logit_rows {
            head_in.extend_from_slice(&final_norm[row * c.n_embd..(row + 1) * c.n_embd]);
        }
        let logits = if logit_rows.is_empty() {
            Vec::new()
        } else {
            self.matmul(self.weights.head(), &head_in, logit_rows.len())
        };

        Ok(Window {
            tokens: tokens.to_vec(),
            start: state.position,
            hidden,
            final_norm,
            logit_rows,
            logits,
            layers: rows,
            n_embd: c.n_embd,
            n_vocab: c.n_vocab,
        })
    }

    /// Runs block `layer` alone on `x`, the hidden states of consecutive rows
    /// that enter it (`rows * n_embd` values), as rows that follow the committed
    /// `state`, and returns the hidden states that leave it. This is the unit a
    /// GPU implementation is compared with block by block: give both the same
    /// input and compare the outputs. The state is not changed.
    ///
    /// # Panics
    ///
    /// Panics when `layer` is out of range or `x` is not whole rows.
    pub fn block(&self, layer: usize, state: &SequenceState, x: &[f32]) -> Vec<f32> {
        assert!(layer < self.config.n_layer, "the model has {} blocks", self.config.n_layer);
        assert!(
            x.len().is_multiple_of(self.config.n_embd),
            "x must be whole rows of n_embd values"
        );
        self.run_block(layer, state, x, x.len() / self.config.n_embd, None).0
    }

    fn run_block(
        &self,
        layer: usize,
        state: &SequenceState,
        x: &[f32],
        n: usize,
        rotary: Option<&[[usize; 3]]>,
    ) -> (Vec<f32>, LayerRows) {
        let weights = &self.weights.layers[layer];
        let normed = self.norm_rows(x, &weights.block.attn_norm);
        let (mixed, kept) = match (&weights.mixer, &state.layers[layer]) {
            (Mixer::Recurrent(w), LayerState::Recurrent { conv, delta }) => {
                self.recurrent(w, conv, delta, &normed, n)
            }
            (Mixer::Attention(w), LayerState::Attention { k, v }) => {
                self.attention(w, k, v, state.position, &normed, n, rotary)
            }
            _ => unreachable!("the state was built for this configuration"),
        };
        let after_mixer: Vec<f32> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let normed = self.norm_rows(&after_mixer, &weights.block.post_attention_norm);
        let ffn = self.ffn(&weights.block, &normed, n);
        (after_mixer.iter().zip(&ffn).map(|(a, b)| a + b).collect(), kept)
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
        let r = &self.config.recurrent;
        let (kv_dim, kk) = (self.config.attention_dims().1, r.conv_kernel - 1);
        for (layer_state, rows) in state.layers.iter_mut().zip(&window.layers) {
            match (layer_state, rows) {
                (LayerState::Attention { k, v }, LayerRows::Attention { k: wk, v: wv }) => {
                    k.extend_from_slice(&wk[..accepted * kv_dim]);
                    v.extend_from_slice(&wv[..accepted * kv_dim]);
                }
                (
                    LayerState::Recurrent { conv, delta },
                    LayerRows::Recurrent { qkv, q, k, v, g, beta },
                ) => {
                    let cd = r.conv_dim();
                    // The new window is the last `kk` columns of the old window
                    // followed by the accepted inputs.
                    let old = conv.clone();
                    for c in 0..cd {
                        for i in 0..kk {
                            let at = accepted + i;
                            conv[c * kk + i] =
                                if at < kk { old[c * kk + at] } else { qkv[(at - kk) * cd + c] };
                        }
                    }
                    for t in 0..accepted {
                        blocks::advance_delta(r, delta, q, k, v, g, beta, t);
                        self.storage.round(delta, r.head_v_dim, r.head_k_dim);
                    }
                }
                _ => unreachable!("the window was run on a state of this configuration"),
            }
        }
        state.position += accepted;
        Ok(())
    }

    /// Runs `tokens` from `state` and commits all of them.
    pub fn forward(
        &self,
        state: &mut SequenceState,
        tokens: &[u32],
        logits: LogitRows,
    ) -> Result<Window> {
        let window = self.window(state, tokens, logits)?;
        self.commit(state, &window, window.rows())?;
        Ok(window)
    }

    /// `rows` of `n_embd` values, each through an RMS norm with `weight`.
    fn norm_rows(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        for (src, dst) in x.chunks_exact(weight.len()).zip(out.chunks_exact_mut(weight.len())) {
            rms_norm(src, weight, self.config.rms_eps, dst);
        }
        out
    }

    /// `y[t] = W x[t]` for `n` rows: row `t` of `x` has `w.cols()` values and
    /// row `t` of the result `w.rows()`.
    fn matmul(&self, w: &Matrix<'_>, x: &[f32], n: usize) -> Vec<f32> {
        let (rows, cols) = (w.rows(), w.cols());
        assert_eq!(x.len(), n * cols);
        let xs = widen(x);
        // `[output row][token]` from the workers, transposed below.
        let by_row = par_ranges(self.threads, rows, cols * n.max(1), |range| {
            let mut stored = vec![0.0f32; cols];
            let mut out = Vec::with_capacity(range.len() * n);
            for r in range {
                w.decode_row(r, &mut stored);
                for t in 0..n {
                    out.push(dot_f32(&xs[t * cols..(t + 1) * cols], &stored) as f32);
                }
            }
            out
        });
        if n == 1 {
            return by_row;
        }
        let mut y = vec![0.0f32; n * rows];
        for (r, per_token) in by_row.chunks_exact(n).enumerate() {
            for (t, &value) in per_token.iter().enumerate() {
                y[t * rows + r] = value;
            }
        }
        y
    }

    /// The feed-forward part: `down(silu(gate(x)) * up(x))`.
    fn ffn(&self, block: &crate::qwen35::weights::Block<'_>, x: &[f32], n: usize) -> Vec<f32> {
        let gate = self.matmul(&block.ffn_gate, x, n);
        let up = self.matmul(&block.ffn_up, x, n);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| (ops::silu(f64::from(g)) * f64::from(u)) as f32)
            .collect();
        self.matmul(&block.ffn_down, &act, n)
    }
}
