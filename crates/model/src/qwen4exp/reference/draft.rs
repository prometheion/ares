//! The draft block of the model (section 4 of the specification): one layer of
//! the attention kind that guesses one token further than the trunk.
//!
//! Cell `i` of the block pairs the trunk's residual after its last layer at
//! cell `i` ([`Window::trunk_out`](super::Window::trunk_out)) with the token at
//! cell `i + 1`; the largest logit of its row is a guess for the token at cell
//! `i + 2`. A chain goes on at cell `i + 1` from the block's own residual of
//! cell `i` ([`DraftWindow::out`]) and the token it guessed.
//!
//! The block keeps cells of its own ([`DraftState`]): keys and values of every
//! cell and the keys of its indexer, as an attention layer of the trunk does.
//! Rows are run as a [`DraftWindow`], which does not change the state, and a
//! prefix of them is committed with [`Reference::draft_commit`].
//!
//! Where Strata and the fork differ (differences 8 to 10 of the
//! specification): every stream of the residual is normed by itself, as the
//! fork does and as the mixers norm the streams (Strata norms the `D` values
//! as one vector unless told otherwise; on the real model the block guessed
//! 63 of 80 tokens with a norm per stream and 58 with one norm, see
//! `tests/qwen4exp_real.rs`); a cell is rotated to its own index, as Strata
//! does; the cells a row reads are selected by the block's own indexer, as
//! the fork does and as the trunk's attention layers do. Up to the dense
//! limit that is every cell, which is what Strata reads at any length.

use crate::qwen4exp::config::Config;
use crate::qwen4exp::error::{Error, Result};
use crate::qwen4exp::weights::{DraftWeights, Mixer};

use super::ops::matmul;
use super::{
    LayerRows, LayerState, LogitRows, Observer, Reference, Routing, commit_attention, pieces,
};

/// The cells of the draft block of one sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftState {
    position: usize,
    layer: LayerState,
}

impl Default for DraftState {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftState {
    /// The state of a block that has seen no cell.
    pub fn new() -> Self {
        Self {
            position: 0,
            layer: LayerState::Attention {
                k: Vec::new(),
                v: Vec::new(),
                pooled: Vec::new(),
                raw_tail: Vec::new(),
            },
        }
    }

    /// The number of cells the block holds: the cell of its next row.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The keys and values of the block's cells, `HKV * HD` values per cell.
    pub fn kv(&self) -> (&[f32], &[f32]) {
        match &self.layer {
            LayerState::Attention { k, v, .. } => (k, v),
            LayerState::Recurrent { .. } => unreachable!("the block is an attention layer"),
        }
    }

    /// The pooled indexer keys of the block's complete blocks of cells and the
    /// raw keys of the incomplete one.
    pub fn indexer(&self) -> (&[f32], &[f32]) {
        match &self.layer {
            LayerState::Attention { pooled, raw_tail, .. } => (pooled, raw_tail),
            LayerState::Recurrent { .. } => unreachable!("the block is an attention layer"),
        }
    }
}

/// What rows of the draft block compute from a committed state.
#[derive(Clone, Debug)]
pub struct DraftWindow {
    /// The paired tokens of the rows.
    pub tokens: Vec<u32>,
    /// The cell of row 0.
    pub start: usize,
    /// The block's input, `rows * D` values: the projected residual of every
    /// stream plus the projected embedding of the paired token.
    pub input: Vec<f32>,
    /// The block's residual after its mixture of experts, `rows * D` values:
    /// what the next link of a chain starts from.
    pub out: Vec<f32>,
    /// The output of the block's final mixer, `rows * N` values.
    pub final_mix: Vec<f32>,
    /// The rows that have logits, in order.
    pub logit_rows: Vec<usize>,
    /// The logits of those rows, `V` values each, from the model's head.
    pub logits: Vec<f32>,
    /// The picks of the block's router.
    pub routing: Routing,
    rows: LayerRows,
    n_vocab: usize,
}

impl DraftWindow {
    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// The logits of `row`, when it has some.
    pub fn logits_of(&self, row: usize) -> Option<&[f32]> {
        let i = self.logit_rows.iter().position(|&r| r == row)?;
        Some(&self.logits[i * self.n_vocab..(i + 1) * self.n_vocab])
    }

    /// The token a row drafts: the largest of its logits, the lowest token of
    /// equal ones. `None` for a row without logits.
    pub fn draft(&self, row: usize) -> Option<u32> {
        let logits = self.logits_of(row)?;
        let mut best = 0;
        for (token, &logit) in logits.iter().enumerate() {
            if logit > logits[best] {
                best = token;
            }
        }
        Some(best as u32)
    }
}

impl Reference<'_, '_> {
    /// The input of the draft block for `n` rows: `rt` is the residual each
    /// row starts from (`n * D` values: the trunk's after its last layer, or
    /// the block's own of the link before) and `tokens` the paired tokens.
    ///
    /// ```text
    /// e     = fc_embedding (rms(token_embd[token]) * norm_embedding)
    /// hn[c] = rms(rt[c]) * norm_hidden[c]
    /// R[c]  = fc_hidden hn[c] + e
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `rt` is not one row per token or a token is not in the
    /// vocabulary.
    pub fn draft_input(&self, head: &DraftWeights<'_>, rt: &[f32], tokens: &[u32]) -> Vec<f32> {
        let c = self.config;
        let (n, n_embd, wide) = (tokens.len(), c.n_embd, c.wide());
        assert_eq!(rt.len(), n * wide, "rt must be one row of the residual per token");
        let mut x = vec![0.0f32; n_embd];
        let mut en = vec![0.0f32; n * n_embd];
        for (&token, en) in tokens.iter().zip(en.chunks_exact_mut(n_embd)) {
            self.weights.token_embd.decode_row(token as usize, &mut x);
            pieces::rms_norm(&x, &head.norm_embedding, c.rms_eps, en);
        }
        let e = matmul(self.threads, &head.fc_embedding, &en, n);
        let mut hn = vec![0.0f32; n * wide];
        for (src, dst) in rt.chunks_exact(wide).zip(hn.chunks_exact_mut(wide)) {
            pieces::stream_norm(src, &head.norm_hidden, n_embd, c.rms_eps, dst);
        }
        // Every stream is a row of the projection.
        let mut r = matmul(self.threads, &head.fc_hidden, &hn, n * c.n_stream);
        for (row, e) in r.chunks_exact_mut(wide).zip(e.chunks_exact(n_embd)) {
            for stream in row.chunks_exact_mut(n_embd) {
                for (r, e) in stream.iter_mut().zip(e) {
                    *r += e;
                }
            }
        }
        r
    }

    /// Runs rows of the draft block from its committed `state`: row `t` is
    /// cell `state.position() + t`, starts from `rt[t]` (`D` values) and is
    /// paired with `tokens[t]`. Rows of one call see the keys and values of
    /// the rows before them, as the rows of a trunk window do. The state is
    /// not changed.
    pub fn draft_window(
        &self,
        head: &DraftWeights<'_>,
        state: &DraftState,
        rt: &[f32],
        tokens: &[u32],
        logits: LogitRows,
    ) -> Result<DraftWindow> {
        self.draft_window_observed(head, state, rt, tokens, logits, &mut ())
    }

    /// [`Reference::draft_window`] with a hook; the block reports as layer
    /// `config.n_layer`.
    pub fn draft_window_observed(
        &self,
        head: &DraftWeights<'_>,
        state: &DraftState,
        rt: &[f32],
        tokens: &[u32],
        logits: LogitRows,
        observer: &mut dyn Observer,
    ) -> Result<DraftWindow> {
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
        let layer = c.n_layer;
        let Mixer::Attention(attention) = &head.layer.mixer else {
            unreachable!("the draft block is loaded as an attention layer");
        };

        let input = self.draft_input(head, rt, tokens);
        observer.tensor(Some(layer), "draft_in", n, &input);
        let mut r = input.clone();
        let (m, inj) = self.read(&head.layer.attn_mixer, &r, n);
        observer.tensor(Some(layer), "attn_in", n, &m);
        let (out, rows) =
            self.attention(attention, &state.layer, state.position, &m, n, layer, observer);
        observer.tensor(Some(layer), "attn_out", n, &out);
        self.write(&mut r, &out, &inj, n);

        let (m, inj) = self.read(&head.layer.ffn_mixer, &r, n);
        observer.tensor(Some(layer), "ffn_in", n, &m);
        let (out, routing) = self.moe(&head.layer.moe, &m, n, layer, observer);
        observer.tensor(Some(layer), "ffn_out", n, &out);
        self.write(&mut r, &out, &inj, n);
        observer.tensor(Some(layer), "layer_out", n, &r);

        let (final_mix, _) = self.read(&head.output_mixer, &r, n);
        observer.tensor(Some(layer), "final_mix", n, &final_mix);
        let logit_rows: Vec<usize> = match logits {
            LogitRows::None => Vec::new(),
            LogitRows::Last => vec![n - 1],
            LogitRows::All => (0..n).collect(),
        };
        let mut head_in = Vec::with_capacity(logit_rows.len() * c.n_embd);
        for &row in &logit_rows {
            head_in.extend_from_slice(&final_mix[row * c.n_embd..(row + 1) * c.n_embd]);
        }
        let logits = if logit_rows.is_empty() {
            Vec::new()
        } else {
            matmul(self.threads, &self.weights.output, &head_in, logit_rows.len())
        };
        Ok(DraftWindow {
            tokens: tokens.to_vec(),
            start: state.position,
            input,
            out: r,
            final_mix,
            logit_rows,
            logits,
            routing,
            rows,
            n_vocab: c.n_vocab,
        })
    }

    /// Applies the first `accepted` rows of `window` to `state`, which must be
    /// where the window started.
    pub fn draft_commit(
        &self,
        state: &mut DraftState,
        window: &DraftWindow,
        accepted: usize,
    ) -> Result<()> {
        if accepted > window.rows() {
            return Err(Error::Accepted { accepted, rows: window.rows() });
        }
        if window.start != state.position {
            return Err(Error::WindowPosition { window: window.start, state: state.position });
        }
        commit_attention(self.config, &mut state.layer, &window.rows, window.start, accepted);
        state.position += accepted;
        Ok(())
    }

    /// A chain of `links` drafts from the committed `state`: link 0 is the
    /// cell `state.position()`, starts from `rt` (`D` values) and is paired
    /// with `token`; every other link starts from the block's residual of the
    /// link before and is paired with its draft. Returns the drafts. The
    /// state is not changed: the cells of a chain are provisional.
    pub fn draft_chain(
        &self,
        head: &DraftWeights<'_>,
        state: &DraftState,
        rt: &[f32],
        token: u32,
        links: usize,
    ) -> Result<Vec<u32>> {
        let mut state = state.clone();
        let (mut rt, mut token) = (rt.to_vec(), token);
        let mut drafts = Vec::with_capacity(links);
        for _ in 0..links {
            let window = self.draft_window(head, &state, &rt, &[token], LogitRows::All)?;
            self.draft_commit(&mut state, &window, 1)?;
            token = window.draft(0).expect("the row has logits");
            drafts.push(token);
            rt = window.out;
        }
        Ok(drafts)
    }
}

/// The configuration of the draft block of a model as a model of one layer:
/// what a device part that runs the block alone is built from.
pub(crate) fn block_config(c: &Config) -> Config {
    Config { n_layer: 1, attention_layers: vec![true], ngram: None, ..c.clone() }
}
