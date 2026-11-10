//! The next-token-prediction block (multi-token prediction): the block a
//! model carries after its own blocks to draft the token after next.
//!
//! The block is an attention block with its own keys and values and one more
//! input. At position `p` it reads the embedding of token `x_p` and the hidden
//! state the model computed at position `p - 1` (the output of its final
//! norm, from which `x_p` was picked), and predicts `x_{p+1}`:
//!
//! ```text
//! e'   = rms_norm(embedding(x_p); enorm)       h' = rms_norm(h_{p-1}; hnorm)
//! x    = eh_proj [e' ; h']                     (embedding first)
//! x    = x + attention(rms_norm(x; attn_norm)) (its own K/V cache, position p)
//! x    = x + ffn(rms_norm(x; post_attention_norm))
//! out  = rms_norm(x; shared_head_norm)         (output_norm when it has none)
//! logits = head @ out
//! ```
//!
//! Before position 0 there is no hidden state; the block reads zeros there.
//!
//! Its keys and values at a position depend on that position's input only, so
//! there are two ways it is run, as in llama.cpp (`common/speculative.cpp`,
//! `common_speculative_impl_draft_mtp`):
//!
//! - **Following the model** ([`Reference::mtp_rows`], then
//!   [`Reference::mtp_commit`]): every position the model has computed, with
//!   the model's own hidden state. This fills the block's cache.
//! - **Drafting** ([`Reference::mtp_draft`]): from the newest token `x_p` and
//!   `h_{p-1}` it predicts `x_{p+1}`, then goes on alone: link `j` reads the
//!   token link `j - 1` picked and, in place of the model's hidden state, the
//!   block's own `out` of link `j - 1`. Nothing a chain computes is kept.

use crate::qwen35::error::{Error, Result};
use crate::qwen35::reference::ops::rms_norm;
use crate::qwen35::reference::{LayerRows, Reference};
use crate::qwen35::weights::{Matrix, NextnWeights};

/// What the block keeps of a sequence: its keys and values of every position
/// the model has committed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MtpState {
    k: Vec<f32>,
    v: Vec<f32>,
    position: usize,
}

impl MtpState {
    /// The state of a sequence that has not seen a token.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of positions the block has followed.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The keys and the values, `[position][kv head][dim]`.
    pub fn kv(&self) -> (&[f32], &[f32]) {
        (&self.k, &self.v)
    }
}

/// What the block computes for consecutive positions.
#[derive(Clone, Debug)]
pub struct MtpRows {
    /// The position of row 0.
    pub start: usize,
    /// The block's input, the output of `eh_proj`: `rows * n_embd` values.
    pub projected: Vec<f32>,
    /// The block's output after its head norm, `rows * n_embd` values: what
    /// the next link of a chain reads as its hidden state.
    pub hidden: Vec<f32>,
    /// The logits of every row, `n_vocab` values each; empty when not asked
    /// for.
    pub logits: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    n_embd: usize,
}

impl MtpRows {
    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.hidden.len() / self.n_embd
    }

    /// The keys and values of the rows, `[row][kv head][dim]`.
    pub fn kv(&self) -> (&[f32], &[f32]) {
        (&self.k, &self.v)
    }
}

/// One link of a draft chain.
#[derive(Clone, Debug)]
pub struct MtpDraft {
    /// The drafted token: the most probable of the draft vocabulary.
    pub token: u32,
    /// Its logit.
    pub logit: f32,
    /// What the link computed (one row, with logits).
    pub rows: MtpRows,
}

impl<'g> Reference<'_, 'g> {
    /// The block's weights, when the model has one.
    pub fn mtp_weights(&self) -> Result<&NextnWeights<'g>> {
        self.weights.nextn.as_ref().ok_or_else(|| Error::Inconsistent {
            what: "the model has no next-token-prediction block".to_owned(),
        })
    }

    fn mtp_embedding(&self) -> Result<&Matrix<'g>> {
        Ok(self.mtp_weights()?.embed_tokens.as_ref().unwrap_or(&self.weights.token_embd))
    }

    /// Runs the block on consecutive positions from `state.position()`: row
    /// `i` reads `tokens[i]` and `hidden[i]` (`n_embd` values: the model's
    /// final-norm output of the position before, or zeros before position 0).
    /// The state is not changed.
    pub fn mtp_rows(
        &self,
        state: &MtpState,
        tokens: &[u32],
        hidden: &[f32],
        logits: bool,
    ) -> Result<MtpRows> {
        let c = self.config;
        let w = self.mtp_weights()?;
        let (n, d) = (tokens.len(), c.n_embd);
        if n == 0 {
            return Err(Error::NoTokens);
        }
        if let Some(&token) = tokens.iter().find(|&&t| t as usize >= c.n_vocab) {
            return Err(Error::Token { token, n_vocab: c.n_vocab });
        }
        assert_eq!(hidden.len(), n * d, "one hidden state of n_embd values per token");

        // [e' ; h'] per row.
        let embedding = self.mtp_embedding()?;
        let mut joined = vec![0.0f32; n * 2 * d];
        let mut e = vec![0.0f32; d];
        for (row, &token) in tokens.iter().enumerate() {
            embedding.decode_row(token as usize, &mut e);
            let (e_out, h_out) = joined[row * 2 * d..(row + 1) * 2 * d].split_at_mut(d);
            rms_norm(&e, &w.enorm, c.rms_eps, e_out);
            rms_norm(&hidden[row * d..(row + 1) * d], &w.hnorm, c.rms_eps, h_out);
        }
        let x = self.matmul(&w.eh_proj, &joined, n);

        let normed = self.norm_rows(&x, &w.block.attn_norm);
        let (mixed, kept) =
            self.attention(&w.attention, &state.k, &state.v, state.position, &normed, n);
        let LayerRows::Attention { k, v } = kept else {
            unreachable!("an attention block keeps keys and values")
        };
        let after: Vec<f32> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let normed = self.norm_rows(&after, &w.block.post_attention_norm);
        let ffn = self.ffn(&w.block, &normed, n);
        let out: Vec<f32> = after.iter().zip(&ffn).map(|(a, b)| a + b).collect();

        let head_norm = w.shared_head_norm.as_deref().unwrap_or(&self.weights.output_norm);
        let hidden_out = self.norm_rows(&out, head_norm);
        let logits = if logits {
            let head = w.shared_head_head.as_ref().unwrap_or(self.weights.head());
            self.matmul(head, &hidden_out, n)
        } else {
            Vec::new()
        };
        Ok(MtpRows {
            start: state.position,
            projected: x,
            hidden: hidden_out,
            logits,
            k,
            v,
            n_embd: d,
        })
    }

    /// Appends the keys and values of the first `accepted` rows to `state`,
    /// which must be where the rows started.
    pub fn mtp_commit(&self, state: &mut MtpState, rows: &MtpRows, accepted: usize) -> Result<()> {
        if accepted > rows.rows() {
            return Err(Error::Accepted { accepted, rows: rows.rows() });
        }
        if rows.start != state.position {
            return Err(Error::WindowPosition { window: rows.start, state: state.position });
        }
        let kv_dim = self.config.attention_dims().1;
        state.k.extend_from_slice(&rows.k[..accepted * kv_dim]);
        state.v.extend_from_slice(&rows.v[..accepted * kv_dim]);
        state.position += accepted;
        Ok(())
    }

    /// Drafts `len` tokens after `token`, the newest token of the sequence at
    /// position `state.position()`, which the model picked from `hidden`.
    /// Link 0 predicts the token after `token`; each further link reads the
    /// token and the output of the link before. A link's pick is the token
    /// with the largest logit among `vocabulary` (all tokens when `None`),
    /// the lowest token on a tie. The state is not changed.
    pub fn mtp_draft(
        &self,
        state: &MtpState,
        token: u32,
        hidden: &[f32],
        len: usize,
        vocabulary: Option<&[u32]>,
    ) -> Result<Vec<MtpDraft>> {
        let mut state = state.clone();
        let mut token = token;
        let mut hidden = hidden.to_vec();
        let mut chain = Vec::with_capacity(len);
        for _ in 0..len {
            let rows = self.mtp_rows(&state, &[token], &hidden, true)?;
            self.mtp_commit(&mut state, &rows, 1)?;
            let mut best = (f32::NEG_INFINITY, u32::MAX);
            let mut consider = |t: u32| {
                let l = rows.logits[t as usize];
                if l > best.0 || (l == best.0 && t < best.1) {
                    best = (l, t);
                }
            };
            match vocabulary {
                Some(list) => list.iter().copied().for_each(&mut consider),
                None => (0..self.config.n_vocab as u32).for_each(&mut consider),
            }
            token = best.1;
            hidden.clone_from(&rows.hidden);
            chain.push(MtpDraft { token, logit: best.0, rows });
        }
        Ok(chain)
    }
}
