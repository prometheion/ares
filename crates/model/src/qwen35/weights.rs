//! The weights of a `qwen35` model: which tensors there are, of which shape,
//! and where their bytes are.
//!
//! The table is built against a [`Config`] and is exact: a tensor that is
//! missing, has other dimensions than the configuration gives it, cannot be
//! decoded, or is in the file without being part of the architecture is an
//! error that names it. Matrices stay as the stored blocks (borrowed from the
//! file) and are decoded row by row when they are used; the small vectors
//! (norms, time-step parameters, the convolution) are decoded to `f32` once.

use ares_gguf::Gguf;

pub use crate::files::Matrix;
use crate::files::Table;
use crate::qwen35::config::Config;
use crate::qwen35::error::{Error, Result};

/// What a tensor of the file that the table does not take is not part of.
const OF: &str = "the qwen35 architecture";

/// The weights shared by every block: the two norms and the feed-forward part.
#[derive(Clone, Debug)]
pub struct Block<'g> {
    /// RMS norm weight in front of the recurrent or attention part
    /// (`attn_norm.weight`).
    pub attn_norm: Vec<f32>,
    /// RMS norm weight in front of the feed-forward part
    /// (`post_attention_norm.weight`).
    pub post_attention_norm: Vec<f32>,
    /// Gate of the feed-forward part, `n_embd` to `n_ff`.
    pub ffn_gate: Matrix<'g>,
    /// Up projection of the feed-forward part, `n_embd` to `n_ff`.
    pub ffn_up: Matrix<'g>,
    /// Down projection of the feed-forward part, `n_ff` to `n_embd`.
    pub ffn_down: Matrix<'g>,
}

/// The weights of a recurrent block.
#[derive(Clone, Debug)]
pub struct RecurrentWeights<'g> {
    /// Query, key and value in one projection, `n_embd` to `conv_dim`
    /// (`attn_qkv.weight`).
    pub qkv: Matrix<'g>,
    /// The gate `z` of the output norm, `n_embd` to `value_dim`
    /// (`attn_gate.weight`).
    pub gate: Matrix<'g>,
    /// The convolution, `conv_kernel` taps for each of `conv_dim` channels,
    /// channel-major (`ssm_conv1d.weight`); tap `conv_kernel - 1` multiplies
    /// the newest input.
    pub conv: Vec<f32>,
    /// Bias added to the decay input of each value head (`ssm_dt.bias`).
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` of each value head, a negative number (`ssm_a`).
    pub a: Vec<f32>,
    /// Beta input, `n_embd` to `n_v_heads` (`ssm_beta.weight`).
    pub beta: Matrix<'g>,
    /// Decay input, `n_embd` to `n_v_heads` (`ssm_alpha.weight`).
    pub alpha: Matrix<'g>,
    /// Weight of the gated RMS norm over a value head, `head_v_dim` values
    /// (`ssm_norm.weight`).
    pub norm: Vec<f32>,
    /// Output projection, `value_dim` to `n_embd` (`ssm_out.weight`).
    pub out: Matrix<'g>,
}

/// The weights of an attention block.
#[derive(Clone, Debug)]
pub struct AttentionWeights<'g> {
    /// Query and output gate, `n_embd` to `n_head * head_dim * 2`: per head the
    /// query, then the gate (`attn_q.weight`).
    pub q: Matrix<'g>,
    /// Key, `n_embd` to `n_head_kv * head_dim` (`attn_k.weight`).
    pub k: Matrix<'g>,
    /// Value, `n_embd` to `n_head_kv * head_dim` (`attn_v.weight`).
    pub v: Matrix<'g>,
    /// Output projection, `n_head * head_dim` to `n_embd`
    /// (`attn_output.weight`).
    pub out: Matrix<'g>,
    /// RMS norm weight over each query head (`attn_q_norm.weight`).
    pub q_norm: Vec<f32>,
    /// RMS norm weight over each key head (`attn_k_norm.weight`).
    pub k_norm: Vec<f32>,
}

/// What differs between the two kinds of block.
#[derive(Clone, Debug)]
pub enum Mixer<'g> {
    /// A recurrent block.
    Recurrent(RecurrentWeights<'g>),
    /// An attention block.
    Attention(AttentionWeights<'g>),
}

/// One block of the model.
#[derive(Clone, Debug)]
pub struct Layer<'g> {
    /// Norms and feed-forward part.
    pub block: Block<'g>,
    /// The recurrent or attention part.
    pub mixer: Mixer<'g>,
}

/// The block that predicts the token after next (multi-token prediction), the
/// last block of a file whose `nextn_predict_layers` is 1. The reference runs
/// it in `reference/mtp.rs`.
#[derive(Clone, Debug)]
pub struct NextnWeights<'g> {
    /// An attention block with its norms and feed-forward part.
    pub block: Block<'g>,
    /// Its attention weights.
    pub attention: AttentionWeights<'g>,
    /// Projection of the normalised embedding of the next token and the
    /// normalised hidden state together, `2 * n_embd` to `n_embd`
    /// (`nextn.eh_proj.weight`).
    pub eh_proj: Matrix<'g>,
    /// RMS norm weight of the embedding (`nextn.enorm.weight`).
    pub enorm: Vec<f32>,
    /// RMS norm weight of the hidden state (`nextn.hnorm.weight`).
    pub hnorm: Vec<f32>,
    /// Final norm of the block, when it has its own
    /// (`nextn.shared_head_norm.weight`); the model's `output_norm` otherwise.
    pub shared_head_norm: Option<Vec<f32>>,
    /// Embeddings of its own, when it has them (`nextn.embed_tokens.weight`);
    /// the model's `token_embd` otherwise.
    pub embed_tokens: Option<Matrix<'g>>,
    /// Output head of its own, when it has one (`nextn.shared_head_head.weight`);
    /// the model's head otherwise.
    pub shared_head_head: Option<Matrix<'g>>,
}

/// The architecture of a file that holds a `qwen35` model's MTP block alone.
pub const MTP_ARCHITECTURE: &str = "qwen35-mtp";

/// Every weight of a `qwen35` model.
#[derive(Clone, Debug)]
pub struct Weights<'g> {
    /// Token embeddings, `n_vocab` rows of `n_embd` (`token_embd.weight`).
    pub token_embd: Matrix<'g>,
    /// The final RMS norm (`output_norm.weight`).
    pub output_norm: Vec<f32>,
    /// The output head, `n_embd` to `n_vocab`; `None` when the model has none
    /// of its own and uses `token_embd` (`output.weight`).
    pub output: Option<Matrix<'g>>,
    /// The blocks, `config.n_layer` of them.
    pub layers: Vec<Layer<'g>>,
    /// The next-token-prediction block, when the configuration has one.
    pub nextn: Option<NextnWeights<'g>>,
}

impl<'g> Weights<'g> {
    /// Builds the table for `config` from the tensors of `gguf`.
    pub fn load(gguf: &'g Gguf, config: &Config) -> Result<Self> {
        let mut table = Table::new(gguf);
        let t = &mut table;
        let c = config;
        let (n_embd, n_vocab) = (c.n_embd, c.n_vocab);

        let token_embd = t.matrix("token_embd.weight", n_embd, n_vocab)?;
        let output_norm = t.vector("output_norm.weight", n_embd)?;
        let output = t.optional_matrix("output.weight", n_embd, n_vocab)?;

        let mut layers = Vec::with_capacity(c.n_layer);
        for layer in 0..c.n_layer {
            let block = block(t, c, layer)?;
            let mixer = if c.is_attention(layer) {
                Mixer::Attention(attention(t, c, layer)?)
            } else {
                Mixer::Recurrent(recurrent(t, c, layer)?)
            };
            layers.push(Layer { block, mixer });
        }

        let nextn = if c.n_nextn == 1 { Some(nextn(t, c, c.n_layer)?) } else { None };

        table.finish(OF, |_| true)?;
        Ok(Self { token_embd, output_norm, output, layers, nextn })
    }

    /// The same weights with the MTP block of `file`, a file of its own
    /// (architecture [`MTP_ARCHITECTURE`]: the block's tensors under the names
    /// the model file gives them, `blk.<n_layer>.*`), in place of the model
    /// file's own block if it has one.
    pub fn with_mtp_file(mut self, file: &'g Gguf, c: &Config) -> Result<Self> {
        if file.architecture() != Some(MTP_ARCHITECTURE) {
            return Err(Error::NotAnMtpFile {
                found: file.architecture().unwrap_or("none").to_owned(),
            });
        }
        let key = |what: &str| file.u64(&format!("{MTP_ARCHITECTURE}.{what}"));
        let (blocks, width) = (key("block_count"), key("embedding_length"));
        if blocks != Some(c.n_layer as u64 + 1) || width != Some(c.n_embd as u64) {
            return Err(Error::MtpFileShape {
                blocks: blocks.unwrap_or(0),
                width: width.unwrap_or(0),
                n_layer: c.n_layer,
                n_embd: c.n_embd,
            });
        }
        let mut table = Table::new(file);
        self.nextn = Some(nextn(&mut table, c, c.n_layer)?);
        table.finish(OF, |_| true)?;
        Ok(self)
    }

    /// The matrix that turns the final hidden state into logits.
    pub fn head(&self) -> &Matrix<'g> {
        self.output.as_ref().unwrap_or(&self.token_embd)
    }
}

fn block<'g>(t: &mut Table<'g>, c: &Config, layer: usize) -> Result<Block<'g>> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    Ok(Block {
        attn_norm: t.vector(&name("attn_norm.weight"), c.n_embd)?,
        post_attention_norm: t.vector(&name("post_attention_norm.weight"), c.n_embd)?,
        ffn_gate: t.matrix(&name("ffn_gate.weight"), c.n_embd, c.n_ff)?,
        ffn_up: t.matrix(&name("ffn_up.weight"), c.n_embd, c.n_ff)?,
        ffn_down: t.matrix(&name("ffn_down.weight"), c.n_ff, c.n_embd)?,
    })
}

fn attention<'g>(t: &mut Table<'g>, c: &Config, layer: usize) -> Result<AttentionWeights<'g>> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let (q_dim, kv_dim) = c.attention_dims();
    Ok(AttentionWeights {
        q: t.matrix(&name("attn_q.weight"), c.n_embd, q_dim * 2)?,
        k: t.matrix(&name("attn_k.weight"), c.n_embd, kv_dim)?,
        v: t.matrix(&name("attn_v.weight"), c.n_embd, kv_dim)?,
        out: t.matrix(&name("attn_output.weight"), q_dim, c.n_embd)?,
        q_norm: t.vector(&name("attn_q_norm.weight"), c.attention.head_dim)?,
        k_norm: t.vector(&name("attn_k_norm.weight"), c.attention.head_dim)?,
    })
}

fn recurrent<'g>(t: &mut Table<'g>, c: &Config, layer: usize) -> Result<RecurrentWeights<'g>> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let r = &c.recurrent;
    let n_v = r.n_v_heads;
    Ok(RecurrentWeights {
        qkv: t.matrix(&name("attn_qkv.weight"), c.n_embd, r.conv_dim())?,
        gate: t.matrix(&name("attn_gate.weight"), c.n_embd, r.value_dim())?,
        conv: t.floats(&name("ssm_conv1d.weight"), &[r.conv_kernel as u64, r.conv_dim() as u64])?,
        dt_bias: t.vector(&name("ssm_dt.bias"), n_v)?,
        a: t.vector(&name("ssm_a"), n_v)?,
        beta: t.matrix(&name("ssm_beta.weight"), c.n_embd, n_v)?,
        alpha: t.matrix(&name("ssm_alpha.weight"), c.n_embd, n_v)?,
        norm: t.vector(&name("ssm_norm.weight"), r.head_v_dim)?,
        out: t.matrix(&name("ssm_out.weight"), r.value_dim(), c.n_embd)?,
    })
}

fn nextn<'g>(t: &mut Table<'g>, c: &Config, layer: usize) -> Result<NextnWeights<'g>> {
    let name = |what: &str| format!("blk.{layer}.{what}");
    let n_embd = c.n_embd;
    Ok(NextnWeights {
        block: block(t, c, layer)?,
        attention: attention(t, c, layer)?,
        eh_proj: t.matrix(&name("nextn.eh_proj.weight"), 2 * n_embd, n_embd)?,
        enorm: t.vector(&name("nextn.enorm.weight"), n_embd)?,
        hnorm: t.vector(&name("nextn.hnorm.weight"), n_embd)?,
        shared_head_norm: t
            .optional_floats(&name("nextn.shared_head_norm.weight"), &[n_embd as u64])?,
        embed_tokens: t.optional_matrix(&name("nextn.embed_tokens.weight"), n_embd, c.n_vocab)?,
        shared_head_head: t.optional_matrix(
            &name("nextn.shared_head_head.weight"),
            n_embd,
            c.n_vocab,
        )?,
    })
}
