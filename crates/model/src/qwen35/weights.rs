//! The weights of a `qwen35` model: which tensors there are, of which shape,
//! and where their bytes are.
//!
//! The table is built against a [`Config`] and is exact: a tensor that is
//! missing, has other dimensions than the configuration gives it, cannot be
//! decoded, or is in the file without being part of the architecture is an
//! error that names it. Matrices stay as the stored blocks (borrowed from the
//! file) and are decoded row by row when they are used; the small vectors
//! (norms, time-step parameters, the convolution) are decoded to `f32` once.

use std::collections::HashSet;

use ares_core::GgmlType;
use ares_gguf::Gguf;

use crate::qwen35::config::Config;
use crate::qwen35::error::{Error, Result};

/// A weight matrix as stored: `rows` output values, each a dot product over
/// `cols` input values, one row after the other in quantized blocks.
///
/// In GGUF the dimensions are `[cols, rows]` (the first one is contiguous),
/// so a matrix that maps `n_in` inputs to `n_out` outputs has `cols == n_in`.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'g> {
    ty: GgmlType,
    rows: usize,
    cols: usize,
    data: &'g [u8],
}

impl<'g> Matrix<'g> {
    /// The element type the rows are stored in.
    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// The number of outputs.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of inputs.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The stored bytes of row `row`.
    pub fn row_bytes(&self, row: usize) -> &'g [u8] {
        let size = self.data.len() / self.rows;
        &self.data[row * size..(row + 1) * size]
    }

    /// The stored bytes of the whole matrix, row after row: what a device
    /// loader uploads.
    pub fn bytes(&self) -> &'g [u8] {
        self.data
    }

    /// Decodes row `row` into `out` (`cols` values).
    ///
    /// # Panics
    ///
    /// Panics when `row` or the length of `out` is out of range. The type was
    /// checked to be decodable when the matrix was built, so decoding itself
    /// cannot fail.
    pub fn decode_row(&self, row: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols, "a row has {} values", self.cols);
        ares_quant::dequantize(self.ty, self.row_bytes(row), out)
            .expect("the type and length of the rows were checked when the matrix was built");
    }
}

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
        let mut loader = Loader { gguf, used: HashSet::new() };
        let c = config;
        let (n_embd, n_vocab) = (c.n_embd, c.n_vocab);

        let token_embd = loader.matrix("token_embd.weight", n_embd, n_vocab)?;
        let output_norm = loader.vector("output_norm.weight", n_embd)?;
        let output = loader.optional_matrix("output.weight", n_embd, n_vocab)?;

        let mut layers = Vec::with_capacity(c.n_layer);
        for layer in 0..c.n_layer {
            let block = loader.block(c, layer)?;
            let mixer = if c.is_attention(layer) {
                Mixer::Attention(loader.attention(c, layer)?)
            } else {
                Mixer::Recurrent(loader.recurrent(c, layer)?)
            };
            layers.push(Layer { block, mixer });
        }

        let nextn = if c.n_nextn == 1 { Some(loader.nextn(c, c.n_layer)?) } else { None };

        if let Some(extra) = gguf.tensors().iter().find(|t| !loader.used.contains(&t.name)) {
            return Err(Error::UnexpectedTensor { name: extra.name.clone() });
        }
        Ok(Self { token_embd, output_norm, output, layers, nextn })
    }

    /// The matrix that turns the final hidden state into logits.
    pub fn head(&self) -> &Matrix<'g> {
        self.output.as_ref().unwrap_or(&self.token_embd)
    }
}

struct Loader<'g> {
    gguf: &'g Gguf,
    used: HashSet<String>,
}

impl<'g> Loader<'g> {
    fn info(&mut self, name: &str, dims: &[u64]) -> Result<Option<&'g ares_gguf::TensorInfo>> {
        let Some(info) = self.gguf.tensor(name) else { return Ok(None) };
        if info.dims.as_slice() != dims {
            return Err(Error::TensorShape {
                name: name.to_owned(),
                found: info.dims.as_slice().to_vec(),
                expected: dims.to_vec(),
            });
        }
        self.used.insert(name.to_owned());
        Ok(Some(info))
    }

    fn optional_matrix(
        &mut self,
        name: &str,
        cols: usize,
        rows: usize,
    ) -> Result<Option<Matrix<'g>>> {
        let Some(info) = self.info(name, &[cols as u64, rows as u64])? else { return Ok(None) };
        let data = self.gguf.data(info);
        let matrix = Matrix { ty: info.ty, rows, cols, data };
        // Decode the first row now, so that a type that cannot be decoded is an
        // error of loading and not of the first multiplication.
        let mut first = vec![0.0; cols];
        ares_quant::dequantize(info.ty, matrix.row_bytes(0), &mut first)
            .map_err(|source| Error::TensorType { name: name.to_owned(), ty: info.ty, source })?;
        Ok(Some(matrix))
    }

    fn matrix(&mut self, name: &str, cols: usize, rows: usize) -> Result<Matrix<'g>> {
        self.optional_matrix(name, cols, rows)?
            .ok_or_else(|| Error::MissingTensor { name: name.to_owned() })
    }

    fn optional_vector(&mut self, name: &str, dims: &[u64]) -> Result<Option<Vec<f32>>> {
        let Some(info) = self.info(name, dims)? else { return Ok(None) };
        let n = dims.iter().product::<u64>() as usize;
        let mut out = vec![0.0; n];
        ares_quant::dequantize(info.ty, self.gguf.data(info), &mut out)
            .map_err(|source| Error::TensorType { name: name.to_owned(), ty: info.ty, source })?;
        Ok(Some(out))
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.optional_vector(name, &[len as u64])?
            .ok_or_else(|| Error::MissingTensor { name: name.to_owned() })
    }

    fn block(&mut self, c: &Config, layer: usize) -> Result<Block<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        Ok(Block {
            attn_norm: self.vector(&name("attn_norm.weight"), c.n_embd)?,
            post_attention_norm: self.vector(&name("post_attention_norm.weight"), c.n_embd)?,
            ffn_gate: self.matrix(&name("ffn_gate.weight"), c.n_embd, c.n_ff)?,
            ffn_up: self.matrix(&name("ffn_up.weight"), c.n_embd, c.n_ff)?,
            ffn_down: self.matrix(&name("ffn_down.weight"), c.n_ff, c.n_embd)?,
        })
    }

    fn attention(&mut self, c: &Config, layer: usize) -> Result<AttentionWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let (q_dim, kv_dim) = c.attention_dims();
        Ok(AttentionWeights {
            q: self.matrix(&name("attn_q.weight"), c.n_embd, q_dim * 2)?,
            k: self.matrix(&name("attn_k.weight"), c.n_embd, kv_dim)?,
            v: self.matrix(&name("attn_v.weight"), c.n_embd, kv_dim)?,
            out: self.matrix(&name("attn_output.weight"), q_dim, c.n_embd)?,
            q_norm: self.vector(&name("attn_q_norm.weight"), c.attention.head_dim)?,
            k_norm: self.vector(&name("attn_k_norm.weight"), c.attention.head_dim)?,
        })
    }

    fn recurrent(&mut self, c: &Config, layer: usize) -> Result<RecurrentWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let r = &c.recurrent;
        let n_v = r.n_v_heads;
        let conv = self
            .optional_vector(
                &name("ssm_conv1d.weight"),
                &[r.conv_kernel as u64, r.conv_dim() as u64],
            )?
            .ok_or_else(|| Error::MissingTensor { name: name("ssm_conv1d.weight") })?;
        Ok(RecurrentWeights {
            qkv: self.matrix(&name("attn_qkv.weight"), c.n_embd, r.conv_dim())?,
            gate: self.matrix(&name("attn_gate.weight"), c.n_embd, r.value_dim())?,
            conv,
            dt_bias: self.vector(&name("ssm_dt.bias"), n_v)?,
            a: self.vector(&name("ssm_a"), n_v)?,
            beta: self.matrix(&name("ssm_beta.weight"), c.n_embd, n_v)?,
            alpha: self.matrix(&name("ssm_alpha.weight"), c.n_embd, n_v)?,
            norm: self.vector(&name("ssm_norm.weight"), r.head_v_dim)?,
            out: self.matrix(&name("ssm_out.weight"), r.value_dim(), c.n_embd)?,
        })
    }

    fn nextn(&mut self, c: &Config, layer: usize) -> Result<NextnWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let n_embd = c.n_embd;
        Ok(NextnWeights {
            block: self.block(c, layer)?,
            attention: self.attention(c, layer)?,
            eh_proj: self.matrix(&name("nextn.eh_proj.weight"), 2 * n_embd, n_embd)?,
            enorm: self.vector(&name("nextn.enorm.weight"), n_embd)?,
            hnorm: self.vector(&name("nextn.hnorm.weight"), n_embd)?,
            shared_head_norm: self
                .optional_vector(&name("nextn.shared_head_norm.weight"), &[n_embd as u64])?,
            embed_tokens: self.optional_matrix(
                &name("nextn.embed_tokens.weight"),
                n_embd,
                c.n_vocab,
            )?,
            shared_head_head: self.optional_matrix(
                &name("nextn.shared_head_head.weight"),
                n_embd,
                c.n_vocab,
            )?,
        })
    }
}
