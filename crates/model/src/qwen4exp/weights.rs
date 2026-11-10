//! The weights of a `qwen4exp` model: which tensors there are, of which shape,
//! and where their bytes are (`docs/reference/qwen4exp-architecture.md`,
//! section 2.3).
//!
//! The table is built against a [`Config`] and is exact: a tensor that is
//! missing, has other dimensions than the configuration gives it, cannot be
//! decoded, or is in the file without being part of the architecture is an
//! error that names it.
//!
//! Nothing large is read when the table is built. Matrices stay as the stored
//! blocks, borrowed from the mapped file, and are decoded row by row when they
//! are used. The routed experts ([`ExpertStack`]) and the n-gram table
//! ([`NgramTable`]) are views too: an expert's rows and a table row are
//! fetched from the mapping when a token asks for them, so a file several
//! times the size of the machine's memory can be run. Only the small vectors
//! (norm weights, time-step parameters, the convolutions) are decoded to `f32`
//! once.

use std::collections::HashSet;

use ares_core::GgmlType;
use ares_gguf::Gguf;

use crate::qwen4exp::config::Config;
use crate::qwen4exp::error::{Error, Result};

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

    /// The stored bytes of the whole matrix.
    pub fn bytes(&self) -> &'g [u8] {
        self.data
    }

    /// The stored bytes of row `row`.
    pub fn row_bytes(&self, row: usize) -> &'g [u8] {
        let size = self.data.len() / self.rows;
        &self.data[row * size..(row + 1) * size]
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

/// The matrices of all routed experts of one kind in a layer, as stored:
/// expert `e` is the `e`-th contiguous slab of `rows` rows. Nothing is read
/// until an expert is asked for.
#[derive(Clone, Copy, Debug)]
pub struct ExpertStack<'g> {
    ty: GgmlType,
    n_expert: usize,
    rows: usize,
    cols: usize,
    data: &'g [u8],
}

impl<'g> ExpertStack<'g> {
    /// The element type the experts are stored in.
    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// The number of experts.
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// The stored bytes of one expert's matrix.
    pub fn expert_bytes(&self) -> usize {
        self.data.len() / self.n_expert
    }

    /// The matrix of expert `expert`: a view of its bytes in the mapped file.
    ///
    /// # Panics
    ///
    /// Panics when `expert` is out of range.
    pub fn expert(&self, expert: usize) -> Matrix<'g> {
        assert!(expert < self.n_expert, "the layer has {} experts", self.n_expert);
        let size = self.expert_bytes();
        Matrix {
            ty: self.ty,
            rows: self.rows,
            cols: self.cols,
            data: &self.data[expert * size..(expert + 1) * size],
        }
    }
}

/// The n-gram table (`per_layer_token_embd.weight`), as stored. A row is
/// fetched from the mapped file when a token hashes to it.
#[derive(Clone, Copy, Debug)]
pub struct NgramTable<'g> {
    ty: GgmlType,
    rows: u64,
    row_dim: usize,
    row_bytes: usize,
    data: &'g [u8],
}

impl<'g> NgramTable<'g> {
    /// The element type the rows are stored in.
    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// The number of rows.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The values of one row.
    pub fn row_dim(&self) -> usize {
        self.row_dim
    }

    /// The stored bytes of row `row`.
    ///
    /// # Panics
    ///
    /// Panics when `row` is out of range.
    pub fn row_bytes(&self, row: u64) -> &'g [u8] {
        assert!(row < self.rows, "the table has {} rows", self.rows);
        let at = row as usize * self.row_bytes;
        &self.data[at..at + self.row_bytes]
    }

    /// Decodes row `row` into `out` (`row_dim` values).
    ///
    /// # Panics
    ///
    /// Panics when `row` or the length of `out` is out of range.
    pub fn decode_row(&self, row: u64, out: &mut [f32]) {
        assert_eq!(out.len(), self.row_dim, "a row has {} values", self.row_dim);
        ares_quant::dequantize(self.ty, self.row_bytes(row), out)
            .expect("the type and length of the rows were checked when the table was built");
    }
}

/// One stream mixer ("hyper-connection"): the read in front of a block and
/// the write-back weights behind it.
#[derive(Clone, Debug)]
pub struct StreamMixer<'g> {
    /// Norm weights, stream `c` at `[c * N, (c + 1) * N)` (`hc_*_norm.weight`).
    pub norm: Vec<f32>,
    /// Down projection, `D` to `LR` (`hc_*_down.weight`).
    pub down: Matrix<'g>,
    /// Up projection, `LR` to `D` (`hc_*_up.weight`).
    pub up: Matrix<'g>,
    /// Write-back weights, `D` to `hc` (`hc_*_inject.weight`); `None` for the
    /// final mixer in front of the output head.
    pub inject: Option<Matrix<'g>>,
}

/// The mixture of experts of one layer.
#[derive(Clone, Debug)]
pub struct Moe<'g> {
    /// The router, `N` to `E` (`ffn_gate_inp.weight`).
    pub router: Matrix<'g>,
    /// Gate of the routed experts, each `N` to `FF` (`ffn_gate_exps.weight`).
    pub gate: ExpertStack<'g>,
    /// Up projection of the routed experts, each `N` to `FF`
    /// (`ffn_up_exps.weight`).
    pub up: ExpertStack<'g>,
    /// Down projection of the routed experts, each `FF` to `N`
    /// (`ffn_down_exps.weight`).
    pub down: ExpertStack<'g>,
    /// The scalar gate of the shared expert, `N` values
    /// (`ffn_gate_inp_shexp.weight`).
    pub shared_gate_inp: Vec<f32>,
    /// Gate of the shared expert (`ffn_gate_shexp.weight`).
    pub shared_gate: Matrix<'g>,
    /// Up projection of the shared expert (`ffn_up_shexp.weight`).
    pub shared_up: Matrix<'g>,
    /// Down projection of the shared expert (`ffn_down_shexp.weight`).
    pub shared_down: Matrix<'g>,
}

/// The weights of a recurrent layer.
#[derive(Clone, Debug)]
pub struct RecurrentWeights<'g> {
    /// Query, key and value in one projection, `N` to `CG`
    /// (`attn_qkv.weight`).
    pub qkv: Matrix<'g>,
    /// The output gate `z`, `N` to `GV * S` (`attn_gate.weight`).
    pub gate: Matrix<'g>,
    /// The convolution, `conv_kernel` taps for each of `CG` channels,
    /// channel-major (`ssm_conv1d.weight`); the last tap multiplies the newest
    /// input.
    pub conv: Vec<f32>,
    /// Bias added to the decay input of each value head (`ssm_dt.bias`).
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` of each value head, a negative number (`ssm_a`).
    pub a: Vec<f32>,
    /// Write strength input, `N` to `GV` (`ssm_beta.weight`).
    pub beta: Matrix<'g>,
    /// Decay input, `N` to `GV` (`ssm_alpha.weight`).
    pub alpha: Matrix<'g>,
    /// Weight of the output norm over a value head, `S` values, shared by all
    /// heads (`ssm_norm.weight`).
    pub norm: Vec<f32>,
    /// Output projection, `GV * S` to `N` (`ssm_out.weight`).
    pub out: Matrix<'g>,
}

/// The weights of an attention layer with its indexer.
#[derive(Clone, Debug)]
pub struct AttentionWeights<'g> {
    /// Query and output gate, `N` to `H * HD * 2`: per head the query, then
    /// the gate (`attn_q.weight`).
    pub q: Matrix<'g>,
    /// Key, `N` to `HKV * HD` (`attn_k.weight`).
    pub k: Matrix<'g>,
    /// Value, `N` to `HKV * HD` (`attn_v.weight`).
    pub v: Matrix<'g>,
    /// Output projection, `H * HD` to `N` (`attn_output.weight`).
    pub out: Matrix<'g>,
    /// Norm weight over each query head (`attn_q_norm.weight`).
    pub q_norm: Vec<f32>,
    /// Norm weight over each key head (`attn_k_norm.weight`).
    pub k_norm: Vec<f32>,
    /// Indexer queries, `N` to `IH * ID` (`indexer.q_proj.weight`).
    pub indexer_q: Matrix<'g>,
    /// The raw indexer key, `N` to `ID` (`indexer.k_proj.weight`).
    pub indexer_k: Matrix<'g>,
    /// Norm weight over each indexer query head (`indexer.q_norm.weight`).
    pub indexer_q_norm: Vec<f32>,
    /// Norm weight over a pooled key (`indexer.k_norm.weight`).
    pub indexer_k_norm: Vec<f32>,
}

/// The weights of the hashed n-gram embedding of its layer.
#[derive(Clone, Debug)]
pub struct NgramWeights<'g> {
    /// Gathered rows to a key per stream, `N` to `D` (`ple_key.weight`).
    pub key: Matrix<'g>,
    /// Gathered rows to a value, `N` to `N` (`ple_value.weight`).
    pub value: Matrix<'g>,
    /// Norm weights of the key, per stream (`ple_norm_key.weight`).
    pub norm_key: Vec<f32>,
    /// Norm weights of the residual as the query (`ple_norm_query.weight`).
    pub norm_query: Vec<f32>,
    /// Norm weights in front of the convolution (`ple_norm_conv.weight`).
    pub norm_conv: Vec<f32>,
    /// The convolution, `conv_kernel` taps for each of `D` channels,
    /// channel-major (`ple_conv1d.weight`); the last tap multiplies the
    /// newest input.
    pub conv: Vec<f32>,
}

/// What differs between the two kinds of layer.
#[derive(Clone, Debug)]
pub enum Mixer<'g> {
    /// A recurrent layer.
    Recurrent(RecurrentWeights<'g>),
    /// An attention layer.
    Attention(AttentionWeights<'g>),
}

/// One layer of the model.
#[derive(Clone, Debug)]
pub struct Layer<'g> {
    /// The stream mixer around the recurrent or attention part (`hc_attn_*`).
    pub attn_mixer: StreamMixer<'g>,
    /// The stream mixer around the mixture of experts (`hc_ffn_*`).
    pub ffn_mixer: StreamMixer<'g>,
    /// The recurrent or attention part.
    pub mixer: Mixer<'g>,
    /// The mixture of experts.
    pub moe: Moe<'g>,
    /// The n-gram embedding, on the layer that has it.
    pub ngram: Option<NgramWeights<'g>>,
}

/// Every weight of a `qwen4exp` model.
#[derive(Clone, Debug)]
pub struct Weights<'g> {
    /// Token embeddings, `V` rows of `N` (`token_embd.weight`).
    pub token_embd: Matrix<'g>,
    /// The output head, `N` to `V` (`output.weight`).
    pub output: Matrix<'g>,
    /// The final mixer in front of the head (`output_hc_*`), without inject.
    pub output_mixer: StreamMixer<'g>,
    /// The n-gram table, when the model has the n-gram embedding.
    pub ngram_table: Option<NgramTable<'g>>,
    /// The layers, `config.n_layer` of them.
    pub layers: Vec<Layer<'g>>,
}

impl<'g> Weights<'g> {
    /// Builds the table for `config` from the tensors of `gguf`.
    pub fn load(gguf: &'g Gguf, config: &Config) -> Result<Self> {
        let mut loader = Loader { gguf, used: HashSet::new() };
        let c = config;
        let (n, wide) = (c.n_embd, c.wide());

        let token_embd = loader.matrix("token_embd.weight", n, c.n_vocab)?;
        let output = loader.matrix("output.weight", n, c.n_vocab)?;
        let output_mixer = StreamMixer {
            norm: loader.vector("output_hc_norm.weight", wide)?,
            down: loader.matrix("output_hc_down.weight", wide, c.mixer_rank)?,
            up: loader.matrix("output_hc_up.weight", c.mixer_rank, wide)?,
            inject: None,
        };
        let ngram_table = match &c.ngram {
            Some(ngram) => Some(loader.table(ngram.row_dim, ngram.table_rows)?),
            None => None,
        };

        let mut layers = Vec::with_capacity(c.n_layer);
        for layer in 0..c.n_layer {
            let mixer = if c.is_attention(layer) {
                Mixer::Attention(loader.attention(c, layer)?)
            } else {
                Mixer::Recurrent(loader.recurrent(c, layer)?)
            };
            let ngram = match &c.ngram {
                Some(ngram) if ngram.layer == layer => Some(loader.ngram(c, layer)?),
                _ => None,
            };
            layers.push(Layer {
                attn_mixer: loader.stream_mixer(c, layer, "attn")?,
                ffn_mixer: loader.stream_mixer(c, layer, "ffn")?,
                mixer,
                moe: loader.moe(c, layer)?,
                ngram,
            });
        }

        if let Some(extra) = gguf.tensors().iter().find(|t| !loader.used.contains(&t.name)) {
            return Err(Error::UnexpectedTensor { name: extra.name.clone() });
        }
        Ok(Self { token_embd, output, output_mixer, ngram_table, layers })
    }
}

struct Loader<'g> {
    gguf: &'g Gguf,
    used: HashSet<String>,
}

impl<'g> Loader<'g> {
    fn info(&mut self, name: &str, dims: &[u64]) -> Result<&'g ares_gguf::TensorInfo> {
        let info =
            self.gguf.tensor(name).ok_or_else(|| Error::MissingTensor { name: name.to_owned() })?;
        if info.dims.as_slice() != dims {
            return Err(Error::TensorShape {
                name: name.to_owned(),
                found: info.dims.as_slice().to_vec(),
                expected: dims.to_vec(),
            });
        }
        self.used.insert(name.to_owned());
        Ok(info)
    }

    /// The bytes of `rows` rows of `cols` values each, after checking that a
    /// row is a whole number of blocks and that its first one decodes: a type
    /// that cannot be decoded is an error of loading, not of the first
    /// multiplication. Only that one row is read.
    fn rows_of(&self, info: &'g ares_gguf::TensorInfo, cols: usize, rows: u64) -> Result<&'g [u8]> {
        let data = self.gguf.data(info);
        let row_bytes = info.ty.bytes_for(cols as u64).filter(|b| b * rows == data.len() as u64);
        let Some(row_bytes) = row_bytes else {
            return Err(Error::TensorRows { name: info.name.clone(), ty: info.ty, cols });
        };
        let mut first = vec![0.0; cols];
        ares_quant::dequantize(info.ty, &data[..row_bytes as usize], &mut first)
            .map_err(|source| Error::TensorType { name: info.name.clone(), ty: info.ty, source })?;
        Ok(data)
    }

    fn matrix(&mut self, name: &str, cols: usize, rows: usize) -> Result<Matrix<'g>> {
        let info = self.info(name, &[cols as u64, rows as u64])?;
        let data = self.rows_of(info, cols, rows as u64)?;
        Ok(Matrix { ty: info.ty, rows, cols, data })
    }

    fn experts(
        &mut self,
        name: &str,
        cols: usize,
        rows: usize,
        n_expert: usize,
    ) -> Result<ExpertStack<'g>> {
        let info = self.info(name, &[cols as u64, rows as u64, n_expert as u64])?;
        let data = self.rows_of(info, cols, (rows * n_expert) as u64)?;
        Ok(ExpertStack { ty: info.ty, n_expert, rows, cols, data })
    }

    fn table(&mut self, row_dim: usize, rows: u64) -> Result<NgramTable<'g>> {
        let info = self.info("per_layer_token_embd.weight", &[row_dim as u64, rows])?;
        let data = self.rows_of(info, row_dim, rows)?;
        let row_bytes = data.len() / rows as usize;
        Ok(NgramTable { ty: info.ty, rows, row_dim, row_bytes, data })
    }

    fn vector_of(&mut self, name: &str, dims: &[u64]) -> Result<Vec<f32>> {
        let info = self.info(name, dims)?;
        let mut out = vec![0.0; dims.iter().product::<u64>() as usize];
        ares_quant::dequantize(info.ty, self.gguf.data(info), &mut out)
            .map_err(|source| Error::TensorType { name: name.to_owned(), ty: info.ty, source })?;
        Ok(out)
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.vector_of(name, &[len as u64])
    }

    fn stream_mixer(&mut self, c: &Config, layer: usize, which: &str) -> Result<StreamMixer<'g>> {
        let name = |what: &str| format!("blk.{layer}.hc_{which}_{what}.weight");
        let wide = c.wide();
        Ok(StreamMixer {
            norm: self.vector(&name("norm"), wide)?,
            down: self.matrix(&name("down"), wide, c.mixer_rank)?,
            up: self.matrix(&name("up"), c.mixer_rank, wide)?,
            inject: Some(self.matrix(&name("inject"), wide, c.n_stream)?),
        })
    }

    fn moe(&mut self, c: &Config, layer: usize) -> Result<Moe<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let (n, e) = (c.n_embd, &c.experts);
        Ok(Moe {
            router: self.matrix(&name("ffn_gate_inp.weight"), n, e.count)?,
            gate: self.experts(&name("ffn_gate_exps.weight"), n, e.ff, e.count)?,
            up: self.experts(&name("ffn_up_exps.weight"), n, e.ff, e.count)?,
            down: self.experts(&name("ffn_down_exps.weight"), e.ff, n, e.count)?,
            shared_gate_inp: self.vector(&name("ffn_gate_inp_shexp.weight"), n)?,
            shared_gate: self.matrix(&name("ffn_gate_shexp.weight"), n, e.shared_ff)?,
            shared_up: self.matrix(&name("ffn_up_shexp.weight"), n, e.shared_ff)?,
            shared_down: self.matrix(&name("ffn_down_shexp.weight"), e.shared_ff, n)?,
        })
    }

    fn attention(&mut self, c: &Config, layer: usize) -> Result<AttentionWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let (q_dim, kv_dim) = c.attention_dims();
        let (n, ix) = (c.n_embd, &c.indexer);
        Ok(AttentionWeights {
            q: self.matrix(&name("attn_q.weight"), n, q_dim * 2)?,
            k: self.matrix(&name("attn_k.weight"), n, kv_dim)?,
            v: self.matrix(&name("attn_v.weight"), n, kv_dim)?,
            out: self.matrix(&name("attn_output.weight"), q_dim, n)?,
            q_norm: self.vector(&name("attn_q_norm.weight"), c.attention.head_dim)?,
            k_norm: self.vector(&name("attn_k_norm.weight"), c.attention.head_dim)?,
            indexer_q: self.matrix(&name("indexer.q_proj.weight"), n, ix.n_head * ix.key_dim)?,
            indexer_k: self.matrix(&name("indexer.k_proj.weight"), n, ix.key_dim)?,
            indexer_q_norm: self.vector(&name("indexer.q_norm.weight"), ix.key_dim)?,
            indexer_k_norm: self.vector(&name("indexer.k_norm.weight"), ix.key_dim)?,
        })
    }

    fn recurrent(&mut self, c: &Config, layer: usize) -> Result<RecurrentWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let r = &c.recurrent;
        let (n, n_v) = (c.n_embd, r.n_v_heads);
        Ok(RecurrentWeights {
            qkv: self.matrix(&name("attn_qkv.weight"), n, r.conv_dim())?,
            gate: self.matrix(&name("attn_gate.weight"), n, r.value_dim())?,
            conv: self.vector_of(
                &name("ssm_conv1d.weight"),
                &[r.conv_kernel as u64, r.conv_dim() as u64],
            )?,
            dt_bias: self.vector(&name("ssm_dt.bias"), n_v)?,
            a: self.vector(&name("ssm_a"), n_v)?,
            beta: self.matrix(&name("ssm_beta.weight"), n, n_v)?,
            alpha: self.matrix(&name("ssm_alpha.weight"), n, n_v)?,
            norm: self.vector(&name("ssm_norm.weight"), r.head_dim)?,
            out: self.matrix(&name("ssm_out.weight"), r.value_dim(), n)?,
        })
    }

    fn ngram(&mut self, c: &Config, layer: usize) -> Result<NgramWeights<'g>> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let (n, wide) = (c.n_embd, c.wide());
        let kernel = c.ngram.as_ref().map_or(0, |g| g.conv_kernel);
        Ok(NgramWeights {
            key: self.matrix(&name("ple_key.weight"), n, wide)?,
            value: self.matrix(&name("ple_value.weight"), n, n)?,
            norm_key: self.vector(&name("ple_norm_key.weight"), wide)?,
            norm_query: self.vector(&name("ple_norm_query.weight"), wide)?,
            norm_conv: self.vector(&name("ple_norm_conv.weight"), wide)?,
            conv: self.vector_of(&name("ple_conv1d.weight"), &[kernel as u64, wide as u64])?,
        })
    }
}
