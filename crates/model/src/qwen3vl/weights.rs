//! The weights of a projector: which tensors there are, of which shape, and
//! where their bytes are.
//!
//! The table is built against a [`Config`] and is exact: a tensor that is
//! missing, has other dimensions than the configuration gives it, cannot be
//! decoded, or is in the file without being part of the projector is an error
//! that names it. The matrices of the blocks and the merger stay as stored
//! (borrowed from the file); the patch kernels, the position table and the
//! vectors are decoded to `f32` once.

use std::collections::HashSet;

use ares_core::GgmlType;
use ares_gguf::Gguf;

use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::{Error, Result};

/// A weight matrix as stored: `rows` outputs, each a dot product over `cols`
/// inputs, one row after the other. In GGUF the dimensions are `[cols, rows]`.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'g> {
    ty: GgmlType,
    rows: usize,
    cols: usize,
    data: &'g [u8],
}

impl<'g> Matrix<'g> {
    /// The element type the matrix is stored in.
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

    /// The stored bytes, row after row: what a device loader uploads.
    pub fn bytes(&self) -> &'g [u8] {
        self.data
    }

    /// The whole matrix as `f32`, row after row.
    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        ares_quant::dequantize(self.ty, self.data, &mut out)
            .expect("the type was decoded when the table was built");
        out
    }
}

/// The weight and bias of a layer norm.
#[derive(Clone, Debug)]
pub struct Norm {
    /// What the normalised value is multiplied by.
    pub weight: Vec<f32>,
    /// What is added after.
    pub bias: Vec<f32>,
}

/// A matrix with the bias added to its product.
#[derive(Clone, Debug)]
pub struct Linear<'g> {
    /// The matrix.
    pub weight: Matrix<'g>,
    /// One value per output.
    pub bias: Vec<f32>,
}

/// One block of the tower.
#[derive(Clone, Debug)]
pub struct Block<'g> {
    /// The norm in front of the attention (`ln1`).
    pub ln1: Norm,
    /// Query, key and value in one matrix of `3 * n_embd` outputs (`attn_qkv`).
    pub qkv: Linear<'g>,
    /// The attention's output matrix (`attn_out`).
    pub out: Linear<'g>,
    /// The norm in front of the MLP (`ln2`).
    pub ln2: Norm,
    /// `n_embd` to `n_ff` (`ffn_up`).
    pub up: Linear<'g>,
    /// `n_ff` to `n_embd` (`ffn_down`).
    pub down: Linear<'g>,
}

/// Every tensor of a projector.
#[derive(Clone, Debug)]
pub struct Weights<'g> {
    /// The two patch kernels (`v.patch_embd.weight`, `.weight.1`), each
    /// `n_embd` rows of `3 * patch * patch` values in the order channel, row,
    /// column. A still image goes through both and the results are added.
    pub patch: [Vec<f32>; 2],
    /// Added to every patch (`v.patch_embd.bias`).
    pub patch_bias: Vec<f32>,
    /// The learned positions: `table_side * table_side` rows of `n_embd`,
    /// row-major over the table (`v.position_embd.weight`).
    pub position: Vec<f32>,
    /// The blocks, first to last.
    pub blocks: Vec<Block<'g>>,
    /// The norm after the last block (`v.post_ln`).
    pub post_ln: Norm,
    /// The merger's first layer, `4 * n_embd` to `4 * n_embd` (`mm.0`).
    pub mm0: Linear<'g>,
    /// The merger's second layer, `4 * n_embd` to `n_out` (`mm.2`).
    pub mm2: Linear<'g>,
}

struct Loader<'g> {
    gguf: &'g Gguf,
    claimed: HashSet<String>,
}

impl<'g> Weights<'g> {
    /// Builds the table of a file against its configuration.
    pub fn load(gguf: &'g Gguf, c: &Config) -> Result<Self> {
        let mut l = Loader { gguf, claimed: HashSet::new() };
        let (side, e, group) = (c.table_side, c.n_embd, 4 * c.n_embd);
        let kernel = [c.patch as u64, c.patch as u64, 3, e as u64];
        let weights = Self {
            patch: [
                l.floats("v.patch_embd.weight", &kernel)?,
                l.floats("v.patch_embd.weight.1", &kernel)?,
            ],
            patch_bias: l.floats("v.patch_embd.bias", &[e as u64])?,
            position: l.floats("v.position_embd.weight", &[e as u64, (side * side) as u64])?,
            blocks: (0..c.n_layer)
                .map(|i| {
                    let name = |part: &str| format!("v.blk.{i}.{part}");
                    Ok(Block {
                        ln1: l.norm(&name("ln1"), e)?,
                        qkv: l.linear(&name("attn_qkv"), e, 3 * e)?,
                        out: l.linear(&name("attn_out"), e, e)?,
                        ln2: l.norm(&name("ln2"), e)?,
                        up: l.linear(&name("ffn_up"), e, c.n_ff)?,
                        down: l.linear(&name("ffn_down"), c.n_ff, e)?,
                    })
                })
                .collect::<Result<_>>()?,
            post_ln: l.norm("v.post_ln", e)?,
            mm0: l.linear("mm.0", group, group)?,
            mm2: l.linear("mm.2", group, c.n_out)?,
        };
        match gguf.tensors().iter().find(|t| !l.claimed.contains(&t.name)) {
            Some(extra) => Err(Error::UnexpectedTensor { name: extra.name.clone() }),
            None => Ok(weights),
        }
    }
}

impl<'g> Loader<'g> {
    fn info(&mut self, name: &str, dims: &[u64]) -> Result<&'g ares_gguf::TensorInfo> {
        let missing = || Error::MissingTensor { name: name.to_owned() };
        let info = self.gguf.tensor(name).ok_or_else(missing)?;
        if info.dims.as_slice() != dims {
            return Err(Error::TensorShape {
                name: name.to_owned(),
                found: info.dims.as_slice().to_vec(),
                expected: dims.to_vec(),
            });
        }
        self.claimed.insert(name.to_owned());
        Ok(info)
    }

    fn floats(&mut self, name: &str, dims: &[u64]) -> Result<Vec<f32>> {
        let info = self.info(name, dims)?;
        let mut out = vec![0.0; dims.iter().product::<u64>() as usize];
        ares_quant::dequantize(info.ty, self.gguf.data(info), &mut out)
            .map_err(|source| Error::TensorType { name: name.to_owned(), ty: info.ty, source })?;
        Ok(out)
    }

    fn norm(&mut self, name: &str, len: usize) -> Result<Norm> {
        Ok(Norm {
            weight: self.floats(&format!("{name}.weight"), &[len as u64])?,
            bias: self.floats(&format!("{name}.bias"), &[len as u64])?,
        })
    }

    fn linear(&mut self, name: &str, cols: usize, rows: usize) -> Result<Linear<'g>> {
        let weight = format!("{name}.weight");
        let info = self.info(&weight, &[cols as u64, rows as u64])?;
        let matrix = Matrix { ty: info.ty, rows, cols, data: self.gguf.data(info) };
        // One row decoded now, so that a type without a decoder is refused here.
        let row = matrix.data.len() / rows;
        ares_quant::dequantize(info.ty, &matrix.data[..row], &mut vec![0.0; cols])
            .map_err(|source| Error::TensorType { name: weight, ty: info.ty, source })?;
        Ok(Linear { weight: matrix, bias: self.floats(&format!("{name}.bias"), &[rows as u64])? })
    }
}
