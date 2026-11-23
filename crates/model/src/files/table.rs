//! The tensors of a GGUF file as a model's weight table takes them: each by
//! name with the dimensions the configuration gives it, and at the end a
//! tensor that no one took is refused.

use std::collections::HashSet;

use ares_core::GgmlType;
use ares_gguf::{Gguf, TensorInfo};

use super::FileError;

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
    /// A matrix of `rows` rows of `cols` values of `ty` in `data`, whose rows
    /// were checked to be whole blocks that decode.
    pub(crate) fn new(ty: GgmlType, rows: usize, cols: usize, data: &'g [u8]) -> Self {
        Self { ty, rows, cols, data }
    }

    /// The rows `rows` of the matrix as a matrix of their own.
    pub(crate) fn slice(&self, rows: std::ops::Range<usize>) -> Self {
        let size = self.data.len() / self.rows;
        Self {
            ty: self.ty,
            rows: rows.len(),
            cols: self.cols,
            data: &self.data[rows.start * size..rows.end * size],
        }
    }

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

    /// The stored bytes of the whole matrix, row after row: what a device
    /// loader uploads.
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

    /// The whole matrix as `f32`, row after row.
    pub fn decode(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        ares_quant::dequantize(self.ty, self.data, &mut out)
            .expect("the type and length of the rows were checked when the matrix was built");
        out
    }
}

/// The tensors of one file, taken by name; the names taken are kept, so that
/// a tensor no one took can be refused at the end.
pub(crate) struct Table<'g> {
    gguf: &'g Gguf,
    claimed: HashSet<String>,
}

impl<'g> Table<'g> {
    pub(crate) fn new(gguf: &'g Gguf) -> Self {
        Self { gguf, claimed: HashSet::new() }
    }

    /// The tensor `name`, `None` when the file has none; one of other
    /// dimensions than `dims` is an error.
    pub(crate) fn find(
        &mut self,
        name: &str,
        dims: &[u64],
    ) -> Result<Option<&'g TensorInfo>, FileError> {
        let Some(info) = self.gguf.tensor(name) else { return Ok(None) };
        if info.dims.as_slice() != dims {
            return Err(FileError::TensorShape {
                name: name.to_owned(),
                found: info.dims.as_slice().to_vec(),
                expected: dims.to_vec(),
            });
        }
        self.claimed.insert(name.to_owned());
        Ok(Some(info))
    }

    /// The tensor `name` of dimensions `dims`, which the file must have.
    pub(crate) fn info(&mut self, name: &str, dims: &[u64]) -> Result<&'g TensorInfo, FileError> {
        self.find(name, dims)?.ok_or_else(|| FileError::MissingTensor { name: name.to_owned() })
    }

    /// The bytes of `info`, taken as `rows` rows of `cols` values each, after
    /// checking that a row is a whole number of blocks and that its first one
    /// decodes: a type that cannot be decoded is an error of loading, not of
    /// the first multiplication. Only that one row is read.
    pub(crate) fn rows(
        &self,
        info: &'g TensorInfo,
        cols: usize,
        rows: u64,
    ) -> Result<&'g [u8], FileError> {
        let data = self.bytes(info);
        let row_bytes = info.ty.bytes_for(cols as u64).filter(|b| b * rows == data.len() as u64);
        let Some(row_bytes) = row_bytes else {
            return Err(FileError::TensorRows { name: info.name.clone(), ty: info.ty, cols });
        };
        let mut first = vec![0.0; cols];
        ares_quant::dequantize(info.ty, &data[..row_bytes as usize], &mut first).map_err(
            |source| FileError::TensorType { name: info.name.clone(), ty: info.ty, source },
        )?;
        Ok(data)
    }

    /// The matrix `name` of `rows` rows of `cols` values, `None` when the
    /// file has none.
    pub(crate) fn optional_matrix(
        &mut self,
        name: &str,
        cols: usize,
        rows: usize,
    ) -> Result<Option<Matrix<'g>>, FileError> {
        let Some(info) = self.find(name, &[cols as u64, rows as u64])? else { return Ok(None) };
        let data = self.rows(info, cols, rows as u64)?;
        Ok(Some(Matrix::new(info.ty, rows, cols, data)))
    }

    /// The matrix `name` of `rows` rows of `cols` values.
    pub(crate) fn matrix(
        &mut self,
        name: &str,
        cols: usize,
        rows: usize,
    ) -> Result<Matrix<'g>, FileError> {
        self.optional_matrix(name, cols, rows)?
            .ok_or_else(|| FileError::MissingTensor { name: name.to_owned() })
    }

    /// The stored bytes of `info`.
    pub(crate) fn bytes(&self, info: &'g TensorInfo) -> &'g [u8] {
        self.gguf.data(info)
    }

    /// All values of `info` decoded to `f32`.
    pub(crate) fn decode(&self, info: &'g TensorInfo) -> Result<Vec<f32>, FileError> {
        let mut out = vec![0.0; info.dims.as_slice().iter().product::<u64>() as usize];
        ares_quant::dequantize(info.ty, self.bytes(info), &mut out).map_err(|source| {
            FileError::TensorType { name: info.name.clone(), ty: info.ty, source }
        })?;
        Ok(out)
    }

    /// The tensor `name` of dimensions `dims` decoded to `f32`, `None` when
    /// the file has none.
    pub(crate) fn optional_floats(
        &mut self,
        name: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<f32>>, FileError> {
        self.find(name, dims)?.map(|info| self.decode(info)).transpose()
    }

    /// The tensor `name` of dimensions `dims` decoded to `f32`.
    pub(crate) fn floats(&mut self, name: &str, dims: &[u64]) -> Result<Vec<f32>, FileError> {
        let info = self.info(name, dims)?;
        self.decode(info)
    }

    /// The vector `name` of `len` values decoded to `f32`.
    pub(crate) fn vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>, FileError> {
        self.floats(name, &[len as u64])
    }

    /// The first tensor of the file that no one took and for which `ours`
    /// holds.
    pub(crate) fn unclaimed(&self, ours: impl Fn(&str) -> bool) -> Option<&'g TensorInfo> {
        self.gguf.tensors().iter().find(|t| ours(&t.name) && !self.claimed.contains(&t.name))
    }

    /// Refuses the first tensor of the file that no one took and for which
    /// `ours` holds: it is not part of `of` (a noun, "the qwen35
    /// architecture").
    pub(crate) fn finish(
        &self,
        of: &'static str,
        ours: impl Fn(&str) -> bool,
    ) -> Result<(), FileError> {
        match self.unclaimed(ours) {
            Some(extra) => Err(FileError::UnexpectedTensor { name: extra.name.clone(), of }),
            None => Ok(()),
        }
    }
}

/// The number of tokens of a model whose embedding table `token_embd.weight`
/// has rows of `n_embd` values: its rows, which the vocabulary of the file
/// (`tokenizer.ggml.tokens`) must match when it has one.
pub(crate) fn vocabulary(gguf: &Gguf, n_embd: usize) -> Result<usize, FileError> {
    let name = "token_embd.weight";
    let embd = gguf.tensor(name).ok_or_else(|| FileError::MissingTensor { name: name.into() })?;
    let dims = embd.dims.as_slice();
    if dims.len() != 2 || dims[0] != n_embd as u64 {
        return Err(FileError::TensorShape {
            name: name.into(),
            found: dims.to_vec(),
            expected: vec![n_embd as u64, dims.get(1).copied().unwrap_or(0)],
        });
    }
    if let Some(tokens) = gguf.array("tokenizer.ggml.tokens")
        && tokens.len() != dims[1]
    {
        return Err(FileError::BadKey {
            key: "tokenizer.ggml.tokens".into(),
            problem: format!("has {} entries but {name} has {} rows", tokens.len(), dims[1]),
        });
    }
    Ok(dims[1] as usize)
}
