//! The weights of a projector: which tensors there are, of which shape, and
//! where their bytes are.
//!
//! The table is built against a [`Config`] and is exact: a tensor that is
//! missing, has other dimensions than the configuration gives it, cannot be
//! decoded, or is in the file without being part of the projector is an error
//! that names it. The matrices of the blocks and the merger stay as stored
//! (borrowed from the file); the patch kernels, the position table and the
//! vectors are decoded to `f32` once.

use ares_gguf::Gguf;

pub use crate::files::Matrix;
use crate::files::Table;
use crate::qwen3vl::config::Config;
use crate::qwen3vl::error::Result;

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

impl<'g> Weights<'g> {
    /// Builds the table of a file against its configuration.
    pub fn load(gguf: &'g Gguf, c: &Config) -> Result<Self> {
        let mut table = Table::new(gguf);
        let t = &mut table;
        let (side, e, group) = (c.table_side, c.n_embd, 4 * c.n_embd);
        let kernel = [c.patch as u64, c.patch as u64, 3, e as u64];
        let weights = Self {
            patch: [
                t.floats("v.patch_embd.weight", &kernel)?,
                t.floats("v.patch_embd.weight.1", &kernel)?,
            ],
            patch_bias: t.floats("v.patch_embd.bias", &[e as u64])?,
            position: t.floats("v.position_embd.weight", &[e as u64, (side * side) as u64])?,
            blocks: (0..c.n_layer)
                .map(|i| {
                    let name = |part: &str| format!("v.blk.{i}.{part}");
                    Ok(Block {
                        ln1: norm(t, &name("ln1"), e)?,
                        qkv: linear(t, &name("attn_qkv"), e, 3 * e)?,
                        out: linear(t, &name("attn_out"), e, e)?,
                        ln2: norm(t, &name("ln2"), e)?,
                        up: linear(t, &name("ffn_up"), e, c.n_ff)?,
                        down: linear(t, &name("ffn_down"), c.n_ff, e)?,
                    })
                })
                .collect::<Result<_>>()?,
            post_ln: norm(t, "v.post_ln", e)?,
            mm0: linear(t, "mm.0", group, group)?,
            mm2: linear(t, "mm.2", group, c.n_out)?,
        };
        table.finish("the projector", |_| true)?;
        Ok(weights)
    }
}

fn norm(t: &mut Table<'_>, name: &str, len: usize) -> Result<Norm> {
    Ok(Norm {
        weight: t.vector(&format!("{name}.weight"), len)?,
        bias: t.vector(&format!("{name}.bias"), len)?,
    })
}

fn linear<'g>(t: &mut Table<'g>, name: &str, cols: usize, rows: usize) -> Result<Linear<'g>> {
    Ok(Linear {
        weight: t.matrix(&format!("{name}.weight"), cols, rows)?,
        bias: t.vector(&format!("{name}.bias"), rows)?,
    })
}
