//! How many rows a step of prompt rows should have, for the types a model
//! file is in.
//!
//! A wide step pays for the matrices that have a tile kernel (tensor cores):
//! the weights are read once for all its rows. A matrix without one is
//! multiplied by the decode kernels in launches of at most `MAX_BATCH` (eight)
//! rows, and it reads its weights once for every eight rows however many the
//! step has. The matrices that read the same input are multiplied the same
//! way, so one of them without a kernel takes its whole group with it
//! (`decode_in_batches` in `choose.rs`). A step as wide as the first kind of
//! file wants costs the second kind the scratch for the rows and gains it
//! nothing. Every quantized type has a tile kernel now; `f32`, `f16` and
//! `bf16` have none, so the narrow step is for the files whose matrices are
//! mostly in those.
//!
//! The step loop reads a prompt in pieces of [`ares_runtime::Model::prompt_rows`]
//! rows, which is the pipeline group (`Options::pipeline_rows`, 256) but at most
//! the rows of a step, and gives a prompt the whole step while nobody generates;
//! so the rows chosen here are also the widest piece and the most rows the
//! windows of all sequences can take together (`Options::window_rows`, which
//! the server sets to `--parallel` times `--window`).
//!
//! Measured on Qwen3.8-27B (2026-11-12, an RTX 4070 and an RTX 3060, prompts of
//! 2048 tokens read in steps of the given number of rows), with the types as
//! they have tile kernels today. The `Q4_K_S` file on a tensor split of
//! 0.625 / 0.375 reads 870, 952, 1238, 1358, 1363 and 1370 tokens per second in
//! steps of 128, 256, 384, 512, 768 and 1024 rows (a scratch of 59, 90, 122,
//! 153, 216 and 279 MiB on the RTX 4070), so 512 is where the rate stops
//! growing and 128 loses 36 %. The 2-bit file (`IQ2_XXS`) on one RTX 4070 reads
//! 1008, 1131, 1172 and 1213 tokens per second in steps of 128, 256, 512 and
//! 1024 rows (a scratch of 76, 119, 205 and 377 MiB): the wide step pays there
//! too, since the type has tile kernels. Before the tile kernels for `IQ2_XXS`
//! were added the same file on the same card read 275, 272, 268 and 259 tokens
//! per second in steps of 128, 256, 512 and 1024 rows, which is what matrices
//! without a tile kernel do with wide steps. The rule below is made from
//! that and from nothing else about the devices: it reads the types of the
//! file, and which types have a kernel is a fact of the kernels, not of a
//! device. Files in `f16` or `bf16` have not been measured.

use ares_cuda::{Kernels, Precision};

use super::Options;
use crate::qwen35::weights::{Matrix, Mixer, Weights};

/// The rows of a step for a file whose matrices mostly have no tile kernel:
/// room for the windows of the default policy several times over (two
/// sequences of eight rows) and for the groups of rows a split hands over.
pub const BATCHED_ROWS: u32 = 128;

/// The rows of the steps that read prompts for `weights`: those of
/// [`Options::default`] when at least half of the bytes of the matrices are in
/// groups that have tile kernels (always with [`Precision::Exact`], which
/// multiplies with them everywhere), [`BATCHED_ROWS`] otherwise.
pub fn step_rows(weights: &Weights<'_>, precision: Precision) -> u32 {
    let paying = Options::default().max_rows;
    if precision == Precision::Exact {
        return paying;
    }
    let (mut tiled, mut all) = (0u64, 0u64);
    let mut group = |matrices: &[&Matrix<'_>]| {
        let bytes: u64 = matrices.iter().map(|m| m.bytes().len() as u64).sum();
        all += bytes;
        if matrices.iter().all(|m| Kernels::has_prefill_matmul(m.ty())) {
            tiled += bytes;
        }
    };
    for layer in &weights.layers {
        let block = &layer.block;
        group(&[&block.ffn_gate, &block.ffn_up]);
        group(&[&block.ffn_down]);
        match &layer.mixer {
            Mixer::Recurrent(r) => {
                group(&[&r.qkv, &r.gate, &r.beta, &r.alpha]);
                group(&[&r.out]);
            }
            Mixer::Attention(a) => {
                group(&[&a.q, &a.k, &a.v]);
                group(&[&a.out]);
            }
        }
    }
    if tiled * 2 >= all { paying } else { BATCHED_ROWS }
}
