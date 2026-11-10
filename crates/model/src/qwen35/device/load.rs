//! The weights of one device: which bytes of the model file it holds and
//! where they are in its arena.
//!
//! All weights of a device live in one allocation (the driver rounds every
//! allocation up, and a model has hundreds of tensors). A tensor is stored as
//! the file stores it. A device that holds a slice of a matrix gets the rows
//! or the columns of its slice, rearranged at load into a matrix of the same
//! type: the same bytes per value, never decoded.

use std::ops::Range;
use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{Device, DeviceBuffer, DevicePtr, RecurrentShape};
use ares_runtime::ModelError;

use super::placement::{BlockSlice, DevicePlan};
use crate::qwen35::Config;
use crate::qwen35::weights::{Matrix, Mixer, Weights};

/// Every tensor starts at a multiple of this in the arena: more than any
/// weight kernel asks for (16 bytes).
const ALIGN: u64 = 64;

/// A matrix on a device: `n_out` rows of `n_in` values of type `ty`, at
/// offset `at` of the arena.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mat {
    pub ty: GgmlType,
    pub n_in: u32,
    pub n_out: u32,
    pub at: u64,
}

/// A device's slice of a recurrent block.
#[derive(Clone, Debug)]
pub(crate) struct RecurrentW {
    /// The geometry of the slice: its key heads and their value heads.
    pub shape: RecurrentShape,
    pub qkv: Mat,
    pub z: Mat,
    pub beta: Mat,
    pub alpha: Mat,
    pub out: Mat,
    /// Offsets of f32 vectors: the taps of the convolution, the decay bias and
    /// factor per value head, the weight of the gated norm.
    pub taps: u64,
    pub dt_bias: u64,
    pub ssm_a: u64,
    pub ssm_norm: u64,
}

/// A device's slice of an attention block.
#[derive(Clone, Debug)]
pub(crate) struct AttentionW {
    /// Query heads and KV heads of the slice.
    pub n_head: u32,
    pub n_kv: u32,
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub out: Mat,
    pub q_norm: u64,
    pub k_norm: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum MixerW {
    Recurrent(RecurrentW),
    Attention(AttentionW),
    /// The device holds no head of this block: its partial result is zero.
    Empty,
}

#[derive(Clone, Debug)]
pub(crate) struct FfnW {
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

/// A device's slice of a block.
#[derive(Clone, Debug)]
pub(crate) struct BlockW {
    /// The block's index in the model.
    pub layer: usize,
    pub attn_norm: u64,
    pub post_norm: u64,
    pub mixer: MixerW,
    /// `None` when the device holds no row of the feed-forward part.
    pub ffn: Option<FfnW>,
}

/// Where the bytes of a tensor on the device come from.
enum Source<'g> {
    /// Stored bytes as they are.
    Bytes(&'g [u8]),
    /// Rows of a matrix, range after range.
    Rows(Matrix<'g>, Vec<Range<usize>>),
    /// The same columns of every row of a matrix, range after range. The
    /// ranges are whole blocks of the matrix's type.
    Columns(Matrix<'g>, Vec<Range<usize>>),
    /// f32 values.
    Floats(Vec<f32>),
}

fn row_bytes(m: &Matrix<'_>) -> usize {
    m.bytes().len() / m.rows().max(1)
}

impl Source<'_> {
    fn len(&self) -> u64 {
        (match self {
            Source::Bytes(b) => b.len(),
            Source::Rows(m, ranges) => ranges.iter().map(|r| r.len()).sum::<usize>() * row_bytes(m),
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                m.rows() * ranges.iter().map(|r| r.len() / bv * bb).sum::<usize>()
            }
            Source::Floats(f) => f.len() * 4,
        }) as u64
    }

    fn write(
        &self,
        arena: &DeviceBuffer,
        at: u64,
        staging: &mut Vec<u8>,
    ) -> Result<(), ModelError> {
        match self {
            Source::Bytes(b) => arena.write(at, b)?,
            Source::Rows(m, ranges) => {
                let size = row_bytes(m);
                let mut at = at;
                for r in ranges {
                    let bytes = &m.bytes()[r.start * size..r.end * size];
                    arena.write(at, bytes)?;
                    at += bytes.len() as u64;
                }
            }
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                let size = row_bytes(m);
                staging.clear();
                for row in m.bytes().chunks_exact(size) {
                    for r in ranges {
                        staging.extend_from_slice(&row[r.start / bv * bb..r.end / bv * bb]);
                    }
                }
                arena.write(at, staging)?;
            }
            Source::Floats(f) => {
                staging.clear();
                staging.extend(f.iter().flat_map(|v| v.to_le_bytes()));
                arena.write(at, staging)?;
            }
        }
        Ok(())
    }
}

/// The arena being laid out.
#[derive(Default)]
struct Layout<'g> {
    items: Vec<(u64, Source<'g>)>,
    len: u64,
}

impl<'g> Layout<'g> {
    fn push(&mut self, source: Source<'g>) -> u64 {
        let at = self.len.next_multiple_of(ALIGN);
        self.len = at + source.len();
        self.items.push((at, source));
        at
    }

    fn floats(&mut self, values: &[f32]) -> u64 {
        self.push(Source::Floats(values.to_vec()))
    }

    /// Consecutive rows of `m` as a matrix of its own.
    fn row_range(&mut self, m: &Matrix<'g>, rows: Range<usize>) -> Mat {
        self.rows(m, std::iter::once(rows).collect())
    }

    /// Consecutive columns of every row of `m` as a matrix of its own.
    fn column_range(&mut self, m: &Matrix<'g>, columns: Range<usize>) -> Result<Mat, ModelError> {
        self.columns(m, std::iter::once(columns).collect())
    }

    /// Rows `rows` of `m` (range after range) as a matrix of its own.
    fn rows(&mut self, m: &Matrix<'g>, rows: Vec<Range<usize>>) -> Mat {
        let n_out = rows.iter().map(|r| r.len()).sum::<usize>() as u32;
        let source = match rows.as_slice() {
            [one] => {
                let size = row_bytes(m);
                Source::Bytes(&m.bytes()[one.start * size..one.end * size])
            }
            _ => Source::Rows(*m, rows),
        };
        Mat { ty: m.ty(), n_in: m.cols() as u32, n_out, at: self.push(source) }
    }

    /// Columns `columns` of every row of `m` as a matrix of its own.
    fn columns(&mut self, m: &Matrix<'g>, columns: Vec<Range<usize>>) -> Result<Mat, ModelError> {
        let bv = m.ty().block_values() as usize;
        if columns.iter().any(|r| !r.start.is_multiple_of(bv) || !r.end.is_multiple_of(bv)) {
            return Err(ModelError::Invalid(format!(
                "columns {columns:?} of a {} matrix are not whole blocks of {bv} values",
                m.ty()
            )));
        }
        let n_in = columns.iter().map(|r| r.len()).sum::<usize>() as u32;
        let source = match columns.as_slice() {
            [one] if one.start == 0 && one.end == m.cols() => Source::Bytes(m.bytes()),
            _ => Source::Columns(*m, columns),
        };
        Ok(Mat { ty: m.ty(), n_in, n_out: m.rows() as u32, at: self.push(source) })
    }
}

/// The value-head ranges that belong to key heads `keys`, each head
/// `per_head` items wide: value head `h` reads key head `h mod n_k_heads`.
fn value_ranges(c: &Config, keys: &Range<u32>, per_head: usize) -> Vec<Range<usize>> {
    let r = &c.recurrent;
    let (k0, k1) = (keys.start as usize, keys.end as usize);
    (0..r.n_v_heads / r.n_k_heads)
        .map(|tile| (tile * r.n_k_heads + k0) * per_head..(tile * r.n_k_heads + k1) * per_head)
        .collect()
}

fn gather(values: &[f32], ranges: &[Range<usize>]) -> Vec<f32> {
    ranges.iter().flat_map(|r| values[r.clone()].iter().copied()).collect()
}

/// The weights of one device.
#[derive(Debug)]
pub(crate) struct DeviceWeights {
    arena: DeviceBuffer,
    pub blocks: Vec<BlockW>,
    /// The token embedding table, on the device that has it.
    pub embedding: Option<Mat>,
    pub output_norm: u64,
    /// The rows of the output head on this device and the token of the first.
    pub head: Option<(Mat, u32)>,
    /// How long the upload took.
    pub upload: Duration,
}

impl DeviceWeights {
    /// The device address of offset `at` of the arena.
    pub fn at(&self, at: u64) -> DevicePtr {
        self.arena.ptr().offset(at)
    }

    /// The address of a matrix.
    pub fn mat(&self, m: &Mat) -> DevicePtr {
        self.at(m.at)
    }

    /// The bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.arena.len()
    }

    /// Lays out and uploads what `plan` gives this device.
    pub fn load(
        device: &Device,
        ledger: &mut Ledger,
        c: &Config,
        w: &Weights<'_>,
        plan: &DevicePlan,
    ) -> Result<Self, ModelError> {
        let mut layout = Layout::default();
        let mut blocks = Vec::new();
        for (layer, slice) in plan.blocks.iter().enumerate() {
            let Some(slice) = slice else { continue };
            blocks.push(block(&mut layout, c, w, layer, slice)?);
        }
        let embedding = plan.embedding.then(|| {
            let m = &w.token_embd;
            layout.row_range(m, 0..m.rows())
        });
        let output_norm = layout.floats(&w.output_norm);
        let head = if plan.head.is_empty() {
            None
        } else {
            let m = w.head();
            let rows = plan.head.start as usize..plan.head.end as usize;
            let offset = (rows.start * row_bytes(m)) as u64;
            // A head that is the embedding table reads the same bytes when the
            // table is here and the slice starts at an aligned row.
            let mat = match (&embedding, w.output.is_none()) {
                (Some(table), true) if offset.is_multiple_of(ALIGN) => Mat {
                    ty: table.ty,
                    n_in: table.n_in,
                    n_out: rows.len() as u32,
                    at: table.at + offset,
                },
                _ => layout.row_range(m, rows),
            };
            Some((mat, plan.head.start))
        };

        let bytes = layout.len.next_multiple_of(ALIGN).max(ALIGN);
        ledger.reserve("qwen35 weights", bytes)?;
        let arena = DeviceBuffer::new(device, bytes)?;
        let begun = Instant::now();
        let mut staging = Vec::new();
        for (at, source) in &layout.items {
            source.write(&arena, *at, &mut staging)?;
        }
        Ok(Self { arena, blocks, embedding, output_norm, head, upload: begun.elapsed() })
    }
}

fn block<'g>(
    layout: &mut Layout<'g>,
    c: &Config,
    w: &Weights<'g>,
    layer: usize,
    slice: &BlockSlice,
) -> Result<BlockW, ModelError> {
    let l = &w.layers[layer];
    let attn_norm = layout.floats(&l.block.attn_norm);
    let post_norm = layout.floats(&l.block.post_attention_norm);
    let heads = slice.heads.start as usize..slice.heads.end as usize;
    let mixer = match &l.mixer {
        _ if heads.is_empty() => MixerW::Empty,
        Mixer::Recurrent(r) => {
            let rc = &c.recurrent;
            let (dk, kd) = (rc.head_k_dim, rc.key_dim());
            let values = value_ranges(c, &slice.heads, rc.head_v_dim);
            let per_head = value_ranges(c, &slice.heads, 1);
            // q of the key heads, k of the key heads, v of their value heads.
            let mut channels = Vec::with_capacity(2 + values.len());
            channels.push(heads.start * dk..heads.end * dk);
            channels.push(kd + heads.start * dk..kd + heads.end * dk);
            channels.extend(values.iter().map(|v| 2 * kd + v.start..2 * kd + v.end));
            let k = rc.conv_kernel;
            let taps: Vec<f32> =
                channels.iter().flat_map(|r| r.start * k..r.end * k).map(|i| r.conv[i]).collect();
            let n_k = heads.len() as u32;
            MixerW::Recurrent(RecurrentW {
                shape: RecurrentShape {
                    n_k_heads: n_k,
                    n_v_heads: n_k * (rc.n_v_heads / rc.n_k_heads) as u32,
                    head_dim: dk as u32,
                    conv_kernel: k as u32,
                },
                qkv: layout.rows(&r.qkv, channels),
                z: layout.rows(&r.gate, values.clone()),
                beta: layout.rows(&r.beta, per_head.clone()),
                alpha: layout.rows(&r.alpha, per_head.clone()),
                out: layout.columns(&r.out, values)?,
                taps: layout.floats(&taps),
                dt_bias: layout.floats(&gather(&r.dt_bias, &per_head)),
                ssm_a: layout.floats(&gather(&r.a, &per_head)),
                ssm_norm: layout.floats(&r.norm),
            })
        }
        Mixer::Attention(a) => {
            let ac = &c.attention;
            let (h, per_kv) = (ac.head_dim, ac.n_head / ac.n_head_kv);
            let q_heads = heads.start * per_kv..heads.end * per_kv;
            MixerW::Attention(AttentionW {
                n_head: q_heads.len() as u32,
                n_kv: heads.len() as u32,
                q: layout.row_range(&a.q, q_heads.start * 2 * h..q_heads.end * 2 * h),
                k: layout.row_range(&a.k, heads.start * h..heads.end * h),
                v: layout.row_range(&a.v, heads.start * h..heads.end * h),
                out: layout.column_range(&a.out, q_heads.start * h..q_heads.end * h)?,
                q_norm: layout.floats(&a.q_norm),
                k_norm: layout.floats(&a.k_norm),
            })
        }
    };
    let rows = slice.ffn.start as usize..slice.ffn.end as usize;
    let ffn = if rows.is_empty() {
        None
    } else {
        Some(FfnW {
            gate: layout.row_range(&l.block.ffn_gate, rows.clone()),
            up: layout.row_range(&l.block.ffn_up, rows.clone()),
            down: layout.column_range(&l.block.ffn_down, rows)?,
        })
    };
    Ok(BlockW { layer, attn_norm, post_norm, mixer, ffn })
}
