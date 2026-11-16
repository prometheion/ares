//! The dense weights of one device: which tensors of the model file it holds
//! and where they are in its arena.
//!
//! All dense weights of a device live in one allocation (the driver rounds
//! every allocation up, and a layer has two dozen tensors), every tensor in
//! the bytes the file stores. Vectors that the reference keeps as `f32` (norm
//! weights, convolution taps) are uploaded as `f32`.
//!
//! What is not here: the routed experts, which live in the slots of the
//! expert store; the n-gram table, whose rows the host fetches per step; and
//! the token embedding, whose rows the host decodes for the tokens of a step.
//!
//! A device that shares its layers with another holds a slice of each
//! ([`LayerShare`]): of a recurrent layer some key heads with the value heads
//! that read them, which are rows of the four projections and columns of the
//! output projection, gathered here into matrices of their own.

use std::ops::Range;
use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{Device, DeviceBuffer, DevicePtr, ExpertShape, RecurrentShape, WeightRange};
use ares_runtime::ModelError;

use crate::qwen4exp::Config;
use crate::qwen4exp::weights::{
    DraftWeights, Layer, Matrix, Mixer, RecurrentWeights, StreamMixer, Weights,
};

/// What a part is to the residual streams of the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The part holds whole layers: the streams, the mixers and everything
    /// the layers compute (one device, or a split by layers).
    Whole,
    /// The part holds the streams, every mixer, the router and the n-gram
    /// block of layers it shares with a worker, and its share of what they
    /// compute.
    Mixer,
    /// The part gets the mixed input of every half block from the mixer and
    /// returns its share of the output. It never sees the streams.
    Worker,
}

/// What a part holds of one layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerShare {
    /// The key heads of a recurrent layer, with the value heads that read
    /// them, or the KV heads of an attention layer, that the part computes.
    /// May be empty.
    pub heads: Range<u32>,
    /// Whether the part computes the shared expert.
    pub shared: bool,
}

impl LayerShare {
    /// The whole of layer `layer`.
    pub fn whole(c: &Config, layer: usize) -> Self {
        let heads =
            if c.is_attention(layer) { c.attention.n_head_kv } else { c.recurrent.n_k_heads };
        Self { heads: 0..heads as u32, shared: true }
    }
}

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
    /// What the size of a float matrix's weights leaves of the tile kernels, which hold
    /// them as f16 (a wide step asks, `part/wide.rs`); [`WeightRange::Any`] for every other
    /// type, and for every matrix of a layout that only measures sizes.
    pub range: WeightRange,
}

/// A stream mixer: the offset of its norm weights (`f32`, a weight per value
/// and stream) and its projections.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MixerW {
    pub norm: u64,
    pub down: Mat,
    pub up: Mat,
    /// `None` for the mixer in front of the output head.
    pub inject: Option<Mat>,
}

/// A recurrent layer's inner part.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecurrentW {
    pub shape: RecurrentShape,
    pub qkv: Mat,
    pub z: Mat,
    pub alpha: Mat,
    pub beta: Mat,
    pub out: Mat,
    /// Offsets of `f32` vectors: the taps of the convolution, the decay bias
    /// and factor per value head, the weight of the gated norm.
    pub taps: u64,
    pub dt_bias: u64,
    pub ssm_a: u64,
    pub ssm_norm: u64,
}

/// An attention layer's inner part with its indexer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionW {
    pub q: Mat,
    pub k: Mat,
    pub v: Mat,
    pub out: Mat,
    pub indexer_q: Mat,
    pub indexer_k: Mat,
    /// Offsets of the `f32` norm weights.
    pub q_norm: u64,
    pub k_norm: u64,
    pub indexer_q_norm: u64,
    pub indexer_k_norm: u64,
}

/// The device part of the n-gram block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NgramW {
    pub key: Mat,
    pub value: Mat,
    /// Offsets of `f32` vectors.
    pub norm_key: u64,
    pub norm_query: u64,
    pub norm_conv: u64,
    pub taps: u64,
}

/// The shared expert of a layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedW {
    /// The scalar gate as a matrix of one row (F32).
    pub inp: Mat,
    pub gate: Mat,
    pub up: Mat,
    pub down: Mat,
}

/// The dense part of a layer's mixture of experts: the router and the shared
/// expert, each where the part has it. The routed experts are in the store's
/// slots.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeW {
    pub shape: ExpertShape,
    /// The router matrix (F32, `n_experts` rows of `n_embd`): not on a worker.
    pub router: Option<u64>,
    /// The range of the router's weights, as [`Mat::range`].
    pub router_range: WeightRange,
    pub shared: Option<SharedW>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CoreW {
    Recurrent(RecurrentW),
    Attention(AttentionW),
    /// The part has no head of the layer.
    None,
}

/// One layer of the model on a device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerW {
    /// The layer's number in the model.
    pub layer: usize,
    /// The mixers in front of the two halves: not on a worker.
    pub mixers: Option<(MixerW, MixerW)>,
    pub core: CoreW,
    pub moe: MoeW,
    pub ngram: Option<NgramW>,
}

/// The mixer in front of the output head and the head.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HeadW {
    pub mixer: MixerW,
    pub output: Mat,
}

/// What makes the input of the model's draft block: the norm weights (`f32`
/// offsets) and the projections of the paired token's embedding and of the
/// residual streams.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DraftInW {
    pub norm_embedding: u64,
    pub norm_hidden: u64,
    pub fc_embedding: Mat,
    pub fc_hidden: Mat,
}

/// The offset that marks the one matrix a part reads in another part's arena
/// (the output head, which a draft block shares with the model).
const BORROWED: u64 = u64::MAX;

/// What goes to one place of the arena.
enum Source<'g> {
    /// Bytes of the mapped file.
    File(&'g [u8]),
    /// Values the reference decoded.
    Floats(Vec<f32>),
    /// Rows of a matrix, range after range.
    Rows(Matrix<'g>, Vec<Range<usize>>),
    /// The same columns of every row of a matrix, range after range, in
    /// whole blocks of the matrix's type.
    Columns(Matrix<'g>, Vec<Range<usize>>),
}

fn row_bytes(m: &Matrix<'_>) -> usize {
    m.bytes().len() / m.rows().max(1)
}

impl Source<'_> {
    fn len(&self) -> u64 {
        (match self {
            Source::File(bytes) => bytes.len(),
            Source::Floats(values) => values.len() * 4,
            Source::Rows(m, ranges) => ranges.iter().map(Range::len).sum::<usize>() * row_bytes(m),
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                m.rows() * ranges.iter().map(|r| r.len() / bv * bb).sum::<usize>()
            }
        }) as u64
    }

    /// The bytes as they go to the arena; `staging` holds them when they are
    /// gathered.
    fn bytes<'a>(&'a self, staging: &'a mut Vec<u8>) -> &'a [u8] {
        staging.clear();
        match self {
            Source::File(bytes) => return bytes,
            Source::Floats(values) => staging.extend(values.iter().flat_map(|v| v.to_le_bytes())),
            Source::Rows(m, ranges) => {
                let size = row_bytes(m);
                for r in ranges {
                    staging.extend_from_slice(&m.bytes()[r.start * size..r.end * size]);
                }
            }
            Source::Columns(m, ranges) => {
                let (bv, bb) = (m.ty().block_values() as usize, m.ty().block_bytes() as usize);
                for row in m.bytes().chunks_exact(row_bytes(m)) {
                    for r in ranges {
                        staging.extend_from_slice(&row[r.start / bv * bb..r.end / bv * bb]);
                    }
                }
            }
        }
        staging
    }
}

/// The ranges of the value heads that read the key heads `keys`, each head
/// `per_head` items wide: value head `h` reads key head `h mod n_k_heads`.
fn value_ranges(c: &Config, keys: &Range<usize>, per_head: usize) -> Vec<Range<usize>> {
    let r = &c.recurrent;
    (0..r.n_v_heads / r.n_k_heads)
        .map(|tile| {
            (tile * r.n_k_heads + keys.start) * per_head..(tile * r.n_k_heads + keys.end) * per_head
        })
        .collect()
}

fn gather(values: &[f32], ranges: &[Range<usize>]) -> Vec<f32> {
    ranges.iter().flat_map(|r| values[r.clone()].iter().copied()).collect()
}

/// The key heads a recurrent layer can be cut at: the fewest whose value
/// columns are whole blocks of the output projection's type.
pub(crate) fn key_head_unit(c: &Config, r: &RecurrentWeights<'_>) -> usize {
    let (block, head) = (r.out.ty().block_values() as usize, c.recurrent.head_dim);
    let mut unit = 1;
    while !(unit * head).is_multiple_of(block) && unit < c.recurrent.n_k_heads {
        unit += 1;
    }
    unit
}

/// An arena being laid out: what goes where.
#[derive(Default)]
struct Layout<'g> {
    len: u64,
    parts: Vec<(u64, Source<'g>)>,
    /// Whether the float matrices are read for the range of their weights: for the layout
    /// that is uploaded, not for the ones that only measure sizes (a pass over the bytes
    /// of every float matrix).
    ranges: bool,
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

impl<'g> Layout<'g> {
    fn push(&mut self, source: Source<'g>) -> u64 {
        let at = self.len;
        self.len = (at + source.len().max(1)).next_multiple_of(ALIGN);
        self.parts.push((at, source));
        at
    }

    fn floats(&mut self, values: &[f32]) -> u64 {
        self.push(Source::Floats(values.to_vec()))
    }

    /// The range of the weights of `m`: of the whole matrix, also where a part holds some
    /// of its rows or columns (no part of it is larger than the whole).
    fn range(&self, m: &Matrix<'_>) -> WeightRange {
        if self.ranges { WeightRange::of(m.ty(), m.bytes()) } else { WeightRange::Any }
    }

    fn matrix(&mut self, m: &Matrix<'g>) -> Mat {
        let at = self.push(Source::File(m.bytes()));
        Mat { ty: m.ty(), n_in: m.cols() as u32, n_out: m.rows() as u32, at, range: self.range(m) }
    }

    /// Rows `rows` of `m`, range after range, as a matrix of their own.
    fn rows(&mut self, m: &Matrix<'g>, rows: Vec<Range<usize>>) -> Mat {
        let n_out = rows.iter().map(Range::len).sum::<usize>() as u32;
        let at = self.push(Source::Rows(*m, rows));
        Mat { ty: m.ty(), n_in: m.cols() as u32, n_out, at, range: self.range(m) }
    }

    /// Columns `columns` of every row of `m` as a matrix of their own.
    fn columns(&mut self, m: &Matrix<'g>, columns: Vec<Range<usize>>) -> Result<Mat, ModelError> {
        let bv = m.ty().block_values() as usize;
        if columns.iter().any(|r| !r.start.is_multiple_of(bv) || !r.end.is_multiple_of(bv)) {
            return Err(invalid(format!(
                "columns {columns:?} of a {} matrix are not whole blocks of {bv} values",
                m.ty()
            )));
        }
        let n_in = columns.iter().map(Range::len).sum::<usize>() as u32;
        let at = self.push(Source::Columns(*m, columns));
        Ok(Mat { ty: m.ty(), n_in, n_out: m.rows() as u32, at, range: self.range(m) })
    }

    /// The key heads `heads` of a recurrent layer with their value heads: a
    /// narrower layer of the same kind.
    fn recurrent(
        &mut self,
        c: &Config,
        r: &RecurrentWeights<'g>,
        heads: Range<usize>,
    ) -> Result<RecurrentW, ModelError> {
        let rc = &c.recurrent;
        let whole = heads == (0..rc.n_k_heads);
        let (s, kd) = (rc.head_dim, rc.key_dim());
        let values = value_ranges(c, &heads, s);
        let per_head = value_ranges(c, &heads, 1);
        // q of the key heads, k of the key heads, v of their value heads.
        let mut channels =
            vec![heads.start * s..heads.end * s, kd + heads.start * s..kd + heads.end * s];
        channels.extend(values.iter().map(|v| 2 * kd + v.start..2 * kd + v.end));
        let k = rc.conv_kernel;
        let taps: Vec<f32> =
            channels.iter().flat_map(|r| r.start * k..r.end * k).map(|i| r.conv[i]).collect();
        let n_k = heads.len() as u32;
        Ok(RecurrentW {
            shape: RecurrentShape {
                n_k_heads: n_k,
                n_v_heads: n_k * (rc.n_v_heads / rc.n_k_heads) as u32,
                head_dim: s as u32,
                conv_kernel: k as u32,
            },
            qkv: if whole { self.matrix(&r.qkv) } else { self.rows(&r.qkv, channels) },
            z: if whole { self.matrix(&r.gate) } else { self.rows(&r.gate, values.clone()) },
            alpha: if whole {
                self.matrix(&r.alpha)
            } else {
                self.rows(&r.alpha, per_head.clone())
            },
            beta: if whole { self.matrix(&r.beta) } else { self.rows(&r.beta, per_head.clone()) },
            out: if whole { self.matrix(&r.out) } else { self.columns(&r.out, values)? },
            taps: self.floats(&taps),
            dt_bias: self.floats(&gather(&r.dt_bias, &per_head)),
            ssm_a: self.floats(&gather(&r.a, &per_head)),
            ssm_norm: self.floats(&r.norm),
        })
    }

    /// A matrix of a stream mixer or of the n-gram block: these multiply the `f32` values a
    /// kernel wrote, unquantized ([`Part::exact_product`](super::part::Part), the fused read),
    /// in any weight type and in the bytes the file stores.
    fn exact_matrix(&mut self, m: &Matrix<'g>) -> Mat {
        self.matrix(m)
    }

    fn mixer(&mut self, m: &StreamMixer<'g>) -> MixerW {
        MixerW {
            norm: self.floats(&m.norm),
            down: self.exact_matrix(&m.down),
            up: self.exact_matrix(&m.up),
            inject: m.inject.as_ref().map(|i| self.exact_matrix(i)),
        }
    }

    fn layer(
        &mut self,
        c: &Config,
        layer: usize,
        w: &Layer<'g>,
        role: Role,
        share: &LayerShare,
    ) -> Result<LayerW, ModelError> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let streams = role != Role::Worker;
        let mixers = if streams {
            Some((self.mixer(&w.attn_mixer), self.mixer(&w.ffn_mixer)))
        } else {
            None
        };
        let heads = share.heads.start as usize..share.heads.end as usize;
        let core = match &w.mixer {
            _ if heads.is_empty() => CoreW::None,
            Mixer::Recurrent(r) => {
                let unit = key_head_unit(c, r);
                if heads.end > c.recurrent.n_k_heads
                    || !heads.start.is_multiple_of(unit)
                    || !heads.end.is_multiple_of(unit)
                {
                    return Err(invalid(format!(
                        "{}: the key heads {heads:?} of {} are not whole units of {unit}",
                        name("attn_qkv"),
                        c.recurrent.n_k_heads
                    )));
                }
                CoreW::Recurrent(self.recurrent(c, r, heads)?)
            }
            Mixer::Attention(_) if heads != (0..c.attention.n_head_kv) => {
                return Err(invalid(format!(
                    "{}: the KV heads {heads:?} of {}; an attention layer is not cut between                      devices yet",
                    name("attn_q"),
                    c.attention.n_head_kv
                )));
            }
            Mixer::Attention(a) => CoreW::Attention(AttentionW {
                q: self.matrix(&a.q),
                k: self.matrix(&a.k),
                v: self.matrix(&a.v),
                out: self.matrix(&a.out),
                indexer_q: self.matrix(&a.indexer_q),
                indexer_k: self.matrix(&a.indexer_k),
                q_norm: self.floats(&a.q_norm),
                k_norm: self.floats(&a.k_norm),
                indexer_q_norm: self.floats(&a.indexer_q_norm),
                indexer_k_norm: self.floats(&a.indexer_k_norm),
            }),
        };
        let moe = &w.moe;
        if moe.gate.ty() != moe.up.ty() {
            return Err(invalid(format!(
                "{}: gate is {} and up is {}; the expert kernels take one type for both",
                name("ffn_*_exps"),
                moe.gate.ty(),
                moe.up.ty()
            )));
        }
        let shape = ExpertShape {
            n_embd: c.n_embd as u32,
            n_ff: c.experts.ff as u32,
            n_experts: c.experts.count as u32,
            top_k: c.experts.used as u32,
            gate_ty: moe.gate.ty(),
            down_ty: moe.down.ty(),
        };
        shape.check()?;
        let router = match moe.router.ty() {
            _ if !streams => None,
            GgmlType::F32 => Some(self.push(Source::File(moe.router.bytes()))),
            // The router kernel multiplies f32 rows: a router in another float
            // type (the draft block's file has BF16) is widened once, here.
            GgmlType::F16 | GgmlType::BF16 => {
                let (rows, cols) = (moe.router.rows(), moe.router.cols());
                let mut values = vec![0.0f32; rows * cols];
                for (row, out) in values.chunks_exact_mut(cols).enumerate() {
                    moe.router.decode_row(row, out);
                }
                Some(self.floats(&values))
            }
            other => {
                return Err(invalid(format!(
                    "{} is {other}: the router kernel takes a float type",
                    name("ffn_gate_inp"),
                )));
            }
        };
        let router_range = if streams { self.range(&moe.router) } else { WeightRange::Any };
        let largest = moe.shared_gate_inp.iter().fold(0.0f32, |m, v| {
            // A value that is not a number counts as beyond every range.
            if v.is_nan() { f32::INFINITY } else { m.max(v.abs()) }
        });
        let shared = share.shared.then(|| SharedW {
            inp: Mat {
                ty: GgmlType::F32,
                n_in: c.n_embd as u32,
                n_out: 1,
                at: self.floats(&moe.shared_gate_inp),
                range: WeightRange::of_largest(largest),
            },
            gate: self.matrix(&moe.shared_gate),
            up: self.matrix(&moe.shared_up),
            down: self.matrix(&moe.shared_down),
        });
        let moe = MoeW { shape, router, router_range, shared };
        let ngram = w.ngram.as_ref().filter(|_| streams).map(|g| NgramW {
            key: self.exact_matrix(&g.key),
            value: self.exact_matrix(&g.value),
            norm_key: self.floats(&g.norm_key),
            norm_query: self.floats(&g.norm_query),
            norm_conv: self.floats(&g.norm_conv),
            taps: self.floats(&g.conv),
        });
        Ok(LayerW { layer, mixers, core, moe, ngram })
    }

    fn head(&mut self, w: &Weights<'g>) -> HeadW {
        HeadW { mixer: self.mixer(&w.output_mixer), output: self.matrix(&w.output) }
    }

    fn draft_in(&mut self, w: &DraftWeights<'g>) -> DraftInW {
        DraftInW {
            norm_embedding: self.floats(&w.norm_embedding),
            norm_hidden: self.floats(&w.norm_hidden),
            fc_embedding: self.matrix(&w.fc_embedding),
            fc_hidden: self.matrix(&w.fc_hidden),
        }
    }
}

/// Bytes of what a part of `role` holds of layer `layer` in an arena.
pub(crate) fn layer_bytes(
    c: &Config,
    w: &Weights<'_>,
    layer: usize,
    role: Role,
    share: &LayerShare,
) -> Result<u64, ModelError> {
    let mut layout = Layout::default();
    layout.layer(c, layer, &w.layers[layer], role, share)?;
    Ok(layout.len)
}

/// Bytes of the final mixer and the output head in an arena.
pub(crate) fn head_bytes(w: &Weights<'_>) -> u64 {
    let mut layout = Layout::default();
    layout.head(w);
    layout.len
}

/// The dense weights of one device.
#[derive(Debug)]
pub(crate) struct DeviceWeights {
    arena: DeviceBuffer,
    pub layers: Vec<LayerW>,
    pub head: Option<HeadW>,
    /// The input projections of a draft block; `None` for a part of the model.
    pub draft_in: Option<DraftInW>,
    /// Where the matrix at [`BORROWED`] is: in the arena of another part on
    /// the same device, which outlives this one.
    borrowed: DevicePtr,
    upload: Duration,
}

impl DeviceWeights {
    /// The address of offset `at` of the arena.
    pub fn at(&self, at: u64) -> DevicePtr {
        self.arena.ptr().offset(at)
    }

    /// The address of a matrix.
    pub fn mat(&self, m: &Mat) -> DevicePtr {
        if m.at == BORROWED { self.borrowed } else { self.at(m.at) }
    }

    /// The output head and where it is on the device, for a draft block on
    /// the same device to read it. `None` for a part without the head.
    pub fn output(&self) -> Option<(Mat, DevicePtr)> {
        self.head.map(|head| (head.output, self.mat(&head.output)))
    }

    /// Bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.arena.len()
    }

    /// How long the upload took, the reads of the file included.
    pub fn upload_time(&self) -> Duration {
        self.upload
    }

    /// Lays out and uploads what a part of `role` holds of the layers from
    /// `first` on (`shares` says what of each), and the final mixer and the
    /// output head when `head` is set, after reserving them in the ledger.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        ledger: &mut Ledger,
        c: &Config,
        w: &Weights<'_>,
        role: Role,
        first: usize,
        shares: &[LayerShare],
        head: bool,
    ) -> Result<Self, ModelError> {
        let mut layout = Layout { ranges: true, ..Layout::default() };
        let on_device = shares
            .iter()
            .enumerate()
            .map(|(i, share)| layout.layer(c, first + i, &w.layers[first + i], role, share))
            .collect::<Result<Vec<_>, _>>()?;
        let head = head.then(|| layout.head(w));
        let len = layout.len.max(ALIGN);
        ledger.reserve("qwen4exp dense weights", len)?;
        let arena = DeviceBuffer::new(device, len)?;
        let started = Instant::now();
        let mut staging = Vec::new();
        for (at, source) in &layout.parts {
            arena.write(*at, source.bytes(&mut staging))?;
        }
        Ok(Self {
            arena,
            layers: on_device,
            head,
            draft_in: None,
            borrowed: DevicePtr::NULL,
            upload: started.elapsed(),
        })
    }

    /// Lays out and uploads the dense weights of a draft block: its one layer
    /// (`c` and `w` describe it as a model of one layer), its input projections
    /// and its own final mixer. Its output head is `output`, the model's, which
    /// is on `device` already.
    pub fn load_block(
        device: &Device,
        ledger: &mut Ledger,
        c: &Config,
        w: &DraftWeights<'_>,
        output: (Mat, DevicePtr),
    ) -> Result<Self, ModelError> {
        let mut layout = Layout { ranges: true, ..Layout::default() };
        let layer = layout.layer(c, 0, &w.layer, Role::Whole, &LayerShare::whole(c, 0))?;
        let mixer = layout.mixer(&w.output_mixer);
        let draft_in = layout.draft_in(w);
        let len = layout.len.max(ALIGN);
        ledger.reserve("qwen4exp draft block weights", len)?;
        let arena = DeviceBuffer::new(device, len)?;
        let started = Instant::now();
        let mut staging = Vec::new();
        for (at, source) in &layout.parts {
            arena.write(*at, source.bytes(&mut staging))?;
        }
        Ok(Self {
            arena,
            layers: vec![layer],
            head: Some(HeadW { mixer, output: Mat { at: BORROWED, ..output.0 } }),
            draft_in: Some(draft_in),
            borrowed: output.1,
            upload: started.elapsed(),
        })
    }
}
