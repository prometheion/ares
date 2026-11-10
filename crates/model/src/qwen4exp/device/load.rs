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

use std::ops::Range;
use std::time::{Duration, Instant};

use ares_core::{GgmlType, Ledger};
use ares_cuda::{Device, DeviceBuffer, DevicePtr, ExpertShape, RecurrentShape};
use ares_runtime::ModelError;

use crate::qwen4exp::Config;
use crate::qwen4exp::weights::{Layer, Matrix, Mixer, StreamMixer, Weights};

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

/// The dense part of a layer's mixture of experts: the router and the shared
/// expert. The routed experts are in the store's slots.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeW {
    pub shape: ExpertShape,
    /// The router matrix (F32, `n_experts` rows of `n_embd`).
    pub router: u64,
    /// The scalar gate of the shared expert as a matrix of one row.
    pub shared_inp: Mat,
    pub shared_gate: Mat,
    pub shared_up: Mat,
    pub shared_down: Mat,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CoreW {
    Recurrent(RecurrentW),
    Attention(AttentionW),
}

/// One layer of the model on a device.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerW {
    /// The layer's number in the model.
    pub layer: usize,
    pub attn_mixer: MixerW,
    pub ffn_mixer: MixerW,
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

/// What goes to one place of the arena.
enum Source<'g> {
    /// Bytes of the mapped file.
    File(&'g [u8]),
    /// Values the reference decoded.
    Floats(Vec<f32>),
}

impl Source<'_> {
    fn len(&self) -> u64 {
        match self {
            Source::File(bytes) => bytes.len() as u64,
            Source::Floats(values) => values.len() as u64 * 4,
        }
    }
}

/// An arena being laid out: what goes where.
#[derive(Default)]
struct Layout<'g> {
    len: u64,
    parts: Vec<(u64, Source<'g>)>,
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

    fn matrix(&mut self, m: &Matrix<'g>) -> Mat {
        let at = self.push(Source::File(m.bytes()));
        Mat { ty: m.ty(), n_in: m.cols() as u32, n_out: m.rows() as u32, at }
    }

    /// A matrix of the stream mixer: these multiply the `f32` values a mixer
    /// kernel wrote, which only the float weight kernels read.
    fn float_matrix(&mut self, what: &str, m: &Matrix<'g>) -> Result<Mat, ModelError> {
        if !matches!(m.ty(), GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(invalid(format!(
                "{what} is {}: the device step multiplies it with unquantized activations and \
                 takes F32, F16 and BF16 there",
                m.ty()
            )));
        }
        Ok(self.matrix(m))
    }

    fn mixer(&mut self, what: &str, m: &StreamMixer<'g>) -> Result<MixerW, ModelError> {
        Ok(MixerW {
            norm: self.floats(&m.norm),
            down: self.float_matrix(what, &m.down)?,
            up: self.float_matrix(what, &m.up)?,
            inject: m.inject.as_ref().map(|i| self.float_matrix(what, i)).transpose()?,
        })
    }

    fn layer(&mut self, c: &Config, layer: usize, w: &Layer<'g>) -> Result<LayerW, ModelError> {
        let name = |what: &str| format!("blk.{layer}.{what}");
        let attn_mixer = self.mixer(&name("hc_attn"), &w.attn_mixer)?;
        let ffn_mixer = self.mixer(&name("hc_ffn"), &w.ffn_mixer)?;
        let core = match &w.mixer {
            Mixer::Recurrent(r) => CoreW::Recurrent(RecurrentW {
                shape: RecurrentShape {
                    n_k_heads: c.recurrent.n_k_heads as u32,
                    n_v_heads: c.recurrent.n_v_heads as u32,
                    head_dim: c.recurrent.head_dim as u32,
                    conv_kernel: c.recurrent.conv_kernel as u32,
                },
                qkv: self.matrix(&r.qkv),
                z: self.matrix(&r.gate),
                alpha: self.matrix(&r.alpha),
                beta: self.matrix(&r.beta),
                out: self.matrix(&r.out),
                taps: self.floats(&r.conv),
                dt_bias: self.floats(&r.dt_bias),
                ssm_a: self.floats(&r.a),
                ssm_norm: self.floats(&r.norm),
            }),
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
        if moe.router.ty() != GgmlType::F32 {
            return Err(invalid(format!(
                "{} is {}: the router kernel takes F32",
                name("ffn_gate_inp"),
                moe.router.ty()
            )));
        }
        let router = self.push(Source::File(moe.router.bytes()));
        let shared_inp = Mat {
            ty: GgmlType::F32,
            n_in: c.n_embd as u32,
            n_out: 1,
            at: self.floats(&moe.shared_gate_inp),
        };
        let moe = MoeW {
            shape,
            router,
            shared_inp,
            shared_gate: self.matrix(&moe.shared_gate),
            shared_up: self.matrix(&moe.shared_up),
            shared_down: self.matrix(&moe.shared_down),
        };
        let ngram = w
            .ngram
            .as_ref()
            .map(|g| -> Result<NgramW, ModelError> {
                Ok(NgramW {
                    key: self.float_matrix(&name("ple_key"), &g.key)?,
                    value: self.float_matrix(&name("ple_value"), &g.value)?,
                    norm_key: self.floats(&g.norm_key),
                    norm_query: self.floats(&g.norm_query),
                    norm_conv: self.floats(&g.norm_conv),
                    taps: self.floats(&g.conv),
                })
            })
            .transpose()?;
        Ok(LayerW { layer, attn_mixer, ffn_mixer, core, moe, ngram })
    }

    fn head(&mut self, w: &Weights<'g>) -> Result<HeadW, ModelError> {
        Ok(HeadW {
            mixer: self.mixer("output_hc", &w.output_mixer)?,
            output: self.matrix(&w.output),
        })
    }
}

/// Bytes of the dense weights of layer `layer` in an arena.
pub(crate) fn layer_bytes(c: &Config, w: &Weights<'_>, layer: usize) -> Result<u64, ModelError> {
    let mut layout = Layout::default();
    layout.layer(c, layer, &w.layers[layer])?;
    Ok(layout.len)
}

/// Bytes of the final mixer and the output head in an arena.
pub(crate) fn head_bytes(w: &Weights<'_>) -> Result<u64, ModelError> {
    let mut layout = Layout::default();
    layout.head(w)?;
    Ok(layout.len)
}

/// The dense weights of one device.
#[derive(Debug)]
pub(crate) struct DeviceWeights {
    arena: DeviceBuffer,
    pub layers: Vec<LayerW>,
    pub head: Option<HeadW>,
    upload: Duration,
}

impl DeviceWeights {
    /// The address of offset `at` of the arena.
    pub fn at(&self, at: u64) -> DevicePtr {
        self.arena.ptr().offset(at)
    }

    /// The address of a matrix.
    pub fn mat(&self, m: &Mat) -> DevicePtr {
        self.at(m.at)
    }

    /// Bytes of the arena.
    pub fn bytes(&self) -> u64 {
        self.arena.len()
    }

    /// How long the upload took, the reads of the file included.
    pub fn upload_time(&self) -> Duration {
        self.upload
    }

    /// Lays out and uploads the dense weights of `layers`, and of the output
    /// head when `head` is set, after reserving them in the ledger.
    pub fn load(
        device: &Device,
        ledger: &mut Ledger,
        c: &Config,
        w: &Weights<'_>,
        layers: Range<usize>,
        head: bool,
    ) -> Result<Self, ModelError> {
        let mut layout = Layout::default();
        let on_device = layers
            .map(|layer| layout.layer(c, layer, &w.layers[layer]))
            .collect::<Result<Vec<_>, _>>()?;
        let head = head.then(|| layout.head(w)).transpose()?;
        let len = layout.len.max(ALIGN);
        ledger.reserve("qwen4exp dense weights", len)?;
        let arena = DeviceBuffer::new(device, len)?;
        let started = Instant::now();
        for (at, source) in &layout.parts {
            match source {
                Source::File(bytes) => arena.write(*at, bytes)?,
                Source::Floats(values) => {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    arena.write(*at, &bytes)?;
                }
            }
        }
        Ok(Self { arena, layers: on_device, head, upload: started.elapsed() })
    }
}
