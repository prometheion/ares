//! The layer kernels of `ares-cuda` (`LayerKernels`) held to the CPU reference. The tests are
//! in `layers/mod.rs`, which `tests/layers_vulkan.rs` runs on `ares-vulkan` with the same
//! inputs, references and limits; this file gives them what CUDA does its own way.
//!
//! These tests open GPUs, so like the other GPU tests they run only with
//! `ARES_TEST_CUDA=1`. The ones on the real 0.8B also need
//! `ARES_TEST_QWEN35_MODEL=<Qwen3.5-0.8B-Q4_K_M.gguf>` and say so when they skip:
//!
//! ```sh
//! ARES_TEST_CUDA=1 ARES_TEST_QWEN35_MODEL=/path/Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p ares-model --test layers_gpu -- --nocapture
//! ```
//!
//! What is measured is printed as `ERR name value` lines (`--nocapture`).
//!
//! - `recurrent_protocol_*`: convolution, delta rule, commit on made-up inputs, several
//!   sequences of different lengths in one launch, a sequence at position 0 and
//!   others deep in their history, window / verify / commit / direct steps one
//!   after the other, for both state formats.
//! - `embedding_gather_*`: rows of the real q6_K token table.
//! - `a_recurrent_block_*`: a whole block of the real 0.8B (norm, five matrix
//!   products, convolution, delta rule, gated norm, output product, residual,
//!   feed-forward part) chained from the existing weight kernels and our layer
//!   kernels, compared with `Reference::block` and, for the state it leaves,
//!   with `Reference::commit`.

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use ares_cuda as backend;
use backend::{
    Device, DeviceBuffer, DevicePtr, EmbedRows, KeptRows, Kernels, LayerKernels, Precision,
    PrefillPath, StateFormat, StepDims, Stream, quantized_activation_bytes,
};
use common::gpu::gpus;
use layers::DeviceMatrix;

mod common;

mod layers;

struct Gpu {
    device: Device,
    stream: Stream,
    layers: LayerKernels,
    kernels: Kernels,
}

impl Gpu {
    fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            stream: Stream::new(device, 0).unwrap(),
            layers: LayerKernels::load(device).unwrap(),
            kernels: Kernels::load(device).unwrap(),
        }
    }
}

/// Whether the delta rule rounds a state held in f16 after every row it advances it by, in a
/// window and in a commit (CUDA's kernels do), rather than once at the end of a commit.
const ROUNDED_EVERY_ROW: bool = true;

/// The largest difference of a delta rule's outputs from the reference's with the state in
/// f16: a value the two compute a few f32 steps apart can round to different neighbours in
/// f16, and the rows after it go on from there (5.2e-5 measured; a neighbour is 1e-3 of one
/// of 128 values).
const F16_OUTPUT_LIMIT: f64 = 5e-4;

/// The layer kernels with each shape of the delta rule the tests run in `format`: CUDA has one.
fn delta_shapes(gpu: &Gpu, _format: StateFormat) -> Vec<(&'static str, LayerKernels)> {
    vec![("d128", gpu.layers.clone())]
}

/// The embedding gather of the rows of `tokens` (in the step at `step`) from a table of
/// `n_vocab` rows of `n_embd` values of `ty` into `out`.
#[allow(clippy::too_many_arguments)]
fn embed_gather(
    gpu: &Gpu,
    step: &DeviceBuffer,
    dims: StepDims,
    ty: ares_core::GgmlType,
    table: &DeviceBuffer,
    n_vocab: u32,
    n_embd: usize,
    out: &DeviceBuffer,
) {
    let rows = EmbedRows {
        ty,
        n_vocab,
        n_embd: n_embd as u32,
        kept: KeptRows { rows: table.ptr(), tokens: DevicePtr::NULL, count: n_vocab },
        staged: DevicePtr::NULL,
    };
    unsafe { gpu.layers.embed_gather(&gpu.stream, step.ptr(), dims, &rows, out.ptr()).unwrap() };
}

/// `x + y` for `n` values, in a buffer of its own.
fn add_residual(gpu: &Gpu, x: &DeviceBuffer, y: DeviceBuffer, n: u32) -> DeviceBuffer {
    let sum = gpu.empty(u64::from(n) * 4);
    unsafe { gpu.kernels.add_to_f32(&gpu.stream, sum.ptr(), x.ptr(), y.ptr(), n).unwrap() };
    sum
}

/// How the matrix products of the chain prepare their activations.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// int8 activations, the decode kernels up to their row limit and the int8 tensor-core path above it.
    Fast,
    /// f16 activations and the f16 tensor-core path: 0.03 % rms from the exact product.
    Exact,
}

/// Prepared activations of `n` rows of `n_in` values for the products that follow.
struct Prepared {
    buf: DeviceBuffer,
    n_in: u32,
    n: u32,
}

struct Chain<'a> {
    gpu: &'a Gpu,
    mode: Mode,
    path: PrefillPath,
}

impl<'a> Chain<'a> {
    fn new(gpu: &'a Gpu, mode: Mode) -> Self {
        let path = match mode {
            Mode::Fast => PrefillPath::Int8,
            Mode::Exact => gpu.kernels.prefill_path(Precision::Exact).unwrap(),
        };
        Self { gpu, mode, path }
    }

    fn prepare(&self, x: DevicePtr, n_in: u32, n: u32) -> Prepared {
        let bytes = Kernels::prefill_activation_bytes(self.path, n_in, n);
        let buf = self.gpu.empty(bytes);
        unsafe {
            self.gpu
                .kernels
                .prepare_prefill_activations(&self.gpu.stream, self.path, x, buf.ptr(), n_in, n)
                .unwrap();
        }
        Prepared { buf, n_in, n }
    }

    /// Prepared activations from a fused kernel's output: the int8 layout as it is, or
    /// the f32 copy it carries converted to f16.
    fn adopt_fused(&self, fused: DeviceBuffer, n_in: u32, n: u32) -> Prepared {
        match self.mode {
            Mode::Fast => Prepared { buf: fused, n_in, n },
            Mode::Exact => {
                let n_values = u64::from(n_in) * u64::from(n);
                let copy = fused.ptr().offset(quantized_activation_bytes(n_values) - n_values * 4);
                let p = self.prepare(copy, n_in, n);
                drop(fused);
                p
            }
        }
    }

    fn product(&self, m: &DeviceMatrix, act: &Prepared) -> DeviceBuffer {
        assert_eq!(m.n_in, act.n_in);
        let y = self.gpu.empty(u64::from(m.n_out) * u64::from(act.n) * 4);
        let stream = &self.gpu.stream;
        unsafe {
            match (self.mode, Kernels::decode_kernel_rows(m.ty)) {
                (Mode::Fast, Some(limit)) if act.n <= limit.min(8) => self
                    .gpu
                    .kernels
                    .matvec(stream, m.ty, m.w.ptr(), act.buf.ptr(), y.ptr(), m.n_in, m.n_out, act.n)
                    .unwrap(),
                _ => self
                    .gpu
                    .kernels
                    .matmul(
                        stream,
                        m.ty,
                        self.path,
                        m.w.ptr(),
                        act.buf.ptr(),
                        y.ptr(),
                        m.n_in,
                        m.n_out,
                        act.n,
                    )
                    .unwrap(),
            }
        }
        y
    }
}
