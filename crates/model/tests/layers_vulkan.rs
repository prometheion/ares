//! The layer kernels of `ares-vulkan` (`LayerKernels`) held to the CPU reference. The tests are
//! in `layers/mod.rs`, which `tests/layers_gpu.rs` runs on `ares-cuda` with the same inputs,
//! references and limits; this file gives them what Vulkan does its own way.
//!
//! These tests open GPUs, so like the other GPU tests they run only with
//! `ARES_TEST_VULKAN=1`. The ones on the real 0.8B also need
//! `ARES_TEST_QWEN35_0_8B=<Qwen3.5-0.8B-Q4_K_M.gguf>` and say so when they skip:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 ARES_TEST_QWEN35_0_8B=/path/Qwen3.5-0.8B-Q4_K_M.gguf \
//!     cargo test -p ares-model --test layers_vulkan -- --nocapture
//! ```
//!
//! What is measured is printed as `ERR name value` lines (`--nocapture`).
//!
//! - `recurrent_protocol_*`: convolution, delta rule, commit on made-up inputs, several
//!   sequences of different lengths in one launch, a sequence at position 0 and
//!   others deep in their history, window / verify / commit / direct steps one
//!   after the other, for every compiled shape of the delta rule and for both
//!   state formats.
//! - `embedding_gather_*`: rows of the real q6_K token table.
//! - `a_recurrent_block_*`: a whole block of the real 0.8B (norm, five matrix
//!   products, convolution, delta rule, gated norm, output product, residual,
//!   feed-forward part) chained from Vulkan's weight kernels and its layer
//!   kernels, compared with `Reference::block` and, for the state it leaves,
//!   with `Reference::commit`.

// The tests launch kernels on buffers they allocated a few lines earlier, with sizes taken from
// the same shapes; a comment on each of the launches would repeat that.
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::OnceLock;

use ares_vulkan as backend;
use backend::{
    Device, DeviceBuffer, DevicePtr, Kernels, LayerKernels, MAX_BATCH, Precision, PrefillPath,
    StateFormat, StepDims, Stream, device_count, quantized_activation_copy_offset,
};
use layers::DeviceMatrix;

mod common;
mod layers;

/// The devices every test shares, opened once and never closed: the process leaves through
/// [`leave`].
static DEVICES: OnceLock<Option<&'static [Device]>> = OnceLock::new();

/// Ends the process at once when it exits, with the status it exits with (glibc's `on_exit`
/// hands it over): no exit handler of the driver runs and no device is destroyed. The NVIDIA
/// driver (615.78) compiles pipelines on threads of its own, and it crashes there when a
/// device is destroyed or its exit handlers run while one of them still works: seen as a
/// segmentation fault with devices that were never destroyed, and as an abort ("corrupted
/// size vs. prev_size" in `libnvidia-gpucomp`) with devices dropped by a handler at exit
/// after a long run that had made hundreds of pipelines. The kernel frees what the process
/// held.
extern "C" fn leave(status: i32, _: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    // SAFETY: `_exit` takes any status and does not return.
    unsafe { _exit(status) }
}

/// Registers [`leave`] for the exit of the process.
fn leave_at_exit() {
    unsafe extern "C" {
        fn on_exit(
            callback: extern "C" fn(i32, *mut std::ffi::c_void),
            arg: *mut std::ffi::c_void,
        ) -> i32;
    }
    // SAFETY: registering a function that ends the process; it uses no argument.
    unsafe { on_exit(leave, std::ptr::null_mut()) };
}

fn gpus() -> Option<&'static [Device]> {
    *DEVICES.get_or_init(|| {
        if std::env::var_os("ARES_TEST_VULKAN").is_none() {
            eprintln!("skipped: set ARES_TEST_VULKAN=1 to run the tests that open GPUs");
            return None;
        }
        let count = device_count().expect("the Vulkan loader");
        assert!(count > 0, "ARES_TEST_VULKAN is set but the loader reports no device");
        let opened: Vec<Device> =
            (0..count).map(|i| Device::open(i).expect("open device")).collect();
        leave_at_exit();
        Some(Box::leak(opened.into_boxed_slice()) as &'static [Device])
    })
}

struct Gpu {
    device: Device,
    stream: Stream,
    layers: LayerKernels,
    kernels: Kernels,
}

impl Gpu {
    fn new(device: &Device) -> Self {
        let kernels = Kernels::load(device).unwrap();
        Self {
            device: device.clone(),
            stream: Stream::new(device).unwrap(),
            layers: LayerKernels::load(&kernels).unwrap(),
            kernels,
        }
    }
}

/// Whether the delta rule rounds a state held in f16 after every row it advances it by; Vulkan's
/// rounds it once, at the end of a commit.
const ROUNDED_EVERY_ROW: bool = false;

/// The largest difference of a delta rule's outputs from the reference's with the state in
/// f16. The two computations round some elements to different neighbours (they differ by f32
/// rounding before the store: about one element in a thousand per step, printed), and such a
/// difference stays in the state of the next step. One step of f16 on one element moves an
/// output by up to 1e-4 of the largest: measured 3.8e-5 to 9.0e-5 over the shapes, against
/// 1e-6 with the state in f32.
const F16_OUTPUT_LIMIT: f64 = 2e-4;

/// The layer kernels with each shape of the delta rule the tests run in `format`: every
/// compiled shape, in both formats.
fn delta_shapes(gpu: &Gpu, _format: StateFormat) -> Vec<(&'static str, LayerKernels)> {
    LayerKernels::delta_variants()
        .map(|variant| {
            let mut layers = gpu.layers.clone();
            layers.select_delta_variant(variant).unwrap();
            (variant, layers)
        })
        .collect()
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
    unsafe {
        gpu.layers
            .embed_gather(
                &gpu.stream,
                step.ptr(),
                dims,
                table.ptr(),
                ty,
                n_vocab,
                n_embd as u32,
                out.ptr(),
            )
            .unwrap();
    }
}

/// `x + y` for `n` values, in the buffer of `y`.
fn add_residual(gpu: &Gpu, x: &DeviceBuffer, y: DeviceBuffer, n: u32) -> DeviceBuffer {
    unsafe { gpu.kernels.add_f32(&gpu.stream, y.ptr(), x.ptr(), n).unwrap() };
    y
}

/// How the matrix products of the chain prepare their activations.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// What a step takes for speed: up to 8 rows int8 activations and the decode kernels
    /// (about 1 % rms from the exact product), above that the tensor-core path of
    /// `Precision::Fast` (f16 activations; f16 accumulation on a capped card).
    Fast,
    /// f16 activations and the f16 tensor-core path with f32 accumulation: 0.03 % rms from
    /// the exact product.
    Exact,
}

/// Prepared activations of `n` rows of `n_in` values for the products that follow.
struct Prepared {
    /// The prepared form, with the buffer that holds it.
    act: DevicePtr,
    _buf: DeviceBuffer,
    /// The rows as f32, which the decode kernels of the float types read as they are.
    x: DevicePtr,
    /// The buffer `x` lies in, when the rows are a fused kernel's own copy.
    _x_owner: Option<DeviceBuffer>,
    n_in: u32,
    n: u32,
}

struct Chain<'a> {
    gpu: &'a Gpu,
    mode: Mode,
}

impl<'a> Chain<'a> {
    fn new(gpu: &'a Gpu, mode: Mode) -> Self {
        Self { gpu, mode }
    }

    /// The tensor-core path of this mode for `n` rows; `None` for the decode kernels.
    fn path(&self, n: u32) -> Option<PrefillPath> {
        match self.mode {
            Mode::Fast if n <= MAX_BATCH => None,
            Mode::Fast => Some(self.gpu.kernels.prefill_path(Precision::Fast).unwrap()),
            Mode::Exact => Some(self.gpu.kernels.prefill_path(Precision::Exact).unwrap()),
        }
    }

    fn prepare_from(
        &self,
        x: DevicePtr,
        x_owner: Option<DeviceBuffer>,
        n_in: u32,
        n: u32,
    ) -> Prepared {
        let kernels = &self.gpu.kernels;
        let buf = match self.path(n) {
            None => {
                let buf = self.gpu.empty(Kernels::quantized_activation_bytes(n_in, n));
                unsafe {
                    kernels.quantize_activations(&self.gpu.stream, x, buf.ptr(), n_in, n).unwrap();
                }
                buf
            }
            Some(path) => {
                let buf = self.gpu.empty(Kernels::prefill_activation_bytes(path, n_in, n));
                unsafe {
                    kernels
                        .prepare_prefill_activations(&self.gpu.stream, path, x, buf.ptr(), n_in, n)
                        .unwrap();
                }
                buf
            }
        };
        Prepared { act: buf.ptr(), _buf: buf, x, _x_owner: x_owner, n_in, n }
    }

    fn prepare(&self, x: DevicePtr, n_in: u32, n: u32) -> Prepared {
        self.prepare_from(x, None, n_in, n)
    }

    /// Prepared activations from a fused kernel's output: its int8 layout as it is for the
    /// decode kernels, or the f32 values it keeps behind converted to f16.
    fn adopt_fused(&self, fused: DeviceBuffer, n_in: u32, n: u32) -> Prepared {
        let n_values = u64::from(n_in) * u64::from(n);
        let copy = fused.ptr().offset(quantized_activation_copy_offset(n_values));
        match self.path(n) {
            None => Prepared { act: fused.ptr(), x: copy, _buf: fused, _x_owner: None, n_in, n },
            Some(_) => self.prepare_from(copy, Some(fused), n_in, n),
        }
    }

    fn product(&self, m: &DeviceMatrix, act: &Prepared) -> DeviceBuffer {
        assert_eq!(m.n_in, act.n_in);
        let y = self.gpu.empty(u64::from(m.n_out) * u64::from(act.n) * 4);
        let stream = &self.gpu.stream;
        let kernels = &self.gpu.kernels;
        unsafe {
            match self.path(act.n) {
                None => {
                    // The decode kernels of the float types read the f32 rows.
                    let rows =
                        if Kernels::needs_quantized_activations(m.ty) { act.act } else { act.x };
                    kernels
                        .matvec(stream, m.ty, m.w.ptr(), rows, y.ptr(), m.n_in, m.n_out, act.n)
                        .unwrap()
                }
                Some(path) => kernels
                    .matmul(stream, m.ty, path, m.w.ptr(), act.act, y.ptr(), m.n_in, m.n_out, act.n)
                    .unwrap(),
            }
        }
        y
    }
}
