//! The layer kernels of `ares-vulkan` (`LayerKernels`) held to the CPU reference: the twin
//! of `tests/layers_gpu.rs`, which holds `ares-cuda`'s to it, with the same inputs, the same
//! references and the same limits.
//!
//! These tests open GPUs, so like the other GPU tests they run only with
//! `ARES_TEST_VULKAN=1`. The ones on the real 0.8B also need
//! `ARES_TEST_QWEN35_MODEL=<Qwen3.5-0.8B-Q4_K_M.gguf>` and say so when they skip:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 ARES_TEST_QWEN35_MODEL=/path/Qwen3.5-0.8B-Q4_K_M.gguf \
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

use std::path::PathBuf;
use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_gguf::Gguf;
use ares_model::qwen35::reference::pieces::{conv_silu, delta_step, l2_norm, sigmoid, softplus};
use ares_model::qwen35::weights::{Matrix, Mixer};
use ares_model::qwen35::{Config, LogitRows, Reference, SequenceState, Weights};
use ares_vulkan::{
    CommitSet, Device, DeviceBuffer, DevicePtr, Kernels, LayerKernels, MAX_BATCH, OutputGate,
    Precision, PrefillPath, RecurrentShape, Segment, StateFormat, StepRecord, Stream, device_count,
    quantized_activation_bytes, quantized_activation_copy_offset,
};

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

fn model_path() -> Option<PathBuf> {
    match std::env::var_os("ARES_TEST_QWEN35_MODEL") {
        Some(p) => Some(p.into()),
        None => {
            eprintln!("skipped: set ARES_TEST_QWEN35_MODEL to run this test");
            None
        }
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn random(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.unit() * scale).collect()
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
    fn bytes(&self, data: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, data.len().max(16) as u64).unwrap();
        if !data.is_empty() {
            buffer.write(0, data).unwrap();
        }
        buffer
    }
    fn floats(&self, data: &[f32]) -> DeviceBuffer {
        self.bytes(&data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
    }
    fn empty(&self, bytes: u64) -> DeviceBuffer {
        let buffer = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        buffer.write(0, &vec![0xFFu8; bytes as usize]).unwrap();
        buffer
    }
    fn read(&self, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 4];
        buffer.read(0, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
    fn read_f16(&self, buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
        self.stream.synchronize().unwrap();
        let mut bytes = vec![0u8; n * 2];
        buffer.read(0, &mut bytes).unwrap();
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| ares_quant::f16_to_f32(u16::from_le_bytes(*b)))
            .collect()
    }
}

/// Largest absolute difference relative to the largest value of the reference.
fn rel_max(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(*v).abs())).max(1e-30);
    got.iter().zip(want).fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - f64::from(*w)).abs()))
        / scale
}

/// `|got - want| / |want|` over all values (L2).
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        d += (f64::from(*g) - f64::from(*w)).powi(2);
        n += f64::from(*w).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

fn report(name: &str, value: f64) {
    eprintln!("ERR {name} {value:.3e}");
}

// ---------------------------------------------------------------------------
// The recurrent kernels on made-up inputs.

/// One sequence on the host: the convolution window `[channel][K-1]` and the
/// matrices `[value head][value index][key index]`, as the kernels hold them.
#[derive(Clone)]
struct HostSeq {
    window: Vec<f32>,
    s: Vec<f32>,
}

struct Params {
    shape: RecurrentShape,
    taps: Vec<f32>,
    dt_bias: Vec<f32>,
    ssm_a: Vec<f32>,
}

/// The kept inputs of a window, as the reference computes them.
struct RefWindow {
    q: Vec<f32>,
    kv: Vec<f32>, // [row][key_dim + value_dim]: k^ then v
    g: Vec<f32>,
    beta: Vec<f32>,
    out: Vec<f32>,
}

fn ref_window(
    p: &Params,
    seq: &HostSeq,
    qkv: &[f32],
    alpha: &[f32],
    beta_in: &[f32],
    n: usize,
) -> RefWindow {
    let sh = &p.shape;
    let (dk, nv, nk) = (sh.head_dim as usize, sh.n_v_heads as usize, sh.n_k_heads as usize);
    let (kd, vd, cd, kk) = (
        sh.key_dim() as usize,
        sh.value_dim() as usize,
        sh.conv_dim() as usize,
        sh.conv_kernel as usize - 1,
    );
    let mixed = conv_silu(&p.taps, kk, &seq.window, qkv, cd, n);
    let mut q = vec![0.0f32; n * kd];
    let mut kv = vec![0.0f32; n * (kd + vd)];
    for t in 0..n {
        let row = &mixed[t * cd..(t + 1) * cd];
        for h in 0..nk {
            l2_norm(&row[h * dk..(h + 1) * dk], 1e-6, &mut q[t * kd + h * dk..][..dk]);
            l2_norm(
                &row[kd + h * dk..kd + (h + 1) * dk],
                1e-6,
                &mut kv[t * (kd + vd) + h * dk..][..dk],
            );
        }
        kv[t * (kd + vd) + kd..(t + 1) * (kd + vd)].copy_from_slice(&row[2 * kd..]);
    }
    let mut g = vec![0.0f32; n * nv];
    let mut beta = vec![0.0f32; n * nv];
    for t in 0..n {
        for h in 0..nv {
            g[t * nv + h] = (softplus(f64::from(alpha[t * nv + h]) + f64::from(p.dt_bias[h]))
                * f64::from(p.ssm_a[h])) as f32;
            beta[t * nv + h] = sigmoid(f64::from(beta_in[t * nv + h])) as f32;
        }
    }
    let mut out = vec![0.0f32; n * vd];
    let mut final_s = seq.s.clone(); // advanced in place by the window; only the outputs are kept
    for h in 0..nv {
        let kh = h % nk;
        let s = &mut final_s[h * dk * dk..(h + 1) * dk * dk];
        for t in 0..n {
            let mut o = vec![0.0f32; dk];
            delta_step(
                s,
                &q[t * kd + kh * dk..][..dk],
                &kv[t * (kd + vd) + kh * dk..][..dk],
                &kv[t * (kd + vd) + kd + h * dk..][..dk],
                g[t * nv + h],
                beta[t * nv + h],
                Some(&mut o),
            );
            out[t * vd + h * dk..][..dk].copy_from_slice(&o);
        }
    }
    RefWindow { q, kv, g, beta, out }
}

/// The reference's commit of the first `accepted` rows of a window: the new
/// window and the matrices advanced over exactly those rows.
fn ref_commit(
    p: &Params,
    seq: &mut HostSeq,
    w: &RefWindow,
    qkv: &[f32],
    accepted: usize,
    storage: ares_model::qwen35::StateStorage,
) {
    let sh = &p.shape;
    let (dk, nv, nk) = (sh.head_dim as usize, sh.n_v_heads as usize, sh.n_k_heads as usize);
    let (kd, vd, cd, kk) = (
        sh.key_dim() as usize,
        sh.value_dim() as usize,
        sh.conv_dim() as usize,
        sh.conv_kernel as usize - 1,
    );
    let old = seq.window.clone();
    for c in 0..cd {
        for i in 0..kk {
            let at = accepted + i;
            seq.window[c * kk + i] =
                if at < kk { old[c * kk + at] } else { qkv[(at - kk) * cd + c] };
        }
    }
    for h in 0..nv {
        let kh = h % nk;
        let s = &mut seq.s[h * dk * dk..(h + 1) * dk * dk];
        for t in 0..accepted {
            delta_step(
                s,
                &w.q[t * kd + kh * dk..][..dk],
                &w.kv[t * (kd + vd) + kh * dk..][..dk],
                &w.kv[t * (kd + vd) + kd + h * dk..][..dk],
                w.g[t * nv + h],
                w.beta[t * nv + h],
                None,
            );
        }
    }
    if accepted > 0 {
        storage.round(&mut seq.s, dk, dk);
    }
}

struct StepSpec {
    /// Rows, direct, accepted rows for the commit.
    rows: usize,
    direct: bool,
    accepted: usize,
}

/// Runs steps of several sequences through the kernels and through the
/// reference, window by window, and compares everything after each step.
fn recurrent_protocol(format: StateFormat, variant: &str, gpu: &Gpu) -> (f64, f64, f64) {
    let storage = match format {
        StateFormat::F32 => ares_model::qwen35::StateStorage::F32,
        StateFormat::F16 => ares_model::qwen35::StateStorage::F16,
    };
    let shape = RecurrentShape { n_k_heads: 2, n_v_heads: 4, head_dim: 128, conv_kernel: 4 };
    let nv = 4usize;
    let (kd, vd, cd, kk) =
        (shape.key_dim() as usize, shape.value_dim() as usize, shape.conv_dim() as usize, 3usize);
    let mut rng = Rng(0xDE17A + variant.len() as u64);
    let p = Params {
        shape,
        taps: random(&mut rng, cd * 4, 0.6),
        dt_bias: random(&mut rng, nv, 1.0),
        ssm_a: random(&mut rng, nv, 1.5).iter().map(|v| -v.exp()).collect(),
    };
    // Five sequences, slots permuted and strided: sequence 0 is at position 0.
    let slots = [3u32, 0, 4, 1, 2];
    let n_slots = 5usize;
    let (conv_len, delta_len) = (shape.conv_state_len() as usize, shape.delta_state_len() as usize);
    let (conv_stride, delta_stride) = (conv_len + 64, delta_len + 64);
    let round_state = |v: f32| match format {
        StateFormat::F32 => v,
        StateFormat::F16 => ares_quant::f16_to_f32(ares_quant::f32_to_f16(v)),
    };
    let mut seqs: Vec<HostSeq> = (0..5)
        .map(|i| {
            if i == 0 {
                HostSeq { window: vec![0.0; cd * kk], s: vec![0.0; delta_len] }
            } else {
                HostSeq {
                    window: random(&mut rng, cd * kk, 1.0),
                    s: random(&mut rng, delta_len, 0.3).into_iter().map(round_state).collect(),
                }
            }
        })
        .collect();

    // Device state arrays.
    let upload_state = |seqs: &[HostSeq]| -> (DeviceBuffer, DeviceBuffer) {
        let mut conv = vec![0.0f32; n_slots * conv_stride];
        let mut delta = vec![0.0f32; n_slots * delta_stride];
        for (i, s) in seqs.iter().enumerate() {
            let slot = slots[i] as usize;
            conv[slot * conv_stride..][..conv_len].copy_from_slice(&s.window);
            delta[slot * delta_stride..][..delta_len].copy_from_slice(&s.s);
        }
        let conv_b = gpu.floats(&conv);
        let delta_b = match format {
            StateFormat::F32 => gpu.floats(&delta),
            StateFormat::F16 => gpu.bytes(
                &delta
                    .iter()
                    .flat_map(|v| ares_quant::f32_to_f16(*v).to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
        };
        (conv_b, delta_b)
    };
    let (conv_b, delta_b) = upload_state(&seqs);
    let (taps_b, dt_b, a_b) = (gpu.floats(&p.taps), gpu.floats(&p.dt_bias), gpu.floats(&p.ssm_a));

    // Steps: prompt slices (direct, odd and long and short and one row), then
    // windows of drafts with commits of every length, then single rows again.
    let steps: Vec<Vec<StepSpec>> = vec![
        vec![
            StepSpec { rows: 5, direct: true, accepted: 5 },
            StepSpec { rows: 1, direct: true, accepted: 1 },
            StepSpec { rows: 70, direct: true, accepted: 70 },
            StepSpec { rows: 3, direct: true, accepted: 3 },
            StepSpec { rows: 2, direct: true, accepted: 2 },
        ],
        vec![
            StepSpec { rows: 4, direct: false, accepted: 2 },
            StepSpec { rows: 8, direct: false, accepted: 8 },
            StepSpec { rows: 1, direct: false, accepted: 0 },
            StepSpec { rows: 3, direct: false, accepted: 1 },
            StepSpec { rows: 6, direct: true, accepted: 6 },
        ],
        vec![
            StepSpec { rows: 1, direct: true, accepted: 1 },
            StepSpec { rows: 2, direct: false, accepted: 1 },
            StepSpec { rows: 1, direct: true, accepted: 1 },
            StepSpec { rows: 7, direct: false, accepted: 3 },
            StepSpec { rows: 1, direct: false, accepted: 1 },
        ],
    ];
    let (mut worst_out, mut worst_conv, mut worst_state) = (0.0f64, 0.0f64, 0.0f64);
    let mut f16_off = 0usize;
    for (step_no, step) in steps.iter().enumerate() {
        // Rows of the step.
        let mut segments = Vec::new();
        let mut row0 = 0u32;
        for (i, s) in step.iter().enumerate() {
            segments.push(Segment {
                row0,
                n_rows: s.rows as u32,
                slot: slots[i],
                pos0: 0,
                direct: s.direct,
            });
            row0 += s.rows as u32;
        }
        let n = row0 as usize;
        let qkv = random(&mut rng, n * cd, 1.5);
        let alpha = random(&mut rng, n * nv, 2.0);
        let beta_in = random(&mut rng, n * nv, 2.0);
        let mut record = StepRecord::new(&segments, &vec![0; n], None, &[]).unwrap();
        let dims = record.dims();
        let step_buf = gpu.bytes(record.as_bytes());

        // The reference's windows.
        let windows: Vec<RefWindow> = step
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let r0 = segments[i].row0 as usize;
                ref_window(
                    &p,
                    &seqs[i],
                    &qkv[r0 * cd..(r0 + s.rows) * cd],
                    &alpha[r0 * nv..(r0 + s.rows) * nv],
                    &beta_in[r0 * nv..(r0 + s.rows) * nv],
                    s.rows,
                )
            })
            .collect();

        // The kernels.
        let (qkv_b, alpha_b, beta_b) = (gpu.floats(&qkv), gpu.floats(&alpha), gpu.floats(&beta_in));
        let q_b = gpu.empty((n * kd * 4) as u64);
        let kv_b = gpu.empty((n * (kd + vd) * 4) as u64);
        let out_b = gpu.empty((n * vd * 4) as u64);
        let mut layers = gpu.layers.clone();
        layers.select_delta_variant(variant).unwrap();
        unsafe {
            layers
                .conv_window(
                    &gpu.stream,
                    step_buf.ptr(),
                    dims,
                    shape,
                    Some(1e-6),
                    qkv_b.ptr(),
                    taps_b.ptr(),
                    conv_b.ptr(),
                    conv_stride as u64,
                    q_b.ptr(),
                    kv_b.ptr(),
                )
                .unwrap();
            layers
                .delta_window(
                    &gpu.stream,
                    step_buf.ptr(),
                    dims,
                    shape,
                    format,
                    q_b.ptr(),
                    kv_b.ptr(),
                    alpha_b.ptr(),
                    beta_b.ptr(),
                    dt_b.ptr(),
                    a_b.ptr(),
                    delta_b.ptr(),
                    delta_stride as u64,
                    out_b.ptr(),
                )
                .unwrap();
            // Direct segments commit their window right away, while the inputs are there.
            layers
                .conv_commit(
                    &gpu.stream,
                    step_buf.ptr(),
                    dims,
                    shape,
                    CommitSet::Direct,
                    qkv_b.ptr(),
                    conv_b.ptr(),
                    conv_stride as u64,
                )
                .unwrap();
        }
        let (got_q, got_kv, got_out) =
            (gpu.read(&q_b, n * kd), gpu.read(&kv_b, n * (kd + vd)), gpu.read(&out_b, n * vd));
        for (i, s) in step.iter().enumerate() {
            let r0 = segments[i].row0 as usize;
            let w = &windows[i];
            let e_q = rel_max(&got_q[r0 * kd..(r0 + s.rows) * kd], &w.q);
            let e_kv = rel_max(&got_kv[r0 * (kd + vd)..(r0 + s.rows) * (kd + vd)], &w.kv);
            let e_out = rel_max(&got_out[r0 * vd..(r0 + s.rows) * vd], &w.out);
            assert!(
                e_q < 2e-6 && e_kv < 2e-6,
                "step {step_no} seq {i}: conv outputs q {e_q}, kv {e_kv}"
            );
            // With the state in f16 the two computations round some elements to different
            // neighbours (they differ by f32 rounding before the store: about one element
            // in a thousand per step, printed below), and such a difference stays in the
            // state of the next step. One step of f16 on one element moves an output by up
            // to 1e-4 of the largest: measured 3.8e-5 to 9.0e-5 over the shapes, against
            // 1e-6 with the state in f32. (CUDA's test has 5e-5 for both formats, which the
            // two shapes it runs with f16 meet; the default shape here does too.)
            let out_limit = if format == StateFormat::F32 { 5e-5 } else { 2e-4 };
            assert!(e_out < out_limit, "step {step_no} seq {i}: delta output {e_out}");
            worst_out = worst_out.max(e_out);
            worst_conv = worst_conv.max(e_q.max(e_kv));
        }

        // Verification: the deferred segments accept a prefix; the commit follows.
        for (i, s) in step.iter().enumerate() {
            record.set_accepted(i, s.accepted as u32).unwrap();
        }
        let step_buf = gpu.bytes(record.as_bytes());
        unsafe {
            layers
                .commit(
                    &gpu.stream,
                    step_buf.ptr(),
                    dims,
                    shape,
                    format,
                    qkv_b.ptr(),
                    kv_b.ptr(),
                    alpha_b.ptr(),
                    beta_b.ptr(),
                    dt_b.ptr(),
                    a_b.ptr(),
                    conv_b.ptr(),
                    conv_stride as u64,
                    delta_b.ptr(),
                    delta_stride as u64,
                )
                .unwrap();
        }
        // The reference's commits (a direct segment is committed whole).
        for (i, s) in step.iter().enumerate() {
            let r0 = segments[i].row0 as usize;
            let accepted = if s.direct { s.rows } else { s.accepted };
            ref_commit(
                &p,
                &mut seqs[i],
                &windows[i],
                &qkv[r0 * cd..(r0 + s.rows) * cd],
                accepted,
                storage,
            );
        }
        // Compare all state, including the slots of sequences that were not in the step.
        let got_conv = gpu.read(&conv_b, n_slots * conv_stride);
        let got_delta: Vec<f32> = match format {
            StateFormat::F32 => gpu.read(&delta_b, n_slots * delta_stride),
            StateFormat::F16 => gpu.read_f16(&delta_b, n_slots * delta_stride),
        };
        for (i, seq) in seqs.iter().enumerate() {
            let slot = slots[i] as usize;
            let e_conv = rel_max(&got_conv[slot * conv_stride..][..conv_len], &seq.window);
            let e_state = rel_max(&got_delta[slot * delta_stride..][..delta_len], &seq.s);
            let limit = if format == StateFormat::F32 { 2e-5 } else { 2e-3 };
            assert!(e_conv < 1e-6, "step {step_no} seq {i}: conv window {e_conv}");
            assert!(e_state < limit, "step {step_no} seq {i}: matrices {e_state}");
            worst_state = worst_state.max(e_state);
            if format == StateFormat::F16 {
                // How many elements the two computations rounded to different f16 values.
                let off = got_delta[slot * delta_stride..][..delta_len]
                    .iter()
                    .zip(&seq.s)
                    .filter(|(g, w)| g != w)
                    .count();
                f16_off = f16_off.max(off);
            }
            // The padding between slots must stay as it was.
            let pad = &got_delta[slot * delta_stride + delta_len..][..64];
            assert!(pad.iter().all(|v| *v == 0.0), "the kernel wrote past a slot");
        }
    }
    if format == StateFormat::F16 {
        eprintln!(
            "{variant}: at most {f16_off} of {delta_len} matrix elements of a sequence differ from the reference's"
        );
    }
    (worst_out, worst_conv, worst_state)
}

#[test]
fn recurrent_protocol_f32_matches_the_reference_for_every_shape() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        for variant in LayerKernels::delta_variants() {
            let (out, conv, state) = recurrent_protocol(StateFormat::F32, variant, &gpu);
            report(&format!("{} f32 {variant} delta_out_rel_max", device.info().name), out);
            report(&format!("{} f32 {variant} conv_rel_max", device.info().name), conv);
            report(&format!("{} f32 {variant} state_rel_max", device.info().name), state);
        }
    }
}

#[test]
fn recurrent_protocol_f16_state_matches_the_reference_holding_f16() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let gpu = Gpu::new(device);
        for variant in LayerKernels::delta_variants() {
            let (out, conv, state) = recurrent_protocol(StateFormat::F16, variant, &gpu);
            report(&format!("{} f16 {variant} delta_out_rel_max", device.info().name), out);
            report(&format!("{} f16 {variant} state_rel_max", device.info().name), state);
            let _ = conv;
        }
    }
}

// ---------------------------------------------------------------------------
// The real 0.8B.

fn upload_matrix(gpu: &Gpu, m: &Matrix<'_>) -> DeviceBuffer {
    let mut bytes = Vec::new();
    for r in 0..m.rows() {
        bytes.extend_from_slice(m.row_bytes(r));
    }
    gpu.bytes(&bytes)
}

#[test]
fn embedding_gather_decodes_rows_of_the_real_table() {
    let (Some(gpus), Some(path)) = (gpus(), model_path()) else { return };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let table = &w.token_embd;
    let ty = table.ty();
    for device in gpus {
        let gpu = Gpu::new(device);
        // A table of selected rows (the first, some in the middle, the last).
        let picks = [0usize, 1, 2, 777, 100_000, 248_318, 248_319];
        let mut bytes = Vec::new();
        for &r in &picks {
            bytes.extend_from_slice(table.row_bytes(r));
        }
        let table_b = gpu.bytes(&bytes);
        let tokens: Vec<u32> = vec![6, 0, 3, 3, 5, 1, 2, 4];
        let segments = [Segment { row0: 0, n_rows: 8, slot: 0, pos0: 0, direct: false }];
        let record = StepRecord::new(&segments, &tokens, None, &[]).unwrap();
        let step = gpu.bytes(record.as_bytes());
        let out = gpu.empty((8 * c.n_embd * 4) as u64);
        unsafe {
            gpu.layers
                .embed_gather(
                    &gpu.stream,
                    step.ptr(),
                    record.dims(),
                    table_b.ptr(),
                    ty,
                    picks.len() as u32,
                    c.n_embd as u32,
                    out.ptr(),
                )
                .unwrap();
        }
        let got = gpu.read(&out, 8 * c.n_embd);
        let mut worst = 0.0f64;
        for (i, &t) in tokens.iter().enumerate() {
            let mut want = vec![0.0f32; c.n_embd];
            table.decode_row(picks[t as usize], &mut want);
            worst = worst.max(rel_max(&got[i * c.n_embd..(i + 1) * c.n_embd], &want));
        }
        report(&format!("{} embed {} rel_max", device.info().name, ty), worst);
        assert!(worst < 1e-6, "{worst}");
    }
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

struct DeviceMatrix {
    ty: GgmlType,
    n_in: u32,
    n_out: u32,
    w: DeviceBuffer,
}

fn device_matrix(gpu: &Gpu, m: &Matrix<'_>) -> DeviceMatrix {
    DeviceMatrix {
        ty: m.ty(),
        n_in: m.cols() as u32,
        n_out: m.rows() as u32,
        w: upload_matrix(gpu, m),
    }
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

struct BlockWeights {
    attn_norm: DeviceBuffer,
    post_norm: DeviceBuffer,
    gate: DeviceMatrix,
    up: DeviceMatrix,
    down: DeviceMatrix,
    qkv: DeviceMatrix,
    z: DeviceMatrix,
    beta: DeviceMatrix,
    alpha: DeviceMatrix,
    out: DeviceMatrix,
    taps: DeviceBuffer,
    dt_bias: DeviceBuffer,
    ssm_a: DeviceBuffer,
    ssm_norm: DeviceBuffer,
}

fn block_weights(gpu: &Gpu, w: &Weights<'_>, layer: usize) -> BlockWeights {
    let l = &w.layers[layer];
    let Mixer::Recurrent(r) = &l.mixer else { panic!("block {layer} is attention") };
    BlockWeights {
        attn_norm: gpu.floats(&l.block.attn_norm),
        post_norm: gpu.floats(&l.block.post_attention_norm),
        gate: device_matrix(gpu, &l.block.ffn_gate),
        up: device_matrix(gpu, &l.block.ffn_up),
        down: device_matrix(gpu, &l.block.ffn_down),
        qkv: device_matrix(gpu, &r.qkv),
        z: device_matrix(gpu, &r.gate),
        beta: device_matrix(gpu, &r.beta),
        alpha: device_matrix(gpu, &r.alpha),
        out: device_matrix(gpu, &r.out),
        taps: gpu.floats(&r.conv),
        dt_bias: gpu.floats(&r.dt_bias),
        ssm_a: gpu.floats(&r.a),
        ssm_norm: gpu.floats(&r.norm),
    }
}

/// One recurrent block of the model on the GPU for one sequence's window of `n`
/// rows. Returns the block's output and leaves the committed state in the buffers.
#[allow(clippy::too_many_arguments)]
fn run_block(
    chain: &Chain<'_>,
    c: &Config,
    bw: &BlockWeights,
    x: &DeviceBuffer,
    n: usize,
    conv_state: &DeviceBuffer,
    delta_state: &DeviceBuffer,
    format: StateFormat,
) -> DeviceBuffer {
    let gpu = chain.gpu;
    let r = &c.recurrent;
    let shape = RecurrentShape {
        n_k_heads: r.n_k_heads as u32,
        n_v_heads: r.n_v_heads as u32,
        head_dim: r.head_k_dim as u32,
        conv_kernel: r.conv_kernel as u32,
    };
    let (d, nv, vd) = (c.n_embd, r.n_v_heads, r.value_dim());
    let segments = [Segment { row0: 0, n_rows: n as u32, slot: 0, pos0: 0, direct: false }];
    let mut record = StepRecord::new(&segments, &vec![0; n], None, &[]).unwrap();
    let dims = record.dims();
    let step = gpu.bytes(record.as_bytes());
    let stream = &gpu.stream;
    let n32 = n as u32;

    // Mixer.
    let xn = gpu.empty((n * d * 4) as u64);
    unsafe {
        gpu.kernels
            .rms_norm_f32(stream, x.ptr(), bw.attn_norm.ptr(), xn.ptr(), d as u32, n32, c.rms_eps)
            .unwrap();
    }
    let act = chain.prepare(xn.ptr(), d as u32, n32);
    let qkv = chain.product(&bw.qkv, &act);
    let z = chain.product(&bw.z, &act);
    let beta_in = chain.product(&bw.beta, &act);
    let alpha = chain.product(&bw.alpha, &act);
    let q = gpu.empty((n * r.key_dim() * 4) as u64);
    let kv = gpu.empty((n * (r.key_dim() + vd) * 4) as u64);
    let o = gpu.empty((n * vd * 4) as u64);
    let (conv_stride, delta_stride) = (shape.conv_state_len(), shape.delta_state_len());
    unsafe {
        gpu.layers
            .conv_window(
                stream,
                step.ptr(),
                dims,
                shape,
                Some(c.rms_eps),
                qkv.ptr(),
                bw.taps.ptr(),
                conv_state.ptr(),
                conv_stride,
                q.ptr(),
                kv.ptr(),
            )
            .unwrap();
        gpu.layers
            .delta_window(
                stream,
                step.ptr(),
                dims,
                shape,
                format,
                q.ptr(),
                kv.ptr(),
                alpha.ptr(),
                beta_in.ptr(),
                bw.dt_bias.ptr(),
                bw.ssm_a.ptr(),
                delta_state.ptr(),
                delta_stride,
                o.ptr(),
            )
            .unwrap();
    }
    let fused = gpu.empty(quantized_activation_bytes((n * vd) as u64));
    unsafe {
        gpu.layers
            .gated_norm_quant(
                stream,
                OutputGate::Silu,
                o.ptr(),
                z.ptr(),
                bw.ssm_norm.ptr(),
                fused.ptr(),
                (n * nv) as u32,
                r.head_v_dim as u32,
                c.rms_eps,
            )
            .unwrap();
    }
    let act = chain.adopt_fused(fused, vd as u32, n32);
    let mixed = chain.product(&bw.out, &act);
    // The residual: x1 = x + mixed, in the buffer of the product.
    let x1 = mixed;
    unsafe {
        gpu.kernels.add_f32(stream, x1.ptr(), x.ptr(), (n * d) as u32).unwrap();
    }

    // Feed-forward part.
    let xn2 = gpu.empty((n * d * 4) as u64);
    unsafe {
        gpu.kernels
            .rms_norm_f32(stream, x1.ptr(), bw.post_norm.ptr(), xn2.ptr(), d as u32, n32, c.rms_eps)
            .unwrap();
    }
    let act = chain.prepare(xn2.ptr(), d as u32, n32);
    let gate = chain.product(&bw.gate, &act);
    let up = chain.product(&bw.up, &act);
    let fused = gpu.empty(quantized_activation_bytes((n * c.n_ff) as u64));
    unsafe {
        gpu.layers
            .swiglu_quant(stream, gate.ptr(), up.ptr(), fused.ptr(), (n * c.n_ff) as u32)
            .unwrap();
    }
    let act = chain.adopt_fused(fused, c.n_ff as u32, n32);
    let down = chain.product(&bw.down, &act);
    let out = down;
    unsafe {
        gpu.kernels.add_f32(stream, out.ptr(), x1.ptr(), (n * d) as u32).unwrap();
    }

    // The commit of all rows (deferred form), from what the window kept.
    record.set_accepted(0, n as u32).unwrap();
    let step = gpu.bytes(record.as_bytes());
    unsafe {
        gpu.layers
            .commit(
                stream,
                step.ptr(),
                dims,
                shape,
                format,
                qkv.ptr(),
                kv.ptr(),
                alpha.ptr(),
                beta_in.ptr(),
                bw.dt_bias.ptr(),
                bw.ssm_a.ptr(),
                conv_state.ptr(),
                conv_stride,
                delta_state.ptr(),
                delta_stride,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    out
}

#[test]
fn a_recurrent_block_of_the_real_0_8b_matches_the_reference() {
    let (Some(gpus), Some(path)) = (gpus(), model_path()) else { return };
    let gguf = Gguf::open(path).unwrap();
    let c = Config::from_gguf(&gguf).unwrap();
    let w = Weights::load(&gguf, &c).unwrap();
    let reference = Reference::new(&c, &w);
    // A prompt of the model's own kind of text: its ids are the first tokens of a real prompt.
    let ids: Vec<u32> = std::env::var_os("ARES_TEST_QWEN35_CAPTURE")
        .map(PathBuf::from)
        .and_then(|d| std::fs::read_to_string(d.join("english/base/prompt.tokens")).ok())
        .map(|t| t.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| (0..80).map(|i| (i * 7919 + 13) % 50_000).collect());
    let history = 40usize;
    let mut state = SequenceState::new(&c);
    reference.forward(&mut state, &ids[..history], LogitRows::None).unwrap();
    // Window sizes: one row (decode kernels), eight (their limit), 21 (odd, tensor-core path).
    // The tensor-core kernels step through a row in 64 values and whole blocks, which every
    // matrix of the model allows.
    let layers: Vec<usize> = (0..c.n_layer).filter(|l| !c.is_attention(*l)).collect();
    for device in gpus {
        let gpu = Gpu::new(device);
        for &n in &[1usize, 8, 21] {
            let tokens = &ids[history..history + n];
            let window = reference.window(&state, tokens, LogitRows::None).unwrap();
            let mut committed = state.clone();
            reference.commit(&mut committed, &window, n).unwrap();
            for mode in [Mode::Fast, Mode::Exact] {
                let chain = Chain::new(&gpu, mode);
                let (mut worst_out, mut worst_conv, mut worst_delta) = (0.0f64, 0.0f64, 0.0f64);
                for &layer in &layers {
                    // The block is fed with the hidden state the reference gave it.
                    let x_in: Vec<f32> = if layer == 0 {
                        (0..n)
                            .flat_map(|i| {
                                let mut row = vec![0.0f32; c.n_embd];
                                w.token_embd.decode_row(tokens[i] as usize, &mut row);
                                row
                            })
                            .collect()
                    } else {
                        window.hidden[layer - 1].clone()
                    };
                    let want = reference.block(layer, &state, &x_in);
                    let bw = block_weights(&gpu, &w, layer);
                    let (conv_b, delta_b) = (
                        gpu.floats(state.conv(layer).unwrap()),
                        gpu.floats(state.delta(layer).unwrap()),
                    );
                    let x_b = gpu.floats(&x_in);
                    let out =
                        run_block(&chain, &c, &bw, &x_b, n, &conv_b, &delta_b, StateFormat::F32);
                    let got = gpu.read(&out, n * c.n_embd);
                    let e_out = rel_l2(&got, &want);
                    let e_conv = rel_max(
                        &gpu.read(&conv_b, state.conv(layer).unwrap().len()),
                        committed.conv(layer).unwrap(),
                    );
                    let e_delta = rel_max(
                        &gpu.read(&delta_b, state.delta(layer).unwrap().len()),
                        committed.delta(layer).unwrap(),
                    );
                    worst_out = worst_out.max(e_out);
                    worst_conv = worst_conv.max(e_conv);
                    worst_delta = worst_delta.max(e_delta);
                    eprintln!(
                        "block {layer:2} n={n:2} {mode:?}: output {e_out:.2e}, conv window {e_conv:.1e}, matrices {e_delta:.1e}"
                    );
                }
                let name = device.info().name.clone();
                report(&format!("{name} block n={n} {mode:?} out_rel_l2_worst"), worst_out);
                report(
                    &format!("{name} block n={n} {mode:?} conv_state_rel_max_worst"),
                    worst_conv,
                );
                report(
                    &format!("{name} block n={n} {mode:?} delta_state_rel_max_worst"),
                    worst_delta,
                );
                let limit = match mode {
                    Mode::Fast => 5e-2,
                    Mode::Exact => 5e-3,
                };
                assert!(worst_out < limit, "{mode:?} n={n}: block output {worst_out}");
                let state_limit = match mode {
                    Mode::Fast => 2e-2,
                    Mode::Exact => 2e-3,
                };
                assert!(
                    worst_conv < state_limit && worst_delta < state_limit,
                    "{mode:?} n={n}: state {worst_conv} {worst_delta}"
                );
            }
        }
    }
}
