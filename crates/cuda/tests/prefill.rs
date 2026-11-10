//! Tests of the prefill weight kernels (`Kernels::matmul`,
//! `prepare_prefill_activations`): the tensor-core family for tens to thousands
//! of activation rows.
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`, like
//! `tests/gpu.rs`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test prefill -- --test-threads=1
//! ```
//!
//! `ARES_TEST_GGUF=<model.gguf>` adds the check on real tensors: the smallest
//! matrix of every type with a prefill kernel that the model contains.
//!
//! What the tests prove. There are two kernels per type, and each has two
//! claims, checked separately as the decode kernels' tests do:
//!
//! - *The kernel computes what it says* (tight): its result equals the weights
//!   (`ares_quant::dequantize`) times the activations in the form that kernel
//!   computes with, built on the CPU in f64: for the int8 path the activations
//!   dequantized from the codes `ares_quantize_act` produces and the exact
//!   weights; for the f16 path the weights and the activations rounded to f16.
//!   What is left is the order of the f32 sums and the last bit of the f32
//!   arithmetic that forms a weight before it is rounded to f16.
//! - *The form costs little* (loose): the result is close to the exact product
//!   of the weights with the unrounded f32 activations. The deviations are
//!   printed (`--nocapture`) so that a change that makes them worse can be
//!   noticed; the assertion only guards against a gross error.

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Kernels, Precision, PrefillPath, Stream, Wait, device_count,
};
use ares_quant::{f16_to_f32, f32_to_f16};

/// The types with a prefill kernel, and the byte offsets of the f16 fields of a
/// block that must be finite for the test data to be meaningful.
const TYPES: [(GgmlType, &[usize]); 11] = [
    (GgmlType::Q4_K, &[0, 2]),
    (GgmlType::Q5_K, &[0, 2]),
    (GgmlType::Q6_K, &[208]),
    (GgmlType::Q3_K, &[108]),
    (GgmlType::IQ4_XS, &[0]),
    (GgmlType::IQ4_NL, &[0]),
    (GgmlType::IQ3_S, &[0]),
    (GgmlType::IQ3_XXS, &[0]),
    (GgmlType::IQ2_S, &[0]),
    (GgmlType::IQ2_XS, &[0]),
    (GgmlType::Q8_0, &[0]),
];

const PATHS: [PrefillPath; 3] = [PrefillPath::Int8, PrefillPath::F16, PrefillPath::F16Flush];

fn gpus() -> Option<&'static [Device]> {
    static DEVICES: OnceLock<Option<Vec<Device>>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            if std::env::var_os("ARES_TEST_CUDA").is_none() {
                eprintln!("skipped: set ARES_TEST_CUDA=1 to run the tests that open GPUs");
                return None;
            }
            let count = device_count().expect("the CUDA driver");
            assert!(count > 0, "ARES_TEST_CUDA is set but the driver reports no device");
            Some((0..count).map(|i| Device::open(i, Wait::Block).expect("open device")).collect())
        })
        .as_deref()
}

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// A value of a normal distribution (Box-Muller), occasionally an outlier:
    /// real activations have a few much larger values per row.
    fn activation(&mut self) -> f32 {
        let u1 = (self.unit() + 1.0) * 0.5 + 1e-7;
        let u2 = self.unit();
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos();
        if self.next().is_multiple_of(400) { normal * 30.0 } else { normal }
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buffer.read(0, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// `n_blocks` blocks of `ty` of random bytes, with finite f16 scales of both
/// signs (any other bytes are valid in every one of these types).
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    let (_, scales) = TYPES.iter().find(|(t, _)| *t == ty).unwrap();
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for b in 0..n_blocks {
        for &offset in *scales {
            let scale = (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5))
                * if rng.next().is_multiple_of(4) { -1.0 } else { 1.0 };
            let at = b * block + offset;
            bytes[at..at + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        }
    }
    bytes
}

/// The quantization of activations as `ares_quantize_act` does it, on the CPU:
/// per group of 32 values `d = max|x| / 127`, `q = round(x / d)` (halves away
/// from zero). Returns the activations as the kernels see them, `d * q`.
fn quantized_like_the_gpu(x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for group in x.chunks(32) {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        for v in group {
            let code = if amax == 0.0 { 0.0 } else { (v / d).round() };
            out.push(f64::from(d) * f64::from(code));
        }
    }
    out
}

fn through_f16(v: f32) -> f64 {
    f64::from(f16_to_f32(f32_to_f16(v)))
}

/// How far a kernel result is from the two references.
#[derive(Debug)]
struct Deviation {
    /// Largest `|have - emulation| / sum of |w x|` over the outputs.
    tight: f64,
    /// `rms(have - exact) / rms(exact)` and `max|have - exact| / max|exact|`,
    /// with `exact` the product of the weights with the unrounded activations.
    loose_rms: f64,
    loose_max: f64,
    /// `rms(have - exact)` over the typical size of a sum of this kind,
    /// `sqrt(sum of (w x)^2)` per output: stable where the two above are not,
    /// for a handful of random sums.
    loose_scaled: f64,
}

impl Deviation {
    fn describe(&self) -> String {
        format!(
            "vs the emulation {:.1e} of the magnitude; vs the f32 product: rms {:.4}%, max {:.4}%, {:.4}% of a sum's size",
            self.tight,
            self.loose_rms * 100.0,
            self.loose_max * 100.0,
            self.loose_scaled * 100.0
        )
    }
}

/// The output of `Kernels::matmul` for `w` (n_out rows of n_in values of `ty`)
/// and the activations `x` (n_rows rows).
#[allow(clippy::too_many_arguments)]
fn run_matmul(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    path: PrefillPath,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> Vec<f32> {
    let stream = Stream::new(device, 0).unwrap();
    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
    dw.write(0, w).unwrap();
    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
    dx.write(0, &f32_bytes(x)).unwrap();
    let bytes = Kernels::prefill_activation_bytes(path, n_in as u32, n_rows as u32);
    let act = DeviceBuffer::new(device, bytes).unwrap();
    let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
    // SAFETY: the buffers were sized for these shapes just above.
    unsafe {
        kernels
            .prepare_prefill_activations(
                &stream,
                path,
                dx.ptr(),
                act.ptr(),
                n_in as u32,
                n_rows as u32,
            )
            .unwrap();
        kernels
            .matmul(
                &stream,
                ty,
                path,
                dw.ptr(),
                act.ptr(),
                dy.ptr(),
                n_in as u32,
                n_out as u32,
                n_rows as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    read_f32(&dy, n_rows * n_out)
}

/// The output of the decode kernel (`Kernels::matvec`, at most 8 rows) on the
/// same data, for the comparison of the families.
#[allow(clippy::too_many_arguments)]
fn run_matvec(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> Vec<f32> {
    let stream = Stream::new(device, 0).unwrap();
    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
    dw.write(0, w).unwrap();
    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
    dx.write(0, &f32_bytes(x)).unwrap();
    let dq =
        DeviceBuffer::new(device, Kernels::quantized_activation_bytes(n_in as u32, n_rows as u32))
            .unwrap();
    let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
    // SAFETY: the buffers were sized for these shapes just above.
    unsafe {
        kernels
            .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_rows as u32)
            .unwrap();
        kernels
            .matvec(
                &stream,
                ty,
                dw.ptr(),
                dq.ptr(),
                dy.ptr(),
                n_in as u32,
                n_out as u32,
                n_rows as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    read_f32(&dy, n_rows * n_out)
}

/// Compares `have` (the kernel's output) with both references.
#[allow(clippy::too_many_arguments)]
fn deviation(
    path: PrefillPath,
    ty: GgmlType,
    w: &[u8],
    x: &[f32],
    have: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> Deviation {
    let mut weights = vec![0f32; n_in * n_out];
    ares_quant::dequantize(ty, w, &mut weights).unwrap();
    // The weights as the kernel computes with them.
    let used: Vec<f64> = match path {
        PrefillPath::Int8 => weights.iter().map(|v| f64::from(*v)).collect(),
        PrefillPath::F16 | PrefillPath::F16Flush => {
            weights.iter().map(|v| through_f16(*v)).collect()
        }
    };
    let mut worst_tight = 0f64;
    let (mut err2, mut ref2, mut err_max, mut ref_max, mut scale2) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for b in 0..n_rows {
        let xb = &x[b * n_in..][..n_in];
        let xs: Vec<f64> = match path {
            PrefillPath::Int8 => quantized_like_the_gpu(xb),
            PrefillPath::F16 | PrefillPath::F16Flush => {
                xb.iter().map(|v| through_f16(*v)).collect()
            }
        };
        for r in 0..n_out {
            let (mut emulated, mut magnitude, mut exact, mut squares) = (0f64, 0f64, 0f64, 0f64);
            let row = &used[r * n_in..][..n_in];
            let exact_row = &weights[r * n_in..][..n_in];
            for k in 0..n_in {
                let p = row[k] * xs[k];
                emulated += p;
                magnitude += p.abs();
                let e = f64::from(exact_row[k]) * f64::from(xb[k]);
                exact += e;
                squares += e * e;
            }
            let h = f64::from(have[b * n_out + r]);
            worst_tight = worst_tight.max((h - emulated).abs() / magnitude.max(1e-30));
            err2 += (h - exact).powi(2);
            ref2 += exact.powi(2);
            scale2 += squares;
            err_max = err_max.max((h - exact).abs());
            ref_max = ref_max.max(exact.abs());
        }
    }
    let n = (n_rows * n_out) as f64;
    Deviation {
        tight: worst_tight,
        loose_rms: (err2 / n).sqrt() / (ref2 / n).sqrt().max(1e-30),
        loose_max: err_max / ref_max.max(1e-30),
        loose_scaled: (err2 / n).sqrt() / (scale2 / n).sqrt().max(1e-30),
    }
}

/// The bounds the deviations are held to. Int8 quantizes the activations to 8
/// bits per group of 32, as the decode kernels do (their tests use the same
/// bounds); f16 rounds the weights and activations to 11 bits.
fn assert_within(path: PrefillPath, what: &str, dev: &Deviation, enough_outputs: bool) {
    // The emulation rounds each weight once to f16 and sums exactly. The int8 kernel
    // computes exactly that. The f16 kernels make the weight from f16 scales with
    // packed f16 arithmetic (up to three roundings of 2^-11 instead of one), and
    // the flushing one also rounds partial sums of 64 products to f16.
    let tight = match path {
        PrefillPath::Int8 => 5e-5,
        PrefillPath::F16 => 2e-3,
        PrefillPath::F16Flush => 4e-3,
    };
    assert!(dev.tight < tight, "{what}: differs from the emulation: {}", dev.describe());
    if enough_outputs {
        let bound = match path {
            PrefillPath::Int8 => 0.03,
            PrefillPath::F16 | PrefillPath::F16Flush => 0.003,
        };
        assert!(dev.loose_scaled < bound, "{what}: far from the f32 product: {}", dev.describe());
    }
}

/// Every kernel for shapes around its edges: one row and one weight row, row
/// counts that do not fill a tile (64 activation rows, 128 or 64 weight rows),
/// rows of a block and of many, a row of 32-value blocks that is not a multiple
/// of the step of 64, and a matrix with many blocks of threads.
#[test]
fn prefill_matmul_computes_the_emulated_product_on_every_type() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            let mut shapes = vec![
                (bv, 1, 1),
                (bv, 3, 9),
                (bv * 5, 13, 70),
                (bv * 20, 129, 65),
                (bv * 8, 100, 130),
                (if bv == 256 { 2048 } else { 1024 }, 200, 33),
            ];
            if bv == 32 {
                // Not a whole number of steps of 64 values.
                shapes.push((96, 40, 17));
                shapes.push((32 * 25, 77, 66));
            }
            for (n_in, n_out, n_rows) in shapes {
                let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                for path in PATHS {
                    let have = run_matmul(device, &kernels, ty, path, &w, &x, n_in, n_out, n_rows);
                    let dev = deviation(path, ty, &w, &x, &have, n_in, n_out, n_rows);
                    eprintln!(
                        "{ty} {n_out}x{n_in} rows {n_rows} {path:?} on {}: {}",
                        device.info().name,
                        dev.describe()
                    );
                    assert_within(
                        path,
                        &format!("{ty} {n_out}x{n_in} x{n_rows} {path:?}"),
                        &dev,
                        n_out * n_rows >= 16,
                    );
                }
            }
        }
    }
}

/// The edges of every shape: the numbers of activation rows around each limit
/// where the choice of shape changes (32, 64) and around the tiles of the shapes
/// (32, 64 and 128 columns of the output), and the numbers of weight rows around
/// the tiles (64 and 128), each on its own and crossing the other. Partial tiles in
/// both dimensions, for both paths and every type.
#[test]
fn prefill_matmul_is_right_around_every_shape_boundary() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x5851f42d4c957f2d);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            let n_in = 512usize;
            let mut cases: Vec<(usize, usize)> = [
                1usize, 2, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 96, 127, 128, 129, 130, 191,
                255, 257,
            ]
            .iter()
            .map(|rows| (130usize, *rows))
            .collect();
            for n_out in [1usize, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 257] {
                for rows in [5usize, 40, 100] {
                    cases.push((n_out, rows));
                }
            }
            let w = random_blocks(&mut rng, ty, n_in / bv * 257);
            let x_all: Vec<f32> = (0..n_in * 257).map(|_| rng.activation()).collect();
            for (n_out, n_rows) in cases {
                let w = &w[..n_in / bv * n_out * ty.block_bytes() as usize];
                let x = &x_all[..n_in * n_rows];
                for path in PATHS {
                    let have = run_matmul(device, &kernels, ty, path, w, x, n_in, n_out, n_rows);
                    let dev = deviation(path, ty, w, x, &have, n_in, n_out, n_rows);
                    assert_within(
                        path,
                        &format!(
                            "{ty} {n_out}x{n_in} x{n_rows} {path:?} on {}",
                            device.info().name
                        ),
                        &dev,
                        n_out * n_rows >= 16,
                    );
                }
            }
        }
    }
}

/// The same on real tensors: the smallest 2-D matrix of each type in the model,
/// its first 200 rows, with activations of a normal distribution and a batch
/// that fills no tile.
#[test]
fn prefill_matmul_on_real_tensors() {
    let Some(gpus) = gpus() else { return };
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!("skipped: set ARES_TEST_GGUF=<model.gguf> to check real tensors");
        return;
    };
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    let mut rng = Rng(0x853c49e6748fea9b);
    let mut checked = Vec::new();
    for (ty, _) in TYPES {
        let Some(tensor) = gguf
            .tensors()
            .iter()
            .filter(|t| t.ty == ty && t.dims.as_slice().len() == 2)
            .min_by_key(|t| t.size)
        else {
            continue;
        };
        let (n_in, n_out) = (tensor.dims.get(0) as usize, tensor.dims.get(1) as usize);
        let rows = n_out.min(200);
        let row_bytes = ty.bytes_for(n_in as u64).unwrap() as usize;
        let w = &gguf.data(tensor)[..rows * row_bytes];
        for n_rows in [8usize, 9, 70] {
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for device in gpus {
                let kernels = Kernels::load(device).unwrap();
                if !kernels.has_prefill_kernels() {
                    continue;
                }
                for path in PATHS {
                    let have = run_matmul(device, &kernels, ty, path, w, &x, n_in, rows, n_rows);
                    let dev = deviation(path, ty, w, &x, &have, n_in, rows, n_rows);
                    eprintln!(
                        "{} ({ty}, {rows} x {n_in}) rows {n_rows} prefill {path:?} on {}: {}",
                        tensor.name,
                        device.info().name,
                        dev.describe()
                    );
                    assert_within(path, &tensor.name, &dev, true);
                }
                if n_rows <= 8 {
                    // The decode family computes what the int8 path does.
                    let have = run_matvec(device, &kernels, ty, w, &x, n_in, rows, n_rows);
                    let dev = deviation(PrefillPath::Int8, ty, w, &x, &have, n_in, rows, n_rows);
                    eprintln!(
                        "{} ({ty}, {rows} x {n_in}) rows {n_rows} decode on {}: {}",
                        tensor.name,
                        device.info().name,
                        dev.describe()
                    );
                    assert_within(PrefillPath::Int8, &tensor.name, &dev, true);
                }
            }
        }
        checked.push(ty);
    }
    assert!(!checked.is_empty(), "the model has no 2-D tensor of a type with a prefill kernel");
    eprintln!("checked types: {}", checked.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "));
}

/// The int8 path is the decode kernels' arithmetic on the tensor cores, so on
/// the same quantized activations the two give the same numbers up to f32
/// rounding of the sums.
#[test]
fn the_int8_path_agrees_with_the_decode_kernels() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xa076_1d64_78bd_642f);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        let stream = Stream::new(device, 0).unwrap();
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            let (n_in, n_out, n_rows) = (bv * 12, 150usize, 8usize);
            let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
            dw.write(0, &w).unwrap();
            let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
            dx.write(0, &f32_bytes(&x)).unwrap();
            let dq = DeviceBuffer::new(
                device,
                Kernels::quantized_activation_bytes(n_in as u32, n_rows as u32),
            )
            .unwrap();
            let y_decode = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
            let y_prefill = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
            // SAFETY: the buffers were sized for these shapes just above.
            unsafe {
                kernels
                    .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_rows as u32)
                    .unwrap();
                kernels
                    .matvec(
                        &stream,
                        ty,
                        dw.ptr(),
                        dq.ptr(),
                        y_decode.ptr(),
                        n_in as u32,
                        n_out as u32,
                        n_rows as u32,
                    )
                    .unwrap();
                kernels
                    .matmul(
                        &stream,
                        ty,
                        PrefillPath::Int8,
                        dw.ptr(),
                        dq.ptr(),
                        y_prefill.ptr(),
                        n_in as u32,
                        n_out as u32,
                        n_rows as u32,
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let (a, b) =
                (read_f32(&y_decode, n_rows * n_out), read_f32(&y_prefill, n_rows * n_out));
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = a.iter().zip(&b).fold(0f32, |m, (p, q)| m.max((p - q).abs())) / scale;
            eprintln!(
                "{ty} on {}: decode vs prefill int8, worst {worst:.1e} of the largest output",
                device.info().name
            );
            assert!(worst < 1e-5, "{ty}: the two families differ by {worst:e}");
        }
    }
}

/// A row of the result does not depend on the other rows of the batch, or on
/// where in the grid it is computed, or on which shape computes it: rows 0 to 19
/// of a batch of 130 (the large shape), of 40 (medium) and of 20 (small) are the
/// same bits, and so are single rows, for both paths. (The sums of an output are
/// formed in the same order whatever the shape.)
#[test]
fn a_row_gives_the_same_result_alone_and_in_a_batch() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xda942042e4dd58b5);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            let (n_in, n_out, n_rows) = (bv * 8, 150usize, 130usize);
            let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for path in PATHS {
                let batch = run_matmul(device, &kernels, ty, path, &w, &x, n_in, n_out, n_rows);
                for rows in [40usize, 20] {
                    let part = run_matmul(
                        device,
                        &kernels,
                        ty,
                        path,
                        &w,
                        &x[..rows * n_in],
                        n_in,
                        n_out,
                        rows,
                    );
                    assert_eq!(
                        &batch[..rows * n_out],
                        &part[..],
                        "{ty} {path:?}, first {rows} rows of 130"
                    );
                }
                for b in [0usize, 31, 32, 63, 64, 129] {
                    let alone = run_matmul(
                        device,
                        &kernels,
                        ty,
                        path,
                        &w,
                        &x[b * n_in..][..n_in],
                        n_in,
                        n_out,
                        1,
                    );
                    assert_eq!(
                        &batch[b * n_out..][..n_out],
                        &alone[..],
                        "{ty} {path:?}, batch row {b}"
                    );
                }
            }
        }
    }
}

/// The shapes are described as they sit on the device: every one holds at least a
/// block of an SM, the shared memory fits the opt-in limit, and the rows they take
/// grow from the small shape to the large.
#[test]
fn the_shapes_fit_the_device() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for (ty, _) in TYPES {
            for path in PATHS {
                let shapes = kernels.prefill_shapes(ty, path).unwrap();
                assert_eq!(shapes.len(), 3);
                for shape in &shapes {
                    assert!(shape.blocks_per_sm >= 1, "{ty} {path:?} {shape:?}");
                    assert!(shape.shared_bytes <= device.info().max_shared_per_block_optin);
                    assert!(shape.registers * shape.threads <= 65536, "{ty} {path:?} {shape:?}");
                }
                assert!(shapes[0].up_to_rows > shapes[1].up_to_rows);
                assert!(shapes[1].up_to_rows > shapes[2].up_to_rows);
            }
        }
    }
}

/// What `Kernels` says about the device is consistent: the path of the exact
/// precision is always f16, the fast one is int8 exactly where the tensor cores
/// are capped. The measurement is printed.
#[test]
fn the_path_follows_the_measured_tensor_core_rate() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            assert!(device.info().compute_capability.0 < 8);
            assert!(!kernels.tensor_cores_full_rate());
            assert!(kernels.prefill_path(Precision::Fast).is_err());
            continue;
        }
        let full = kernels.tensor_cores_full_rate();
        eprintln!("{}: f16 mma with f32 accumulation at full rate: {full}", device.info().name);
        assert_eq!(kernels.prefill_path(Precision::Exact).unwrap(), PrefillPath::F16);
        let fast = kernels.prefill_path(Precision::Fast).unwrap();
        assert_eq!(fast, if full { PrefillPath::F16 } else { PrefillPath::Int8 });
        for ty in [GgmlType::Q4_K, GgmlType::IQ2_S, GgmlType::IQ2_XS, GgmlType::Q6_K] {
            let flush = !full && ty != GgmlType::Q6_K;
            let expected = if flush { PrefillPath::F16Flush } else { fast };
            assert_eq!(kernels.prefill_path_for(ty, Precision::Fast).unwrap(), expected, "{ty}");
            assert_eq!(kernels.prefill_path_for(ty, Precision::Exact).unwrap(), PrefillPath::F16);
        }
    }
}

/// Refusals happen before anything is launched: a type without a prefill kernel
/// names the types that have one, and the sizes and alignments the kernels rely
/// on are checked.
#[test]
fn bad_arguments_are_refused() {
    let Some(gpus) = gpus() else { return };
    let device = &gpus[0];
    let kernels = Kernels::load(device).unwrap();
    if !kernels.has_prefill_kernels() {
        return;
    }
    let stream = Stream::new(device, 0).unwrap();
    let buffer = DeviceBuffer::new(device, 1 << 20).unwrap();
    let p = buffer.ptr();
    let null = DevicePtr::NULL;
    let int8 = PrefillPath::Int8;

    for ty in [GgmlType::F16, GgmlType::Q4_0, GgmlType::Q2_K, GgmlType::IQ1_S, GgmlType::MXFP4] {
        assert!(!Kernels::has_prefill_matmul(ty));
        // SAFETY: refused before anything is launched.
        let error = unsafe { kernels.matmul(&stream, ty, int8, p, p, p, 256, 1, 1) }
            .unwrap_err()
            .to_string();
        assert!(error.contains(ty.name()) && error.contains("no prefill kernel"), "{error}");
        assert!(error.contains("q4_K") && error.contains("iq4_xs"), "{error}");
    }
    for (ty, _) in TYPES {
        assert!(Kernels::has_prefill_matmul(ty));
        // Both families exist for the type, and the rule says where one hands over to the other.
        assert!(Kernels::has_fast_matvec(ty));
        let rows = Kernels::decode_kernel_rows(ty).unwrap();
        assert!((1..=32).contains(&rows), "{ty}: {rows}");
    }
    assert_eq!(Kernels::decode_kernel_rows(GgmlType::F16), None);

    // SAFETY: every call below is refused before anything is launched.
    unsafe {
        // Not a whole number of blocks.
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 300, 1, 1).is_err());
        assert!(kernels.matmul(&stream, GgmlType::Q8_0, int8, p, p, p, 48, 1, 1).is_err());
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 0, 1, 1).is_err());
        // No rows, or too many to count.
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 256, 1, 0).is_err());
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 4096, 1, 1 << 20).is_err());
        assert!(kernels.prepare_prefill_activations(&stream, int8, p, p, 256, 0).is_err());
        assert!(kernels.prepare_prefill_activations(&stream, int8, p, p, 48, 1).is_err());
        // Alignment: Q4_K needs 16 bytes of the weights, IQ4_XS 8, Q8_0 2; the activations 16.
        assert!(
            kernels.matmul(&stream, GgmlType::Q4_K, int8, p.offset(8), p, p, 256, 1, 1).is_err()
        );
        assert!(
            kernels.matmul(&stream, GgmlType::IQ4_XS, int8, p.offset(4), p, p, 256, 1, 1).is_err()
        );
        assert!(
            kernels.matmul(&stream, GgmlType::Q8_0, int8, p.offset(1), p, p, 32, 1, 1).is_err()
        );
        assert!(
            kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p.offset(8), p, 256, 1, 1).is_err()
        );
        assert!(
            kernels.prepare_prefill_activations(&stream, int8, p, p.offset(4), 256, 1).is_err()
        );
        assert!(
            kernels
                .prepare_prefill_activations(&stream, PrefillPath::F16, p, p.offset(4), 256, 1)
                .is_err()
        );
        // No weight rows is nothing to do, not an error, and a null address is not touched.
        kernels.matmul(&stream, GgmlType::Q4_K, int8, null, null, null, 256, 0, 1).unwrap();
    }
}

/// The output of `Kernels::matmul_rearranged` for a Q4_K matrix `w` (file form,
/// rearranged on the device first).
#[allow(clippy::too_many_arguments)]
fn run_matmul_rearranged(
    device: &Device,
    kernels: &Kernels,
    path: PrefillPath,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> Vec<f32> {
    let stream = Stream::new(device, 0).unwrap();
    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
    dw.write(0, w).unwrap();
    let bytes = Kernels::rearranged_bytes(GgmlType::Q4_K, n_out as u32, n_in as u32).unwrap();
    let dr = DeviceBuffer::new(device, bytes).unwrap();
    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
    dx.write(0, &f32_bytes(x)).unwrap();
    let act = DeviceBuffer::new(
        device,
        Kernels::prefill_activation_bytes(path, n_in as u32, n_rows as u32),
    )
    .unwrap();
    let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
    // SAFETY: the buffers were sized for these shapes just above.
    unsafe {
        kernels
            .rearrange(&stream, GgmlType::Q4_K, dw.ptr(), dr.ptr(), n_out as u32, n_in as u32)
            .unwrap();
        kernels
            .prepare_prefill_activations(
                &stream,
                path,
                dx.ptr(),
                act.ptr(),
                n_in as u32,
                n_rows as u32,
            )
            .unwrap();
        kernels
            .matmul_rearranged(
                &stream,
                GgmlType::Q4_K,
                path,
                dr.ptr(),
                act.ptr(),
                dy.ptr(),
                n_in as u32,
                n_out as u32,
                n_rows as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    read_f32(&dy, n_rows * n_out)
}

/// The rearranged form of Q4_K is the same number of bytes (rows rounded up to 16),
/// and the kernel that reads it gives the product of the tile kernel on the file's
/// form (the same f16 weights and sums; only the order of the f32 sums can differ)
/// and is as close to the f32 product as the f16 paths are, for matrices that
/// leave tiles of 16 and 128 rows and of 128 activation rows partly empty.
#[test]
fn the_rearranged_form_gives_the_product_of_the_file_form() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    assert_eq!(
        Kernels::rearranged_bytes(GgmlType::Q4_K, 17408, 5120),
        Some(17408 * 5120 / 256 * 144)
    );
    assert_eq!(Kernels::rearranged_bytes(GgmlType::Q4_K, 17, 256), Some(32 * 144));
    assert_eq!(Kernels::rearranged_bytes(GgmlType::Q6_K, 16, 256), None);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        let shapes = [
            (256, 1, 1),
            (256, 16, 8),
            (512, 17, 3),
            (256, 100, 65),
            (1280, 129, 127),
            (768, 130, 128),
            (512, 300, 129),
            (1024, 257, 300),
            (2560, 512, 33),
        ];
        for (n_in, n_out, n_rows) in shapes {
            let w = random_blocks(&mut rng, GgmlType::Q4_K, n_in / 256 * n_out);
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for path in [PrefillPath::F16, PrefillPath::F16Flush] {
                let have =
                    run_matmul_rearranged(device, &kernels, path, &w, &x, n_in, n_out, n_rows);
                let tile =
                    run_matmul(device, &kernels, GgmlType::Q4_K, path, &w, &x, n_in, n_out, n_rows);
                let dev = deviation(path, GgmlType::Q4_K, &w, &x, &have, n_in, n_out, n_rows);
                eprintln!(
                    "Q4_K {n_out}x{n_in} rows {n_rows} {path:?} rearranged on {}: {}",
                    device.info().name,
                    dev.describe()
                );
                assert_within(
                    path,
                    &format!("rearranged Q4_K {n_out}x{n_in} x{n_rows} {path:?}"),
                    &dev,
                    n_out * n_rows >= 16,
                );
                // Against the tile kernel: relative to the size of the sums.
                let mut worst = 0f64;
                for b in 0..n_rows {
                    for r in 0..n_out {
                        let d = f64::from(have[b * n_out + r] - tile[b * n_out + r]).abs();
                        let m = f64::from(have[b * n_out + r].abs().max(tile[b * n_out + r].abs()));
                        worst = worst.max(d / m.max(1e-3));
                    }
                }
                assert!(
                    worst < 2e-3,
                    "rearranged Q4_K {n_out}x{n_in} x{n_rows} {path:?}: differs from the tile kernel by {worst:.2e}"
                );
            }
        }
    }
}
