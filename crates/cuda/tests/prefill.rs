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
    Device, DeviceBuffer, DevicePtr, Kernels, Precision, PrefillPath, Stream, Wait, WeightRange,
    device_count,
};
use ares_quant::{f16_to_f32, f32_to_bf16, f32_to_f16};

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

/// The types of the second module (`kernels/prefill_more.cu`): the f16 variants only.
const MORE_TYPES: [(GgmlType, &[usize]); 14] = [
    (GgmlType::IQ2_XXS, &[0]),
    (GgmlType::IQ1_S, &[0]),
    (GgmlType::IQ1_M, &[]),
    (GgmlType::Q2_0, &[0]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::TQ1_0, &[52]),
    (GgmlType::TQ2_0, &[64]),
    (GgmlType::MXFP4, &[]),
    (GgmlType::NVFP4, &[]),
    (GgmlType::Q1_0, &[0]),
];

/// The float types, also of the second module: no scales in the data.
const FLOAT_TYPES: [GgmlType; 3] = [GgmlType::F32, GgmlType::F16, GgmlType::BF16];

const PATHS: [PrefillPath; 3] = [PrefillPath::Int8, PrefillPath::F16, PrefillPath::F16Flush];

/// Every type with a tile kernel.
fn all_types() -> impl Iterator<Item = (GgmlType, &'static [usize])> {
    TYPES.into_iter().chain(MORE_TYPES).chain(FLOAT_TYPES.map(|ty| (ty, &[][..])))
}

/// The values of a row the kernels take at once: a block, or 32 for the float types, whose
/// blocks are one value.
fn unit(ty: GgmlType) -> usize {
    (ty.block_values() as usize).max(32)
}

/// The paths a type has kernels for.
fn paths_of(kernels: &Kernels, ty: GgmlType) -> Vec<PrefillPath> {
    PATHS.into_iter().filter(|p| kernels.has_prefill_path(ty, *p)).collect()
}

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

/// The values as weights of a float type.
fn float_bytes(ty: GgmlType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match ty {
            GgmlType::F32 => v.to_le_bytes().to_vec(),
            GgmlType::F16 => f32_to_f16(*v).to_le_bytes().to_vec(),
            _ => f32_to_bf16(*v).to_le_bytes().to_vec(),
        })
        .collect()
}

/// `n` weights of a float type: normal numbers of the size of a weight, a few much larger.
fn random_floats(rng: &mut Rng, ty: GgmlType, n: usize) -> Vec<u8> {
    let values: Vec<f32> = (0..n).map(|_| rng.activation() * 0.05).collect();
    float_bytes(ty, &values)
}

/// `n_blocks` blocks of `ty` of random bytes, with finite scales (f16 of both signs; a
/// small power of two for MXFP4's exponent byte and NVFP4's four UE4M3 bytes), where the
/// bytes of a scale could otherwise be infinite or NaN. Any other bytes are valid in every
/// one of these types.
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    if FLOAT_TYPES.contains(&ty) {
        return random_floats(rng, ty, n_blocks);
    }
    let (_, scales) = all_types().find(|(t, _)| *t == ty).unwrap();
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for b in 0..n_blocks {
        for &offset in scales {
            let scale = (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5))
                * if rng.next().is_multiple_of(4) { -1.0 } else { 1.0 };
            let at = b * block + offset;
            bytes[at..at + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        }
        match ty {
            // The E8M0 exponent byte: 2^-7 to 2^3.
            GgmlType::MXFP4 => bytes[b * block] = 120 + (rng.next() % 11) as u8,
            // The four UE4M3 scale bytes: 0.125 to 2.
            GgmlType::NVFP4 => {
                for i in 0..4 {
                    bytes[b * block + i] = 0x20 + (rng.next() % 0x21) as u8;
                }
            }
            _ => {}
        }
        if ty == GgmlType::IQ1_M {
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            let d = f32_to_f16(0.002 + 0.02 * (rng.unit() * 0.5 + 0.5));
            for i in 0..4 {
                let at = b * block + 48 + 2 * i;
                let word = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
                bytes[at..at + 2].copy_from_slice(&word.to_le_bytes());
            }
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
    /// Largest `|have - emulation| / sum of |w x|` over the outputs. For the path with f16
    /// accumulation, what the rounding of small partial sums may cost is taken off the
    /// difference first: the f16 sum is rounded after every 8 products, and where it is below
    /// 2^-13 of the row's factor a rounding is absolute, at most 2^-24 of the factor (derived
    /// at `tiny_float_weights_are_off_by_at_most_the_spacing_of_f16_subnormals`). An output of
    /// a row of small weights among larger ones is off by that, whatever its own size.
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
        let small_sums = if path == PrefillPath::F16Flush && xb.iter().any(|v| *v != 0.0) {
            n_in.div_ceil(8) as f64 * 2f64.powi(-24) * f64::from(row_factor(xb))
        } else {
            0.0
        };
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
            let difference = ((h - emulated).abs() - small_sums).max(0.0);
            worst_tight = worst_tight.max(difference / magnitude.max(1e-30));
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
        for (ty, _) in all_types() {
            let bv = ty.block_values() as usize;
            let step = unit(ty);
            let mut shapes = vec![
                (step, 1, 1),
                (step, 3, 9),
                (step * 5, 13, 70),
                (step * 20, 129, 65),
                (step * 8, 100, 130),
                (if bv == 256 { 2048 } else { 1024 }, 200, 33),
            ];
            if step == 32 {
                // Not a whole number of steps of 64 values.
                shapes.push((96, 40, 17));
                shapes.push((32 * 25, 77, 66));
            }
            for (n_in, n_out, n_rows) in shapes {
                let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                for path in paths_of(&kernels, ty) {
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
        for (ty, _) in all_types() {
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
                for path in paths_of(&kernels, ty) {
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
    for (ty, _) in all_types() {
        let candidates = gguf.tensors().iter().filter(|t| {
            t.ty == ty
                && t.dims.as_slice().len() == 2
                && (t.dims.get(0) as usize).is_multiple_of(unit(ty))
        });
        // The smallest matrix of a type; of a float type (the few that models keep, of many
        // kinds) the first of each kind: the same name without the layer's number.
        let mut chosen = Vec::new();
        if FLOAT_TYPES.contains(&ty) {
            let mut kinds = Vec::new();
            for tensor in candidates {
                let kind: String = tensor.name.chars().filter(|c| !c.is_ascii_digit()).collect();
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                    chosen.push(tensor);
                }
            }
        } else {
            chosen.extend(candidates.min_by_key(|t| t.size));
        }
        if chosen.is_empty() {
            continue;
        }
        for tensor in chosen {
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
                    for path in paths_of(&kernels, ty) {
                        let have =
                            run_matmul(device, &kernels, ty, path, w, &x, n_in, rows, n_rows);
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
                        let have = run_matvec(device, &kernels, ty, w, &x, n_in, rows, n_rows);
                        let dev =
                            deviation(PrefillPath::Int8, ty, w, &x, &have, n_in, rows, n_rows);
                        eprintln!(
                            "{} ({ty}, {rows} x {n_in}) rows {n_rows} decode on {}: {}",
                            tensor.name,
                            device.info().name,
                            dev.describe()
                        );
                        if FLOAT_TYPES.contains(&ty) {
                            // The decode kernels of the float types multiply the exact activations.
                            assert!(dev.loose_scaled < 1e-5, "{}: {}", tensor.name, dev.describe());
                        } else {
                            // The decode family computes what the int8 path does.
                            assert_within(PrefillPath::Int8, &tensor.name, &dev, true);
                        }
                    }
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
        for (ty, _) in all_types() {
            let bv = ty.block_values() as usize;
            let (n_in, n_out, n_rows) = (unit(ty) * 8, 150usize, 130usize);
            let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for path in paths_of(&kernels, ty) {
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

/// A float weight beyond the range of f16 must not turn into a wrong finite number: the
/// tile holds an infinity, and the outputs that read it are infinite or NaN, on every path.
/// The other outputs are right.
#[test]
fn a_float_weight_beyond_f16_makes_its_outputs_infinite_and_no_others_wrong() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let (n_in, n_out, n_rows, bad_row) = (256usize, 70usize, 40usize, 7usize);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for ty in [GgmlType::F32, GgmlType::BF16] {
            let mut values: Vec<f32> = (0..n_in * n_out).map(|_| rng.activation() * 0.05).collect();
            values[bad_row * n_in + 100] = -7.0e4;
            values[(bad_row + 1) * n_in + 3] = 1.0e30;
            let w = float_bytes(ty, &values);
            // Which is why a loader asks for the range of a matrix and takes the decode kernels.
            assert_eq!(WeightRange::of(ty, &w), WeightRange::Beyond);
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation() + 2.0).collect();
            for path in paths_of(&kernels, ty) {
                let have = run_matmul(device, &kernels, ty, path, &w, &x, n_in, n_out, n_rows);
                let mut right = vec![true; n_out];
                right[bad_row] = false;
                right[bad_row + 1] = false;
                for b in 0..n_rows {
                    for r in 0..n_out {
                        let y = have[b * n_out + r];
                        if right[r] {
                            assert!(y.is_finite(), "{ty} {path:?} row {b} output {r}: {y}");
                        } else {
                            assert!(!y.is_finite(), "{ty} {path:?} row {b} output {r}: {y}");
                        }
                    }
                }
                // The finite outputs are those of the matrix without the two weights.
                let ok_rows: Vec<usize> = (0..n_out).filter(|r| right[*r]).collect();
                let mut kept = Vec::new();
                for r in &ok_rows {
                    kept.extend_from_slice(&values[r * n_in..][..n_in]);
                }
                let w_ok = float_bytes(ty, &kept);
                let reference =
                    run_matmul(device, &kernels, ty, path, &w_ok, &x, n_in, ok_rows.len(), n_rows);
                for b in 0..n_rows {
                    for (i, r) in ok_rows.iter().enumerate() {
                        assert_eq!(have[b * n_out + r], reference[b * ok_rows.len() + i]);
                    }
                }
            }
        }
    }
}

/// Weights that f16 holds but whose sums it does not. Above `WeightRange::F16_SUMS_MAX` the
/// path with f16 accumulation can overflow: a run of 64 weights of 8000 against activations of
/// one sign gives an infinite output there, and the product on the f16 path, whose sums are
/// f32. At the limit itself it cannot: 64 weights of 512 against the largest activations a
/// divided row holds give the product.
#[test]
fn the_f16_sums_take_weights_up_to_their_limit_and_overflow_above() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x94d049bb133111eb);
    let (n_in, n_out, n_rows, large_row) = (256usize, 70usize, 40usize, 7usize);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for ty in [GgmlType::F32, GgmlType::BF16] {
            for (weight, range) in [(8000.0, WeightRange::F32Sums), (512.0, WeightRange::Any)] {
                let mut values: Vec<f32> =
                    (0..n_in * n_out).map(|_| rng.activation() * 0.05).collect();
                values[large_row * n_in + 64..large_row * n_in + 128].fill(weight);
                let w = float_bytes(ty, &values);
                assert_eq!(WeightRange::of(ty, &w), range, "{ty} {weight}");
                let mut exact_weights = vec![0f32; n_in * n_out];
                ares_quant::dequantize(ty, &w, &mut exact_weights).unwrap();
                // Just below a power of two, so that the divided values are just below 1.
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| 0.96 + 0.03 * rng.unit()).collect();
                for path in paths_of(&kernels, ty) {
                    let have = run_matmul(device, &kernels, ty, path, &w, &x, n_in, n_out, n_rows);
                    let held = range.path(Some(path)) == Some(path);
                    for b in 0..n_rows {
                        for r in 0..n_out {
                            let y = f64::from(have[b * n_out + r]);
                            if r == large_row && !held {
                                assert!(y.is_infinite(), "{ty} {path:?} row {b}: {y}");
                                continue;
                            }
                            let (mut exact, mut magnitude) = (0f64, 0f64);
                            for k in 0..n_in {
                                let p = f64::from(exact_weights[r * n_in + k])
                                    * f64::from(x[b * n_in + k]);
                                exact += p;
                                magnitude += p.abs();
                            }
                            // The activations are rounded by 2^-11 of each product, and the
                            // f16 sum of a group by at most 2^-10 of each of its 8 partial
                            // sums, the last of which is the group: 4.9e-3 in all.
                            assert!(
                                (y - exact).abs() <= 5e-3 * magnitude,
                                "{ty} {weight} {path:?} row {b} output {r}: {y} for {exact}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The distance between neighbouring f16 values at the magnitude `v`: 2^-24 below 2^-13 (the
/// subnormals and the first binade of normal numbers), 2^-10 of the binade above.
fn f16_spacing(v: f64) -> f64 {
    let exponent = if v > 0.0 { ((v.to_bits() >> 52) & 0x7ff) as i32 - 1023 } else { -14 };
    2f64.powi(exponent.max(-14) - 10)
}

/// `v` rounded to the nearest f16, halves to even.
fn nearest_f16(v: f64) -> f64 {
    let spacing = f16_spacing(v.abs());
    (v / spacing).round_ties_even() * spacing
}

/// The power of two `ares_convert_f16` divides a row of activations by: the one above the
/// row's largest magnitude, so that the divided values are below 1.
fn row_factor(x: &[f32]) -> f32 {
    let largest = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(largest.is_normal());
    2f32.powi(((largest.to_bits() >> 23) & 0xff) as i32 - 126)
}

/// Weights below 2^-13 (1.2e-4) lie where f16 values are 2^-24 apart (the subnormals and the
/// first binade of normal numbers), so the tile holds each of them within 2^-25, whatever its
/// size. This is what that and the other roundings of the kernels do to an output, and the
/// test holds the kernels to the sum of it.
///
/// For a row of activations `x` with the factor `s` (the power of two above its largest
/// magnitude, by which `ares_convert_f16` divides the row) a kernel computes
/// `y = s * sum of f16(w[k]) * f16(x[k] / s)`. In the units of `y`:
///
/// - *Weights.* `|f16(w) - w| <= 2^-25`: an F32 weight is rounded once, to nearest (`cvt`), a
///   BF16 weight is first widened to f32, which changes nothing. Times the activations as the
///   kernel has them: `2^-25 * sum |x^[k]|`, with `x^ = s * f16(x / s)`.
/// - *Activations.* `|x^ - x| <= max(2^-11 |x|, 2^-25 s)`, the second for the values below
///   `2^-14 s`, which are subnormal once divided: `sum |w[k]| * |x^[k] - x[k]|` (the test takes
///   the difference itself).
/// - *Sums of the f16 path.* The products are exact (22 bits). An `mma` adds 16 of them to an
///   f32 sum and leaves it within a spacing of f32, `2^-23` of its magnitude, `n_in / 16` times.
/// - *Sums of the path with f16 accumulation.* The sum of each 64 values is kept as an f16, and
///   it is rounded after every 8 products: the tensor cores take the 16 products of an `mma` in
///   two steps (measured on the two cards at hand: rounding the sum to the nearest f16 after
///   every 8 products on the CPU gives the bits of the kernel for the share of the outputs the
///   test prints, 99.7 to 99.8 %; after every 16, or truncating, for 15 % at most). A step is
///   off by at most the spacing of f16 at the partial sum `h`, and the partial sum is in the
///   units of the divided row: `s * 2^-24` for `|h| < 2^-13`, at most `s * 2^-10 |h|` above
///   (the bound allows the whole spacing, so it does not depend on the rounding being to
///   nearest). So with weights this small every step loses up to `2^-24` of the row's factor,
///   3.8e-6 for a row whose largest activation is 40, where the f16 path loses nothing; over
///   `n_in / 8` steps that is the larger part of this path's error. Each group's sum is then
///   added to an f32: `n_in / 64` times a spacing of f32.
///
/// The measured error is printed with the bound and the three parts of the bound.
#[test]
fn tiny_float_weights_are_off_by_at_most_the_spacing_of_f16_subnormals() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    let (n_in, n_out, n_rows) = (512usize, 130usize, 33usize);
    let (weight_step, f32_step) = (2f64.powi(-25), 2f64.powi(-23));
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        for ty in [GgmlType::F32, GgmlType::BF16] {
            // From 1e-7 to 1e-4, the range where f16 stops being exact for a bf16.
            let values: Vec<f32> = (0..n_in * n_out)
                .map(|_| {
                    let magnitude = 10f32.powf(-7.0 + 3.0 * (rng.unit() * 0.5 + 0.5));
                    magnitude * if rng.next().is_multiple_of(2) { -1.0 } else { 1.0 }
                })
                .collect();
            assert!(values.iter().all(|v| v.abs() < 2f32.powi(-13)));
            let w = float_bytes(ty, &values);
            let mut exact_weights = vec![0f32; n_in * n_out];
            ares_quant::dequantize(ty, &w, &mut exact_weights).unwrap();
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for path in paths_of(&kernels, ty) {
                let flush = path == PrefillPath::F16Flush;
                let have = run_matmul(device, &kernels, ty, path, &w, &x, n_in, n_out, n_rows);
                // The output that comes closest to its bound: error, bound and the bound's parts.
                let mut worst = (0f64, 1f64, [0f64; 3]);
                let mut emulated = 0usize;
                for b in 0..n_rows {
                    let xb = &x[b * n_in..][..n_in];
                    let factor = row_factor(xb);
                    let s = f64::from(factor);
                    // The activations as the kernel has them, in the units of the divided row.
                    let divided: Vec<f64> = xb.iter().map(|v| through_f16(*v / factor)).collect();
                    for r in 0..n_out {
                        let row = &exact_weights[r * n_in..][..n_in];
                        let (mut exact, mut weights, mut activations) = (0f64, 0f64, 0f64);
                        // In the units of the divided row: the sum of the magnitudes so far, and
                        // what the f32 sum may be off by.
                        let (mut magnitude, mut off) = (0f64, 0f64);
                        // The emulation of the path with f16 accumulation.
                        let mut sum = 0f32;
                        for group in 0..n_in / 64 {
                            // The f16 sum of the group: its magnitude at most, what it may be
                            // off by, and its emulation.
                            let (mut partial, mut partial_off, mut h) = (0f64, 0f64, 0f64);
                            for step in 0..8 {
                                let mut products = 0f64;
                                for k in group * 64 + step * 8..group * 64 + step * 8 + 8 {
                                    let (wv, xv) = (f64::from(row[k]), f64::from(xb[k]));
                                    let tile = through_f16(row[k]);
                                    exact += wv * xv;
                                    weights += weight_step * (divided[k] * s).abs();
                                    activations += wv.abs() * (divided[k] * s - xv).abs();
                                    partial += (tile * divided[k]).abs();
                                    products += tile * divided[k];
                                }
                                if flush {
                                    partial_off += f16_spacing(partial + partial_off);
                                    h = nearest_f16(h + products);
                                } else if step % 2 == 1 {
                                    off += f32_step * (magnitude + partial + off);
                                }
                            }
                            magnitude += partial;
                            if flush {
                                off += partial_off;
                                off += f32_step * (magnitude + off);
                                sum += h as f32;
                            }
                        }
                        let parts = [weights, activations, off * s];
                        let bound: f64 = parts.iter().sum();
                        let error = (f64::from(have[b * n_out + r]) - exact).abs();
                        assert!(
                            error <= bound,
                            "{ty} {path:?} row {b} output {r}: {error:e} > {bound:e} (weights, activations, sums: {parts:?})"
                        );
                        if error / bound > worst.0 / worst.1 {
                            worst = (error, bound, parts);
                        }
                        emulated += usize::from(have[b * n_out + r] == sum * factor);
                    }
                }
                let (error, bound, [weights, activations, sums]) = worst;
                eprintln!(
                    "{ty} weights from 1e-7 to 1e-4, {path:?} on {}: closest to its bound is the error {error:.2e} against {bound:.2e} (weights {weights:.2e}, activations {activations:.2e}, sums {sums:.2e})",
                    device.info().name
                );
                if flush {
                    eprintln!(
                        "  {emulated} of {} outputs are the bits of f16 sums rounded to nearest after every 8 products",
                        n_rows * n_out
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
        for (ty, _) in all_types() {
            for path in paths_of(&kernels, ty) {
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

/// The path measured per type, on every card, capped or not: the most exact of the paths
/// within 5 % of the fastest, kept for the type and taken by `prefill_path_for`. Before the
/// measurement the type takes the kept rule. The times are printed.
#[test]
fn the_measured_path_is_the_most_exact_of_the_fastest() {
    let Some(gpus) = gpus() else { return };
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        if !kernels.has_prefill_kernels() {
            continue;
        }
        let full = kernels.tensor_cores_full_rate();
        for ty in [GgmlType::Q4_K, GgmlType::Q8_0, GgmlType::IQ3_S, GgmlType::Q4_0, GgmlType::F16] {
            let kept = kernels.prefill_path_for(ty, Precision::Fast).unwrap();
            let choice = kernels.measure_prefill_path(ty).unwrap();
            eprintln!("{}: {ty} {choice:?}", device.info().name);
            assert_eq!(choice.ty, ty);
            assert!(choice.measured, "{ty}");
            assert_eq!(kernels.prefill_path_for(ty, Precision::Fast).unwrap(), choice.path);
            assert_eq!(kernels.prefill_path_for(ty, Precision::Exact).unwrap(), PrefillPath::F16);
            assert_eq!(kernels.measure_prefill_path(ty).unwrap(), choice, "measured once");
            assert_eq!(kernels.prefill_choice(ty).unwrap(), choice);
            if full {
                assert_eq!(kept, PrefillPath::F16, "{ty}: unmeasured, the exact path");
            }
            let ms = |path: PrefillPath| match path {
                PrefillPath::Int8 => choice.probe_ms[0],
                PrefillPath::F16 => choice.probe_ms[1],
                PrefillPath::F16Flush => choice.probe_ms[2],
            };
            for path in PATHS {
                assert_eq!(
                    ms(path).is_finite(),
                    kernels.has_prefill_path(ty, path),
                    "{ty} {path:?}"
                );
            }
            let fastest = choice.probe_ms.iter().copied().fold(f32::INFINITY, f32::min);
            assert!(ms(choice.path) <= fastest * 1.05, "{ty} {choice:?}");
            // Every path more exact than the chosen one is slower by more than 5 %.
            let order = [PrefillPath::F16, PrefillPath::F16Flush, PrefillPath::Int8];
            for path in order.iter().take_while(|p| **p != choice.path) {
                assert!(ms(*path) > fastest * 1.05, "{ty} {path:?} {choice:?}");
            }
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

    for ty in [GgmlType::Q8_1, GgmlType::Q8_K, GgmlType::F64, GgmlType::I32] {
        // SAFETY: refused before anything is launched.
        let error = unsafe { kernels.matmul(&stream, ty, int8, p, p, p, 256, 1, 1) }
            .unwrap_err()
            .to_string();
        assert!(error.contains(ty.name()) && error.contains("no prefill kernel"), "{error}");
        assert!(error.contains("q4_K") && error.contains("iq4_xs"), "{error}");
        assert!(error.contains("f32") && error.contains("bf16"), "{error}");
    }
    for ty in ares_core::ggml_type::ALL_TYPES.into_iter().filter(|t| t.is_weight()) {
        // Every weight type has both families, and the rule says where one hands over to the other.
        assert!(Kernels::has_fast_matvec(ty), "{ty}");
        assert!(all_types().any(|(t, _)| t == ty), "{ty} has no tile kernel");
        if let Some(rows) = Kernels::decode_kernel_rows(ty) {
            assert!((1..=32).contains(&rows), "{ty}: {rows}");
        }
    }

    // SAFETY: every call below is refused before anything is launched.
    unsafe {
        // Not a whole number of blocks.
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 300, 1, 1).is_err());
        assert!(kernels.matmul(&stream, GgmlType::Q8_0, int8, p, p, p, 48, 1, 1).is_err());
        assert!(kernels.matmul(&stream, GgmlType::Q4_K, int8, p, p, p, 0, 1, 1).is_err());
        // The float types have blocks of one value; the kernels still take 32 at a time, and
        // 16 bytes of alignment, and there is no int8 form of them.
        for ty in FLOAT_TYPES {
            let f16 = PrefillPath::F16;
            assert!(kernels.matmul(&stream, ty, f16, p, p, p, 40, 1, 1).is_err(), "{ty}");
            assert!(kernels.matmul(&stream, ty, f16, p.offset(8), p, p, 64, 1, 1).is_err(), "{ty}");
            let error = kernels.matmul(&stream, ty, int8, p, p, p, 64, 1, 1).unwrap_err();
            assert!(error.to_string().contains("Int8"), "{ty}: {error}");
        }
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
