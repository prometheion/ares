//! Tests of the quantized weight kernels (`Kernels::matvec`, `quantize_activations`).
//!
//! This file repeats the CPU emulation and the checks of `crates/cuda/tests/mmvq.rs`
//! (the helpers `Rng`, `random_blocks`, `quantize_like_the_gpu`, `Deviation` and
//! `check_matvec`, and the shapes of the tests), so that the two backends are held to
//! one standard. It is a copy for now: once the CUDA side is quiet it moves to a
//! shared place and both test files use it.
//!
//! They open the GPUs, so they run only with `ARES_TEST_VULKAN=1`, like
//! `tests/gpu.rs`:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test mmvq -- --test-threads=1
//! ```
//!
//! `ARES_TEST_GGUF=<model.gguf>` adds the check on real tensors: the
//! smallest matrix of every type with a fast kernel that the model contains.
//!
//! What the tests prove. The kernels multiply a quantized weight matrix with
//! activations that were quantized to int8, so there are two claims, checked
//! separately:
//!
//! - *The kernel computes what it says* (tight): its result equals the exact
//!   weights (`ares_quant::dequantize`) times the dequantized activations,
//!   which the test builds on the CPU with the same arithmetic as the
//!   quantizing kernel, up to f32 rounding of the sums.
//! - *Quantizing the activations costs little* (loose): the result is close to
//!   the exact product of the weights with the unquantized activations. The
//!   observed deviation is printed (`--nocapture`) so that a change that makes
//!   it worse can be noticed; the assertion only guards against a gross error.

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_vulkan::{
    Device, DeviceBuffer, DevicePtr, Kernels, MAX_BATCH, PrefillPath, Stream, device_count,
};

/// The types with a kernel, and the byte offsets of the f16 fields of a block that must
/// be finite for the test data to be meaningful (everything else in a block may be any
/// bytes). The first eleven are the types of the reference model, whose kernels have the
/// CUDA counterparts; the rest are the other types a GGUF file can carry. Types with a
/// scale that is not a plain f16 field (IQ1_M, MXFP4, NVFP4, the float types) are
/// handled in `random_blocks`.
const TYPES: [(GgmlType, &[usize]); 28] = [
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
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::IQ2_XXS, &[0]),
    (GgmlType::IQ1_S, &[0]),
    (GgmlType::IQ1_M, &[]),
    (GgmlType::TQ1_0, &[52]),
    (GgmlType::TQ2_0, &[64]),
    (GgmlType::MXFP4, &[]),
    (GgmlType::NVFP4, &[]),
    (GgmlType::Q1_0, &[0]),
    (GgmlType::Q2_0, &[0]),
    (GgmlType::F32, &[]),
    (GgmlType::F16, &[]),
    (GgmlType::BF16, &[]),
];

/// Whether the kernel of `ty` reads f32 activations as they are.
fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// How many of `random_blocks`'s blocks a matrix of `n_out` rows of `n_in` values takes:
/// whole blocks, or one per value for the float types.
fn blocks_of(ty: GgmlType, n_in: usize, n_out: usize) -> usize {
    if is_float(ty) { n_in * n_out } else { n_in / ty.block_values() as usize * n_out }
}

/// Values per unit of a row of `ty` in the tests: a block, or 8 for the float types.
fn unit(ty: GgmlType) -> usize {
    if is_float(ty) { 8 } else { ty.block_values() as usize }
}

struct Gpu {
    device: Device,
    kernels: Kernels,
}

/// The devices every test shares, opened once and never closed: the process leaves through
/// [`leave`].
static GPUS: OnceLock<Option<&'static [Gpu]>> = OnceLock::new();

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

fn gpus() -> Option<&'static [Gpu]> {
    *GPUS.get_or_init(|| {
        if std::env::var_os("ARES_TEST_VULKAN").is_none() {
            eprintln!("skipped: set ARES_TEST_VULKAN=1 to run the tests that open GPUs");
            return None;
        }
        let count = device_count().expect("the Vulkan loader");
        assert!(count > 0, "ARES_TEST_VULKAN is set but the loader reports no device");
        let opened: Vec<Gpu> = (0..count)
            .map(|i| {
                let device = Device::open(i).expect("open device");
                let kernels = Kernels::load(&device).expect("load kernels");
                assert!(kernels.has_quantized_kernels(), "{:?}", device.info());
                Gpu { device, kernels }
            })
            .collect();
        leave_at_exit();
        Some(Box::leak(opened.into_boxed_slice()) as &'static [Gpu])
    })
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
/// signs (any other bytes are valid in every one of these types), and for the types
/// whose scale is built another way, a scale in a moderate range.
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    if is_float(ty) {
        // n_blocks values.
        return match ty {
            GgmlType::F32 => (0..n_blocks).flat_map(|_| rng.unit().to_le_bytes()).collect(),
            GgmlType::F16 => (0..n_blocks)
                .flat_map(|_| ares_quant::f32_to_f16(rng.unit()).to_le_bytes())
                .collect(),
            _ => (0..n_blocks)
                .flat_map(|_| ares_quant::f32_to_bf16(rng.unit()).to_le_bytes())
                .collect(),
        };
    }
    let (_, scales) = TYPES.iter().find(|(t, _)| *t == ty).unwrap();
    let block = ty.block_bytes() as usize;
    let mut bytes: Vec<u8> = (0..n_blocks * block).map(|_| (rng.next() >> 32) as u8).collect();
    for b in 0..n_blocks {
        let blk = &mut bytes[b * block..(b + 1) * block];
        for &offset in *scales {
            let scale = (0.002 + 0.02 * (rng.unit() * 0.5 + 0.5))
                * if rng.next().is_multiple_of(4) { -1.0 } else { 1.0 };
            blk[offset..offset + 2].copy_from_slice(&ares_quant::f32_to_f16(scale).to_le_bytes());
        }
        match ty {
            // The f16 d is spread over the top nibbles of the four 16-bit scale words.
            GgmlType::IQ1_M => {
                let d = ares_quant::f32_to_f16(0.002 + 0.02 * (rng.unit() * 0.5 + 0.5));
                for i in 0..4 {
                    let at = 49 + 2 * i;
                    blk[at] = (blk[at] & 0x0F) | ((((d >> (4 * i)) & 0xF) as u8) << 4);
                }
            }
            // An E8M0 exponent near 1.
            GgmlType::MXFP4 => blk[0] = 120 + (rng.next() >> 40) as u8 % 10,
            // UE4M3 scales: 4 exponent bits (bias 7) and 3 mantissa bits, around 2^-3.
            GgmlType::NVFP4 => {
                for scale in &mut blk[..4] {
                    *scale =
                        ((3 + (rng.next() >> 40) as u8 % 5) << 3) | ((rng.next() >> 33) as u8 & 7);
                }
            }
            _ => {}
        }
    }
    bytes
}

/// The quantization of activations as `ares_quantize_act` does it, on the CPU:
/// per group of 32 values `d = max|x| / 127`, `q = round(x / d)` (halves away
/// from zero) and `s = d * sum(q)`. Returns the codes and the `(d, s)` pairs.
fn quantize_like_the_gpu(x: &[f32]) -> (Vec<i8>, Vec<(f32, f32)>) {
    let mut codes = Vec::with_capacity(x.len());
    let mut pairs = Vec::with_capacity(x.len() / 32);
    for group in x.chunks(32) {
        let amax = group.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let mut sum = 0f32;
        for v in group {
            let code = if amax == 0.0 { 0.0 } else { (v / d).round() };
            codes.push(code as i8);
            sum += code;
        }
        pairs.push((d, d * sum));
    }
    (codes, pairs)
}

/// How far a kernel result is from the two references.
#[derive(Debug)]
struct Deviation {
    /// Largest `|have - emulation| / sum of |w x^|` over the outputs: rounding
    /// of the f32 sums only.
    tight: f64,
    /// `rms(have - exact) / rms(exact)` and `max|have - exact| / max|exact|`,
    /// with `exact` the product of the weights with the unquantized
    /// activations: the cost of quantizing the activations.
    loose_rms: f64,
    loose_max: f64,
    /// `rms(have - exact)` over the typical size of a sum of this kind,
    /// `sqrt(sum of (w x)^2)` per output. The two above are relative to the
    /// outputs, which for a handful of random sums can be small by chance; this
    /// one is stable and is what the tests assert on.
    loose_scaled: f64,
}

impl Deviation {
    /// The numbers in one line, for the log of a run.
    fn describe(&self) -> String {
        format!(
            "vs the emulation {:.1e} of the magnitude; vs the f32 product: rms {:.3}%, max {:.3}%, {:.3}% of a sum's size",
            self.tight,
            self.loose_rms * 100.0,
            self.loose_max * 100.0,
            self.loose_scaled * 100.0
        )
    }
}

/// Runs `Kernels::matvec` on `w` (n_out rows of n_in values of `ty`) with the
/// activations `x` (n_batch rows) and compares it with both references.
#[allow(clippy::too_many_arguments)]
fn check_matvec(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_batch: usize,
) -> Deviation {
    assert_eq!(x.len(), n_in * n_batch);
    let stream = Stream::new(device).unwrap();
    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
    dw.write(0, w).unwrap();
    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
    dx.write(0, &f32_bytes(x)).unwrap();
    let dq =
        DeviceBuffer::new(device, Kernels::quantized_activation_bytes(n_in as u32, n_batch as u32))
            .unwrap();
    let dy = DeviceBuffer::new(device, (n_batch * n_out * 4) as u64).unwrap();
    // SAFETY: the buffers were sized for these shapes just above.
    unsafe {
        // The float types read the f32 activations as they are.
        let act = if is_float(ty) {
            dx.ptr()
        } else {
            kernels
                .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_batch as u32)
                .unwrap();
            dq.ptr()
        };
        kernels
            .matvec(&stream, ty, dw.ptr(), act, dy.ptr(), n_in as u32, n_out as u32, n_batch as u32)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let have = read_f32(&dy, n_batch * n_out);

    deviation(ty, w, x, n_in, n_out, n_batch, &have)
}

/// How far `have` (n_batch rows of n_out outputs) is from the two references.
fn deviation(
    ty: GgmlType,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_batch: usize,
    have: &[f32],
) -> Deviation {
    let mut weights = vec![0f32; n_in * n_out];
    ares_quant::dequantize(ty, w, &mut weights).unwrap();
    let mut worst_tight = 0f64;
    let (mut err2, mut ref2, mut err_max, mut ref_max, mut scale2) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for b in 0..n_batch {
        let xb = &x[b * n_in..][..n_in];
        let (codes, pairs) = quantize_like_the_gpu(xb);
        for r in 0..n_out {
            let row = &weights[r * n_in..][..n_in];
            let (mut emulated, mut magnitude, mut exact, mut squares) = (0f64, 0f64, 0f64, 0f64);
            for (k, weight) in row.iter().enumerate() {
                let w = f64::from(*weight);
                // The float kernels multiply by the activations themselves.
                let xq = if is_float(ty) {
                    f64::from(xb[k])
                } else {
                    f64::from(pairs[k / 32].0) * f64::from(codes[k])
                };
                emulated += w * xq;
                magnitude += (w * xq).abs();
                exact += w * f64::from(xb[k]);
                squares += (w * f64::from(xb[k])).powi(2);
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
    let n = (n_batch * n_out) as f64;
    Deviation {
        tight: worst_tight,
        loose_rms: (err2 / n).sqrt() / (ref2 / n).sqrt().max(1e-30),
        loose_max: err_max / ref_max.max(1e-30),
        loose_scaled: (err2 / n).sqrt() / (scale2 / n).sqrt().max(1e-30),
    }
}

/// The values of the quantizing kernel are exactly the CPU's: the int8 codes,
/// the scales and the sums, for rows with all-zero groups, one large outlier
/// per group, negative values, and sizes that do not fill a thread block.
#[test]
fn activation_quantization_matches_the_cpu_bit_for_bit() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        for (n_in, n_batch) in [(32usize, 1usize), (256, 3), (5120, 8), (96, 5)] {
            let mut x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.activation()).collect();
            // A group of zeros, a group with a single value, a group of equal values.
            x[..32].fill(0.0);
            if x.len() >= 96 {
                x[32..64].fill(0.0);
                x[40] = -3.5;
                x[64..96].fill(0.25);
            }
            let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
            dx.write(0, &f32_bytes(&x)).unwrap();
            let bytes = Kernels::quantized_activation_bytes(n_in as u32, n_batch as u32);
            let dq = DeviceBuffer::new(device, bytes).unwrap();
            // SAFETY: the buffers were sized for these shapes just above.
            unsafe {
                kernels
                    .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_batch as u32)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let mut got = vec![0u8; bytes as usize];
            dq.read(0, &mut got).unwrap();

            let n_values = n_in * n_batch;
            let (codes, pairs) = quantize_like_the_gpu(&x);
            let got_codes: Vec<i8> = got[..n_values].iter().map(|b| *b as i8).collect();
            assert_eq!(got_codes, codes, "codes, {n_batch} x {n_in}");
            let ds_at = n_values.next_multiple_of(16);
            for (g, (d, s)) in pairs.iter().enumerate() {
                let at = ds_at + 8 * g;
                let gd = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
                let gs = f32::from_le_bytes(got[at + 4..at + 8].try_into().unwrap());
                assert_eq!((gd.to_bits(), gs.to_bits()), (d.to_bits(), s.to_bits()), "group {g}");
            }
        }
    }
}

/// Every fast kernel, for shapes around its edges: the smallest matrix (one
/// block), one row, row counts that do not fill a thread block of 8 rows, a
/// row longer than a warp's iteration, many thread blocks, and every batch
/// size. Random weights (valid in every type) and activations with outliers.
#[test]
fn fast_matvec_computes_the_dequantized_product_on_every_type() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for Gpu { device, kernels } in gpus {
        for (ty, _) in TYPES {
            let bv = unit(ty);
            let shapes = [
                (bv, 1, 1),
                (bv, 3, 8),
                (bv * 5, 13, 4),
                (bv * 20, 33, 1),
                (bv * 20, 9, 7),
                (if bv == 256 { 17408 } else { 4096 }, 17, 2),
                // Many thread blocks.
                (bv * 4, 3000, 3),
            ];
            for (n_in, n_out, n_batch) in shapes {
                let w = random_blocks(&mut rng, ty, blocks_of(ty, n_in, n_out));
                let x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.activation()).collect();
                let dev = check_matvec(device, kernels, ty, &w, &x, n_in, n_out, n_batch);
                eprintln!(
                    "{ty} {n_out}x{n_in} rows {n_batch} on {}: {}",
                    device.info().name,
                    dev.describe()
                );
                assert!(
                    dev.tight < 5e-5,
                    "{ty} {n_out}x{n_in} x{n_batch}: differs from the emulation: {}",
                    dev.describe()
                );
                // A handful of outputs say little about the error of quantizing the
                // activations (it is a statistic over many sums), so only shapes with
                // enough of them are held to the loose bound.
                if n_out * n_batch >= 16 {
                    assert!(
                        dev.loose_scaled < 0.03,
                        "{ty} {n_out}x{n_in} x{n_batch}: far from the f32 product: {}",
                        dev.describe()
                    );
                }
            }
        }
    }
}

/// The same on real tensors: the smallest 2-D matrix of each fast type in the
/// model, its first 512 rows, with activations of a normal distribution.
#[test]
fn fast_matvec_on_real_tensors() {
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
            // Expert tensors have a third dimension; the first rows of the first expert
            // are a matrix like any other.
            .filter(|t| {
                matches!(t.dims.as_slice().len(), 2 | 3)
                    && t.ty == ty
                    && (t.dims.get(0) as usize).is_multiple_of(unit(ty))
            })
            .min_by_key(|t| t.size)
        else {
            continue;
        };
        let (n_in, n_out) = (tensor.dims.get(0) as usize, tensor.dims.get(1) as usize);
        let rows = n_out.min(512);
        let row_bytes = ty.bytes_for(n_in as u64).unwrap() as usize;
        let w = &gguf.data(tensor)[..rows * row_bytes];
        for n_batch in [1usize, 3, 8] {
            let x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.activation()).collect();
            for Gpu { device, kernels } in gpus {
                let dev = check_matvec(device, kernels, ty, w, &x, n_in, rows, n_batch);
                eprintln!(
                    "{} ({ty}, {rows} x {n_in}) rows {n_batch} on {}: {}",
                    tensor.name,
                    device.info().name,
                    dev.describe()
                );
                assert!(
                    dev.tight < 5e-5,
                    "{}: differs from the emulation: {}",
                    tensor.name,
                    dev.describe()
                );
                assert!(
                    dev.loose_scaled < 0.03,
                    "{}: far from the f32 product: {}",
                    tensor.name,
                    dev.describe()
                );
            }
        }
        checked.push(ty);
    }
    assert!(!checked.is_empty(), "the model has no 2-D tensor of a type with a fast kernel");
    eprintln!("checked types: {}", checked.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "));
}

/// The kernels read every weight once: the result does not depend on how many
/// rows are in the batch, so row `b` of a batch of 8 equals the same row alone.
#[test]
fn a_row_gives_the_same_result_alone_and_in_a_batch() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xda942042e4dd58b5);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        for (ty, _) in TYPES {
            let bv = unit(ty);
            for n_out in [19usize, 1500] {
                let n_in = bv * 8;
                let w = random_blocks(&mut rng, ty, blocks_of(ty, n_in, n_out));
                let x: Vec<f32> = (0..n_in * 8).map(|_| rng.activation()).collect();
                let run = |x: &[f32], n_batch: usize| -> Vec<f32> {
                    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
                    dw.write(0, &w).unwrap();
                    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
                    dx.write(0, &f32_bytes(x)).unwrap();
                    let dq = DeviceBuffer::new(
                        device,
                        Kernels::quantized_activation_bytes(n_in as u32, n_batch as u32),
                    )
                    .unwrap();
                    let dy = DeviceBuffer::new(device, (n_batch * n_out * 4) as u64).unwrap();
                    // SAFETY: the buffers were sized for these shapes just above.
                    unsafe {
                        let act = if is_float(ty) {
                            dx.ptr()
                        } else {
                            kernels
                                .quantize_activations(
                                    &stream,
                                    dx.ptr(),
                                    dq.ptr(),
                                    n_in as u32,
                                    n_batch as u32,
                                )
                                .unwrap();
                            dq.ptr()
                        };
                        kernels
                            .matvec(
                                &stream,
                                ty,
                                dw.ptr(),
                                act,
                                dy.ptr(),
                                n_in as u32,
                                n_out as u32,
                                n_batch as u32,
                            )
                            .unwrap();
                    }
                    stream.synchronize().unwrap();
                    read_f32(&dy, n_batch * n_out)
                };
                let batch = run(&x, 8);
                for b in [0usize, 3, 7] {
                    let alone = run(&x[b * n_in..][..n_in], 1);
                    assert_eq!(
                        &batch[b * n_out..][..n_out],
                        &alone[..],
                        "{ty}, {n_out} rows, batch row {b}"
                    );
                }
            }
        }
    }
}

/// Refusals happen before anything is launched: a type without a fast kernel
/// names the types that have one, and the sizes and alignments the kernels
/// rely on are checked.
#[test]
fn bad_arguments_are_refused() {
    let Some(gpus) = gpus() else { return };
    let Gpu { device, kernels } = &gpus[0];
    let stream = Stream::new(device).unwrap();
    let buffer = DeviceBuffer::new(device, 1 << 20).unwrap();
    let p = buffer.ptr();
    let null = DevicePtr::NULL;

    for ty in [GgmlType::F64, GgmlType::Q8_K, GgmlType::Q8_1, GgmlType::I32, GgmlType::I8] {
        assert!(!Kernels::has_fast_matvec(ty));
        // SAFETY: refused before anything is launched.
        let error =
            unsafe { kernels.matvec(&stream, ty, p, p, p, 256, 1, 1) }.unwrap_err().to_string();
        assert!(error.contains(ty.name()) && error.contains("no fast weight kernel"), "{error}");
        assert!(error.contains("q4_K") && error.contains("iq4_xs"), "{error}");
    }
    for (ty, _) in TYPES {
        assert!(Kernels::has_fast_matvec(ty));
    }

    // SAFETY: every call below is refused before anything is launched.
    unsafe {
        // Not a whole number of blocks.
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p, p, p, 300, 1, 1).is_err());
        assert!(kernels.matvec(&stream, GgmlType::Q8_0, p, p, p, 48, 1, 1).is_err());
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p, p, p, 0, 1, 1).is_err());
        // Batch sizes.
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p, p, p, 256, 1, 0).is_err());
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p, p, p, 256, 1, MAX_BATCH + 1).is_err());
        assert!(kernels.quantize_activations(&stream, p, p, 256, 0).is_err());
        assert!(kernels.quantize_activations(&stream, p, p, 256, MAX_BATCH + 1).is_err());
        assert!(kernels.quantize_activations(&stream, p, p, 48, 1).is_err());
        // Alignment: Q4_K needs 16 bytes of the weights, IQ4_XS 8, Q8_0 2; the activations 16.
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p.offset(8), p, p, 256, 1, 1).is_err());
        assert!(kernels.matvec(&stream, GgmlType::IQ4_XS, p.offset(4), p, p, 256, 1, 1).is_err());
        assert!(kernels.matvec(&stream, GgmlType::Q8_0, p.offset(1), p, p, 32, 1, 1).is_err());
        assert!(kernels.matvec(&stream, GgmlType::Q4_K, p, p.offset(8), p, 256, 1, 1).is_err());
        assert!(kernels.quantize_activations(&stream, p, p.offset(4), 256, 1).is_err());
        // No rows is nothing to do, not an error, and a null address is not touched.
        kernels.matvec(&stream, GgmlType::Q4_K, null, null, null, 256, 0, 1).unwrap();
    }
}

/// The quantizer against the CPU on a million random values of every magnitude: the
/// scales `d = max / 127` and the codes `round(x / d)` depend on a division that is
/// exact on the CPU, and GLSL's `/` is not required to be.
#[test]
fn activation_quantization_matches_the_cpu_on_random_values() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xfeed_5eed_0dd5_a11e);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        let (n_in, n_batch) = (4096usize, 8usize);
        for round in 0..32 {
            // Magnitudes over many octaves, both signs, and groups with a single outlier.
            let x: Vec<f32> = (0..n_in * n_batch)
                .map(|i| {
                    let octave = ((rng.next() >> 20) % 24) as i32 - 12;
                    let v = rng.unit() * 2f32.powi(octave);
                    if i % 97 == 0 { v * 40.0 } else { v }
                })
                .collect();
            let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
            dx.write(0, &f32_bytes(&x)).unwrap();
            let bytes = Kernels::quantized_activation_bytes(n_in as u32, n_batch as u32);
            let dq = DeviceBuffer::new(device, bytes).unwrap();
            // SAFETY: the buffers were sized for these shapes just above.
            unsafe {
                kernels
                    .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_batch as u32)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            let mut got = vec![0u8; bytes as usize];
            dq.read(0, &mut got).unwrap();
            let (codes, pairs) = quantize_like_the_gpu(&x);
            let wrong_codes: Vec<usize> =
                (0..x.len()).filter(|&i| got[i] as i8 != codes[i]).collect();
            let ds_at = x.len().next_multiple_of(16);
            let mut wrong_ds = Vec::new();
            for (g, (d, s)) in pairs.iter().enumerate() {
                let at = ds_at + 8 * g;
                let gd = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
                let gs = f32::from_le_bytes(got[at + 4..at + 8].try_into().unwrap());
                if (gd.to_bits(), gs.to_bits()) != (d.to_bits(), s.to_bits()) {
                    wrong_ds.push((g, gd, *d, gs, *s));
                }
            }
            assert!(
                wrong_codes.is_empty() && wrong_ds.is_empty(),
                "round {round} on {}: {} codes and {} (d, s) pairs differ; first pairs {:?}",
                device.info().name,
                wrong_codes.len(),
                wrong_ds.len(),
                &wrong_ds[..wrong_ds.len().min(4)]
            );
        }
    }
}

/// The tensor-core kernel against the same two references as the decode kernel: it must
/// compute what the decode kernel computes, to f32 rounding, for any number of rows, with
/// partial tiles on both sides (64 weight rows and 64 activation rows per workgroup).
#[test]
fn matmul_q4k_computes_the_dequantized_product() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x7a11_f00d_0bad_cafe);
    for Gpu { device, kernels } in gpus {
        if !kernels.has_matmul(GgmlType::Q4_K, PrefillPath::Int8) {
            eprintln!("skipped on {}: no int8 cooperative matrices", device.info().name);
            continue;
        }
        let stream = Stream::new(device).unwrap();
        for (n_in, n_out, n_rows) in [
            (256usize, 1usize, 1usize),
            (256, 64, 64),
            (512, 70, 65),
            (1280, 130, 3),
            (5120, 129, 200),
            (2560, 64, 1000),
            (17408, 77, 70),
        ] {
            let w = random_blocks(&mut rng, GgmlType::Q4_K, blocks_of(GgmlType::Q4_K, n_in, n_out));
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
            let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
            // SAFETY: the buffers were sized for these shapes just above.
            unsafe {
                kernels
                    .quantize_activation_rows(
                        &stream,
                        dx.ptr(),
                        dq.ptr(),
                        n_in as u32,
                        n_rows as u32,
                    )
                    .unwrap();
                kernels
                    .matmul(
                        &stream,
                        GgmlType::Q4_K,
                        PrefillPath::Int8,
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
            let have = read_f32(&dy, n_rows * n_out);
            let dev = deviation(GgmlType::Q4_K, &w, &x, n_in, n_out, n_rows, &have);
            eprintln!(
                "q4_K matmul {n_out}x{n_in} rows {n_rows} on {}: {}",
                device.info().name,
                dev.describe()
            );
            assert!(dev.tight < 5e-5, "{n_out}x{n_in} x{n_rows}: {}", dev.describe());
            if n_out * n_rows >= 16 {
                assert!(dev.loose_scaled < 0.03, "{n_out}x{n_in} x{n_rows}: {}", dev.describe());
            }
        }
    }
}

/// The exact product of the dequantized weights with the unquantized activations in f64,
/// and the typical size of a sum of this kind (`sqrt(sum of (w x)^2)`) per output.
fn exact_product(
    ty: GgmlType,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut weights = vec![0f32; n_in * n_out];
    ares_quant::dequantize(ty, w, &mut weights).unwrap();
    let mut exact = vec![0f64; n_rows * n_out];
    let mut size = vec![0f64; n_rows * n_out];
    for b in 0..n_rows {
        let xb = &x[b * n_in..][..n_in];
        for r in 0..n_out {
            let (mut sum, mut squares) = (0f64, 0f64);
            for (wk, xk) in weights[r * n_in..][..n_in].iter().zip(xb) {
                let p = f64::from(*wk) * f64::from(*xk);
                sum += p;
                squares += p * p;
            }
            exact[b * n_out + r] = sum;
            size[b * n_out + r] = squares.sqrt();
        }
    }
    (exact, size)
}

/// `rms(have - exact)` over the rms size of a sum, and `rms(have - exact) / rms(exact)`.
fn relative_error(have: &[f32], exact: &[f64], size: &[f64]) -> (f64, f64) {
    let n = have.len() as f64;
    let err: f64 = have.iter().zip(exact).map(|(h, e)| (f64::from(*h) - e).powi(2)).sum();
    let reference: f64 = exact.iter().map(|e| e * e).sum();
    let scale: f64 = size.iter().map(|s| s * s).sum();
    (
        (err / n).sqrt() / (scale / n).sqrt().max(1e-30),
        (err / n).sqrt() / (reference / n).sqrt().max(1e-30),
    )
}

/// The f16 prefill kernels on the file's layout, for every type that has one: the error
/// against the exact f32 product must be the rounding of weights and activations to f16
/// (and of partial sums, with f16 accumulation) and nothing more, for shapes with partial
/// tiles on both sides.
#[test]
fn f16_prefill_on_the_file_layout_is_close_to_the_exact_product() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x51de_cafe_f00d_0002);
    let mut failures = Vec::new();
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        for (ty, _) in TYPES {
            if !kernels.has_matmul(ty, PrefillPath::F16) {
                continue;
            }
            for (n_in, n_out, n_rows) in
                [(256usize, 1usize, 1usize), (256, 64, 64), (1280, 70, 65), (2560, 129, 130)]
            {
                let w = random_blocks(&mut rng, ty, blocks_of(ty, n_in, n_out));
                let x: Vec<f32> = (0..n_in * n_rows)
                    .map(|i| rng.activation() * 4f32.powi(((i / n_in) % 7) as i32 - 3))
                    .collect();
                let (exact, size) = exact_product(ty, &w, &x, n_in, n_out, n_rows);
                let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
                dw.write(0, &w).unwrap();
                let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
                dx.write(0, &f32_bytes(&x)).unwrap();
                for path in [PrefillPath::F16, PrefillPath::F16Flush] {
                    let act = DeviceBuffer::new(
                        device,
                        Kernels::prefill_activation_bytes(path, n_in as u32, n_rows as u32),
                    )
                    .unwrap();
                    let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
                    // SAFETY: the buffers are sized for these shapes just above.
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
                    let have = read_f32(&dy, n_rows * n_out);
                    let (scaled, rel) = relative_error(&have, &exact, &size);
                    eprintln!(
                        "{ty} {path:?} {n_out}x{n_in} rows {n_rows} on {}: {:.4}% of a sum's size, {:.4}% of the rms",
                        device.info().name,
                        scaled * 100.0,
                        rel * 100.0
                    );
                    if scaled.is_nan() || scaled >= 1.5e-3 {
                        failures.push(format!("{ty} {path:?} {n_out}x{n_in} x{n_rows}: {scaled}"));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// What a device without cooperative matrices runs for a prompt: the decode kernels in
/// batches of 8 rows. Any number of rows gives what the decode kernel gives for each.
#[test]
fn matmul_decode_runs_any_number_of_rows_through_the_decode_kernels() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xdec0_dede_0000_0004);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        for ty in [GgmlType::Q4_K, GgmlType::Q8_0, GgmlType::IQ2_XS, GgmlType::F16] {
            for n_rows in [1usize, 8, 9, 19, 64] {
                let (n_in, n_out) = (unit(ty) * 20, 70usize);
                let w = random_blocks(&mut rng, ty, blocks_of(ty, n_in, n_out));
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
                dw.write(0, &w).unwrap();
                let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
                dx.write(0, &f32_bytes(&x)).unwrap();
                let scratch = DeviceBuffer::new(
                    device,
                    Kernels::quantized_activation_bytes(n_in as u32, MAX_BATCH),
                )
                .unwrap();
                let dy = DeviceBuffer::new(device, (n_rows * n_out * 4) as u64).unwrap();
                // SAFETY: the buffers are sized for these shapes just above.
                unsafe {
                    kernels
                        .matmul_decode(
                            &stream,
                            ty,
                            dw.ptr(),
                            dx.ptr(),
                            scratch.ptr(),
                            dy.ptr(),
                            n_in as u32,
                            n_out as u32,
                            n_rows as u32,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
                let have = read_f32(&dy, n_rows * n_out);
                let dev = deviation(ty, &w, &x, n_in, n_out, n_rows, &have);
                assert!(dev.tight < 5e-5, "{ty} x{n_rows}: {}", dev.describe());
            }
        }
    }
}
