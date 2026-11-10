//! Tests of the fast weight kernels (`Kernels::matvec`, `quantize_activations`).
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`, like
//! `tests/gpu.rs`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test mmvq -- --test-threads=1
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
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Kernels, MAX_BATCH, Stream, Wait, device_count};

/// The types with a fast kernel, and the byte offsets of the f16 fields of a
/// block that must be finite for the test data to be meaningful (everything
/// else in a block may be any bytes).
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
    // The types of the second round. The float types and the three whose scales are
    // not plain f16 fields (IQ1_M, MXFP4, NVFP4) are made by `random_blocks`.
    (GgmlType::F32, &[]),
    (GgmlType::F16, &[]),
    (GgmlType::BF16, &[]),
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::IQ1_S, &[0]),
    (GgmlType::IQ1_M, &[]),
    (GgmlType::IQ2_XXS, &[0]),
    (GgmlType::TQ1_0, &[52]),
    (GgmlType::TQ2_0, &[64]),
    (GgmlType::MXFP4, &[]),
    (GgmlType::NVFP4, &[]),
    (GgmlType::Q1_0, &[0]),
    (GgmlType::Q2_0, &[0]),
];

/// The types whose weights are floats: their kernels multiply the activations exactly.
fn is_float(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// The shapes `(n_in, n_out, n_batch)` a type is tested on: its smallest matrix (one
/// block), one row, row counts that do not fill a thread block of 8 rows, a row
/// longer than a warp's iteration, many thread blocks, every batch size. The float
/// types have no blocks; theirs include row lengths that are not a multiple of 8,
/// whose rows are not 16-byte aligned.
fn shapes_for(ty: GgmlType) -> Vec<(usize, usize, usize)> {
    if is_float(ty) {
        return vec![
            (1, 3, 1),
            (3, 5, 2),
            (8, 7, 3),
            (100, 9, 4),
            (1000, 33, 1),
            (4100, 17, 8),
            (5120, 20, 5),
            (1024, 3000, 3),
        ];
    }
    let bv = ty.block_values() as usize;
    vec![
        (bv, 1, 1),
        (bv, 3, 8),
        (bv * 5, 13, 4),
        (bv * 20, 33, 1),
        (bv * 20, 9, 7),
        (if bv == 256 { 17408 } else { 4096 }, 17, 2),
        // Many thread blocks.
        (bv * 4, 3000, 3),
    ]
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

/// `n_blocks` blocks of `ty` of random bytes, with finite f16 scales of both
/// signs (any other bytes are valid in every one of these types).
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
    if is_float(ty) {
        // n_blocks values (a block of a float type is one value), of both signs
        let mut bytes = Vec::with_capacity(n_blocks * ty.block_bytes() as usize);
        for _ in 0..n_blocks {
            let v = rng.unit() * 2.0;
            match ty {
                GgmlType::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
                GgmlType::F16 => bytes.extend_from_slice(&ares_quant::f32_to_f16(v).to_le_bytes()),
                _ => bytes.extend_from_slice(&ares_quant::f32_to_bf16(v).to_le_bytes()),
            }
        }
        return bytes;
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
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            GgmlType::IQ1_M => {
                let d = ares_quant::f32_to_f16(0.002 + 0.02 * (rng.unit() * 0.5 + 0.5));
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let word = u16::from_le_bytes([blk[at], blk[at + 1]]);
                    let word = (word & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12);
                    blk[at..at + 2].copy_from_slice(&word.to_le_bytes());
                }
            }
            // An E8M0 exponent around 1 (2^-9 to 2^10), not the huge ones.
            GgmlType::MXFP4 => blk[0] = 118 + (rng.next() % 20) as u8,
            // Four UE4M3 scales of exponent 3..10, never the NaN code.
            GgmlType::NVFP4 => {
                for byte in blk.iter_mut().take(4) {
                    *byte = ((3 + rng.next() % 8) << 3 | (rng.next() % 8)) as u8;
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
    let stream = Stream::new(device, 0).unwrap();
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
        kernels
            .quantize_activations(&stream, dx.ptr(), dq.ptr(), n_in as u32, n_batch as u32)
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
                n_batch as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let have = read_f32(&dy, n_batch * n_out);

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
                // The float types multiply the activations as they are.
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
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
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
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            let shapes = shapes_for(ty);
            for (n_in, n_out, n_batch) in shapes {
                let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
                let x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.activation()).collect();
                let dev = check_matvec(device, &kernels, ty, &w, &x, n_in, n_out, n_batch);
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
                    // (a float type has nothing to quantize: its bound is f32 rounding)
                    let bound = if is_float(ty) { 1e-5 } else { 0.03 };
                    assert!(
                        dev.loose_scaled < bound,
                        "{ty} {n_out}x{n_in} x{n_batch}: far from the f32 product: {}",
                        dev.describe()
                    );
                }
            }
        }
    }
}

/// The same on real tensors: the smallest 2-D matrix of each fast type in the
/// model (of a 3-D tensor of experts, the first expert), its first 512 rows, with activations of a normal distribution.
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
            .filter(|t| t.ty == ty && matches!(t.dims.as_slice().len(), 2 | 3))
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
            for device in gpus {
                let kernels = Kernels::load(device).unwrap();
                let dev = check_matvec(device, &kernels, ty, w, &x, n_in, rows, n_batch);
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
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        for (ty, _) in TYPES {
            let bv = ty.block_values() as usize;
            for n_out in [19usize, 1500] {
                let n_in = bv * 8;
                let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
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
                        kernels
                            .quantize_activations(
                                &stream,
                                dx.ptr(),
                                dq.ptr(),
                                n_in as u32,
                                n_batch as u32,
                            )
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
    let device = &gpus[0];
    let kernels = Kernels::load(device).unwrap();
    let stream = Stream::new(device, 0).unwrap();
    let buffer = DeviceBuffer::new(device, 1 << 20).unwrap();
    let p = buffer.ptr();
    let null = DevicePtr::NULL;

    for ty in [GgmlType::F64, GgmlType::I8, GgmlType::I32, GgmlType::Q8_1, GgmlType::Q8_K] {
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
        assert!(kernels.quantize_activations(&stream, p, p, 0, 1).is_err());
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
