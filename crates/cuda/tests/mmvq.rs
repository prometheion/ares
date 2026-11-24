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

use ares_core::GgmlType;
use ares_cuda::{Device, DeviceBuffer, DevicePtr, Kernels, MAX_BATCH, Stream};
use ares_quant::act;
use common::blocks::{TYPES, is_float, random_blocks};
use common::{Rng, f32_bytes, gpus, read_f32};

mod common;

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
    // NaN in every output, so that one the kernel leaves unwritten fails the comparison.
    dy.write(0, &vec![0xFF; n_batch * n_out * 4]).unwrap();
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
        // The float types take any row length; their activations are not quantized.
        let groups: Vec<act::Group> =
            if is_float(ty) { Vec::new() } else { act::groups(xb).collect() };
        for r in 0..n_out {
            let row = &weights[r * n_in..][..n_in];
            let (mut emulated, mut magnitude, mut exact, mut squares) = (0f64, 0f64, 0f64, 0f64);
            for (k, weight) in row.iter().enumerate() {
                let w = f64::from(*weight);
                // The float types multiply the activations as they are.
                let xq = if is_float(ty) {
                    f64::from(xb[k])
                } else {
                    let group = &groups[k / act::GROUP];
                    f64::from(group.d) * f64::from(group.codes[k % act::GROUP])
                };
                emulated += w * xq;
                magnitude += (w * xq).abs();
                exact += w * f64::from(xb[k]);
                squares += (w * f64::from(xb[k])).powi(2);
            }
            let h = f64::from(have[b * n_out + r]);
            let off = (h - emulated).abs() / magnitude.max(1e-30);
            worst_tight = worst_tight.max(if off.is_nan() { f64::INFINITY } else { off });
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
            let groups: Vec<act::Group> = act::groups(&x).collect();
            let got_codes: Vec<i8> = got[..n_values].iter().map(|b| *b as i8).collect();
            let codes: Vec<i8> = groups.iter().flat_map(|g| g.codes).collect();
            assert_eq!(got_codes, codes, "codes, {n_batch} x {n_in}");
            let ds_at = act::Layout::of(n_values as u64).pairs_at as usize;
            for (g, group) in groups.iter().enumerate() {
                let at = ds_at + 8 * g;
                let gd = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
                let gs = f32::from_le_bytes(got[at + 4..at + 8].try_into().unwrap());
                let want = (group.d.to_bits(), group.s.to_bits());
                assert_eq!((gd.to_bits(), gs.to_bits()), want, "group {g}");
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
        for ty in TYPES {
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

/// A matrix with more tiles of rows than the launch of a type's kernel has thread blocks:
/// the kernels whose codebook is large walk several tiles per block, and the launch gives
/// them fewer blocks than there are tiles. Every row must still be written, for every type,
/// whichever of them walk (sixteen blocks of 8 rows per multiprocessor is more than any
/// launch has).
#[test]
fn every_row_of_a_matrix_taller_than_a_launch_is_written() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x6a09e667f3bcc909);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let n_out = 16 * 8 * device.info().multiprocessors as usize + 5;
        for ty in TYPES {
            let n_in = ty.block_values().max(32) as usize;
            let w = random_blocks(&mut rng, ty, n_in / ty.block_values() as usize * n_out);
            let x: Vec<f32> = (0..n_in * 2).map(|_| rng.activation()).collect();
            let dev = check_matvec(device, &kernels, ty, &w, &x, n_in, n_out, 2);
            assert!(
                dev.tight < 5e-5,
                "{ty} {n_out}x{n_in} on {}: differs from the emulation: {}",
                device.info().name,
                dev.describe()
            );
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
    for ty in TYPES {
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

/// The types whose decoder has an offset per block (`d * q + m`): the device may fuse the
/// multiply and the add where the CPU rounds twice, so a weight can differ from the CPU's in
/// its last bit. Every other type decodes exactly as `ares_quant::dequantize` does.
const OFFSET_TYPES: [GgmlType; 5] =
    [GgmlType::Q4_1, GgmlType::Q5_1, GgmlType::Q2_K, GgmlType::Q4_K, GgmlType::Q5_K];

/// The weight kernel for unquantized rows (`Kernels::matvec_exact`), on every type and the
/// shapes of the fast kernels: it reads the f32 section of the rows only (the codes and the
/// scales are poisoned here), its products are those of the decoded weights in f64 up to f32
/// rounding, and a quantized matrix gives the bits its decoded copy in F32 gives on the float
/// kernel, except for the last bit of a weight of a type with an offset.
#[test]
fn exact_matvec_multiplies_unquantized_rows_on_every_type() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x7c3f_19a2_e4d1_5b07);
    for device in gpus {
        let kernels = Kernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        for ty in TYPES {
            let bv = ty.block_values() as usize;
            let (mut worst, mut differ, mut outputs) = (0f64, 0usize, 0usize);
            for (n_in, n_out, n_batch) in shapes_for(ty) {
                if !is_float(ty) && n_in % bv != 0 {
                    continue;
                }
                let w = random_blocks(&mut rng, ty, n_in / bv * n_out);
                let x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.activation()).collect();
                let bytes = Kernels::quantized_activation_bytes(n_in as u32, n_batch as u32);
                let x_at = bytes - (x.len() * 4) as u64;
                let dq = DeviceBuffer::new(device, bytes).unwrap();
                dq.write(0, &vec![0x7Fu8; x_at as usize]).unwrap();
                dq.write(x_at, &f32_bytes(&x)).unwrap();
                let run = |ty: GgmlType, w: &[u8]| -> Vec<f32> {
                    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
                    dw.write(0, w).unwrap();
                    let dy = DeviceBuffer::new(device, (n_batch * n_out * 4) as u64).unwrap();
                    // SAFETY: the buffers were sized for these shapes just above.
                    unsafe {
                        kernels
                            .matvec_exact(
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
                let have = run(ty, &w);
                let mut weights = vec![0f32; n_in * n_out];
                ares_quant::dequantize(ty, &w, &mut weights).unwrap();
                for b in 0..n_batch {
                    let xb = &x[b * n_in..][..n_in];
                    for r in 0..n_out {
                        let row = &weights[r * n_in..][..n_in];
                        let (mut exact, mut magnitude) = (0f64, 0f64);
                        for (w, x) in row.iter().zip(xb) {
                            exact += f64::from(*w) * f64::from(*x);
                            magnitude += (f64::from(*w) * f64::from(*x)).abs();
                        }
                        let h = f64::from(have[b * n_out + r]);
                        worst = worst.max((h - exact).abs() / magnitude.max(1e-30));
                    }
                }
                if !is_float(ty) {
                    let copy = f32_bytes(&weights);
                    let want = run(GgmlType::F32, &copy);
                    differ +=
                        have.iter().zip(&want).filter(|(h, w)| h.to_bits() != w.to_bits()).count();
                    outputs += have.len();
                }
            }
            eprintln!(
                "{ty} on {}: {worst:.1e} of the magnitude from the f64 product; {differ} of {outputs} \
                 outputs differ from the decoded copy in F32",
                device.info().name
            );
            assert!(worst < 2e-6, "{ty}: {worst:.2e} from the f64 product");
            if !OFFSET_TYPES.contains(&ty) {
                assert_eq!(differ, 0, "{ty}: outputs differ from the decoded copy in F32");
            }
        }
    }
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
        for ty in TYPES {
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
    for ty in TYPES {
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
