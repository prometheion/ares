//! Tests of the device layouts of weight matrices (`Kernels::rearrange`,
//! `Kernels::product`): one rearranged copy of a matrix, in the bytes of the
//! file's, and one family of kernels for every number of activation rows.
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test layout -- --test-threads=1
//! ```
//!
//! `ARES_TEST_GGUF=<model.gguf>` adds the check on real tensors: the smallest
//! matrix of every type with a device layout that the model contains.
//!
//! Each kernel is held to two references, as the other weight kernels are:
//!
//! - *tight*: the weights (`ares_quant::dequantize`) and the activations rounded to
//!   f16, multiplied on the CPU in f64. What is left is that the kernel forms a
//!   weight from f16 scales with f16 arithmetic, and, with f16 accumulation, the
//!   rounding of partial sums of 64 products.
//! - *loose*: the exact product of the weights with the unrounded activations. The
//!   deviations are printed (`--nocapture`); the assertion guards against a gross
//!   error.

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, DevicePtr, FEW_ROWS, FewPlan, Kernels, Precision, Stream,
    Wait, WeightLayout, device_count,
};
use ares_quant::{f16_to_f32, f32_to_f16};

/// The types with a device layout, and the byte offsets of the f16 fields of a
/// block that must be finite for the test data to be meaningful.
const TYPES: [(GgmlType, &[usize]); 12] = [
    (GgmlType::Q4_K, &[0, 2]),
    (GgmlType::Q5_K, &[0, 2]),
    (GgmlType::Q6_K, &[208]),
    (GgmlType::Q3_K, &[108]),
    (GgmlType::Q2_K, &[80, 82]),
    (GgmlType::Q4_0, &[0]),
    (GgmlType::Q4_1, &[0, 2]),
    (GgmlType::Q5_0, &[0]),
    (GgmlType::Q5_1, &[0, 2]),
    (GgmlType::Q8_0, &[0]),
    (GgmlType::IQ4_XS, &[0]),
    (GgmlType::IQ4_NL, &[0]),
];

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

/// The kernels of the devices that have the tensor-core family.
fn loaded() -> Vec<(&'static Device, Kernels)> {
    gpus()
        .unwrap_or(&[])
        .iter()
        .map(|d| (d, Kernels::load(d).unwrap()))
        .filter(|(_, k)| k.has_prefill_kernels())
        .collect()
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

    /// A value of a normal distribution, occasionally an outlier: real activations
    /// have a few much larger values per row.
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

/// `n_blocks` blocks of `ty` of random bytes, with finite f16 scales of both signs.
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

fn through_f16(v: f32) -> f64 {
    f64::from(f16_to_f32(f32_to_f16(v)))
}

/// Which kernel a test asks for.
#[derive(Clone, Copy, Debug)]
enum How<'a> {
    /// The one the device's table names.
    Table(Precision),
    /// One kernel of the family above [`FEW_ROWS`] rows by name.
    Kernel(&'a str),
    /// One launch of the few-row kernels.
    Few(FewPlan),
    /// The kernels over the file's layout.
    File(Precision),
}

/// The product of `w` (n_out rows of n_in values of `ty`, as in the file) with the
/// rows `x`, through the device layout or, for [`How::File`], the file's.
#[allow(clippy::too_many_arguments)]
fn run(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    how: How<'_>,
    w: &[u8],
    x: &[f32],
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> Vec<f32> {
    let (n_in32, n_out32, n_rows32) = (n_in as u32, n_out as u32, n_rows as u32);
    let stream = Stream::new(device, 0).unwrap();
    let file = DeviceBuffer::new(device, w.len() as u64).unwrap();
    file.write(0, w).unwrap();
    let bytes = Kernels::layout_bytes(ty, WeightLayout::Device, n_out32, n_in32).unwrap();
    let rearranged = DeviceBuffer::new(device, bytes).unwrap();
    let dx = DeviceBuffer::new(device, x.len() as u64 * 4).unwrap();
    dx.write(0, &f32_bytes(x)).unwrap();
    let (layout, precision) = match how {
        How::Table(p) => (WeightLayout::Device, p),
        How::Kernel(_) | How::Few(_) => (WeightLayout::Device, Precision::Exact),
        How::File(p) => (WeightLayout::File, p),
    };
    let form = kernels.activation_form(ty, layout, precision, n_rows32).unwrap();
    let act = DeviceBuffer::new(device, Kernels::activation_bytes(form, n_in32, n_rows32)).unwrap();
    let y = DeviceBuffer::new(device, (n_rows * n_out) as u64 * 4).unwrap();
    // SAFETY: every buffer is as large as the launch it is given to reads or writes:
    // the file's matrix, its rearranged form, the rows, their prepared form and the
    // products.
    unsafe {
        kernels.rearrange(&stream, ty, file.ptr(), rearranged.ptr(), n_out32, n_in32).unwrap();
        kernels.prepare_activations(&stream, form, dx.ptr(), act.ptr(), n_in32, n_rows32).unwrap();
        match how {
            How::Kernel(name) => kernels
                .product_with(
                    &stream,
                    ty,
                    name,
                    rearranged.ptr(),
                    act.ptr(),
                    y.ptr(),
                    n_in32,
                    n_out32,
                    n_rows32,
                )
                .unwrap(),
            How::Few(plan) => kernels
                .product_few(
                    &stream,
                    ty,
                    plan,
                    rearranged.ptr(),
                    act.ptr(),
                    y.ptr(),
                    n_in32,
                    n_out32,
                    n_rows32,
                )
                .unwrap(),
            _ => {
                let at = if layout == WeightLayout::Device { rearranged.ptr() } else { file.ptr() };
                kernels
                    .product(
                        &stream,
                        ty,
                        layout,
                        precision,
                        at,
                        act.ptr(),
                        y.ptr(),
                        n_in32,
                        n_out32,
                        n_rows32,
                    )
                    .unwrap()
            }
        }
    }
    stream.synchronize().unwrap();
    read_f32(&y, n_rows * n_out)
}

/// How far a result is from the two references.
#[derive(Debug)]
struct Deviation {
    /// Largest `|have - emulation| / sum of |w x|` over the outputs.
    tight: f64,
    /// `rms(have - exact)` over the typical size of a sum, `sqrt(sum of (w x)^2)`
    /// per output.
    loose: f64,
}

impl Deviation {
    fn describe(&self) -> String {
        format!(
            "vs the f16 emulation {:.1e} of the magnitude; vs the f32 product {:.4}% of a sum's size",
            self.tight,
            self.loose * 100.0
        )
    }

    /// The kernels with f32 accumulation differ from the emulation by the f16
    /// arithmetic that forms a weight (up to three roundings of 2^-11 instead of
    /// one); f16 accumulation adds the rounding of the partial sums.
    fn assert_within(&self, what: &str, enough_outputs: bool) {
        assert!(self.tight < 4e-3, "{what}: differs from the emulation: {}", self.describe());
        if enough_outputs {
            assert!(self.loose < 0.003, "{what}: far from the f32 product: {}", self.describe());
        }
    }
}

fn deviation(
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
    let used: Vec<f64> = weights.iter().map(|v| through_f16(*v)).collect();
    let (mut tight, mut err2, mut scale2) = (0f64, 0f64, 0f64);
    for b in 0..n_rows {
        let xb = &x[b * n_in..][..n_in];
        let xs: Vec<f64> = xb.iter().map(|v| through_f16(*v)).collect();
        for r in 0..n_out {
            let (mut emulated, mut magnitude, mut exact, mut squares) = (0f64, 0f64, 0f64, 0f64);
            for k in 0..n_in {
                let p = used[r * n_in + k] * xs[k];
                emulated += p;
                magnitude += p.abs();
                let e = f64::from(weights[r * n_in + k]) * f64::from(xb[k]);
                exact += e;
                squares += e * e;
            }
            let h = f64::from(have[b * n_out + r]);
            tight = tight.max((h - emulated).abs() / magnitude.max(1e-30));
            err2 += (h - exact).powi(2);
            scale2 += squares;
        }
    }
    Deviation { tight, loose: (err2 / scale2.max(1e-60)).sqrt() }
}

#[test]
fn a_device_layout_takes_the_bytes_of_the_file() {
    let file =
        |ty: GgmlType, n_out, n_in| Kernels::layout_bytes(ty, WeightLayout::File, n_out, n_in);
    let device =
        |ty: GgmlType, n_out, n_in| Kernels::layout_bytes(ty, WeightLayout::Device, n_out, n_in);
    for (ty, _) in TYPES {
        assert!(Kernels::has_device_layout(ty, 5120));
        // Whole tiles of 16 rows: the same bytes.
        assert_eq!(device(ty, 17408, 5120), file(ty, 17408, 5120));
        assert_eq!(device(ty, 17408, 5120), ty.bytes_for(17408 * 5120));
        // Rows are rounded up to the tile.
        assert_eq!(device(ty, 17, 256), file(ty, 32, 256));
        assert_eq!(Kernels::layout_alignment(ty, WeightLayout::Device), 16);
    }
    // A row that is not whole steps has no device layout, nor has a type without one.
    assert!(!Kernels::has_device_layout(GgmlType::Q4_K, 96));
    assert_eq!(device(GgmlType::Q4_K, 16, 96), None);
    assert!(!Kernels::has_device_layout(GgmlType::IQ3_S, 5120));
    assert_eq!(device(GgmlType::IQ3_S, 16, 256), None);
    assert_eq!(file(GgmlType::IQ3_S, 16, 256), Some(16 * 110));
}

/// The rearrangement kernel writes, byte for byte, the layout of the CPU reference
/// (`ares_quant::layout`, which the Vulkan backend is held to as well), including
/// the zero rows that fill the last tile.
#[test]
fn the_rearrangement_is_the_reference_layout() {
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for (ty, _) in TYPES {
        for (n_out, n_in) in [(1usize, 256usize), (16, 256), (37, 768), (128, 1024)] {
            let w = random_blocks(&mut rng, ty, n_out * n_in / ty.block_values() as usize);
            let bytes = ares_quant::layout::rearranged_bytes(ty, n_out, n_in).unwrap();
            assert_eq!(
                Kernels::layout_bytes(ty, WeightLayout::Device, n_out as u32, n_in as u32),
                Some(bytes as u64)
            );
            let mut want = vec![0u8; bytes];
            ares_quant::layout::rearrange(ty, n_out, n_in, &w, &mut want).unwrap();
            for (device, kernels) in loaded() {
                let stream = Stream::new(device, 0).unwrap();
                let file = DeviceBuffer::new(device, w.len() as u64).unwrap();
                file.write(0, &w).unwrap();
                let out = DeviceBuffer::new(device, bytes as u64).unwrap();
                // SAFETY: the two buffers hold the matrix in the file's and the device layout.
                unsafe {
                    kernels
                        .rearrange(&stream, ty, file.ptr(), out.ptr(), n_out as u32, n_in as u32)
                        .unwrap();
                }
                stream.synchronize().unwrap();
                let mut have = vec![0xAAu8; bytes];
                out.read(0, &mut have).unwrap();
                assert!(have == want, "{ty} {n_out} x {n_in} on {}", device.info().name);
            }
        }
    }
}

/// Every kernel of the family around its edges: one weight row, rows that do not
/// fill a tile of 16 or a block, one activation row and counts that do not fill a
/// fragment of 8 or the block's rows, more than one activation tile, one
/// super-block and many (the few-row kernels cut K into 8 parts: fewer super-blocks
/// than parts leave parts empty, 10 give them one or two).
#[test]
fn every_kernel_computes_the_product() {
    let shapes: [(usize, usize, &[usize]); 8] = [
        (1, 256, &[1, 9]),
        (17, 256, &[3, 17]),
        (130, 512, &[8, 33]),
        (64, 768, &[16, 65]),
        (129, 1024, &[1, 32, 70]),
        (300, 256, &[7, 130]),
        (33, 2560, &[1, 2, 8, 16, 20]),
        (200, 1536, &[4, 9]),
    ];
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for (ty, _) in TYPES {
        for (n_out, n_in, rows) in shapes {
            let w = random_blocks(&mut rng, ty, n_out * n_in / ty.block_values() as usize);
            for &n_rows in rows {
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                for (device, kernels) in loaded() {
                    let mut worst = (0f64, 0f64);
                    let mut hows: Vec<(String, How)> = Kernels::layout_kernel_names()
                        .into_iter()
                        .map(|name| (name.to_string(), How::Kernel(name)))
                        .collect();
                    if n_rows <= FEW_ROWS as usize {
                        hows.extend(
                            FewPlan::all().into_iter().map(|p| (format!("{p:?}"), How::Few(p))),
                        );
                    }
                    let mut kernels_run = 0;
                    for (name, how) in hows {
                        let have = run(device, &kernels, ty, how, &w, &x, n_in, n_out, n_rows);
                        let dev = deviation(ty, &w, &x, &have, n_in, n_out, n_rows);
                        let what = format!(
                            "{ty} {n_out} x {n_in}, {n_rows} rows, {name} on {}",
                            device.info().name
                        );
                        dev.assert_within(&what, n_out * n_rows >= 64);
                        worst = (worst.0.max(dev.tight), worst.1.max(dev.loose));
                        kernels_run += 1;
                    }
                    eprintln!(
                        "{ty} {n_out} x {n_in}, {n_rows} rows on {}: worst of the {kernels_run} kernels {:.1e} of the magnitude, {:.4}% of a sum's size",
                        device.info().name,
                        worst.0,
                        worst.1 * 100.0
                    );
                }
            }
        }
    }
}

/// `Kernels::product` over the device layout for every number of rows up to past
/// the widest block, with the table the device keeps and the one it measures, and
/// against the kernels over the file's layout.
#[test]
fn the_table_serves_every_number_of_rows() {
    let (n_out, n_in) = (144usize, 512usize);
    let mut rng = Rng(0xd1342543de82ef95);
    for (ty, _) in TYPES {
        let w = random_blocks(&mut rng, ty, n_out * n_in / ty.block_values() as usize);
        for (device, kernels) in loaded() {
            let kept = kernels.layout_table(ty).unwrap();
            assert!(!kept.measured);
            for pass in 0..2 {
                if pass == 1 {
                    let table = kernels.measure_layout(ty).unwrap();
                    assert!(table.measured);
                    assert_eq!(kernels.layout_table(ty).unwrap(), table);
                    for rows in table.rows {
                        assert!(rows.exact.f32_accumulation);
                        assert!(rows.fast.probe_ms > 0.0 && rows.exact.probe_ms > 0.0);
                        // Fast is the exact kernel, or one more than 5 % faster than every exact one.
                        assert!(rows.fast.probe_ms <= rows.exact.probe_ms);
                        eprintln!(
                            "{ty} on {}: up to {} rows fast {} ({:.3} ms), exact {} ({:.3} ms)",
                            device.info().name,
                            rows.up_to_rows,
                            rows.fast.name,
                            rows.fast.probe_ms,
                            rows.exact.name,
                            rows.exact.probe_ms
                        );
                    }
                }
                for n_rows in [1usize, 2, 5, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129] {
                    let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                    for precision in [Precision::Fast, Precision::Exact] {
                        let have = run(
                            device,
                            &kernels,
                            ty,
                            How::Table(precision),
                            &w,
                            &x,
                            n_in,
                            n_out,
                            n_rows,
                        );
                        let dev = deviation(ty, &w, &x, &have, n_in, n_out, n_rows);
                        let what = format!(
                            "{ty} {n_rows} rows {precision:?} on {} (pass {pass})",
                            device.info().name
                        );
                        dev.assert_within(&what, true);
                        // The tile kernel over the file's layout computes the same
                        // product from the same f16 operands.
                        if pass == 0 && precision == Precision::Exact {
                            let file = run(
                                device,
                                &kernels,
                                ty,
                                How::File(Precision::Exact),
                                &w,
                                &x,
                                n_in,
                                n_out,
                                n_rows,
                            );
                            let dev = deviation(ty, &w, &x, &file, n_in, n_out, n_rows);
                            dev.assert_within(&format!("{what}, file layout"), true);
                        }
                    }
                }
            }
        }
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The product of a row up to [`FEW_ROWS`] rows is the same bits whatever the other rows
/// of the batch are, how many there are, which kernel and which launch computes it, and
/// whether the launch was measured: a verify window of drafts computes the rows it shares
/// with a decode step as that step does, so that drafts cannot change the text.
#[test]
fn a_row_is_the_same_bits_in_any_batch_and_launch() {
    let mut rng = Rng(0x6a09e667f3bcc908);
    // Shapes of 5 super-blocks (parts of one or two), a ragged tile, and 17 super-blocks.
    for (n_out, n_in) in [(70usize, 1280usize), (33, 4352), (16, 256)] {
        for (ty, _) in TYPES {
            let w = random_blocks(&mut rng, ty, n_out * n_in / ty.block_values() as usize);
            let x: Vec<f32> = (0..n_in * FEW_ROWS as usize).map(|_| rng.activation()).collect();
            for (device, kernels) in loaded() {
                let all =
                    run(device, &kernels, ty, How::Table(Precision::Fast), &w, &x, n_in, n_out, 16);
                let what = format!("{ty} {n_out} x {n_in} on {}", device.info().name);
                for n_rows in [1usize, 2, 3, 4, 5, 8, 9, 12, 16] {
                    let part = run(
                        device,
                        &kernels,
                        ty,
                        How::Table(Precision::Fast),
                        &w,
                        &x[..n_rows * n_in],
                        n_in,
                        n_out,
                        n_rows,
                    );
                    assert_eq!(
                        bits(&part),
                        bits(&all[..n_rows * n_out]),
                        "{what}: {n_rows} rows differ from the first {n_rows} of 16"
                    );
                    let exact = run(
                        device,
                        &kernels,
                        ty,
                        How::Table(Precision::Exact),
                        &w,
                        &x[..n_rows * n_in],
                        n_in,
                        n_out,
                        n_rows,
                    );
                    assert_eq!(bits(&exact), bits(&part), "{what}: exact and fast differ");
                }
                // One row alone, in every launch, with 1 and 16 rows.
                for plan in FewPlan::all() {
                    for n_rows in [1usize, 16] {
                        let have = run(
                            device,
                            &kernels,
                            ty,
                            How::Few(plan),
                            &w,
                            &x[..n_rows * n_in],
                            n_in,
                            n_out,
                            n_rows,
                        );
                        assert_eq!(
                            bits(&have),
                            bits(&all[..n_rows * n_out]),
                            "{what}: {plan:?} with {n_rows} rows"
                        );
                    }
                }
            }
        }
    }
}

/// Measuring the launch of a shape keeps a launch of the family and changes no bit: the product
/// after it is the one before, and so is the product of every launch the measurement could have
/// chosen (the untuned table's launch need not be of the same kind as the measured one).
#[test]
fn tuning_a_shape_changes_no_bit() {
    let mut rng = Rng(0xbb67ae8584caa73b);
    for (n_out, n_in) in [(144usize, 2048usize), (2000, 1024)] {
        for (ty, _) in [TYPES[0], TYPES[2], TYPES[9], TYPES[10]] {
            let w = random_blocks(&mut rng, ty, n_out * n_in / ty.block_values() as usize);
            let x: Vec<f32> = (0..n_in * 4).map(|_| rng.activation()).collect();
            for (device, kernels) in loaded() {
                let (n_out32, n_in32) = (n_out as u32, n_in as u32);
                let before =
                    run(device, &kernels, ty, How::Table(Precision::Fast), &w, &x, n_in, n_out, 4);
                let plan = kernels.tune_layout(ty, n_out32, n_in32).unwrap();
                assert!(FewPlan::all().contains(&plan));
                assert_eq!(kernels.few_plan(ty, n_out32, n_in32).unwrap(), plan);
                assert_eq!(kernels.tune_layout(ty, n_out32, n_in32).unwrap(), plan);
                eprintln!("{ty} {n_out} x {n_in} on {}: {plan:?}", device.info().name);
                let after =
                    run(device, &kernels, ty, How::Table(Precision::Fast), &w, &x, n_in, n_out, 4);
                assert_eq!(bits(&before), bits(&after), "{ty} {n_out} x {n_in}");
                for candidate in FewPlan::all() {
                    let have =
                        run(device, &kernels, ty, How::Few(candidate), &w, &x, n_in, n_out, 4);
                    assert_eq!(
                        bits(&before),
                        bits(&have),
                        "{ty} {n_out} x {n_in}: {candidate:?} differs from the untuned table"
                    );
                }
            }
        }
    }
}

/// The smallest matrix of every type with a device layout that the model contains,
/// its first 200 rows (not a whole tile), for row counts of every bucket.
#[test]
fn the_device_layout_on_real_tensors() {
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!("skipped: set ARES_TEST_GGUF=<model.gguf> to check real tensors");
        return;
    };
    let kernels_of = loaded();
    if kernels_of.is_empty() {
        return;
    }
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    let mut rng = Rng(0x853c49e6748fea9b);
    let mut checked = Vec::new();
    for (ty, _) in TYPES {
        let Some(tensor) = gguf
            .tensors()
            .iter()
            .filter(|t| {
                t.ty == ty
                    && t.dims.as_slice().len() == 2
                    && Kernels::has_device_layout(ty, t.dims.get(0) as u32)
            })
            .min_by_key(|t| t.size)
        else {
            continue;
        };
        let (n_in, n_out) = (tensor.dims.get(0) as usize, tensor.dims.get(1) as usize);
        let rows = n_out.min(200);
        let row_bytes = ty.bytes_for(n_in as u64).unwrap() as usize;
        let w = &gguf.data(tensor)[..rows * row_bytes];
        for n_rows in [1usize, 8, 12, 24, 70] {
            let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
            for (device, kernels) in &kernels_of {
                for precision in [Precision::Fast, Precision::Exact] {
                    let have =
                        run(device, kernels, ty, How::Table(precision), w, &x, n_in, rows, n_rows);
                    let dev = deviation(ty, w, &x, &have, n_in, rows, n_rows);
                    eprintln!(
                        "{} ({ty}, {rows} x {n_in}) rows {n_rows} {precision:?} on {}: {}",
                        tensor.name,
                        device.info().name,
                        dev.describe()
                    );
                    dev.assert_within(&tensor.name, true);
                }
            }
        }
        checked.push(ty);
    }
    assert!(!checked.is_empty(), "the model has no 2-D tensor of a type with a device layout");
    eprintln!("checked types: {}", checked.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "));
}

#[test]
fn bad_arguments_are_refused() {
    for (device, kernels) in loaded() {
        let stream = Stream::new(device, 0).unwrap();
        let buffer = DeviceBuffer::new(device, 1 << 16).unwrap();
        let (p, null) = (buffer.ptr(), DevicePtr::NULL);
        let q4 = GgmlType::Q4_K;
        let fast = Precision::Fast;
        // SAFETY: every call is refused before anything is launched, or has no weight
        // rows and launches nothing.
        unsafe {
            // A type without a device layout.
            assert!(kernels.rearrange(&stream, GgmlType::IQ3_S, p, p, 16, 256).is_err());
            assert!(
                kernels
                    .product(
                        &stream,
                        GgmlType::IQ3_S,
                        WeightLayout::Device,
                        fast,
                        p,
                        p,
                        p,
                        256,
                        16,
                        1
                    )
                    .is_err()
            );
            assert!(kernels.layout_table(GgmlType::IQ3_S).is_err());
            assert!(
                kernels.activation_form(GgmlType::IQ3_S, WeightLayout::Device, fast, 1).is_err()
            );
            // Rows that are not whole blocks, no rows of activations.
            assert!(kernels.rearrange(&stream, q4, p, p, 16, 96).is_err());
            assert!(
                kernels
                    .product(&stream, q4, WeightLayout::Device, fast, p, p, p, 96, 16, 1)
                    .is_err()
            );
            assert!(
                kernels
                    .product(&stream, q4, WeightLayout::Device, fast, p, p, p, 256, 16, 0)
                    .is_err()
            );
            // Misaligned addresses.
            let odd = p.offset(8);
            assert!(kernels.rearrange(&stream, q4, odd, p, 16, 256).is_err());
            assert!(kernels.rearrange(&stream, q4, p, odd, 16, 256).is_err());
            assert!(
                kernels
                    .product(&stream, q4, WeightLayout::Device, fast, odd, p, p, 256, 16, 1)
                    .is_err()
            );
            assert!(
                kernels
                    .product(&stream, q4, WeightLayout::Device, fast, p, odd, p, 256, 16, 1)
                    .is_err()
            );
            // A kernel that does not exist, a launch that is not one of the family, and the
            // few-row kernels with more rows than they take.
            assert!(kernels.product_with(&stream, q4, "r7", p, p, p, 256, 16, 1).is_err());
            let plan = FewPlan::all()[0];
            assert!(
                kernels.product_few(&stream, q4, plan, p, p, p, 256, 16, FEW_ROWS + 1).is_err()
            );
            let odd_plan = FewPlan { block_warps: 6, log2_warps_per_tile: 1 };
            assert!(kernels.product_few(&stream, q4, odd_plan, p, p, p, 256, 16, 1).is_err());
            assert!(kernels.product_few(&stream, q4, plan, p, p, p, 96, 16, 1).is_err());
            assert!(kernels.tune_layout(q4, 16, 96).is_err());
            assert!(kernels.tune_layout(GgmlType::IQ3_S, 16, 256).is_err());
            // No weight rows is nothing to do, and a null address is not touched.
            kernels.rearrange(&stream, q4, null, null, 0, 256).unwrap();
            kernels
                .product(&stream, q4, WeightLayout::Device, fast, null, null, null, 256, 0, 1)
                .unwrap();
        }
        assert_eq!(
            kernels.activation_form(q4, WeightLayout::Device, fast, 1).unwrap(),
            ActivationForm::F16
        );
        assert_eq!(
            kernels.activation_form(q4, WeightLayout::File, fast, 1).unwrap(),
            ActivationForm::Int8
        );
    }
}
