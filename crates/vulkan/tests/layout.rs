//! Tests of the device layouts of weight matrices (`Kernels::rearrange`,
//! `Kernels::product`): the twin of `crates/cuda/tests/layout.rs`, held to the same
//! references.
//!
//! - The rearrangement makes the bytes of `ares_quant::layout::rearrange`, which CUDA is
//!   tested against too: one upload format serves both backends.
//! - Every kernel of the family over a layout computes the product: its error against the
//!   exact f32 product (in f64) is the rounding of weights and activations to f16 and
//!   nothing more, for one row up to more than a tile, on shapes with partial tiles on
//!   both sides.
//! - The decode kernel on the file's layout gives the same product within the error of its
//!   8-bit activations.
//!
//! They open the GPUs, so they run only with `ARES_TEST_VULKAN=1`:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test layout -- --test-threads=1
//! ```
//!
//! `ARES_TEST_GGUF=<model.gguf>` adds real tensors.

use std::sync::OnceLock;

use ares_core::GgmlType;
use ares_vulkan::{
    ActivationForm, Device, DeviceBuffer, Kernels, Layout, Precision, Stream, device_count,
};

/// The types with a device layout and the byte offsets of the f16 fields of a block that
/// must be finite for the test data to be meaningful.
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

    /// A value of a normal distribution, occasionally an outlier.
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

/// `n_blocks` blocks of `ty` of random bytes with finite f16 scales of both signs (any
/// other bytes are valid in every one of these types).
fn random_blocks(rng: &mut Rng, ty: GgmlType, n_blocks: usize) -> Vec<u8> {
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
    }
    bytes
}

/// The exact product in f64 and, per output, the size of a sum (the root of the sum of the
/// squared terms): what an error is measured against.
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

/// `rms(have - exact)` over the rms size of a sum.
fn relative_error(have: &[f32], exact: &[f64], size: &[f64]) -> f64 {
    let n = have.len() as f64;
    let err: f64 = have.iter().zip(exact).map(|(h, e)| (f64::from(*h) - e).powi(2)).sum();
    let scale: f64 = size.iter().map(|s| s * s).sum();
    (err / n).sqrt() / (scale / n).sqrt().max(1e-30)
}

/// A matrix on a device in both layouts.
struct Matrix {
    file: DeviceBuffer,
    device: DeviceBuffer,
}

fn upload(gpu: &Gpu, stream: &Stream, ty: GgmlType, w: &[u8], n_out: usize, n_in: usize) -> Matrix {
    let file = DeviceBuffer::new(&gpu.device, w.len().next_multiple_of(16) as u64).unwrap();
    file.write(0, w).unwrap();
    let bytes = Kernels::layout_bytes(ty, Layout::Device, n_out as u32, n_in as u32).unwrap();
    let device = DeviceBuffer::new(&gpu.device, bytes).unwrap();
    // SAFETY: the buffers hold the matrix in the two layouts.
    unsafe {
        gpu.kernels
            .rearrange(stream, ty, file.ptr(), device.ptr(), n_out as u32, n_in as u32)
            .unwrap();
    }
    stream.synchronize().unwrap();
    Matrix { file, device }
}

/// The rearrangement on the device makes the bytes of the CPU reference, for every type,
/// with whole and partial tiles of rows.
#[test]
fn the_rearrangement_is_the_reference_byte_for_byte() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x1a70_07f0_0d5e_ed01);
    for gpu in gpus {
        if !gpu.kernels.has_layout_kernels() {
            eprintln!("skipped on {}: no kernels over the device layouts", gpu.device.info().name);
            continue;
        }
        let stream = Stream::new(&gpu.device).unwrap();
        for (ty, _) in TYPES {
            for (n_out, n_in) in [(1usize, 256usize), (16, 256), (21, 512), (48, 768), (130, 1280)]
            {
                assert_eq!(gpu.kernels.layout_for(ty, n_in as u32), Layout::Device);
                let blocks = n_in / ty.block_values() as usize * n_out;
                // Any bytes: the rearrangement moves codes and headers and reads no value.
                let w: Vec<u8> = (0..blocks * ty.block_bytes() as usize)
                    .map(|_| (rng.next() >> 32) as u8)
                    .collect();
                let matrix = upload(gpu, &stream, ty, &w, n_out, n_in);
                let mut want =
                    vec![0u8; ares_quant::layout::rearranged_bytes(ty, n_out, n_in).unwrap()];
                ares_quant::layout::rearrange(ty, n_out, n_in, &w, &mut want).unwrap();
                assert_eq!(matrix.device.len(), want.len() as u64);
                let mut have = vec![0u8; want.len()];
                matrix.device.read(0, &mut have).unwrap();
                let first = have.iter().zip(&want).position(|(h, w)| h != w);
                assert!(
                    first.is_none(),
                    "{ty} {n_out} x {n_in} on {}: byte {} of {} differs",
                    gpu.device.info().name,
                    first.unwrap(),
                    want.len()
                );
            }
        }
        // A type without a device layout, and rows that are not whole steps, keep the file's.
        assert_eq!(gpu.kernels.layout_for(GgmlType::IQ3_S, 512), Layout::File);
        assert_eq!(gpu.kernels.layout_for(GgmlType::Q4_0, 96), Layout::File);
    }
}

/// Every kernel of the family, on every type: close to the exact product for 1 row up to
/// more than a tile of rows. The decode kernel on the file's layout agrees within the error
/// of its 8-bit activations.
#[test]
fn every_kernel_of_a_layout_computes_the_product() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x51de_cafe_f00d_0002);
    let mut failures = Vec::new();
    let mut worst: Vec<(GgmlType, &str, f64)> = Vec::new();
    // The pipelines of all kernels are made up front on all cores: the driver takes up to a
    // second for each the first time it sees one.
    let types = TYPES.map(|(ty, _)| ty);
    for gpu in gpus {
        if gpu.kernels.has_layout_kernels() {
            gpu.kernels.prepare(&types).unwrap();
        }
    }
    // The data and the exact product are made once per type and shape, off the devices.
    for (ty, _) in TYPES {
        for (n_in, n_out, n_rows) in [
            (256usize, 1usize, 1usize),
            (512, 70, 8),
            (1280, 130, 3),
            (2560, 64, 17),
            (5120, 129, 33),
            (768, 200, 100),
        ] {
            let blocks = n_in / ty.block_values() as usize * n_out;
            let w = random_blocks(&mut rng, ty, blocks);
            // Rows of different magnitudes: the kernels divide each by a power of two.
            let x: Vec<f32> = (0..n_in * n_rows)
                .map(|i| rng.activation() * 4f32.powi(((i / n_in) % 7) as i32 - 3))
                .collect();
            let (exact, size) = exact_product(ty, &w, &x, n_in, n_out, n_rows);
            for gpu in gpus {
                if !gpu.kernels.has_layout_kernels() {
                    continue;
                }
                let name = &gpu.device.info().name;
                let stream = Stream::new(&gpu.device).unwrap();
                let matrix = upload(gpu, &stream, ty, &w, n_out, n_in);
                let dx = DeviceBuffer::new(&gpu.device, x.len() as u64 * 4).unwrap();
                dx.write(0, &f32_bytes(&x)).unwrap();
                let act_bytes = [ActivationForm::F16, ActivationForm::Int8]
                    .map(|f| Kernels::activation_bytes(f, n_in as u32, n_rows as u32))
                    .into_iter()
                    .max()
                    .unwrap();
                let act = DeviceBuffer::new(&gpu.device, act_bytes).unwrap();
                let dy = DeviceBuffer::new(&gpu.device, (n_rows * n_out * 4) as u64).unwrap();
                let (ni, no, nr) = (n_in as u32, n_out as u32, n_rows as u32);
                // SAFETY: the buffers are sized for these shapes.
                unsafe {
                    gpu.kernels
                        .prepare_activations(
                            &stream,
                            ActivationForm::F16,
                            dx.ptr(),
                            act.ptr(),
                            ni,
                            nr,
                        )
                        .unwrap();
                }
                for kernel in Kernels::layout_kernel_names() {
                    // SAFETY: as above.
                    unsafe {
                        stream.fill(dy.ptr(), 0xFF, n_rows * n_out * 4).unwrap();
                        gpu.kernels
                            .product_with(
                                &stream,
                                ty,
                                kernel,
                                matrix.device.ptr(),
                                act.ptr(),
                                dy.ptr(),
                                ni,
                                no,
                                nr,
                            )
                            .unwrap();
                    }
                    stream.synchronize().unwrap();
                    let have = read_f32(&dy, n_rows * n_out);
                    let error = relative_error(&have, &exact, &size);
                    match worst.iter_mut().find(|(t, k, _)| *t == ty && *k == kernel) {
                        Some(entry) => entry.2 = entry.2.max(error),
                        None => worst.push((ty, kernel, error)),
                    }
                    // The rounding of the operands to f16 is about 4e-4 of a sum's size
                    // and that of partial sums in f16 as much again; small shapes scatter.
                    if error.is_nan() || error >= 2.5e-3 {
                        failures.push(format!(
                            "{ty} {kernel} {n_out}x{n_in} x{n_rows} on {name}: {error:.5}; have {:?} exact {:?}",
                            &have[..have.len().min(4)],
                            &exact[..exact.len().min(4)]
                        ));
                    }
                }
                // The one call a model makes, in both layouts and precisions.
                for (layout, precision, limit) in [
                    (Layout::Device, Precision::Fast, 2.5e-3),
                    (Layout::Device, Precision::Exact, 2.5e-3),
                    (Layout::File, Precision::Fast, 3e-2),
                    (Layout::File, Precision::Exact, 2.5e-3),
                ] {
                    let form = gpu.kernels.activation_form(ty, layout, precision, nr).unwrap();
                    let w_ptr =
                        if layout == Layout::Device { &matrix.device } else { &matrix.file };
                    // SAFETY: as above.
                    unsafe {
                        gpu.kernels
                            .prepare_activations(&stream, form, dx.ptr(), act.ptr(), ni, nr)
                            .unwrap();
                        stream.fill(dy.ptr(), 0xFF, n_rows * n_out * 4).unwrap();
                        gpu.kernels
                            .product(
                                &stream,
                                ty,
                                layout,
                                precision,
                                w_ptr.ptr(),
                                act.ptr(),
                                dy.ptr(),
                                ni,
                                no,
                                nr,
                            )
                            .unwrap();
                    }
                    stream.synchronize().unwrap();
                    let have = read_f32(&dy, n_rows * n_out);
                    let error = relative_error(&have, &exact, &size);
                    // A handful of sums of 8-bit activations scatter more than many.
                    let limit = if n_out * n_rows < 64 { limit * 3.0 } else { limit };
                    if error.is_nan() || error >= limit {
                        failures.push(format!(
                            "{ty} product {layout:?} {precision:?} ({form:?}) {n_out}x{n_in} x{n_rows} on {name}: {error:.5}"
                        ));
                    }
                }
            }
        }
    }
    for (ty, kernel, error) in &worst {
        eprintln!("{ty} {kernel}: at worst {:.4} % of a sum's size", error * 100.0);
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

/// The table: without a measurement the narrowest shapes, after one the fastest, and the
/// product goes through it.
#[test]
fn a_table_is_measured_and_kept() {
    let Some(gpus) = gpus() else { return };
    for gpu in gpus {
        if !gpu.kernels.has_layout_kernels() {
            continue;
        }
        let ty = GgmlType::Q4_K;
        let kept = gpu.kernels.layout_table(ty).unwrap();
        assert!(!kept.measured);
        let measured = gpu.kernels.measure_layout(ty).unwrap();
        assert!(measured.measured);
        assert_eq!(gpu.kernels.layout_table(ty).unwrap(), measured);
        for (rows, file) in measured.rows.iter().zip(measured.file_ms) {
            assert!(rows.exact.f32_accumulation);
            assert!(rows.fast.probe_ms > 0.0 && rows.exact.probe_ms >= rows.fast.probe_ms * 0.97);
            eprintln!(
                "{} q4_K up to {} rows: fast {} {:.4} ms, exact {} {:.4} ms, file layout {:.4} ms",
                gpu.device.info().name,
                rows.up_to_rows,
                rows.fast.name,
                rows.fast.probe_ms,
                rows.exact.name,
                rows.exact.probe_ms,
                file
            );
        }
        assert!(gpu.kernels.layout_table(GgmlType::IQ3_S).is_err());
        assert!(gpu.kernels.measure_layout(GgmlType::F16).is_err());
    }
}

/// Real tensors: the smallest matrix of every type with a device layout that the model
/// carries, its first 512 rows.
#[test]
fn real_tensors_in_the_device_layout() {
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
            .filter(|t| {
                matches!(t.dims.as_slice().len(), 2 | 3)
                    && t.ty == ty
                    && Kernels::has_device_layout(ty, t.dims.get(0) as u32)
            })
            .min_by_key(|t| t.size)
        else {
            continue;
        };
        let (n_in, n_out) = (tensor.dims.get(0) as usize, tensor.dims.get(1) as usize);
        let rows = n_out.min(512);
        let row_bytes = ty.bytes_for(n_in as u64).unwrap() as usize;
        let w = &gguf.data(tensor)[..rows * row_bytes];
        let mut want = vec![0u8; ares_quant::layout::rearranged_bytes(ty, rows, n_in).unwrap()];
        ares_quant::layout::rearrange(ty, rows, n_in, w, &mut want).unwrap();
        for gpu in gpus {
            if !gpu.kernels.has_layout_kernels() {
                continue;
            }
            let stream = Stream::new(&gpu.device).unwrap();
            let matrix = upload(gpu, &stream, ty, w, rows, n_in);
            let mut have = vec![0u8; want.len()];
            matrix.device.read(0, &mut have).unwrap();
            assert!(have == want, "{}: the layout differs from the reference", tensor.name);
            for n_rows in [1usize, 8, 40] {
                let x: Vec<f32> = (0..n_in * n_rows).map(|_| rng.activation()).collect();
                let (exact, size) = exact_product(ty, w, &x, n_in, rows, n_rows);
                let dx = DeviceBuffer::new(&gpu.device, x.len() as u64 * 4).unwrap();
                dx.write(0, &f32_bytes(&x)).unwrap();
                let (ni, no, nr) = (n_in as u32, rows as u32, n_rows as u32);
                let act = DeviceBuffer::new(
                    &gpu.device,
                    Kernels::activation_bytes(ActivationForm::F16, ni, nr),
                )
                .unwrap();
                let dy = DeviceBuffer::new(&gpu.device, (n_rows * rows * 4) as u64).unwrap();
                for precision in [Precision::Fast, Precision::Exact] {
                    // SAFETY: the buffers are sized for these shapes.
                    unsafe {
                        gpu.kernels
                            .prepare_activations(
                                &stream,
                                ActivationForm::F16,
                                dx.ptr(),
                                act.ptr(),
                                ni,
                                nr,
                            )
                            .unwrap();
                        gpu.kernels
                            .product(
                                &stream,
                                ty,
                                Layout::Device,
                                precision,
                                matrix.device.ptr(),
                                act.ptr(),
                                dy.ptr(),
                                ni,
                                no,
                                nr,
                            )
                            .unwrap();
                    }
                    stream.synchronize().unwrap();
                    let error = relative_error(&read_f32(&dy, n_rows * rows), &exact, &size);
                    eprintln!(
                        "{} ({ty}, {rows} x {n_in}) rows {n_rows} {precision:?} on {}: {:.4} % of a sum's size",
                        tensor.name,
                        gpu.device.info().name,
                        error * 100.0
                    );
                    assert!(error < 2.5e-3, "{}: {error}", tensor.name);
                }
            }
        }
        checked.push(ty);
    }
    assert!(!checked.is_empty(), "the model has no matrix of a type with a device layout");
    eprintln!("checked types: {}", checked.iter().map(|t| t.name()).collect::<Vec<_>>().join(", "));
}
