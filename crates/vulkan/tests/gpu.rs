//! Tests on real devices.
//!
//! Opening a device and running kernels uses the GPUs, which may be busy with
//! a production model. So these tests run only when `ARES_TEST_VULKAN=1` is
//! set and report themselves as skipped otherwise:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test gpu
//! ```
//!
//! `ARES_TEST_GGUF=<model.gguf>` adds a check of the weight kernel against a
//! real Q4_K matrix of that model.
//!
//! The Vulkan validation layers are not needed, and not used unless the
//! environment turns them on (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`);
//! then run with `-- --nocapture` to see what they say.

use std::sync::OnceLock;
use std::time::Instant;

use ares_core::GgmlType;
use ares_vulkan::{
    Args, DEFAULT_BATCH, Device, DeviceBuffer, DevicePtr, DeviceType, Error, Event, Kernels,
    MAX_BATCH, Module, Result, StagingBuffer, Stream, Wait, device_count, devices,
};

struct Gpu {
    device: Device,
    kernels: Kernels,
}

/// The opened devices with their kernels, shared by all tests of this binary so
/// each pipeline is compiled once. `None` when the tests are not enabled.
fn gpus() -> Option<&'static [Gpu]> {
    static GPUS: OnceLock<Option<Vec<Gpu>>> = OnceLock::new();
    GPUS.get_or_init(|| {
        if std::env::var_os("ARES_TEST_VULKAN").is_none() {
            eprintln!("skipped: set ARES_TEST_VULKAN=1 to run the tests that open GPUs");
            return None;
        }
        let count = device_count().expect("the Vulkan loader");
        assert!(count > 0, "ARES_TEST_VULKAN is set but the loader reports no device");
        Some(
            (0..count)
                .map(|i| {
                    let device = Device::open(i).expect("open device");
                    let kernels = Kernels::load(&device).expect("load kernels");
                    Gpu { device, kernels }
                })
                .collect(),
        )
    })
    .as_deref()
}

/// Devices of their own, for tests that count what a stream has submitted.
/// Dropping a buffer synchronises its whole device (as `cuMemFree` does),
/// which flushes every stream on it, so the counts of a stream are only exact
/// on a device that no other test thread uses. A logical device costs next to
/// nothing, unlike a CUDA context, so tests can simply open their own.
fn isolated() -> Vec<Gpu> {
    (0..device_count().unwrap())
        .map(|i| {
            let device = Device::open(i).expect("open device");
            let kernels = Kernels::load(&device).expect("load kernels");
            Gpu { device, kernels }
        })
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
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn read_f32(buffer: &DeviceBuffer, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buffer.read(0, &mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn upload_f32(device: &Device, values: &[f32]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, values.len() as u64 * 4).unwrap();
    buffer.write(0, &f32_bytes(values)).unwrap();
    buffer
}

#[test]
fn describes_devices_without_opening_them() {
    if gpus().is_none() {
        return;
    }
    let infos = devices().unwrap();
    assert_eq!(infos.len() as u32, device_count().unwrap());
    assert!(!infos.is_empty());
    for (i, info) in infos.iter().enumerate() {
        assert_eq!(info.ordinal as usize, i);
        assert!(!info.name.is_empty());
        assert_ne!(info.vendor_id, 0, "{info:?}");
        assert!(info.api_version >= (1, 2, 0), "the engine needs Vulkan 1.2: {info:?}");
        assert!(info.total_memory > 1 << 30, "{info:?}");
        assert_ne!(info.uuid, [0; 16]);
        assert!(info.subgroup_size >= 8, "{info:?}");
        // llvmpipe and the like are never listed.
        assert!(matches!(
            info.device_type,
            DeviceType::Discrete | DeviceType::Integrated | DeviceType::Virtual | DeviceType::Other
        ));
        if info.vendor_id == 0x10de {
            // NVIDIA implements VK_EXT_pci_bus_info.
            let pci = info.pci.expect("an NVIDIA device reports its PCI address");
            let text = pci.to_string();
            assert_eq!(text.len(), "0000:01:00.0".len(), "{text}");
            assert_eq!(text.as_bytes()[4], b':');
            assert_eq!(text.as_bytes()[10], b'.');
        }
    }
    // Two devices are two different GPUs.
    for (i, a) in infos.iter().enumerate() {
        for b in &infos[i + 1..] {
            assert_ne!(a.uuid, b.uuid);
        }
    }
    let missing = infos.len() as u32;
    assert!(matches!(
        Device::open(missing),
        Err(Error::NoSuchDevice { ordinal, count }) if ordinal == missing && count == missing
    ));
}

#[test]
fn buffers_round_trip_and_check_their_bounds() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, .. } in gpus {
        let info = device.mem_info().unwrap();
        assert!(info.free <= info.total);
        assert_eq!(info.total, device.info().total_memory);

        let buffer = DeviceBuffer::new(device, 1 << 20).unwrap();
        let data: Vec<u8> = (0..1usize << 20).map(|i| (i * 31 % 251) as u8).collect();
        buffer.write(0, &data).unwrap();
        let mut back = vec![0u8; data.len()];
        buffer.read(0, &mut back).unwrap();
        assert_eq!(back, data);

        // A window in the middle.
        buffer.write(1000, &[1, 2, 3, 4]).unwrap();
        let mut window = [0u8; 6];
        buffer.read(999, &mut window).unwrap();
        assert_eq!(window, [data[999], 1, 2, 3, 4, data[1004]]);

        assert!(buffer.write((1 << 20) - 3, &[0; 4]).is_err());
        assert!(buffer.read(1 << 20, &mut [0; 1]).is_err());
        assert!(buffer.slice(u64::MAX, 2).is_err());
        assert!(DeviceBuffer::new(device, 0).is_err());
        assert!(StagingBuffer::new(device, 0).is_err());

        // Empty transfers are accepted and do nothing.
        buffer.write(5, &[]).unwrap();
        buffer.read(1 << 20, &mut []).unwrap();
    }
}

/// Transfers go through two 8 MiB slots of staging memory; sizes and offsets
/// around and across the chunk boundaries must all arrive intact.
#[test]
fn large_transfers_cross_chunk_boundaries() {
    let Some(gpus) = gpus() else { return };
    let chunk = 8usize << 20;
    for Gpu { device, .. } in gpus {
        let buffer = DeviceBuffer::new(device, 3 * chunk as u64 + 4096).unwrap();
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let data: Vec<u8> = (0..buffer.len() as usize).map(|_| (rng.next() >> 33) as u8).collect();
        buffer.write(0, &data).unwrap();
        for (offset, len) in [
            (0, chunk - 1),
            (0, chunk),
            (0, chunk + 1),
            (7, 2 * chunk),
            (13, 2 * chunk + 5),
            (4096, 3 * chunk - 100),
            (0, 3 * chunk + 4096),
            (chunk - 3, 6),
        ] {
            let mut back = vec![0u8; len];
            buffer.read(offset as u64, &mut back).unwrap();
            assert!(back == data[offset..offset + len], "read {len} bytes at {offset}");

            // Write the complement, check, and put the original back.
            let flipped: Vec<u8> = data[offset..offset + len].iter().map(|b| !b).collect();
            buffer.write(offset as u64, &flipped).unwrap();
            let mut check = vec![0u8; len];
            buffer.read(offset as u64, &mut check).unwrap();
            assert!(check == flipped, "write {len} bytes at {offset}");
            // Nothing outside the range was touched.
            if offset > 0 {
                let mut before = [0u8; 1];
                buffer.read(offset as u64 - 1, &mut before).unwrap();
                assert_eq!(before[0], data[offset - 1]);
            }
            let end = offset + len;
            if end < data.len() {
                let mut after = [0u8; 1];
                buffer.read(end as u64, &mut after).unwrap();
                assert_eq!(after[0], data[end]);
            }
            buffer.write(offset as u64, &data[offset..end]).unwrap();
        }
    }
}

#[test]
fn an_allocation_that_cannot_fit_is_reported_as_out_of_memory() {
    let Some(gpus) = gpus() else { return };
    let error = DeviceBuffer::new(&gpus[0].device, 1 << 40).unwrap_err();
    assert!(error.is_out_of_memory(), "{error}");
}

#[test]
fn a_stream_runs_its_work_in_order() {
    if gpus().is_none() {
        return;
    }
    for Gpu { device, .. } in &isolated() {
        let stream = Stream::new(device).unwrap();
        let len = 4usize << 20;
        let a = DeviceBuffer::new(device, len as u64).unwrap();
        let b = DeviceBuffer::new(device, len as u64).unwrap();
        let mut staging_in = StagingBuffer::new(device, len).unwrap();
        let staging_out = StagingBuffer::new(device, len).unwrap();
        for (i, byte) in staging_in.iter_mut().enumerate() {
            *byte = (i % 253) as u8;
        }
        let done = Event::new(device).unwrap();

        // SAFETY: all ranges are whole buffers of `len` bytes on this device
        // that outlive the synchronisation below; the staging buffers are not
        // touched until then.
        unsafe {
            stream.fill(a.ptr(), 7, len).unwrap();
            stream.copy_from_staging(a.ptr(), &staging_in, 0, len).unwrap();
            stream.copy(b.ptr(), a.ptr(), len).unwrap();
            stream.fill(a.ptr(), 9, len).unwrap();
            stream.copy_to_staging(&staging_out, 0, b.ptr(), len).unwrap();
        }
        // Nothing has reached the device yet: five commands are recorded.
        assert_eq!(stream.pending_commands(), 5);
        assert_eq!(stream.submissions(), 0);
        done.record(&stream).unwrap();
        // Recording the event submitted them.
        assert_eq!(stream.pending_commands(), 0);
        assert_eq!(stream.submissions(), 1);
        done.synchronize().unwrap();
        assert!(done.is_done().unwrap());
        stream.synchronize().unwrap();
        assert!(stream.is_idle().unwrap());

        // The host data went through `a` into `b` before `a` was overwritten.
        assert_eq!(&staging_out[..], &staging_in[..]);
        let mut first = [0u8; 16];
        a.read(0, &mut first).unwrap();
        assert_eq!(first, [9; 16]);
    }
}

#[test]
fn copies_and_fills_refuse_what_vulkan_cannot_do() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, .. } in gpus {
        let stream = Stream::new(device).unwrap();
        let buffer = DeviceBuffer::new(device, 1 << 16).unwrap();
        let other = DeviceBuffer::new(device, 1 << 16).unwrap();
        let staging = StagingBuffer::new(device, 1 << 10).unwrap();
        let outside = DevicePtr::NULL;
        let past_the_end = buffer.ptr().offset(1 << 16);
        // SAFETY: every call is refused before anything is recorded.
        unsafe {
            // Addresses that are not inside one live buffer.
            assert!(stream.fill(outside, 0, 16).is_err());
            assert!(stream.fill(past_the_end, 0, 4).is_err());
            assert!(stream.fill(buffer.ptr().offset((1 << 16) - 4), 0, 8).is_err());
            assert!(stream.copy(buffer.ptr(), outside, 16).is_err());
            assert!(stream.copy(outside, buffer.ptr(), 16).is_err());
            // A copy cannot straddle two buffers even if they were adjacent.
            assert!(stream.copy(other.ptr(), buffer.ptr().offset(1 << 15), 1 << 16).is_err());
            // Fills are in whole, aligned 32-bit words.
            assert!(stream.fill(buffer.ptr().offset(1), 0, 8).is_err());
            assert!(stream.fill(buffer.ptr(), 0, 6).is_err());
            // Overlapping ranges of one buffer.
            assert!(stream.copy(buffer.ptr().offset(8), buffer.ptr(), 16).is_err());
            // Staging bounds.
            assert!(stream.copy_from_staging(buffer.ptr(), &staging, 1000, 25).is_err());
            assert!(stream.copy_to_staging(&staging, usize::MAX, buffer.ptr(), 2).is_err());
        }
        // Nothing was recorded for any of them.
        assert_eq!(stream.pending_commands(), 0);

        // Adjacent ranges of one buffer are not an overlap.
        // SAFETY: both ranges are inside `buffer`.
        unsafe {
            stream.copy(buffer.ptr().offset(16), buffer.ptr(), 16).unwrap();
            // Zero-length operations do nothing, even on bad addresses.
            stream.copy(outside, outside, 0).unwrap();
            stream.fill(outside, 0, 0).unwrap();
        }
        stream.synchronize().unwrap();
    }
}

#[test]
fn add_and_rms_norm_match_the_cpu() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();

        // Sizes around the block and warp boundaries.
        for n in [1u32, 31, 32, 255, 256, 257, 2560, 10240] {
            let x: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let y: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
            let (dx, dy) = (upload_f32(device, &x), upload_f32(device, &y));
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&stream, dy.ptr(), dx.ptr(), n).unwrap() };
            stream.synchronize().unwrap();
            let got = read_f32(&dy, n as usize);
            let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
            assert_eq!(got, want, "add, n = {n}");
        }

        for (n, rows) in [(32u32, 1u32), (160, 3), (2560, 4), (10240, 2), (1000, 5)] {
            let total = (n * rows) as usize;
            let x: Vec<f32> = (0..total).map(|_| rng.unit() * 3.0).collect();
            let weight: Vec<f32> = (0..n).map(|_| rng.unit() + 1.5).collect();
            let (dx, dw) = (upload_f32(device, &x), upload_f32(device, &weight));
            let dy = DeviceBuffer::new(device, total as u64 * 4).unwrap();
            let eps = 1e-6f32;
            for with_weight in [true, false] {
                let w = if with_weight { dw.ptr() } else { DevicePtr::NULL };
                // SAFETY: the buffers hold `rows * n` and `n` values on this device.
                unsafe {
                    kernels.rms_norm_f32(&stream, dx.ptr(), w, dy.ptr(), n, rows, eps).unwrap();
                }
                stream.synchronize().unwrap();
                let got = read_f32(&dy, total);
                for row in 0..rows as usize {
                    let xr = &x[row * n as usize..][..n as usize];
                    let mean: f64 =
                        xr.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / f64::from(n);
                    let scale = 1.0 / (mean + f64::from(eps)).sqrt();
                    for (i, value) in xr.iter().enumerate() {
                        let w = if with_weight { f64::from(weight[i]) } else { 1.0 };
                        let want = f64::from(*value) * scale * w;
                        let have = f64::from(got[row * n as usize + i]);
                        assert!(
                            (have - want).abs() <= 2e-5 * want.abs().max(1.0),
                            "rms_norm n = {n}, row {row}, value {i}: {have} != {want}"
                        );
                    }
                }
            }
        }
    }
}

/// Normalising in place gives the same result as normalising into another
/// buffer (the kernel documents that `y` may be `x`).
#[test]
fn rms_norm_can_work_in_place() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        let (n, rows) = (2560u32, 6u32);
        let total = (n * rows) as usize;
        let x: Vec<f32> = (0..total).map(|_| rng.unit() * 3.0).collect();
        let weight: Vec<f32> = (0..n).map(|_| rng.unit() + 1.5).collect();
        let (dx, dw) = (upload_f32(device, &x), upload_f32(device, &weight));
        let copy = upload_f32(device, &x);
        let out = DeviceBuffer::new(device, total as u64 * 4).unwrap();
        // SAFETY: the buffers hold `rows * n` and `n` values on this device.
        unsafe {
            kernels.rms_norm_f32(&stream, dx.ptr(), dw.ptr(), out.ptr(), n, rows, 1e-6).unwrap();
            kernels.rms_norm_f32(&stream, copy.ptr(), dw.ptr(), copy.ptr(), n, rows, 1e-6).unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(read_f32(&copy, total), read_f32(&out, total));
    }
}

/// A Q4_K matrix of random blocks with finite scales.
fn random_q4k(rng: &mut Rng, n_in: usize, n_out: usize) -> Vec<u8> {
    // f16 bit patterns of small finite values, both signs.
    const SCALES: [u16; 6] = [0x2e66, 0x3266, 0xae66, 0x1c00, 0x3800, 0xb000];
    let blocks = n_in / 256 * n_out;
    let mut bytes = Vec::with_capacity(blocks * 144);
    for _ in 0..blocks {
        bytes.extend_from_slice(&SCALES[(rng.next() % 6) as usize].to_le_bytes());
        bytes.extend_from_slice(&SCALES[(rng.next() % 6) as usize].to_le_bytes());
        for _ in 0..140 {
            bytes.push((rng.next() >> 32) as u8);
        }
    }
    bytes
}

/// One launch of a Q4_K matrix kernel: stream, weights, activations, scratch,
/// output, columns, rows, activation rows.
type Launch<'a> =
    &'a dyn Fn(&Stream, DevicePtr, DevicePtr, DevicePtr, DevicePtr, u32, u32, u32) -> Result<()>;

/// Runs `launch` on `w` and checks it against decoding the matrix on the CPU
/// and multiplying in f64. The activations are placed `x_offset` bytes into
/// their buffer, which makes them as misaligned as that.
#[allow(clippy::too_many_arguments)]
fn check_matvec_with(
    device: &Device,
    launch: Launch<'_>,
    rng: &mut Rng,
    w: &[u8],
    n_in: usize,
    n_out: usize,
    n_batch: usize,
    x_offset: u64,
) {
    let stream = Stream::new(device).unwrap();
    let x: Vec<f32> = (0..n_in * n_batch).map(|_| rng.unit()).collect();
    let dw = DeviceBuffer::new(device, w.len() as u64).unwrap();
    dw.write(0, w).unwrap();
    let dx = DeviceBuffer::new(device, x_offset + (x.len() * 4) as u64).unwrap();
    dx.write(x_offset, &f32_bytes(&x)).unwrap();
    let scratch = DeviceBuffer::new(device, (n_batch * n_in / 32 * 4) as u64).unwrap();
    let dy = DeviceBuffer::new(device, (n_batch * n_out * 4) as u64).unwrap();
    launch(
        &stream,
        dw.ptr(),
        dx.ptr().offset(x_offset),
        scratch.ptr(),
        dy.ptr(),
        n_in as u32,
        n_out as u32,
        n_batch as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got = read_f32(&dy, n_batch * n_out);

    let mut weights = vec![0f32; n_in * n_out];
    ares_quant::dequantize(GgmlType::Q4_K, w, &mut weights).unwrap();
    for b in 0..n_batch {
        let xb = &x[b * n_in..][..n_in];
        for r in 0..n_out {
            let row = &weights[r * n_in..][..n_in];
            let (mut want, mut magnitude) = (0f64, 0f64);
            for (weight, value) in row.iter().zip(xb) {
                let product = f64::from(*weight) * f64::from(*value);
                want += product;
                magnitude += product.abs();
            }
            let have = f64::from(got[b * n_out + r]);
            // The kernel sums f32 products in another order than this loop.
            assert!(
                (have - want).abs() <= 2e-5 * magnitude + 1e-6,
                "matvec {n_out}x{n_in}, batch row {b}, output {r}, activations {x_offset} bytes \
                 in: {have} != {want} (sum of magnitudes {magnitude})"
            );
        }
    }
}

/// Checks the shipped kernel.
fn check_matvec_q4k(
    device: &Device,
    kernels: &Kernels,
    rng: &mut Rng,
    w: &[u8],
    shape: (usize, usize, usize),
) {
    let (n_in, n_out, n_batch) = shape;
    let launch = |stream: &Stream, w, x, scratch, y, n_in, n_out, n_batch| {
        // SAFETY: the buffers were sized for this shape by the caller.
        unsafe { kernels.matvec_q4_k(stream, w, x, scratch, y, n_in, n_out, n_batch) }
    };
    check_matvec_with(device, &launch, rng, w, n_in, n_out, n_batch, 0);
}

#[test]
fn matvec_q4k_matches_the_cpu_decoder() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x2545f4914f6cdd1d);
    for Gpu { device, kernels } in gpus {
        // One block per row; fewer units than a warp; more units than the
        // widest thread block (17408 columns are 272 units); every batch size.
        for (n_in, n_out, n_batch) in
            [(256, 3, 1), (2560, 67, 1), (2560, 5, 3), (5120, 33, 8), (17408, 9, 2)]
        {
            let w = random_q4k(&mut rng, n_in, n_out);
            check_matvec_q4k(device, kernels, &mut rng, &w, (n_in, n_out, n_batch));
        }

        let stream = Stream::new(device).unwrap();
        let null = DevicePtr::NULL;
        // SAFETY: both calls are refused before anything is launched.
        unsafe {
            assert!(kernels.matvec_q4_k(&stream, null, null, null, null, 300, 1, 1).is_err());
            let too_many = MAX_BATCH + 1;
            assert!(
                kernels.matvec_q4_k(&stream, null, null, null, null, 256, 1, too_many).is_err()
            );
            assert!(kernels.matvec_q4_k(&stream, null, null, null, null, 256, 1, 0).is_err());
        }
        assert_eq!(stream.pending_commands(), 0);
    }
}

/// Every workgroup size the matrix kernel is compiled for (32 to 256 in steps
/// of 32) gets used by some row length, and all of them must give the same
/// answer as the CPU.
#[test]
fn matvec_q4k_is_right_for_every_workgroup_size() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    for Gpu { device, kernels } in gpus {
        // 64 values per invocation: 256 columns want 32 invocations, 3072
        // want 64 (48 rounded up), ... up to 256 for 16384 and more.
        for blocks in [1usize, 5, 8, 11, 16, 20, 24, 28, 32, 40, 64, 80] {
            let n_in = blocks * 256;
            let w = random_q4k(&mut rng, n_in, 7);
            check_matvec_q4k(device, kernels, &mut rng, &w, (n_in, 7, 2));
        }
    }
}

/// The kernel reads activations that are only promised to be 4-byte aligned,
/// with 16-byte loads when they happen to be 16-byte aligned: both paths must
/// agree with the CPU, and so must row counts that are not a multiple of the
/// number of rows a workgroup computes.
#[test]
fn matvec_q4k_accepts_activations_at_any_4_byte_alignment() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x5eed_0fa1_19ed_cafe);
    for Gpu { device, kernels } in gpus {
        for (n_out, n_batch) in [(1usize, 1usize), (5, 8), (6, 3), (9, 8)] {
            let w = random_q4k(&mut rng, 1536, n_out);
            for offset in [0u64, 4, 8, 12, 16, 20] {
                let launch = |stream: &Stream, w, x, scratch, y, n_in, n_out, n_batch| {
                    // SAFETY: the buffers were sized for this shape by the caller.
                    unsafe { kernels.matvec_q4_k(stream, w, x, scratch, y, n_in, n_out, n_batch) }
                };
                check_matvec_with(device, &launch, &mut rng, &w, 1536, n_out, n_batch, offset);
            }
        }
    }
}

/// The port of the CUDA kernel's thread layout, which the benchmark compares
/// the shipped kernel with, loaded through the public `Module` interface (the
/// way a kernel outside this crate would be). It must be right too, or the
/// comparison is meaningless.
#[test]
fn the_port_of_the_cuda_thread_layout_matches_the_cpu_decoder() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0x1234_dead_beef_5678);
    for Gpu { device, .. } in gpus {
        let sum32 =
            Module::load(device, include_bytes!(concat!(env!("OUT_DIR"), "/sum32_f32.spv")))
                .unwrap()
                .function("main")
                .unwrap();
        let matvec = Module::load(
            device,
            include_bytes!(concat!(env!("OUT_DIR"), "/matvec_q4_K_cuda_layout.spv")),
        )
        .unwrap();
        for (n_in, n_out, n_batch) in [(256usize, 3usize, 1usize), (5120, 33, 8), (17408, 9, 2)] {
            let threads = (n_in as u32 / 64).next_multiple_of(32).min(256);
            let function = matvec.function_with("main", &[(0, threads)]).unwrap();
            let launch = |stream: &Stream,
                          w,
                          x,
                          scratch,
                          y,
                          n_in: u32,
                          n_out: u32,
                          n_batch: u32| {
                let n_groups = n_batch * (n_in / 32);
                let args = Args::new().ptr(x).ptr(scratch).u32(n_groups);
                // SAFETY: the argument list matches sum32_f32.comp.
                unsafe { sum32.launch(stream, [n_groups.div_ceil(256), 1, 1], args)? };
                let args =
                    Args::new().ptr(w).ptr(x).ptr(scratch).ptr(y).u32(n_in).u32(n_out).u32(n_batch);
                // SAFETY: the argument list matches the shader's push constants.
                unsafe { function.launch(stream, [n_out, 1, 1], args) }
            };
            let w = random_q4k(&mut rng, n_in, n_out);
            check_matvec_with(device, &launch, &mut rng, &w, n_in, n_out, n_batch, 0);
        }
    }
}

#[test]
fn matvec_q4k_matches_the_cpu_decoder_on_a_real_model() {
    let Some(gpus) = gpus() else { return };
    let Some(path) = std::env::var_os("ARES_TEST_GGUF") else {
        eprintln!("skipped: set ARES_TEST_GGUF=<model.gguf> to check a real Q4_K matrix");
        return;
    };
    let gguf = ares_gguf::Gguf::open(&path).unwrap();
    let tensor = gguf
        .tensors()
        .iter()
        .filter(|t| t.ty == GgmlType::Q4_K && t.dims.as_slice().len() == 2)
        .min_by_key(|t| t.size)
        .expect("the model has no 2-D Q4_K tensor");
    let (n_in, n_out) = (tensor.dims.get(0) as usize, tensor.dims.get(1) as usize);
    eprintln!("checking {} ({n_out} x {n_in})", tensor.name);
    let rows = n_out.min(512);
    let w = &gguf.data(tensor)[..rows * (n_in / 256) * 144];
    let mut rng = Rng(0x853c49e6748fea9b);
    for Gpu { device, kernels } in gpus {
        check_matvec_q4k(device, kernels, &mut rng, w, (n_in, rows, 2));
    }
}

/// Vulkan has no copy between two logical devices, so data between GPUs goes
/// through the host: device-to-staging on the first, then staging-to-device on
/// the second, ordered by the host's `synchronize`. (The CUDA layer's
/// counterpart moves it by peer copy, ordered by an event across devices.)
#[test]
fn data_moves_between_devices_through_the_host() {
    let Some(gpus) = gpus() else { return };
    let [first, second, ..] = gpus else {
        eprintln!("skipped: needs two devices");
        return;
    };
    let len = 1usize << 20;
    let data: Vec<u8> = (0..len).map(|i| (i * 7 % 241) as u8).collect();

    for (src_device, dst_device) in
        [(&first.device, &second.device), (&second.device, &first.device)]
    {
        let src = DeviceBuffer::new(src_device, len as u64).unwrap();
        let dst = DeviceBuffer::new(dst_device, len as u64).unwrap();
        let src_stream = Stream::new(src_device).unwrap();
        let dst_stream = Stream::new(dst_device).unwrap();
        src.write(0, &data).unwrap();
        let mut over_the_host = StagingBuffer::new(src_device, len).unwrap();
        let from_host = StagingBuffer::new(dst_device, len).unwrap();

        // SAFETY: whole buffers of `len` bytes on the named devices, alive
        // until both streams are synchronised below; the staging buffers are
        // not touched while a copy that names them runs.
        unsafe {
            dst_stream.fill(dst.ptr(), 0xee, len).unwrap();
            src_stream.copy_to_staging(&over_the_host, 0, src.ptr(), len).unwrap();
        }
        src_stream.synchronize().unwrap();
        // A staging buffer belongs to one device.
        // SAFETY: the call is refused before anything is recorded.
        let refused = unsafe { dst_stream.copy_from_staging(dst.ptr(), &over_the_host, 0, len) };
        assert!(refused.is_err());
        let mut from_host = from_host;
        from_host.copy_from_slice(&over_the_host);
        over_the_host.fill(0);
        // SAFETY: as above, with the right staging buffer.
        unsafe { dst_stream.copy_from_staging(dst.ptr(), &from_host, 0, len).unwrap() };
        dst_stream.synchronize().unwrap();
        let mut back = vec![0u8; len];
        dst.read(0, &mut back).unwrap();
        assert!(back == data, "{src_device:?} -> {dst_device:?}");
    }

    // Events do not span devices.
    let event = Event::new(&first.device).unwrap();
    let on_first = Stream::new(&first.device).unwrap();
    let on_second = Stream::new(&second.device).unwrap();
    event.record(&on_first).unwrap();
    assert!(on_second.wait_event(&event).is_err());
    assert!(event.record(&on_second).is_err());
}

/// `DeviceBuffer::write` copies from ordinary memory. A kernel launched right
/// after it returns, on any stream, must see the new data in full; the CUDA
/// layer once returned when the driver had staged the data and not when the
/// device had it, and the tail of large uploads read stale.
#[test]
fn a_kernel_launched_right_after_a_write_sees_all_of_it() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, kernels } in gpus {
        // The upload and the clearing are on one stream and the kernel on
        // another, so no stream order can be what makes this work.
        let setup = Stream::new(device).unwrap();
        let compute = Stream::new(device).unwrap();
        let n = 8u32 << 20;
        let bytes = n as usize * 4;
        let x = DeviceBuffer::new(device, bytes as u64).unwrap();
        let y = DeviceBuffer::new(device, bytes as u64).unwrap();
        for round in 0..4u32 {
            // Distinct, representable values, so a misplaced chunk shows up.
            let data: Vec<f32> = (0..n).map(|i| (i % 4093) as f32 + round as f32).collect();
            // SAFETY: whole buffers of `bytes` bytes on this device.
            unsafe {
                setup.fill(x.ptr(), 0, bytes).unwrap();
                setup.fill(y.ptr(), 0, bytes).unwrap();
            }
            setup.synchronize().unwrap();
            x.write(0, &f32_bytes(&data)).unwrap();
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&compute, y.ptr(), x.ptr(), n).unwrap() };
            compute.synchronize().unwrap();
            let got = read_f32(&y, n as usize);
            let stale = got.iter().zip(&data).filter(|(g, d)| g != d).count();
            assert_eq!(
                stale, 0,
                "round {round}: {stale} of {n} values were read before the upload"
            );
        }
    }
}

/// Thousands of small launches in a row: they are recorded into batches, the
/// batches are submitted as they fill, and command buffers are reused. Each
/// launch depends on the previous one (they all add into the same buffer), so
/// a missing barrier or a lost command shows in the sum.
#[test]
fn thousands_of_launches_in_a_row_batch_and_reuse_their_resources() {
    if gpus().is_none() {
        return;
    }
    const LAUNCHES: u32 = 2_000;
    for Gpu { device, kernels } in &isolated() {
        let n = 1024u32;
        let x = upload_f32(device, &vec![1.0; n as usize]);
        // The default batch, a small one that cycles through command buffers
        // and the in-flight limit many times over, and one command per batch.
        for batch in [DEFAULT_BATCH, 7, 1] {
            let stream = Stream::with_batch(device, batch).unwrap();
            let y = upload_f32(device, &vec![0.0; n as usize]);
            let started = Instant::now();
            for _ in 0..LAUNCHES {
                // SAFETY: both buffers hold `n` values on this device and
                // outlive the synchronisation below.
                unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
            }
            let recorded = started.elapsed();
            stream.synchronize().unwrap();
            let total = started.elapsed();
            assert!(stream.is_idle().unwrap());
            assert_eq!(stream.pending_commands(), 0);
            eprintln!(
                "{:?}: {LAUNCHES} launches, batch {batch}: {} submissions, recorded in {:?} \
                 ({:?} each), done after {:?}",
                device,
                stream.submissions(),
                recorded,
                recorded / LAUNCHES,
                total
            );
            // Every `batch` commands one submission, and one more on
            // `synchronize` for the remainder.
            assert_eq!(stream.submissions(), u64::from(LAUNCHES.div_ceil(batch)));
            let got = read_f32(&y, n as usize);
            assert!(
                got.iter().all(|v| *v == LAUNCHES as f32),
                "batch {batch}: {} of {n} sums are not {LAUNCHES}",
                got.iter().filter(|v| **v != LAUNCHES as f32).count()
            );
        }
    }
}

/// Launches of different kernels in a row, so the pipeline changes between
/// commands (and is rebound), checked against the CPU.
#[test]
fn alternating_kernels_in_one_batch_give_the_right_answers() {
    let Some(gpus) = gpus() else { return };
    let mut rng = Rng(0xfeed_face_0bad_cafe);
    for Gpu { device, kernels } in gpus {
        let stream = Stream::new(device).unwrap();
        let (n, rows) = (512u32, 3u32);
        let total = (n * rows) as usize;
        let x: Vec<f32> = (0..total).map(|_| rng.unit()).collect();
        let step: Vec<f32> = (0..total).map(|_| rng.unit() * 0.01).collect();
        let dx = upload_f32(device, &x);
        let dstep = upload_f32(device, &step);
        let rounds = 40;
        // x = rms_norm(x + step), `rounds` times, with no host involvement.
        for _ in 0..rounds {
            // SAFETY: the buffers hold `rows * n` values on this device.
            unsafe {
                kernels.add_f32(&stream, dx.ptr(), dstep.ptr(), total as u32).unwrap();
                kernels
                    .rms_norm_f32(&stream, dx.ptr(), DevicePtr::NULL, dx.ptr(), n, rows, 1e-6)
                    .unwrap();
            }
        }
        stream.synchronize().unwrap();
        let got = read_f32(&dx, total);

        let mut want: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        for _ in 0..rounds {
            for row in 0..rows as usize {
                let r = &mut want[row * n as usize..][..n as usize];
                for (v, s) in r.iter_mut().zip(&step[row * n as usize..]) {
                    *v += f64::from(*s);
                }
                let mean = r.iter().map(|v| v * v).sum::<f64>() / f64::from(n);
                let scale = 1.0 / (mean + 1e-6).sqrt();
                r.iter_mut().for_each(|v| *v *= scale);
            }
        }
        for (i, (have, want)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*have) - want).abs() <= 1e-4 * want.abs().max(1.0),
                "value {i}: {have} != {want}"
            );
        }
    }
}

#[test]
fn recorded_work_is_not_lost_when_the_stream_is_dropped_or_a_buffer_is() {
    if gpus().is_none() {
        return;
    }
    for Gpu { device, kernels } in &isolated() {
        let n = 1u32 << 16;
        let x = upload_f32(device, &vec![1.0; n as usize]);
        let y = upload_f32(device, &vec![0.0; n as usize]);
        {
            let stream = Stream::new(device).unwrap();
            for _ in 0..10 {
                // SAFETY: both buffers hold `n` values on this device.
                unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
            }
            assert_eq!(stream.pending_commands(), 10);
            // Dropped without flush or synchronize.
        }
        device.synchronize().unwrap();
        assert!(read_f32(&y, n as usize).iter().all(|v| *v == 10.0));

        // `Device::synchronize` flushes streams that still hold recorded work.
        let stream = Stream::new(device).unwrap();
        // SAFETY: as above.
        unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
        assert_eq!(stream.pending_commands(), 1);
        device.synchronize().unwrap();
        assert_eq!(stream.pending_commands(), 0);
        assert!(stream.is_idle().unwrap());
        assert!(read_f32(&y, n as usize).iter().all(|v| *v == 11.0));

        // Freeing a buffer first runs the commands that name it, so this
        // neither crashes nor reads freed memory.
        let temporary = upload_f32(device, &vec![2.0; n as usize]);
        for _ in 0..5 {
            // SAFETY: both buffers hold `n` values; `temporary` is dropped
            // below, which waits for the launches.
            unsafe { kernels.add_f32(&stream, y.ptr(), temporary.ptr(), n).unwrap() };
        }
        drop(temporary);
        stream.synchronize().unwrap();
        assert!(read_f32(&y, n as usize).iter().all(|v| *v == 21.0));
    }
}

#[test]
fn events_order_streams_of_one_device_and_time_work() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, kernels } in gpus {
        let producer = Stream::new(device).unwrap();
        let consumer = Stream::new(device).unwrap();
        let n = 1u32 << 22;
        let x = upload_f32(device, &vec![1.0; n as usize]);
        let y = upload_f32(device, &vec![0.0; n as usize]);
        let z = DeviceBuffer::new(device, u64::from(n) * 4).unwrap();

        // A never-recorded event is done, and waiting for it does nothing.
        let never = Event::new(device).unwrap();
        assert!(never.is_done().unwrap());
        never.synchronize().unwrap();
        consumer.wait_event(&never).unwrap();

        let (start, stop) = (Event::timed(device).unwrap(), Event::timed(device).unwrap());
        let ready = Event::new(device).unwrap();
        start.record(&producer).unwrap();
        let rounds = 200;
        for _ in 0..rounds {
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&producer, y.ptr(), x.ptr(), n).unwrap() };
        }
        stop.record(&producer).unwrap();
        ready.record(&producer).unwrap();

        // The consumer copies `y` once the producer has finished adding.
        consumer.wait_event(&ready).unwrap();
        // SAFETY: whole buffers of the same size.
        unsafe { consumer.copy(z.ptr(), y.ptr(), n as usize * 4).unwrap() };
        consumer.synchronize().unwrap();
        assert!(ready.is_done().unwrap());
        assert!(read_f32(&z, n as usize).iter().all(|v| *v == rounds as f32));

        let ms = stop.elapsed_ms_since(&start).unwrap();
        // 200 launches over 16 MiB of reads and 16 MiB of writes: at least
        // some microseconds each, and not longer than the host waited.
        eprintln!("{device:?}: {rounds} launches of {n} values took {ms:.3} ms on the device");
        assert!(ms > 0.05 && ms < 5000.0, "{ms}");
        // Events made without timing refuse to be timed.
        assert!(ready.elapsed_ms_since(&start).is_err());
        assert!(never.elapsed_ms_since(&start).is_err());

        // An event can be recorded again; it then means the new point.
        ready.record(&consumer).unwrap();
        ready.synchronize().unwrap();
        assert!(ready.is_done().unwrap());
    }
}

#[test]
fn streams_can_be_recorded_from_several_threads() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, kernels } in gpus {
        let n = 4096u32;
        let shared = Stream::new(device).unwrap();
        let threads = 4usize;
        let per_thread = 300u32;
        let x = upload_f32(device, &vec![1.0; n as usize]);
        let outputs: Vec<DeviceBuffer> =
            (0..threads).map(|_| upload_f32(device, &vec![0.0; n as usize])).collect();
        std::thread::scope(|scope| {
            for output in &outputs {
                let (shared, x) = (&shared, &x);
                scope.spawn(move || {
                    // Each thread has its own stream as well as the shared one.
                    let own = Stream::new(device).unwrap();
                    for i in 0..per_thread {
                        let stream = if i % 2 == 0 { shared } else { &own };
                        // SAFETY: both buffers hold `n` values on this device
                        // and outlive the synchronisation below.
                        unsafe { kernels.add_f32(stream, output.ptr(), x.ptr(), n).unwrap() };
                    }
                    own.synchronize().unwrap();
                });
            }
        });
        shared.synchronize().unwrap();
        for output in &outputs {
            assert!(read_f32(output, n as usize).iter().all(|v| *v == per_thread as f32));
        }
    }
}

#[test]
fn kernels_can_be_loaded_again_and_the_modules_check_their_input() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, .. } in gpus {
        let started = Instant::now();
        let again = Kernels::load(device).unwrap();
        eprintln!("{device:?}: Kernels::load took {:?}", started.elapsed());
        assert_eq!(again.device().ordinal(), device.ordinal());

        assert!(ares_vulkan::Module::load(device, &[0u8; 64]).is_err());
        assert!(ares_vulkan::Module::load(device, b"abc").is_err());
        let image = std::fs::read(concat!(env!("OUT_DIR"), "/add_f32.spv")).unwrap();
        let module = ares_vulkan::Module::load(device, &image).unwrap();
        assert!(matches!(module.function("nope"), Err(Error::NoSuchKernel(_))));
        assert!(module.function("main").is_ok());
    }
}

/// Threads can wait for the device by blocking in the driver, by polling, or by
/// polling and yielding; every way must see the work finish.
#[test]
fn every_wait_mode_sees_the_work_finish() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device: shared, .. } in gpus {
        for wait in [Wait::Block, Wait::Spin, Wait::Yield] {
            let device = Device::open_with(shared.ordinal(), wait).unwrap();
            let kernels = Kernels::load(&device).unwrap();
            let stream = Stream::new(&device).unwrap();
            let n = 1u32 << 20;
            let x = upload_f32(&device, &vec![1.0; n as usize]);
            let y = upload_f32(&device, &vec![0.0; n as usize]);
            for _ in 0..20 {
                // SAFETY: both buffers hold `n` values on this device.
                unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
            }
            stream.synchronize().unwrap();
            assert!(stream.is_idle().unwrap());
            assert!(read_f32(&y, n as usize).iter().all(|v| *v == 20.0), "{wait:?}");
            let event = Event::new(&device).unwrap();
            event.record(&stream).unwrap();
            event.synchronize().unwrap();
        }
    }
}

/// A host that records far ahead of the device is held back by the limit on
/// batches in flight instead of piling up command buffers; nothing is lost.
#[test]
fn a_host_far_ahead_of_the_device_is_slowed_down_and_loses_nothing() {
    let Some(gpus) = gpus() else { return };
    for Gpu { device, kernels } in gpus {
        // One command per batch, each a kernel that takes the device a while,
        // so the limit of 32 unfinished batches is reached over and over.
        let stream = Stream::with_batch(device, 1).unwrap();
        let n = 1u32 << 22;
        let x = upload_f32(device, &vec![1.0; n as usize]);
        let y = upload_f32(device, &vec![0.0; n as usize]);
        let launches = 5 * ares_vulkan::MAX_IN_FLIGHT as u32;
        for _ in 0..launches {
            // SAFETY: both buffers hold `n` values on this device.
            unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), n).unwrap() };
        }
        stream.synchronize().unwrap();
        assert_eq!(stream.submissions(), u64::from(launches));
        assert!(read_f32(&y, n as usize).iter().all(|v| *v == launches as f32));
    }
}
