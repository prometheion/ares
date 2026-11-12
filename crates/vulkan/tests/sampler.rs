//! Tests of the sampler's kernels (`SamplerKernels`) against the same arithmetic on the host
//! (`ares_cuda::sample_reference`, which CUDA's kernels are held to by `crates/cuda/tests/sampler.rs`).
//!
//! They open the GPUs, so they run only with `ARES_TEST_VULKAN=1`:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test --release -p ares-vulkan --test sampler -- --test-threads=1 --nocapture
//! ```
//!
//! What is checked: every record the device writes (pick, score, status and
//! candidates) equals the host's bit for bit, for vocabularies from one token
//! to the size of Qwen's, for rows that are peaked, flat, tied and partly
//! masked, for every combination of the cuts, with a token map, and with the
//! vocabulary cut into slices as several devices hold it (the exchange between
//! the slices is added on the host here). The time of a pick is printed.

use std::sync::OnceLock;
use std::time::Instant;

use ares_vulkan::{
    Device, DeviceBuffer, SAMPLE_STATUS_NONFINITE, SampleBatch, SampleParams, SamplePick,
    SamplePlan, SampleRow, SamplerKernels, Stream, device_count,
};

/// The host's definition of the pick of one row (`ares_cuda::sample_reference`), for a row of
/// this crate's settings: the records of the two crates are the same bytes.
fn sample_reference(
    logits: &[f32],
    token_of: impl Fn(usize) -> u32,
    row: &SampleRow,
) -> ares_cuda::SamplePick {
    ares_cuda::sample_reference(
        logits,
        token_of,
        &ares_cuda::SampleRow::from_bytes(&row.to_bytes()),
    )
}

/// The devices every test shares, opened once and never closed: the process leaves through
/// [`leave`].
static DEVICES: OnceLock<Option<&'static [Device]>> = OnceLock::new();

/// Ends the process at once when it exits (see `tests/layers.rs`): the NVIDIA driver crashes
/// when a device is destroyed or its exit handlers run while a compiler thread still works.
extern "C" fn leave(status: i32, _: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    // SAFETY: `_exit` takes any status and does not return.
    unsafe { _exit(status) }
}

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

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.unit() + 1e-7;
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The shapes of rows the sampler must get right.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// As a model's: most tokens far below a few likely ones.
    Peaked,
    /// Every token about as likely.
    Flat,
    /// Few distinct values, so every cut falls on a tie.
    Tied,
    /// Some tokens masked with minus infinity, some logits negative zero.
    Masked,
}

fn row(rng: &mut Rng, n: usize, shape: Shape) -> Vec<f32> {
    let mut logits: Vec<f32> = match shape {
        Shape::Peaked => (0..n).map(|_| rng.normal() * 2.5 - 4.0).collect(),
        Shape::Flat => (0..n).map(|_| rng.normal() * 0.01).collect(),
        Shape::Tied => (0..n).map(|_| (rng.normal() * 3.0).round() * 0.5).collect(),
        Shape::Masked => (0..n)
            .map(|_| match rng.below(4) {
                0 => f32::NEG_INFINITY,
                1 => -0.0,
                2 => 0.0,
                _ => rng.normal() * 2.0,
            })
            .collect(),
    };
    if matches!(shape, Shape::Peaked) {
        // A handful of likely tokens, as after a prompt.
        for _ in 0..n.min(12) {
            let at = rng.below(n);
            logits[at] = 8.0 + rng.normal() * 3.0;
        }
    }
    logits
}

fn settings() -> Vec<SampleParams> {
    let p = |temperature, top_k, top_p, min_p, seed| SampleParams {
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
    };
    vec![
        SampleParams::GREEDY,
        p(1.0, 0, 1.0, 0.0, 1),
        p(0.7, 0, 1.0, 0.0, 2),
        p(1.0, 1, 1.0, 0.0, 3),
        p(1.0, 20, 1.0, 0.0, 4),
        p(1.0, 1000, 1.0, 0.0, 5),
        p(1.0, 10_000_000, 1.0, 0.0, 6),
        p(1.0, 0, 0.1, 0.0, 7),
        p(1.0, 0, 0.95, 0.0, 8),
        p(1.0, 0, 0.9999, 0.0, 9),
        p(1.0, 0, 1.0, 0.05, 10),
        p(1.0, 0, 1.0, 0.9, 11),
        p(1.0, 20, 0.95, 0.0, 0xdead_beef_0123_4567),
        p(1.3, 40, 0.9, 0.02, 13),
        p(0.2, 3, 0.5, 0.5, 14),
        p(4.0, 64, 0.99, 0.001, 15),
    ]
}

/// The settings of a request become the bytes CUDA's kernels read (no device needed).
#[test]
fn settings_become_the_records_cuda_reads() {
    for (i, p) in settings().iter().enumerate() {
        let cuda = ares_cuda::SampleParams {
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
            seed: p.seed,
        };
        let position = 1000 * i as u32 + 7;
        assert_eq!(p.row(position).to_bytes(), cuda.row(position).to_bytes(), "{p:?}");
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The buffers of one slice of the vocabulary on a device.
struct Slice {
    logits: DeviceBuffer,
    tokens: Option<DeviceBuffer>,
    params: DeviceBuffer,
    out: DeviceBuffer,
    scratch: DeviceBuffer,
    exchange: DeviceBuffer,
    batch: SampleBatch,
}

impl Slice {
    /// Columns `range` of `rows` (each of the whole vocabulary) as device
    /// `rank` of `devices` holds them. With `map`, column `i` is token
    /// `map[i]`; without, token `i`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &Device,
        stream: &Stream,
        kernels: &SamplerKernels,
        rows: &[Vec<f32>],
        params: &[SampleRow],
        range: std::ops::Range<usize>,
        map: Option<&[u32]>,
        devices: u32,
        rank: u32,
    ) -> Self {
        let n_rows = rows.len() as u32;
        let n_local = range.len();
        let mut flat = Vec::with_capacity(rows.len() * n_local);
        for row in rows {
            flat.extend_from_slice(&row[range.clone()]);
        }
        let logits = DeviceBuffer::new(device, (flat.len() * 4) as u64).unwrap();
        logits.write(0, &f32_bytes(&flat)).unwrap();
        let tokens = map.map(|map| {
            let buffer = DeviceBuffer::new(device, (map.len() * 4) as u64).unwrap();
            buffer.write(0, &u32_bytes(map)).unwrap();
            buffer
        });
        let params_buffer =
            DeviceBuffer::new(device, (params.len() * SampleRow::BYTES) as u64).unwrap();
        let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_bytes()).collect();
        params_buffer.write(0, &bytes).unwrap();
        let out = DeviceBuffer::new(device, (rows.len() * SamplePick::BYTES) as u64).unwrap();
        let scratch = DeviceBuffer::new(device, SamplerKernels::scratch_bytes(n_rows)).unwrap();
        let exchange = DeviceBuffer::new(
            device,
            u64::from(SamplerKernels::exchange_values(n_rows, devices)) * 4,
        )
        .unwrap();
        // SAFETY: the scratch buffer has the size the sampler asks for.
        unsafe { kernels.clear(stream, scratch.ptr(), n_rows).unwrap() };
        let batch = SampleBatch {
            logits: logits.ptr(),
            stride: n_local as u32,
            n_rows,
            capacity: n_rows,
            n_local: n_local as u32,
            tokens: tokens.as_ref().map_or(ares_vulkan::DevicePtr::NULL, DeviceBuffer::ptr),
            first: range.start as u32,
            params: params_buffer.ptr(),
            out: out.ptr(),
            scratch: scratch.ptr(),
            exchange: exchange.ptr(),
            devices,
            rank,
        };
        Self { logits, tokens, params: params_buffer, out, scratch, exchange, batch }
    }

    fn picks(&self) -> Vec<SamplePick> {
        let mut bytes = vec![0u8; self.out.len() as usize];
        self.out.read(0, &mut bytes).unwrap();
        bytes
            .as_chunks::<{ SamplePick::BYTES }>()
            .0
            .iter()
            .map(|b| SamplePick::from_bytes(b))
            .collect()
    }

    fn keep(&self) -> usize {
        // The buffers live as long as the slice; this only names them.
        [&self.logits, &self.params, &self.scratch].iter().map(|b| b.len() as usize).sum::<usize>()
            + self.tokens.as_ref().map_or(0, |t| t.len() as usize)
    }
}

/// The picks of `rows` with the vocabulary cut at `cuts` into slices, each
/// run as a device of its own; the exchange between them is added here.
/// Returns the records of every slice.
fn run(
    device: &Device,
    kernels: &SamplerKernels,
    rows: &[Vec<f32>],
    params: &[SampleRow],
    cuts: &[usize],
    map: Option<&[u32]>,
) -> Vec<Vec<SamplePick>> {
    let stream = Stream::new(device).unwrap();
    let n = rows[0].len();
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(n);
    let devices = (bounds.len() - 1) as u32;
    let slices: Vec<Slice> = bounds
        .windows(2)
        .enumerate()
        .map(|(rank, w)| {
            Slice::new(
                device,
                &stream,
                kernels,
                rows,
                params,
                w[0]..w[1],
                map,
                devices,
                rank as u32,
            )
        })
        .collect();
    let plan = SamplePlan::of(params);
    for step in plan.steps(devices) {
        let mut values = 0;
        for slice in &slices {
            // SAFETY: the slice's buffers have the sizes the batch documents.
            values = unsafe { kernels.enqueue(&stream, &slice.batch, step).unwrap() };
        }
        stream.synchronize().unwrap();
        if values > 0 {
            let mut sum = vec![0f32; values as usize];
            let mut bytes = vec![0u8; values as usize * 4];
            for slice in &slices {
                slice.exchange.read(0, &mut bytes).unwrap();
                for (s, b) in sum.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *s += f32::from_le_bytes(*b);
                }
            }
            let bytes = f32_bytes(&sum);
            for slice in &slices {
                slice.exchange.write(0, &bytes).unwrap();
            }
        }
    }
    stream.synchronize().unwrap();
    assert!(slices.iter().all(|s| s.keep() > 0));
    slices.iter().map(Slice::picks).collect()
}

fn same(a: &SamplePick, b: &ares_cuda::SamplePick) -> bool {
    a.token == b.token
        && a.status == b.status
        && a.value.to_bits() == b.value.to_bits()
        && a.cand_token == b.cand_token
        && a.cand_logit.map(f32::to_bits) == b.cand_logit.map(f32::to_bits)
}

/// Checks every slice's records of `rows` against the host.
fn check(
    device: &Device,
    kernels: &SamplerKernels,
    rows: &[Vec<f32>],
    params: &[SampleRow],
    cuts: &[usize],
    map: Option<&[u32]>,
    what: &str,
) {
    let got = run(device, kernels, rows, params, cuts, map);
    for (r, (row, p)) in rows.iter().zip(params).enumerate() {
        let token_of = |i: usize| map.map_or(i as u32, |m| m[i]);
        let want = sample_reference(row, token_of, p);
        for (slice, picks) in got.iter().enumerate() {
            assert!(
                same(&picks[r], &want),
                "{what}: row {r} ({} tokens), slice {slice} of {}, {p:?}:\n device {:?}\n host   {want:?}",
                row.len(),
                got.len(),
                picks[r]
            );
        }
    }
}

#[test]
fn picks_equal_the_host_arithmetic() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    let all = settings();
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let mut checked = 0;
        for n in [1usize, 2, 7, 255, 4096, 4097, 50_000, 248_320] {
            for shape in [Shape::Peaked, Shape::Flat, Shape::Tied, Shape::Masked] {
                // Every setting once per size and shape, a few rows per batch
                // with different settings.
                for batch in all.chunks(4) {
                    let rows: Vec<Vec<f32>> =
                        batch.iter().map(|_| row(&mut rng, n, shape)).collect();
                    let params: Vec<SampleRow> =
                        batch.iter().map(|p| p.row(rng.below(100_000) as u32)).collect();
                    check(device, &kernels, &rows, &params, &[], None, &format!("{shape:?}"));
                    checked += rows.len();
                }
            }
        }
        println!("{}: {checked} rows equal the host's picks", device.info().name);
    }
}

#[test]
fn a_token_map_and_slices_give_the_same_picks() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0002);
    let all = settings();
    let device = &devices[0];
    let kernels = SamplerKernels::load(device).unwrap();
    for n in [9usize, 5000, 248_320] {
        // A permutation: column i is token map[i].
        let mut map: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            map.swap(i, rng.below(i + 1));
        }
        for shape in [Shape::Peaked, Shape::Tied, Shape::Masked, Shape::Flat] {
            for batch in all.chunks(8) {
                let rows: Vec<Vec<f32>> = batch.iter().map(|_| row(&mut rng, n, shape)).collect();
                let params: Vec<SampleRow> =
                    batch.iter().map(|p| p.row(rng.below(100_000) as u32)).collect();
                let what = format!("{shape:?}");
                check(device, &kernels, &rows, &params, &[], Some(&map), &what);
                // Two slices as the tensor split cuts the head, three uneven ones.
                check(device, &kernels, &rows, &params, &[n * 5 / 8], Some(&map), &what);
                check(device, &kernels, &rows, &params, &[n * 5 / 8], None, &what);
                check(device, &kernels, &rows, &params, &[n / 9, n / 2], Some(&map), &what);
            }
        }
    }
}

/// Batches of different sizes one after another in the same scratch area, as
/// the steps of an engine are: a step of four rows, then one row, then three.
/// (The histograms of a row must be where the batch before left them clear.)
#[test]
fn batches_of_different_sizes_share_one_scratch() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0004);
    let device = &devices[0];
    let kernels = SamplerKernels::load(device).unwrap();
    let stream = Stream::new(device).unwrap();
    // The whole vocabulary: with fewer logits the chunk lists of a batch are
    // too short to reach where the histograms of a smaller batch would lie
    // if their place depended on the batch.
    let n = 248_320;
    let capacity = 4;
    let sampled = SampleParams { temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 4711 };
    let rows: Vec<Vec<f32>> = (0..capacity).map(|_| row(&mut rng, n, Shape::Peaked)).collect();
    let params = vec![SampleParams::GREEDY.row(0); capacity];
    let slice = Slice::new(device, &stream, &kernels, &rows, &params, 0..n, None, 1, 0);
    let mut checked = 0;
    for round in 0..12 {
        let k = [4, 1, 3, 1, 2, 4][round % 6];
        let rows: Vec<Vec<f32>> = (0..k).map(|_| row(&mut rng, n, Shape::Peaked)).collect();
        // Rows that are sampled and rows that are not, in turn.
        let params: Vec<SampleRow> = (0..k)
            .map(|r| {
                let setting = if (round + r) % 3 == 0 { SampleParams::GREEDY } else { sampled };
                setting.row((round * 10 + r) as u32)
            })
            .collect();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        stream.synchronize().unwrap();
        slice.logits.write(0, &f32_bytes(&flat)).unwrap();
        let bytes: Vec<u8> = params.iter().flat_map(|p| p.to_bytes()).collect();
        slice.params.write(0, &bytes).unwrap();
        let batch = SampleBatch { n_rows: k as u32, ..slice.batch };
        // SAFETY: the slice's buffers hold `capacity` rows and `k` is at most that.
        unsafe {
            kernels
                .sample(&stream, &batch, SamplePlan::of(&params), |_, _| {
                    Ok::<(), ares_vulkan::Error>(())
                })
                .unwrap();
        }
        stream.synchronize().unwrap();
        let got = slice.picks();
        for (r, (logits, p)) in rows.iter().zip(&params).enumerate() {
            let want = sample_reference(logits, |i| i as u32, p);
            assert!(same(&got[r], &want), "round {round}, row {r} of {k}: {:?}\n{want:?}", got[r]);
            checked += 1;
        }
    }
    println!("{checked} rows in batches of 1 to 4 rows in one scratch equal the host's picks");
}

#[test]
fn a_logit_that_is_not_a_number_is_reported() {
    let Some(devices) = gpus() else { return };
    let device = &devices[0];
    let kernels = SamplerKernels::load(device).unwrap();
    let mut rng = Rng(77);
    for bad in [f32::NAN, f32::INFINITY] {
        let mut logits = row(&mut rng, 10_000, Shape::Peaked);
        logits[6789] = bad;
        let rows = vec![logits.clone(), row(&mut rng, 10_000, Shape::Peaked)];
        let params = vec![SampleParams::GREEDY.row(0); 2];
        for cuts in [vec![], vec![5000]] {
            let got = run(device, &kernels, &rows, &params, &cuts, None);
            for picks in &got {
                assert_eq!(picks[0].status, SAMPLE_STATUS_NONFINITE, "{bad}");
                assert_eq!(picks[1].status, 0, "the other row is not touched");
            }
        }
    }
}

#[test]
fn the_time_of_a_pick() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0003);
    let n = 248_320;
    let greedy = SampleParams::GREEDY;
    let sampled = SampleParams { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 9 };
    let top_p_only = SampleParams { temperature: 1.0, top_k: 0, top_p: 0.95, min_p: 0.0, seed: 9 };
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let stream = Stream::new(device).unwrap();
        for (name, setting) in
            [("greedy", greedy), ("top-k 20, top-p 0.95", sampled), ("top-p 0.95", top_p_only)]
        {
            for n_rows in [1usize, 4, 16] {
                for shape in [Shape::Peaked, Shape::Flat] {
                    let rows: Vec<Vec<f32>> =
                        (0..n_rows).map(|_| row(&mut rng, n, shape)).collect();
                    let params: Vec<SampleRow> =
                        (0..n_rows).map(|r| setting.row(r as u32)).collect();
                    let slice =
                        Slice::new(device, &stream, &kernels, &rows, &params, 0..n, None, 1, 0);
                    let plan = SamplePlan::of(&params);
                    let mut times = Vec::new();
                    for _ in 0..40 {
                        stream.synchronize().unwrap();
                        let begun = Instant::now();
                        // SAFETY: the slice's buffers have the sizes the batch documents.
                        unsafe {
                            kernels
                                .sample(&stream, &slice.batch, plan, |_, _| {
                                    Ok::<(), ares_vulkan::Error>(())
                                })
                                .unwrap();
                        }
                        stream.synchronize().unwrap();
                        times.push(begun.elapsed().as_secs_f64() * 1e6);
                    }
                    times.sort_by(f64::total_cmp);
                    println!(
                        "{}: {name}, {n_rows} rows of {n} ({shape:?}): {:.0} us median, {:.0} us fastest, {} launches",
                        device.info().name,
                        times[times.len() / 2],
                        times[0],
                        plan.steps(1).len()
                    );
                    assert!(slice.keep() > 0);
                }
            }
        }
    }
}
