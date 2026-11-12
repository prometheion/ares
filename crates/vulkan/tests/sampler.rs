//! Tests of the sampler's kernels (`SamplerKernels`) against the reference of
//! the `ares-sampler` crate: the twin of `crates/cuda/tests/sampler.rs`.
//!
//! They open the GPUs, so they run only with `ARES_TEST_VULKAN=1`:
//!
//! ```sh
//! ARES_TEST_VULKAN=1 cargo test -p ares-vulkan --test sampler --release -- --test-threads=1 --nocapture
//! ```
//!
//! What is checked: every record the device writes (pick, score, status,
//! candidates, the surprise of a mirostat row and the probabilities a row
//! reports) equals the host's bit for bit, for vocabularies from one token to
//! the size of Qwen's, for rows that are peaked, flat, tied and partly masked,
//! for every stage of the chain alone and together in several orders, with
//! corrections and masks, with a token map, and with the vocabulary cut into
//! slices as several devices hold it (the exchange between the slices is added
//! on the host here). The time of a pick is printed.

use std::sync::OnceLock;
use std::time::Instant;

use ares_sampler::arith;
use ares_sampler::{
    Entry, LogprobOut, Logprobs, Mirostat, Order, SAMPLE_STATUS_EMPTY, SAMPLE_STATUS_NONFINITE,
    SampleParams, SamplePick, SampleRow, Stage, TokenMask, sample_reference,
};
use ares_vulkan::{
    Args, Device, DeviceBuffer, DevicePtr, Module, SampleBatch, SamplePlan, SamplerKernels,
    Segment, StepRecord, Stream, device_count,
};

/// Ends the process at once when it exits, with the status it exits with: no exit handler of
/// the driver runs and no device is destroyed. The NVIDIA driver compiles pipelines on threads
/// of its own, and it crashes there when a device is destroyed or its exit handlers run while
/// one of them still works. The kernel frees what the process held.
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
    static DEVICES: OnceLock<Option<&'static [Device]>> = OnceLock::new();
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

/// The settings every kind of row is run with, for a vocabulary of `vocab`.
fn settings(vocab: u32) -> Vec<SampleParams> {
    let mut n = 0u64;
    let mut p = move |f: &dyn Fn(&mut SampleParams)| {
        n += 1;
        let mut params =
            SampleParams { temperature: 1.0, seed: n * 0x9e37_79b9, vocab, ..SampleParams::GREEDY };
        f(&mut params);
        params
    };
    let order = |stages: &[Stage]| Order::new(stages).unwrap();
    let lp = |top, after_chain| Logprobs { top, chosen: true, after_chain };
    vec![
        SampleParams::GREEDY,
        p(&|_| {}),
        p(&|s| s.temperature = 0.7),
        p(&|s| s.top_k = 1),
        p(&|s| s.top_k = 20),
        p(&|s| s.top_k = 1000),
        p(&|s| s.top_k = 10_000_000),
        p(&|s| s.top_p = 0.1),
        p(&|s| s.top_p = 0.95),
        p(&|s| s.top_p = 0.9999),
        p(&|s| s.min_p = 0.05),
        p(&|s| s.min_p = 0.9),
        p(&|s| (s.top_k, s.top_p, s.seed) = (20, 0.95, 0xdead_beef_0123_4567)),
        p(&|s| (s.temperature, s.top_k, s.top_p, s.min_p) = (1.3, 40, 0.9, 0.02)),
        p(&|s| (s.temperature, s.top_k, s.top_p, s.min_p) = (0.2, 3, 0.5, 0.5)),
        p(&|s| (s.temperature, s.top_k, s.top_p, s.min_p) = (4.0, 64, 0.99, 0.001)),
        // The stages beyond the cuts of the default chain.
        p(&|s| s.typical_p = 0.8),
        p(&|s| (s.typical_p, s.top_p, s.temperature) = (0.95, 0.9, 0.6)),
        p(&|s| s.top_n_sigma = 1.5),
        p(&|s| (s.top_n_sigma, s.top_k, s.temperature) = (3.0, 100, 0.8)),
        p(&|s| (s.xtc_probability, s.xtc_threshold) = (1.0, 0.1)),
        p(&|s| (s.xtc_probability, s.xtc_threshold, s.top_p) = (0.5, 0.2, 0.9)),
        p(&|s| (s.dynatemp_range, s.dynatemp_exponent) = (0.5, 1.0)),
        p(&|s| {
            (s.dynatemp_range, s.dynatemp_exponent, s.top_k, s.temperature) = (0.8, 1.7, 40, 0.9)
        }),
        p(&|s| (s.temperature, s.dynatemp_range) = (0.0, 0.6)),
        // Orders: the temperature before the cuts, a cut twice.
        p(&|s| {
            (s.top_k, s.top_p, s.min_p, s.temperature) = (300, 0.9, 0.02, 1.7);
            s.order = order(&[Stage::Temperature, Stage::TopK, Stage::TopP, Stage::MinP]);
        }),
        p(&|s| {
            (s.top_p, s.min_p, s.typical_p, s.temperature) = (0.9, 0.02, 0.7, 0.8);
            s.order = order(&[Stage::MinP, Stage::TopP, Stage::Typical, Stage::Temperature]);
        }),
        p(&|s| {
            (s.top_k, s.top_p, s.temperature, s.xtc_probability) = (200, 0.9, 1.4, 1.0);
            s.order = order(&[
                Stage::TopK,
                Stage::Temperature,
                Stage::Xtc,
                Stage::TopP,
                Stage::Temperature,
            ]);
        }),
        p(&|s| {
            (s.top_n_sigma, s.xtc_probability, s.xtc_threshold, s.min_p) = (2.5, 1.0, 0.15, 0.01);
            s.order = order(&[Stage::Xtc, Stage::TopNSigma, Stage::MinP, Stage::Temperature]);
        }),
        // Mirostat.
        p(&|s| (s.mirostat, s.mirostat_mu, s.temperature) = (Some(Mirostat::V2), 6.0, 0.8)),
        p(&|s| (s.mirostat, s.mirostat_mu, s.temperature) = (Some(Mirostat::V2), 2.5, 1.0)),
        p(&|s| (s.mirostat, s.mirostat_mu, s.temperature) = (Some(Mirostat::V1), 8.0, 0.8)),
        p(&|s| (s.mirostat, s.mirostat_mu, s.temperature) = (Some(Mirostat::V1), 3.0, 1.0)),
        // Probabilities.
        p(&|s| s.logprobs = lp(5, false)),
        p(&|s| (s.top_k, s.top_p, s.logprobs) = (40, 0.9, lp(20, false))),
        p(&|s| (s.top_k, s.top_p, s.logprobs) = (40, 0.9, lp(8, true))),
        p(&|s| (s.min_p, s.temperature, s.logprobs) = (0.1, 0.5, lp(20, true))),
        p(&|s| s.logprobs = Logprobs { top: 0, chosen: true, after_chain: false }),
        SampleParams { logprobs: lp(3, false), ..SampleParams::GREEDY },
        SampleParams { logprobs: lp(3, true), ..SampleParams::GREEDY },
        p(&|s| (s.mirostat, s.mirostat_mu, s.logprobs) = (Some(Mirostat::V2), 5.0, lp(4, true))),
    ]
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// What the rows of a batch carry besides their settings.
#[derive(Clone, Default)]
struct Extras {
    /// The corrections of each row.
    entries: Vec<Vec<Entry>>,
    /// The mask of each row.
    masks: Vec<Option<TokenMask>>,
}

impl Extras {
    fn none(rows: usize) -> Self {
        Extras { entries: vec![Vec::new(); rows], masks: vec![None; rows] }
    }

    /// The rows with their corrections and masks named, and the flat arrays
    /// of both as the step record holds them.
    fn attach(&self, params: &[SampleRow], words: usize) -> (Vec<SampleRow>, Vec<Entry>, Vec<u32>) {
        let (mut rows, mut flat, mut masks) = (Vec::new(), Vec::new(), Vec::new());
        for (r, row) in params.iter().enumerate() {
            let mut row = *row;
            if !self.entries[r].is_empty() {
                row = row.with_entries(flat.len() as u32, self.entries[r].len() as u32);
                flat.extend_from_slice(&self.entries[r]);
            }
            if let Some(mask) = &self.masks[r] {
                row = row.with_mask((masks.len() / words) as u32);
                masks.extend_from_slice(mask.words());
            }
            rows.push(row);
        }
        (rows, flat, masks)
    }
}

/// The buffers of one slice of the vocabulary on a device.
struct Slice {
    logits: DeviceBuffer,
    tokens: Option<DeviceBuffer>,
    inverse: Option<DeviceBuffer>,
    params: DeviceBuffer,
    entries: Option<DeviceBuffer>,
    masks: Option<DeviceBuffer>,
    out: DeviceBuffer,
    lp: DeviceBuffer,
    scratch: DeviceBuffer,
    exchange: DeviceBuffer,
    batch: SampleBatch,
}

fn buffer(device: &Device, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64).unwrap();
    buffer.write(0, bytes).unwrap();
    buffer
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
        extras: &Extras,
        range: std::ops::Range<usize>,
        map: Option<&[u32]>,
        devices: u32,
        rank: u32,
    ) -> Self {
        let n_rows = rows.len() as u32;
        let vocab = rows[0].len();
        let n_local = range.len();
        let mut flat = Vec::with_capacity(rows.len() * n_local);
        for row in rows {
            flat.extend_from_slice(&row[range.clone()]);
        }
        let logits = buffer(device, &f32_bytes(&flat));
        let words = TokenMask::words_for(vocab as u32);
        let (rows_with, entries, masks) = extras.attach(params, words);
        let tokens = map.map(|map| buffer(device, &u32_bytes(&map[range.clone()])));
        let inverse = tokens.as_ref().map(|tokens| {
            let inverse = DeviceBuffer::new(device, vocab as u64 * 4).unwrap();
            // SAFETY: the buffers hold `n_local` and `vocab` values.
            unsafe {
                kernels.invert(stream, tokens.ptr(), n_local as u32, inverse.ptr(), vocab as u32)
            }
            .unwrap();
            inverse
        });
        let params_bytes: Vec<u8> = rows_with.iter().flat_map(|p| p.to_bytes()).collect();
        let params_buffer = buffer(device, &params_bytes);
        let entry_bytes: Vec<u8> = entries.iter().flat_map(Entry::to_bytes).collect();
        let entries = (!entries.is_empty()).then(|| buffer(device, &entry_bytes));
        let masks = (!masks.is_empty()).then(|| buffer(device, &u32_bytes(&masks)));
        let out = DeviceBuffer::new(device, (rows.len() * SamplePick::BYTES) as u64).unwrap();
        let lp = DeviceBuffer::new(device, (rows.len() * LogprobOut::BYTES) as u64).unwrap();
        let scratch = DeviceBuffer::new(device, SamplerKernels::scratch_bytes(n_rows)).unwrap();
        let exchange = DeviceBuffer::new(
            device,
            u64::from(SamplerKernels::exchange_values(n_rows, devices)) * 4,
        )
        .unwrap();
        // SAFETY: the scratch buffer has the size the sampler asks for.
        unsafe { kernels.clear(stream, scratch.ptr(), n_rows).unwrap() };
        let ptr = |b: &Option<DeviceBuffer>| b.as_ref().map_or(DevicePtr::NULL, DeviceBuffer::ptr);
        let batch = SampleBatch {
            logits: logits.ptr(),
            stride: n_local as u32,
            n_rows,
            capacity: n_rows,
            n_local: n_local as u32,
            tokens: ptr(&tokens),
            // A token list holds this slice's tokens only.
            first: if map.is_some() { 0 } else { range.start as u32 },
            inverse: ptr(&inverse),
            vocab: vocab as u32,
            params: params_buffer.ptr(),
            entries: ptr(&entries),
            masks: ptr(&masks),
            mask_words: words as u32,
            out: out.ptr(),
            logprobs: lp.ptr(),
            scratch: scratch.ptr(),
            exchange: exchange.ptr(),
            devices,
            rank,
        };
        Self {
            logits,
            tokens,
            inverse,
            params: params_buffer,
            entries,
            masks,
            out,
            lp,
            scratch,
            exchange,
            batch,
        }
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

    fn logprobs(&self) -> Vec<LogprobOut> {
        let mut bytes = vec![0u8; self.lp.len() as usize];
        self.lp.read(0, &mut bytes).unwrap();
        bytes
            .as_chunks::<{ LogprobOut::BYTES }>()
            .0
            .iter()
            .map(|b| LogprobOut::from_bytes(b))
            .collect()
    }

    fn keep(&self) -> usize {
        // The buffers live as long as the slice; this only names them.
        [&self.logits, &self.params, &self.scratch].iter().map(|b| b.len() as usize).sum::<usize>()
            + self.tokens.as_ref().map_or(0, |t| t.len() as usize)
            + self.inverse.as_ref().map_or(0, |t| t.len() as usize)
            + self.entries.as_ref().map_or(0, |t| t.len() as usize)
            + self.masks.as_ref().map_or(0, |t| t.len() as usize)
    }
}

/// What the device wrote for the rows.
struct Records {
    picks: Vec<SamplePick>,
    logprobs: Vec<LogprobOut>,
}

/// The picks of `rows` with the vocabulary cut at `cuts` into slices, each
/// run as a device of its own; the exchange between them is added here.
/// Returns the records of every slice.
fn run(
    device: &Device,
    kernels: &SamplerKernels,
    rows: &[Vec<f32>],
    params: &[SampleRow],
    extras: &Extras,
    cuts: &[usize],
    map: Option<&[u32]>,
) -> Vec<Records> {
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
                extras,
                w[0]..w[1],
                map,
                devices,
                rank as u32,
            )
        })
        .collect();
    let (with, _, _) = extras.attach(params, TokenMask::words_for(n as u32));
    let plan = SamplePlan::of(&with);
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
    slices.iter().map(|s| Records { picks: s.picks(), logprobs: s.logprobs() }).collect()
}

fn same(a: &SamplePick, b: &SamplePick) -> bool {
    a.token == b.token
        && a.status == b.status
        && a.value.to_bits() == b.value.to_bits()
        && a.aux.to_bits() == b.aux.to_bits()
        && a.cand_token == b.cand_token
        && a.cand_logit.map(f32::to_bits) == b.cand_logit.map(f32::to_bits)
}

/// The probabilities as far as the row reported them.
fn same_logprobs(a: &LogprobOut, b: &LogprobOut) -> bool {
    let n = b.n as usize;
    a.n == b.n
        && a.chosen.to_bits() == b.chosen.to_bits()
        && a.token[..n] == b.token[..n]
        && a.logprob[..n].iter().map(|v| v.to_bits()).eq(b.logprob[..n].iter().map(|v| v.to_bits()))
}

/// Checks every slice's records of `rows` against the host.
#[allow(clippy::too_many_arguments)]
fn check(
    device: &Device,
    kernels: &SamplerKernels,
    rows: &[Vec<f32>],
    params: &[SampleRow],
    extras: &Extras,
    cuts: &[usize],
    map: Option<&[u32]>,
    what: &str,
) {
    let got = run(device, kernels, rows, params, extras, cuts, map);
    let words = TokenMask::words_for(rows[0].len() as u32);
    let (with, _, _) = extras.attach(params, words);
    for (r, (row, p)) in rows.iter().zip(&with).enumerate() {
        let token_of = |i: usize| map.map_or(i as u32, |m| m[i]);
        let mask = extras.masks[r].as_ref().map(TokenMask::words);
        let want = sample_reference(row, token_of, p, &extras.entries[r], mask);
        for (slice, records) in got.iter().enumerate() {
            assert!(
                same(&records.picks[r], &want.pick),
                "{what}: row {r} ({} tokens), slice {slice} of {}, {p:?} (program {:?}):\n device {:?}\n host   {:?}",
                row.len(),
                got.len(),
                p.program(),
                records.picks[r],
                want.pick
            );
            if p.logprobs().is_some() {
                assert!(
                    same_logprobs(&records.logprobs[r], &want.logprobs),
                    "{what}: row {r} ({} tokens), slice {slice} of {}, {p:?}:\n device {:?}\n host   {:?}",
                    row.len(),
                    got.len(),
                    records.logprobs[r],
                    want.logprobs
                );
            }
        }
    }
}

#[test]
fn picks_equal_the_host_arithmetic() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0001);
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let mut checked = 0;
        for n in [1usize, 2, 7, 255, 4096, 4097, 50_000, 248_320] {
            let all = settings(n as u32);
            for shape in [Shape::Peaked, Shape::Flat, Shape::Tied, Shape::Masked] {
                // Every setting once per size and shape, a few rows per batch
                // with different settings.
                for batch in all.chunks(5) {
                    let rows: Vec<Vec<f32>> =
                        batch.iter().map(|_| row(&mut rng, n, shape)).collect();
                    let params: Vec<SampleRow> =
                        batch.iter().map(|p| p.row(rng.below(100_000) as u32)).collect();
                    let extras = Extras::none(rows.len());
                    let what = format!("{shape:?}");
                    check(device, &kernels, &rows, &params, &extras, &[], None, &what);
                    checked += rows.len();
                }
            }
        }
        println!("{}: {checked} rows equal the host's records", device.info().name);
    }
}

#[test]
fn a_token_map_and_slices_give_the_same_picks() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0002);
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        for n in [9usize, 5000, 248_320] {
            let all = settings(n as u32);
            // A permutation: column i is token map[i].
            let mut map: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                map.swap(i, rng.below(i + 1));
            }
            for shape in [Shape::Peaked, Shape::Tied, Shape::Masked, Shape::Flat] {
                for batch in all.chunks(8) {
                    let rows: Vec<Vec<f32>> =
                        batch.iter().map(|_| row(&mut rng, n, shape)).collect();
                    let params: Vec<SampleRow> =
                        batch.iter().map(|p| p.row(rng.below(100_000) as u32)).collect();
                    let extras = Extras::none(rows.len());
                    let what = format!("{shape:?}");
                    check(device, &kernels, &rows, &params, &extras, &[], Some(&map), &what);
                    // Two slices as the tensor split cuts the head, three uneven ones.
                    check(
                        device,
                        &kernels,
                        &rows,
                        &params,
                        &extras,
                        &[n * 5 / 8],
                        Some(&map),
                        &what,
                    );
                    check(device, &kernels, &rows, &params, &extras, &[n * 5 / 8], None, &what);
                    check(
                        device,
                        &kernels,
                        &rows,
                        &params,
                        &extras,
                        &[n / 9, n / 2],
                        Some(&map),
                        &what,
                    );
                }
            }
        }
    }
}

/// Corrections of random tokens and random masks for `rows` rows of a
/// vocabulary of `vocab` tokens.
fn random_extras(rng: &mut Rng, vocab: usize, rows: usize) -> Extras {
    let mut extras = Extras::none(rows);
    for r in 0..rows {
        if rng.below(3) != 0 {
            let count = 1 + rng.below(40.min(vocab));
            let mut tokens: Vec<u32> = Vec::new();
            while tokens.len() < count {
                let t = rng.below(vocab) as u32;
                if !tokens.contains(&t) {
                    tokens.push(t);
                }
            }
            extras.entries[r] = tokens
                .into_iter()
                .map(|token| Entry {
                    token,
                    bias: [0.0, 0.0, -2.5, 4.0, f32::NEG_INFINITY][rng.below(5)],
                    rep: [1.0, 1.1, 1.5, 0.8][rng.below(4)],
                    sub: [0.0, 0.5, 2.0][rng.below(3)],
                })
                .collect();
        }
        if rng.below(3) == 0 {
            let mut mask = TokenMask::none(vocab as u32);
            for _ in 0..1 + rng.below(vocab.min(3000)) {
                mask.set(rng.below(vocab) as u32, true);
            }
            if rng.below(4) == 0 {
                mask = TokenMask::all(vocab as u32);
                mask.set(rng.below(vocab) as u32, false);
            }
            extras.masks[r] = Some(mask);
        }
    }
    extras
}

#[test]
fn corrections_and_masks_equal_the_host() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0005);
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        for n in [3usize, 300, 4097, 50_000, 248_320] {
            let all = settings(n as u32);
            let mut map: Vec<u32> = (0..n as u32).collect();
            for i in (1..n).rev() {
                map.swap(i, rng.below(i + 1));
            }
            for shape in [Shape::Peaked, Shape::Tied, Shape::Masked] {
                for batch in all.chunks(6) {
                    let rows: Vec<Vec<f32>> =
                        batch.iter().map(|_| row(&mut rng, n, shape)).collect();
                    let params: Vec<SampleRow> =
                        batch.iter().map(|p| p.row(rng.below(100_000) as u32)).collect();
                    let extras = random_extras(&mut rng, n, rows.len());
                    let what = format!("{shape:?} with corrections");
                    check(device, &kernels, &rows, &params, &extras, &[], None, &what);
                    let thirds = [n / 3, n * 2 / 3];
                    check(device, &kernels, &rows, &params, &extras, &thirds, None, &what);
                    check(device, &kernels, &rows, &params, &extras, &[n / 2], Some(&map), &what);
                }
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
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let stream = Stream::new(device).unwrap();
        // The whole vocabulary: with fewer logits the chunk lists of a batch are
        // too short to reach where the histograms of a smaller batch would lie
        // if their place depended on the batch.
        let n = 248_320;
        let capacity = 4;
        let all = settings(n as u32);
        let rows: Vec<Vec<f32>> = (0..capacity).map(|_| row(&mut rng, n, Shape::Peaked)).collect();
        let params = vec![SampleParams::GREEDY.row(0); capacity];
        let extras = Extras::none(capacity);
        let slice =
            Slice::new(device, &stream, &kernels, &rows, &params, &extras, 0..n, None, 1, 0);
        let mut checked = 0;
        for round in 0..24 {
            let k = [4, 1, 3, 1, 2, 4][round % 6];
            let rows: Vec<Vec<f32>> = (0..k).map(|_| row(&mut rng, n, Shape::Peaked)).collect();
            // Rows of every kind, in turn.
            let params: Vec<SampleRow> = (0..k)
                .map(|r| all[(round * 5 + r * 7) % all.len()].row((round * 10 + r) as u32))
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
            let lp = slice.logprobs();
            for (r, (logits, p)) in rows.iter().zip(&params).enumerate() {
                let want = sample_reference(logits, |i| i as u32, p, &[], None);
                assert!(
                    same(&got[r], &want.pick),
                    "round {round}, row {r} of {k}: {:?}\n{:?}",
                    got[r],
                    want.pick
                );
                if p.logprobs().is_some() {
                    assert!(same_logprobs(&lp[r], &want.logprobs), "round {round}, row {r} of {k}");
                }
                checked += 1;
            }
        }
        println!(
            "{checked} rows in batches of 1 to 4 rows in one scratch equal the host's records"
        );
    }
}

#[test]
fn a_logit_that_is_not_a_number_is_reported() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let mut rng = Rng(77);
        for bad in [f32::NAN, f32::INFINITY] {
            let mut logits = row(&mut rng, 10_000, Shape::Peaked);
            logits[6789] = bad;
            let rows = vec![logits.clone(), row(&mut rng, 10_000, Shape::Peaked)];
            let params = vec![SampleParams::GREEDY.row(0); 2];
            for cuts in [vec![], vec![5000]] {
                let got = run(device, &kernels, &rows, &params, &Extras::none(2), &cuts, None);
                for records in &got {
                    assert_eq!(records.picks[0].status, SAMPLE_STATUS_NONFINITE, "{bad}");
                    assert_eq!(records.picks[1].status, 0, "the other row is not touched");
                }
            }
        }
    }
}

/// A row with no token left (a mask that allows nothing) is reported, and does
/// not disturb the row beside it.
#[test]
fn a_row_with_nothing_left_is_reported() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let mut rng = Rng(79);
        let n = 5000;
        let rows = vec![row(&mut rng, n, Shape::Peaked), row(&mut rng, n, Shape::Peaked)];
        let sampled = SampleParams { temperature: 0.8, top_k: 20, seed: 5, ..SampleParams::GREEDY };
        let params = vec![sampled.row(3), sampled.row(3)];
        let mut extras = Extras::none(2);
        extras.masks[0] = Some(TokenMask::none(n as u32));
        for cuts in [vec![], vec![n / 2]] {
            let got = run(device, &kernels, &rows, &params, &extras, &cuts, None);
            for records in &got {
                assert_eq!(records.picks[0].token, u32::MAX);
                assert_ne!(records.picks[0].status & SAMPLE_STATUS_EMPTY, 0);
                assert_eq!(records.picks[1].status, 0);
            }
        }
    }
}

#[test]
fn the_time_of_a_pick() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0003);
    let n = 248_320;
    let sampled =
        SampleParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 9, ..SampleParams::GREEDY };
    let top_p_only = SampleParams { top_k: 0, ..sampled };
    let lp = |after_chain| Logprobs { top: 5, chosen: true, after_chain };
    let raw = SampleParams { logprobs: lp(false), ..SampleParams::GREEDY };
    let post = SampleParams { logprobs: lp(true), ..sampled };
    let typical = SampleParams { typical_p: 0.9, top_n_sigma: 2.0, ..sampled };
    // The setting, whether the rows have 64 corrections, whether they have a mask.
    let cases: Vec<(&str, SampleParams, bool, bool)> = vec![
        ("greedy", SampleParams::GREEDY, false, false),
        ("top-k 20, top-p 0.95", sampled, false, false),
        ("top-p 0.95", top_p_only, false, false),
        ("greedy with penalties", SampleParams::GREEDY, true, false),
        ("greedy with a mask", SampleParams::GREEDY, false, true),
        ("top-k 20, top-p 0.95 with penalties and a mask", sampled, true, true),
        ("greedy with logprobs", raw, false, false),
        ("top-k 20, top-p 0.95 with logprobs after the chain", post, false, false),
        ("top-n-sigma 2, typical 0.9, top-k 20, top-p 0.95", typical, false, false),
    ];
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let stream = Stream::new(device).unwrap();
        for (name, setting, with_entries, with_mask) in &cases {
            for n_rows in [1usize, 4, 16] {
                let rows: Vec<Vec<f32>> =
                    (0..n_rows).map(|_| row(&mut rng, n, Shape::Peaked)).collect();
                let params: Vec<SampleRow> = (0..n_rows).map(|r| setting.row(r as u32)).collect();
                let mut extras = Extras::none(n_rows);
                for r in 0..n_rows {
                    if *with_entries {
                        // As a penalty over the last 64 tokens takes.
                        extras.entries[r] = (0..64)
                            .map(|i| Entry {
                                rep: 1.1,
                                sub: 0.3,
                                ..Entry::new(i * 3571 % n as u32)
                            })
                            .collect();
                    }
                    if *with_mask {
                        extras.masks[r] = Some(TokenMask::only(n as u32, (0..n as u32).step_by(3)));
                    }
                }
                let slice = Slice::new(
                    device,
                    &stream,
                    &kernels,
                    &rows,
                    &params,
                    &extras,
                    0..n,
                    None,
                    1,
                    0,
                );
                // The corrections change the logits, so every round starts
                // from a fresh copy.
                let host = f32_bytes(&rows.iter().flatten().copied().collect::<Vec<_>>());
                let (with, entries, _) = extras.attach(&params, TokenMask::words_for(n as u32));
                let entry_bytes: Vec<u8> = entries.iter().flat_map(Entry::to_bytes).collect();
                let plan = SamplePlan::of(&with);
                let mut times = Vec::new();
                for _ in 0..40 {
                    slice.logits.write(0, &host).unwrap();
                    if let Some(buffer) = &slice.entries {
                        buffer.write(0, &entry_bytes).unwrap();
                    }
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
                    "{}: {name}, {n_rows} rows of {n}: {:.0} us median, {:.0} us fastest, {} launches",
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

/// The arithmetic the stages are made of, one function at a time (`shaders/sample_probe.comp`)
/// against `ares_sampler::arith`, over inputs from every range the sampler reaches: the shaders
/// repeat the host's IEEE operations bit for bit, and the conversions, the division and the
/// square root of the device are the host's.
#[test]
fn the_arithmetic_of_the_shaders_equals_the_hosts() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0006);
    // A positive normal f64 or f32 of any exponent in [lo, hi) (powers of two).
    let wide64 = |rng: &mut Rng, lo: i32, hi: i32| {
        let e = lo + rng.below((hi - lo) as usize) as i32;
        f64::from_bits(((e + 1023) as u64) << 52 | (rng.next() >> 12))
    };
    let wide32 = |rng: &mut Rng, lo: i32, hi: i32| {
        let e = lo + rng.below((hi - lo) as usize) as i32;
        f32::from_bits(((e + 127) as u32) << 23 | (rng.next() as u32 >> 9))
    };
    let n = 20_000usize;
    for device in devices {
        let module =
            Module::load(device, include_bytes!(concat!(env!("OUT_DIR"), "/sample_probe.spv")))
                .unwrap();
        let probe = module.function("main").unwrap();
        let stream = Stream::new(device).unwrap();
        let (a_buf, b_buf) = (
            DeviceBuffer::new(device, (n as u64 + 16) * 8).unwrap(),
            DeviceBuffer::new(device, (n as u64 + 16) * 8).unwrap(),
        );
        let out = DeviceBuffer::new(device, (n as u64 + 16) * 40).unwrap();
        let run = |mode: u32, inputs: &[(u64, u64)]| -> Vec<[u64; 5]> {
            let bytes = |f: fn(&(u64, u64)) -> u64| -> Vec<u8> {
                inputs.iter().flat_map(|p| f(p).to_le_bytes()).collect()
            };
            a_buf.write(0, &bytes(|p| p.0)).unwrap();
            b_buf.write(0, &bytes(|p| p.1)).unwrap();
            let args = Args::new()
                .ptr(a_buf.ptr())
                .ptr(b_buf.ptr())
                .ptr(out.ptr())
                .u32(inputs.len() as u32)
                .u32(mode);
            // SAFETY: the buffers hold `n` items of the sizes the shader reads and writes.
            unsafe { probe.launch(&stream, [(inputs.len() as u32).div_ceil(64), 1, 1], args) }
                .unwrap();
            stream.synchronize().unwrap();
            let mut raw = vec![0u8; inputs.len() * 40];
            out.read(0, &mut raw).unwrap();
            raw.as_chunks::<40>()
                .0
                .iter()
                .map(|c| {
                    let mut r = [0u64; 5];
                    for (k, w) in r.iter_mut().zip(c.as_chunks::<8>().0) {
                        *k = u64::from_le_bytes(*w);
                    }
                    r
                })
                .collect()
        };
        let check =
            |name: &str, mode: u32, inputs: Vec<(u64, u64)>, want: &dyn Fn(u64, u64) -> u64| {
                let got = run(mode, &inputs);
                let bad = inputs
                    .iter()
                    .zip(&got)
                    .filter(|((a, b), g)| g[0] != want(*a, *b))
                    .take(3)
                    .map(|((a, b), g)| {
                        format!("({a:#x}, {b:#x}): device {:#x}, host {:#x}", g[0], want(*a, *b))
                    })
                    .collect::<Vec<_>>();
                assert!(bad.is_empty(), "{}: {name}: {}", device.info().name, bad.join("; "));
            };
        // exp(-x) as a fixed-point number: keys over the whole range, the edges included.
        let mut keys: Vec<(u64, u64)> =
            (0..n).map(|_| ((wide32(&mut rng, -30, 7).to_bits() as u64), 0)).collect();
        keys.extend(
            [0u32, 1, 0x3f80_0000, 0x41e0_0000, 0x4220_0000, 0x7f7f_ffff, 0x7f80_0000]
                .map(|k| (u64::from(k), 0)),
        );
        check("the weight of a distance", 0, keys, &|a, _| arith::weight_fixed(a as u32));
        let lns: Vec<(u64, u64)> =
            (0..n).map(|_| (wide32(&mut rng, -60, 60).to_bits() as u64, 0)).collect();
        check("ln in single precision", 1, lns, &|a, _| {
            u64::from(arith::ln_f32(f32::from_bits(a as u32)).to_bits())
        });
        let gumbels: Vec<(u64, u64)> = (0..n)
            .map(|_| (rng.next(), (rng.below(300_000) as u64) << 32 | rng.below(100_000) as u64))
            .collect();
        check("the Gumbel noise", 2, gumbels, &|a, b| {
            u64::from(arith::gumbel(a, b as u32, (b >> 32) as u32).to_bits())
        });
        let exps: Vec<(u64, u64)> = (0..n)
            .map(|_| ((rng.unit() as f64 * 1400.0 - 700.0 + rng.unit() as f64).to_bits(), 0))
            .collect();
        check("exp in double precision", 3, exps, &|a, _| {
            arith::exp_d(f64::from_bits(a)).to_bits()
        });
        let lnd: Vec<(u64, u64)> =
            (0..n).map(|_| (wide64(&mut rng, -900, 900).to_bits(), 0)).collect();
        check("ln in double precision", 4, lnd, &|a, _| arith::ln_d(f64::from_bits(a)).to_bits());
        let pows: Vec<(u64, u64)> = (0..n)
            .map(|i| {
                let x = if i % 50 == 0 { 0.0 } else { wide64(&mut rng, -20, 10) };
                (x.to_bits(), ((rng.unit() * 8.0 - 3.0) as f64).to_bits())
            })
            .collect();
        check("pow in double precision", 5, pows, &|a, b| {
            arith::pow_d(f64::from_bits(a), f64::from_bits(b)).to_bits()
        });
        let sums: Vec<(u64, u64)> =
            (0..n).map(|_| (1 + (rng.next() >> rng.below(24)), 0)).collect();
        check("the logarithm of a fixed-point sum", 6, sums, &|a, _| arith::ln_fixed(a).to_bits());
        let doubles = |rng: &mut Rng| -> Vec<(u64, u64)> {
            (0..n)
                .map(|_| {
                    let sign = |rng: &mut Rng| if rng.below(2) == 0 { -1.0 } else { 1.0 };
                    (
                        (sign(rng) * wide64(rng, -200, 200)).to_bits(),
                        (sign(rng) * wide64(rng, -200, 200)).to_bits(),
                    )
                })
                .collect()
        };
        let pairs = doubles(&mut rng);
        check("a division in double precision", 7, pairs.clone(), &|a, b| {
            (f64::from_bits(a) / f64::from_bits(b)).to_bits()
        });
        let roots: Vec<(u64, u64)> =
            (0..n).map(|_| (wide64(&mut rng, -200, 200).to_bits(), 0)).collect();
        check("a square root in double precision", 8, roots, &|a, _| {
            f64::from_bits(a).sqrt().to_bits()
        });
        let big: Vec<(u64, u64)> = (0..n).map(|_| (rng.next() >> rng.below(64), 0)).collect();
        check("a 64-bit integer as f32", 9, big.clone(), &|a, _| u64::from((a as f32).to_bits()));
        check("a 64-bit integer as f64", 10, big, &|a, _| (a as f64).to_bits());
        let narrow: Vec<(u64, u64)> = (0..n)
            .map(|_| ((wide64(&mut rng, -100, 100) * rng.unit() as f64).to_bits(), 0))
            .collect();
        check("f64 as f32", 11, narrow, &|a, _| u64::from((f64::from_bits(a) as f32).to_bits()));
        let singles: Vec<(u64, u64)> = (0..n)
            .map(|_| {
                (
                    wide32(&mut rng, -60, 60).to_bits() as u64,
                    wide32(&mut rng, -60, 60).to_bits() as u64,
                )
            })
            .collect();
        check("a division in single precision", 12, singles, &|a, b| {
            u64::from((f32::from_bits(a as u32) / f32::from_bits(b as u32)).to_bits())
        });
        let draws: Vec<(u64, u64)> =
            (0..n).map(|_| (rng.next(), rng.below(300_000) as u64)).collect();
        check("the draw of XTC", 13, draws, &|a, b| {
            u64::from(arith::xtc_draw(a, b as u32).to_bits())
        });
        // The terms one token adds to the sums of a set: a distance and its scaled form.
        let terms: Vec<(u64, u64)> = (0..n)
            .map(|_| {
                let d = wide32(&mut rng, -10, 12);
                let x = d * wide32(&mut rng, -3, 5);
                (u64::from(d.to_bits()), u64::from(x.to_bits()))
            })
            .collect();
        let got = run(14, &terms);
        for ((a, b), g) in terms.iter().zip(&got) {
            let t = arith::Sums::of(f32::from_bits(*a as u32), f32::from_bits(*b as u32));
            assert_eq!(g, &[t.count, t.z, t.a, t.q1, t.q2], "terms of {a:#x}, {b:#x}");
        }
        let shares: Vec<(u64, u64)> = (0..n).map(|_| (rng.next() >> 2, rng.next() >> 32)).collect();
        check("a share of a sum", 15, shares, &|a, b| arith::share(a, b as u32));
        println!("{}: the arithmetic of the shaders equals the host's", device.info().name);
    }
}

/// The kernels that keep a draft chain on the device move exactly what they name: the picks into
/// the records the next kernels read, the rows of the hidden state a chain starts from, and the
/// carry of each sequence through a step, a window that verified and its commit.
#[test]
fn the_draft_kernels_move_what_they_name() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5eed_1234_abcd_0007);
    let d = 37usize;
    let carry_rows = 4usize;
    let floats = |rng: &mut Rng, n: usize| -> Vec<f32> { (0..n).map(|_| rng.normal()).collect() };
    let read_f32 = |buffer: &DeviceBuffer| -> Vec<f32> {
        let mut bytes = vec![0u8; buffer.len() as usize];
        buffer.read(0, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    };
    let read_u32 = |buffer: &DeviceBuffer| -> Vec<u32> {
        let mut bytes = vec![0u8; buffer.len() as usize];
        buffer.read(0, &mut bytes).unwrap();
        bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
    };
    for device in devices {
        let kernels = SamplerKernels::load(device).unwrap();
        let stream = Stream::new(device).unwrap();

        // The picks go to the next link's tokens, the verifying rows' tokens and the drafts.
        let tokens = [11u32, 22, 33, 44, 55];
        let picks: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| {
                let mut record = [0u8; SamplePick::BYTES];
                record[..4].copy_from_slice(&t.to_le_bytes());
                record
            })
            .collect();
        let (record_rows, draft_at) = ([9u32, 1, 4, 0, 7], [7u32, 6, 5, 0, 3]);
        let picks_buffer = buffer(device, &picks);
        let next = buffer(device, &u32_bytes(&[0xdead; 5]));
        let record = buffer(device, &u32_bytes(&[0xbeef; 10]));
        let drafts = buffer(device, &u32_bytes(&[0xfeed; 8]));
        let rows_buffer = buffer(device, &u32_bytes(&record_rows));
        let at_buffer = buffer(device, &u32_bytes(&draft_at));
        // SAFETY: the buffers hold the picks, indices and arrays the call names.
        unsafe {
            kernels
                .draft_put(
                    &stream,
                    picks_buffer.ptr(),
                    5,
                    next.ptr(),
                    3,
                    record.ptr(),
                    rows_buffer.ptr(),
                    drafts.ptr(),
                    at_buffer.ptr(),
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(read_u32(&next), [11, 22, 33, 0xdead, 0xdead]);
        let mut want = vec![0xbeef; 10];
        let mut want_drafts = vec![0xfeed; 8];
        for (c, &t) in tokens.iter().enumerate() {
            want[record_rows[c] as usize] = t;
            want_drafts[draft_at[c] as usize] = t;
        }
        assert_eq!(read_u32(&record), want);
        assert_eq!(read_u32(&drafts), want_drafts);

        // Rows of the hidden state by index.
        let src = floats(&mut rng, 9 * d);
        let index = [6u32, 0, 6, 3, 8];
        let src_buffer = buffer(device, &f32_bytes(&src));
        let dst = DeviceBuffer::new(device, (5 * d * 4) as u64).unwrap();
        let index_buffer = buffer(device, &u32_bytes(&index));
        // SAFETY: the buffers hold the rows and indices the call names.
        unsafe {
            kernels
                .gather_hidden(
                    &stream,
                    dst.ptr(),
                    src_buffer.ptr(),
                    index_buffer.ptr(),
                    5,
                    d as u32,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let got = read_f32(&dst);
        for (r, &i) in index.iter().enumerate() {
            assert_eq!(
                got[r * d..(r + 1) * d].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                src[i as usize * d..(i as usize + 1) * d]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "row {r} of the gather"
            );
        }

        // A step of three windows: a first window of a sequence at position 0 (slot 1), a direct
        // window (slot 0), and a window that continues a sequence (slot 2).
        let seg = |row0, n_rows, slot, pos0, direct| Segment { row0, n_rows, slot, pos0, direct };
        let segments = [seg(0, 3, 1, 0, false), seg(3, 4, 0, 5, true), seg(7, 2, 2, 9, false)];
        let mut step = StepRecord::new(&segments, &[0; 9], None, &[]).unwrap();
        let hidden = floats(&mut rng, 9 * d);
        let carry = floats(&mut rng, 3 * carry_rows * d);
        let step_buffer = buffer(device, step.as_bytes());
        let hidden_buffer = buffer(device, &f32_bytes(&hidden));
        let carry_buffer = buffer(device, &f32_bytes(&carry));
        let out = DeviceBuffer::new(device, (9 * d * 4) as u64).unwrap();
        let dims = step.dims();
        // SAFETY: the buffers hold the record, the rows and the carries the calls name.
        unsafe {
            kernels
                .shift_hidden(
                    &stream,
                    step_buffer.ptr(),
                    dims.n_segs,
                    dims.max_seg_rows,
                    hidden_buffer.ptr(),
                    carry_buffer.ptr(),
                    carry_rows as u32,
                    out.ptr(),
                    d as u32,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let shifted = read_f32(&out);
        let carry_row = |slot: usize, row: usize| &carry[(slot * carry_rows + row) * d..][..d];
        for s in &segments {
            for i in 0..s.n_rows as usize {
                let r = s.row0 as usize + i;
                let want: Vec<f32> = if i > 0 {
                    hidden[(r - 1) * d..r * d].to_vec()
                } else if s.pos0 == 0 {
                    vec![0.0; d]
                } else {
                    carry_row(s.slot as usize, 0).to_vec()
                };
                assert_eq!(
                    shifted[r * d..(r + 1) * d].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "row {r} of the shift"
                );
            }
        }

        // Keeping the step's final hidden states in the carries: every row of a window that is
        // not direct (up to the carry's rows), the last row of a direct one in row 0.
        // SAFETY: as above.
        unsafe {
            kernels
                .keep_hidden(
                    &stream,
                    step_buffer.ptr(),
                    dims.n_segs,
                    dims.max_seg_rows,
                    hidden_buffer.ptr(),
                    carry_buffer.ptr(),
                    carry_rows as u32,
                    d as u32,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut want = carry.clone();
        for s in &segments {
            for i in 0..s.n_rows as usize {
                let r = s.row0 as usize + i;
                let at = if s.direct {
                    if i + 1 != s.n_rows as usize {
                        continue;
                    }
                    0
                } else if i >= carry_rows {
                    continue;
                } else {
                    i
                };
                want[(s.slot as usize * carry_rows + at) * d..][..d]
                    .copy_from_slice(&hidden[r * d..(r + 1) * d]);
            }
        }
        let kept = read_f32(&carry_buffer);
        assert_eq!(
            kept.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the carries after keep"
        );

        // The commit: a window that verified two of its three rows moves row 1 to row 0; one
        // that kept one row, and a direct one, leave the carry as it is.
        step.set_accepted(0, 2).unwrap();
        step.set_accepted(2, 1).unwrap();
        let commit = buffer(device, step.as_bytes());
        // SAFETY: as above.
        unsafe {
            kernels
                .commit_hidden(
                    &stream,
                    commit.ptr(),
                    dims.n_segs,
                    carry_buffer.ptr(),
                    carry_rows as u32,
                    d as u32,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        let row1 = want[(carry_rows + 1) * d..][..d].to_vec();
        want[carry_rows * d..][..d].copy_from_slice(&row1);
        let committed = read_f32(&carry_buffer);
        assert_eq!(
            committed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the carries after the commit"
        );
        println!("{}: the draft kernels move what they name", device.info().name);
    }
}
