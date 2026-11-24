//! Tests of the attention module (`AttentionKernels`): the kernels that write
//! K and V into the cache and the attention kernels that read it.
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test -p ares-cuda --test attention -- --test-threads=1 --nocapture
//! ```
//!
//! What they prove, against `ares_quant` (the specification and the reference):
//!
//! - *The codec*: the encoder kernel writes the bytes of `encode_tile` /
//!   `encode_row`, for every format, equal on the golden files of
//!   `crates/quant/tests/golden/kv` and on random and extreme tiles.
//! - *Writing*: appending the rows of a step and converting entries leaves in
//!   the pools exactly the bytes `KvHead` stores, for pairs of formats, with
//!   drafts (pending cells) and settling.
//! - *Reading*: attention equals `KvHead::attend` within the stated error.

use std::path::PathBuf;

use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
    PlanConfig, Planner, Stream,
};
use ares_quant::{KvFormat, KvHead, KvPolicy, encode_row, encode_tile};
use common::{Rng, download, f32_bytes, gpus, u32_bytes, upload};

mod common;

const TILE: usize = 128;

/// What the rows of K and V are drawn from.
impl Rng {
    fn gaussian(&mut self) -> f32 {
        let u1 = (self.unit() + 1.0) * 0.5 + 1e-7;
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos()
    }

    /// A value like the rows of K and V: a per-channel offset (`channel`), noise,
    /// and now and then an outlier.
    fn kv_value(&mut self, channel: f32) -> f32 {
        let v = 0.5 * channel + self.gaussian();
        if self.next().is_multiple_of(300) { v * 12.0 } else { v }
    }
}

fn cache_format(f: KvFormat) -> CacheFormat {
    CacheFormat::ALL[KvFormat::ALL.iter().position(|x| *x == f).unwrap()]
}

fn zeroed(device: &Device, len: u64) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, len.max(16)).unwrap();
    buffer.write(0, &vec![0u8; len.max(16) as usize]).unwrap();
    buffer
}

fn status_of(buffer: &DeviceBuffer) -> u32 {
    u32::from_le_bytes(download(buffer, 4).try_into().unwrap())
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    if a.len() != b.len() {
        return Some(a.len().min(b.len()));
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

// ---------------------------------------------------------------------------
// The codec

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../quant/tests/golden/kv")
}

fn read_golden(name: &str) -> Vec<u8> {
    std::fs::read(golden_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Encodes `rows` rows of `d` values with the kernel: (mean, rows).
fn device_encode(
    kernels: &AttentionKernels,
    stream: &Stream,
    format: CacheFormat,
    d: usize,
    values: &[f32],
    use_mean: bool,
) -> (Vec<u8>, Vec<u8>) {
    let device = kernels.device();
    let rows = values.len() / d;
    let src = upload(device, &f32_bytes(values));
    let mean_len = if use_mean { format.mean_bytes(d as u32) as usize } else { 0 };
    let mean = zeroed(device, mean_len as u64);
    let out_len = rows * format.row_bytes(d as u32) as usize;
    let out = zeroed(device, out_len as u64);
    let status = zeroed(device, 4);
    // SAFETY: the buffers have the sizes the kernel writes.
    unsafe {
        kernels
            .encode_tile_f32(
                stream,
                src.ptr(),
                rows as u32,
                d as u32,
                format,
                use_mean,
                mean.ptr(),
                out.ptr(),
                status.ptr(),
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(status_of(&status), 0, "status");
    (download(&mean, mean_len), download(&out, out_len))
}

#[test]
fn the_encoder_kernel_reproduces_the_golden_files_byte_for_byte() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        for (suffix, input_name) in [("", "input.f32"), ("_search", "input_search.f32")] {
            let input: Vec<f32> = read_golden(input_name)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            for format in KvFormat::ALL {
                let file = read_golden(&format!("{}{suffix}.bin", format.name()));
                let encoded_len = u32::from_le_bytes(file[20..24].try_into().unwrap()) as usize;
                let expected = &file[32..32 + encoded_len];
                let (_, got) = device_encode(
                    &kernels,
                    &stream,
                    cache_format(format),
                    input.len(),
                    &input,
                    false,
                );
                assert_eq!(
                    first_difference(&got, expected),
                    None,
                    "{format}{suffix}: bytes differ on {}",
                    device.info().name
                );
            }
        }
        // The tile of 5 rows of 96 values, with its mean.
        let input: Vec<f32> = read_golden("input_tile.f32")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        for format in [KvFormat::Q7, KvFormat::Q6, KvFormat::Q5, KvFormat::Q4] {
            let file = read_golden(&format!("{}_tile.bin", format.name()));
            let word = |i: usize| u32::from_le_bytes(file[4 * i..4 * i + 4].try_into().unwrap());
            let (mean_len, enc_len) = (word(6) as usize, word(7) as usize);
            let expected_mean = &file[32..32 + mean_len];
            let expected_rows = &file[32 + mean_len..32 + mean_len + enc_len];
            let (mean, rows) =
                device_encode(&kernels, &stream, cache_format(format), 96, &input, true);
            assert_eq!(first_difference(&mean, expected_mean), None, "{format} tile mean");
            assert_eq!(first_difference(&rows, expected_rows), None, "{format} tile rows");
        }
    }
}

/// Rows of the kinds that stress the encoder: the realistic kind, constant rows,
/// zeros, tiny and huge values, single large values.
fn stress_rows(rng: &mut Rng, rows: usize, d: usize, kind: usize) -> Vec<f32> {
    let channel: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
    let mut out = Vec::with_capacity(rows * d);
    for r in 0..rows {
        for (c, &mean) in channel.iter().enumerate() {
            let v = match kind {
                0 => rng.kv_value(mean),
                1 => mean,
                2 => 0.0,
                3 => 1e-6 * rng.gaussian(),
                4 => 3e4 * rng.gaussian(),
                5 => {
                    if c == (r * 7) % d {
                        900.0
                    } else {
                        0.01 * rng.gaussian()
                    }
                }
                6 => {
                    (if rng.next().is_multiple_of(2) { 1.0 } else { -1.0 })
                        * (1.0 + 1e-3 * rng.gaussian())
                }
                _ => 40.0 + rng.gaussian(),
            };
            out.push(v.clamp(-6.0e4, 6.0e4));
        }
    }
    out
}

#[test]
fn the_encoder_kernel_equals_encode_tile_on_random_and_extreme_tiles() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for format in KvFormat::ALL {
            for (d, rows) in [(256, 128), (128, 128), (64, 5), (256, 1)] {
                for kind in 0..8 {
                    let values = stress_rows(&mut rng, rows, d, kind);
                    let n_mean = format.mean_bytes(d).unwrap();
                    let mut mean = vec![0u8; n_mean];
                    let mut out = vec![0u8; rows * format.row_bytes(d).unwrap()];
                    encode_tile(format, d, &values, &mut mean, &mut out).unwrap();
                    let (got_mean, got) =
                        device_encode(&kernels, &stream, cache_format(format), d, &values, true);
                    assert_eq!(
                        first_difference(&got_mean, &mean),
                        None,
                        "{format} d={d} kind={kind} mean"
                    );
                    assert_eq!(
                        first_difference(&got, &out),
                        None,
                        "{format} d={d} rows={rows} kind={kind} rows"
                    );
                    if rows == 1 {
                        let mut row = vec![0u8; format.row_bytes(d).unwrap()];
                        encode_row(format, &values, &mut row).unwrap();
                        let (_, got) = device_encode(
                            &kernels,
                            &stream,
                            cache_format(format),
                            d,
                            &values,
                            false,
                        );
                        assert_eq!(first_difference(&got, &row), None, "{format} d={d} encode_row");
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A sequence on the device, mirrored by KvHead

/// The pools of one layer and the slot bookkeeping the state crate will do.
struct Pools {
    geo: CacheGeometry,
    exact: DeviceBuffer,
    converted: DeviceBuffer,
    free_exact: Vec<u32>,
    free_converted: Vec<u32>,
}

impl Pools {
    fn new(device: &Device, geo: CacheGeometry, n_exact: u32, n_converted: u32) -> Self {
        Self {
            geo,
            exact: zeroed(device, n_exact as u64 * geo.exact_entry_bytes()),
            converted: zeroed(device, n_converted as u64 * geo.converted_entry_bytes()),
            free_exact: (0..n_exact).rev().collect(),
            free_converted: (0..n_converted).rev().collect(),
        }
    }
}

/// One sequence: `KvHead` per KV head (the reference) and its table on the
/// device. Rows are `[row][kv_heads][D]`.
struct DevSeq {
    policy: KvPolicy,
    heads: Vec<KvHead>,
    table: Vec<u32>,
    len: usize,
    /// Tiles already converted (device side).
    converted_tiles: std::ops::Range<usize>,
    table_dev: DeviceBuffer,
}

const CONVERTED: u32 = 0x8000_0000;

impl DevSeq {
    fn new(device: &Device, geo: &CacheGeometry, policy: KvPolicy, max_entries: usize) -> Self {
        Self {
            policy,
            heads: (0..geo.kv_heads)
                .map(|_| KvHead::new(policy, geo.head_dim as usize).unwrap())
                .collect(),
            table: Vec::new(),
            len: 0,
            converted_tiles: 0..0,
            table_dev: zeroed(device, 4 * max_entries as u64),
        }
    }

    fn upload_table(&self) {
        self.table_dev.write(0, &u32_bytes(&self.table)).unwrap();
    }

    /// Appends `rows` rows. `pending`: draft cells, converted only by `settle`.
    #[allow(clippy::too_many_arguments)]
    fn append(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pools: &mut Pools,
        k: &[f32],
        v: &[f32],
        pending: bool,
        status: &DeviceBuffer,
    ) {
        let geo = pools.geo;
        let d = geo.head_dim as usize;
        let (total, first) = (geo.kv_total as usize, geo.first_kv_head as usize);
        let n = k.len() / (total * d);
        let device = kernels.device();
        let mut rows = Vec::new();
        for i in 0..n {
            let pos = self.len + i;
            if pos / TILE == self.table.len() {
                let entry = if self.policy.has_mean() {
                    pools.free_exact.pop().expect("exact slots")
                } else {
                    pools.free_converted.pop().expect("converted slots") | CONVERTED
                };
                self.table.push(entry);
            }
            rows.push(AttnRow { table: 0, pos: pos as u32, slot_base: 0, n_slots: 0 });
            for (hi, head) in self.heads.iter_mut().enumerate() {
                let at = (i * total + first + hi) * d;
                let (kr, vr) = (&k[at..at + d], &v[at..at + d]);
                if pending {
                    head.append_pending(kr, vr).unwrap();
                } else {
                    head.append(kr, vr).unwrap();
                }
            }
        }
        self.len += n;
        self.upload_table();
        let rows_dev = upload(device, &AttnRow::to_bytes(&rows));
        let k_dev = upload(device, &f32_bytes(k));
        let v_dev = upload(device, &f32_bytes(v));
        // SAFETY: sizes follow the geometry; the slots were taken from the pools.
        unsafe {
            kernels
                .append(
                    stream,
                    &geo,
                    rows_dev.ptr(),
                    n as u32,
                    k_dev.ptr(),
                    v_dev.ptr(),
                    self.table_dev.ptr(),
                    pools.exact.ptr(),
                    pools.converted.ptr(),
                    ares_cuda::DevicePtr::NULL,
                    status.ptr(),
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        if !pending {
            self.convert_due(kernels, stream, pools, status);
        }
    }

    fn settle(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pools: &mut Pools,
        status: &DeviceBuffer,
    ) {
        for head in &mut self.heads {
            head.settle().unwrap();
        }
        self.convert_due(kernels, stream, pools, status);
    }

    /// Converts the tiles the policy converts at the current length.
    fn convert_due(
        &mut self,
        kernels: &AttentionKernels,
        stream: &Stream,
        pools: &mut Pools,
        status: &DeviceBuffer,
    ) {
        if !self.policy.has_mean() {
            return;
        }
        let range = self.policy.quantized_cells(self.len);
        let tiles = range.start / TILE..range.end / TILE;
        let mut jobs = Vec::new();
        for tile in tiles.clone() {
            if self.converted_tiles.contains(&tile) || self.table[tile] & CONVERTED != 0 {
                continue;
            }
            let from = self.table[tile];
            let to = pools.free_converted.pop().expect("converted slots");
            jobs.extend([from, to]);
            self.table[tile] = to | CONVERTED;
            pools.free_exact.push(from);
        }
        if jobs.is_empty() {
            return;
        }
        let jobs_dev = upload(kernels.device(), &u32_bytes(&jobs));
        // SAFETY: the jobs name live slots.
        unsafe {
            kernels
                .convert(
                    stream,
                    &pools.geo,
                    jobs_dev.ptr(),
                    (jobs.len() / 2) as u32,
                    pools.exact.ptr(),
                    pools.converted.ptr(),
                    ares_cuda::DevicePtr::NULL,
                    status.ptr(),
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        self.upload_table();
        self.converted_tiles = tiles;
    }

    /// Compares every stored byte with the reference.
    fn check_bytes(&self, pools: &Pools, label: &str) {
        let geo = pools.geo;
        let (d, h) = (geo.head_dim as usize, geo.kv_heads as usize);
        let exact = download(&pools.exact, pools.exact.len() as usize);
        let converted = download(&pools.converted, pools.converted.len() as usize);
        let (ee, ce) = (geo.exact_entry_bytes() as usize, geo.converted_entry_bytes() as usize);
        let ch = geo.converted_head_bytes() as usize;
        let (rbk, rbv) =
            (geo.k.row_bytes(geo.head_dim) as usize, geo.v.row_bytes(geo.head_dim) as usize);
        let (km, vm) =
            (geo.k.mean_bytes(geo.head_dim) as usize, geo.v.mean_bytes(geo.head_dim) as usize);
        for (hi, head) in self.heads.iter().enumerate() {
            assert!(hi < h);
            for cell in 0..self.len {
                let entry = self.table[cell / TILE];
                let slot = (entry & !CONVERTED) as usize;
                let r = cell % TILE;
                if let Some((k, v)) = head.exact(cell) {
                    assert_eq!(
                        entry & CONVERTED,
                        0,
                        "{label}: exact cell {cell} is in a converted entry"
                    );
                    let base = slot * ee + hi * (TILE * 4 * d);
                    let got_k = &exact[base + r * 2 * d..base + (r + 1) * 2 * d];
                    let got_v = &exact
                        [base + TILE * 2 * d + r * 2 * d..base + TILE * 2 * d + (r + 1) * 2 * d];
                    assert_eq!(
                        first_difference(got_k, k),
                        None,
                        "{label}: head {hi} cell {cell} exact K"
                    );
                    assert_eq!(
                        first_difference(got_v, v),
                        None,
                        "{label}: head {hi} cell {cell} exact V"
                    );
                } else {
                    assert_ne!(
                        entry & CONVERTED,
                        0,
                        "{label}: converted cell {cell} is in an exact entry"
                    );
                    let (unit, i) = head.unit_of(cell).expect("converted cell");
                    let base = slot * ce + hi * ch;
                    let rk = base + km + r * rbk;
                    assert_eq!(
                        first_difference(&converted[rk..rk + rbk], unit.k_row(i)),
                        None,
                        "{label}: head {hi} cell {cell} K"
                    );
                    let rv = base + km + TILE * rbk + vm + r * rbv;
                    assert_eq!(
                        first_difference(&converted[rv..rv + rbv], unit.v_row(i)),
                        None,
                        "{label}: head {hi} cell {cell} V"
                    );
                    if r == 0 {
                        assert_eq!(
                            first_difference(&converted[base..base + km], &unit.k_mean),
                            None,
                            "{label}: head {hi} K mean"
                        );
                        let vb = base + km + TILE * rbk;
                        assert_eq!(
                            first_difference(&converted[vb..vb + vm], &unit.v_mean),
                            None,
                            "{label}: head {hi} V mean"
                        );
                    }
                }
            }
        }
    }
}

fn rows_of(rng: &mut Rng, n: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * h * d);
    let channel: Vec<f32> = (0..h * d).map(|_| rng.gaussian()).collect();
    for _ in 0..n {
        for &mean in &channel {
            out.push(rng.kv_value(mean));
        }
    }
    out
}

fn pair_label(k: KvFormat, v: KvFormat) -> String {
    format!("{k}/{v}")
}

#[test]
fn appending_and_converting_leave_the_bytes_kvhead_stores() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(77);
        let pairs = [
            (KvFormat::Q8, KvFormat::Q8),
            (KvFormat::F16, KvFormat::F16),
            (KvFormat::F16, KvFormat::Q8),
            (KvFormat::Q8, KvFormat::F16),
            (KvFormat::Q4, KvFormat::Q4),
            (KvFormat::Q7, KvFormat::Q5),
            (KvFormat::Q6, KvFormat::Q8),
            (KvFormat::Q8, KvFormat::Q4),
            (KvFormat::F16, KvFormat::Q5),
        ];
        for (k, v) in pairs {
            for d in [256usize, 128] {
                let geo = CacheGeometry {
                    k: cache_format(k),
                    v: cache_format(v),
                    head_dim: d as u32,
                    kv_heads: 2,
                    kv_total: 3,
                    first_kv_head: 1,
                    q_per_kv: 4,
                };
                let policy = KvPolicy::new(k, v);
                let mut pools = Pools::new(device, geo, 6, 8);
                let status = zeroed(device, 4);
                let mut seq = DevSeq::new(device, &geo, policy, 16);
                let label = pair_label(k, v);
                // The model has 3 KV heads of which this device holds heads 1 and 2.
                // Appends in steps of various sizes (a prompt, decode rows, drafts).
                // Without a mean the device writes drafts straight into the format (the
                // reference keeps pending cells exact until they are settled, so only
                // `append` is comparable) and nothing is converted later.
                let drafts = policy.has_mean();
                for (n, pending) in
                    [(300usize, false), (1, false), (1, false), (150, false), (5, drafts)]
                {
                    let full_k = rows_of(&mut rng, n, 3, d);
                    let full_v = rows_of(&mut rng, n, 3, d);
                    seq.append(&kernels, &stream, &mut pools, &full_k, &full_v, pending, &status);
                    if !pending {
                        seq.check_bytes(&pools, &format!("{label} d={d} after {n}"));
                    }
                }
                seq.check_bytes(&pools, &format!("{label} d={d} with drafts"));
                if drafts {
                    // Accept three of the five drafts: cut the rest, settle.
                    for head in &mut seq.heads {
                        head.truncate(seq.len - 2).unwrap();
                    }
                    seq.len -= 2;
                    seq.settle(&kernels, &stream, &mut pools, &status);
                    seq.check_bytes(&pools, &format!("{label} d={d} settled"));
                }
                assert_eq!(status_of(&status), 0, "{label} status");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reading

/// The error of the kernel against `KvHead::attend` on one case.
#[derive(Clone, Copy, Debug, Default)]
struct Error {
    /// The largest |kernel - reference| over rows, heads and elements, relative to
    /// the largest |reference| of its row and head.
    max_rel: f64,
    /// The rms of the same differences, relative to the rms of the reference.
    rms_rel: f64,
}

impl Error {
    fn worse(self, other: Error) -> Error {
        Error { max_rel: self.max_rel.max(other.max_rel), rms_rel: self.rms_rel.max(other.rms_rel) }
    }
}

struct Case {
    k: KvFormat,
    v: KvFormat,
    d: usize,
    hist: usize,
    rows: usize,
}

/// A sequence built through the writing kernels, with the rows of its last step
/// still to attend: `rows` cells appended in one step after `hist` earlier ones.
fn build(
    kernels: &AttentionKernels,
    stream: &Stream,
    case: &Case,
    geo: &CacheGeometry,
    rng: &mut Rng,
) -> (Pools, DevSeq) {
    let device = kernels.device();
    let entries = (case.hist + case.rows).div_ceil(TILE) + 2;
    let mut pools = Pools::new(device, *geo, entries as u32 + 6, entries as u32 + 2);
    let status = zeroed(device, 4);
    let mut seq = DevSeq::new(device, geo, KvPolicy::new(case.k, case.v), entries + 2);
    let total = geo.kv_total as usize;
    // The history in prompt-sized steps, then the rows of the window.
    let mut left = case.hist;
    while left > 0 {
        let n = left.min(700);
        let (k, v) = (rows_of(rng, n, total, case.d), rows_of(rng, n, total, case.d));
        seq.append(kernels, stream, &mut pools, &k, &v, false, &status);
        left -= n;
    }
    let (k, v) = (rows_of(rng, case.rows, total, case.d), rows_of(rng, case.rows, total, case.d));
    seq.append(kernels, stream, &mut pools, &k, &v, false, &status);
    assert_eq!(status_of(&status), 0);
    (pools, seq)
}

/// What the rows of one sequence in a step see.
enum Look {
    /// `n` consecutive rows, the last `n` cells of the sequence, attending causally.
    Window(usize),
    /// `n` consecutive rows starting at the cell `pos0`, attending causally.
    At(usize, usize),
    /// One row, the last cell, attending to these ranges only.
    Ranges(Vec<std::ops::Range<usize>>),
}

/// Runs attention for the rows described by `looks` (one sequence each) and
/// compares every row with the reference. `check` limits how many rows of a wide
/// window are compared (the reference is slow): every row up to that many, spread.
#[allow(clippy::too_many_arguments)]
fn attend_many(
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: &CacheGeometry,
    pools: &Pools,
    seqs: &[&DevSeq],
    looks: &[Look],
    gated: bool,
    check: usize,
    rng: &mut Rng,
) -> Error {
    attend_many_q(kernels, stream, geo, pools, seqs, looks, gated, check, None, rng)
}

/// As `attend_many`, with the queries given (`[row][q_heads][D]`) instead of random.
#[allow(clippy::too_many_arguments)]
fn attend_many_q(
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: &CacheGeometry,
    pools: &Pools,
    seqs: &[&DevSeq],
    looks: &[Look],
    gated: bool,
    check: usize,
    q_given: Option<&[f32]>,
    rng: &mut Rng,
) -> Error {
    let device = kernels.device();
    let (g, d) = (geo.q_per_kv as usize, geo.head_dim as usize);
    let q_total = geo.kv_total as usize * g;
    let n: usize = looks
        .iter()
        .map(|l| match l {
            Look::Window(n) | Look::At(_, n) => *n,
            Look::Ranges(_) => 1,
        })
        .sum();
    let q: Vec<f32> = match q_given {
        Some(q) => {
            assert_eq!(q.len(), n * q_total * d);
            q.to_vec()
        }
        None => (0..n * q_total * d).map(|_| rng.kv_value(0.0)).collect(),
    };
    let gate: Vec<f32> = (0..n * q_total * d).map(|_| 2.0 * rng.gaussian()).collect();
    let q_dev = upload(device, &f32_bytes(&q));
    let gate_dev = upload(device, &f32_bytes(&gate));
    let out_dev = zeroed(device, (n * q_total * d * 4) as u64);

    // One table array for all sequences.
    let mut table = Vec::new();
    let offsets: Vec<u32> = seqs
        .iter()
        .map(|s| {
            let at = table.len() as u32;
            table.extend_from_slice(&s.table);
            at
        })
        .collect();
    let table_dev = upload(device, &u32_bytes(&table));

    let sm = device.info().multiprocessors;
    let mut planner = Planner::new(*geo, PlanConfig::for_multiprocessors(sm), n as u32);
    let mut row0 = 0usize;
    let mut checks: Vec<(usize, usize, usize, Option<std::ops::Range<usize>>)> = Vec::new(); // seq, row, pos, ranges
    for (i, look) in looks.iter().enumerate() {
        let seq = seqs[i];
        match look {
            Look::Window(_) | Look::At(..) => {
                let (w, pos0) = match look {
                    Look::Window(w) => (w, seq.len - w),
                    Look::At(p, w) => (w, *p),
                    Look::Ranges(_) => unreachable!(),
                };
                planner.dense(offsets[i], &seq.table, row0 as u32, *w as u32, pos0 as u32);
                let stride = (*w).div_ceil(check.max(1)).max(1);
                for r in (0..*w).filter(|r| r % stride == 0 || *r == w - 1) {
                    checks.push((i, row0 + r, pos0 + r, None));
                }
                row0 += w;
            }
            Look::Ranges(ranges) => {
                let r32: Vec<std::ops::Range<u32>> =
                    ranges.iter().map(|r| r.start as u32..r.end as u32).collect();
                planner.sparse(offsets[i], &seq.table, row0 as u32, (seq.len - 1) as u32, &r32);
                checks.push((i, row0, seq.len - 1, Some(0..0)));
                row0 += 1;
            }
        }
    }
    let plan = planner.finish();
    let scratch = zeroed(device, plan.state_bytes(geo));
    let buffers = plan.upload(device).unwrap();
    let args = AttendBuffers {
        tables: table_dev.ptr(),
        exact_pool: pools.exact.ptr(),
        converted_pool: pools.converted.ptr(),
        far_pool: ares_cuda::DevicePtr::NULL,
        q: q_dev.ptr(),
        gate: gated.then(|| gate_dev.ptr()),
        out: out_dev.ptr(),
        scratch: scratch.ptr(),
    };
    let scale = 1.0 / (d as f32).sqrt();
    // SAFETY: the buffers have the sizes the plan and the geometry give.
    unsafe { kernels.attend(stream, geo, &buffers, &args, scale).unwrap() };
    stream.synchronize().unwrap();
    let got: Vec<f32> = download(&out_dev, n * q_total * d * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let mut worst = Error::default();
    for (i, row, pos, sparse) in checks {
        let visible: Vec<std::ops::Range<usize>> = match (&looks[i], sparse) {
            (Look::Ranges(r), Some(_)) => r.clone(),
            _ => std::iter::once(0..pos + 1).collect(),
        };
        for kvh in 0..geo.kv_heads as usize {
            for gi in 0..g {
                let hg = (geo.first_kv_head as usize + kvh) * g + gi;
                let at = (row * q_total + hg) * d;
                let reference = seqs[i].heads[kvh].attend(&q[at..at + d], scale, &visible).unwrap();
                let mut want: Vec<f64> = reference.output.iter().map(|x| f64::from(*x)).collect();
                if gated {
                    for (w, gt) in want.iter_mut().zip(&gate[at..at + d]) {
                        *w *= 1.0 / (1.0 + (-f64::from(*gt)).exp());
                    }
                }
                let big = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
                let rms = (want.iter().map(|x| x * x).sum::<f64>() / d as f64).sqrt().max(1e-30);
                let (mut max, mut sq) = (0.0f64, 0.0f64);
                for (w, k) in want.iter().zip(&got[at..at + d]) {
                    let diff = (w - f64::from(*k)).abs();
                    max = max.max(diff);
                    sq += diff * diff;
                }
                worst = worst
                    .worse(Error { max_rel: max / big, rms_rel: (sq / d as f64).sqrt() / rms });
            }
        }
    }
    worst
}

/// Runs attention for the rows of the window of `case`.
#[allow(clippy::too_many_arguments)]
fn attend_and_compare(
    kernels: &AttentionKernels,
    stream: &Stream,
    case: &Case,
    geo: &CacheGeometry,
    pools: &Pools,
    seq: &DevSeq,
    gated: bool,
    rng: &mut Rng,
) -> Error {
    attend_many(kernels, stream, geo, pools, &[seq], &[Look::Window(case.rows)], gated, 16, rng)
}

fn test_geometry(k: KvFormat, v: KvFormat, d: usize) -> CacheGeometry {
    CacheGeometry {
        k: cache_format(k),
        v: cache_format(v),
        head_dim: d as u32,
        kv_heads: 2,
        kv_total: 3,
        first_kv_head: 1,
        q_per_kv: 6,
    }
}

#[test]
fn attention_equals_the_reference_for_every_pair_of_formats_in_decode() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(2024);
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut worst = Error::default();
        let mut bad = Vec::new();
        for k in KvFormat::ALL {
            for v in KvFormat::ALL {
                for (hist, rows) in [(900usize, 1usize), (1500, 4)] {
                    let case = Case { k, v, d: 256, hist, rows };
                    let geo = test_geometry(k, v, 256);
                    let (pools, seq) = build(&kernels, &stream, &case, &geo, &mut rng);
                    let err = attend_and_compare(
                        &kernels, &stream, &case, &geo, &pools, &seq, true, &mut rng,
                    );
                    if std::env::var_os("ARES_TEST_VERBOSE").is_some() {
                        println!(
                            "{} {k}/{v} hist {hist} rows {rows}: max {:.2e} rms {:.2e}",
                            device.info().name,
                            err.max_rel,
                            err.rms_rel
                        );
                    }
                    if err.max_rel >= 8e-3 {
                        bad.push(format!("{k}/{v} hist {hist}: max error {:.2e}", err.max_rel));
                    }
                    worst = worst.worse(err);
                }
            }
        }
        println!(
            "{}: worst over all pairs: max {:.2e}, rms {:.2e}",
            device.info().name,
            worst.max_rel,
            worst.rms_rel
        );
        assert!(bad.is_empty(), "{bad:#?}");
    }
}

/// The head size of 128 (rows of 4 groups, 8-byte aligned in the converted pool): one row
/// (the direct kernel where K and V are quantized), a window of 4 rows and a wide window.
#[test]
fn attention_with_a_head_size_of_128_equals_the_reference() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(128);
    let pairs = [
        (KvFormat::F16, KvFormat::F16),
        (KvFormat::Q8, KvFormat::Q8),
        (KvFormat::Q4, KvFormat::Q4),
        (KvFormat::Q6, KvFormat::Q5),
        (KvFormat::Q7, KvFormat::F16),
        (KvFormat::F16, KvFormat::Q5),
    ];
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut worst = Error::default();
        let mut bad = Vec::new();
        for (k, v) in pairs {
            for (hist, rows) in [(901usize, 1usize), (1500, 4), (700, 40)] {
                let case = Case { k, v, d: 128, hist, rows };
                let geo = test_geometry(k, v, 128);
                let (pools, seq) = build(&kernels, &stream, &case, &geo, &mut rng);
                let err = attend_and_compare(
                    &kernels, &stream, &case, &geo, &pools, &seq, true, &mut rng,
                );
                println!(
                    "{} head 128 {k}/{v} hist {hist} rows {rows}: max {:.2e} rms {:.2e}",
                    device.info().name,
                    err.max_rel,
                    err.rms_rel
                );
                if err.max_rel >= 8e-3 {
                    bad.push(format!("{k}/{v} hist {hist} rows {rows}: {:.2e}", err.max_rel));
                }
                worst = worst.worse(err);
            }
        }
        println!(
            "{}: head 128, worst: max {:.2e}, rms {:.2e}",
            device.info().name,
            worst.max_rel,
            worst.rms_rel
        );
        assert!(bad.is_empty(), "{bad:#?}");
    }
}

/// One case from the environment, for investigations:
/// `ARES_ATTN_CASE=k,v,hist,rows[,d]`.
#[test]
fn one_case_from_the_environment() {
    let Some(devices) = gpus() else { return };
    let Ok(spec) = std::env::var("ARES_ATTN_CASE") else { return };
    let f: Vec<&str> = spec.split(',').collect();
    let (k, v): (KvFormat, KvFormat) = (f[0].parse().unwrap(), f[1].parse().unwrap());
    let (hist, rows): (usize, usize) = (f[2].parse().unwrap(), f[3].parse().unwrap());
    let d: usize = f.get(4).map_or(256, |x| x.parse().unwrap());
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut rng = Rng(5);
        let case = Case { k, v, d, hist, rows };
        let mut geo = test_geometry(k, v, d);
        if let Ok(g) = std::env::var("ARES_ATTN_G") {
            geo.q_per_kv = g.parse().unwrap();
        }
        let (pools, seq) = build(&kernels, &stream, &case, &geo, &mut rng);
        for gated in [false, true] {
            let err =
                attend_and_compare(&kernels, &stream, &case, &geo, &pools, &seq, gated, &mut rng);
            println!(
                "{} {spec} gated {gated}: max {:.2e} rms {:.2e}",
                device.info().name,
                err.max_rel,
                err.rms_rel
            );
        }
    }
}

fn build_seq(
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: &CacheGeometry,
    pools: &mut Pools,
    len: usize,
    step: usize,
    rng: &mut Rng,
) -> DevSeq {
    let device = kernels.device();
    let policy = KvPolicy::new(cache_to_kv(geo.k), cache_to_kv(geo.v));
    let mut seq = DevSeq::new(device, geo, policy, len.div_ceil(TILE) + 4);
    let status = zeroed(device, 4);
    let total = geo.kv_total as usize;
    let mut left = len;
    while left > 0 {
        let n = left.min(step);
        let (k, v) = (
            rows_of(rng, n, total, geo.head_dim as usize),
            rows_of(rng, n, total, geo.head_dim as usize),
        );
        seq.append(kernels, stream, pools, &k, &v, false, &status);
        left -= n;
    }
    assert_eq!(status_of(&status), 0);
    seq
}

fn cache_to_kv(f: CacheFormat) -> KvFormat {
    KvFormat::ALL[f.id() as usize]
}

#[test]
fn batches_of_sequences_ranges_and_wide_windows_equal_the_reference() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(31337);
    let pairs = [
        (KvFormat::F16, KvFormat::F16),
        (KvFormat::Q8, KvFormat::Q8),
        (KvFormat::Q4, KvFormat::Q4),
        (KvFormat::Q6, KvFormat::Q5),
        (KvFormat::F16, KvFormat::Q7),
    ];
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let mut bad = Vec::new();
        for (k, v) in pairs {
            let geo = test_geometry(k, v, 256);
            let mut pools = Pools::new(device, geo, 60, 160);
            // Four sequences of different lengths, one of them with a window of four rows.
            let lens = [700usize, 2100, 1300, 330, 1100];
            let seqs: Vec<DevSeq> = lens
                .iter()
                .map(|&l| build_seq(&kernels, &stream, &geo, &mut pools, l, 600, &mut rng))
                .collect();
            let refs: Vec<&DevSeq> = seqs.iter().collect();
            let looks = [
                Look::Window(1),
                Look::Window(1),
                Look::Window(4),
                Look::Window(1),
                Look::Window(1),
            ];
            let err =
                attend_many(&kernels, &stream, &geo, &pools, &refs, &looks, true, 16, &mut rng);
            if err.max_rel > 8e-3 {
                bad.push(format!("{k}/{v} batch: {:.2e}", err.max_rel));
            }
            println!(
                "{} {k}/{v} batch of 5 sequences: max {:.2e} rms {:.2e}",
                device.info().name,
                err.max_rel,
                err.rms_rel
            );
            // A row that sees selected blocks of 4 cells (and its own newest cells), the kind of
            // list qwen4exp selects, over a sequence of 3000 cells.
            let long = build_seq(&kernels, &stream, &geo, &mut pools, 3000, 700, &mut rng);
            let mut ranges = Vec::new();
            let mut block = 0usize;
            while block < 2900 / 4 {
                block += 1 + (rng.next() % 5) as usize;
                if block * 4 + 4 <= 2990 {
                    ranges.push(block * 4..block * 4 + 4);
                }
            }
            ranges.dedup();
            ranges.push(2996..3000);
            let err = attend_many(
                &kernels,
                &stream,
                &geo,
                &pools,
                &[&long],
                &[Look::Ranges(ranges.clone())],
                false,
                1,
                &mut rng,
            );
            if err.max_rel > 8e-3 {
                bad.push(format!("{k}/{v} ranges: {:.2e}", err.max_rel));
            }
            println!(
                "{} {k}/{v} {} ranges: max {:.2e} rms {:.2e}",
                device.info().name,
                ranges.len(),
                err.max_rel,
                err.rms_rel
            );
            // Sequences with a common prefix (the same entries, seen at three lengths, one of
            // them with two rows): the planner reads the prefix once, as items with the rows
            // of all three, and each tail on its own.
            let err = attend_many(
                &kernels,
                &stream,
                &geo,
                &pools,
                &[&long, &long, &long],
                &[Look::At(2990, 1), Look::At(2600, 1), Look::At(1500, 2)],
                true,
                4,
                &mut rng,
            );
            if err.max_rel > 8e-3 {
                bad.push(format!("{k}/{v} shared prefix: {:.2e}", err.max_rel));
            }
            println!(
                "{} {k}/{v} shared prefix: max {:.2e} rms {:.2e}",
                device.info().name,
                err.max_rel,
                err.rms_rel
            );
            // A wide window.
            let wide = build_seq(&kernels, &stream, &geo, &mut pools, 1400, 700, &mut rng);
            let err = attend_many(
                &kernels,
                &stream,
                &geo,
                &pools,
                &[&wide],
                &[Look::Window(300)],
                true,
                12,
                &mut rng,
            );
            if err.max_rel > 8e-3 {
                bad.push(format!("{k}/{v} wide: {:.2e}", err.max_rel));
            }
            println!(
                "{} {k}/{v} wide window of 300 rows: max {:.2e} rms {:.2e}",
                device.info().name,
                err.max_rel,
                err.rms_rel
            );
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }
}

/// The outputs of the rows of one step that `plan` describes (`n` rows of one
/// sequence, whose table is `table`), for the queries and gates given:
/// `[row][q_heads][D]`.
#[allow(clippy::too_many_arguments)]
fn attend_planned(
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: &CacheGeometry,
    pools: &Pools,
    table: &[u32],
    cfg: PlanConfig,
    n: usize,
    q: &[f32],
    gate: &[f32],
    plan: impl FnOnce(&mut Planner),
) -> Vec<f32> {
    let device = kernels.device();
    let values = n * (geo.kv_total * geo.q_per_kv * geo.head_dim) as usize;
    assert_eq!((q.len(), gate.len()), (values, values));
    let (q_dev, gate_dev) = (upload(device, &f32_bytes(q)), upload(device, &f32_bytes(gate)));
    let out_dev = zeroed(device, values as u64 * 4);
    let table_dev = upload(device, &u32_bytes(table));
    let mut planner = Planner::new(*geo, cfg, n as u32);
    plan(&mut planner);
    let plan = planner.finish();
    let scratch = zeroed(device, plan.state_bytes(geo).max(16));
    let buffers = plan.upload(device).unwrap();
    let args = AttendBuffers {
        tables: table_dev.ptr(),
        exact_pool: pools.exact.ptr(),
        converted_pool: pools.converted.ptr(),
        far_pool: ares_cuda::DevicePtr::NULL,
        q: q_dev.ptr(),
        gate: Some(gate_dev.ptr()),
        out: out_dev.ptr(),
        scratch: scratch.ptr(),
    };
    let scale = 1.0 / (geo.head_dim as f32).sqrt();
    // SAFETY: the buffers have the sizes the plan and the geometry give.
    unsafe { kernels.attend(stream, geo, &buffers, &args, scale).unwrap() };
    stream.synchronize().unwrap();
    download(&out_dev, values * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The rows of the model's heads on this device, of `[row][q_heads][D]`: the
/// kernels write no other.
fn own_heads(geo: &CacheGeometry, rows: &[f32], n: usize) -> Vec<u32> {
    let (g, d) = (geo.q_per_kv as usize, geo.head_dim as usize);
    let per_row = geo.kv_total as usize * g * d;
    let first = geo.first_kv_head as usize * g * d;
    let len = geo.kv_heads as usize * g * d;
    (0..n)
        .flat_map(|r| rows[r * per_row + first..r * per_row + first + len].iter())
        .map(|x| x.to_bits())
        .collect()
}

/// A row of a window planned with [`Planner::window`] has, bit for bit, the
/// output it has as the only row of a step: for every pair of formats, for
/// windows of one tile of vectors and of several, for a window that begins a
/// new entry (where the pieces of a row change), and with as few pieces as
/// make a piece span several entries. Drafts rely on this: what a window
/// holds besides a row must not change the row.
#[test]
fn a_row_of_a_window_has_the_bits_it_has_alone() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0xA104E);
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let sm = device.info().multiprocessors;
        let mut compared = 0usize;
        for k in KvFormat::ALL {
            for v in KvFormat::ALL {
                // A head of 256 with 6 query heads per KV head (one row is 6
                // vectors, 8 rows fill the widest decode shape), and one of 128
                // with 4 for a few pairs.
                let small = (k == v) || (k == KvFormat::Q8 && v == KvFormat::F16);
                for d in [256usize, 128] {
                    if d == 128 && !small {
                        continue;
                    }
                    let mut geo = test_geometry(k, v, d);
                    if d == 128 {
                        geo.q_per_kv = 4;
                    }
                    // 1500 cells and then the window: with a tile mean the
                    // first tile and the newest are exact, the others converted.
                    let hist = 1500usize;
                    let case = Case { k, v, d, hist, rows: 40 };
                    let (pools, seq) = build(&kernels, &stream, &case, &geo, &mut rng);
                    let per_row = (geo.kv_total * geo.q_per_kv) as usize * d;
                    // (first position, rows): inside an entry, across the entry
                    // that begins at 1536, more rows than one tile of vectors.
                    for (pos0, rows) in [(1501usize, 2usize), (1505, 8), (1530, 8), (1520, 19)] {
                        for cfg in [
                            PlanConfig::for_multiprocessors(sm),
                            PlanConfig { target_blocks: 8, ..PlanConfig::for_multiprocessors(sm) },
                        ] {
                            let q: Vec<f32> =
                                (0..rows * per_row).map(|_| rng.kv_value(0.0)).collect();
                            let gate: Vec<f32> =
                                (0..rows * per_row).map(|_| 2.0 * rng.gaussian()).collect();
                            let together = attend_planned(
                                &kernels,
                                &stream,
                                &geo,
                                &pools,
                                &seq.table,
                                cfg,
                                rows,
                                &q,
                                &gate,
                                |p| p.window(0, &seq.table, 0, rows as u32, pos0 as u32, None),
                            );
                            for r in 0..rows {
                                let range = r * per_row..(r + 1) * per_row;
                                let alone = attend_planned(
                                    &kernels,
                                    &stream,
                                    &geo,
                                    &pools,
                                    &seq.table,
                                    cfg,
                                    1,
                                    &q[range.clone()],
                                    &gate[range.clone()],
                                    |p| p.window(0, &seq.table, 0, 1, (pos0 + r) as u32, None),
                                );
                                assert!(
                                    own_heads(&geo, &alone, 1).iter().any(|x| *x != 0),
                                    "the row was computed"
                                );
                                assert_eq!(
                                    own_heads(&geo, &together[range], 1),
                                    own_heads(&geo, &alone, 1),
                                    "{} {k}/{v} d {d}: row {r} of the {rows} rows at {pos0}, \
                                     {} blocks",
                                    device.info().name,
                                    cfg.target_blocks
                                );
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        println!("{}: {compared} rows of windows equal the rows alone", device.info().name);
    }
}

/// A window can read one tile from an entry it names instead of the table's:
/// the rows that read a tile converted while the table still has its exact
/// entry for the rows before them. Such rows have the bits they have once the
/// table names the converted entry.
#[test]
fn a_window_reads_a_swapped_tile_as_if_the_table_named_it() {
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(0x5A9);
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        let cfg = PlanConfig::for_multiprocessors(device.info().multiprocessors);
        for format in [KvFormat::Q7, KvFormat::Q4] {
            let (d, rows) = (256usize, 6usize);
            let geo = test_geometry(format, format, d);
            // 400 cells: every tile is still exact with q4 (256 newest cells),
            // and tile 2 is the last full one, exact with q7 too (32 newest).
            let case = Case { k: format, v: format, d, hist: 400 - rows, rows };
            let (mut pools, seq) = build(&kernels, &stream, &case, &geo, &mut rng);
            let tile = 2usize;
            let exact = seq.table[tile];
            assert_eq!(exact & CONVERTED, 0, "{format}: tile {tile} is an exact entry");
            // Convert the tile into a free converted slot; the table keeps the
            // exact entry.
            let converted = pools.free_converted.pop().unwrap();
            let jobs = upload(device, &u32_bytes(&[exact, converted]));
            let status = zeroed(device, 4);
            // SAFETY: both slots are in the pools.
            unsafe {
                kernels
                    .convert(
                        &stream,
                        &geo,
                        jobs.ptr(),
                        1,
                        pools.exact.ptr(),
                        pools.converted.ptr(),
                        ares_cuda::DevicePtr::NULL,
                        status.ptr(),
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            assert_eq!(status_of(&status), 0);
            let per_row = (geo.kv_total * geo.q_per_kv) as usize * d;
            let q: Vec<f32> = (0..rows * per_row).map(|_| rng.kv_value(0.0)).collect();
            let gate: Vec<f32> = (0..rows * per_row).map(|_| 2.0 * rng.gaussian()).collect();
            let pos0 = (400 - rows) as u32;
            let swap = Some((tile as u32, converted | CONVERTED));
            let run = |table: &[u32], swap: Option<(u32, u32)>| {
                attend_planned(&kernels, &stream, &geo, &pools, table, cfg, rows, &q, &gate, |p| {
                    p.window(0, table, 0, rows as u32, pos0, swap)
                })
            };
            let swapped = run(&seq.table, swap);
            let mut named = seq.table.clone();
            named[tile] = converted | CONVERTED;
            let by_table = run(&named, None);
            let unswapped = run(&seq.table, None);
            assert_eq!(
                own_heads(&geo, &swapped, rows),
                own_heads(&geo, &by_table, rows),
                "{} {format}",
                device.info().name
            );
            assert_ne!(
                own_heads(&geo, &swapped, rows),
                own_heads(&geo, &unswapped, rows),
                "{format}: the converted tile is not the exact one"
            );
        }
    }
}

fn read_f32_file(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The captured Q, K and V of Qwen3.8-27B (3686 tokens, 24 query heads, 4 KV heads of 256
/// values; `crates/quant/tools/kv_real/README.md`): attention of rows in the middle and at
/// the end of the text, against the reference on the same stored bytes.
#[test]
fn attention_on_the_captured_rows_of_a_real_model_equals_the_reference() {
    let dir = std::path::Path::new("/var/tmp/kvreal-data/q27b");
    if !dir.exists() {
        eprintln!("skipped: no captured data at {}", dir.display());
        return;
    }
    let Some(devices) = gpus() else { return };
    let mut rng = Rng(1);
    let pairs = [
        (KvFormat::F16, KvFormat::F16),
        (KvFormat::Q8, KvFormat::Q8),
        (KvFormat::Q6, KvFormat::Q5),
        (KvFormat::Q4, KvFormat::Q4),
        (KvFormat::Q4, KvFormat::Q8),
    ];
    for device in devices {
        let kernels = AttentionKernels::load(device).unwrap();
        let stream = Stream::new(device, 0).unwrap();
        for layer in [3usize, 31, 63] {
            let k = read_f32_file(&dir.join(format!("L{layer:02}_k.f32")));
            let v = read_f32_file(&dir.join(format!("L{layer:02}_v.f32")));
            let q = read_f32_file(&dir.join(format!("L{layer:02}_q.f32")));
            let n_tokens = k.len() / (4 * 256);
            let q_row = 24 * 256;
            for (kf, vf) in pairs {
                let geo = CacheGeometry {
                    k: cache_format(kf),
                    v: cache_format(vf),
                    head_dim: 256,
                    kv_heads: 4,
                    kv_total: 4,
                    first_kv_head: 0,
                    q_per_kv: 6,
                };
                let mut pools = Pools::new(device, geo, 12, 40);
                let mut seq = DevSeq::new(device, &geo, KvPolicy::new(kf, vf), 40);
                let status = zeroed(device, 4);
                let mut at = 0;
                while at < n_tokens {
                    let n = 700.min(n_tokens - at);
                    let (a, b) = (at * 1024, (at + n) * 1024);
                    seq.append(&kernels, &stream, &mut pools, &k[a..b], &v[a..b], false, &status);
                    at += n;
                }
                // Rows: one and two in the middle, a window of 8 at the end.
                let spans = [(1500usize, 1usize), (2900, 2), (n_tokens - 8, 8)];
                let looks: Vec<Look> = spans.iter().map(|&(p, n)| Look::At(p, n)).collect();
                let mut qs = Vec::new();
                for (p, n) in spans {
                    qs.extend_from_slice(&q[p * q_row..(p + n) * q_row]);
                }
                let err = attend_many_q(
                    &kernels,
                    &stream,
                    &geo,
                    &pools,
                    &[&seq, &seq, &seq],
                    &looks,
                    false,
                    16,
                    Some(&qs),
                    &mut rng,
                );
                println!(
                    "{} real data, layer {layer}, {kf}/{vf}: max {:.2e} rms {:.2e}",
                    device.info().name,
                    err.max_rel,
                    err.rms_rel
                );
                assert!(err.max_rel < 5e-3, "{kf}/{vf} layer {layer}: {:.2e}", err.max_rel);
            }
        }
    }
}
