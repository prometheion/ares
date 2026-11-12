//! Tests of the exact pool and the group's table on a device: sequences go
//! through the page pool, the kernels write their rows, and what is stored is
//! compared with what `ares_quant::KvHead` stores for the same rows.
//!
//! They open the GPUs, so they run only with `ARES_TEST_CUDA=1`:
//!
//! ```sh
//! ARES_TEST_CUDA=1 cargo test --release -p ares-model --lib kv:: -- --test-threads=1
//! ```

use std::sync::OnceLock;

use ares_core::Ledger;
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
    DevicePtr, PlanConfig, Planner, Stream, Wait, device_count,
};
use ares_quant::{KvFormat, KvHead, KvPolicy};
use ares_state::{PagePool, Sequence};

use super::super::forms::Reach;
use super::{Layer, Pages};
use crate::qwen35::device::PAGE_ENTRY;

const TILE: usize = 128;

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

/// A deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn gaussian(&mut self) -> f32 {
        let unit = |r: &mut Rng| ((r.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        let u1 = (unit(self) + 1.0) * 0.5 + 1e-7;
        let u2 = unit(self);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::PI * u2).cos()
    }
}

fn cache_format(f: KvFormat) -> CacheFormat {
    CacheFormat::ALL[KvFormat::ALL.iter().position(|x| *x == f).unwrap()]
}

fn download(buffer: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buffer.len() as usize];
    buffer.read(0, &mut out).unwrap();
    out
}

fn bytes_of<T: Copy>(values: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| f(*v)).collect()
}

fn upload(device: &Device, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64).unwrap();
    buffer.write(0, bytes).unwrap();
    buffer
}

fn zeroed(device: &Device, len: u64) -> DeviceBuffer {
    let buffer = DeviceBuffer::new(device, len.max(16)).unwrap();
    buffer.write(0, &vec![0u8; len.max(16) as usize]).unwrap();
    buffer
}

/// The rows of the cell whose text up to it hashes to `path`: K and V of every
/// KV head, `[head][D]`.
fn cell_rows(path: u64, heads: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(path | 1);
    for _ in 0..4 {
        rng.next();
    }
    let channel: Vec<f32> = (0..heads * d).map(|_| rng.gaussian()).collect();
    let row = |rng: &mut Rng| -> Vec<f32> {
        channel
            .iter()
            .map(|c| {
                let v = 0.5 * c + rng.gaussian();
                if rng.next().is_multiple_of(300) { v * 12.0 } else { v }
            })
            .collect()
    };
    (row(&mut rng), row(&mut rng))
}

fn mix(path: u64, token: u32) -> u64 {
    let mut h = path ^ (u64::from(token) + 0x9e37_79b9_7f4a_7c15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// One sequence in a slot: its pages and the hash of the text up to each cell.
struct Seq {
    slot: u32,
    cells: Sequence,
    path: Vec<u64>,
    /// The reference, for a sequence that does not share its pages.
    heads: Vec<KvHead>,
}

struct Rig<'a> {
    kernels: AttentionKernels,
    stream: Stream,
    geo: CacheGeometry,
    policy: KvPolicy,
    pages: Pages,
    converted: DeviceBuffer,
    status: DeviceBuffer,
    pool: PagePool,
    page_tab: Vec<u32>,
    stride: u32,
    device: &'a Device,
}

impl<'a> Rig<'a> {
    fn new(
        device: &'a Device,
        k: KvFormat,
        v: KvFormat,
        d: u32,
        exact: u32,
        pages: u32,
        slots: u32,
    ) -> Self {
        let geo = CacheGeometry {
            k: cache_format(k),
            v: cache_format(v),
            head_dim: d,
            kv_heads: 2,
            kv_total: 2,
            first_kv_head: 0,
            q_per_kv: 4,
        };
        let policy = KvPolicy::new(k, v);
        let stride = 24;
        let mut ledger = Ledger::new("test", 1 << 34);
        let group = Pages::new(
            device,
            &mut ledger,
            "test",
            &[geo],
            exact,
            policy.exact_newest_cells as u32,
            pages,
            slots,
            stride,
            true,
        )
        .unwrap();
        Self {
            kernels: AttentionKernels::load(device).unwrap(),
            stream: Stream::new(device, 0).unwrap(),
            geo,
            policy,
            pages: group,
            converted: zeroed(device, u64::from(pages) * geo.converted_entry_bytes()),
            status: zeroed(device, 4),
            pool: PagePool::new(pages, TILE as u32),
            page_tab: vec![0; (slots * stride) as usize],
            stride,
            device,
        }
    }

    fn layer(&self) -> Layer {
        Layer { geo: self.geo, pool: 0, exact: 0 }
    }

    fn seq(&self, slot: u32) -> Seq {
        let heads = (0..self.geo.kv_heads)
            .map(|_| KvHead::new(self.policy, self.geo.head_dim as usize).unwrap())
            .collect();
        Seq { slot, cells: Sequence::new(), path: Vec::new(), heads }
    }

    /// Tells the group what the pool freed since the last step.
    fn frees(&mut self) {
        let mut freed = Vec::new();
        self.pool.take_freed(1024, &mut freed);
        self.pages.release(freed);
    }

    /// Writes the conductor's page table for the pages `seq` has from `first`.
    fn tabulate(&mut self, seq: &Seq, first: usize) {
        for (i, page) in seq.cells.pages().iter().enumerate().skip(first) {
            self.page_tab[(seq.slot * self.stride) as usize + i] = page.0 | PAGE_ENTRY;
        }
    }

    /// One step of the sequences: `(sequence, tokens, a draft window?)`. Returns
    /// nothing; a draft window is settled with [`Rig::accept`].
    fn step(&mut self, ops: &mut [(&mut Seq, Vec<u32>, bool)]) {
        let mut reaches = Vec::new();
        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut rows = Vec::new();
        let (mut k_all, mut v_all) = (Vec::new(), Vec::new());
        let d = self.geo.head_dim as usize;
        let heads = self.geo.kv_heads as usize;
        for (seq, tokens, draft) in ops.iter_mut() {
            let first = seq.cells.len() as u32;
            let grown = seq.cells.append(&mut self.pool, tokens.len() as u64).unwrap();
            let dirty_from = grown.first_cell as usize / TILE;
            if let Some(copy) = grown.copy {
                self.frees();
                let layer = [self.layer()];
                self.pages
                    .copy_page(
                        &self.kernels,
                        &self.stream,
                        self.converted.ptr(),
                        self.status.ptr(),
                        &layer,
                        copy.from.0,
                        copy.to.0,
                    )
                    .unwrap();
            }
            self.tabulate(seq, dirty_from);
            for (i, token) in tokens.iter().enumerate() {
                let before = seq.path.last().copied().unwrap_or(0);
                let path = mix(before, *token);
                seq.path.push(path);
                let (k, v) = cell_rows(path, heads, d);
                for (h, head) in seq.heads.iter_mut().enumerate() {
                    if *draft {
                        head.append_pending(&k[h * d..(h + 1) * d], &v[h * d..(h + 1) * d])
                            .unwrap();
                    } else {
                        head.append(&k[h * d..(h + 1) * d], &v[h * d..(h + 1) * d]).unwrap();
                    }
                }
                k_all.extend(k);
                v_all.extend(v);
                rows.push(AttnRow {
                    table: seq.slot * self.stride,
                    pos: first + i as u32,
                    slot_base: 0,
                    n_slots: 0,
                });
            }
            reaches.push(Reach {
                slot: seq.slot,
                pos0: first,
                rows: tokens.len() as u32,
                writes: true,
                settles: !*draft,
            });
            plan.push((first as usize, tokens.len()));
        }
        self.frees();
        let jobs =
            self.pages.prepare(&self.stream, &reaches, &self.page_tab, DevicePtr::NULL).unwrap();
        let layer = [self.layer()];
        self.pages
            .convert(
                &self.kernels,
                &self.stream,
                self.converted.ptr(),
                self.status.ptr(),
                &layer,
                &jobs,
                1,
            )
            .unwrap();
        let rows_dev = upload(self.device, &AttnRow::to_bytes(&rows));
        let k_dev = upload(self.device, &bytes_of(&k_all, f32::to_le_bytes));
        let v_dev = upload(self.device, &bytes_of(&v_all, f32::to_le_bytes));
        // SAFETY: the buffers have the sizes the geometry gives; the table has
        // the entries the rows name.
        unsafe {
            self.kernels
                .append(
                    &self.stream,
                    &self.geo,
                    rows_dev.ptr(),
                    rows.len() as u32,
                    k_dev.ptr(),
                    v_dev.ptr(),
                    self.pages.table(DevicePtr::NULL),
                    self.pages.exact(),
                    self.converted.ptr(),
                    self.status.ptr(),
                )
                .unwrap();
        }
        self.stream.synchronize().unwrap();
    }

    /// Settles a draft window: the first `keep` of its rows are accepted.
    fn accept(&mut self, seq: &mut Seq, rows: usize, keep: usize) {
        let len = seq.cells.len() as usize - (rows - keep);
        seq.cells.truncate(&mut self.pool, len as u64);
        seq.path.truncate(len);
        for head in &mut seq.heads {
            head.truncate(len).unwrap();
            head.settle().unwrap();
        }
        self.pages.settled(seq.slot, len as u32, &self.page_tab);
    }

    /// The step a sequence would start with, which converts what left its
    /// window.
    fn flush(&mut self, seq: &Seq) {
        self.frees();
        let r = Reach {
            slot: seq.slot,
            pos0: seq.cells.len() as u32,
            rows: 0,
            writes: false,
            settles: false,
        };
        let jobs = self.pages.prepare(&self.stream, &[r], &self.page_tab, DevicePtr::NULL).unwrap();
        let layer = [self.layer()];
        self.pages
            .convert(
                &self.kernels,
                &self.stream,
                self.converted.ptr(),
                self.status.ptr(),
                &layer,
                &jobs,
                1,
            )
            .unwrap();
        self.stream.synchronize().unwrap();
    }

    /// The entry of tile `tile` of `seq` as the group's table names it.
    fn entry(&self, seq: &Seq, tile: usize) -> u32 {
        self.pages.entries(&self.page_tab)[(seq.slot * self.stride) as usize + tile]
    }

    /// Checks that every stored cell of `seq` is what its form should hold for
    /// the rows of its text: exact cells as f16, converted tiles as `encode_tile`
    /// writes them. With `policy`, also that the forms are those the policy asks
    /// for (a sequence that shares nothing).
    fn check(&self, seq: &Seq, policy: bool, label: &str) {
        let geo = self.geo;
        let (d, heads) = (geo.head_dim as usize, geo.kv_heads as usize);
        let exact = self.pages.exact_buffer().map(download).unwrap_or_default();
        let converted = download(&self.converted);
        let (ee, ce) = (geo.exact_entry_bytes() as usize, geo.converted_entry_bytes() as usize);
        let ch = geo.converted_head_bytes() as usize;
        let (rbk, rbv) =
            (geo.k.row_bytes(geo.head_dim) as usize, geo.v.row_bytes(geo.head_dim) as usize);
        let (km, vm) =
            (geo.k.mean_bytes(geo.head_dim) as usize, geo.v.mean_bytes(geo.head_dim) as usize);
        let len = seq.cells.len() as usize;
        if policy {
            let range = self.policy.quantized_cells(len);
            for tile in 0..len.div_ceil(TILE) {
                let converted = self.entry(seq, tile) & PAGE_ENTRY != 0;
                let want = range.contains(&(tile * TILE)) && self.policy.has_mean();
                assert_eq!(converted, want, "{label}: tile {tile} of {len} cells");
            }
        }
        // A reference for every tile that is converted: the tile alone, converted
        // as soon as it is full.
        let all = KvPolicy { exact_first_cells: 0, exact_newest_cells: 0, ..self.policy };
        let never =
            KvPolicy { exact_first_cells: usize::MAX / 2, exact_newest_cells: 0, ..self.policy };
        for tile in 0..len.div_ceil(TILE) {
            let entry = self.entry(seq, tile);
            let first = tile * TILE;
            let n = (len - first).min(TILE);
            for h in 0..heads {
                if entry & PAGE_ENTRY == 0 {
                    let mut reference = KvHead::new(never, d).unwrap();
                    for c in 0..n {
                        let (k, v) = cell_rows(seq.path[first + c], heads, d);
                        reference.append(&k[h * d..(h + 1) * d], &v[h * d..(h + 1) * d]).unwrap();
                    }
                    let base = entry as usize * ee + h * (TILE * 4 * d);
                    for c in 0..n {
                        let (rk, rv) = reference.exact(c).unwrap();
                        let (gk, gv) = (
                            &exact[base + c * 2 * d..base + (c + 1) * 2 * d],
                            &exact[base + TILE * 2 * d + c * 2 * d
                                ..base + TILE * 2 * d + (c + 1) * 2 * d],
                        );
                        assert!(gk == rk, "{label}: tile {tile} cell {c} head {h}: exact K");
                        assert!(gv == rv, "{label}: tile {tile} cell {c} head {h}: exact V");
                    }
                } else {
                    assert_eq!(n, TILE, "{label}: tile {tile} is converted before it is full");
                    let mut reference = KvHead::new(all, d).unwrap();
                    for c in 0..TILE {
                        let (k, v) = cell_rows(seq.path[first + c], heads, d);
                        reference.append(&k[h * d..(h + 1) * d], &v[h * d..(h + 1) * d]).unwrap();
                    }
                    let (unit, _) = reference.unit_of(0).unwrap();
                    let page = (entry & !PAGE_ENTRY) as usize;
                    let base = page * ce + h * ch;
                    assert!(
                        converted[base..base + km] == unit.k_mean[..],
                        "{label}: tile {tile} K mean"
                    );
                    assert!(
                        converted[base + km..base + km + TILE * rbk] == unit.k_rows[..],
                        "{label}: tile {tile} head {h}: converted K"
                    );
                    let vb = base + km + TILE * rbk;
                    assert!(
                        converted[vb..vb + vm] == unit.v_mean[..],
                        "{label}: tile {tile} V mean"
                    );
                    assert!(
                        converted[vb + vm..vb + vm + TILE * rbv] == unit.v_rows[..],
                        "{label}: tile {tile} head {h}: converted V"
                    );
                }
            }
        }
    }

    /// Attention of the last row of `seq` against the reference, relative to the
    /// largest output element.
    fn attention_error(&mut self, seq: &Seq) -> f64 {
        let geo = self.geo;
        let (d, g) = (geo.head_dim as usize, geo.q_per_kv as usize);
        let q_total = geo.kv_total as usize * g;
        let len = seq.cells.len() as usize;
        let mut rng = Rng(len as u64 + 5);
        let q: Vec<f32> = (0..q_total * d).map(|_| rng.gaussian()).collect();
        let q_dev = upload(self.device, &bytes_of(&q, f32::to_le_bytes));
        let out = zeroed(self.device, (q_total * d * 4) as u64);
        let table = seq.slot * self.stride;
        let entries = &self.pages.entries(&self.page_tab)[table as usize..];
        let mut planner = Planner::new(
            geo,
            PlanConfig::for_multiprocessors(self.device.info().multiprocessors),
            1,
        );
        planner.dense(table, entries, 0, 1, (len - 1) as u32);
        let plan = planner.finish();
        let scratch = zeroed(self.device, plan.state_bytes(&geo));
        let buffers = plan.upload(self.device).unwrap();
        let args = AttendBuffers {
            tables: self.pages.table(DevicePtr::NULL),
            exact_pool: self.pages.exact(),
            converted_pool: self.converted.ptr(),
            q: q_dev.ptr(),
            gate: None,
            out: out.ptr(),
            scratch: scratch.ptr(),
        };
        let scale = 1.0 / (d as f32).sqrt();
        // SAFETY: the buffers have the sizes the plan and the geometry give.
        unsafe { self.kernels.attend(&self.stream, &geo, &buffers, &args, scale).unwrap() };
        self.stream.synchronize().unwrap();
        let got: Vec<f32> =
            download(&out).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let mut worst = 0.0f64;
        for h in 0..geo.kv_heads as usize {
            for gi in 0..g {
                let hg = h * g + gi;
                let want = seq.heads[h]
                    .attend(&q[hg * d..(hg + 1) * d], scale, std::slice::from_ref(&(0..len)))
                    .unwrap()
                    .output;
                let big = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30) as f64;
                for (w, k) in want.iter().zip(&got[hg * d..(hg + 1) * d]) {
                    worst = worst.max((f64::from(*w) - f64::from(*k)).abs() / big);
                }
            }
        }
        worst
    }

    fn end(&mut self, seq: Seq) {
        seq.cells.release(&mut self.pool);
        self.frees();
    }
}

fn tokens(rng: &mut Rng, n: usize) -> Vec<u32> {
    (0..n).map(|_| rng.next() as u32).collect()
}

#[test]
fn a_lone_sequence_is_stored_the_way_kvhead_stores_it() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let pairs = [
            (KvFormat::Q4, KvFormat::Q4),
            (KvFormat::Q5, KvFormat::Q5),
            (KvFormat::Q7, KvFormat::Q6),
            (KvFormat::Q8, KvFormat::Q4),
            (KvFormat::F16, KvFormat::Q5),
        ];
        for (k, v) in pairs {
            for d in [256u32, 128] {
                let label = format!("{k}/{v} d={d} on {}", device.info().name);
                let policy = KvPolicy::new(k, v);
                let exact =
                    super::super::forms::exact_slots(policy.exact_newest_cells as u32, 1, 300, 0);
                let mut rig = Rig::new(device, k, v, d, exact, 16, 1);
                let mut seq = rig.seq(0);
                let mut rng = Rng(7);
                // A prompt in steps of various sizes, then decoding, a window of
                // drafts of which three are accepted, and more decoding.
                for n in [300usize, 1, 1, 150, 280, 1] {
                    let t = tokens(&mut rng, n);
                    rig.step(&mut [(&mut seq, t, false)]);
                }
                let t = tokens(&mut rng, 5);
                rig.step(&mut [(&mut seq, t, true)]);
                rig.accept(&mut seq, 5, 3);
                for _ in 0..70 {
                    let t = tokens(&mut rng, 1);
                    rig.step(&mut [(&mut seq, t, false)]);
                }
                rig.flush(&seq);
                rig.check(&seq, true, &label);
                let error = rig.attention_error(&seq);
                assert!(error < 6e-3, "{label}: attention is off by {error}");
                let (capacity, in_use) = rig.pages.slots();
                assert!(in_use <= capacity);
                rig.end(seq);
                let (_, in_use) = rig.pages.slots();
                assert_eq!(in_use, 0, "{label}: the slots of a sequence that ended are free");
            }
        }
    }
}

#[test]
fn sequences_that_share_pages_find_every_page_in_a_form_that_holds_its_text() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        for (k, v) in [(KvFormat::Q4, KvFormat::Q4), (KvFormat::Q5, KvFormat::Q5)] {
            let label = format!("{k}/{v} on {}", device.info().name);
            let policy = KvPolicy::new(k, v);
            let exact =
                super::super::forms::exact_slots(policy.exact_newest_cells as u32, 2, 300, 2);
            let mut rig = Rig::new(device, k, v, 256, exact, 40, 2);
            let mut rng = Rng(11);
            let mut a = rig.seq(0);
            for n in [300usize, 300, 100, 1, 1, 1] {
                let t = tokens(&mut rng, n);
                rig.step(&mut [(&mut a, t, false)]);
            }
            // B goes on from the end of A's text: it shares every page and copies
            // the partly used last one when it writes.
            let mut b = Seq {
                slot: 1,
                cells: a.cells.fork(&mut rig.pool),
                path: a.path.clone(),
                heads: Vec::new(),
            };
            rig.tabulate(&b, 0);
            let shared = a.cells.pages().to_vec();
            for (who, n) in [(1, 200usize), (0, 300), (1, 400), (0, 1), (1, 1)] {
                let t = tokens(&mut rng, n);
                if who == 0 {
                    rig.step(&mut [(&mut a, t, false)]);
                } else {
                    rig.step(&mut [(&mut b, t, false)]);
                }
            }
            assert_eq!(a.cells.pages()[..5], shared[..5], "{label}: the full pages stay shared");
            assert_ne!(a.cells.pages()[5], b.cells.pages()[5], "{label}: the last page was copied");
            rig.flush(&a);
            rig.check(&a, false, &format!("{label}: A"));
            rig.flush(&b);
            rig.check(&b, false, &format!("{label}: B"));
            // A step of both: neither finds a table that is out of date.
            let (ta, tb) = (tokens(&mut rng, 1), tokens(&mut rng, 1));
            rig.step(&mut [(&mut a, ta, false), (&mut b, tb, false)]);
            rig.flush(&a);
            rig.check(&a, false, &format!("{label}: A again"));
            rig.flush(&b);
            rig.check(&b, false, &format!("{label}: B again"));
            let (capacity, in_use) = rig.pages.slots();
            assert!(in_use <= capacity);
            rig.end(a);
            rig.flush(&b);
            rig.check(&b, false, &format!("{label}: B alone"));
            rig.end(b);
            assert_eq!(rig.pages.slots().1, 0, "{label}: nothing is left");
        }
    }
}

#[test]
fn a_cached_page_is_converted_to_make_room_and_keeps_its_text() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        let (k, v) = (KvFormat::Q4, KvFormat::Q4);
        let label = format!("{k}/{v} on {}", device.info().name);
        // Room for one sequence and a little more.
        let exact = super::super::forms::exact_slots(256, 1, 128, 0);
        let mut rig = Rig::new(device, k, v, 256, exact, 40, 2);
        let mut rng = Rng(21);
        let mut a = rig.seq(0);
        for _ in 0..8 {
            let t = tokens(&mut rng, 128);
            rig.step(&mut [(&mut a, t, false)]);
        }
        let t = tokens(&mut rng, 30);
        rig.step(&mut [(&mut a, t, false)]);
        let before = rig.pages.slots().1;
        // A ends; the cache keeps its pages, still exact where the window was.
        let mut cached = Seq {
            slot: 0,
            cells: a.cells.fork(&mut rig.pool),
            path: a.path.clone(),
            heads: Vec::new(),
        };
        rig.end(a);
        assert_eq!(rig.pages.slots().1, before, "{label}: the cache keeps the exact entries");
        // Another sequence needs the slots: pages of the cache are converted.
        let mut b = rig.seq(1);
        for _ in 0..6 {
            let t = tokens(&mut rng, 128);
            rig.step(&mut [(&mut b, t, false)]);
        }
        rig.flush(&b);
        rig.check(&b, false, &format!("{label}: B"));
        rig.flush(&cached);
        rig.check(&cached, false, &format!("{label}: the cache"));
        // A's own policy would convert tiles 1 to 5 of its 9 (the first and the
        // window of 1054 cells stay exact); pages beyond that were converted to
        // make room, and the partly used last page was not.
        let converted = (0..9).filter(|t| rig.entry(&cached, *t) & PAGE_ENTRY != 0).count();
        assert!(converted > 5, "{label}: {converted} of the cache's 9 tiles are converted");
        assert_eq!(rig.entry(&cached, 8) & PAGE_ENTRY, 0, "{label}: the last page is exact");
        cached.cells.truncate(&mut rig.pool, 0);
        rig.end(cached);
        rig.end(b);
        assert_eq!(rig.pages.slots().1, 0, "{label}: nothing is left");
    }
}

#[test]
fn the_table_of_a_new_group_is_zero_on_the_device_whatever_the_memory_held() {
    let Some(devices) = gpus() else { return };
    for device in devices {
        // Memory that held something when it was freed, of the sizes the
        // group's table takes.
        for _ in 0..4 {
            let dirty = DeviceBuffer::new(device, 96 * 4).unwrap();
            dirty.write(0, &[0xAB; 96 * 4]).unwrap();
        }
        let rig = Rig::new(device, KvFormat::Q4, KvFormat::Q4, 256, 6, 16, 4);
        let table = download(rig.pages.own_table().expect("a table of its own"));
        assert!(
            table.iter().all(|b| *b == 0),
            "{}: the table starts as its mirror does",
            device.info().name
        );
    }
}
