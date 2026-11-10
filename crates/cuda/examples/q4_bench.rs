//! Times the `qwen4exp` kernels (`Q4Kernels`) at the shapes of
//! Qwen3.8-Flash-Next: 48 layers, 4 streams of 2560, a mixer rank of 320, 12
//! attention layers with an indexer of 4 heads of 128 over blocks of 4 cells
//! and a budget of 2048 cells, an n-gram block of 16 rows of 160 values.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example q4_bench -- [device ...]
//! ```
//!
//! With `attention` as an argument it times instead the selection and the
//! attention over the selected lists through the attention kernels, with K
//! and V in q8/q8 and q5/q4.
//!
//! For one row, a window of 8 rows and a prompt piece of 1024 rows it prints
//! the time per launch of every kernel, the bytes the launch reads and writes
//! and the rate that gives, next to the time of a launch that does nothing
//! (the floor). The sums at the end are what the kernels cost per step of the
//! whole model. Inputs are pseudo-random; nothing is checked here (the tests
//! do that).

// The launches are on buffers allocated a few lines earlier with sizes from the same shapes.
#![allow(clippy::undocumented_unsafe_blocks)]

use ares_core::GgmlType;
use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnItem, AttnRow, AttnShape, CacheFormat, CacheGeometry,
    CommitSet, Device, DeviceBuffer, Event, IndexerShape, KeyFormat, LayerKernels, ListHandOver,
    NgramShape, Plan, Q4Kernels, Result, RopeParams, Segment, SelectionRule, StepRecord, Stream,
    Wait, quantized_activation_bytes,
};

const HC: u32 = 4;
const N: u32 = 2560;
const D: u32 = HC * N;
const RANK: u32 = 320;
const LAYERS: f64 = 48.0;
const ATTENTION_LAYERS: f64 = 12.0;
const EPS: f32 = 1e-6;
const INDEXER: IndexerShape =
    IndexerShape { n_head: 4, key_dim: 128, block: 4, top_k: 2048, rope_dim: 64 };
const NGRAM: NgramShape =
    NgramShape { n_stream: HC, n_embd: N, heads: 16, row_dim: 160, conv_kernel: 4, dilation: 3 };
const ROPE: RopeParams = RopeParams { dim: 64, freq_base: 10_000_000.0, sections: [11, 11, 10, 0] };

struct Rig {
    device: Device,
    stream: Stream,
    q4: Q4Kernels,
    start: Event,
    stop: Event,
    seed: u64,
}

impl Rig {
    fn next(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        self.seed
    }

    /// `n` f32 values in ±[0.5, 1).
    fn floats(&mut self, n: u64) -> DeviceBuffer {
        let data: Vec<u8> = (0..n)
            .flat_map(|_| (0x3f00_0000u32 | (self.next() as u32 & 0x807f_ffff)).to_le_bytes())
            .collect();
        self.bytes(&data)
    }

    /// `n` f16 values in ±[0.5, 1).
    fn halves(&mut self, n: u64) -> DeviceBuffer {
        let data: Vec<u8> = (0..n)
            .flat_map(|_| (0x3800u16 | (self.next() as u16 & 0x83ff)).to_le_bytes())
            .collect();
        self.bytes(&data)
    }

    fn bytes(&self, data: &[u8]) -> DeviceBuffer {
        let b = DeviceBuffer::new(&self.device, data.len().max(16) as u64).unwrap();
        b.write(0, data).unwrap();
        b
    }

    fn zeros(&self, bytes: u64) -> DeviceBuffer {
        let b = DeviceBuffer::new(&self.device, bytes.max(16)).unwrap();
        unsafe { self.stream.fill(b.ptr(), 0, bytes as usize).unwrap() };
        self.stream.synchronize().unwrap();
        b
    }

    /// Microseconds per launch: the median of five batches of `iters` launches between two
    /// events, after a warm-up.
    fn time(&self, iters: u32, mut launch: impl FnMut()) -> f64 {
        for _ in 0..iters.min(4) {
            launch();
        }
        self.stream.synchronize().unwrap();
        let mut batches: Vec<f64> = (0..5)
            .map(|_| {
                let wall = std::time::Instant::now();
                self.start.record(&self.stream).unwrap();
                for _ in 0..iters {
                    launch();
                }
                self.stop.record(&self.stream).unwrap();
                self.stop.synchronize().unwrap();
                let wall_us = wall.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
                let event_us = f64::from(self.stop.elapsed_ms_since(&self.start).unwrap()) * 1000.0
                    / f64::from(iters);
                // On a device shared with other processes the events sometimes report far too
                // little; the wall clock is then the better number.
                if event_us < 0.9 * wall_us { wall_us } else { event_us }
            })
            .collect();
        batches.sort_by(f64::total_cmp);
        batches[2]
    }
}

/// Prints one line and returns the time.
fn line(name: &str, us: f64, bytes: u64, floor: f64) -> f64 {
    let rate = bytes as f64 / 1e9 / ((us - floor).max(0.05) * 1e-6);
    println!(
        "    {name:<44} {us:>9.1} us   {:>9.2} MiB  {rate:>7.1} GB/s above the floor",
        bytes as f64 / (1 << 20) as f64
    );
    us
}

fn iters_for(rows: u32) -> u32 {
    if rows >= 512 { 20 } else { 200 }
}

/// The mixer and the n-gram block for a step of `rows` rows. Returns (mixer time per layer's
/// two reads and writes, n-gram time per step).
fn mixer_and_ngram(rig: &mut Rig, rows: u32, floor: f64) -> Result<(f64, f64)> {
    let it = iters_for(rows);
    let (rd, rn, rr) =
        (u64::from(rows) * u64::from(D), u64::from(rows) * u64::from(N), rows * RANK);
    let r = rig.floats(rd);
    let norm = rig.floats(u64::from(D));
    let xn = rig.floats(rd);
    let out = rig.floats(rn);
    let inj = rig.floats(u64::from(rows * HC));
    let gate = rig.floats(rd);
    let down = rig.floats(u64::from(rr));
    let lo = rig.zeros(u64::from(rr) * 4);
    let act = rig.zeros(quantized_activation_bytes(rn));
    let (q4, s) = (rig.q4.clone(), &rig.stream);
    let norm_us = rig.time(it, || unsafe {
        q4.stream_norm(s, r.ptr(), norm.ptr(), xn.ptr(), rows, HC, N, EPS).unwrap()
    });
    let write_us = rig.time(it, || unsafe {
        q4.stream_write(s, r.ptr(), out.ptr(), inj.ptr(), rows, HC, N).unwrap()
    });
    let write_norm_us = rig.time(it, || unsafe {
        q4.stream_write_norm(
            s,
            r.ptr(),
            out.ptr(),
            inj.ptr(),
            norm.ptr(),
            xn.ptr(),
            rows,
            HC,
            N,
            EPS,
        )
        .unwrap()
    });
    let squash_us = rig.time(it, || unsafe { q4.squash(s, down.ptr(), lo.ptr(), rr, HC).unwrap() });
    let mix_us = rig.time(it, || unsafe {
        q4.mix_quant(s, xn.ptr(), gate.ptr(), act.ptr(), rows, HC, N).unwrap()
    });
    line("stream_norm (R -> xn)", norm_us, rd * 8 + u64::from(D) * 4, floor);
    line("stream_write (R += out * f)", write_us, rd * 8 + rn * 4, floor);
    line("stream_write_norm (both)", write_norm_us, rd * 12 + rn * 4 + u64::from(D) * 4, floor);
    line("squash (down -> lo)", squash_us, u64::from(rr) * 8, floor);
    line("mix_quant (xn, gate -> m, quantized)", mix_us, rd * 8 + rn * 5 + rn / 4, floor);
    // A layer: norm, squash, mix for the first read; write + norm, squash, mix for the second;
    // the write of the experts' output.
    let per_layer = norm_us + write_norm_us + write_us + 2.0 * (squash_us + mix_us);

    let fetched = rig.halves(u64::from(rows) * NGRAM.fetched_row_bytes() / 2);
    let emb = rig.zeros(rn * 4);
    let key = rig.floats(rd);
    let value = rig.floats(rn);
    let weights = [rig.floats(u64::from(D)), rig.floats(u64::from(D)), rig.floats(u64::from(D))];
    let gated = rig.zeros(rd * 4);
    let nrm = rig.zeros(rd * 4);
    let taps = rig.halves(u64::from(D) * 4);
    let history = rig.floats(NGRAM.history_len());
    let segments = [Segment { row0: 0, n_rows: rows, slot: 0, pos0: 5000, direct: false }];
    let record = StepRecord::new(&segments, &vec![0u32; rows as usize], None, &[])?;
    let step = rig.bytes(record.as_bytes());
    let dims = record.dims();
    let s = &rig.stream;
    let decode_us = rig
        .time(it, || unsafe { q4.ngram_decode(s, NGRAM, fetched.ptr(), emb.ptr(), rows).unwrap() });
    let gate_us = rig.time(it, || unsafe {
        q4.ngram_gate(
            s,
            NGRAM,
            EPS,
            key.ptr(),
            value.ptr(),
            r.ptr(),
            weights[0].ptr(),
            weights[1].ptr(),
            weights[2].ptr(),
            gated.ptr(),
            nrm.ptr(),
            None,
            rows,
        )
        .unwrap()
    });
    let window_us = rig.time(it, || unsafe {
        q4.ngram_window(
            s,
            step.ptr(),
            dims,
            NGRAM,
            r.ptr(),
            gated.ptr(),
            nrm.ptr(),
            taps.ptr(),
            GgmlType::F16,
            history.ptr(),
            NGRAM.history_len(),
        )
        .unwrap()
    });
    let commit_us = rig.time(it, || unsafe {
        q4.ngram_commit(
            s,
            step.ptr(),
            dims,
            NGRAM,
            CommitSet::Deferred,
            nrm.ptr(),
            history.ptr(),
            NGRAM.history_len(),
        )
        .unwrap()
    });
    line(
        "ngram_decode (16 rows of IQ4_NL per row)",
        decode_us,
        u64::from(rows) * 1440 + rn * 4,
        floor,
    );
    line(
        "ngram_gate (key, value, R -> gated, nrm)",
        gate_us,
        rd * 16 + rn * 4 + u64::from(D) * 12,
        floor,
    );
    line("ngram_window (conv, R += ...)", window_us, rd * 28, floor);
    line("ngram_commit", commit_us, NGRAM.history_len() * 8, floor);
    Ok((per_layer, decode_us + gate_us + window_us + commit_us))
}

/// The indexer of one attention layer for `rows` rows that end at cell `cells - 1`. Returns
/// the time per attention layer.
fn indexer(
    rig: &mut Rig,
    layers: &LayerKernels,
    rows: u32,
    cells: u32,
    format: KeyFormat,
    floor: f64,
) -> Result<f64> {
    let it = iters_for(rows);
    let (id, ih) = (u64::from(INDEXER.key_dim), u64::from(INDEXER.n_head));
    let entries = cells.div_ceil(128);
    let n_blocks = cells / INDEXER.block;
    let key_values = u64::from(entries) * 32 * id;
    let pool = match format {
        KeyFormat::F32 => rig.floats(key_values),
        KeyFormat::F16 => rig.halves(key_values),
    };
    let table: Vec<u32> = (0..entries).collect();
    let key_table = rig.bytes(&table.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let c0 = cells - rows;
    let attn_rows: Vec<AttnRow> =
        (0..rows).map(|t| AttnRow { table: 0, pos: c0 + t, slot_base: 0, n_slots: 0 }).collect();
    let rows_b = rig.bytes(&AttnRow::to_bytes(&attn_rows));
    let segments = [Segment { row0: 0, n_rows: rows, slot: 0, pos0: c0 as i32, direct: false }];
    let record = StepRecord::new(&segments, &vec![0u32; rows as usize], None, &[])?;
    let step = rig.bytes(record.as_bytes());
    let dims = record.dims();
    let rope = rig.zeros(u64::from(rows) * 32 * 8);
    let iq = rig.floats(u64::from(rows) * ih * id);
    let raw = rig.floats(u64::from(rows) * id);
    let u = rig.zeros(u64::from(rows) * ih * id * 4);
    let (q_norm, k_norm) = (rig.floats(id), rig.floats(id));
    let carry = rig.floats(INDEXER.carry_len());
    // Rows are scored and selected in pieces that keep the score scratch below 64 MiB.
    let piece = rows.min(((64u64 << 20) / (u64::from(n_blocks) * 4)).max(1) as u32);
    let pieces = f64::from(rows.div_ceil(piece));
    let scores = rig.zeros(u64::from(piece) * u64::from(n_blocks) * 4);
    let cap = INDEXER.max_ranges(0);
    let lists = rig.zeros(u64::from(piece) * u64::from(cap) * 12);
    let counts = rig.zeros(u64::from(piece) * 16);
    let (plan_pieces, heads) = (8u32, 2u32);
    let hands: Vec<ListHandOver> = (0..piece)
        .map(|row| ListHandOver {
            ranges_off: row * cap * 3,
            item_conv: row * plan_pieces * heads,
            pieces_conv: plan_pieces,
            kv_heads: heads,
            ..ListHandOver::default()
        })
        .collect();
    let hands_b = rig.bytes(&ListHandOver::to_bytes(&hands));
    let items = rig.zeros(u64::from(piece * plan_pieces * heads) * 48);
    let (q4, s) = (rig.q4.clone(), &rig.stream);
    unsafe { layers.rope_table(s, step.ptr(), dims, rope.ptr(), ROPE)? };
    let window_us = rig.time(it, || unsafe {
        q4.indexer_window(
            s,
            step.ptr(),
            dims,
            INDEXER,
            format,
            EPS,
            rows_b.ptr(),
            iq.ptr(),
            raw.ptr(),
            q_norm.ptr(),
            k_norm.ptr(),
            rope.ptr(),
            u.ptr(),
            carry.ptr(),
            INDEXER.carry_len(),
            key_table.ptr(),
            pool.ptr(),
        )
        .unwrap()
    });
    let commit_us = rig.time(it, || unsafe {
        q4.indexer_commit(
            s,
            step.ptr(),
            dims,
            INDEXER,
            CommitSet::Deferred,
            rows_b.ptr(),
            raw.ptr(),
            rope.ptr(),
            carry.ptr(),
            INDEXER.carry_len(),
        )
        .unwrap()
    });
    let scores_us = rig.time(it, || unsafe {
        q4.block_scores(
            s,
            INDEXER,
            format,
            rows_b.ptr(),
            u.ptr(),
            key_table.ptr(),
            pool.ptr(),
            scores.ptr(),
            0,
            piece,
            n_blocks,
        )
        .unwrap()
    });
    // Repeating one launch reads the keys from the device's cache once they fit there, which a
    // step of the model does not: between two attention layers everything else has run. For few
    // rows the launch is therefore also timed behind a fill of 96 MiB, less the time of the fill.
    let key_bytes = u64::from(n_blocks) * id * format.bytes();
    let cold_us = if rows <= 8 && key_bytes >= (1 << 20) {
        let evict = rig.zeros(96 << 20);
        let fill_us = rig.time(it, || unsafe { s.fill(evict.ptr(), 0, 96 << 20).unwrap() });
        let both_us = rig.time(it, || unsafe {
            s.fill(evict.ptr(), 0, 96 << 20).unwrap();
            q4.block_scores(
                s,
                INDEXER,
                format,
                rows_b.ptr(),
                u.ptr(),
                key_table.ptr(),
                pool.ptr(),
                scores.ptr(),
                0,
                piece,
                n_blocks,
            )
            .unwrap()
        });
        Some(both_us - fill_us)
    } else {
        None
    };
    let select_us = rig.time(it, || unsafe {
        q4.select(
            s,
            INDEXER,
            SelectionRule::CellBudget,
            rows_b.ptr(),
            scores.ptr(),
            None,
            lists.ptr(),
            counts.ptr(),
            0,
            piece,
            n_blocks,
            cap,
        )
        .unwrap()
    });
    let hand_us = rig.time(it, || unsafe {
        q4.hand_over(s, hands_b.ptr(), counts.ptr(), lists.ptr(), items.ptr(), piece).unwrap()
    });
    let score_bytes = u64::from(piece) * u64::from(n_blocks) * 4;
    let label = |what: &str| {
        if piece < rows { format!("{what} ({piece} of the rows)") } else { what.to_owned() }
    };
    line(
        "indexer_window (queries, pooled keys)",
        window_us,
        u64::from(rows) * (ih * id * 8 + id * 8),
        floor,
    );
    line("indexer_commit", commit_us, 3 * id * 8, floor);
    // One row reads every key once; more rows read them again, mostly from the cache.
    line(&label("block_scores"), scores_us, key_bytes * u64::from(piece) + score_bytes, floor);
    if let Some(cold) = cold_us {
        line(
            "block_scores, keys not in the cache",
            cold,
            key_bytes * u64::from(piece) + score_bytes,
            floor,
        );
    }
    line(&label("select"), select_us, score_bytes * 6 + u64::from(piece) * 2051 * 3, floor);
    line(&label("hand_over"), hand_us, u64::from(piece) * u64::from(cap) * 12, floor);
    Ok(window_us + commit_us + pieces * (cold_us.unwrap_or(scores_us) + select_us + hand_us))
}

/// The attention of `rows` rows over lists made on the device, through the attention kernels
/// as they are: K and V of `cells` cells in converted entries of the formats `k` / `v`, random
/// scores (so a list is about 513 ranges of 4 cells scattered over the sequence), every row its
/// own list, `pieces` items per row and KV head. Prints the time of the selection and of the
/// attention and returns the latter.
fn attention_path(
    rig: &mut Rig,
    attention: &AttentionKernels,
    (k, v): (CacheFormat, CacheFormat),
    cells: u32,
    rows: u32,
    pieces: u32,
) -> Result<f64> {
    let (hd, heads, g) = (256u32, 2u32, 12u32);
    let geo = CacheGeometry {
        k,
        v,
        head_dim: hd,
        kv_heads: heads,
        kv_total: heads,
        first_kv_head: 0,
        q_per_kv: g,
    };
    let entries = cells.div_ceil(128);
    let n_blocks = cells / INDEXER.block;
    // One tile of random rows, encoded once per KV head for K and for V, and copied to every
    // entry: the layout of a converted slot is, per head, [K mean][K rows][V mean][V rows].
    let entry_bytes = geo.converted_entry_bytes();
    let pool = rig.zeros(entry_bytes * u64::from(entries));
    let exact = rig.zeros(geo.exact_entry_bytes());
    let status = rig.zeros(16);
    for head in 0..heads {
        let mut at = pool.ptr().offset(u64::from(head) * geo.converted_head_bytes());
        for format in [k, v] {
            let src = rig.floats(128 * u64::from(hd));
            let rows_at = at.offset(u64::from(format.mean_bytes(hd)));
            unsafe {
                attention.encode_tile_f32(
                    &rig.stream,
                    src.ptr(),
                    128,
                    hd,
                    format,
                    format.has_mean(),
                    at,
                    rows_at,
                    status.ptr(),
                )?;
            }
            rig.stream.synchronize()?;
            at = rows_at.offset(128 * u64::from(format.row_bytes(hd)));
        }
    }
    for e in 1..u64::from(entries) {
        unsafe {
            rig.stream.copy(pool.ptr().offset(e * entry_bytes), pool.ptr(), entry_bytes as usize)?
        };
    }
    let table: Vec<u32> = (0..entries).map(|e| e | 1 << 31).collect();
    let tables = rig.bytes(&table.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>());

    // The plan: every row's list is made on the device.
    let cap = INDEXER.max_ranges(0);
    let c0 = cells - rows;
    let single = pieces == 1;
    let mut plan = Plan {
        rows: (0..rows)
            .map(|t| AttnRow {
                table: 0,
                pos: c0 + t,
                slot_base: if single { 0 } else { t * pieces },
                n_slots: if single { 0 } else { pieces },
            })
            .collect(),
        row_ids: (0..rows).collect(),
        ranges: vec![0; (rows * cap * 3) as usize],
        state_slots: if single { 0 } else { rows * pieces },
        needs_merge: !single,
        ..Plan::default()
    };
    for row in 0..rows {
        for piece in 0..pieces {
            for kv_head in 0..heads {
                // Flags: 1 reads converted entries, 4 finishes its row (the only piece).
                let flags = 1 | if single { 4 } else { 0 };
                plan.items.push(AttnItem {
                    row0: row,
                    n_rows: 1,
                    kv_head,
                    flags,
                    slot: piece,
                    ..AttnItem::default()
                });
            }
        }
    }
    let shape = AttnShape::for_vecs(g);
    plan.groups.push((shape, 0..plan.items.len()));
    let hands: Vec<ListHandOver> = (0..rows)
        .map(|row| ListHandOver {
            ranges_off: row * cap * 3,
            item_conv: row * pieces * heads,
            pieces_conv: pieces,
            kv_heads: heads,
            ..ListHandOver::default()
        })
        .collect();
    let layout = plan.layout();
    let mut bytes = vec![0u8; layout.bytes as usize];
    plan.write(&mut bytes);
    let plan_b = rig.bytes(&bytes);
    let buffers = unsafe { plan.on_device(plan_b.ptr()) };
    let hands_b = rig.bytes(&ListHandOver::to_bytes(&hands));
    let scores = rig.floats(u64::from(rows) * u64::from(n_blocks));
    let counts = rig.zeros(u64::from(rows) * 16);
    let q = rig.floats(u64::from(rows) * u64::from(heads * g * hd));
    let out = rig.zeros(u64::from(rows) * u64::from(heads * g * hd) * 4);
    let scratch = rig.zeros(plan.state_bytes(&geo).max(16));
    let (q4, s) = (rig.q4.clone(), &rig.stream);
    let (rows_ptr, ranges_ptr) =
        (plan_b.ptr().offset(layout.rows), plan_b.ptr().offset(layout.ranges));
    let select = || unsafe {
        q4.select(
            s,
            INDEXER,
            SelectionRule::CellBudget,
            rows_ptr,
            scores.ptr(),
            Some(tables.ptr()),
            ranges_ptr,
            counts.ptr(),
            0,
            rows,
            n_blocks,
            cap,
        )
        .unwrap()
    };
    let it = if rows >= 512 { 8 } else { 100 };
    let select_us = rig.time(it, select);
    select();
    unsafe {
        q4.hand_over(
            s,
            hands_b.ptr(),
            counts.ptr(),
            ranges_ptr,
            plan_b.ptr().offset(layout.items),
            rows,
        )?
    };
    let attend_buffers = AttendBuffers {
        tables: tables.ptr(),
        exact_pool: exact.ptr(),
        converted_pool: pool.ptr(),
        q: q.ptr(),
        gate: None,
        out: out.ptr(),
        scratch: scratch.ptr(),
    };
    let attend_us = rig.time(it, || unsafe {
        attention.attend(s, &geo, &buffers, &attend_buffers, 1.0 / 16.0).unwrap()
    });
    // What a row reads: K and V of 2051 cells of both heads.
    let read =
        u64::from(rows) * 2051 * u64::from(heads) * u64::from(k.row_bytes(hd) + v.row_bytes(hd));
    println!(
        "    {k:?}/{v:?} {cells:>6} cells {rows:>4} rows x {pieces:>2} pieces ({shape:?}): select {select_us:>8.1} us, \
         attention {attend_us:>9.1} us = {:>6.1} us per row, {:.1} GB/s of K and V",
        attend_us / f64::from(rows),
        read as f64 / 1e9 / (attend_us * 1e-6)
    );
    Ok(attend_us)
}

fn main() -> Result<()> {
    let attention_only = std::env::args().any(|a| a == "attention");
    let args: Vec<u32> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let ordinals = if args.is_empty() { vec![0] } else { args };
    for ordinal in ordinals {
        let device = Device::open(ordinal, Wait::Block)?;
        println!("{} (device {ordinal})", device.info().name);
        if attention_only {
            let mut rig = Rig {
                stream: Stream::new(&device, 0)?,
                q4: Q4Kernels::load(&device)?,
                start: Event::timed(&device)?,
                stop: Event::timed(&device)?,
                device: device.clone(),
                seed: 0x9e37_79b9_7f4a_7c15,
            };
            let attention = AttentionKernels::load(&device)?;
            println!("  attention over lists made on the device, per attention layer");
            for formats in [(CacheFormat::Q8, CacheFormat::Q8), (CacheFormat::Q5, CacheFormat::Q4)]
            {
                for cells in [32_768u32, 262_144] {
                    for (rows, pieces) in [(1u32, 1u32), (1, 4), (1, 16), (8, 1), (8, 4), (1024, 1)]
                    {
                        attention_path(&mut rig, &attention, formats, cells, rows, pieces)?;
                    }
                }
            }
            continue;
        }
        let mut rig = Rig {
            stream: Stream::new(&device, 0)?,
            q4: Q4Kernels::load(&device)?,
            start: Event::timed(&device)?,
            stop: Event::timed(&device)?,
            device: device.clone(),
            seed: 0x9e37_79b9_7f4a_7c15,
        };
        let layers = LayerKernels::load(&device)?;
        // The floor: a launch of one thread block that computes one value.
        let one = rig.floats(16);
        let floor = {
            let (q4, s) = (rig.q4.clone(), &rig.stream);
            rig.time(400, || unsafe { q4.squash(s, one.ptr(), one.ptr(), 1, HC).unwrap() })
        };
        println!("  floor: a launch of one thread block, {floor:.1} us");
        for rows in [1u32, 8, 1024] {
            println!("  {rows} row(s)");
            let (mixer, ngram) = mixer_and_ngram(&mut rig, rows, floor)?;
            println!(
                "    => the mixers of a layer {mixer:.1} us, of {LAYERS} layers {:.2} ms; the n-gram block {ngram:.1} us",
                mixer * LAYERS / 1000.0
            );
            for (cells, format) in [
                (4096u32, KeyFormat::F32),
                (32_768, KeyFormat::F32),
                (32_768, KeyFormat::F16),
                (262_144, KeyFormat::F32),
                (262_144, KeyFormat::F16),
            ] {
                println!("   indexer at {cells} cells, {format:?} keys");
                let per_layer = indexer(&mut rig, &layers, rows, cells, format, floor)?;
                println!(
                    "    => per attention layer {per_layer:.1} us, {ATTENTION_LAYERS} layers {:.2} ms",
                    per_layer * ATTENTION_LAYERS / 1000.0
                );
            }
        }
    }
    Ok(())
}
