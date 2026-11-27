//! Times the attention kernels (`AttentionKernels::attend`) on each device, with
//! the shape of Qwen3.8-27B: head size 256, 4 KV heads, 6 query heads per KV head.
//!
//! Decode steps: 1, 4 and 8 query rows of 1 and 4 sequences against histories of
//! 4k, 32k and 262k cells, per pair of K and V formats. Prompt steps: 1024 and
//! 2048 rows of one sequence. The time is that of one attention layer (a step of
//! the model has 16), K and V cycle over several layers' pools so that the
//! reads come from memory and not from the L2 cache (at least 8 times the L2 size
//! when the pools allow it). The bytes of K and V read per second are the bytes
//! the blocks load, per item, against the memory bandwidth of the card.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example attention_bench -- \
//!     [--pairs q8/q8,f16/f16] [--hist 4096,32768,262144] [--rows 1,4,8] [--seqs 1,4] \
//!     [--prompt 1024,2048] [--device N] [--iters N]
//! ```
//!
//! Run it under `flock` when other work shares the GPUs.

use std::time::Instant;

use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
    Event, PlanConfig, Planner, Result, Stream, Wait, devices,
};

#[path = "../tests/common/mod.rs"]
mod common;
use common::{Rng, f32_bytes};

const TILE: usize = 128;
const CONVERTED: u32 = 0x8000_0000;
const D: usize = 256;
const KV: usize = 4;
const G: usize = 6;

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(device: &Device, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64)?;
    buffer.write(0, bytes)?;
    Ok(buffer)
}

fn parse_format(s: &str) -> CacheFormat {
    match s {
        "f16" => CacheFormat::F16,
        "q8" => CacheFormat::Q8,
        "q7" => CacheFormat::Q7,
        "q6" => CacheFormat::Q6,
        "q5" => CacheFormat::Q5,
        "q4" => CacheFormat::Q4,
        other => panic!("unknown format {other}"),
    }
}

fn name(f: CacheFormat) -> &'static str {
    ["f16", "q8", "q7", "q6", "q5", "q4"][f.id() as usize]
}

/// The default exact windows of `KvPolicy::new`: (first cells, newest cells) of the
/// narrower format, and whether the pair has a mean.
fn policy(k: CacheFormat, v: CacheFormat) -> (usize, usize, bool) {
    let newest = |f: CacheFormat| [0, 0, 32, 64, 128, 256][f.id() as usize];
    let has_mean = k.has_mean() || v.has_mean();
    (if has_mean { TILE } else { 0 }, newest(k).max(newest(v)), has_mean)
}

/// The cells `quantized_cells(n)` of the reference policy, as whole tiles.
fn converted_tiles(n: usize, first: usize, newest: usize) -> std::ops::Range<usize> {
    let start = first.min(n).next_multiple_of(TILE);
    let end = n.saturating_sub(newest) / TILE * TILE;
    if start < end { start / TILE..end / TILE } else { 0..0 }
}

struct Layer {
    exact: DeviceBuffer,
    converted: DeviceBuffer,
}

struct Setup {
    geo: CacheGeometry,
    layers: Vec<Layer>,
    table: Vec<u32>,
    table_dev: DeviceBuffer,
    /// (table offset, length in cells) of each sequence.
    seqs: Vec<(u32, usize)>,
}

struct Step {
    rows: DeviceBuffer,
    n: u32,
    table: DeviceBuffer,
    jobs: Option<(DeviceBuffer, u32)>,
}

/// Builds `n_seqs` sequences of `len` cells in every layer through the writing
/// kernels, in steps of 4096 rows as a prompt would arrive. With `convert` the
/// tiles the policy converts are converted after each step (the state of a
/// sequence that has been running); without, everything is still exact (the
/// state a prompt step reads). The slots are handed out as the state manager
/// will: an exact slot is given back when its tile is converted.
#[allow(clippy::too_many_arguments)]
fn build(
    device: &Device,
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: CacheGeometry,
    n_seqs: usize,
    len: usize,
    n_layers: usize,
    convert: bool,
) -> Result<Setup> {
    let (first, newest, has_mean) = policy(geo.k, geo.v);
    let per_seq = len.div_ceil(TILE);
    let mut table = vec![u32::MAX; n_seqs * per_seq];
    let (mut free_exact, mut next_exact, mut next_converted) = (Vec::<u32>::new(), 0u32, 0u32);
    let mut steps = Vec::new();
    let mut seqs = Vec::new();
    for s in 0..n_seqs {
        let off = (s * per_seq) as u32;
        seqs.push((off, len));
        let mut at = 0;
        while at < len {
            let n = 4096usize.min(len - at);
            for e in at / TILE..(at + n).div_ceil(TILE) {
                let slot = &mut table[s * per_seq + e];
                if *slot == u32::MAX {
                    *slot = if has_mean {
                        free_exact.pop().unwrap_or_else(|| {
                            next_exact += 1;
                            next_exact - 1
                        })
                    } else {
                        next_converted += 1;
                        (next_converted - 1) | CONVERTED
                    };
                }
            }
            let rows: Vec<AttnRow> = (0..n)
                .map(|i| AttnRow { table: off, pos: (at + i) as u32, slot_base: 0, n_slots: 0 })
                .collect();
            let rows_dev = upload(device, &AttnRow::to_bytes(&rows))?;
            let table_dev = upload(device, &u32_bytes(&table))?;
            at += n;
            let mut jobs = Vec::new();
            if has_mean && convert {
                for e in converted_tiles(at, first, newest) {
                    let slot = &mut table[s * per_seq + e];
                    if *slot & CONVERTED == 0 {
                        jobs.extend([*slot, next_converted]);
                        free_exact.push(*slot);
                        *slot = next_converted | CONVERTED;
                        next_converted += 1;
                    }
                }
            }
            let jobs = if jobs.is_empty() {
                None
            } else {
                Some((upload(device, &u32_bytes(&jobs))?, (jobs.len() / 2) as u32))
            };
            steps.push(Step { rows: rows_dev, n: n as u32, table: table_dev, jobs });
        }
    }
    let table_dev = upload(device, &u32_bytes(&table))?;

    let mut rng = Rng(99);
    let step = 4096usize.min(len);
    let k_host: Vec<f32> = (0..step * KV * D).map(|_| rng.normal()).collect();
    let v_host: Vec<f32> = (0..step * KV * D).map(|_| rng.normal()).collect();
    let k_dev = upload(device, &f32_bytes(&k_host))?;
    let v_dev = upload(device, &f32_bytes(&v_host))?;
    let status = upload(device, &[0u8; 4])?;
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        let exact =
            DeviceBuffer::new(device, u64::from(next_exact.max(1)) * geo.exact_entry_bytes())?;
        let converted = DeviceBuffer::new(
            device,
            u64::from(next_converted.max(1)) * geo.converted_entry_bytes(),
        )?;
        for st in &steps {
            // SAFETY: the pools hold every slot the tables name.
            unsafe {
                kernels.append(
                    stream,
                    &geo,
                    st.rows.ptr(),
                    st.n,
                    k_dev.ptr(),
                    v_dev.ptr(),
                    st.table.ptr(),
                    exact.ptr(),
                    converted.ptr(),
                    ares_cuda::DevicePtr::NULL,
                    status.ptr(),
                )?;
                if let Some((jobs, n)) = &st.jobs {
                    kernels.convert(
                        stream,
                        &geo,
                        jobs.ptr(),
                        *n,
                        exact.ptr(),
                        converted.ptr(),
                        ares_cuda::DevicePtr::NULL,
                        status.ptr(),
                    )?;
                }
            }
        }
        stream.synchronize()?;
        layers.push(Layer { exact, converted });
    }
    Ok(Setup { geo, layers, table, table_dev, seqs })
}

/// Bytes of K and V a block reads for `cells` cells of an entry of one kind.
fn item_bytes(geo: &CacheGeometry, converted: bool, cells: u32) -> f64 {
    let d = geo.head_dim;
    if !converted {
        return f64::from(cells) * f64::from(4 * d);
    }
    let per_cell = f64::from(geo.k.row_bytes(d) + geo.v.row_bytes(d));
    let means = f64::from(geo.k.mean_bytes(d) + geo.v.mean_bytes(d));
    f64::from(cells) * per_cell + f64::from(cells.div_ceil(TILE as u32)) * means
}

struct Timing {
    us: f64,
    gbps: f64,
    pieces: usize,
}

#[allow(clippy::too_many_arguments)]
fn time_step(
    device: &Device,
    kernels: &AttentionKernels,
    stream: &Stream,
    setup: &Setup,
    rows: usize,
    views: usize,
    iters: usize,
) -> Result<Timing> {
    let geo = &setup.geo;
    // `views` > 1: the first sequence seen at that many lengths, 200 cells apart, as so many
    // sequences with a common prefix would be.
    let n_rows = rows * setup.seqs.len().max(views);
    let q_total = KV * G;
    let mut rng = Rng(7);
    let q: Vec<f32> = (0..n_rows * q_total * D).map(|_| rng.normal()).collect();
    let q_dev = upload(device, &f32_bytes(&q))?;
    let out = DeviceBuffer::new(device, (n_rows * q_total * D * 4) as u64)?;
    let mut planner = Planner::new(
        *geo,
        PlanConfig::for_multiprocessors(device.info().multiprocessors),
        n_rows as u32,
    );
    if views > 1 {
        let (off, len) = setup.seqs[0];
        for s in 0..views {
            let len = len - 200 * s;
            let entries = &setup.table[off as usize..off as usize + len.div_ceil(TILE)];
            planner.dense(off, entries, (s * rows) as u32, rows as u32, (len - rows) as u32);
        }
    } else {
        for (s, &(off, len)) in setup.seqs.iter().enumerate() {
            let entries = &setup.table[off as usize..off as usize + len.div_ceil(TILE)];
            planner.dense(off, entries, (s * rows) as u32, rows as u32, (len - rows) as u32);
        }
    }
    let plan = planner.finish();
    let bytes: f64 = plan.items.iter().map(|i| item_bytes(geo, i.flags & 1 != 0, i.total)).sum();
    let scratch = DeviceBuffer::new(device, plan.state_bytes(geo).max(16))?;
    let buffers = plan.upload(device)?;
    let launch = |layer: &Layer| -> Result<()> {
        let args = AttendBuffers {
            tables: setup.table_dev.ptr(),
            exact_pool: layer.exact.ptr(),
            converted_pool: layer.converted.ptr(),
            far_pool: ares_cuda::DevicePtr::NULL,
            q: q_dev.ptr(),
            gate: None,
            out: out.ptr(),
            scratch: scratch.ptr(),
        };
        // SAFETY: the buffers were sized from the plan and the geometry.
        unsafe { kernels.attend(stream, geo, &buffers, &args, 1.0 / (D as f32).sqrt()) }
    };
    // Warm up for a third of a second: the cards clock down when idle and take a few
    // hundred milliseconds to come back, which would be measured as slow kernels.
    let warm = Instant::now();
    let mut i = 0;
    while warm.elapsed().as_millis() < 350 {
        for _ in 0..8 {
            launch(&setup.layers[i % setup.layers.len()])?;
            i += 1;
        }
        stream.synchronize()?;
    }
    let (start, stop) = (Event::timed(device)?, Event::timed(device)?);
    start.record(stream)?;
    for i in 0..iters {
        launch(&setup.layers[i % setup.layers.len()])?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    let us = f64::from(stop.elapsed_ms_since(&start)?) * 1000.0 / iters as f64;
    Ok(Timing { us, gbps: bytes / us / 1e3, pieces: plan.items.len() })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let list = |flag: &str, default: &str| -> Vec<String> {
        arg(flag).unwrap_or_else(|| default.to_owned()).split(',').map(str::to_owned).collect()
    };
    let pairs: Vec<(CacheFormat, CacheFormat)> = list("--pairs", "q8/q8,f16/f16,f16/q8,q4/q4")
        .iter()
        .map(|p| {
            let (k, v) = p.split_once('/').expect("pairs are k/v");
            (parse_format(k), parse_format(v))
        })
        .collect();
    let hist: Vec<usize> =
        list("--hist", "4096,32768,262144").iter().map(|x| x.parse().unwrap()).collect();
    let rows: Vec<usize> = list("--rows", "1,4,8").iter().map(|x| x.parse().unwrap()).collect();
    let seqs: Vec<usize> = list("--seqs", "1,4").iter().map(|x| x.parse().unwrap()).collect();
    let prompts: Vec<usize> =
        list("--prompt", "1024,2048").iter().map(|x| x.parse().unwrap()).collect();
    let only = arg("--device").map(|d| d.parse::<u32>().unwrap());
    let iters: usize = arg("--iters").map_or(30, |x| x.parse().unwrap());
    // `--views N`: every decode case also with N sequences that share their prefix.
    let views: usize = arg("--views").map_or(1, |x| x.parse().unwrap());
    // `--long H,..`: a prompt step of `--step` rows (256) at the end of a history of H cells,
    // converted as the policy says, as the step of a long prompt reads it.
    let long: Vec<usize> = arg("--long")
        .map(|l| l.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_default();
    let step: usize = arg("--step").map_or(256, |x| x.parse().unwrap());
    // `--heads N`: the KV heads this device holds (of 4), as a placement gives them.
    let heads: usize = arg("--heads").map_or(KV, |x| x.parse().unwrap());
    // With `--long` only those steps run.
    let (hist, prompts) = if long.is_empty() { (hist, prompts) } else { (Vec::new(), Vec::new()) };

    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal, Wait::Block)?;
        let kernels = AttentionKernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        let bandwidth =
            f64::from(info.memory_bus_width_bits) / 8.0 * f64::from(info.memory_clock_khz) * 2.0
                / 1e6;
        println!(
            "\n## {} (memory limit about {bandwidth:.0} GB/s, L2 {} MiB)\n",
            info.name,
            info.l2_cache_bytes >> 20
        );
        println!(
            "| step | K/V | rows | seqs | hist | pieces | us/layer | ms/16 layers | GB/s | % of limit |"
        );
        println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for &(k, v) in &pairs {
            let geo = CacheGeometry {
                k,
                v,
                head_dim: D as u32,
                kv_heads: heads as u32,
                kv_total: KV as u32,
                first_kv_head: 0,
                q_per_kv: G as u32,
            };
            for &n_seqs in &seqs {
                for &h in &hist {
                    let len = h + 8;
                    // Layers: enough footprint that the L2 does not hold it, if memory allows.
                    let per_layer = n_seqs as f64
                        * len as f64
                        * 4.0
                        * f64::from(k.row_bytes(D as u32) + v.row_bytes(D as u32));
                    let free = device.mem_info()?.free as f64 * 0.6;
                    let want = (8.0 * f64::from(info.l2_cache_bytes) / per_layer).ceil() as usize;
                    let n_layers =
                        want.clamp(1, 16).min(((free / per_layer).floor() as usize).max(1));
                    if per_layer > free {
                        println!(
                            "| decode | {}/{} | - | {n_seqs} | {h} | skipped: needs {:.1} GB | | | | |",
                            name(k),
                            name(v),
                            per_layer / 1e9
                        );
                        continue;
                    }
                    let t0 = Instant::now();
                    let setup =
                        build(&device, &kernels, &stream, geo, n_seqs, len, n_layers, true)?;
                    eprintln!(
                        "built {}/{} x{n_seqs} {h} in {:.1}s ({n_layers} layers)",
                        name(k),
                        name(v),
                        t0.elapsed().as_secs_f64()
                    );
                    for &r in &rows {
                        if views > 1 && n_seqs == 1 {
                            let t = time_step(&device, &kernels, &stream, &setup, r, views, iters)?;
                            println!(
                                "| shared x{views} | {}/{} | {r} | {n_seqs} | {h} | {} | {:.1} | {:.2} | {:.0} | {:.0} |",
                                name(k),
                                name(v),
                                t.pieces,
                                t.us,
                                t.us * 16.0 / 1e3,
                                t.gbps,
                                100.0 * t.gbps / bandwidth
                            );
                        }
                        let t = time_step(&device, &kernels, &stream, &setup, r, 1, iters)?;
                        println!(
                            "| decode | {}/{} | {r} | {n_seqs} | {h} | {} | {:.1} | {:.2} | {:.0} | {:.0} |",
                            name(k),
                            name(v),
                            t.pieces,
                            t.us,
                            t.us * 16.0 / 1e3,
                            t.gbps,
                            100.0 * t.gbps / bandwidth
                        );
                    }
                }
            }
            for &h in &long {
                let len = h + step;
                let per_layer = len as f64
                    * heads as f64
                    * f64::from(k.row_bytes(D as u32) + v.row_bytes(D as u32));
                let free = device.mem_info()?.free as f64 * 0.6;
                let n_layers = ((free / per_layer).floor() as usize).clamp(1, 2);
                let setup = build(&device, &kernels, &stream, geo, 1, len, n_layers, true)?;
                let t = time_step(&device, &kernels, &stream, &setup, step, 1, iters.min(8))?;
                // Causal: row i sees h + i + 1 cells; 4 * D flops per cell and query head.
                let flops = (0..step).map(|i| (h + i + 1) as f64).sum::<f64>()
                    * 4.0
                    * D as f64
                    * (G * heads) as f64;
                println!(
                    "| long | {}/{} | {step} | {heads} heads | {h} | {} | {:.0} | {:.2} | {:.0} | {:.1} TFLOPS |",
                    name(k),
                    name(v),
                    t.pieces,
                    t.us,
                    t.us * 16.0 / 1e3,
                    t.gbps,
                    flops / t.us / 1e6
                );
            }
            for &p in &prompts {
                let setup = build(&device, &kernels, &stream, geo, 1, p, 2, false)?;
                let t = time_step(&device, &kernels, &stream, &setup, p, 1, iters.min(8))?;
                println!(
                    "| prompt | {}/{} | {p} | 1 | 0 | {} | {:.0} | {:.2} | {:.0} | {:.0} |",
                    name(k),
                    name(v),
                    t.pieces,
                    t.us,
                    t.us * 16.0 / 1e3,
                    t.gbps,
                    100.0 * t.gbps / bandwidth
                );
            }
        }
    }
    Ok(())
}
