// The attention benchmark for one backend. Included twice by `main.rs`, once inside a module
// that imports `ares_vulkan` and once `ares_cuda`; the three functions the modules define
// (`open`, `new_stream`, `plan_config`) are the only places where the two crates differ.

use std::time::Instant;

use super::{BenchCase, Case, Opts};

const TILE: usize = 128;
const CONVERTED: u32 = 0x8000_0000;
pub const D: usize = 256;
pub const KV: usize = 4;
pub const G: usize = 6;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn gaussian(&mut self) -> f32 {
        let u1 = (self.next() >> 40) as f32 / (1u64 << 24) as f32 + 1e-7;
        let u2 = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(device: &Device, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64)?;
    buffer.write(0, bytes)?;
    Ok(buffer)
}

fn format_of(name: &str) -> CacheFormat {
    match name {
        "f16" => CacheFormat::F16,
        "q8" => CacheFormat::Q8,
        "q7" => CacheFormat::Q7,
        "q6" => CacheFormat::Q6,
        "q5" => CacheFormat::Q5,
        "q4" => CacheFormat::Q4,
        other => panic!("unknown format {other}"),
    }
}

/// The default exact windows of `KvPolicy::new`: (first cells, newest cells) of the narrower
/// format, and whether the pair has a mean.
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
    len: usize,
}

struct Step {
    rows: DeviceBuffer,
    n: u32,
    table: DeviceBuffer,
    jobs: Option<(DeviceBuffer, u32)>,
}

/// Builds a sequence of `len` cells in `n_layers` layers through the writing kernels, in steps
/// of 4096 rows as a prompt would arrive, and converts the tiles the policy converts, as a
/// sequence that has been running has them.
fn build(
    device: &Device,
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: CacheGeometry,
    len: usize,
    n_layers: usize,
) -> Result<Setup> {
    let (first, newest, has_mean) = policy(geo.k, geo.v);
    let per_seq = len.div_ceil(TILE);
    let mut table = vec![u32::MAX; per_seq];
    let (mut free_exact, mut next_exact, mut next_converted) = (Vec::<u32>::new(), 0u32, 0u32);
    let mut steps = Vec::new();
    let mut at = 0;
    while at < len {
        let n = 4096usize.min(len - at);
        for entry in table.iter_mut().take((at + n).div_ceil(TILE)).skip(at / TILE) {
            if *entry == u32::MAX {
                *entry = if has_mean {
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
            .map(|i| AttnRow { table: 0, pos: (at + i) as u32, slot_base: 0, n_slots: 0 })
            .collect();
        let rows_dev = upload(device, &AttnRow::to_bytes(&rows))?;
        let table_dev = upload(device, &u32_bytes(&table))?;
        at += n;
        let mut jobs = Vec::new();
        if has_mean {
            for e in converted_tiles(at, first, newest) {
                if table[e] & CONVERTED == 0 {
                    jobs.extend([table[e], next_converted]);
                    free_exact.push(table[e]);
                    table[e] = next_converted | CONVERTED;
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
    let table_dev = upload(device, &u32_bytes(&table))?;

    let mut rng = Rng(99);
    let step = 4096usize.min(len);
    let k_host: Vec<f32> = (0..step * KV * D).map(|_| rng.gaussian()).collect();
    let v_host: Vec<f32> = (0..step * KV * D).map(|_| rng.gaussian()).collect();
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
                        status.ptr(),
                    )?;
                }
            }
        }
        stream.synchronize()?;
        layers.push(Layer { exact, converted });
    }
    Ok(Setup { geo, layers, table, table_dev, len })
}

/// The ranges of the list case: blocks of 4 cells at a regular distance, and the newest 3.
fn list_ranges(len: usize, cells: usize) -> Vec<std::ops::Range<u32>> {
    let blocks = (cells - 3) / 4;
    let stride = (len - 8) / blocks;
    let mut ranges: Vec<std::ops::Range<u32>> =
        (0..blocks).map(|b| (b * stride) as u32..(b * stride + 4) as u32).collect();
    ranges.push((len - 3) as u32..len as u32);
    ranges
}

/// The microseconds of one layer's attention for a case, and the pieces the plan has.
fn time_step(
    device: &Device,
    kernels: &AttentionKernels,
    stream: &Stream,
    setup: &Setup,
    case: &Case,
    opts: &Opts,
) -> Result<(f64, usize)> {
    let geo = &setup.geo;
    let rows = match case {
        Case::Window(rows) => *rows,
        Case::List(_) => 1,
    };
    let q_total = KV * G;
    let mut rng = Rng(7);
    let q: Vec<f32> = (0..rows * q_total * D).map(|_| rng.gaussian()).collect();
    let q_dev = upload(device, &f32_bytes(&q))?;
    let out = DeviceBuffer::new(device, (rows * q_total * D * 4) as u64)?;
    let mut cfg = plan_config(device, kernels);
    if let Some(blocks) = opts.blocks {
        cfg.target_blocks = blocks;
    }
    let mut planner = Planner::new(*geo, cfg, rows as u32);
    let entries = &setup.table[..setup.len.div_ceil(TILE)];
    match case {
        Case::Window(rows) => {
            planner.dense(0, entries, 0, *rows as u32, (setup.len - rows) as u32);
        }
        Case::List(cells) => {
            planner.sparse(0, entries, 0, (setup.len - 1) as u32, &list_ranges(setup.len, *cells));
        }
    }
    let plan = planner.finish();
    let scratch = DeviceBuffer::new(device, plan.state_bytes(geo).max(16))?;
    let buffers = plan.upload(device)?;
    let launch = |layer: &Layer| -> Result<()> {
        let args = AttendBuffers {
            tables: setup.table_dev.ptr(),
            exact_pool: layer.exact.ptr(),
            converted_pool: layer.converted.ptr(),
            q: q_dev.ptr(),
            gate: None,
            out: out.ptr(),
            scratch: scratch.ptr(),
        };
        // SAFETY: the buffers were sized from the plan and the geometry.
        unsafe { kernels.attend(stream, geo, &buffers, &args, 1.0 / (D as f32).sqrt()) }
    };
    // Warm up for a third of a second: the cards clock down when idle and take a few hundred
    // milliseconds to come back, which would be measured as slow kernels.
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
    for i in 0..opts.iters {
        launch(&setup.layers[i % setup.layers.len()])?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    let us = f64::from(stop.elapsed_ms_since(&start)?) * 1000.0 / opts.iters as f64;
    Ok((us, plan.items.len()))
}

/// Times the cases on the device with this ordinal: the microseconds of one layer's attention
/// and the items of the plan, or `None` where the sequence does not fit the device's memory.
pub fn run(ordinal: u32, opts: &Opts, cases: &[BenchCase]) -> Result<Vec<Option<(f64, usize)>>> {
    let device = open(ordinal)?;
    let kernels = AttentionKernels::load(&device)?;
    let stream = new_stream(&device)?;
    let mut results = Vec::new();
    // Consecutive cases of one pair and history share their sequence.
    let mut built: Option<((CacheFormat, CacheFormat, usize), Setup)> = None;
    for c in cases {
        let (k, v) = (format_of(c.k_name), format_of(c.v_name));
        let key = (k, v, c.hist);
        if built.as_ref().map(|(b, _)| *b) != Some(key) {
            built = None;
            let geo = CacheGeometry {
                k,
                v,
                head_dim: D as u32,
                kv_heads: KV as u32,
                kv_total: KV as u32,
                first_kv_head: 0,
                q_per_kv: G as u32,
            };
            let len = c.hist + 8;
            // Layers: enough footprint that the L2 does not hold it, if memory allows.
            let per_layer = len as f64
                * KV as f64
                * f64::from(k.row_bytes(D as u32) + v.row_bytes(D as u32));
            let free = device.mem_info()?.free as f64 * 0.6;
            if per_layer > free {
                eprintln!("[{BACKEND}] {} needs {:.1} GB: skipped", c.label(), per_layer / 1e9);
                results.push(None);
                continue;
            }
            let want = (opts.l2_bytes * 8.0 / per_layer).ceil() as usize;
            let n_layers = want.clamp(1, 16).min(((free / per_layer).floor() as usize).max(1));
            let t0 = Instant::now();
            built = Some((key, build(&device, &kernels, &stream, geo, len, n_layers)?));
            eprintln!(
                "[{BACKEND}] built {} {}/{} in {:.1}s ({n_layers} layers)",
                c.hist,
                c.k_name,
                c.v_name,
                t0.elapsed().as_secs_f64()
            );
        }
        let Some((_, setup)) = built.as_ref() else {
            results.push(None);
            continue;
        };
        results.push(Some(time_step(&device, &kernels, &stream, setup, &c.case, opts)?));
    }
    Ok(results)
}
