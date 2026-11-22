//! Times one decode row's attention over K and V in VRAM and over the same bytes in
//! page-locked host memory, which the kernels read over the card's link (entries with
//! `FAR_ENTRY`: the K and V of a context beyond VRAM), and what a copy of the layer's K and V
//! from host memory to VRAM takes instead.
//!
//! Two shapes: Qwen3.8-27B's attention layers (head size 256, 4 KV heads, 6 query heads per
//! KV head; dense: a row reads every cell) and Qwen3.8-Flash-Next's (2 KV heads, 12 query heads
//! per KV head; a row past 2051 cells reads 512 blocks of 4 cells the indexer picks and its
//! tail, here picked at random). K and V in `q8`, one sequence; the time is that of one layer
//! (the 27B has 16 such layers, Flash-Next 12).
//!
//! ```sh
//! cargo run --release -p ares-cuda --example far_attention_bench -- \
//!     [--shape 27b,flash] [--hist 32768,131072,262144] [--device N] [--iters N]
//! ```
//!
//! Run it under the GPU lock when other work shares the card.

use std::time::Instant;

use ares_cuda::{
    AttendBuffers, AttentionKernels, AttnRow, CacheFormat, CacheGeometry, Device, DeviceBuffer,
    DevicePtr, Event, FAR_ENTRY, PinnedBuffer, PlanConfig, Planner, Result, Stream, Wait, devices,
};

const TILE: usize = 128;
const CONVERTED: u32 = 0x8000_0000;
const D: usize = 256;

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

fn bytes_of<T: Copy>(values: &[T], f: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|v| f(*v)).collect()
}

fn upload(device: &Device, bytes: &[u8]) -> Result<DeviceBuffer> {
    let buffer = DeviceBuffer::new(device, bytes.len().max(16) as u64)?;
    buffer.write(0, bytes)?;
    Ok(buffer)
}

/// One layer of one sequence of `len` cells, written by the append kernel: its converted
/// pool in VRAM and the table.
fn build(
    device: &Device,
    kernels: &AttentionKernels,
    stream: &Stream,
    geo: &CacheGeometry,
    len: usize,
) -> Result<(DeviceBuffer, Vec<u32>)> {
    let tiles = len.div_ceil(TILE);
    let table: Vec<u32> = (0..tiles as u32).map(|t| t | CONVERTED).collect();
    let table_dev = upload(device, &bytes_of(&table, u32::to_le_bytes))?;
    let pool = DeviceBuffer::new(device, tiles as u64 * geo.converted_entry_bytes())?;
    let kv = geo.kv_heads as usize;
    let step = 4096usize.min(len);
    let mut rng = Rng(99);
    let k: Vec<f32> = (0..step * kv * D).map(|_| rng.gaussian()).collect();
    let v: Vec<f32> = (0..step * kv * D).map(|_| rng.gaussian()).collect();
    let (k, v) = (
        upload(device, &bytes_of(&k, f32::to_le_bytes))?,
        upload(device, &bytes_of(&v, f32::to_le_bytes))?,
    );
    let status = upload(device, &[0u8; 16])?;
    let mut at = 0;
    while at < len {
        let n = step.min(len - at);
        let rows: Vec<AttnRow> = (0..n)
            .map(|i| AttnRow { table: 0, pos: (at + i) as u32, slot_base: 0, n_slots: 0 })
            .collect();
        let rows = upload(device, &AttnRow::to_bytes(&rows))?;
        // SAFETY: the pool has an entry for every tile the rows name.
        unsafe {
            kernels.append(
                stream,
                geo,
                rows.ptr(),
                n as u32,
                k.ptr(),
                v.ptr(),
                table_dev.ptr(),
                DevicePtr::NULL,
                pool.ptr(),
                DevicePtr::NULL,
                status.ptr(),
            )?;
        }
        stream.synchronize()?;
        at += n;
    }
    Ok((pool, table))
}

/// The cells a Flash-Next row at cell `len - 1` reads: 512 blocks of 4 cells at random and
/// the tail, as ranges.
fn picked(len: usize, rng: &mut Rng) -> Vec<std::ops::Range<u32>> {
    let blocks = len / 4;
    let mut chosen = std::collections::BTreeSet::new();
    while chosen.len() < 512.min(blocks) {
        chosen.insert((rng.next() % blocks as u64) as u32);
    }
    let mut ranges: Vec<std::ops::Range<u32>> = Vec::new();
    for b in chosen {
        match ranges.last_mut() {
            Some(r) if r.end == 4 * b => r.end += 4,
            _ => ranges.push(4 * b..4 * b + 4),
        }
    }
    let tail = (blocks * 4) as u32..len as u32;
    if !tail.is_empty() {
        ranges.push(tail);
    }
    ranges
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg =
        |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let list = |flag: &str, default: &str| -> Vec<String> {
        arg(flag).unwrap_or_else(|| default.to_owned()).split(',').map(str::to_owned).collect()
    };
    let shapes = list("--shape", "27b,flash");
    let hist: Vec<usize> =
        list("--hist", "32768,131072,262144").iter().map(|x| x.parse().unwrap()).collect();
    let only = arg("--device").map(|d| d.parse::<u32>().unwrap());
    let iters: usize = arg("--iters").map_or(20, |x| x.parse().unwrap());

    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal, Wait::Block)?;
        let kernels = AttentionKernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        println!("\n## {}\n", info.name);
        println!(
            "| shape | cells | K/V of the layer | cells read | VRAM, us | RAM, us | RAM / VRAM | \
             copy to VRAM, ms | GB/s of the copy |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        for shape in &shapes {
            let (kv, g, layers) = match shape.as_str() {
                "27b" => (4u32, 6u32, 16),
                "flash" => (2, 12, 12),
                other => panic!("unknown shape {other}"),
            };
            let geo = CacheGeometry {
                k: CacheFormat::Q8,
                v: CacheFormat::Q8,
                head_dim: D as u32,
                kv_heads: kv,
                kv_total: kv,
                first_kv_head: 0,
                q_per_kv: g,
            };
            for &len in &hist {
                let (pool, table) = build(&device, &kernels, &stream, &geo, len)?;
                let bytes = pool.len() as usize;
                let mut host = PinnedBuffer::new(&device, bytes)?;
                // SAFETY: the buffer holds `bytes` and is waited for below.
                unsafe { stream.copy_to_host(host.as_mut_ptr(), pool.ptr(), bytes)? };
                stream.synchronize()?;
                let far_table: Vec<u32> =
                    table.iter().map(|e| (e & !CONVERTED) | CONVERTED | FAR_ENTRY).collect();
                let near_dev = upload(&device, &bytes_of(&table, u32::to_le_bytes))?;
                let far_dev = upload(&device, &bytes_of(&far_table, u32::to_le_bytes))?;
                let mut planner = Planner::new(
                    geo,
                    PlanConfig::for_multiprocessors(device.info().multiprocessors),
                    1,
                );
                let mut rng = Rng(7);
                let read = if shape == "flash" && len > 2051 {
                    let ranges = picked(len, &mut rng);
                    let cells = ranges.iter().map(|r| r.end - r.start).sum::<u32>();
                    planner.sparse(0, &table, 0, (len - 1) as u32, &ranges);
                    cells
                } else {
                    planner.dense(0, &table, 0, 1, (len - 1) as u32);
                    len as u32
                };
                let plan = planner.finish();
                let buffers = plan.upload(&device)?;
                let q_len = (kv * g) as usize * D;
                let q: Vec<f32> = (0..q_len).map(|_| rng.gaussian()).collect();
                let q = upload(&device, &bytes_of(&q, f32::to_le_bytes))?;
                let out = DeviceBuffer::new(&device, (q_len * 4) as u64)?;
                let scratch = DeviceBuffer::new(&device, plan.state_bytes(&geo).max(16))?;
                let time = |far: bool| -> Result<f64> {
                    let args = AttendBuffers {
                        tables: if far { far_dev.ptr() } else { near_dev.ptr() },
                        exact_pool: DevicePtr::NULL,
                        converted_pool: pool.ptr(),
                        far_pool: DevicePtr::NULL.offset(host.as_ptr() as u64),
                        q: q.ptr(),
                        gate: None,
                        out: out.ptr(),
                        scratch: scratch.ptr(),
                    };
                    let launch = || {
                        // SAFETY: the plan names the sequence's cells, which both pools hold.
                        unsafe {
                            kernels.attend(&stream, &geo, &buffers, &args, 1.0 / (D as f32).sqrt())
                        }
                    };
                    let warm = Instant::now();
                    while warm.elapsed().as_millis() < 300 {
                        launch()?;
                        stream.synchronize()?;
                    }
                    let (start, stop) = (Event::timed(&device)?, Event::timed(&device)?);
                    start.record(&stream)?;
                    for _ in 0..iters {
                        launch()?;
                    }
                    stop.record(&stream)?;
                    stop.synchronize()?;
                    Ok(f64::from(stop.elapsed_ms_since(&start)?) * 1000.0 / iters as f64)
                };
                let (near_us, far_us) = (time(false)?, time(true)?);
                let (start, stop) = (Event::timed(&device)?, Event::timed(&device)?);
                start.record(&stream)?;
                // SAFETY: both hold `bytes`.
                unsafe { stream.copy_from_host(pool.ptr(), host.as_ptr(), bytes)? };
                stop.record(&stream)?;
                stop.synchronize()?;
                let copy_ms = f64::from(stop.elapsed_ms_since(&start)?);
                println!(
                    "| {shape} | {len} | {:.0} MiB | {read} | {near_us:.1} | {far_us:.1} | {:.1} | \
                     {copy_ms:.1} | {:.1} |",
                    bytes as f64 / (1 << 20) as f64,
                    far_us / near_us,
                    bytes as f64 / copy_ms / 1e6,
                );
                eprintln!(
                    "{shape} {len}: a step of {layers} layers {:.2} ms in VRAM, {:.2} ms from RAM",
                    near_us * f64::from(layers) / 1e3,
                    far_us * f64::from(layers) / 1e3
                );
            }
        }
    }
    Ok(())
}
