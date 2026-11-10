//! Times the kernels of the mixture of experts on the real expert tensors of
//! Qwen3.8-Flash-Next, with every picked expert in a slot.
//!
//! For one layer of each type gate and up have in the file, and 1 to 8 rows:
//! what the router costs, what the experts cost, and how that compares with
//! the bytes the layer must read (the router matrix once, each picked expert
//! once) at the device's memory bandwidth. The kernels run the way a step
//! runs them, 48 layers in a row between two events, over several copies of
//! the slots so that every layer reads its weights from memory and not from
//! the L2 cache.
//!
//! Then the price of telling the host what the router picked, which is what
//! the step protocol for missing experts is built on: the 48 layers enqueued
//! whole; the same with each layer's route record copied to pinned memory and
//! the host following the copies while the device runs on; and the host
//! reading each record before it enqueues the layer's experts (a round trip
//! per layer).
//!
//! ```sh
//! cargo run --release -p ares-cuda --example expert_layer_bench -- <model-00001-of-00002.gguf> [device]
//! ```

use std::time::Instant;

use ares_cuda::{
    Device, DeviceBuffer, Event, ExpertKernels, ExpertShape, Kernels, PinnedBuffer, Result,
    RouteRecord, Stream, Wait, devices,
};
use ares_gguf::Gguf;

/// Layers of a pass, and copies of the slots it cycles through.
const LAYERS: u32 = 48;
const COPIES: u32 = 4;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn normal(&mut self) -> f32 {
        let unit = |r: &mut Self| (r.next() >> 40) as f32 / (1u64 << 24) as f32;
        let (u1, u2) = (unit(self) + 1e-7, unit(self));
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One layer on a device with the experts a set of rows picks, in `COPIES`
/// copies of its slots.
struct Bench<'a> {
    kernels: &'a ExpertKernels,
    stream: &'a Stream,
    shape: ExpertShape,
    n_rows: u32,
    groups: u32,
    router: DeviceBuffer,
    act: DeviceBuffer,
    logits: DeviceBuffer,
    /// `COPIES` tables.
    tables: DeviceBuffer,
    /// One record per layer of the pass.
    records: DeviceBuffer,
    hidden: DeviceBuffer,
    out: DeviceBuffer,
    pool: DeviceBuffer,
}

impl Bench<'_> {
    fn table(&self, layer: u32) -> ares_cuda::DevicePtr {
        self.tables.ptr().offset(u64::from(layer % COPIES) * 4 * u64::from(self.shape.n_experts))
    }

    fn pool(&self, layer: u32) -> ares_cuda::DevicePtr {
        let copy = u64::from(self.groups) * self.shape.slot_bytes().unwrap_or(0);
        self.pool.ptr().offset(u64::from(layer % COPIES) * copy)
    }

    fn record(&self, layer: u32) -> ares_cuda::DevicePtr {
        self.records.ptr().offset(u64::from(layer) * self.record_stride())
    }

    fn record_stride(&self) -> u64 {
        self.shape.record_bytes(self.n_rows).next_multiple_of(16)
    }

    fn route(&self, layer: u32) -> Result<()> {
        // SAFETY: the buffers were sized for this shape and number of rows.
        unsafe {
            self.kernels.route(
                self.stream,
                &self.shape,
                self.router.ptr(),
                self.act.ptr(),
                self.logits.ptr(),
                self.table(layer),
                self.record(layer),
                self.n_rows,
                layer,
            )
        }
    }

    fn experts(&self, layer: u32, groups: Option<u32>) -> Result<()> {
        // SAFETY: as above; every picked expert is in a slot of every copy.
        unsafe {
            self.kernels.experts(
                self.stream,
                &self.shape,
                self.pool(layer),
                self.table(layer),
                self.record(layer),
                self.act.ptr(),
                self.hidden.ptr(),
                self.out.ptr(),
                self.n_rows,
                groups,
            )
        }
    }

    /// Milliseconds of a pass of `LAYERS` layers enqueued by `layer`, the
    /// best of five.
    fn time(&self, mut layer: impl FnMut(u32) -> Result<()>) -> Result<f64> {
        let device = self.stream.device();
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            start.record(self.stream)?;
            for l in 0..LAYERS {
                layer(l)?;
            }
            end.record(self.stream)?;
            end.synchronize()?;
            best = best.min(f64::from(end.elapsed_ms_since(&start)?));
        }
        Ok(best)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: expert_layer_bench <model.gguf> [device]");
        std::process::exit(2);
    };
    let only_device: Option<u32> = args.get(1).and_then(|a| a.parse().ok());
    let gguf = Gguf::open(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    // One layer of each type: IQ2_S, IQ2_XXS, IQ1_M in Qwen3.8-Flash-Next.
    let mut layers: Vec<(String, usize)> = Vec::new();
    for tensor in gguf.tensors() {
        let Some(rest) = tensor.name.strip_prefix("blk.") else { continue };
        let Some((index, "ffn_gate_exps.weight")) = rest.split_once('.') else { continue };
        if !layers.iter().any(|(ty, _)| ty == tensor.ty.name()) {
            layers.push((tensor.ty.name().to_owned(), index.parse().unwrap_or(0)));
        }
    }

    for info in devices()? {
        if only_device.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let dense = Kernels::load(&device)?;
        let kernels = ExpertKernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        println!("CUDA{} {}", info.ordinal, info.name);
        println!(
            "  {:<22} {:>4} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "layer", "rows", "groups", "MB read", "route us", "expert us", "layer us", "GB/s"
        );
        for (_, index) in &layers {
            let tensor = |suffix: &str| {
                gguf.tensor(&format!("blk.{index}.{suffix}.weight")).expect("an expert tensor")
            };
            let (gate, down) = (tensor("ffn_gate_exps"), tensor("ffn_down_exps"));
            let shape = ExpertShape {
                n_embd: gate.dims.get(0) as u32,
                n_ff: gate.dims.get(1) as u32,
                n_experts: gate.dims.get(2) as u32,
                top_k: 10,
                gate_ty: gate.ty,
                down_ty: down.ty,
            };
            shape.check()?;
            let name = format!("blk.{index} {}/{}", gate.ty, down.ty);
            for (n_rows, same_rows) in
                [(1u32, false), (2, false), (4, false), (8, false), (8, true)]
            {
                let mut rng = Rng(0x2545f4914f6cdd1d + u64::from(n_rows));
                let n = shape.n_embd as usize;
                let mut x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.normal()).collect();
                if same_rows {
                    let (first, rest) = x.split_at_mut(n);
                    rest.chunks_mut(n).for_each(|row| row.copy_from_slice(first));
                }
                let bench =
                    prepare(&device, &dense, &kernels, &stream, &gguf, shape, *index, &x, n_rows)?;
                let route = bench.time(|l| bench.route(l))?;
                let experts = bench.time(|l| bench.experts(l, Some(bench.groups)))?;
                let both = bench.time(|l| {
                    bench.route(l)?;
                    bench.experts(l, None)
                })?;
                let bytes = shape.router_bytes()
                    + u64::from(bench.groups) * shape.expert_bytes().unwrap_or(0);
                let per_layer = |ms: f64| ms * 1e3 / f64::from(LAYERS);
                println!(
                    "  {:<22} {:>4} {:>6} {:>9.2} {:>9.1} {:>9.1} {:>9.1} {:>8.0}{}",
                    name,
                    n_rows,
                    bench.groups,
                    bytes as f64 / 1e6,
                    per_layer(route),
                    per_layer(experts),
                    per_layer(both),
                    bytes as f64 / 1e9 / (both / 1e3 / f64::from(LAYERS)),
                    if same_rows { "  (rows pick the same experts)" } else { "" }
                );
                if *index == layers[0].1 && (n_rows == 1 || (n_rows == 8 && !same_rows)) {
                    protocols(&device, &bench, both)?;
                }
            }
        }
    }
    Ok(())
}

/// Loads the layer's router, routes `x` and puts the picked experts into
/// slots, `COPIES` times.
#[allow(clippy::too_many_arguments)]
fn prepare<'a>(
    device: &Device,
    dense: &Kernels,
    kernels: &'a ExpertKernels,
    stream: &'a Stream,
    gguf: &Gguf,
    shape: ExpertShape,
    index: usize,
    x: &[f32],
    n_rows: u32,
) -> Result<Bench<'a>> {
    let tensor =
        |suffix: &str| gguf.tensor(&format!("blk.{index}.{suffix}.weight")).expect("a tensor");
    let values = u64::from(n_rows) * u64::from(shape.n_embd);
    let table_bytes = 4 * u64::from(shape.n_experts);
    let stride = shape.slot_bytes().unwrap_or(0);
    let router = DeviceBuffer::new(device, shape.router_bytes())?;
    router.write(0, gguf.data(tensor("ffn_gate_inp")))?;
    let xs = DeviceBuffer::new(device, values * 4)?;
    xs.write(0, &f32_bytes(x))?;
    let tables = DeviceBuffer::new(device, u64::from(COPIES) * table_bytes)?;
    tables.write(0, &vec![0xFF; (u64::from(COPIES) * table_bytes) as usize])?;
    let mut bench = Bench {
        kernels,
        stream,
        shape,
        n_rows,
        groups: 0,
        router,
        act: DeviceBuffer::new(device, Kernels::quantized_activation_bytes(shape.n_embd, n_rows))?,
        logits: DeviceBuffer::new(device, shape.logits_bytes(n_rows))?,
        tables,
        records: DeviceBuffer::new(device, 16)?,
        hidden: DeviceBuffer::new(device, shape.hidden_bytes(n_rows))?,
        out: DeviceBuffer::new(device, values * 4)?,
        pool: DeviceBuffer::new(device, 16)?,
    };
    bench.records = DeviceBuffer::new(device, u64::from(LAYERS) * bench.record_stride())?;
    // SAFETY: the buffers were sized for this shape and number of rows.
    unsafe {
        dense.quantize_activations(stream, xs.ptr(), bench.act.ptr(), shape.n_embd, n_rows)?;
        stream.fill(bench.out.ptr(), 0, values as usize * 4)?;
    }
    bench.route(0)?;
    stream.synchronize()?;
    let mut record = vec![0u8; shape.record_bytes(n_rows) as usize];
    bench.records.read(0, &mut record)?;
    let record = RouteRecord::parse(&record, shape.top_k, n_rows).expect("a route record");
    let picked: Vec<u32> = record.groups().map(|g| g.expert).collect();
    bench.groups = picked.len() as u32;

    bench.pool = DeviceBuffer::new(device, u64::from(COPIES) * u64::from(bench.groups) * stride)?;
    let (gate_bytes, down_bytes) =
        (shape.gate_bytes().unwrap_or(0) as usize, shape.down_bytes().unwrap_or(0) as usize);
    let parts = [
        (tensor("ffn_gate_exps"), gate_bytes, 0),
        (tensor("ffn_up_exps"), gate_bytes, shape.up_offset().unwrap_or(0)),
        (tensor("ffn_down_exps"), down_bytes, shape.down_offset().unwrap_or(0)),
    ];
    for copy in 0..COPIES {
        let mut table = vec![0xFFu8; table_bytes as usize];
        for (i, expert) in picked.iter().enumerate() {
            // A different order in every copy.
            let slot = (i as u32 + 3 * copy) % bench.groups;
            let at = u64::from(copy * bench.groups + slot) * stride;
            for (tensor, bytes, offset) in parts {
                let e = *expert as usize;
                bench.pool.write(at + offset, &gguf.data(tensor)[e * bytes..(e + 1) * bytes])?;
            }
            table[4 * *expert as usize..][..4].copy_from_slice(&(slot as i32).to_le_bytes());
        }
        bench.tables.write(u64::from(copy) * table_bytes, &table)?;
    }
    Ok(bench)
}

/// What it costs to tell the host what the routers picked.
fn protocols(device: &Device, bench: &Bench<'_>, whole_ms: f64) -> Result<()> {
    let record_bytes = bench.shape.record_bytes(bench.n_rows) as usize;
    let stride = bench.record_stride() as usize;
    let mut host = PinnedBuffer::new(device, LAYERS as usize * stride)?;
    let host_ptr = host.as_mut_ptr();
    let events: Vec<Event> = (0..LAYERS).map(|_| Event::new(device)).collect::<Result<_>>()?;
    let copy_record = |l: u32| -> Result<()> {
        // SAFETY: the pinned buffer has a record's bytes per layer and lives
        // until the stream is idle; nobody reads a record before its event.
        unsafe {
            bench.stream.copy_to_host(
                host_ptr.add(l as usize * stride),
                bench.record(l),
                record_bytes,
            )?;
        }
        events[l as usize].record(bench.stream)
    };
    let parse = |l: u32| {
        // SAFETY: the copy of layer `l` has finished (its event is done) and
        // the range is inside the pinned buffer.
        let bytes =
            unsafe { std::slice::from_raw_parts(host_ptr.add(l as usize * stride), record_bytes) };
        RouteRecord::parse(bytes, bench.shape.top_k, bench.n_rows).map(|r| r.n_groups() as u32)
    };

    // The step enqueued whole; the host follows the copies while it runs.
    let mut followed = f64::INFINITY;
    let mut lag_us = 0.0f64;
    for _ in 0..5 {
        let started = Instant::now();
        for l in 0..LAYERS {
            bench.route(l)?;
            copy_record(l)?;
            bench.experts(l, None)?;
        }
        let enqueued = started.elapsed();
        let mut seen = 0;
        while seen < LAYERS {
            if events[seen as usize].is_done()? {
                assert_eq!(parse(seen), Some(bench.groups));
                seen += 1;
            }
        }
        bench.stream.synchronize()?;
        let total = started.elapsed().as_secs_f64() * 1e3;
        if total < followed {
            followed = total;
            lag_us = enqueued.as_secs_f64() * 1e6;
        }
    }

    // The same, with the copies on a second stream that waits for each
    // router: the step's own stream never stops for a copy.
    let side = Stream::new(device, 0)?;
    let routed: Vec<Event> = (0..LAYERS).map(|_| Event::new(device)).collect::<Result<_>>()?;
    let mut side_ms = f64::INFINITY;
    for _ in 0..5 {
        let started = Instant::now();
        for l in 0..LAYERS {
            bench.route(l)?;
            routed[l as usize].record(bench.stream)?;
            side.wait_event(&routed[l as usize])?;
            // SAFETY: as in `copy_record`.
            unsafe {
                side.copy_to_host(
                    host_ptr.add(l as usize * stride),
                    bench.record(l),
                    record_bytes,
                )?;
            }
            events[l as usize].record(&side)?;
            bench.experts(l, None)?;
        }
        let mut seen = 0;
        while seen < LAYERS {
            if events[seen as usize].is_done()? {
                assert_eq!(parse(seen), Some(bench.groups));
                seen += 1;
            }
        }
        bench.stream.synchronize()?;
        side_ms = side_ms.min(started.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "    {LAYERS} layers, {} rows: record copies on a second stream {side_ms:.2} ms",
        bench.n_rows
    );

    // A round trip per layer: the host reads the record, then enqueues the experts.
    let mut round_trip = f64::INFINITY;
    for _ in 0..5 {
        let started = Instant::now();
        for l in 0..LAYERS {
            bench.route(l)?;
            copy_record(l)?;
            while !events[l as usize].is_done()? {
                std::hint::spin_loop();
            }
            bench.experts(l, parse(l))?;
        }
        bench.stream.synchronize()?;
        round_trip = round_trip.min(started.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "    {LAYERS} layers, {} rows: enqueued whole {:.2} ms; with a record copy per layer and the \
         host following {:.2} ms (enqueue took {:.0} us); host round trip per layer {:.2} ms \
         (+{:.1} us per layer)",
        bench.n_rows,
        whole_ms,
        followed,
        lag_us,
        round_trip,
        (round_trip - followed) * 1e3 / f64::from(LAYERS)
    );
    Ok(())
}
