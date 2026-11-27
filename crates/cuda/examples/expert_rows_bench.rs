//! Times the many-row expert kernels (`ExpertRowKernels`) on the real expert tensors of
//! Qwen3.8-Flash-Next, with the routes the CPU reference recorded, against the decode
//! kernels (`ExpertKernels`) in launches of up to eight rows.
//!
//! For one layer of each type gate and up have in the file, and for 1 to 4096 rows: the
//! time of one layer's MoE block (plan, gate and up, down), the share of each
//! stage, rows per second, and how close that is to the memory and the tensor-core
//! limits. Every expert the rows pick is in device memory (in several copies when few
//! are picked, taken in turn, so that the weights come from memory and not from the L2
//! cache). The decode kernels get the same experts, in batches of eight rows above eight.
//!
//! Routes: `long`, `mixed`, `german` and `english` of the trace directory, token after
//! token, taken again from the start when more rows are asked for than there are
//! tokens (3570). The activations are random: the kernels' time does not depend on them.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example expert_rows_bench -- <model-00001-of-00002.gguf> <trace dir> [device] [rows,...]
//! ```

use ares_cuda::{
    Device, DeviceBuffer, DevicePtr, Event, ExpertKernels, ExpertRowKernels, ExpertShape, Kernels,
    PrefillPath, ROWS_TILES, Result, Stream, Wait, devices,
};
use ares_gguf::Gguf;

#[path = "../tests/common/mod.rs"]
mod common;
use common::{Rng, f32_bytes};

/// The picks of one layer of a trace for every token of the trace files.
fn read_routes(dir: &str, layer: usize, top_k: usize) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut routes = Vec::new();
    for name in ["long", "mixed", "german", "english"] {
        let bytes = std::fs::read(format!("{dir}/{name}.q4route")).unwrap_or_else(|e| {
            eprintln!("{dir}/{name}.q4route: {e}");
            std::process::exit(1);
        });
        assert_eq!(&bytes[..8], b"Q4ROUTE1");
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let (layers, k, tokens, size) = (word(12), word(20), word(24), word(32));
        assert_eq!(k, top_k);
        for t in 0..tokens {
            let at = 64 + 4 * tokens + (t * layers + layer) * size;
            let ids = (0..k)
                .map(|j| u32::from(u16::from_le_bytes([bytes[at + 2 * j], bytes[at + 2 * j + 1]])))
                .collect();
            let weights = (0..k)
                .map(|j| f32::from_le_bytes(bytes[at + 2 * k + 4 * j..][..4].try_into().unwrap()))
                .collect();
            routes.push((ids, weights));
        }
    }
    routes
}

/// The record of the decode kernels for up to eight rows, built on the host: the
/// header, ids, weights and the groups (each picked expert once, ascending, with the
/// rows that picked it) with the slot of the expert in the pool.
fn old_record(shape: &ExpertShape, rows: &[(Vec<u32>, Vec<f32>)], slot_of: &[i32]) -> Vec<u8> {
    let k = shape.top_k as usize;
    let cap = rows.len() * k;
    let mut bytes = vec![0u8; shape.record_bytes(rows.len() as u32) as usize];
    let mut put = |at: usize, v: u32| bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let mut groups: std::collections::BTreeMap<u32, (u32, [f32; 8])> = Default::default();
    for (b, (ids, weights)) in rows.iter().enumerate() {
        for j in 0..k {
            put(16 + 4 * (b * k + j), ids[j]);
            put(16 + 4 * cap + 4 * (b * k + j), weights[j].to_bits());
            let entry = groups.entry(ids[j]).or_insert((0, [0.0; 8]));
            entry.0 |= 1 << b;
            entry.1[b] = weights[j];
        }
    }
    put(4, groups.len() as u32);
    put(12, cap as u32);
    let mut first_pair = 0;
    for (g, (expert, (mask, weight))) in groups.iter().enumerate() {
        let at = 16 + 8 * cap + 48 * g;
        put(at, *expert);
        put(at + 4, slot_of[*expert as usize] as u32);
        put(at + 8, *mask);
        put(at + 12, first_pair);
        for (b, w) in weight.iter().enumerate() {
            put(at + 16 + 4 * b, w.to_bits());
        }
        first_pair += mask.count_ones();
    }
    bytes
}

/// Milliseconds per call of `f` run `reps` times between two events, the best of
/// `rounds`.
fn time(
    stream: &Stream,
    reps: u32,
    rounds: u32,
    mut f: impl FnMut(u32) -> Result<()>,
) -> Result<f64> {
    let device = stream.device();
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        start.record(stream)?;
        for i in 0..reps {
            f(i)?;
        }
        end.record(stream)?;
        end.synchronize()?;
        best = best.min(f64::from(end.elapsed_ms_since(&start)?) / f64::from(reps));
    }
    Ok(best)
}

/// The experts a set of rows picks, ascending.
fn picked(routes: &[(Vec<u32>, Vec<f32>)]) -> Vec<u32> {
    let mut set: Vec<u32> = routes.iter().flat_map(|(ids, _)| ids.iter().copied()).collect();
    set.sort_unstable();
    set.dedup();
    set
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(path), Some(trace)) = (args.first(), args.get(1)) else {
        eprintln!("usage: expert_rows_bench <model.gguf> <trace dir> [device] [rows,...]");
        std::process::exit(2);
    };
    let only_device: Option<u32> = args.get(2).and_then(|a| a.parse().ok());
    let row_counts: Vec<u32> = args
        .get(3)
        .map(|a| a.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 64, 256, 1024, 4096]);
    let only_type = std::env::var("ARES_BENCH_TYPE").ok();
    let gguf = Gguf::open(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
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
        let old = ExpertKernels::load(&device)?;
        let rows = ExpertRowKernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        let bandwidth =
            f64::from(info.memory_bus_width_bits) / 8.0 * f64::from(info.memory_clock_khz) * 2.0
                / 1e6;
        println!(
            "CUDA{} {} ({} SMs, memory {:.0} GB/s peak)",
            info.ordinal, info.name, info.multiprocessors, bandwidth
        );
        for (ty_name, index) in &layers {
            if only_type.as_ref().is_some_and(|t| t != ty_name) {
                continue;
            }
            let tensor = |suffix: &str| {
                gguf.tensor(&format!("blk.{index}.{suffix}.weight")).expect("an expert tensor")
            };
            let (gate, up, down) =
                (tensor("ffn_gate_exps"), tensor("ffn_up_exps"), tensor("ffn_down_exps"));
            let shape = ExpertShape {
                n_embd: gate.dims.get(0) as u32,
                n_ff: gate.dims.get(1) as u32,
                n_experts: gate.dims.get(2) as u32,
                top_k: 10,
                gate_ty: gate.ty,
                down_ty: down.ty,
            };
            shape.check()?;
            let routes = read_routes(trace, *index, shape.top_k as usize);
            println!("  blk.{index} {}/{} (gate and up {ty_name})", gate.ty, down.ty);
            println!(
                "  {:>5} {:>4} {:>4} {:>7} | {:>8} {:>8} {:>8} | {:>9} {:>7} {:>7} {:>7} | {:>9} {:>6}",
                "rows",
                "exp",
                "tile",
                "items",
                "plan us",
                "g+u us",
                "down us",
                "block us",
                "rows/s",
                "GB/s",
                "TFLOPS",
                "old us",
                "x old"
            );
            for &n_rows in &row_counts {
                let batch: Vec<(Vec<u32>, Vec<f32>)> =
                    (0..n_rows as usize).map(|t| routes[t % routes.len()].clone()).collect();
                let set = picked(&batch);
                let stride = shape.slot_bytes().expect("a slot size");
                // Copies of the picked experts, so that a repeated call reads memory.
                let copies = (160_000_000 / (set.len() as u64 * stride)).clamp(1, 8) as u32;
                let pool =
                    DeviceBuffer::new(&device, u64::from(copies) * set.len() as u64 * stride)?;
                let tables =
                    DeviceBuffer::new(&device, u64::from(copies) * 8 * u64::from(shape.n_experts))?;
                let slot_tables =
                    DeviceBuffer::new(&device, u64::from(copies) * 4 * u64::from(shape.n_experts))?;
                let (gate_bytes, down_bytes) =
                    (shape.gate_bytes().unwrap() as usize, shape.down_bytes().unwrap() as usize);
                let mut slot_of: Vec<Vec<i32>> = Vec::new();
                for copy in 0..copies {
                    let mut table = vec![0u64; shape.n_experts as usize];
                    let mut slots = vec![-1i32; shape.n_experts as usize];
                    for (i, e) in set.iter().enumerate() {
                        let slot = (i as u32 + 5 * copy) % set.len() as u32;
                        let at = (u64::from(copy) * set.len() as u64 + u64::from(slot)) * stride;
                        let e = *e as usize;
                        pool.write(at, &gguf.data(gate)[e * gate_bytes..(e + 1) * gate_bytes])?;
                        pool.write(
                            at + shape.up_offset().unwrap(),
                            &gguf.data(up)[e * gate_bytes..(e + 1) * gate_bytes],
                        )?;
                        pool.write(
                            at + shape.down_offset().unwrap(),
                            &gguf.data(down)[e * down_bytes..(e + 1) * down_bytes],
                        )?;
                        table[e] = pool.ptr().as_raw() + at;
                        slots[e] = slot as i32;
                    }
                    tables.write(
                        u64::from(copy) * 8 * u64::from(shape.n_experts),
                        &table.iter().flat_map(|a| a.to_le_bytes()).collect::<Vec<u8>>(),
                    )?;
                    slot_tables.write(
                        u64::from(copy) * 4 * u64::from(shape.n_experts),
                        &slots.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>(),
                    )?;
                    slot_of.push(slots);
                }
                let table_at = |copy: u32| -> DevicePtr {
                    tables.ptr().offset(u64::from(copy % copies) * 8 * u64::from(shape.n_experts))
                };

                // Activations: random, in the f16 form of the many-row kernels.
                let mut rng = Rng(0x2545f4914f6cdd1d ^ u64::from(n_rows));
                let n = shape.n_embd as usize;
                let x: Vec<f32> = (0..n_rows as usize * n).map(|_| rng.normal()).collect();
                let x_buf = DeviceBuffer::new(&device, 4 * x.len() as u64)?;
                x_buf.write(0, &f32_bytes(&x))?;
                let act = DeviceBuffer::new(&device, shape.rows_activation_bytes(n_rows))?;
                // SAFETY: the buffers were sized for this shape and number of rows.
                unsafe {
                    dense.prepare_prefill_activations(
                        &stream,
                        PrefillPath::F16,
                        x_buf.ptr(),
                        act.ptr(),
                        shape.n_embd,
                        n_rows,
                    )?;
                }
                let mut record = vec![0u8; 16];
                record.extend(
                    batch.iter().flat_map(|(ids, _)| ids.iter().flat_map(|v| v.to_le_bytes())),
                );
                record
                    .extend(batch.iter().flat_map(|(_, w)| w.iter().flat_map(|v| v.to_le_bytes())));
                let record_buf = DeviceBuffer::new(&device, record.len() as u64)?;
                record_buf.write(0, &record)?;
                let hidden = DeviceBuffer::new(&device, shape.rows_hidden_bytes(n_rows))?;
                let out = DeviceBuffer::new(&device, 4 * u64::from(n_rows) * n as u64)?;
                // SAFETY: as above.
                unsafe { stream.fill(out.ptr(), 0, 4 * n_rows as usize * n)? };

                // The decode kernels: the same rows in batches of eight.
                let old_us = {
                    let n_batches = n_rows.div_ceil(8);
                    let old_act = DeviceBuffer::new(
                        &device,
                        Kernels::quantized_activation_bytes(shape.n_embd, 8),
                    )?;
                    let old_x = DeviceBuffer::new(&device, 4 * 8 * n as u64)?;
                    old_x.write(0, &f32_bytes(&x[..8.min(n_rows as usize) * n]))?;
                    let old_rows = 8.min(n_rows);
                    // SAFETY: as above.
                    unsafe {
                        dense.quantize_activations(
                            &stream,
                            old_x.ptr(),
                            old_act.ptr(),
                            shape.n_embd,
                            old_rows,
                        )?;
                    }
                    let old_hidden = DeviceBuffer::new(&device, shape.hidden_bytes(8))?;
                    let old_out = DeviceBuffer::new(&device, 4 * 8 * n as u64)?;
                    let rec_stride = shape.record_bytes(8).next_multiple_of(16);
                    let records = DeviceBuffer::new(&device, u64::from(n_batches) * rec_stride)?;
                    for b in 0..n_batches {
                        let lo = (b * 8) as usize;
                        let hi = (lo + 8).min(batch.len());
                        let rec =
                            old_record(&shape, &batch[lo..hi], &slot_of[(b % copies) as usize]);
                        records.write(u64::from(b) * rec_stride, &rec)?;
                    }
                    stream.synchronize()?;
                    let ms = time(&stream, n_batches, 3, |b| {
                        let lo = (b * 8) as usize;
                        let rows_here = (batch.len() - lo).min(8) as u32;
                        let copy = b % copies;
                        // SAFETY: the pool and tables hold every picked expert; the
                        // records were written for these batches.
                        unsafe {
                            old.experts(
                                &stream,
                                &shape,
                                pool.ptr().offset(u64::from(copy) * set.len() as u64 * stride),
                                slot_tables
                                    .ptr()
                                    .offset(u64::from(copy) * 4 * u64::from(shape.n_experts)),
                                records.ptr().offset(u64::from(b) * rec_stride),
                                old_act.ptr(),
                                old_hidden.ptr(),
                                old_out.ptr(),
                                None,
                                rows_here,
                                None,
                            )
                        }
                    })?;
                    ms * 1e3 * f64::from(n_batches)
                };

                let mut best: Option<(f64, u32)> = None;
                let mut lines = Vec::new();
                for tile in ROWS_TILES {
                    let plan = DeviceBuffer::new(&device, shape.rows_plan_bytes(n_rows, tile))?;
                    // SAFETY: as above.
                    unsafe {
                        rows.plan_rows(
                            &stream,
                            &shape,
                            n_rows,
                            tile,
                            record_buf.ptr(),
                            plan.ptr(),
                            0,
                        )?;
                    }
                    stream.synchronize()?;
                    let reps = (2000 / n_rows.max(1)).clamp(3, 200);
                    let ms_plan = time(&stream, reps, 3, |_| {
                        // SAFETY: as above.
                        unsafe {
                            rows.plan_rows(
                                &stream,
                                &shape,
                                n_rows,
                                tile,
                                record_buf.ptr(),
                                plan.ptr(),
                                0,
                            )
                        }
                    })?;
                    let ms_gate_up = time(&stream, reps, 3, |i| {
                        // SAFETY: as above; the table has every picked expert.
                        unsafe {
                            rows.rows_gate_up(
                                &stream,
                                &shape,
                                n_rows,
                                tile,
                                table_at(i),
                                plan.ptr(),
                                act.ptr(),
                                hidden.ptr(),
                            )
                        }
                    })?;
                    let ms_down = time(&stream, reps, 3, |i| {
                        // SAFETY: as above.
                        unsafe {
                            rows.rows_down(
                                &stream,
                                &shape,
                                n_rows,
                                tile,
                                table_at(i),
                                plan.ptr(),
                                hidden.ptr(),
                                out.ptr(),
                            )
                        }
                    })?;
                    let ms_block = time(&stream, reps, 3, |i| {
                        // SAFETY: as above.
                        unsafe {
                            rows.plan_rows(
                                &stream,
                                &shape,
                                n_rows,
                                tile,
                                record_buf.ptr(),
                                plan.ptr(),
                                0,
                            )?;
                            rows.rows_experts(
                                &stream,
                                &shape,
                                n_rows,
                                tile,
                                table_at(i),
                                plan.ptr(),
                                act.ptr(),
                                hidden.ptr(),
                                out.ptr(),
                            )
                        }
                    })?;
                    let mut head = vec![0u8; 16];
                    plan.read(0, &mut head)?;
                    let items = u32::from_le_bytes(head[..4].try_into().unwrap());
                    let flops = 2.0
                        * f64::from(n_rows)
                        * f64::from(shape.top_k)
                        * 3.0
                        * f64::from(shape.n_ff)
                        * f64::from(shape.n_embd);
                    let bytes = set.len() as f64 * stride as f64;
                    lines.push((tile, items, ms_plan, ms_gate_up, ms_down, ms_block, flops, bytes));
                    if best.is_none_or(|(b, _)| ms_block < b) {
                        best = Some((ms_block, tile));
                    }
                }
                for (tile, items, plan_ms, gu_ms, down_ms, block_ms, flops, bytes) in lines {
                    let mark = if best.is_some_and(|(_, t)| t == tile) { "*" } else { " " };
                    println!(
                        "  {:>5} {:>4} {:>3}{} {:>7} | {:>8.1} {:>8.1} {:>8.1} | {:>9.1} {:>7.0} {:>7.0} {:>7.1} | {:>9.1} {:>6.2}",
                        n_rows,
                        set.len(),
                        tile,
                        mark,
                        items,
                        plan_ms * 1e3,
                        gu_ms * 1e3,
                        down_ms * 1e3,
                        block_ms * 1e3,
                        f64::from(n_rows) / (block_ms / 1e3),
                        bytes / 1e9 / (block_ms / 1e3),
                        flops / 1e12 / (block_ms / 1e3),
                        old_us,
                        old_us / (block_ms * 1e3),
                    );
                }
            }
        }
    }
    Ok(())
}
