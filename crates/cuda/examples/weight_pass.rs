//! Times one pass of every weight matrix of a real model on the GPUs.
//!
//! A decode step reads each weight once, so the time of a step is bounded
//! below by the time to multiply every matrix of the model with the step's
//! activations; prompt processing is bound by the arithmetic of the same
//! products, with hundreds to thousands of rows. This example measures exactly
//! that: it opens a GGUF file, uploads every 2-D weight tensor whose type has a
//! fast kernel, and then times passes in which each matrix is multiplied with
//! `n` rows of random activations once.
//!
//! Every matrix is one buffer in one layout, and every product one call
//! (`Kernels::product`). A matrix whose type has a device layout is rearranged
//! into it at load and multiplied by the kernels over that layout whatever the
//! number of rows; the others stay in the file's layout, where 1 to 8 rows run
//! the decode kernels and more the tile kernels. `--layout file` keeps every
//! matrix in the file's layout, for comparison.
//!
//! - With 1 to 8 rows the result is a LOWER BOUND for a decode step, to set next
//!   to the step time of another engine.
//! - With more rows it is a LOWER BOUND for the time of a chunk of that many
//!   prompt tokens, and `rows / pass time` an UPPER BOUND for prompt processing
//!   in tokens per second, to set next to the figure of another engine's whole
//!   forward pass. The output head multiplies only the last row of a chunk
//!   (the logits of the next token), so it runs with one row.
//!
//! There is no attention, no recurrent layer, no normalisation, no sum across
//! devices and no sampling in it, and the activations are prepared once per
//! layer and input width, not after every norm.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example weight_pass -- <model.gguf> \
//!     [--split 0.625,0.375] [--rows 1,3,8] [--passes N] [--device 0] \
//!     [--precision fast|exact] [--layout device|file]
//! ```
//!
//! - `--split a,b,...`: the shares of every matrix's rows that devices 0, 1,
//!   ... get (they are normalised). Each device is driven by its own host
//!   thread and stream, all started together. Without it, one device (`--device`,
//!   default 0) gets everything; a model that does not fit there is refused.
//! - `--rows`: the numbers of activation rows to measure. A number the scratch
//!   memory of a device does not allow (the weights take most of it) is skipped
//!   and says so.
//! - `--passes`: passes per measurement (default 20 for up to 8 rows, 5 for more).
//! - `--precision`: which kernels run (`Precision`): `fast` (the default) takes
//!   the fastest of the device, `exact` always the f16 kernels with f32
//!   accumulation.
//! - `--layout`: `device` (the default) rearranges the matrices of the types that
//!   have a device layout; the table of which kernel serves which number of rows
//!   is measured per device before the weights are placed and printed.
//! - Peer access is not needed: the devices do not exchange data here.
//!
//! Loading is timed too, because it is part of what a user waits for: the
//! weights are uploaded twice, with the buffers freed in between, and for each
//! upload (the first from a file that may not be in the page cache, the second
//! from a file that surely is) the example prints per device the bytes, the
//! seconds and the GB/s, and how long freeing the buffers takes, also at the
//! end. The upload path is the plain one: the mapped file is ordinary
//! (pageable) memory, copied with `DeviceBuffer::write`, so the driver stages
//! every copy through its own pinned buffers; one host thread per device. It
//! is not tuned: page-locked staging on a stream, or locking the mapped pages in
//! place, would be the next steps. A matrix that gets a device layout is copied
//! to a scratch buffer as large as the largest of them and rearranged from there
//! into its place, so the device never holds a matrix twice.
//!
//! Not every tensor is a matrix product with a fast kernel. The report says
//! what was covered and what was skipped and why: the token embedding is a
//! row lookup, not a product; 1-D tensors are norms and biases; the other
//! types have no fast kernel (yet).

use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Barrier;
use std::time::{Duration, Instant};

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, DevicePtr, Event, Kernels, MAX_BATCH, Precision, Result,
    Stream, Wait, WeightLayout, device_count,
};
use ares_gguf::{Gguf, TensorInfo};

/// Rows of a split are multiples of this: a tile of a device layout, which is two
/// thread blocks of the decode kernels, is never split across devices.
const ROW_GRANULE: u64 = 16;
/// Matrices start at multiples of this in a device's weight buffer.
const MATRIX_ALIGN: u64 = 256;
/// The output head: in a prefill pass only the last row of a chunk goes through it.
const HEAD: &str = "output.weight";

fn main() {
    if let Err(message) = run() {
        eprintln!("weight_pass: {message}");
        std::process::exit(1);
    }
}

struct Options {
    path: String,
    shares: Vec<f64>,
    first_device: u32,
    rows: Vec<u32>,
    passes: Option<usize>,
    precision: Precision,
    /// Whether the types that have a device layout get it.
    device_layout: bool,
}

fn parse_args() -> std::result::Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut shares = Vec::new();
    let mut rows = vec![1, 3, 8];
    let mut passes = None;
    let mut precision = Precision::Fast;
    let mut device_layout = true;
    let mut device = 0;
    while let Some(arg) = args.next() {
        let mut value = |what: &str| args.next().ok_or(format!("{what} needs a value"));
        match arg.as_str() {
            "--split" => {
                shares = value("--split")?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>().map_err(|e| format!("--split '{s}': {e}")))
                    .collect::<std::result::Result<_, _>>()?;
            }
            "--rows" => {
                rows = value("--rows")?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map_err(|e| format!("--rows '{s}': {e}")))
                    .collect::<std::result::Result<_, _>>()?;
            }
            "--passes" => {
                passes = Some(value("--passes")?.parse().map_err(|e| format!("--passes: {e}"))?);
            }
            "--precision" => {
                precision = match value("--precision")?.as_str() {
                    "fast" => Precision::Fast,
                    "exact" => Precision::Exact,
                    other => return Err(format!("--precision '{other}': fast or exact")),
                };
            }
            "--layout" => {
                device_layout = match value("--layout")?.as_str() {
                    "device" => true,
                    "file" => false,
                    other => return Err(format!("--layout: device or file, not {other}")),
                };
            }
            "--device" => {
                device = value("--device")?.parse().map_err(|e| format!("--device: {e}"))?;
            }
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => path = Some(other.to_owned()),
        }
    }
    let path = path.ok_or(
        "usage: weight_pass <model.gguf> [--split a,b] [--rows 1,3,8] [--passes N] [--device N] \
         [--precision fast|exact] [--layout device|file]",
    )?;
    if rows.contains(&0) || rows.is_empty() {
        return Err("--rows takes values of 1 or more".into());
    }
    if shares.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return Err("--split shares must be positive".into());
    }
    if passes == Some(0) {
        return Err("--passes must be at least 1".into());
    }
    let first_device = if shares.is_empty() { device } else { 0 };
    if shares.is_empty() {
        shares.push(1.0);
    }
    Ok(Options { path, shares, first_device, rows, passes, precision, device_layout })
}

/// For each pass of one device: when it started and ended (since the start of
/// the run, on the host) and how long it took on the device (ms).
type PassTimes = Vec<(Duration, Duration, f32)>;

/// Why a tensor is not part of the pass.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Skip {
    NotAMatrix,
    EmbeddingLookup,
    /// The type, as its GGUF id (the enum itself has no order).
    NoFastKernel(u32),
}

impl Skip {
    fn describe(self) -> String {
        match self {
            Skip::NotAMatrix => "1-D tensors (norms, biases)".into(),
            Skip::EmbeddingLookup => "token embedding: a row lookup, not a product".into(),
            Skip::NoFastKernel(id) => {
                let name = GgmlType::from_id(id).map_or("unknown", GgmlType::name);
                format!("{name} matrices: no fast kernel")
            }
        }
    }
}

fn classify(t: &TensorInfo) -> std::result::Result<(), Skip> {
    if t.dims.as_slice().len() != 2 {
        return Err(Skip::NotAMatrix);
    }
    if t.name == "token_embd.weight" {
        return Err(Skip::EmbeddingLookup);
    }
    if !Kernels::has_fast_matvec(t.ty) {
        return Err(Skip::NoFastKernel(t.ty.id()));
    }
    Ok(())
}

/// One device's slice of one matrix.
struct Slice {
    tensor: usize,
    row0: u64,
    rows: u64,
    offset: u64,
    /// How the slice lies on the device.
    layout: WeightLayout,
}

/// The rows each device gets of a matrix of `n_out` rows.
fn split_rows(n_out: u64, shares: &[f64]) -> Vec<(u64, u64)> {
    let total: f64 = shares.iter().sum();
    let mut bounds = vec![0u64];
    let mut cumulative = 0.0;
    for (i, share) in shares.iter().enumerate() {
        cumulative += share / total;
        let end = if i + 1 == shares.len() {
            n_out
        } else {
            ((cumulative * n_out as f64 / ROW_GRANULE as f64).round() as u64 * ROW_GRANULE)
                .min(n_out)
        };
        bounds.push(end.max(*bounds.last().unwrap()));
    }
    bounds.windows(2).map(|w| (w[0], w[1] - w[0])).collect()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn run() -> std::result::Result<(), String> {
    let options = parse_args()?;
    let gguf = Gguf::open(&options.path).map_err(|e| format!("{}: {e}", options.path))?;
    let n_devices = device_count().map_err(|e| e.to_string())?;
    let last = options.first_device + options.shares.len() as u32;
    if last > n_devices {
        return Err(format!(
            "{} device(s) are needed from device {}, the driver has {n_devices}",
            options.shares.len(),
            options.first_device
        ));
    }

    // What is covered, what is not.
    let tensors = gguf.tensors();
    let mut covered_bytes = 0u64;
    let mut covered_by_type: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    let mut skipped: BTreeMap<Skip, (u64, u64)> = BTreeMap::new();
    let mut matrices = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        match classify(t) {
            Ok(()) => {
                matrices.push(i);
                covered_bytes += t.size;
                let entry = covered_by_type.entry(t.ty.name()).or_default();
                entry.0 += 1;
                entry.1 += t.size;
            }
            Err(reason) => {
                let entry = skipped.entry(reason).or_default();
                entry.0 += 1;
                entry.1 += t.size;
            }
        }
    }
    let skipped_bytes: u64 = skipped.values().map(|(_, b)| b).sum();

    println!(
        "model: {} ({:.2} GiB in {} tensors)",
        options.path,
        gib(gguf.file_size()),
        tensors.len()
    );
    println!(
        "covered: {} matrices, {:.3} GiB ({:.2}% of the tensor bytes)",
        matrices.len(),
        gib(covered_bytes),
        100.0 * covered_bytes as f64 / (covered_bytes + skipped_bytes) as f64
    );
    for (name, (count, bytes)) in &covered_by_type {
        println!("    {name:<8} {count:>4} matrices {:>8.3} GiB", gib(*bytes));
    }
    println!("skipped: {:.3} GiB", gib(skipped_bytes));
    for (reason, (count, bytes)) in &skipped {
        println!("    {:>4} tensors {:>8.3} GiB  {}", count, gib(*bytes), reason.describe());
    }

    // The plan: which rows of which matrix go to which device, and where.
    let n_dev = options.shares.len();
    let mut slices: Vec<Vec<Slice>> = (0..n_dev).map(|_| Vec::new()).collect();
    let mut device_bytes = vec![0u64; n_dev];
    for &i in &matrices {
        let t = &tensors[i];
        let n_in = t.dims.get(0);
        let n_out = t.dims.get(1);
        let row_bytes = t.ty.bytes_for(n_in).ok_or_else(|| format!("{}: ragged row", t.name))?;
        for (d, (row0, rows)) in split_rows(n_out, &options.shares).into_iter().enumerate() {
            if rows == 0 {
                continue;
            }
            let offset = device_bytes[d].next_multiple_of(MATRIX_ALIGN);
            let layout = if options.device_layout && Kernels::has_device_layout(t.ty, n_in as u32) {
                WeightLayout::Device
            } else {
                WeightLayout::File
            };
            // The device layout rounds the rows of the slice up to a tile of 16.
            let bytes = Kernels::layout_bytes(t.ty, layout, rows as u32, n_in as u32)
                .unwrap_or(rows * row_bytes);
            slices[d].push(Slice { tensor: i, row0, rows, offset, layout });
            device_bytes[d] = offset + bytes;
        }
    }
    for (d, bytes) in device_bytes.iter().enumerate() {
        println!("device {}: {:.3} GiB of weights", options.first_device + d as u32, gib(*bytes));
    }

    // Open the devices. Opening takes a few hundred MiB of each device's memory.
    let mut devices = Vec::new();
    for (d, bytes) in device_bytes.iter().enumerate() {
        let ordinal = options.first_device + d as u32;
        let device = Device::open(ordinal, Wait::Spin).map_err(|e| e.to_string())?;
        let info = device.mem_info().map_err(|e| e.to_string())?;
        let needed = bytes + (512 << 20);
        if needed > info.free {
            return Err(format!(
                "device {ordinal} ({}) has {:.2} GiB free, the weights need {:.2} GiB: spread the model with --split",
                device.info().name,
                gib(info.free),
                gib(device_bytes[d])
            ));
        }
        devices.push(device);
    }

    // The kernels of each device, and for the types that get a device layout the table
    // of which kernel serves which number of rows, measured while the memory is free.
    let kernels =
        devices.iter().map(Kernels::load).collect::<Result<Vec<_>>>().map_err(|e| e.to_string())?;
    for (d, kernels) in kernels.iter().enumerate() {
        let mut types: Vec<GgmlType> = slices[d]
            .iter()
            .filter(|s| s.layout == WeightLayout::Device)
            .map(|s| tensors[s.tensor].ty)
            .collect();
        types.sort_by_key(|t| t.id());
        types.dedup();
        for ty in types {
            let table = kernels.measure_layout(ty).map_err(|e| e.to_string())?;
            let row = |r: &ares_cuda::LayoutRows| {
                let upto = if r.up_to_rows == u32::MAX {
                    "more".to_string()
                } else {
                    format!("<={}", r.up_to_rows)
                };
                format!("{upto}: {} / {}", r.fast.name, r.exact.name)
            };
            println!(
                "device {} {:<7} kernels by rows (fast / exact): {}",
                options.first_device + d as u32,
                ty.name(),
                table.rows.iter().map(row).collect::<Vec<_>>().join(", ")
            );
        }
    }

    // Upload, free, upload again: the first upload reads the file from wherever it
    // is (the page cache, or the disk), the second surely from the page cache.
    println!();
    println!("loading (ordinary memory through DeviceBuffer::write, one host thread per device):");
    let first =
        upload(&devices, &kernels, &slices, &device_bytes, &gguf, &options, "first upload ")?;
    free("free after the first upload", first);
    let buffers =
        upload(&devices, &kernels, &slices, &device_bytes, &gguf, &options, "second upload")?;

    // Per device: the launch list. A launch is one matrix slice; before the
    // first matrix of each (layer, input width) the activations of that width
    // are quantized.
    let widths: Vec<u64> = {
        let mut w: Vec<u64> = matrices.iter().map(|&i| tensors[i].dims.get(0)).collect();
        w.sort_unstable();
        w.dedup();
        w
    };
    let layer_of =
        |name: &str| name.strip_prefix("blk.").and_then(|r| r.split('.').next()).map(str::to_owned);

    // What a device has left once the weights are on it, for activations and outputs.
    let free_after: Vec<u64> = devices
        .iter()
        .map(|d| d.mem_info().map(|i| i.free).map_err(|e| e.to_string()))
        .collect::<std::result::Result<_, _>>()?;
    let max_out_of = |d: usize| {
        slices[d]
            .iter()
            .map(|s| &tensors[s.tensor])
            .filter(|t| t.name != HEAD)
            .map(|t| t.dims.get(1))
            .max()
            .unwrap_or(1)
    };

    for &n_rows in &options.rows {
        // The scratch of the largest pass: per input width the f32 activations
        // and the prepared copy (f16 is the larger), the outputs, and a margin
        // for the context's own needs.
        let scratch = |d: usize| -> u64 {
            let rows = u64::from(n_rows);
            let inputs: u64 = widths.iter().map(|w| w * rows * (4 + 2)).sum();
            inputs + max_out_of(d) * rows * 4 + (512 << 20)
        };
        if let Some(d) = (0..n_dev).find(|&d| scratch(d) > free_after[d]) {
            println!();
            println!(
                "{n_rows} activation row(s): skipped, device {} ({}) has {:.2} GiB left after the weights and the pass needs {:.2} GiB",
                options.first_device + d as u32,
                devices[d].info().name,
                gib(free_after[d]),
                gib(scratch(d))
            );
            continue;
        }
        let passes = options.passes.unwrap_or(if n_rows > MAX_BATCH { 5 } else { 20 });
        let barrier = Barrier::new(n_dev);
        let t0 = Instant::now();
        let results: Vec<std::result::Result<PassTimes, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n_dev)
                .map(|d| {
                    let (device, buffer, slices, barrier) =
                        (&kernels[d], &buffers[d], &slices[d], &barrier);
                    let (widths, gguf, options) = (&widths, &gguf, &options);
                    scope.spawn(move || -> std::result::Result<PassTimes, String> {
                        run_device(
                            device,
                            buffer,
                            slices,
                            widths,
                            gguf,
                            n_rows,
                            passes,
                            options.precision,
                            barrier,
                            t0,
                            &layer_of,
                        )
                        .map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("a device thread panicked".into())))
                .collect()
        });
        let results: Vec<PassTimes> = results.into_iter().collect::<std::result::Result<_, _>>()?;

        // Per pass: the wall time from the first start to the last end, and each
        // device's own time from its events.
        let mut wall: Vec<f64> = (0..passes)
            .map(|p| {
                let start = results.iter().map(|r| r[p].0).min().unwrap();
                let end = results.iter().map(|r| r[p].1).max().unwrap();
                (end - start).as_secs_f64() * 1e3
            })
            .collect();
        wall.sort_by(f64::total_cmp);
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!();
        if n_rows > MAX_BATCH {
            println!(
                "{n_rows} activation rows, {passes} passes (medians; prefill kernels, precision {:?}; the output head takes the last row only)",
                options.precision
            );
            println!(
                "    rows per second is rows / pass time: an UPPER BOUND for prompt processing in tokens per second, since attention, recurrent layers, normalisation and sums across devices are not in the pass"
            );
        } else {
            println!(
                "{n_rows} activation row(s), {passes} passes (medians; lower bound of a decode step: weights only)"
            );
        }
        let mut total_bytes = 0u64;
        let mut total_ops = 0f64;
        for d in 0..n_dev {
            let mut gpu: Vec<f64> = results[d].iter().map(|r| f64::from(r.2)).collect();
            let gpu = median(&mut gpu);
            let mut span: Vec<f64> =
                results[d].iter().map(|r| (r.1 - r.0).as_secs_f64() * 1e3).collect();
            let span = median(&mut span);
            total_bytes += device_bytes[d];
            // 2 operations per weight and row; the head has one row in a prefill pass.
            let ops: f64 = slices[d]
                .iter()
                .map(|s| {
                    let t = &tensors[s.tensor];
                    let rows = if n_rows > MAX_BATCH && t.name == HEAD { 1 } else { n_rows };
                    2.0 * s.rows as f64 * t.dims.get(0) as f64 * f64::from(rows)
                })
                .sum();
            total_ops += ops;
            println!(
                "    device {} ({}): {:>7.3} ms on the device, {:>7.3} ms from first launch to completion, {:>6.1} GB/s of weights ({:.3} GiB), {:>5.1} TOPS",
                options.first_device + d as u32,
                devices[d].info().name,
                gpu,
                span,
                device_bytes[d] as f64 / (gpu * 1e-3) / 1e9,
                gib(device_bytes[d]),
                ops / (gpu * 1e-3) / 1e12
            );
        }
        let total = median(&mut wall);
        println!(
            "    total: {:>7.3} ms per pass, {:>6.1} GB/s of weights ({:.3} GiB); {:.2} passes/s, {:.0} rows/s, {:.1} TOPS",
            total,
            total_bytes as f64 / (total * 1e-3) / 1e9,
            gib(total_bytes),
            1e3 / total,
            f64::from(n_rows) / (total * 1e-3),
            total_ops / (total * 1e-3) / 1e12
        );
    }
    println!();
    free("free at the end      ", buffers);
    Ok(())
}

/// Allocates each device's weight buffer and copies its slices in, one thread
/// per device, and prints the bytes, seconds and GB/s per device and in total.
fn upload(
    devices: &[Device],
    kernels: &[Kernels],
    slices: &[Vec<Slice>],
    device_bytes: &[u64],
    gguf: &Gguf,
    options: &Options,
    label: &str,
) -> std::result::Result<Vec<DeviceBuffer>, String> {
    let started = Instant::now();
    let results: Vec<std::result::Result<(DeviceBuffer, f64, f64), String>> =
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..devices.len())
                .map(|d| {
                    let (device, slices, bytes) = (&devices[d], &slices[d], device_bytes[d]);
                    let kernels = &kernels[d];
                    scope.spawn(move || -> std::result::Result<(DeviceBuffer, f64, f64), String> {
                        let copy = || -> Result<(DeviceBuffer, f64, f64)> {
                            let file_bytes = |s: &Slice| {
                                let tensor = &gguf.tensors()[s.tensor];
                                s.rows * tensor.ty.bytes_for(tensor.dims.get(0)).unwrap()
                            };
                            let t = Instant::now();
                            let buffer = DeviceBuffer::new(device, bytes.max(1))?;
                            // The slices that get a device layout pass through here.
                            let largest = slices
                                .iter()
                                .filter(|s| s.layout == WeightLayout::Device)
                                .map(file_bytes)
                                .max();
                            let scratch = match largest {
                                Some(bytes) => Some((
                                    DeviceBuffer::new(device, bytes)?,
                                    Stream::new(device, 0)?,
                                )),
                                None => None,
                            };
                            let allocated = t.elapsed().as_secs_f64();
                            for s in slices {
                                let tensor = &gguf.tensors()[s.tensor];
                                let n_in = tensor.dims.get(0);
                                let row_bytes = tensor.ty.bytes_for(n_in).unwrap();
                                let data = gguf.data(tensor);
                                let range = (s.row0 * row_bytes) as usize
                                    ..((s.row0 + s.rows) * row_bytes) as usize;
                                match (s.layout, &scratch) {
                                    (WeightLayout::Device, Some((staged, stream))) => {
                                        staged.write(0, &data[range])?;
                                        // SAFETY: the scratch buffer holds the slice in the
                                        // file's layout and the plan gave its device layout
                                        // room at `offset`.
                                        unsafe {
                                            kernels.rearrange(
                                                stream,
                                                tensor.ty,
                                                staged.ptr(),
                                                buffer.ptr().offset(s.offset),
                                                s.rows as u32,
                                                n_in as u32,
                                            )?;
                                        }
                                        // The scratch buffer is written again right after.
                                        stream.synchronize()?;
                                    }
                                    _ => buffer.write(s.offset, &data[range])?,
                                }
                            }
                            Ok((buffer, allocated, t.elapsed().as_secs_f64()))
                        };
                        copy().map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err("an upload thread panicked".to_string())))
                .collect()
        });
    let wall = started.elapsed().as_secs_f64();
    let mut buffers = Vec::new();
    let mut total_bytes = 0u64;
    for (d, result) in results.into_iter().enumerate() {
        let (buffer, allocated, seconds) = result?;
        total_bytes += device_bytes[d];
        println!(
            "  {label} device {} ({}): {:.3} GiB in {:.3} s ({:.3} s of it allocating) = {:.2} GB/s",
            options.first_device + d as u32,
            devices[d].info().name,
            gib(device_bytes[d]),
            seconds,
            allocated,
            device_bytes[d] as f64 / seconds / 1e9
        );
        buffers.push(buffer);
    }
    println!(
        "  {label} total: {:.3} GiB in {:.3} s = {:.2} GB/s (devices in parallel)",
        gib(total_bytes),
        wall,
        total_bytes as f64 / wall / 1e9
    );
    Ok(buffers)
}

/// Frees the weight buffers, one thread per device, and prints how long it took.
fn free(label: &str, buffers: Vec<DeviceBuffer>) {
    let started = Instant::now();
    let times: Vec<(u64, f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = buffers
            .into_iter()
            .map(|buffer| {
                scope.spawn(move || {
                    let (bytes, t) = (buffer.len(), Instant::now());
                    drop(buffer);
                    (bytes, t.elapsed().as_secs_f64())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or((0, 0.0))).collect()
    });
    let per_device: Vec<String> = times
        .iter()
        .map(|(bytes, s)| format!("{:.3} GiB in {:.1} ms", gib(*bytes), s * 1e3))
        .collect();
    println!(
        "  {label}: {} (all: {:.1} ms)",
        per_device.join(", "),
        started.elapsed().as_secs_f64() * 1e3
    );
}

/// Runs `passes` passes on one device, in step with the other threads, and
/// returns for each pass the time it started and ended (since `t0`, on the
/// host) and its time on the device.
///
/// A device that fails (an allocation that does not fit, say) still takes its
/// turns at the barrier, so that the other devices' threads are not left
/// waiting for it and the error is reported.
#[allow(clippy::too_many_arguments)]
fn run_device(
    kernels: &Kernels,
    weights: &DeviceBuffer,
    slices: &[Slice],
    widths: &[u64],
    gguf: &Gguf,
    n_rows: u32,
    passes: usize,
    precision: Precision,
    barrier: &Barrier,
    t0: Instant,
    layer_of: &dyn Fn(&str) -> Option<String>,
) -> Result<PassTimes> {
    let waited = Cell::new(0usize);
    let result = run_device_passes(
        kernels, weights, slices, widths, gguf, n_rows, passes, precision, barrier, t0, layer_of,
        &waited,
    );
    if result.is_err() {
        for _ in waited.get()..passes {
            barrier.wait();
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run_device_passes(
    kernels: &Kernels,
    weights: &DeviceBuffer,
    slices: &[Slice],
    widths: &[u64],
    gguf: &Gguf,
    n_rows: u32,
    passes: usize,
    precision: Precision,
    barrier: &Barrier,
    t0: Instant,
    layer_of: &dyn Fn(&str) -> Option<String>,
    waited: &Cell<usize>,
) -> Result<PassTimes> {
    let device = kernels.device();
    let stream = Stream::new(device, 0)?;
    // In a pass of more rows than a decode step has, only the last row goes through the
    // output head.
    let prefill = n_rows > MAX_BATCH;
    let is_head = |name: &str| prefill && name == HEAD;
    let rows_of = |name: &str| if is_head(name) { 1 } else { n_rows };

    // Random activations per input width, and room for their two prepared forms: which
    // one a product reads depends on the type and the layout of its matrix.
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut inputs: BTreeMap<u64, (DeviceBuffer, [DeviceBuffer; 2])> = BTreeMap::new();
    for &n_in in widths {
        let x: Vec<u8> = (0..n_in * u64::from(n_rows)).flat_map(|_| next().to_le_bytes()).collect();
        let dx = DeviceBuffer::new(device, x.len() as u64)?;
        dx.write(0, &x)?;
        let prepared = [ActivationForm::Int8, ActivationForm::F16].map(|form| {
            DeviceBuffer::new(device, Kernels::activation_bytes(form, n_in as u32, n_rows))
        });
        let [int8, f16] = prepared;
        inputs.insert(n_in, (dx, [int8?, f16?]));
    }
    // The head of a prefill pass has one row, prepared apart.
    let head_width = slices
        .iter()
        .map(|s| &gguf.tensors()[s.tensor])
        .find(|t| t.name == HEAD)
        .map(|t| t.dims.get(0));
    let head_act = DeviceBuffer::new(
        device,
        Kernels::activation_bytes(ActivationForm::Int8, head_width.unwrap_or(32) as u32, 1).max(
            Kernels::activation_bytes(ActivationForm::F16, head_width.unwrap_or(32) as u32, 1),
        ),
    )?;
    let max_out = slices
        .iter()
        .map(|s| (s.rows, rows_of(&gguf.tensors()[s.tensor].name)))
        .map(|(rows, n)| rows * u64::from(n))
        .max()
        .unwrap_or(1);
    let y = DeviceBuffer::new(device, max_out * 4)?;

    let one_pass = |timed: Option<(&Event, &Event)>| -> Result<()> {
        if let Some((start, _)) = timed {
            start.record(&stream)?;
        }
        let mut prepared: Option<(Option<String>, u64, ActivationForm)> = None;
        for s in slices {
            let t = &gguf.tensors()[s.tensor];
            let n_in = t.dims.get(0);
            let rows = rows_of(&t.name);
            // A type with no kernel for this many rows in its layout (the float matrices of
            // the recurrent layers, 4 to 10240 values wide, in a prefill pass) is left out:
            // it is a few MB.
            let Ok(form) = kernels.activation_form(t.ty, s.layout, precision, rows) else {
                continue;
            };
            let (dx, both) = &inputs[&n_in];
            let w: DevicePtr = weights.ptr().offset(s.offset);
            let act = if is_head(&t.name) {
                // SAFETY: `head_act` holds one row of the head's width in either form, and
                // `dx` has at least that many values.
                unsafe {
                    kernels.prepare_activations(
                        &stream,
                        form,
                        dx.ptr(),
                        head_act.ptr(),
                        n_in as u32,
                        1,
                    )?
                };
                head_act.ptr()
            } else {
                let dq = &both[usize::from(form == ActivationForm::F16)];
                let key = (layer_of(&t.name), n_in, form);
                if prepared.as_ref() != Some(&key) {
                    // SAFETY: both buffers were sized for this width, batch and form above.
                    unsafe {
                        kernels.prepare_activations(
                            &stream,
                            form,
                            dx.ptr(),
                            dq.ptr(),
                            n_in as u32,
                            n_rows,
                        )?
                    };
                    prepared = Some(key);
                }
                dq.ptr()
            };
            // SAFETY: the slice lies inside the uploaded weights in the layout the plan
            // gave it, `act` holds the activations of this width in the form the product
            // reads, and `y` has room for the largest output.
            unsafe {
                kernels.product(
                    &stream,
                    t.ty,
                    s.layout,
                    precision,
                    w,
                    act,
                    y.ptr(),
                    n_in as u32,
                    s.rows as u32,
                    rows,
                )?
            };
        }
        if let Some((_, end)) = timed {
            end.record(&stream)?;
        }
        Ok(())
    };

    // Warm-up: the first launches load the module's code and the GPU clocks
    // ramp up.
    let done = Event::new(device)?;
    for _ in 0..3 {
        one_pass(None)?;
    }
    done.record(&stream)?;
    done.synchronize()?;

    let start = Event::timed(device)?;
    let end = Event::timed(device)?;
    let mut out = Vec::with_capacity(passes);
    for _ in 0..passes {
        barrier.wait();
        waited.set(waited.get() + 1);
        let began = t0.elapsed();
        one_pass(Some((&start, &end)))?;
        end.synchronize()?;
        let finished = t0.elapsed();
        out.push((began, finished, end.elapsed_ms_since(&start)?));
    }
    Ok(out)
}
