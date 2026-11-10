//! Times one pass of every weight matrix of a real model on the GPUs: the Vulkan twin of
//! `crates/cuda/examples/weight_pass.rs`, cut down to the measurement.
//!
//! A decode step reads each weight once, so its time is bounded below by the time to
//! multiply every matrix of the model with the step's activations; a prompt chunk by the
//! same products with many rows. The example opens a GGUF file, gives each device its
//! share of the rows of every 2-D weight matrix, and times passes in which each slice is
//! multiplied once with `n` rows of activations through the one call a model makes
//! (`Kernels::product`). A matrix whose type has a device layout is rearranged into it at
//! load (`--layout device`, the default) or stays in the file's layout (`--layout file`),
//! where 1 to 8 rows run the decode kernels and more the tile kernels. The small matrices
//! of a model count here as they do in a step: a launch has a floor, and a model has
//! hundreds of them.
//!
//! There is no attention, no recurrent layer, no sum across devices and no sampling in it,
//! and the activations are prepared once per input width outside the timed pass.
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_weight_pass -- <model.gguf> \
//!     [--split 0.625,0.375] [--rows 1,8,64] [--passes N] [--device 0] \
//!     [--precision fast|exact] [--layout device|file] [--measure]
//! ```
//!
//! `--split a,b`: the shares of every matrix's rows that devices `--device`, the next, ...
//! get; each device is driven by its own host thread, all passes started together.
//! `--measure` fills the table of each type by `Kernels::measure_layout` before the weights
//! are placed (otherwise the table that follows from what the device reports is used).

use std::collections::BTreeMap;
use std::sync::Barrier;
use std::time::Instant;

use ares_core::GgmlType;
use ares_gguf::{Gguf, TensorInfo};
use ares_vulkan::{Device, DeviceBuffer, Kernels, Layout, Precision, Stream};

/// Rows of a split are multiples of this: a workgroup of the widest kernel.
const ROW_GRANULE: u64 = 128;

struct Options {
    path: String,
    shares: Vec<f64>,
    first_device: u32,
    rows: Vec<u32>,
    passes: Option<u32>,
    precision: Precision,
    device_layout: bool,
    measure: bool,
}

fn parse_args() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut options = Options {
        path: String::new(),
        shares: vec![1.0],
        first_device: 0,
        rows: vec![1, 8],
        passes: None,
        precision: Precision::Fast,
        device_layout: true,
        measure: false,
    };
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--split" => {
                options.shares = value("--split")?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>().map_err(|e| format!("--split '{s}': {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--rows" => {
                options.rows = value("--rows")?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map_err(|e| format!("--rows '{s}': {e}")))
                    .collect::<Result<_, _>>()?;
            }
            "--passes" => {
                options.passes = Some(value("--passes")?.parse().map_err(|e| format!("{e}"))?);
            }
            "--device" => {
                options.first_device = value("--device")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--precision" => {
                options.precision = match value("--precision")?.as_str() {
                    "fast" => Precision::Fast,
                    "exact" => Precision::Exact,
                    other => return Err(format!("--precision: fast or exact, not {other}")),
                };
            }
            "--layout" => {
                options.device_layout = match value("--layout")?.as_str() {
                    "device" => true,
                    "file" => false,
                    other => return Err(format!("--layout: device or file, not {other}")),
                };
            }
            "--measure" => options.measure = true,
            other if options.path.is_empty() && !other.starts_with("--") => {
                options.path = other.to_owned();
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if options.path.is_empty() || options.shares.iter().any(|s| *s <= 0.0) {
        return Err("usage: vk_weight_pass <model.gguf> [--split a,b] [--rows 1,8] [--passes N] \
                    [--device N] [--precision fast|exact] [--layout device|file] [--measure]"
            .into());
    }
    Ok(options)
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
        bounds.push(end.max(*bounds.last().unwrap_or(&0)));
    }
    bounds.windows(2).map(|w| (w[0], w[1] - w[0])).collect()
}

/// Per number of rows: the rows, and the fastest and the mean pass in milliseconds.
type PassTimes = Vec<(u32, f64, f64)>;

/// One device's slice of one matrix, on the device.
struct Slice {
    ty: GgmlType,
    n_in: u32,
    rows: u32,
    layout: Layout,
    buffer: DeviceBuffer,
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let options = parse_args()?;
    let gguf = Gguf::open(&options.path).map_err(|e| e.to_string())?;
    let matrices: Vec<&TensorInfo> = gguf
        .tensors()
        .iter()
        .filter(|t| {
            t.dims.as_slice().len() == 2
                && t.name != "token_embd.weight"
                && Kernels::has_fast_matvec(t.ty)
                && (t.dims.get(0) as u32).is_multiple_of(t.ty.block_values().max(32))
        })
        .collect();
    let bytes: u64 = matrices.iter().map(|t| t.size).sum();
    println!(
        "{}: {} matrices, {:.2} GiB of {} tensors",
        options.path,
        matrices.len(),
        bytes as f64 / (1u64 << 30) as f64,
        gguf.tensors().len()
    );
    let n_devices = options.shares.len();
    let start = Barrier::new(n_devices);
    let results: Vec<Result<PassTimes, String>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_devices)
            .map(|d| {
                let (options, gguf, matrices, start) = (&options, &gguf, &matrices, &start);
                scope.spawn(move || run_device(d, options, gguf, matrices, start))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("a device thread panicked".into())))
            .collect()
    });
    let results: Vec<PassTimes> = results.into_iter().collect::<Result<_, _>>()?;
    for (i, &rows) in options.rows.iter().enumerate() {
        let per_device: Vec<String> =
            results.iter().map(|r| format!("{:.2} (mean {:.2})", r[i].1, r[i].2)).collect();
        let pass = results.iter().map(|r| r[i].1).fold(0.0, f64::max);
        println!(
            "rows {rows:>5}: pass {pass:.2} ms ({:.0} rows/s); per device, fastest (mean): {}",
            f64::from(rows) / pass * 1e3,
            per_device.join(" | ")
        );
    }
    Ok(())
}

fn run_device(
    d: usize,
    options: &Options,
    gguf: &Gguf,
    matrices: &[&TensorInfo],
    start: &Barrier,
) -> Result<PassTimes, String> {
    let result = device_passes(d, options, gguf, matrices, start);
    if result.is_err() {
        // The other devices wait at the barrier of every measurement.
        for _ in &options.rows {
            start.wait();
        }
    }
    result
}

fn device_passes(
    d: usize,
    options: &Options,
    gguf: &Gguf,
    matrices: &[&TensorInfo],
    start: &Barrier,
) -> Result<PassTimes, String> {
    let e = |e: ares_vulkan::Error| e.to_string();
    let device = Device::open(options.first_device + d as u32).map_err(e)?;
    let kernels = Kernels::load(&device).map_err(e)?;
    let name = device.info().name.clone();
    let stream = Stream::new(&device).map_err(e)?;
    let layout_of = |t: &TensorInfo| {
        if options.device_layout {
            kernels.layout_for(t.ty, t.dims.get(0) as u32)
        } else {
            Layout::File
        }
    };
    let mut types: Vec<GgmlType> = matrices.iter().map(|t| t.ty).collect();
    types.sort_by_key(|t| t.id());
    types.dedup();
    kernels.prepare(&types).map_err(e)?;
    if options.measure {
        for &ty in &types {
            if matrices.iter().any(|t| t.ty == ty && layout_of(t) == Layout::Device) {
                let table = kernels.measure_layout(ty).map_err(e)?;
                let picks: Vec<String> = table
                    .rows
                    .iter()
                    .zip(table.file_ms)
                    .map(|(r, file)| {
                        format!("{} {:.3} (file {:.3})", r.fast.name, r.fast.probe_ms, file)
                    })
                    .collect();
                println!("{name}: {} table, ms: {}", ty.name(), picks.join(", "));
            }
        }
    }

    // Upload: the slice in the file's layout, rearranged on the device where it gets the
    // device layout.
    let loading = Instant::now();
    let largest = matrices
        .iter()
        .map(|t| {
            let (_, rows) = split_rows(t.dims.get(1), &options.shares)[d];
            rows * t.ty.bytes_for(t.dims.get(0)).unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    let scratch = DeviceBuffer::new(&device, largest.next_multiple_of(16).max(16)).map_err(e)?;
    let mut slices = Vec::new();
    let mut held = 0u64;
    for t in matrices {
        let n_in = t.dims.get(0) as u32;
        let (row0, rows) = split_rows(t.dims.get(1), &options.shares)[d];
        if rows == 0 {
            continue;
        }
        let row_bytes = t.ty.bytes_for(u64::from(n_in)).ok_or("a row of whole blocks")? as usize;
        let data = &gguf.data(t)[row0 as usize * row_bytes..][..rows as usize * row_bytes];
        let layout = layout_of(t);
        let bytes = Kernels::layout_bytes(t.ty, layout, rows as u32, n_in).ok_or("layout bytes")?;
        let buffer = DeviceBuffer::new(&device, bytes.next_multiple_of(16)).map_err(e)?;
        match layout {
            Layout::File => buffer.write(0, data).map_err(e)?,
            Layout::Device => {
                scratch.write(0, data).map_err(e)?;
                // SAFETY: the scratch holds the slice in the file's layout and the buffer
                // has the bytes of its device layout.
                unsafe {
                    kernels
                        .rearrange(&stream, t.ty, scratch.ptr(), buffer.ptr(), rows as u32, n_in)
                        .map_err(e)?;
                }
                stream.synchronize().map_err(e)?;
            }
        }
        held += bytes;
        slices.push(Slice { ty: t.ty, n_in, rows: rows as u32, layout, buffer });
    }
    drop(scratch);
    let in_device_layout = slices.iter().filter(|s| s.layout == Layout::Device).count();
    println!(
        "{name}: {} slices ({} in the device layout), {:.2} GiB, loaded in {:.1} s",
        slices.len(),
        in_device_layout,
        held as f64 / (1u64 << 30) as f64,
        loading.elapsed().as_secs_f64()
    );

    let max_rows = options.rows.iter().copied().max().unwrap_or(1);
    let max_out = slices.iter().map(|s| s.rows).max().unwrap_or(0);
    let max_in = slices.iter().map(|s| s.n_in).max().unwrap_or(0);
    let x = DeviceBuffer::new(&device, u64::from(max_in) * u64::from(max_rows) * 4).map_err(e)?;
    // Activations near 0.01: what they are does not change the time of a product.
    // SAFETY: a whole buffer of this device.
    unsafe { stream.fill(x.ptr(), 0x3c, x.len() as usize).map_err(e)? };
    let mut out = Vec::new();
    for &rows in &options.rows {
        let measure = || -> Result<(f64, f64), String> {
            let y =
                DeviceBuffer::new(&device, u64::from(max_out) * u64::from(rows) * 4).map_err(e)?;
            // One prepared buffer per input width and form.
            let mut prepared: BTreeMap<(u32, u8), DeviceBuffer> = BTreeMap::new();
            let mut plan = Vec::new();
            for s in &slices {
                let Ok(form) = kernels.activation_form(s.ty, s.layout, options.precision, rows)
                else {
                    // No kernel for this many rows in this layout on this device.
                    continue;
                };
                let key = (s.n_in, form as u8);
                if let std::collections::btree_map::Entry::Vacant(slot) = prepared.entry(key) {
                    let bytes = Kernels::activation_bytes(form, s.n_in, rows).next_multiple_of(16);
                    let buffer = DeviceBuffer::new(&device, bytes).map_err(e)?;
                    // SAFETY: `x` holds the rows, the buffer their prepared form.
                    unsafe {
                        kernels
                            .prepare_activations(&stream, form, x.ptr(), buffer.ptr(), s.n_in, rows)
                            .map_err(e)?;
                    }
                    slot.insert(buffer);
                }
                plan.push((s, key));
            }
            if plan.len() < slices.len() {
                println!(
                    "{name}: rows {rows}: {} slices have no kernel for this many rows and are left out",
                    slices.len() - plan.len()
                );
            }
            stream.synchronize().map_err(e)?;
            let passes = options.passes.unwrap_or(if rows <= 8 { 20 } else { 5 });
            let pass = || -> Result<f64, String> {
                let began = Instant::now();
                for (s, key) in &plan {
                    // SAFETY: the slice, its prepared activations and room for its products.
                    unsafe {
                        kernels
                            .product(
                                &stream,
                                s.ty,
                                s.layout,
                                options.precision,
                                s.buffer.ptr(),
                                prepared[key].ptr(),
                                y.ptr(),
                                s.n_in,
                                s.rows,
                                rows,
                            )
                            .map_err(e)?;
                    }
                }
                stream.synchronize().map_err(e)?;
                Ok(began.elapsed().as_secs_f64() * 1e3)
            };
            // Two passes to warm up, then all devices start together.
            pass()?;
            pass()?;
            start.wait();
            let times: Vec<f64> = (0..passes).map(|_| pass()).collect::<Result<_, _>>()?;
            let best = times.iter().copied().fold(f64::INFINITY, f64::min);
            Ok((best, times.iter().sum::<f64>() / times.len() as f64))
        };
        match measure() {
            Ok((best, mean)) => out.push((rows, best, mean)),
            Err(message) => {
                println!("{name}: rows {rows} skipped: {message}");
                start.wait();
                out.push((rows, f64::NAN, f64::NAN));
            }
        }
    }
    Ok(out)
}
