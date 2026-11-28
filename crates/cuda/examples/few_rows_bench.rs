//! Times a decode step's products of one matrix shape: in the file's layout for every
//! type with a fast weight kernel, and for a type with a device layout also in that
//! layout, at 1 to 16 activation rows.
//!
//! The file's layout runs what `Kernels::product` runs there: the integer decode kernels
//! up to their rows (`Kernels::decode_kernel_rows`), the tile kernels above. The device
//! layout runs the few-row kernels. The matrices of a pool larger than any L2 cache are
//! walked in turn, as the matrices of a model are, so every launch reads its weights from
//! memory; the products of both layouts are timed in turn, in bursts, and the fastest
//! burst of each counts. With `--plans` each launch of the few-row kernels
//! ([`FewPlan::all`]) is timed too, whatever `Kernels::tune_layout` chose.
//!
//! Decode is bound by how fast the weights can be read, so each product also gives GB/s
//! of weight bytes, to set against the card's memory bandwidth (479 GB/s on an RTX 4070
//! and 341 on an RTX 3060, measured with a plain streaming read).
//!
//! ```sh
//! cargo run --release -p ares-cuda --example few_rows_bench -- \
//!     [--device N] [--rows 1,2,3,8] [--pool-mib 1536] [--plans] q4_K:6400x5120 iq4_xs:5120x6656
//! cargo run --release -p ares-cuda --example few_rows_bench -- --rows 1,2,4,8 all:17408x5120
//! ```
//!
//! A shape is `<type>:<rows>x<columns>` of the matrix; the type `all` is every type with a
//! fast kernel, and without a shape it is `all:17408x5120` (the FFN of Qwen3.8-27B). The
//! output is microseconds per launch and GB/s of weights, and, in percent, the device
//! layout's time against the file's (below 100: the device layout is the faster).

use ares_core::GgmlType;
use ares_core::ggml_type::ALL_TYPES;
use ares_cuda::{
    Device, DeviceBuffer, Event, FewPlan, Kernels, Precision, Result, Stream, Wait, WeightLayout,
    devices,
};

const BURST: u32 = 16;
const ROUNDS: usize = 12;

/// The shape timed when none is named: the largest matrix of Qwen3.8-27B, the FFN of a
/// layer.
const DEFAULT_SHAPE: &str = "all:17408x5120";

fn main() -> Result<()> {
    let mut rows = vec![1u32, 2, 3, 8];
    let mut only_device: Option<u32> = None;
    let mut pool_mib = 1536u64;
    let mut plans = false;
    let mut shapes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--rows" => {
                let value = args.next().expect("--rows needs a value");
                rows = value.split(',').map(|v| v.parse().expect("--rows")).collect();
            }
            "--device" => {
                only_device = Some(args.next().expect("--device").parse().expect("--device"))
            }
            "--pool-mib" => {
                pool_mib = args.next().expect("--pool-mib").parse().expect("--pool-mib")
            }
            "--plans" => plans = true,
            shape => shapes.extend(parse_shape(shape)),
        }
    }
    if shapes.is_empty() {
        shapes = parse_shape(DEFAULT_SHAPE);
    }
    for info in devices()?.into_iter().filter(|i| only_device.is_none_or(|d| d == i.ordinal)) {
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let kernels = Kernels::load(&device)?;
        println!("CUDA{} {}", info.ordinal, info.name);
        for &(ty, n_out, n_in) in &shapes {
            shape(&device, &kernels, ty, n_out, n_in, &rows, pool_mib << 20, plans)?;
        }
    }
    Ok(())
}

/// The matrices of `<type>:<rows>x<columns>`: one, or one of every type with a fast
/// kernel for the type `all`.
fn parse_shape(shape: &str) -> Vec<(GgmlType, u32, u32)> {
    let (name, dims) = shape.split_once(':').expect("<type>:<rows>x<columns>");
    let (o, i) = dims.split_once('x').expect("<type>:<rows>x<columns>");
    let (n_out, n_in) = (o.parse::<u32>().expect("rows"), i.parse::<u32>().expect("columns"));
    let fast = ALL_TYPES.into_iter().filter(|t| Kernels::has_fast_matvec(*t));
    let types: Vec<GgmlType> = if name == "all" {
        fast.collect()
    } else {
        let ty = fast
            .clone()
            .find(|t| t.name().eq_ignore_ascii_case(name))
            .unwrap_or_else(|| panic!("{name} has no fast kernel; these do: {}", names(fast)));
        vec![ty]
    };
    types.into_iter().map(|ty| (ty, n_out, n_in)).collect()
}

fn names(types: impl Iterator<Item = GgmlType>) -> String {
    types.map(GgmlType::name).collect::<Vec<_>>().join(", ")
}

/// `copies` matrices of the bytes `host` on the device.
fn pool(device: &Device, copies: usize, host: &[u8]) -> Result<Vec<DeviceBuffer>> {
    (0..copies)
        .map(|_| {
            let b = DeviceBuffer::new(device, host.len() as u64)?;
            b.write(0, host)?;
            Ok(b)
        })
        .collect()
}

/// One kind of launch the bench times.
#[derive(Clone, Copy)]
enum Launch {
    /// `Kernels::product` over the file's layout.
    File,
    /// `Kernels::product` over the device layout, as a model runs it.
    Device,
    /// One launch of the few-row kernels over the device layout.
    Plan(FewPlan),
}

#[allow(clippy::too_many_arguments)]
fn shape(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    n_out: u32,
    n_in: u32,
    rows: &[u32],
    pool_bytes: u64,
    all_plans: bool,
) -> Result<()> {
    let stream = Stream::new(device, 0)?;
    let Some(file_bytes) = Kernels::layout_bytes(ty, WeightLayout::File, n_out, n_in) else {
        println!("{} {n_out}x{n_in}: rows of {n_in} are not whole blocks", ty.name());
        return Ok(());
    };
    // The device layout, if the type has one for rows this wide.
    let device_bytes = Kernels::layout_bytes(ty, WeightLayout::Device, n_out, n_in);
    let copies = (pool_bytes / file_bytes.max(1)).clamp(2, 4096) as usize;
    // Random bytes with small scales (the high bits of every byte cleared: finite f16
    // fields, and finite f32 and bf16 values).
    let mut state = 0x9e3779b97f4a7c15u64;
    let host: Vec<u8> = (0..file_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as u8) & 0x6f
        })
        .collect();
    let files = pool(device, copies, &host)?;
    let rearranged = match device_bytes {
        Some(bytes) => files
            .iter()
            .map(|f| {
                let r = DeviceBuffer::new(device, bytes)?;
                // SAFETY: whole matrices of `ty` and the size the function names.
                unsafe { kernels.rearrange(&stream, ty, f.ptr(), r.ptr(), n_out, n_in)? };
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    let max_rows = *rows.iter().max().expect("rows");
    let x = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_in) * 4)?;
    let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_out) * 4)?;
    // SAFETY: whole buffers on this device.
    unsafe { stream.fill(x.ptr(), 0x3c, (u64::from(max_rows) * u64::from(n_in) * 4) as usize)? };
    let mut head = format!(
        "{} {n_out}x{n_in} ({:.1} MB, {copies} copies)",
        ty.name(),
        file_bytes as f64 / 1e6
    );
    if device_bytes.is_some() {
        let plans = kernels.tune_layout(ty, n_out, n_in)?;
        let name = |p: FewPlan| format!("{}w/{}", p.block_warps, 1u32 << p.log2_warps_per_tile);
        head += &format!(
            ", launches {} up to 8 rows, {} up to 16",
            name(plans.rows_8),
            name(plans.rows_16)
        );
    } else {
        head += ", no device layout";
    }
    println!("{head}");
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    for &n in rows {
        // The launches: the file's layout, the device layout as a model runs it, and each
        // launch of the family; each reads the activations in the form its layout asks.
        let mut names = vec!["file".to_string()];
        let mut launches = vec![Launch::File];
        if device_bytes.is_some() {
            names.push("device".to_string());
            launches.push(Launch::Device);
            if all_plans {
                for p in FewPlan::all() {
                    names.push(format!("{}w/{}", p.block_warps, 1u32 << p.log2_warps_per_tile));
                    launches.push(Launch::Plan(p));
                }
            }
        }
        let mut acts = Vec::new();
        for launch in &launches {
            let layout = match launch {
                Launch::File => WeightLayout::File,
                Launch::Device | Launch::Plan(_) => WeightLayout::Device,
            };
            let Ok(form) = kernels.activation_form(ty, layout, Precision::Fast, n) else {
                break;
            };
            let act = DeviceBuffer::new(device, Kernels::activation_bytes(form, n_in, n))?;
            // SAFETY: the buffer was sized for these rows just above.
            unsafe { kernels.prepare_activations(&stream, form, x.ptr(), act.ptr(), n_in, n)? };
            acts.push(act);
        }
        if acts.len() < launches.len() {
            println!("  {n:>2} rows: no kernel for this many rows");
            continue;
        }
        stream.synchronize()?;
        let mut best = vec![f32::INFINITY; names.len()];
        let mut host = vec![f64::INFINITY; names.len()];
        for round in 0..=ROUNDS {
            for (k, best) in best.iter_mut().enumerate() {
                start.record(&stream)?;
                let issued = std::time::Instant::now();
                for i in 0..BURST as usize {
                    let at = i % copies;
                    let act = acts[k].ptr();
                    // SAFETY: the pools hold whole matrices, `act` the rows prepared in the
                    // form of the launch's layout and `y` the products.
                    unsafe {
                        match launches[k] {
                            Launch::File => kernels.product(
                                &stream,
                                ty,
                                WeightLayout::File,
                                Precision::Fast,
                                files[at].ptr(),
                                act,
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                            Launch::Device => kernels.product(
                                &stream,
                                ty,
                                WeightLayout::Device,
                                Precision::Fast,
                                rearranged[at].ptr(),
                                act,
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                            Launch::Plan(p) => kernels.product_few(
                                &stream,
                                ty,
                                p,
                                rearranged[at].ptr(),
                                act,
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                        }
                    }
                }
                let issue_us = issued.elapsed().as_secs_f64() * 1e6 / f64::from(BURST);
                end.record(&stream)?;
                end.synchronize()?;
                if round > 0 {
                    *best = best.min(end.elapsed_ms_since(&start)? / BURST as f32);
                    host[k] = host[k].min(issue_us);
                }
            }
        }
        let gb_per_s = |bytes: u64, ms: f32| bytes as f64 / (f64::from(ms) * 1e-3) / 1e9;
        let file_us = best[0] * 1000.0;
        let mut line = format!(
            "  {n:>2} rows: file {file_us:>8.2} us {:>6.1} GB/s (host {:.2})",
            gb_per_s(file_bytes, best[0]),
            host[0]
        );
        for (name, ms) in names.iter().zip(&best).skip(1) {
            line += &format!(
                " | {name} {:>8.2} us {:>6.1} GB/s {:>4.0}%",
                ms * 1000.0,
                gb_per_s(device_bytes.unwrap_or(file_bytes), *ms),
                100.0 * ms * 1000.0 / file_us
            );
        }
        println!("{line}");
    }
    Ok(())
}
