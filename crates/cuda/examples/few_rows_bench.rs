//! Times a decode step's products of one matrix shape in both layouts: the integer
//! decode kernels over the file's layout against the few-row kernels over the device
//! layout, at 1 to 16 activation rows.
//!
//! The matrices of a pool larger than any L2 cache are walked in turn, as the matrices
//! of a model are, so every launch reads its weights from memory; the products of both
//! layouts are timed in turn, in bursts, and the fastest burst of each counts.
//! `Kernels::product` is the call a model makes; with `--plans` each launch of the
//! few-row kernels ([`FewPlan::all`]) is timed too, whatever `Kernels::tune_layout`
//! chose.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example few_rows_bench -- \
//!     [--device N] [--rows 1,2,3,8] [--pool-mib 1536] [--plans] q4_K:6400x5120 iq4_xs:5120x6656
//! ```
//!
//! A shape is `<type>:<rows>x<columns>` of the matrix. The output is microseconds per
//! launch and, in percent, the device layout's time against the file's (below 100: the
//! device layout is the faster).

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, Event, FewPlan, Kernels, Precision, Result, Stream, Wait,
    WeightLayout, devices,
};

const BURST: u32 = 16;
const ROUNDS: usize = 12;

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
            shape => {
                let (ty, dims) = shape.split_once(':').expect("<type>:<rows>x<columns>");
                let (o, i) = dims.split_once('x').expect("<type>:<rows>x<columns>");
                let ty = *[
                    GgmlType::Q4_K,
                    GgmlType::Q5_K,
                    GgmlType::Q6_K,
                    GgmlType::Q3_K,
                    GgmlType::Q2_K,
                    GgmlType::Q4_0,
                    GgmlType::Q4_1,
                    GgmlType::Q5_0,
                    GgmlType::Q5_1,
                    GgmlType::Q8_0,
                    GgmlType::IQ4_XS,
                    GgmlType::IQ4_NL,
                ]
                .iter()
                .find(|t| t.name().eq_ignore_ascii_case(ty))
                .unwrap_or_else(|| panic!("unknown type {ty}"));
                shapes.push((
                    ty,
                    o.parse::<u32>().expect("rows"),
                    i.parse::<u32>().expect("columns"),
                ));
            }
        }
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

#[allow(clippy::too_many_arguments)]
fn shape(
    device: &Device,
    kernels: &Kernels,
    ty: GgmlType,
    n_out: u32,
    n_in: u32,
    rows: &[u32],
    pool: u64,
    all_plans: bool,
) -> Result<()> {
    let stream = Stream::new(device, 0)?;
    let file_bytes =
        Kernels::layout_bytes(ty, WeightLayout::File, n_out, n_in).expect("whole blocks");
    let device_bytes =
        Kernels::layout_bytes(ty, WeightLayout::Device, n_out, n_in).expect("has the layout");
    let copies = (pool / file_bytes.max(1)).clamp(2, 4096) as usize;
    // Random bytes with small scales (the high bits of every byte cleared: finite f16 fields).
    let mut state = 0x9e3779b97f4a7c15u64;
    let host: Vec<u8> = (0..file_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as u8) & 0x6f
        })
        .collect();
    let files = (0..copies)
        .map(|_| {
            let b = DeviceBuffer::new(device, file_bytes)?;
            b.write(0, &host)?;
            Ok(b)
        })
        .collect::<Result<Vec<_>>>()?;
    let rearranged = files
        .iter()
        .map(|f| {
            let r = DeviceBuffer::new(device, device_bytes)?;
            // SAFETY: whole matrices of `ty` and the size the function names.
            unsafe { kernels.rearrange(&stream, ty, f.ptr(), r.ptr(), n_out, n_in)? };
            Ok(r)
        })
        .collect::<Result<Vec<_>>>()?;
    let max_rows = *rows.iter().max().expect("rows");
    let x = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_in) * 4)?;
    let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_out) * 4)?;
    // SAFETY: whole buffers on this device.
    unsafe { stream.fill(x.ptr(), 0x3c, (u64::from(max_rows) * u64::from(n_in) * 4) as usize)? };
    let plan = kernels.tune_layout(ty, n_out, n_in)?;
    println!(
        "{} {n_out}x{n_in} ({:.1} MB, {copies} copies), launch {} warps, {} per tile",
        ty.name(),
        file_bytes as f64 / 1e6,
        plan.block_warps,
        1u32 << plan.log2_warps_per_tile
    );
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);
    for &n in rows {
        let q =
            DeviceBuffer::new(device, Kernels::activation_bytes(ActivationForm::Int8, n_in, n))?;
        let h = DeviceBuffer::new(device, Kernels::activation_bytes(ActivationForm::F16, n_in, n))?;
        // SAFETY: the buffers were sized for these shapes just above.
        unsafe {
            kernels.prepare_activations(
                &stream,
                ActivationForm::Int8,
                x.ptr(),
                q.ptr(),
                n_in,
                n,
            )?;
            kernels.prepare_activations(&stream, ActivationForm::F16, x.ptr(), h.ptr(), n_in, n)?;
        }
        stream.synchronize()?;
        // The launches: the file's layout, the device layout as a model runs it, and
        // each launch of the family.
        let mut names = vec!["file".to_string(), "device".to_string()];
        let mut launches: Vec<Option<FewPlan>> = vec![None, None];
        if all_plans {
            for p in FewPlan::all() {
                names.push(format!("{}w/{}", p.block_warps, 1u32 << p.log2_warps_per_tile));
                launches.push(Some(p));
            }
        }
        let mut best = vec![f32::INFINITY; names.len()];
        let mut host = vec![f64::INFINITY; names.len()];
        for round in 0..=ROUNDS {
            for (k, best) in best.iter_mut().enumerate() {
                start.record(&stream)?;
                let issued = std::time::Instant::now();
                for i in 0..BURST as usize {
                    let at = i % copies;
                    // SAFETY: the pools hold whole matrices, `q` and `h` the prepared rows
                    // and `y` the products.
                    unsafe {
                        match (k, launches[k]) {
                            (0, _) => kernels.product(
                                &stream,
                                ty,
                                WeightLayout::File,
                                Precision::Fast,
                                files[at].ptr(),
                                q.ptr(),
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                            (1, _) => kernels.product(
                                &stream,
                                ty,
                                WeightLayout::Device,
                                Precision::Fast,
                                rearranged[at].ptr(),
                                h.ptr(),
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                            (_, Some(p)) => kernels.product_few(
                                &stream,
                                ty,
                                p,
                                rearranged[at].ptr(),
                                h.ptr(),
                                y.ptr(),
                                n_in,
                                n_out,
                                n,
                            )?,
                            _ => unreachable!("every launch past the first two is a plan"),
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
        let file_us = best[0] * 1000.0;
        let mut line = format!("  {n:>2} rows: file {file_us:>8.2} us (host {:.2})", host[0]);
        for (name, ms) in names.iter().zip(&best).skip(1) {
            line += &format!(
                " | {name} {:>8.2} us {:>4.0}%",
                ms * 1000.0,
                100.0 * ms * 1000.0 / file_us
            );
        }
        println!("{line}");
    }
    Ok(())
}
