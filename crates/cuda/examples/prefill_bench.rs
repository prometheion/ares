//! Times the prefill weight kernels (the tensor-core family) on each device.
//!
//! Prompt processing is bound by arithmetic, so the number to watch is TOPS
//! (2 * n_out * n_in * rows per second) against what the tensor cores can do:
//! on an RTX 4070 about 234 for the integer `mma`, 117 for f16 with f32
//! accumulation once the cap is lifted and 58 while it is capped; on an RTX
//! 3060 102, 51 and 26 (measured with a loop of `mma` alone).
//!
//! For every type, a 17408 x 5120 matrix (the FFN of Qwen3.8-27B) is multiplied
//! with `rows` activation rows by each path, 10 launches in a row between two
//! events, over several copies of the matrix that together are larger than any
//! L2 cache. The decode kernels are timed for comparison at up to `--decode-rows` rows (64
//! unless said otherwise, in launches of 8). The fast path of the device is marked.
//!
//! For a type with a device layout (`Kernels::has_device_layout`), `layout` and
//! `layout-exact` time `Kernels::product` over the rearranged matrix with the kernel
//! the device's measured table names (from 17 rows on; up to 16 the few-row kernels, as
//! `Kernels::tune_layout` measured the shape), and a kernel's name (`r32f`, `r64h`, ...)
//! times that kernel whatever the table says; `few-<warps in a block>-<log2 of warps on a
//! tile>` (`few-4-0`, `few-4-2`, `few-8-3`) times one launch of the few-row kernels (up
//! to 16 rows).
//!
//! ```sh
//! cargo run --release -p ares-cuda --example prefill_bench -- \
//!     [--rows 64,256,1024] [--types q4_K,iq4_xs] [--device N] [--shape 17408x5120] \
//!     [--decode-rows 64] [--paths int8,f16,f16flush,decode,layout,layout-exact,r8f,...]
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, Event, FewPlan, Kernels, MAX_BATCH, Precision,
    PrefillPath, Result, Stream, Wait, WeightLayout, devices,
};

/// The largest matrix of Qwen3.8-27B, the FFN of a layer, unless `--shape` says another.
static SHAPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(17408 << 32 | 5120);

fn n_in() -> u32 {
    SHAPE.load(std::sync::atomic::Ordering::Relaxed) as u32
}

fn n_out() -> u32 {
    (SHAPE.load(std::sync::atomic::Ordering::Relaxed) >> 32) as u32
}

/// The types with a tile kernel over the file's layout, in the order of their bytes
/// in Qwen3.8-27B, then the others that have a device layout, then the float types.
const TYPES: [GgmlType; 28] = [
    GgmlType::IQ4_XS,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q3_K,
    GgmlType::IQ3_S,
    GgmlType::IQ4_NL,
    GgmlType::IQ3_XXS,
    GgmlType::IQ2_S,
    GgmlType::IQ2_XS,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::IQ2_XXS,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::Q2_0,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::BF16,
    GgmlType::F16,
    GgmlType::F32,
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Int8,
    F16,
    /// f16 with f16 accumulation flushed to f32 ([`PrefillPath::F16Flush`]).
    F16Flush,
    /// The device layout, with the kernel of the device's table for a precision.
    Layout(Precision),
    /// The device layout, with one kernel of the family above 16 rows by name.
    Kernel(&'static str),
    /// The device layout, with one launch of the few-row kernels (up to 16 rows).
    Few(FewPlan),
    Decode,
}

fn main() -> Result<()> {
    let mut rows = vec![64u32, 256, 1024];
    let mut types: Vec<GgmlType> = TYPES.to_vec();
    let mut paths = vec![Which::Int8, Which::F16, Which::Decode];
    let mut only_device: Option<u32> = None;
    let mut decode_rows = 64u32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--rows" => rows = value.split(',').map(|v| v.parse().expect("--rows")).collect(),
            "--types" => {
                types = value
                    .split(',')
                    .map(|name| {
                        *TYPES
                            .iter()
                            .find(|t| t.name().eq_ignore_ascii_case(name))
                            .unwrap_or_else(|| panic!("--types: unknown type {name}"))
                    })
                    .collect();
            }
            "--device" => only_device = Some(value.parse().expect("--device")),
            "--decode-rows" => decode_rows = value.parse().expect("--decode-rows"),
            "--shape" => {
                let (o, i) = value.split_once('x').expect("--shape <rows>x<columns>");
                let (o, i): (u64, u64) = (o.parse().expect("--shape"), i.parse().expect("--shape"));
                SHAPE.store(o << 32 | i, std::sync::atomic::Ordering::Relaxed);
            }
            "--paths" => {
                paths = value
                    .split(',')
                    .map(|p| match p {
                        "int8" => Which::Int8,
                        "f16" => Which::F16,
                        "f16flush" => Which::F16Flush,
                        "layout" => Which::Layout(Precision::Fast),
                        "layout-exact" => Which::Layout(Precision::Exact),
                        "decode" => Which::Decode,
                        name if name.starts_with("few-") => {
                            let parts: Vec<u32> =
                                name[4..].split('-').filter_map(|p| p.parse().ok()).collect();
                            let plan = FewPlan::all()
                                .into_iter()
                                .find(|p| parts == [p.block_warps, p.log2_warps_per_tile]);
                            Which::Few(plan.unwrap_or_else(|| {
                                panic!("--paths: few-<warps>-<log2 per tile> of {:?}", FewPlan::all())
                            }))
                        }
                        name => Which::Kernel(
                            Kernels::layout_kernel_names()
                                .into_iter()
                                .find(|n| *n == name)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "--paths: int8, f16, f16flush, decode, layout, layout-exact or one of {:?}",
                                        Kernels::layout_kernel_names()
                                    )
                                }),
                        ),
                    })
                    .collect();
            }
            other => panic!("unknown option {other}"),
        }
    }

    for info in devices()?.into_iter().filter(|i| only_device.is_none_or(|d| d == i.ordinal)) {
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let kernels = Kernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        let fast = kernels.prefill_path(Precision::Fast)?;
        println!(
            "CUDA{} {}: {} x {}; f16 mma with f32 accumulation at full rate: {} (fast path {fast:?})",
            info.ordinal,
            info.name,
            n_out(),
            n_in(),
            kernels.tensor_cores_full_rate()
        );
        let tiled = types.iter().copied().find(|t| kernels.has_prefill_path(*t, PrefillPath::Int8));
        for path in [PrefillPath::Int8, PrefillPath::F16, PrefillPath::F16Flush] {
            let Some(tiled) = tiled else { break };
            for shape in kernels.prefill_shapes(tiled, path)? {
                println!(
                    "  {path:?} shape {}x{} rows of weights x activations, {} threads, {} B of shared memory (for {}), {} registers, {} block(s) per SM, up to {} rows",
                    shape.weight_rows,
                    shape.batch_rows,
                    shape.threads,
                    shape.shared_bytes,
                    tiled.name(),
                    shape.registers,
                    shape.blocks_per_sm,
                    if shape.up_to_rows == u32::MAX {
                        "any".to_string()
                    } else {
                        shape.up_to_rows.to_string()
                    }
                );
            }
        }
        for &ty in &types {
            time_type(&device, &kernels, &stream, ty, &rows, &paths, fast, decode_rows)?;
        }
    }
    Ok(())
}

/// Launches per timed run.
const RUN: usize = 10;

#[allow(clippy::too_many_arguments)]
fn time_type(
    device: &Device,
    kernels: &Kernels,
    stream: &Stream,
    ty: GgmlType,
    rows: &[u32],
    paths: &[Which],
    fast: PrefillPath,
    decode_rows: u32,
) -> Result<()> {
    let w_bytes = ty.bytes_for(u64::from(n_out()) * u64::from(n_in())).expect("whole blocks");
    let copies = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
    // Random bytes: any block is a valid block of these types, and uniform data
    // would draw less power than weights do. The f16 scales stay finite.
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state >> 32
    };
    let mut host: Vec<u8> = if matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        // Weights of the size of a weight: random bits would be NaNs and infinities.
        (0..w_bytes / u64::from(ty.block_bytes()))
            .flat_map(|_| {
                let v = (next() as f32 / (1u64 << 31) as f32 - 1.0) * 0.1;
                match ty {
                    GgmlType::F32 => v.to_le_bytes().to_vec(),
                    GgmlType::F16 => ares_quant::f32_to_f16(v).to_le_bytes().to_vec(),
                    _ => ares_quant::f32_to_bf16(v).to_le_bytes().to_vec(),
                }
            })
            .collect()
    } else {
        (0..w_bytes).map(|_| next() as u8).collect()
    };
    for block in host.chunks_exact_mut(ty.block_bytes() as usize) {
        for scale_at in scale_offsets(ty) {
            block[*scale_at..*scale_at + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
        if ty == GgmlType::IQ1_M {
            // The f16 d is spread over the top nibbles of the four u16 of the scales.
            for i in 0..4 {
                let word = u16::from_le_bytes([block[48 + 2 * i], block[49 + 2 * i]]);
                let word = (word & 0x0FFF) | (((0x2c00u16 >> (4 * i)) & 0xF) << 12);
                block[48 + 2 * i..50 + 2 * i].copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    let weights = (0..copies)
        .map(|_| {
            let buffer = DeviceBuffer::new(device, w_bytes)?;
            buffer.write(0, &host)?;
            Ok(buffer)
        })
        .collect::<Result<Vec<_>>>()?;
    // The rearranged copies of the matrices, when asked for and the type has the layout,
    // and the table of the device.
    let wants_layout =
        paths.iter().any(|p| matches!(p, Which::Layout(_) | Which::Kernel(_) | Which::Few(_)));
    let rearranged = if wants_layout && Kernels::has_device_layout(ty, n_in()) {
        let plan = kernels.tune_layout(ty, n_out(), n_in())?;
        println!("  {} few-row launch: {plan:?}", ty.name());
        let table = kernels.measure_layout(ty)?;
        for rows in table.rows {
            println!(
                "  {} table: up to {} rows fast {} ({:.3} ms on the probe), exact {} ({:.3} ms)",
                ty.name(),
                rows.up_to_rows,
                rows.fast.name,
                rows.fast.probe_ms,
                rows.exact.name,
                rows.exact.probe_ms
            );
        }
        let bytes = Kernels::layout_bytes(ty, WeightLayout::Device, n_out(), n_in())
            .expect("has the layout");
        let buffers = weights
            .iter()
            .map(|w| {
                let r = DeviceBuffer::new(device, bytes)?;
                // SAFETY: whole matrices of `ty` and the size the function names.
                unsafe { kernels.rearrange(stream, ty, w.ptr(), r.ptr(), n_out(), n_in())? };
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        stream.synchronize()?;
        buffers
    } else {
        Vec::new()
    };
    let max_rows = *rows.iter().max().unwrap_or(&1);
    let x = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_in()) * 4)?;
    let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(n_out()) * 4)?;
    // SAFETY: whole buffers on this device.
    unsafe { stream.fill(x.ptr(), 0x3c, (u64::from(max_rows) * u64::from(n_in()) * 4) as usize)? };
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);

    for &n in rows {
        let mut line = format!("  {:<8} {n:>5} rows", ty.name());
        for &which in paths {
            if which == Which::Decode && n > decode_rows {
                continue;
            }
            if matches!(which, Which::Layout(_) | Which::Kernel(_) | Which::Few(_))
                && rearranged.is_empty()
            {
                continue;
            }
            if which == Which::Int8 && !kernels.has_prefill_path(ty, PrefillPath::Int8) {
                continue;
            }
            let ms = match which {
                Which::Layout(_) | Which::Kernel(_) | Which::Few(_) => {
                    let act = DeviceBuffer::new(
                        device,
                        Kernels::activation_bytes(ActivationForm::F16, n_in(), n),
                    )?;
                    // SAFETY: the buffers were sized for these shapes just above.
                    unsafe {
                        kernels.prepare_activations(
                            stream,
                            ActivationForm::F16,
                            x.ptr(),
                            act.ptr(),
                            n_in(),
                            n,
                        )?
                    };
                    time(device, stream, &start, &end, |i| {
                        let w = rearranged[i % copies].ptr();
                        // SAFETY: as above; the matrices were rearranged just before.
                        unsafe {
                            match which {
                                Which::Kernel(name) => kernels.product_with(
                                    stream,
                                    ty,
                                    name,
                                    w,
                                    act.ptr(),
                                    y.ptr(),
                                    n_in(),
                                    n_out(),
                                    n,
                                ),
                                Which::Few(plan) => kernels.product_few(
                                    stream,
                                    ty,
                                    plan,
                                    w,
                                    act.ptr(),
                                    y.ptr(),
                                    n_in(),
                                    n_out(),
                                    n,
                                ),
                                Which::Layout(precision) => kernels.product(
                                    stream,
                                    ty,
                                    WeightLayout::Device,
                                    precision,
                                    w,
                                    act.ptr(),
                                    y.ptr(),
                                    n_in(),
                                    n_out(),
                                    n,
                                ),
                                _ => unreachable!("the arm is the two layout paths"),
                            }
                        }
                    })?
                }
                Which::Int8 | Which::F16 | Which::F16Flush => {
                    let path = match which {
                        Which::Int8 => PrefillPath::Int8,
                        Which::F16 => PrefillPath::F16,
                        _ => PrefillPath::F16Flush,
                    };
                    let act = DeviceBuffer::new(
                        device,
                        Kernels::prefill_activation_bytes(path, n_in(), n),
                    )?;
                    // SAFETY: the buffers were sized for these shapes just above.
                    unsafe {
                        kernels.prepare_prefill_activations(
                            stream,
                            path,
                            x.ptr(),
                            act.ptr(),
                            n_in(),
                            n,
                        )?
                    };
                    time(device, stream, &start, &end, |i| {
                        // SAFETY: as above; the matrices are whole n_out() x n_in() matrices of `ty`.
                        unsafe {
                            kernels.matmul(
                                stream,
                                ty,
                                path,
                                weights[i % copies].ptr(),
                                act.ptr(),
                                y.ptr(),
                                n_in(),
                                n_out(),
                                n,
                            )
                        }
                    })?
                }
                Which::Decode => {
                    let q = DeviceBuffer::new(
                        device,
                        Kernels::quantized_activation_bytes(n_in(), MAX_BATCH),
                    )?;
                    // SAFETY: as above.
                    unsafe {
                        kernels.quantize_activations(stream, x.ptr(), q.ptr(), n_in(), MAX_BATCH)?
                    };
                    time(device, stream, &start, &end, |i| {
                        let mut left = n;
                        while left > 0 {
                            let chunk = left.min(MAX_BATCH);
                            // SAFETY: as above; only the timing matters, not the activations.
                            unsafe {
                                kernels.matvec(
                                    stream,
                                    ty,
                                    weights[i % copies].ptr(),
                                    q.ptr(),
                                    y.ptr(),
                                    n_in(),
                                    n_out(),
                                    chunk,
                                )?
                            };
                            left -= chunk;
                        }
                        Ok(())
                    })?
                }
            };
            let tops =
                2.0 * f64::from(n_out()) * f64::from(n_in()) * f64::from(n) / (ms * 1e-3) / 1e12;
            let name = match which {
                Which::Int8 => "int8",
                Which::F16 => "f16",
                Which::F16Flush => "f16h",
                Which::Layout(Precision::Fast) => "layout",
                Which::Layout(Precision::Exact) => "layout-exact",
                Which::Kernel(name) => name,
                Which::Few(plan) => match (plan.block_warps, plan.log2_warps_per_tile) {
                    (4, 0) => "few-4-0",
                    (4, 1) => "few-4-1",
                    (4, 2) => "few-4-2",
                    _ => "few-8-3",
                },
                Which::Decode => "decode",
            };
            let mark = match (which, fast) {
                (Which::Int8, PrefillPath::Int8) | (Which::F16, PrefillPath::F16) => "*",
                _ => " ",
            };
            line += &format!(
                " | {name}{mark} {ms:>7.3} ms {:>7.0} rows/s {tops:>6.1} TOPS",
                f64::from(n) / (ms * 1e-3)
            );
        }
        println!("{line}");
    }
    Ok(())
}

/// Median over 5 runs of `RUN` launches of the time of one launch, in ms, after
/// a warm-up of at least 300 ms (module load and clocks ramping up).
fn time(
    _device: &Device,
    stream: &Stream,
    start: &Event,
    end: &Event,
    launch: impl Fn(usize) -> Result<()>,
) -> Result<f64> {
    let warm = Instant::now();
    while warm.elapsed().as_millis() < 300 {
        for i in 0..RUN {
            launch(i)?;
        }
        stream.synchronize()?;
    }
    let mut times = Vec::new();
    for _ in 0..5 {
        start.record(stream)?;
        for i in 0..RUN {
            launch(i)?;
        }
        end.record(stream)?;
        end.synchronize()?;
        times.push(f64::from(end.elapsed_ms_since(start)?) / RUN as f64);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

/// The byte offsets of the f16 scales of a block of `ty`.
fn scale_offsets(ty: GgmlType) -> &'static [usize] {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => &[],
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
        GgmlType::Q6_K => &[208],
        GgmlType::Q3_K => &[108],
        GgmlType::Q2_K => &[80, 82],
        _ => &[0],
    }
}
