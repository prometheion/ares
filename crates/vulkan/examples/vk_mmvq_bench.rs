//! Times the quantized decode weight kernels on each Vulkan device, the way
//! `crates/cuda/examples/matvec_bench.rs` times CUDA's: the same matrix (17408 x
//! 5120, the FFN of Qwen3.8-27B), the same types in the same order, 1, 2, 4 and 8
//! rows of activations, `quantize_activations` once and then 40 launches of `matvec`
//! in a row between two timed events, over copies of the matrix that together are
//! far larger than any L2 cache, the median of 5 runs after a warm-up of at least
//! 300 ms.
//!
//! Decode is bound by how fast the weights can be read, so each line also gives the
//! share of a plain read of the same memory (`stream_read.comp`, measured first on
//! the same copies) and of the figure CUDA's benchmark quotes for the card (479 GB/s
//! for an RTX 4070, 341 for an RTX 3060: a plain reading kernel under CUDA).
//!
//! It opens the devices and takes about 350 MiB of video memory on each, so run it
//! only when that is free:
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_mmvq_bench [device [type,type,...]]
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_vulkan::{
    Args, Device, DeviceBuffer, Event, Function, Kernels, Module, Result, Stream, devices,
};

/// The largest matrix of Qwen3.8-27B: the FFN of a layer.
const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;

/// The types with a kernel: first the eleven of Qwen3.8-27B, in the order of their bytes
/// there (the ones CUDA's benchmark times), then the other seventeen a GGUF file can carry.
const FAST_TYPES: [GgmlType; 28] = [
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
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q2_K,
    GgmlType::IQ2_XXS,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
];

/// Launches per timed run.
const RUN: usize = 40;

/// What CUDA's benchmark quotes as the memory limit of the card, by name.
fn cuda_limit(name: &str) -> Option<f64> {
    if name.contains("4070") {
        Some(479.0)
    } else if name.contains("3060") {
        Some(341.0)
    } else {
        None
    }
}

fn random_bytes(len: u64) -> Vec<u8> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        })
        .collect()
}

/// Copies of a `w_bytes` matrix that together are far larger than the L2 cache.
fn copies(device: &Device, w_bytes: u64, host: &[u8]) -> Result<Vec<DeviceBuffer>> {
    let n = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
    (0..n)
        .map(|_| {
            let buffer = DeviceBuffer::new(device, w_bytes)?;
            buffer.write(0, host)?;
            Ok(buffer)
        })
        .collect()
}

/// Median of 5 runs of `RUN` launches, in ms per launch, after a warm-up.
fn time_launches(
    stream: &Stream,
    (start, end): (&Event, &Event),
    launch: &dyn Fn(usize) -> Result<()>,
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

/// GB/s of a plain read of the memory the weights of `ty` are in.
fn plain_read(
    device: &Device,
    stream: &Stream,
    events: (&Event, &Event),
    ty: GgmlType,
) -> Result<f64> {
    let w_bytes = ty.bytes_for(u64::from(N_OUT) * u64::from(N_IN)).expect("whole blocks");
    let module =
        Module::load(device, include_bytes!(concat!(env!("OUT_DIR"), "/stream_read.spv")))?;
    let read: Function = module.function("main")?;
    let weights = copies(device, w_bytes, &random_bytes(w_bytes))?;
    let sink = DeviceBuffer::new(device, 4096)?;
    // Whole 64 KiB chunks only; the rest of the matrix is under 64 KiB.
    let n_chunks = (w_bytes / 65536) as u32;
    let launch = |i: usize| -> Result<()> {
        let args = Args::new().ptr(weights[i % weights.len()].ptr()).ptr(sink.ptr()).u32(n_chunks);
        // SAFETY: the matrix holds `n_chunks` whole chunks; the sink is 4096 bytes.
        unsafe { read.launch(stream, [n_chunks, 1, 1], args) }
    };
    let ms = time_launches(stream, events, &launch)?;
    Ok((u64::from(n_chunks) * 65536) as f64 / (ms * 1e-3) / 1e9)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let only: Option<u32> = args.next().and_then(|arg| arg.parse().ok());
    let only_types: Option<Vec<String>> =
        args.next().map(|list| list.split(',').map(str::to_owned).collect());
    for info in devices()? {
        if only.is_some_and(|ordinal| ordinal != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        let started = Instant::now();
        let kernels = Kernels::load(&device)?;
        println!(
            "Vulkan{} {}: {N_OUT} x {N_IN} (Kernels::load {:.0} ms, subgroup {}, integer dot product accelerated: {})",
            info.ordinal,
            info.name,
            started.elapsed().as_secs_f64() * 1e3,
            info.subgroup_size,
            info.integer_dot_product
        );
        let stream = Stream::new(&device)?;
        let (start, end) = (Event::timed(&device)?, Event::timed(&device)?);
        let x = DeviceBuffer::new(&device, 8 * u64::from(N_IN) * 4)?;
        let y = DeviceBuffer::new(&device, 8 * u64::from(N_OUT) * 4)?;
        // Activations of a few different magnitudes.
        let values: Vec<u8> = (0..8 * N_IN)
            .flat_map(|i| {
                (((i.wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32) - 0.5)
                    .to_le_bytes()
            })
            .collect();
        x.write(0, &values)?;

        let limit = cuda_limit(&info.name);
        let ceiling = plain_read(&device, &stream, (&start, &end), GgmlType::Q4_K)?;
        println!(
            "  plain read (stream_read.comp, 50.1 MB per matrix): {ceiling:.0} GB/s{}",
            limit.map_or(String::new(), |l| format!(
                " (CUDA's quoted limit {l:.0} GB/s: {:.0} %)",
                ceiling / l * 100.0
            ))
        );
        println!(
            "  quantized kernels (median of 5 runs of {RUN} launches; share of the plain read / of CUDA's limit):"
        );
        for ty in FAST_TYPES {
            if only_types.as_ref().is_some_and(|list| !list.iter().any(|t| t == ty.name())) {
                continue;
            }
            let w_bytes = ty.bytes_for(u64::from(N_OUT) * u64::from(N_IN)).expect("whole blocks");
            let weights = copies(&device, w_bytes, &random_bytes(w_bytes))?;
            let mut line = format!("  {:<8}", ty.name());
            for n_batch in [1u32, 2, 4, 8] {
                let q =
                    DeviceBuffer::new(&device, Kernels::quantized_activation_bytes(N_IN, n_batch))?;
                // SAFETY: the buffers were sized for 8 rows of N_IN activations.
                unsafe { kernels.quantize_activations(&stream, x.ptr(), q.ptr(), N_IN, n_batch)? };
                // The float types read the f32 activations themselves.
                let act = if Kernels::needs_quantized_activations(ty) { q.ptr() } else { x.ptr() };
                let launch = |i: usize| -> Result<()> {
                    // SAFETY: whole N_OUT x N_IN matrices of `ty`; `y` holds 8 rows.
                    unsafe {
                        kernels.matvec(
                            &stream,
                            ty,
                            weights[i % weights.len()].ptr(),
                            act,
                            y.ptr(),
                            N_IN,
                            N_OUT,
                            n_batch,
                        )
                    }
                };
                let ms = time_launches(&stream, (&start, &end), &launch)?;
                let gbs = w_bytes as f64 / (ms * 1e-3) / 1e9;
                line += &format!(
                    " | {n_batch} row(s) {ms:>7.3} ms {gbs:>6.1} GB/s {:>3.0}% {}",
                    gbs / ceiling * 100.0,
                    limit.map_or("   ".to_string(), |l| format!("{:>3.0}%", gbs / l * 100.0))
                );
            }
            println!("{line}");
        }
    }
    Ok(())
}
