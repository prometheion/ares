//! Times the weight kernels on each device.
//!
//! Decode is bound by how fast the weights can be read, so the number to
//! watch is GB/s of weight bytes against the device's memory bandwidth (about
//! 500 GB/s on an RTX 4070 and 360 on an RTX 3060 by the data sheet, 479 and
//! 341 measured with a plain streaming read).
//!
//! First the exact f32 Q4_K kernel, the reference, one launch at a time. Then
//! the fast integer kernels of every type, `quantize_activations` once and
//! `matvec` after it, the way a step uses them: 40 launches in a row between
//! two events, over several copies of the matrix that together are larger than
//! any L2 cache, so that every launch reads its weights from memory. For a type
//! with a device layout a second line times the same products over the rearranged
//! matrix (`Kernels::product`, the kernel of the device's measured table), which
//! is what a model that holds the matrix in that layout runs.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example matvec_bench
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_cuda::{
    ActivationForm, Device, DeviceBuffer, Event, Kernels, Precision, Result, Stream, Wait,
    WeightLayout, devices,
};

/// The largest matrix of Qwen3.8-27B: the FFN of a layer.
const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;

fn main() -> Result<()> {
    // Optional arguments: a device ordinal, and a list of type names ("q4_K,f16"); with
    // the list the f32 baseline is skipped.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let only_device: Option<u32> = args.first().and_then(|a| a.parse().ok());
    let only_types: Option<Vec<&str>> = args.get(1).map(|a| a.split(',').collect());
    for info in devices()? {
        if only_device.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let kernels = Kernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        'baseline: {
            if only_types.is_some() {
                break 'baseline;
            }
            let w_bytes = u64::from(N_OUT) * u64::from(N_IN / 256) * 144;
            let w = DeviceBuffer::new(&device, w_bytes)?;
            let x = DeviceBuffer::new(&device, 8 * u64::from(N_IN) * 4)?;
            let scratch = DeviceBuffer::new(&device, 8 * u64::from(N_IN / 32) * 4)?;
            let y = DeviceBuffer::new(&device, 8 * u64::from(N_OUT) * 4)?;
            // Any bytes are a valid Q4_K matrix for timing; the scales stay finite.
            // SAFETY: whole buffers on this device.
            unsafe {
                stream.fill(w.ptr(), 0x11, w_bytes as usize)?;
                stream.fill(x.ptr(), 0, 8 * N_IN as usize * 4)?;
            }
            stream.synchronize()?;

            println!(
                "CUDA{} {}: {N_OUT} x {N_IN} Q4_K, {:.1} MiB",
                info.ordinal,
                info.name,
                w_bytes as f64 / (1 << 20) as f64
            );
            for n_batch in [1u32, 2, 4, 8] {
                let mut times = Vec::new();
                for _ in 0..60 {
                    let start = Instant::now();
                    // SAFETY: the buffers were sized for 8 activation rows above.
                    unsafe {
                        kernels.matvec_q4_k(
                            &stream,
                            w.ptr(),
                            x.ptr(),
                            scratch.ptr(),
                            y.ptr(),
                            N_IN,
                            N_OUT,
                            n_batch,
                        )?;
                    }
                    stream.synchronize()?;
                    times.push(start.elapsed().as_secs_f64());
                }
                let mut times = times.split_off(10);
                times.sort_by(f64::total_cmp);
                let median = times[times.len() / 2];
                println!(
                    "  {n_batch} row(s): {:>7.3} ms  {:>6.1} GB/s of weights",
                    median * 1e3,
                    w_bytes as f64 / median / 1e9
                );
            }
        }

        println!("fast kernels, {N_OUT} x {N_IN} (median of 5 runs of {RUN} launches):");
        for ty in FAST_TYPES {
            if only_types.as_ref().is_some_and(|names| !names.contains(&ty.name())) {
                continue;
            }
            time_fast(&device, &kernels, &stream, ty)?;
        }
    }
    Ok(())
}

/// The types with a fast kernel: first the eleven of Qwen3.8-27B in the order of their
/// bytes there, then the others.
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
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q2_K,
    GgmlType::IQ1_S,
    GgmlType::IQ1_M,
    GgmlType::IQ2_XXS,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
];

/// Launches per timed run.
const RUN: usize = 40;

fn time_fast(device: &Device, kernels: &Kernels, stream: &Stream, ty: GgmlType) -> Result<()> {
    let w_bytes = ty.bytes_for(u64::from(N_OUT) * u64::from(N_IN)).expect("whole blocks");
    // Enough copies that the matrices together are far larger than the L2 cache.
    let copies = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
    // Random bytes: any block is a valid block of these types, and uniform data
    // would draw less power than weights do.
    let mut state = 0x9e3779b97f4a7c15u64;
    let host: Vec<u8> = (0..w_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        })
        .collect();
    let weights = (0..copies)
        .map(|_| {
            let buffer = DeviceBuffer::new(device, w_bytes)?;
            buffer.write(0, &host)?;
            Ok(buffer)
        })
        .collect::<Result<Vec<_>>>()?;
    let x = DeviceBuffer::new(device, 8 * u64::from(N_IN) * 4)?;
    let y = DeviceBuffer::new(device, 8 * u64::from(N_OUT) * 4)?;
    // SAFETY: whole buffers on this device.
    unsafe { stream.fill(x.ptr(), 0x3c, 8 * N_IN as usize * 4)? };
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);

    let mut line = format!("  {:<8}", ty.name());
    for n_batch in [1u32, 2, 4, 8] {
        let q = DeviceBuffer::new(device, Kernels::quantized_activation_bytes(N_IN, n_batch))?;
        let launch = |i: usize| -> Result<()> {
            // SAFETY: the buffers were sized for 8 rows of N_IN activations, the
            // matrices are whole N_OUT x N_IN matrices of `ty`.
            unsafe {
                kernels.matvec(
                    stream,
                    ty,
                    weights[i % copies].ptr(),
                    q.ptr(),
                    y.ptr(),
                    N_IN,
                    N_OUT,
                    n_batch,
                )
            }
        };
        // SAFETY: as above.
        unsafe { kernels.quantize_activations(stream, x.ptr(), q.ptr(), N_IN, n_batch)? };
        let ms = median_ms(stream, &start, &end, &launch)?;
        line += &format!(
            " | {n_batch} row(s) {:>7.3} ms {:>6.1} GB/s",
            ms,
            w_bytes as f64 / (ms * 1e-3) / 1e9
        );
    }
    println!("{line}");

    if kernels.layout_for(ty, N_IN) != WeightLayout::Device {
        return Ok(());
    }
    // The same matrices in the device layout, in place of the file's.
    let table = kernels.measure_layout(ty)?;
    let bytes =
        Kernels::layout_bytes(ty, WeightLayout::Device, N_OUT, N_IN).expect("has the layout");
    let rearranged = weights
        .iter()
        .map(|w| {
            let r = DeviceBuffer::new(device, bytes)?;
            // SAFETY: whole matrices of `ty` and the size the function names.
            unsafe { kernels.rearrange(stream, ty, w.ptr(), r.ptr(), N_OUT, N_IN)? };
            Ok(r)
        })
        .collect::<Result<Vec<_>>>()?;
    stream.synchronize()?;
    drop(weights);
    let mut line = format!("  {:<8}", format!("  {}", table.rows[0].fast.name));
    for n_batch in [1u32, 2, 4, 8] {
        let form = kernels.activation_form(ty, WeightLayout::Device, Precision::Fast, n_batch)?;
        debug_assert_eq!(form, ActivationForm::F16);
        let act = DeviceBuffer::new(device, Kernels::activation_bytes(form, N_IN, n_batch))?;
        // SAFETY: the buffers were sized for these rows.
        unsafe { kernels.prepare_activations(stream, form, x.ptr(), act.ptr(), N_IN, n_batch)? };
        let launch = |i: usize| -> Result<()> {
            // SAFETY: as above; the matrices were rearranged just before.
            unsafe {
                kernels.product(
                    stream,
                    ty,
                    WeightLayout::Device,
                    Precision::Fast,
                    rearranged[i % copies].ptr(),
                    act.ptr(),
                    y.ptr(),
                    N_IN,
                    N_OUT,
                    n_batch,
                )
            }
        };
        let ms = median_ms(stream, &start, &end, &launch)?;
        line += &format!(
            " | {n_batch} row(s) {:>7.3} ms {:>6.1} GB/s",
            ms,
            bytes as f64 / (ms * 1e-3) / 1e9
        );
    }
    println!("{line}");
    Ok(())
}

/// The median over 5 runs of `RUN` launches of the time of one launch, in ms, after
/// a warm-up of at least 300 ms (module load and the clocks ramping up).
fn median_ms(
    stream: &Stream,
    start: &Event,
    end: &Event,
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
