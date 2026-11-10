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
//! any L2 cache, so that every launch reads its weights from memory.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example matvec_bench
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_cuda::{Device, DeviceBuffer, Event, Kernels, Result, Stream, Wait, devices};

/// The largest matrix of Qwen3.8-27B: the FFN of a layer.
const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;

fn main() -> Result<()> {
    for info in devices()? {
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let kernels = Kernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
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

        println!("fast kernels, {N_OUT} x {N_IN} (median of 5 runs of {RUN} launches):");
        for ty in FAST_TYPES {
            time_fast(&device, &kernels, &stream, ty)?;
        }
    }
    Ok(())
}

/// The types with a fast kernel, in the order of their bytes in Qwen3.8-27B.
const FAST_TYPES: [GgmlType; 11] = [
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
        // Warm up: module load and the clocks ramping up, for at least 300 ms.
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
            times.push(f64::from(end.elapsed_ms_since(&start)?) / RUN as f64);
        }
        times.sort_by(f64::total_cmp);
        let ms = times[times.len() / 2];
        line += &format!(
            " | {n_batch} row(s) {:>7.3} ms {:>6.1} GB/s",
            ms,
            w_bytes as f64 / (ms * 1e-3) / 1e9
        );
    }
    println!("{line}");
    Ok(())
}
