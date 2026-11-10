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
//! L2 cache. The decode kernels are timed for comparison at up to 64 rows (in
//! launches of 8). The fast path of the device is marked.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example prefill_bench -- \
//!     [--rows 64,256,1024] [--types q4_K,iq4_xs] [--paths int8,f16,decode]
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_cuda::{
    Device, DeviceBuffer, Event, Kernels, MAX_BATCH, Precision, PrefillPath, Result, Stream, Wait,
    devices,
};

/// The largest matrix of Qwen3.8-27B: the FFN of a layer.
const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;

/// The types with a prefill kernel, in the order of their bytes in Qwen3.8-27B.
const TYPES: [GgmlType; 11] = [
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum Which {
    Int8,
    F16,
    Decode,
}

fn main() -> Result<()> {
    let mut rows = vec![64u32, 256, 1024];
    let mut types: Vec<GgmlType> = TYPES.to_vec();
    let mut paths = vec![Which::Int8, Which::F16, Which::Decode];
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
            "--paths" => {
                paths = value
                    .split(',')
                    .map(|p| match p {
                        "int8" => Which::Int8,
                        "f16" => Which::F16,
                        "decode" => Which::Decode,
                        _ => panic!("--paths: int8, f16 or decode"),
                    })
                    .collect();
            }
            other => panic!("unknown option {other}"),
        }
    }

    for info in devices()? {
        let device = Device::open(info.ordinal, Wait::Spin)?;
        let kernels = Kernels::load(&device)?;
        let stream = Stream::new(&device, 0)?;
        let fast = kernels.prefill_path(Precision::Fast)?;
        println!(
            "CUDA{} {}: {N_OUT} x {N_IN}; f16 mma with f32 accumulation at full rate: {} (fast path {fast:?})",
            info.ordinal,
            info.name,
            kernels.tensor_cores_full_rate()
        );
        for &ty in &types {
            time_type(&device, &kernels, &stream, ty, &rows, &paths, fast)?;
        }
    }
    Ok(())
}

/// Launches per timed run.
const RUN: usize = 10;

fn time_type(
    device: &Device,
    kernels: &Kernels,
    stream: &Stream,
    ty: GgmlType,
    rows: &[u32],
    paths: &[Which],
    fast: PrefillPath,
) -> Result<()> {
    let w_bytes = ty.bytes_for(u64::from(N_OUT) * u64::from(N_IN)).expect("whole blocks");
    let copies = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
    // Random bytes: any block is a valid block of these types, and uniform data
    // would draw less power than weights do. The f16 scales stay finite.
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut host: Vec<u8> = (0..w_bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        })
        .collect();
    for block in host.chunks_exact_mut(ty.block_bytes() as usize) {
        for scale_at in scale_offsets(ty) {
            block[*scale_at..*scale_at + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        }
    }
    let weights = (0..copies)
        .map(|_| {
            let buffer = DeviceBuffer::new(device, w_bytes)?;
            buffer.write(0, &host)?;
            Ok(buffer)
        })
        .collect::<Result<Vec<_>>>()?;
    let max_rows = *rows.iter().max().unwrap_or(&1);
    let x = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(N_IN) * 4)?;
    let y = DeviceBuffer::new(device, u64::from(max_rows) * u64::from(N_OUT) * 4)?;
    // SAFETY: whole buffers on this device.
    unsafe { stream.fill(x.ptr(), 0x3c, (u64::from(max_rows) * u64::from(N_IN) * 4) as usize)? };
    let (start, end) = (Event::timed(device)?, Event::timed(device)?);

    for &n in rows {
        let mut line = format!("  {:<8} {n:>5} rows", ty.name());
        for &which in paths {
            if which == Which::Decode && n > 64 {
                continue;
            }
            let ms = match which {
                Which::Int8 | Which::F16 => {
                    let path =
                        if which == Which::Int8 { PrefillPath::Int8 } else { PrefillPath::F16 };
                    let act = DeviceBuffer::new(
                        device,
                        Kernels::prefill_activation_bytes(path, N_IN, n),
                    )?;
                    // SAFETY: the buffers were sized for these shapes just above.
                    unsafe {
                        kernels.prepare_prefill_activations(
                            stream,
                            path,
                            x.ptr(),
                            act.ptr(),
                            N_IN,
                            n,
                        )?
                    };
                    time(device, stream, &start, &end, |i| {
                        // SAFETY: as above; the matrices are whole N_OUT x N_IN matrices of `ty`.
                        unsafe {
                            kernels.matmul(
                                stream,
                                ty,
                                path,
                                weights[i % copies].ptr(),
                                act.ptr(),
                                y.ptr(),
                                N_IN,
                                N_OUT,
                                n,
                            )
                        }
                    })?
                }
                Which::Decode => {
                    let q = DeviceBuffer::new(
                        device,
                        Kernels::quantized_activation_bytes(N_IN, MAX_BATCH),
                    )?;
                    // SAFETY: as above.
                    unsafe {
                        kernels.quantize_activations(stream, x.ptr(), q.ptr(), N_IN, MAX_BATCH)?
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
                                    N_IN,
                                    N_OUT,
                                    chunk,
                                )?
                            };
                            left -= chunk;
                        }
                        Ok(())
                    })?
                }
            };
            let tops = 2.0 * f64::from(N_OUT) * f64::from(N_IN) * f64::from(n) / (ms * 1e-3) / 1e12;
            let name = match which {
                Which::Int8 => "int8",
                Which::F16 => "f16",
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
        GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q6_K => &[208],
        GgmlType::Q3_K => &[108],
        _ => &[0],
    }
}
