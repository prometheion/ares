//! Times the tensor-core prefill kernel (`Kernels::matmul`) on each Vulkan device, the way
//! `crates/cuda/examples/prefill_bench.rs` times CUDA's: a 17408 x 5120 matrix (the FFN of
//! Qwen3.8-27B) times `rows` activation rows, 10 launches in a row between two events, over
//! copies of the matrix that together are larger than any L2 cache; the number to watch is
//! TOPS (2 * n_out * n_in * rows per second) against what the tensor cores can do (234 on an
//! RTX 4070 and 102 on an RTX 3060 for int8, measured with `vk_coop_rate` and CUDA's probe).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_mm_bench [device [rows,rows,...]]
//! ```

use std::time::Instant;

use ares_core::GgmlType;
use ares_vulkan::{Device, DeviceBuffer, Event, Kernels, Result, Stream, devices};

const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;
const RUN: usize = 10;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let only: Option<u32> = args.next().and_then(|a| a.parse().ok());
    let rows: Vec<u32> = args
        .next()
        .map(|list| list.split(',').filter_map(|r| r.parse().ok()).collect())
        .unwrap_or_else(|| vec![32, 64, 128, 256, 512, 1024, 2048]);
    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        let kernels = Kernels::load(&device)?;
        println!("Vulkan{} {}", info.ordinal, info.name);
        if !kernels.has_matmul(GgmlType::Q4_K) {
            println!("  no tensor-core kernel on this device");
            continue;
        }
        let stream = Stream::new(&device)?;
        let (start, end) = (Event::timed(&device)?, Event::timed(&device)?);
        let ty = GgmlType::Q4_K;
        let w_bytes = ty.bytes_for(u64::from(N_OUT) * u64::from(N_IN)).unwrap();
        let mut state = 0x9e3779b97f4a7c15u64;
        let host: Vec<u8> = (0..w_bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect();
        let copies = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
        let weights = (0..copies)
            .map(|_| {
                let buffer = DeviceBuffer::new(&device, w_bytes)?;
                buffer.write(0, &host)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>>>()?;
        for &n_rows in &rows {
            let x = DeviceBuffer::new(&device, u64::from(n_rows) * u64::from(N_IN) * 4)?;
            let values: Vec<u8> = (0..u64::from(n_rows) * u64::from(N_IN))
                .flat_map(|i| {
                    (((i.wrapping_mul(2654435761) >> 8) as u32 as f32 / (1u32 << 24) as f32) - 0.5)
                        .to_le_bytes()
                })
                .collect();
            x.write(0, &values)?;
            let q = DeviceBuffer::new(&device, Kernels::quantized_activation_bytes(N_IN, n_rows))?;
            let y = DeviceBuffer::new(&device, u64::from(n_rows) * u64::from(N_OUT) * 4)?;
            // SAFETY: the buffers were sized for `n_rows` rows just above.
            unsafe { kernels.quantize_activation_rows(&stream, x.ptr(), q.ptr(), N_IN, n_rows)? };
            let launch = |i: usize| -> Result<()> {
                // SAFETY: whole N_OUT x N_IN matrices; `y` holds `n_rows` rows.
                unsafe {
                    kernels.matmul(
                        &stream,
                        ty,
                        weights[i % copies].ptr(),
                        q.ptr(),
                        y.ptr(),
                        N_IN,
                        N_OUT,
                        n_rows,
                    )
                }
            };
            let warm = Instant::now();
            while warm.elapsed().as_millis() < 300 {
                for i in 0..RUN {
                    launch(i)?;
                }
                stream.synchronize()?;
            }
            let mut times = Vec::new();
            for _ in 0..5 {
                start.record(&stream)?;
                for i in 0..RUN {
                    launch(i)?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                times.push(f64::from(end.elapsed_ms_since(&start)?) / RUN as f64);
            }
            times.sort_by(f64::total_cmp);
            let ms = times[2];
            let tops =
                2.0 * f64::from(N_OUT) * f64::from(N_IN) * f64::from(n_rows) / (ms * 1e-3) / 1e12;
            println!("  {ty} {n_rows:>5} rows: {ms:8.3} ms  {tops:6.1} TOPS");
        }
    }
    Ok(())
}
