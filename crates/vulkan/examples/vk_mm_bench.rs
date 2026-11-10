//! Times the tensor-core prefill kernels (`Kernels::matmul`, `Kernels::matmul_rearranged`) on
//! each Vulkan device, the way `crates/cuda/examples/prefill_bench.rs` times CUDA's: a
//! 17408 x 5120 matrix (the FFN of Qwen3.8-27B) times `rows` activation rows by each path,
//! 10 launches in a row between two events, over copies of the matrix that together are
//! larger than any L2 cache. The number to watch is TOPS (2 * n_out * n_in * rows per second)
//! against what the tensor cores can do (`vk_coop_rate`: 231 for int8, 130 for f16 with f16
//! accumulation and 65 with f32 accumulation on an RTX 4070; 107 / 56 / 28 on an RTX 3060).
//! The decode kernels are timed for comparison (launches of 8 rows).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_mm_bench -- \
//!     [--rows 64,256,1024] [--types q4_K,q8_0|all] [--paths int8,f16,flush,re-f16,re-flush,decode] \
//!     [--device N]
//! ```
//!
//! `re-f16` and `re-flush` run on the rearranged form (Q4_K only), the others on the file's.

use std::time::Instant;

use ares_core::GgmlType;
use ares_vulkan::{
    Device, DeviceBuffer, Event, Kernels, MAX_BATCH, PrefillPath, Result, Stream, devices,
};

const N_IN: u32 = 5120;
const N_OUT: u32 = 17408;
const RUN: usize = 10;

/// Every weight type with a decode kernel; the tensor-core paths exist for those that have
/// them (`Kernels::has_matmul`).
fn all_types() -> Vec<GgmlType> {
    ares_core::ggml_type::ALL_TYPES
        .iter()
        .copied()
        .filter(|ty| Kernels::has_fast_matvec(*ty))
        .collect()
}

fn main() -> Result<()> {
    let mut rows: Vec<u32> = vec![64, 256, 1024];
    let mut types: Vec<GgmlType> = vec![GgmlType::Q4_K];
    let mut paths: Vec<String> =
        ["int8", "f16", "flush", "re-f16", "re-flush", "decode"].map(String::from).to_vec();
    let mut only: Option<u32> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--rows" => rows = value.split(',').filter_map(|r| r.parse().ok()).collect(),
            "--types" => {
                types = value
                    .split(',')
                    .flat_map(|n| {
                        if n == "all" {
                            return all_types();
                        }
                        vec![
                            *all_types()
                                .iter()
                                .find(|t| t.name() == n)
                                .unwrap_or_else(|| panic!("no type {n}")),
                        ]
                    })
                    .collect()
            }
            "--paths" => paths = value.split(',').map(String::from).collect(),
            "--device" => only = value.parse().ok(),
            other => panic!("unknown option {other}"),
        }
    }
    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        let loading = Instant::now();
        let kernels = Kernels::load(&device)?;
        println!(
            "Vulkan{} {} (Kernels::load {:.0} ms; f16 accumulation at full rate: {}; operand in registers: {})",
            info.ordinal,
            info.name,
            loading.elapsed().as_secs_f64() * 1e3,
            kernels.tensor_cores_full_rate(),
            kernels.cooperative_operand_in_registers()
        );
        let stream = Stream::new(&device)?;
        let (start, end) = (Event::timed(&device)?, Event::timed(&device)?);
        for &ty in &types {
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
            // Random bytes are valid blocks of every type but the floats (f16 and bf16 with
            // an exponent of all ones are infinities and NaN), which the timing does not mind.
            let copies = ((320u64 << 20) / w_bytes + 1).max(2) as usize;
            let weights = (0..copies)
                .map(|_| {
                    let buffer = DeviceBuffer::new(&device, w_bytes)?;
                    buffer.write(0, &host)?;
                    Ok(buffer)
                })
                .collect::<Result<Vec<_>>>()?;
            let re_bytes = Kernels::rearranged_bytes(ty, N_OUT, N_IN);
            let rearranged = match re_bytes {
                Some(bytes) if kernels.has_rearranged_form(ty) => Some(
                    weights
                        .iter()
                        .map(|src| {
                            let dst = DeviceBuffer::new(&device, bytes)?;
                            // SAFETY: whole matrices of the sizes above on this device.
                            unsafe {
                                kernels.rearrange(
                                    &stream,
                                    ty,
                                    src.ptr(),
                                    dst.ptr(),
                                    N_OUT,
                                    N_IN,
                                )?;
                            }
                            Ok(dst)
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                _ => None,
            };
            stream.synchronize()?;
            for &n_rows in &rows {
                let x = DeviceBuffer::new(&device, u64::from(n_rows) * u64::from(N_IN) * 4)?;
                let values: Vec<u8> = (0..u64::from(n_rows) * u64::from(N_IN))
                    .flat_map(|i| {
                        (((i.wrapping_mul(2654435761) >> 8) as u32 as f32 / (1u32 << 24) as f32)
                            - 0.5)
                            .to_le_bytes()
                    })
                    .collect();
                x.write(0, &values)?;
                let y = DeviceBuffer::new(&device, u64::from(n_rows) * u64::from(N_OUT) * 4)?;
                for path in &paths {
                    let (file_path, rearranged_form) = match path.as_str() {
                        "int8" => (Some(PrefillPath::Int8), false),
                        "f16" => (Some(PrefillPath::F16), false),
                        "flush" => (Some(PrefillPath::F16Flush), false),
                        "re-f16" => (Some(PrefillPath::F16), true),
                        "re-flush" => (Some(PrefillPath::F16Flush), true),
                        "decode" => (None, false),
                        other => panic!("unknown path {other}"),
                    };
                    if let Some(p) = file_path {
                        let supported = if rearranged_form {
                            rearranged.is_some() && kernels.has_prefill_path(p)
                        } else {
                            kernels.has_matmul(ty, p)
                        };
                        if !supported {
                            continue;
                        }
                    } else if !Kernels::has_fast_matvec(ty) {
                        continue;
                    }
                    let prepared = file_path.unwrap_or(PrefillPath::Int8);
                    let act = DeviceBuffer::new(
                        &device,
                        Kernels::prefill_activation_bytes(prepared, N_IN, n_rows),
                    )?;
                    // SAFETY: the buffers were sized for `n_rows` rows just above.
                    unsafe {
                        kernels.prepare_prefill_activations(
                            &stream,
                            prepared,
                            x.ptr(),
                            act.ptr(),
                            N_IN,
                            n_rows,
                        )?;
                    }
                    let launch = |i: usize| -> Result<()> {
                        // SAFETY: whole N_OUT x N_IN matrices; `y` holds `n_rows` rows.
                        unsafe {
                            match file_path {
                                None => {
                                    // The decode kernel takes 8 rows at a time.
                                    let mut done = 0;
                                    while done < n_rows {
                                        let n = (n_rows - done).min(MAX_BATCH);
                                        kernels.matvec(
                                            &stream,
                                            ty,
                                            weights[i % copies].ptr(),
                                            act.ptr().offset(u64::from(done) * u64::from(N_IN)),
                                            y.ptr(),
                                            N_IN,
                                            N_OUT,
                                            n,
                                        )?;
                                        done += n;
                                    }
                                    Ok(())
                                }
                                Some(p) if rearranged_form => kernels.matmul_rearranged(
                                    &stream,
                                    ty,
                                    p,
                                    rearranged.as_ref().unwrap()[i % copies].ptr(),
                                    act.ptr(),
                                    y.ptr(),
                                    N_IN,
                                    N_OUT,
                                    n_rows,
                                ),
                                Some(p) => kernels.matmul(
                                    &stream,
                                    ty,
                                    p,
                                    weights[i % copies].ptr(),
                                    act.ptr(),
                                    y.ptr(),
                                    N_IN,
                                    N_OUT,
                                    n_rows,
                                ),
                            }
                        }
                    };
                    let warm = Instant::now();
                    while warm.elapsed().as_millis() < 200 {
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
                    let tops = 2.0 * f64::from(N_OUT) * f64::from(N_IN) * f64::from(n_rows)
                        / (ms * 1e-3)
                        / 1e12;
                    println!("  {ty:>6} {path:>9} {n_rows:>5} rows: {ms:8.3} ms  {tops:6.1} TOPS");
                }
            }
        }
    }
    Ok(())
}
