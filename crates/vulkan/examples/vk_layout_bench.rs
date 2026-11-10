//! Times the kernels over the device layouts (`Kernels::product_with`) against the kernels
//! of the file's layout on the same matrix: the decode kernels (`Kernels::matvec`, in
//! launches of 8 rows) and the tile kernels (`Kernels::matmul`).
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_layout_bench -- [--device N] \
//!     [--types q4_K,iq4_xs] [--rows 1,8,64,1024] [--shape 17408x5120] [--kernels r8f,r64h]
//! ```
//!
//! One line per type and number of rows: milliseconds per product for every kernel of the
//! family that was asked for (TOPS = 2 * n_out * n_in * rows per second, in brackets),
//! then `dec` (decode kernels; above 8 rows with the quantization of each launch) and `tile` (the f16 tile kernel `Precision::Fast` takes).
//! Each time is the fastest of five runs of four launches, each launch over another copy of
//! the matrix (the copies are together larger than any L2 cache, so a launch reads its
//! weights from memory as a step of a model does). The matrix is constant bytes with finite
//! scales; no kernel's time depends on the values.

use ares_core::GgmlType;
use ares_vulkan::{
    ActivationForm, Device, DeviceBuffer, Event, Kernels, Layout, MAX_BATCH, Precision, Result,
    Stream, devices,
};

const TYPES: [GgmlType; 12] = [
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
];

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    let only: Option<u32> = arg("--device").and_then(|a| a.parse().ok());
    let types: Vec<GgmlType> = match arg("--types") {
        Some(list) => {
            list.split(',').filter_map(|n| TYPES.into_iter().find(|t| t.name() == n)).collect()
        }
        None => vec![GgmlType::Q4_K],
    };
    let rows: Vec<u32> = arg("--rows")
        .unwrap_or_else(|| "1,8,16,32,64,256,1024".into())
        .split(',')
        .filter_map(|r| r.parse().ok())
        .collect();
    let (n_out, n_in) = arg("--shape")
        .and_then(|s| {
            let (o, i) = s.split_once('x')?;
            Some((o.parse::<u32>().ok()?, i.parse::<u32>().ok()?))
        })
        .unwrap_or((17408, 5120));
    let names = Kernels::layout_kernel_names();
    let wanted: Vec<&str> = match arg("--kernels") {
        Some(list) => names.into_iter().filter(|n| list.split(',').any(|w| w == *n)).collect(),
        None => names.to_vec(),
    };
    let max_rows = rows.iter().copied().max().unwrap_or(1);

    for info in devices()? {
        if only.is_some_and(|d| d != info.ordinal) {
            continue;
        }
        let device = Device::open(info.ordinal)?;
        let kernels = Kernels::load(&device)?;
        println!(
            "Vulkan{} {}: layout kernels {}, f32 accumulation at full rate {}",
            info.ordinal,
            info.name,
            kernels.has_layout_kernels(),
            kernels.tensor_cores_full_rate()
        );
        if !kernels.has_layout_kernels() {
            continue;
        }
        kernels.prepare(&types)?;
        let stream = Stream::new(&device)?;
        let (start, end) = (Event::timed(&device)?, Event::timed(&device)?);
        let act_bytes = [ActivationForm::F16, ActivationForm::Int8]
            .map(|f| Kernels::activation_bytes(f, n_in, max_rows))
            .into_iter()
            .max()
            .unwrap_or(0);
        let x = DeviceBuffer::new(&device, u64::from(n_in) * u64::from(max_rows) * 4)?;
        let act = DeviceBuffer::new(&device, act_bytes)?;
        let act8 = DeviceBuffer::new(&device, act_bytes)?;
        let y = DeviceBuffer::new(&device, u64::from(n_out) * u64::from(max_rows) * 4)?;
        // Activations near 1 (the f32 0x3c3c3c3c is 0.0115).
        // SAFETY: a whole buffer of this device.
        unsafe { stream.fill(x.ptr(), 0x3c, x.len() as usize)? };
        for &ty in &types {
            let file_bytes = Kernels::layout_bytes(ty, Layout::File, n_out, n_in).expect("blocks");
            let dev_bytes = Kernels::layout_bytes(ty, Layout::Device, n_out, n_in).expect("layout");
            // Copies of the matrix that together are larger than any L2 cache: launch i reads
            // copy i, so no launch finds its weights cached by the one before.
            let copies = (((256u64 << 20) / dev_bytes) as usize + 1).max(2);
            let mut file = Vec::new();
            let mut dev = Vec::new();
            for _ in 0..copies {
                let f = DeviceBuffer::new(&device, file_bytes.next_multiple_of(16))?;
                let d = DeviceBuffer::new(&device, dev_bytes)?;
                // SAFETY: whole buffers; 0x11 bytes are small codes and finite scales in
                // every type, and the rearrangement keeps them a valid matrix.
                unsafe {
                    stream.fill(f.ptr(), 0x11, f.len() as usize)?;
                    start.record(&stream)?;
                    kernels.rearrange(&stream, ty, f.ptr(), d.ptr(), n_out, n_in)?;
                    end.record(&stream)?;
                }
                end.synchronize()?;
                file.push(f);
                dev.push(d);
            }
            println!(
                "{} {n_out}x{n_in}: rearranged in {:.3} ms ({:.1} MiB, {copies} copies)",
                ty.name(),
                end.elapsed_ms_since(&start)?,
                dev_bytes as f64 / (1 << 20) as f64
            );
            for &n in &rows {
                let best = |launch: &dyn Fn(usize) -> Result<()>| -> Result<f32> {
                    let mut best = f32::INFINITY;
                    for round in 0..6 {
                        start.record(&stream)?;
                        for i in 0..4 {
                            launch((round * 4 + i) % copies)?;
                        }
                        end.record(&stream)?;
                        end.synchronize()?;
                        if round > 0 {
                            best = best.min(end.elapsed_ms_since(&start)? / 4.0);
                        }
                    }
                    Ok(best)
                };
                let tops = |ms: f32| {
                    2.0 * f64::from(n_out) * f64::from(n_in) * f64::from(n)
                        / (f64::from(ms) * 1e-3)
                        / 1e12
                };
                let mut line = format!("  {:>7} rows {n:>5}:", ty.name());
                // SAFETY: the buffers hold the matrix in both layouts, max_rows rows of
                // activations in either prepared form and their products.
                unsafe {
                    kernels.prepare_activations(
                        &stream,
                        ActivationForm::F16,
                        x.ptr(),
                        act.ptr(),
                        n_in,
                        n,
                    )?;
                }
                for kernel in &wanted {
                    // A kernel far from its bucket is skipped: it would run for seconds.
                    let batch: u32 = kernel[1..kernel.len() - 1].parse().unwrap_or(64);
                    if n > batch * 4 && batch < 64 {
                        continue;
                    }
                    // SAFETY: as above, over copy `i` of the matrix.
                    let ms = best(&|i| unsafe {
                        kernels.product_with(
                            &stream,
                            ty,
                            kernel,
                            dev[i].ptr(),
                            act.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            n,
                        )
                    })?;
                    line.push_str(&format!(" {kernel} {ms:.4} ({:.1})", tops(ms)));
                }
                // The decode kernels on the file's layout, in launches of 8 rows.
                if n <= 64 {
                    // Up to 8 rows are quantized once, outside the time, like the f16 rows
                    // of the other kernels; more rows are quantized launch by launch.
                    if n <= MAX_BATCH {
                        // SAFETY: as above.
                        unsafe {
                            kernels.quantize_activations(&stream, x.ptr(), act8.ptr(), n_in, n)?;
                        }
                    }
                    let ms = best(&|i| {
                        let mut done = 0;
                        while done < n {
                            let b = MAX_BATCH.min(n - done);
                            // SAFETY: as above; the quantized rows of one launch.
                            unsafe {
                                if n > MAX_BATCH {
                                    kernels.quantize_activations(
                                        &stream,
                                        x.ptr().offset(u64::from(done) * u64::from(n_in) * 4),
                                        act8.ptr(),
                                        n_in,
                                        b,
                                    )?;
                                }
                                kernels.matvec(
                                    &stream,
                                    ty,
                                    file[i].ptr(),
                                    act8.ptr(),
                                    y.ptr().offset(u64::from(done) * u64::from(n_out) * 4),
                                    n_in,
                                    n_out,
                                    b,
                                )?;
                            }
                            done += b;
                        }
                        Ok(())
                    })?;
                    line.push_str(&format!(" | dec {ms:.4} ({:.1})", tops(ms)));
                }
                if n > MAX_BATCH {
                    let path = kernels.prefill_path(Precision::Fast)?;
                    // SAFETY: as above, over copy `i` of the matrix.
                    let ms = best(&|i| unsafe {
                        kernels.matmul(
                            &stream,
                            ty,
                            path,
                            file[i].ptr(),
                            act.ptr(),
                            y.ptr(),
                            n_in,
                            n_out,
                            n,
                        )
                    })?;
                    line.push_str(&format!(" | tile {ms:.4} ({:.1})", tops(ms)));
                }
                println!("{line}");
            }
        }
    }
    Ok(())
}
