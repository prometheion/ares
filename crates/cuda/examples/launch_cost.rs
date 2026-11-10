//! What one kernel launch costs, on the host and on the device.
//!
//! A decode step is hundreds of launches, most of them of kernels that run for
//! a few microseconds, so the cost of a launch and of the gap between two
//! kernels on a stream decide how much of a step is left for the weights. This
//! example measures:
//!
//! - the host time of one `Kernels::add_f32` launch (the typed launch path:
//!   argument list, thread-local context check, `cuLaunchKernel`), while the
//!   stream's queue is not full;
//! - whether a launch ever waits for the device: the same call behind a long
//!   kernel must cost the same;
//! - the same with one host thread per device, both driving their device at
//!   once;
//! - the time per kernel on the device when tiny kernels follow each other on a
//!   stream (what a chain of small kernels costs, and what fusing two of them
//!   would save);
//! - the latency of launching one kernel and waiting for it;
//! - the device time of the small kernels of the per-token path (`add`,
//!   `rms_norm`, the activation quantizer) next to the floor of a tiny
//!   kernel, to see how much of each is the launch and how much work.
//!
//! ```sh
//! cargo run --release -p ares-cuda --example launch_cost
//! ```

use std::time::{Duration, Instant};

use ares_cuda::{Device, DeviceBuffer, Event, Kernels, Result, Stream, Wait, device_count};

/// Launches per batch: below the driver's queue depth (about a thousand), so
/// that a launch never waits for a free slot.
const BATCH: usize = 512;

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

fn summary(label: &str, mut times: Vec<Duration>) {
    times.sort();
    println!(
        "  {label:<58} median {:>5.2} us   p10 {:>5.2}   p90 {:>5.2}   p99 {:>5.2}",
        percentile(&times, 0.5).as_secs_f64() * 1e6,
        percentile(&times, 0.1).as_secs_f64() * 1e6,
        percentile(&times, 0.9).as_secs_f64() * 1e6,
        percentile(&times, 0.99).as_secs_f64() * 1e6,
    );
}

/// Host time of `batches * BATCH` launches of `launch`, each timed alone,
/// with a synchronisation after each batch.
fn host_cost(
    stream: &Stream,
    batches: usize,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<Vec<Duration>> {
    let mut times = Vec::with_capacity(batches * BATCH);
    for _ in 0..batches {
        for _ in 0..BATCH {
            let start = Instant::now();
            launch()?;
            times.push(start.elapsed());
        }
        stream.synchronize()?;
    }
    Ok(times)
}

fn main() -> Result<()> {
    let count = device_count()?;
    let devices: Vec<Device> =
        (0..count).map(|i| Device::open(i, Wait::Spin)).collect::<Result<_>>()?;

    for device in &devices {
        println!("{} ({})", device.info().name, device.ordinal());
        let kernels = Kernels::load(device)?;
        let stream = Stream::new(device, 0)?;
        let x = DeviceBuffer::new(device, 1 << 20)?;
        let y = DeviceBuffer::new(device, 1 << 20)?;
        // SAFETY: whole buffers on this device.
        unsafe {
            stream.fill(x.ptr(), 0, 1 << 20)?;
            stream.fill(y.ptr(), 0, 1 << 20)?;
        }
        stream.synchronize()?;

        // 1. Host cost of a launch of a tiny kernel.
        // SAFETY: both buffers hold the 64 values the launch touches.
        let launch_tiny = || unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), 64) };
        host_cost(&stream, 4, launch_tiny)?; // warm-up
        summary(
            "host cost of Kernels::add_f32 (64 values), queue not full",
            host_cost(&stream, 40, launch_tiny)?,
        );

        // 2. The same behind a long kernel: if a launch waited for the device,
        // this would cost as much as the long kernel.
        let long_n = 1u32 << 18;
        // SAFETY: both buffers hold 2^18 f32 values (1 MiB).
        let launch_long = || unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), long_n) };
        let mut behind = Vec::new();
        for _ in 0..40 {
            // a batch of long kernels is in the queue while the next launch is timed
            for _ in 0..BATCH / 2 {
                launch_long()?;
            }
            let start = Instant::now();
            launch_tiny()?;
            behind.push(start.elapsed());
            stream.synchronize()?;
        }
        summary("same, with 256 queued kernels of 2^18 values ahead of it", behind);

        // 3. Time per kernel on the device for back-to-back tiny kernels, and for
        // one kernel of the same total work.
        let (start, end) = (Event::timed(device)?, Event::timed(device)?);
        for _ in 0..3 {
            for _ in 0..BATCH {
                launch_tiny()?;
            }
        }
        stream.synchronize()?;
        let mut per_kernel = Vec::new();
        for _ in 0..20 {
            start.record(&stream)?;
            for _ in 0..BATCH {
                launch_tiny()?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            per_kernel.push(Duration::from_secs_f64(
                f64::from(end.elapsed_ms_since(&start)?) * 1e-3 / BATCH as f64,
            ));
        }
        summary("device time per tiny kernel, back to back on a stream", per_kernel);

        // 3b. The small kernels of the per-token path, back to back, device time each.
        let width = 5120u32;
        let w = DeviceBuffer::new(device, 4 * u64::from(width))?;
        let q = DeviceBuffer::new(device, Kernels::quantized_activation_bytes(width, 8))?;
        let big = DeviceBuffer::new(device, 8 * 4 * u64::from(width))?;
        // SAFETY: whole buffers on this device.
        unsafe { stream.fill(w.ptr(), 0, 4 * width as usize)? };
        stream.synchronize()?;
        let timed = |label: &str, launch: &dyn Fn() -> Result<()>| -> Result<()> {
            for _ in 0..BATCH {
                launch()?;
            }
            stream.synchronize()?;
            let mut per = Vec::new();
            for _ in 0..20 {
                start.record(&stream)?;
                for _ in 0..BATCH {
                    launch()?;
                }
                end.record(&stream)?;
                end.synchronize()?;
                per.push(Duration::from_secs_f64(
                    f64::from(end.elapsed_ms_since(&start)?) * 1e-3 / BATCH as f64,
                ));
            }
            summary(label, per);
            Ok(())
        };
        // SAFETY: the buffers hold 8 rows of 5120 values, and `q` their quantized
        // form; every launch below stays inside them.
        let add = || unsafe { kernels.add_f32(&stream, big.ptr(), big.ptr(), width) };
        timed("device time per add_f32, 5120 values", &add)?;
        // SAFETY: as above; `w` holds the 5120 weights of the norm.
        let norm1 = || unsafe {
            kernels.rms_norm_f32(&stream, big.ptr(), w.ptr(), big.ptr(), width, 1, 1e-6)
        };
        timed("device time per rms_norm_f32, 1 row of 5120", &norm1)?;
        // SAFETY: as above.
        let norm8 = || unsafe {
            kernels.rms_norm_f32(&stream, big.ptr(), w.ptr(), big.ptr(), width, 8, 1e-6)
        };
        timed("device time per rms_norm_f32, 8 rows of 5120", &norm8)?;
        // SAFETY: as above.
        let quantize1 =
            || unsafe { kernels.quantize_activations(&stream, big.ptr(), q.ptr(), width, 1) };
        timed("device time per quantize_activations, 1 row of 5120", &quantize1)?;
        // SAFETY: as above.
        let quantize8 =
            || unsafe { kernels.quantize_activations(&stream, big.ptr(), q.ptr(), width, 8) };
        timed("device time per quantize_activations, 8 rows of 5120", &quantize8)?;

        // 4. Launch and wait.
        let mut round_trip = Vec::new();
        for _ in 0..2000 {
            let t = Instant::now();
            launch_tiny()?;
            stream.synchronize()?;
            round_trip.push(t.elapsed());
        }
        summary("launch one tiny kernel and wait for it (cuStreamSynchronize)", round_trip);
        println!();
    }

    // 5. One host thread per device, both launching at once.
    if devices.len() > 1 {
        println!("one host thread per device, launching at the same time:");
        let barrier = std::sync::Barrier::new(devices.len());
        let results: Vec<Result<Vec<Duration>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = devices
                .iter()
                .map(|device| {
                    let barrier = &barrier;
                    scope.spawn(move || -> Result<Vec<Duration>> {
                        let kernels = Kernels::load(device)?;
                        let stream = Stream::new(device, 0)?;
                        let x = DeviceBuffer::new(device, 1 << 20)?;
                        let y = DeviceBuffer::new(device, 1 << 20)?;
                        // SAFETY: whole buffers on this device.
                        unsafe {
                            stream.fill(x.ptr(), 0, 1 << 20)?;
                            stream.fill(y.ptr(), 0, 1 << 20)?;
                        }
                        stream.synchronize()?;
                        // SAFETY: both buffers hold the 64 values the launch touches.
                        let launch = || unsafe { kernels.add_f32(&stream, y.ptr(), x.ptr(), 64) };
                        host_cost(&stream, 4, launch)?;
                        barrier.wait();
                        host_cost(&stream, 40, launch)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("thread")).collect()
        });
        for (device, times) in devices.iter().zip(results) {
            summary(&format!("host cost on {}", device.info().name), times?);
        }
    }
    Ok(())
}
