//! Measures how fast two Vulkan devices exchange data, per route and message size: the
//! counterpart of the CUDA layer's `linkbench`, with the same sizes and the same chain, so the
//! numbers can be put next to `docs/dev/architecture/devices.md`'s table.
//!
//! A tensor split sums partial results across devices twice per layer, so the cost of a small
//! message (latency) and of a large one (bandwidth) decides how a model is split and which route
//! the sums take. It opens both devices, so run it only when their memory is free:
//!
//! ```sh
//! cargo run --release -p ares-vulkan --example vk_link_bench
//! ```

use std::sync::Arc;
use std::time::Instant;

use ares_vulkan::{
    Device, DeviceBuffer, HostPages, Mesh, Result, Route, SharedTimeline, Stream, Wait,
    device_count, devices,
};

const SIZES: &[usize] = &[4 << 10, 40 << 10, 256 << 10, 1 << 20, 8 << 20, 64 << 20];

fn main() -> Result<()> {
    for info in devices()? {
        println!(
            "Vulkan{}: {} ({} MiB{})",
            info.ordinal,
            info.name,
            info.total_memory >> 20,
            info.pci.map_or(String::new(), |p| format!(", PCI {p}")),
        );
    }
    if device_count()? < 2 {
        println!("one device: nothing to measure");
        return Ok(());
    }
    let a = Device::open_with(0, Wait::Spin)?;
    let b = Device::open_with(1, Wait::Spin)?;
    let pair = [a.clone(), b.clone()];

    let made = Instant::now();
    let mesh = Mesh::new(&pair, 1 << 20)?;
    println!("\nthe mesh takes the {:?} route (made in {:?})", mesh.route(), made.elapsed());
    for (route, why) in mesh.rejected() {
        println!("  not {route:?}: {why}");
    }
    for note in mesh.notes() {
        println!("  note: {note}");
    }
    println!("  push kernels: {} values", mesh.push_values());
    drop(mesh);

    println!("\n{:<44} {:>9} {:>12} {:>10}", "route", "size", "median", "rate");
    println!(
        "(no peer copy between the devices exists in Vulkan on this driver; see vk_mesh_probe)"
    );
    for (src, dst) in [(&a, &b), (&b, &a)] {
        copy_via_host_pages(src, dst)?;
    }

    // What a tensor split pays per layer: the sum of both devices' partial results, enqueued
    // back to back as a forward pass does.
    println!("\n{:<44} {:>9} {:>12}", "sum across devices", "values", "per sum");
    for route in [Route::Peer, Route::Host, Route::Staged] {
        for values in [2560u32, 5120, 10_240, 40_960, 163_840] {
            let mesh = match Mesh::with_route(&pair, values, route) {
                Ok(mesh) => mesh,
                Err(error) => {
                    if values == 2560 {
                        println!("{:<44} not available: {error}", format!("{route:?} route"));
                    }
                    break;
                }
            };
            let streams = [Stream::new(&pair[0])?, Stream::new(&pair[1])?];
            let partials = [
                DeviceBuffer::new(&pair[0], u64::from(values) * 4)?,
                DeviceBuffer::new(&pair[1], u64::from(values) * 4)?,
            ];
            let pointers = [partials[0].ptr(), partials[1].ptr()];
            let stream_refs = [&streams[0], &streams[1]];
            // The staged route waits for the devices inside every sum: fewer rounds.
            let chain: u32 = if route == Route::Staged { 16 } else { 128 };
            let mut times = Vec::new();
            let mut enqueue = Vec::new();
            for _ in 0..25 {
                // SAFETY: whole buffers on their devices, alive past the synchronisation
                // below. Zeros stay zeros under addition.
                unsafe {
                    for (stream, partial) in streams.iter().zip(&partials) {
                        stream.fill(partial.ptr(), 0, values as usize * 4)?;
                    }
                }
                streams[0].synchronize()?;
                streams[1].synchronize()?;
                let start = Instant::now();
                for _ in 0..chain {
                    // SAFETY: as above; the streams belong to the devices.
                    unsafe { mesh.sum_f32(&stream_refs, &pointers, values)? };
                }
                enqueue.push(start.elapsed().as_secs_f64() / f64::from(chain));
                streams[0].synchronize()?;
                streams[1].synchronize()?;
                times.push(start.elapsed().as_secs_f64() / f64::from(chain));
            }
            let mut times = times.split_off(5);
            times.sort_by(f64::total_cmp);
            let mut enqueue = enqueue.split_off(5);
            enqueue.sort_by(f64::total_cmp);
            println!(
                "{:<44} {:>9} {:>9.1} us   (best {:.1}, host enqueues in {:.1})",
                format!("{route:?} route (push {} values)", mesh.push_values()),
                values,
                times[times.len() / 2] * 1e6,
                times[0] * 1e6,
                enqueue[enqueue.len() / 2] * 1e6
            );
        }
    }
    Ok(())
}

/// Device to shared host pages on the source's stream, then host pages to device on the
/// destination's stream, which waits on a timeline semaphore the source's stream sets: the
/// two copies and the hand-over of the mesh's host route, without the add.
fn copy_via_host_pages(src_device: &Device, dst_device: &Device) -> Result<()> {
    let max = *SIZES.last().expect("sizes");
    let src = DeviceBuffer::new(src_device, max as u64)?;
    let dst = DeviceBuffer::new(dst_device, max as u64)?;
    let pages = Arc::new(HostPages::new(&[src_device, dst_device], max)?);
    let src_view = DeviceBuffer::import_host(src_device, &pages)?;
    let dst_view = DeviceBuffer::import_host(dst_device, &pages)?;
    let timeline = SharedTimeline::new(&[src_device.clone(), dst_device.clone()])?;
    let src_stream = Stream::new(src_device)?;
    let dst_stream = Stream::new(dst_device)?;
    src_stream.link(&dst_stream);
    // SAFETY: the whole source buffer, alive for the function.
    unsafe { src_stream.fill(src.ptr(), 0x5a, max)? };
    src_stream.synchronize()?;

    let label = format!(
        "host pages + semaphore Vulkan{} -> Vulkan{}",
        src_device.ordinal(),
        dst_device.ordinal()
    );
    let mut value = 0u64;
    for &size in SIZES {
        let reps = if size <= 1 << 20 { 200 } else { 20 };
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps + 5 {
            value += 1;
            let start = Instant::now();
            // SAFETY: `size` bytes within buffers of `max` bytes on the named devices; the
            // pages are untouched until the destination's stream has finished.
            unsafe {
                src_stream.copy(src_view.ptr(), src.ptr(), size)?;
                timeline.signal(&src_stream, value)?;
                timeline.wait(&dst_stream, value)?;
                dst_stream.copy(dst.ptr(), dst_view.ptr(), size)?;
            }
            dst_stream.synchronize()?;
            times.push(start.elapsed().as_secs_f64());
        }
        // The first runs include one-time setup.
        let mut times = times.split_off(5);
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        println!(
            "{label:<44} {:>7} K {:>9.1} us {:>7.2} GB/s",
            size >> 10,
            median * 1e6,
            size as f64 / median / 1e9
        );
    }
    Ok(())
}
