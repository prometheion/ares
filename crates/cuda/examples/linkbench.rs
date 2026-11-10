//! Measures how fast two devices exchange data, per route and message size.
//!
//! A tensor split sums partial results across devices twice per layer, so the
//! cost of a small message (latency) and of a large one (bandwidth) decides
//! how a model is split and which route the sums take. This prints the
//! numbers the split planner needs for the two devices of this machine.
//!
//! It opens both devices, so run it only when their memory is free:
//!
//! ```sh
//! cargo run --release -p ares-cuda --example linkbench
//! ```

use std::time::Instant;

use ares_cuda::{
    Device, DeviceBuffer, Event, PinnedBuffer, Result, Stream, Wait, devices, peer_access,
};

const SIZES: &[usize] = &[4 << 10, 40 << 10, 256 << 10, 1 << 20, 8 << 20, 64 << 20];

fn main() -> Result<()> {
    let infos = devices()?;
    for info in &infos {
        println!(
            "CUDA{}: {} (sm_{}{}, {} MiB, PCI {})",
            info.ordinal,
            info.name,
            info.compute_capability.0,
            info.compute_capability.1,
            info.total_memory >> 20,
            info.pci
        );
    }
    if infos.len() < 2 {
        println!("one device: nothing to measure");
        return Ok(());
    }
    let a = Device::open(0, Wait::Spin)?;
    let b = Device::open(1, Wait::Spin)?;
    let (ab, rank_ab) = peer_access(0, 1)?;
    let (ba, rank_ba) = peer_access(1, 0)?;
    println!("peer access 0->1: {ab} (rank {rank_ab}), 1->0: {ba} (rank {rank_ba})\n");

    println!("{:<34} {:>9} {:>12} {:>10}", "route", "size", "median", "rate");
    for (src, dst) in [(&a, &b), (&b, &a)] {
        bench(src, dst, Route::HostStaged)?;
        bench(src, dst, Route::DriverPeer)?;
    }
    if ab && ba {
        a.enable_peer_access(&b)?;
        b.enable_peer_access(&a)?;
        println!();
        for (src, dst) in [(&a, &b), (&b, &a)] {
            bench(src, dst, Route::DriverPeer)?;
        }
    }

    // What a tensor split pays per layer: the sum of both devices' partial
    // results, enqueued back to back as a forward pass does.
    println!("\n{:<34} {:>9} {:>12}", "sum across devices", "values", "per sum");
    let devices = [a, b];
    let mut routes = vec![ares_cuda::Route::Host];
    if ab && ba {
        routes.push(ares_cuda::Route::Peer);
    }
    for route in routes {
        for values in [2560u32, 10_240, 40_960, 163_840] {
            let mesh = ares_cuda::Mesh::with_route(&devices, values, route)?;
            let streams = [Stream::new(&devices[0], 0)?, Stream::new(&devices[1], 0)?];
            let partials = [
                DeviceBuffer::new(&devices[0], u64::from(values) * 4)?,
                DeviceBuffer::new(&devices[1], u64::from(values) * 4)?,
            ];
            let pointers = [partials[0].ptr(), partials[1].ptr()];
            let stream_refs = [&streams[0], &streams[1]];
            const CHAIN: u32 = 128;
            let mut times = Vec::new();
            for _ in 0..25 {
                // SAFETY: whole buffers on their devices, alive past the
                // synchronisation below. Zeros stay zeros under addition.
                unsafe {
                    for (stream, partial) in streams.iter().zip(&partials) {
                        stream.fill(partial.ptr(), 0, values as usize * 4)?;
                    }
                }
                streams[0].synchronize()?;
                streams[1].synchronize()?;
                let start = Instant::now();
                for _ in 0..CHAIN {
                    // SAFETY: as above; the streams belong to the devices.
                    unsafe { mesh.sum_f32(&stream_refs, &pointers, values)? };
                }
                streams[0].synchronize()?;
                streams[1].synchronize()?;
                times.push(start.elapsed().as_secs_f64() / f64::from(CHAIN));
            }
            let mut times = times.split_off(5);
            times.sort_by(f64::total_cmp);
            println!(
                "{:<34} {:>9} {:>9.1} us",
                format!("{route:?} route"),
                values,
                times[times.len() / 2] * 1e6
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Route {
    /// Device to pinned host memory on the source's stream, then host to
    /// device on the destination's stream, which waits for the first copy.
    HostStaged,
    /// One peer copy; the driver goes direct with peer access enabled and
    /// through the host without.
    DriverPeer,
}

fn bench(src_device: &Device, dst_device: &Device, route: Route) -> Result<()> {
    let max = *SIZES.last().expect("sizes");
    let src = DeviceBuffer::new(src_device, max as u64)?;
    let dst = DeviceBuffer::new(dst_device, max as u64)?;
    let mut staging = PinnedBuffer::new(src_device, max)?;
    let src_stream = Stream::new(src_device, 0)?;
    let dst_stream = Stream::new(dst_device, 0)?;
    let staged = Event::new(src_device)?;
    // SAFETY: the whole source buffer, alive for the function.
    unsafe { src_stream.fill(src.ptr(), 0x5a, max)? };
    src_stream.synchronize()?;

    let label = format!(
        "{} CUDA{} -> CUDA{}",
        if route == Route::HostStaged { "host-staged" } else { "peer copy  " },
        src_device.ordinal(),
        dst_device.ordinal()
    );
    for &size in SIZES {
        let reps = if size <= 1 << 20 { 200 } else { 20 };
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps + 5 {
            let start = Instant::now();
            // SAFETY: `size` bytes within buffers of `max` bytes on the named
            // devices; the staging buffer is untouched until both streams
            // are synchronised.
            unsafe {
                match route {
                    Route::HostStaged => {
                        src_stream.copy_to_host(staging.as_mut_ptr(), src.ptr(), size)?;
                        staged.record(&src_stream)?;
                        dst_stream.wait_event(&staged)?;
                        dst_stream.copy_from_host(dst.ptr(), staging.as_ptr(), size)?;
                    }
                    Route::DriverPeer => {
                        dst_stream.copy_peer(dst_device, dst.ptr(), src_device, src.ptr(), size)?;
                    }
                }
            }
            dst_stream.synchronize()?;
            times.push(start.elapsed().as_secs_f64());
        }
        // The first runs include one-time setup.
        let mut times = times.split_off(5);
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        println!(
            "{label:<34} {:>7} K {:>9.1} us {:>7.2} GB/s",
            size >> 10,
            median * 1e6,
            size as f64 / median / 1e9
        );
    }
    Ok(())
}
