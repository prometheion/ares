//! Measures what the link from page-locked host memory to one device gives for copies the
//! size of an expert's matrices, the way the expert store uploads them: copies of `size`
//! bytes enqueued back to back on one stream (or spread over two), a total of `--mib` per
//! run, from places of a large page-locked buffer (as the RAM tier is) into places of a
//! device buffer (as the slots are).
//!
//! ```sh
//! cargo run --release -p ares-cuda --example upload_bench -- [--device 0] [--mib 2048]
//! ```

use std::time::Instant;

use ares_cuda::{Device, DeviceBuffer, PinnedBuffer, Result, Stream, Wait};

const SIZES: &[usize] = &[256 << 10, 921_600, 1_228_800, 3 << 20, 16 << 20, 64 << 20];

fn main() -> Result<()> {
    let mut device = 0u32;
    let mut mib = 2048usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        match arg.as_str() {
            "--device" => device = value() as u32,
            "--mib" => mib = value().max(64),
            _ => {
                eprintln!("usage: upload_bench [--device N] [--mib M]");
                return Ok(());
            }
        }
    }
    let dev = Device::open(device, Wait::Spin)?;
    println!("device {}: {}", device, dev.info().name);
    let host_bytes = 1usize << 30;
    let device_bytes = 512usize << 20;
    let mut host = PinnedBuffer::new(&dev, host_bytes)?;
    // Touch the pages, as a filled RAM tier would have them.
    host.fill(0x5a);
    let slots = DeviceBuffer::new(&dev, device_bytes as u64)?;
    let streams = [Stream::new(&dev, 0)?, Stream::new(&dev, 0)?];
    println!("{:>10} {:>8} {:>10} {:>10}", "copy", "streams", "GB/s", "us a copy");
    for &size in SIZES {
        for n_streams in [1usize, 2] {
            let copies = (mib << 20) / size;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                for i in 0..copies {
                    let src = (i * size * 7) % (host_bytes - size);
                    let dst = (i * size) % (device_bytes - size);
                    // SAFETY: `size` bytes inside both buffers, which outlive the
                    // synchronisation below; nothing else uses them.
                    unsafe {
                        streams[i % n_streams].copy_from_host(
                            slots.ptr().offset(dst as u64),
                            host.as_ptr().add(src),
                            size,
                        )?;
                    }
                }
                for stream in &streams {
                    stream.synchronize()?;
                }
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "{:>8} K {:>8} {:>10.2} {:>10.1}",
                size >> 10,
                n_streams,
                (copies * size) as f64 / best / 1e9,
                best / copies as f64 * 1e6
            );
        }
    }
    Ok(())
}
