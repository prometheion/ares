//! Measures the image tower (`qwen3vl::DeviceTower`) of a projector file on one
//! device, in both homes of its weights.
//!
//! ```text
//! qwen3vl_tower <mmproj.gguf> [--device 0] [--rows 256,1024,4096] [--rounds 3]
//!     [--homes vram,ram]
//! ```
//!
//! For each home: the load, the VRAM it holds between images and the scratch of
//! the largest image (both as the driver reports free memory), the page-locked
//! RAM; then for each size an image of that many rows (a square grid where the
//! rows are a square, made of a smooth pattern; the content does not change the
//! time), its patch rows uploaded, and `--rounds` passes timed on the device
//! (events around the pass) and on the host (to the end of the pass). In the
//! RAM home also the uploads alone, on the tower's copy stream: what of them a
//! pass does not hide is its time less the VRAM home's. The homes take turns,
//! round by round.

use std::time::Instant;

use ares_cuda::{Device, DeviceBuffer, Event, Kernels, Stream, Wait};
use ares_gguf::Gguf;
use ares_model::qwen3vl::{Config, DeviceTower, Home, Weights, patch_rows};

const MIB: f64 = 1024.0 * 1024.0;

struct Args {
    file: String,
    device: u32,
    rows: Vec<u32>,
    rounds: usize,
    homes: Vec<Home>,
}

fn parse() -> Args {
    let mut args = std::env::args().skip(1);
    let mut parsed = Args {
        file: String::new(),
        device: 0,
        rows: vec![256, 1024, 4096],
        rounds: 3,
        homes: vec![Home::Vram, Home::Ram],
    };
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} wants a value"));
        match arg.as_str() {
            "--device" => parsed.device = value().parse().unwrap(),
            "--rows" => parsed.rows = value().split(',').map(|r| r.parse().unwrap()).collect(),
            "--rounds" => parsed.rounds = value().parse().unwrap(),
            "--homes" => {
                parsed.homes = value()
                    .split(',')
                    .map(|h| match h {
                        "vram" => Home::Vram,
                        "ram" => Home::Ram,
                        other => panic!("--homes: {other} is neither vram nor ram"),
                    })
                    .collect()
            }
            file if parsed.file.is_empty() && !file.starts_with("--") => parsed.file = file.into(),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!parsed.file.is_empty(), "usage: qwen3vl_tower <mmproj.gguf> [options]");
    parsed
}

/// The sides in pixels of an image of `rows` rows: a square grid when `rows`
/// is a square, else the widest grid of at most twice as many columns as lines.
fn sides(rows: u32) -> (usize, usize) {
    let mut ny = (rows as f64).sqrt() as u32;
    while !rows.is_multiple_of(ny) {
        ny -= 1;
    }
    ((rows / ny) as usize * 32, ny as usize * 32)
}

/// Normalised pixels of a smooth pattern.
fn pixels(width: usize, height: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (x as f32 / width as f32, y as f32 / height as f32);
            out.extend([
                (6.0 * u).sin() * 0.8,
                (5.0 * v + 2.0 * u).cos() * 0.8,
                ((7.0 * u * v).sin() - 0.2).clamp(-1.0, 1.0),
            ]);
        }
    }
    out
}

fn main() {
    let args = parse();
    let gguf = Gguf::open(&args.file).expect("projector file");
    let config = Config::from_gguf(&gguf).expect("projector configuration");
    let weights = Weights::load(&gguf, &config).expect("projector weights");
    let device = Device::open(args.device, Wait::Block).expect("device");
    let kernels = Kernels::load(&device).expect("kernels");
    let stream = Stream::new(&device, 0).expect("stream");
    let largest = *args.rows.iter().max().unwrap();
    println!(
        "{} on {} ({} MiB free), load average {}",
        args.file,
        device.info().name,
        device.mem_info().unwrap().free / (1 << 20),
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim()
    );

    struct Loaded {
        tower: DeviceTower,
        scratch: ares_model::qwen3vl::Scratch,
    }
    let mut loaded = Vec::new();
    for home in &args.homes {
        let free = device.mem_info().unwrap().free;
        let started = Instant::now();
        let tower = DeviceTower::load(&device, &kernels, &config, &weights, *home).expect("load");
        let seconds = started.elapsed().as_secs_f64();
        let held = free - device.mem_info().unwrap().free;
        let scratch = tower.scratch(largest).expect("scratch");
        let with_scratch = free - device.mem_info().unwrap().free;
        println!(
            "{home:?}: load {seconds:.2} s; VRAM held between images {:.1} MiB (driver: {:.1}), \
             scratch for {largest} rows {:.1} MiB (driver: {:.1}); page-locked RAM {:.1} MiB",
            tower.vram_bytes() as f64 / MIB,
            held as f64 / MIB,
            scratch.bytes() as f64 / MIB,
            (with_scratch - held) as f64 / MIB,
            tower.pinned_bytes() as f64 / MIB,
        );
        loaded.push(Loaded { tower, scratch });
    }

    let (start, end) = (Event::timed(&device).unwrap(), Event::timed(&device).unwrap());
    for &rows in &args.rows {
        let (width, height) = sides(rows);
        let image = pixels(width, height);
        let (patches, nx, ny) = patch_rows(&config, &image, width, height).unwrap();
        let bytes: Vec<u8> = patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = DeviceBuffer::new(&device, u64::from(rows) * config.n_out as u64 * 4).unwrap();
        let mut times: Vec<Vec<(f64, f64)>> = vec![Vec::new(); loaded.len()];
        let mut uploads = Vec::new();
        // One pass of each home to warm up, then the rounds, the homes in turn.
        for round in 0..=args.rounds {
            for (i, l) in loaded.iter_mut().enumerate() {
                let input = l.scratch.input().as_raw() - l.scratch.buffer().ptr().as_raw();
                l.scratch.buffer().write(input, &bytes).unwrap();
                let started = Instant::now();
                start.record(&stream).unwrap();
                // SAFETY: `out` holds the rows and outlives the pass, waited for below.
                unsafe {
                    l.tower.enqueue(&stream, &l.scratch, nx as u32, ny as u32, out.ptr()).unwrap()
                };
                end.record(&stream).unwrap();
                end.synchronize().unwrap();
                let host = started.elapsed().as_secs_f64() * 1000.0;
                let device_ms = f64::from(end.elapsed_ms_since(&start).unwrap());
                if round > 0 {
                    times[i].push((device_ms, host));
                    if let Some(upload) = l.tower.upload_time().unwrap() {
                        uploads.push(upload.as_secs_f64() * 1000.0);
                    }
                }
            }
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let vram = args.homes.iter().position(|h| *h == Home::Vram).map(|i| {
            median(times[i].iter().map(|t| t.0).collect())
        });
        for (i, home) in args.homes.iter().enumerate() {
            let device_ms: Vec<f64> = times[i].iter().map(|t| t.0).collect();
            let host_ms: Vec<f64> = times[i].iter().map(|t| t.1).collect();
            let all = device_ms.iter().map(|t| format!("{t:.1}")).collect::<Vec<_>>().join(" ");
            print!(
                "{rows} rows ({width} x {height} px) {home:?}: {:.1} ms on the device (rounds: \
                 {all}), {:.1} ms on the host",
                median(device_ms.clone()),
                median(host_ms)
            );
            if *home == Home::Ram && !uploads.is_empty() {
                let upload = median(uploads.clone());
                let image = loaded[i].tower.plan().image_bytes() as f64;
                print!(
                    "; uploads alone {upload:.1} ms ({:.2} GB/s)",
                    image / upload / 1e6
                );
                if let Some(vram) = vram {
                    let unhidden = (median(device_ms) - vram).max(0.0);
                    print!(
                        ", {unhidden:.1} ms more than in VRAM: {:.0} % of the uploads hidden",
                        100.0 * (1.0 - unhidden / upload).clamp(0.0, 1.0)
                    );
                }
            }
            println!();
        }
    }
}
