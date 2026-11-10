//! Measures what copies of a file on several drives are worth: random reads of
//! the size of an expert from each copy alone, then from all of them together.
//!
//! The reads bypass the page cache, so a run measures the drives. Each line
//! keeps `depth` reads in flight (a read is one chunk here) and reports the
//! throughput, the latency of a read and how the bytes were shared between the
//! drives.
//!
//! It only reads. The copies must be the same bytes (the reader checks sizes
//! and samples at open).
//!
//! ```sh
//! cargo run --release -p ares-io --example mirror_bench -- FILE COPY [COPY...] \
//!     [--mib 2048] [--kib 1440] [--depths 8,32,64] [--seed 1]
//! ```

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ares_io::{Buffer, Class, Config, FileId, Read, Reader};

struct Args {
    paths: Vec<PathBuf>,
    mib: u64,
    kib: usize,
    depths: Vec<usize>,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args =
        Args { paths: Vec::new(), mib: 2048, kib: 1440, depths: vec![8, 32, 64], seed: 1 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--mib" => args.mib = value("--mib")?.parse().map_err(|e| format!("--mib: {e}"))?,
            "--kib" => args.kib = value("--kib")?.parse().map_err(|e| format!("--kib: {e}"))?,
            "--seed" => args.seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--depths" => {
                args.depths = value("--depths")?
                    .split(',')
                    .map(|d| d.trim().parse().map_err(|e| format!("--depths: {e}")))
                    .collect::<Result<_, _>>()?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            path => args.paths.push(PathBuf::from(path)),
        }
    }
    if args.paths.len() < 2 {
        return Err("give the file and at least one copy of it".into());
    }
    if args.kib == 0 || args.depths.contains(&0) {
        return Err("--kib and --depths must be above zero".into());
    }
    Ok(args)
}

/// xorshift64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One line of the table: `total` bytes of random reads of `len` bytes through
/// `reader` with `depth` of them in flight.
fn run(
    reader: &Reader,
    what: &str,
    depth: usize,
    len: usize,
    total: u64,
    seed: u64,
) -> Result<(), String> {
    let size = reader.files()[0].size;
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let reads = (total / len as u64).max(depth as u64 * 4);
    let align = reader.buffer_alignment();
    let span = (len + 2 * align).next_multiple_of(align);
    let mut free: Vec<Buffer> = (0..depth).map(|_| Buffer::aligned(span, align)).collect();
    let mut started: Vec<Option<Instant>> = vec![None; reads as usize];
    let mut latencies = Vec::with_capacity(reads as usize);
    let before = reader.drive_stats();
    let (mut next, mut done) = (0u64, 0u64);
    let t0 = Instant::now();
    while done < reads {
        while next < reads {
            let Some(buffer) = free.pop() else { break };
            let offset = rng.next() % (size - len as u64);
            started[next as usize] = Some(Instant::now());
            reader
                .submit(Read {
                    key: next,
                    file: FileId(0),
                    offset,
                    len,
                    class: Class::Demand,
                    buffer,
                })
                .map_err(|r| r.to_string())?;
            next += 1;
        }
        let c = reader.wait_any(Duration::from_secs(60)).ok_or("a read did not finish")?;
        c.result.as_ref().map_err(|e| e.to_string())?;
        latencies.push(started[c.key as usize].expect("a started read").elapsed());
        free.push(c.buffer);
        done += 1;
    }
    let seconds = t0.elapsed().as_secs_f64();
    latencies.sort_unstable();
    let at = |p: f64| latencies[((latencies.len() as f64 * p) as usize).min(latencies.len() - 1)];
    let after = reader.drive_stats();
    let shares: Vec<String> = reader
        .drives()
        .iter()
        .zip(before.iter().zip(&after))
        .map(|(drive, (b, a))| format!("{} {} MiB", drive.name, (a.bytes - b.bytes) >> 20))
        .collect();
    println!(
        "{what:<24} {depth:>5} {:>8.0} {:>10.2} {:>8.2} {:>8.2}   {}",
        (reads * len as u64) as f64 / seconds / (1 << 20) as f64,
        at(0.5).as_secs_f64() * 1e3,
        at(0.95).as_secs_f64() * 1e3,
        at(0.99).as_secs_f64() * 1e3,
        shares.join(", ")
    );
    Ok(())
}

fn real_main() -> Result<(), String> {
    let args = parse_args()?;
    let len = args.kib * 1024;
    let total = args.mib << 20;

    // Each copy alone, then all together.
    let mut setups: Vec<(String, Vec<PathBuf>)> = Vec::new();
    for path in &args.paths {
        setups.push((String::new(), vec![path.clone()]));
    }
    setups.push(("all copies".into(), args.paths.clone()));

    println!("reads of {} KiB, {} MiB per line", args.kib, args.mib);
    println!(
        "{:<24} {:>5} {:>8} {:>10} {:>8} {:>8}   bytes per drive",
        "copies", "depth", "MiB/s", "median ms", "p95 ms", "p99 ms"
    );
    for (name, paths) in &mut setups {
        let reader = Reader::open_copies(&[paths.as_slice()], Config::default())
            .map_err(|e| e.to_string())?;
        if name.is_empty() {
            *name = format!("only {}", reader.drives()[0].name);
        }
        let file = &reader.files()[0];
        for (copy, path) in file.copies.iter().zip(paths.iter()) {
            if copy.mode != ares_io::IoMode::Direct {
                println!("note: {} is read with {}", path.display(), copy.mode);
            }
        }
        if reader.drives().len() < paths.len() {
            println!("note: {} copies are on {} drive(s)", paths.len(), reader.drives().len());
        }
        for &depth in &args.depths {
            run(&reader, name, depth, len, total, args.seed)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
