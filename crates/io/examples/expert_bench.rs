//! Measures how fast random experts of a real model can be read from the SSD
//! with direct I/O, at several queue depths.
//!
//! The experts are read the way the engine will read the ones that are not
//! in VRAM or RAM: each expert is one demand read per stacked tensor (gate,
//! up, down), straight from the GGUF at `offset + e * stride`. For each queue
//! depth the benchmark keeps that many experts in flight, picks the next one
//! at random whenever one finishes, and reports throughput and the latency of
//! a whole expert (from submitting its first read to receiving its last).
//! Reads bypass the page cache, so a run measures the drive, not memory.
//!
//! It only reads. The disk may be shared with other work, so each run is kept
//! to about a gigabyte by default.
//!
//! ```sh
//! cargo run --release -p ares-io --example expert_bench -- MODEL.gguf \
//!     [--mib 1024] [--depths 1,4,16,64] [--patterns ffn_gate_exps,ffn_up_exps,ffn_down_exps] \
//!     [--seed 1] [--buffered] [--queue-depth 32] [--hot N] [--fixed] [--mixed N] \
//!     [--mirror DIR]...
//! ```
//!
//! `--mirror DIR` names a directory on another drive that holds a copy of every
//! part of the model; the reads are then spread over the drives, and the last
//! lines say how much each delivered.
//!
//! `--hot N` draws the experts from a fixed set of N, so that after the first
//! pass the drive answers from its own cache and what is left to see is the
//! cost on the host. `--fixed` registers the read buffers with the ring
//! (limited by `ulimit -l`) to compare that cost with and without. `--mixed N`
//! adds a second measurement: demand experts one at a time while N experts of
//! background traffic are in flight, once with the background as prefetches and
//! once as demand reads, to show what the priority classes buy.

use std::collections::HashMap;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use ares_gguf::Gguf;
use ares_io::{Buffer, Class, Config, ExpertMap, FileId, FixedRegion, Read, Reader};

struct Args {
    model: String,
    mib: u64,
    depths: Vec<usize>,
    patterns: Vec<String>,
    seed: u64,
    direct: bool,
    queue_depth: usize,
    /// Draw from this many distinct experts instead of all of them.
    hot: Option<usize>,
    /// Register the read buffers with the ring.
    fixed: bool,
    /// Also measure demand latency next to this many experts of background
    /// prefetch.
    mixed: Option<usize>,
    /// Directories on other drives that hold a copy of the model.
    mirrors: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        model: String::new(),
        mib: 1024,
        depths: vec![1, 4, 16, 64],
        patterns: ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].map(String::from).to_vec(),
        seed: 1,
        direct: true,
        queue_depth: Config::default().queue_depth,
        hot: None,
        fixed: false,
        mixed: None,
        mirrors: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--mib" => args.mib = value("--mib")?.parse().map_err(|e| format!("--mib: {e}"))?,
            "--mirror" => args.mirrors.push(value("--mirror")?),
            "--seed" => args.seed = value("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--queue-depth" => {
                args.queue_depth =
                    value("--queue-depth")?.parse().map_err(|e| format!("--queue-depth: {e}"))?
            }
            "--depths" => {
                args.depths = value("--depths")?
                    .split(',')
                    .map(|d| d.parse().map_err(|e| format!("--depths: {e}")))
                    .collect::<Result<_, _>>()?
            }
            "--patterns" => {
                args.patterns = value("--patterns")?.split(',').map(String::from).collect()
            }
            "--buffered" => args.direct = false,
            "--fixed" => args.fixed = true,
            "--mixed" => {
                args.mixed = Some(value("--mixed")?.parse().map_err(|e| format!("--mixed: {e}"))?)
            }
            "--hot" => args.hot = Some(value("--hot")?.parse().map_err(|e| format!("--hot: {e}"))?),
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => args.model = arg,
        }
    }
    if args.model.is_empty() {
        return Err("usage: expert_bench MODEL.gguf [--mib N] [--depths 1,4,16,64] \
                    [--patterns a,b,c] [--seed N] [--buffered] [--queue-depth N] [--hot N] [--fixed] [--mixed N]"
            .into());
    }
    Ok(args)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// CPU time of this process (all threads), user plus system.
fn cpu_time() -> Duration {
    // SAFETY: an all-zero `rusage` is a valid value, and getrusage fills it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid out-pointer.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let gguf = Gguf::open(&args.model).map_err(|e| e.to_string())?;
    let patterns: Vec<&str> = args.patterns.iter().map(String::as_str).collect();
    let map = ExpertMap::discover(&gguf, &patterns).map_err(|e| e.to_string())?;
    let layers: Vec<usize> = map.moe_layers().collect();
    let first = layers[0];
    let n_experts = map.n_experts(first).expect("a MoE layer has experts");
    let expert_bytes = map.expert_bytes(first).expect("a MoE layer has experts");

    // With --fixed, all buffers are slices of one region registered when the
    // reader starts. It is declared first so that it outlives the reader.
    // The layers of a file may store their experts in different types (Unsloth's dynamic
    // quants do), so a buffer is as large as the matrix is in the largest of them.
    let align = 4096;
    let sizes = largest_ranges(&map, &layers, 2 * align, align);
    let per_slot: usize = sizes.iter().sum();
    let max_depth = args.depths.iter().copied().max().unwrap_or(1);
    let mut region = args.fixed.then(|| Buffer::aligned(per_slot * max_depth, align));
    let fixed_regions = match region.as_mut() {
        // SAFETY: `region` is declared before `reader` and so dropped after
        // it, and is not freed or moved meanwhile.
        Some(r) => vec![unsafe { FixedRegion::new(r.as_mut_ptr(), r.len()) }],
        None => Vec::new(),
    };

    let config = Config {
        direct: args.direct,
        queue_depth: args.queue_depth,
        prefetch_depth: Config::default().prefetch_depth.min(args.queue_depth),
        fixed_regions,
        ..Config::default()
    };
    let reader =
        Reader::open_gguf_mirrored(&gguf, &args.mirrors, config).map_err(|e| e.to_string())?;

    println!("model      {}", args.model);
    println!(
        "           architecture {}, {} part(s), {:.2} GiB, {} tensors",
        gguf.architecture().unwrap_or("?"),
        gguf.n_parts(),
        gguf.file_size() as f64 / (1u64 << 30) as f64,
        gguf.tensors().len()
    );
    println!(
        "experts    {} MoE layers x {} experts, {} bytes ({:.2} MiB) per expert in {} reads",
        layers.len(),
        n_experts,
        expert_bytes,
        expert_bytes as f64 / (1 << 20) as f64,
        patterns.len()
    );
    for (i, file) in reader.files().iter().enumerate() {
        println!(
            "file {i}     {}\n           {}; read granule {} bytes; kernel alignment {:?}",
            file.path.display(),
            file.mode,
            file.block,
            file.direct_align
        );
    }
    let disk = locate_disk(&args.model);
    if let Some(disk) = &disk {
        println!("device     {}", disk.description);
    }
    println!();
    println!(
        "{:>5} {:>8} {:>9} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9}",
        "depth",
        "experts",
        "MiB",
        "seconds",
        "GB/s",
        "expert/s",
        "median ms",
        "p95 ms",
        "p99 ms",
        "cpu s/GiB",
        "disk MiB"
    );

    let mut rng = Rng(args.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n_tensors = patterns.len();
    let hot: Option<Vec<(usize, usize)>> = args.hot.map(|n| {
        (0..n)
            .map(|_| {
                let layer = layers[(rng.next() % layers.len() as u64) as usize];
                (layer, (rng.next() % n_experts as u64) as usize)
            })
            .collect()
    });

    for &depth in &args.depths {
        let total = ((args.mib << 20) / expert_bytes).max(depth as u64 * 2);
        // A buffer per tensor of an expert, large enough for any range widened
        // to blocks.
        let mut free: Vec<Vec<Buffer>> = (0..depth)
            .map(|slot| {
                let mut at = slot * per_slot;
                sizes
                    .iter()
                    .map(|size| match region.as_mut() {
                        Some(r) => {
                            // SAFETY: slot `slot` is a disjoint part of the
                            // region, which outlives the reader.
                            let b = unsafe { Buffer::from_raw(r.as_mut_ptr().add(at), *size) };
                            at += size;
                            b
                        }
                        None => Buffer::aligned(*size, align),
                    })
                    .collect()
            })
            .collect();
        // sequence number -> (started, reads outstanding, buffers received)
        let mut live: HashMap<u64, (Instant, usize, Vec<Option<Buffer>>)> = HashMap::new();
        let (mut next, mut done) = (0u64, 0u64);
        let mut latencies = Vec::with_capacity(total as usize);
        let cpu0 = cpu_time();
        let disk0 = disk.as_ref().and_then(Disk::mib_read);
        let t0 = Instant::now();
        while done < total {
            while next < total && !free.is_empty() {
                let mut buffers = free.pop().expect("not empty");
                let (layer, expert) = match &hot {
                    Some(set) => set[(rng.next() % set.len() as u64) as usize],
                    None => {
                        let layer = layers[(rng.next() % layers.len() as u64) as usize];
                        (layer, (rng.next() % n_experts as u64) as usize)
                    }
                };
                let started = Instant::now();
                let ranges = map.ranges(layer, expert).expect("valid");
                for (m, (range, buffer)) in ranges.zip(buffers.drain(..)).enumerate() {
                    reader
                        .submit(Read {
                            key: next * 4 + m as u64,
                            file: FileId(range.file.0),
                            offset: range.offset,
                            len: range.len,
                            class: Class::Demand,
                            buffer,
                        })
                        .map_err(|r| r.to_string())?;
                }
                live.insert(next, (started, n_tensors, (0..n_tensors).map(|_| None).collect()));
                next += 1;
            }
            let c = reader.wait_any(Duration::from_secs(60)).ok_or("a read did not finish")?;
            c.result.as_ref().map_err(|e| e.to_string())?;
            let seq = c.key / 4;
            let entry = live.get_mut(&seq).expect("a live expert");
            entry.1 -= 1;
            entry.2[(c.key % 4) as usize] = Some(c.buffer);
            if entry.1 == 0 {
                let (started, _, buffers) = live.remove(&seq).expect("present");
                latencies.push(started.elapsed());
                free.push(buffers.into_iter().map(|b| b.expect("all reads came back")).collect());
                done += 1;
            }
        }
        let elapsed = t0.elapsed();
        let cpu = cpu_time() - cpu0;
        let disk_mib = disk.as_ref().and_then(Disk::mib_read).zip(disk0).map(|(b, a)| b - a);
        latencies.sort();
        let bytes = total * expert_bytes;
        println!(
            "{:>5} {:>8} {:>9.0} {:>8.2} {:>8.2} {:>9.0} {:>9.2} {:>9.2} {:>9.2} {:>10.3} {:>9.0}",
            depth,
            total,
            bytes as f64 / (1 << 20) as f64,
            elapsed.as_secs_f64(),
            bytes as f64 / 1e9 / elapsed.as_secs_f64(),
            total as f64 / elapsed.as_secs_f64(),
            percentile(&latencies, 0.5).as_secs_f64() * 1e3,
            percentile(&latencies, 0.95).as_secs_f64() * 1e3,
            percentile(&latencies, 0.99).as_secs_f64() * 1e3,
            cpu.as_secs_f64() / (bytes as f64 / (1u64 << 30) as f64),
            disk_mib.unwrap_or(f64::NAN),
        );
    }
    if let Some(background) = args.mixed {
        println!(
            "\ndemand experts one at a time next to {background} experts of background reads:"
        );
        println!(
            "{:>22} {:>9} {:>9} {:>9} {:>9} {:>8}",
            "background class", "median ms", "p95 ms", "p99 ms", "max ms", "bg GB/s"
        );
        for class in [Class::Prefetch, Class::Demand] {
            let (latencies, bg_bytes, elapsed) =
                mixed(&reader, &map, &layers, n_experts, background, class, 150, &mut rng)?;
            println!(
                "{:>22} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>8.2}",
                format!("{class:?}"),
                percentile(&latencies, 0.5).as_secs_f64() * 1e3,
                percentile(&latencies, 0.95).as_secs_f64() * 1e3,
                percentile(&latencies, 0.99).as_secs_f64() * 1e3,
                latencies.last().map_or(0.0, |d| d.as_secs_f64() * 1e3),
                bg_bytes as f64 / 1e9 / elapsed.as_secs_f64(),
            );
        }
    }
    let stats = reader.stats();
    println!(
        "\nreader     {} demand reads, {:.0} MiB wanted, {:.0} MiB read from the device, \
         peak queue depth {}",
        stats.demand.requests,
        stats.demand.bytes as f64 / (1 << 20) as f64,
        stats.demand.device_bytes as f64 / (1 << 20) as f64,
        stats.peak_in_flight
    );
    println!(
        "           per-read latency (submit to completion in the reader): median {:.2} ms, \
         p95 {:.2} ms, p99 {:.2} ms",
        stats.demand_latency.p50.as_secs_f64() * 1e3,
        stats.demand_latency.p95.as_secs_f64() * 1e3,
        stats.demand_latency.p99.as_secs_f64() * 1e3,
    );
    if reader.drives().len() > 1 {
        for (drive, s) in reader.drives().iter().zip(reader.drive_stats()) {
            println!(
                "drive      {}: {:.0} MiB in {} chunks, peak {} in flight",
                drive.name,
                s.bytes as f64 / (1 << 20) as f64,
                s.chunks,
                s.peak_in_flight
            );
        }
    }
    Ok(())
}

/// Keeps `background` experts in flight as `bg_class` reads and times
/// `count` experts read one after the other as demand reads. Returns the sorted
/// latencies of those, the background bytes read, and the time it took.
#[allow(clippy::too_many_arguments)]
fn mixed(
    reader: &Reader,
    map: &ExpertMap,
    layers: &[usize],
    n_experts: usize,
    background: usize,
    bg_class: Class,
    count: usize,
    rng: &mut Rng,
) -> Result<(Vec<Duration>, u64, Duration), String> {
    const BG: u64 = 1 << 62;
    let sizes = largest_ranges(map, layers, 8192, 4096);
    let alloc = || -> Vec<Buffer> { sizes.iter().map(|s| Buffer::aligned(*s, 4096)).collect() };
    let mut free: Vec<Vec<Buffer>> = (0..background + 1).map(|_| alloc()).collect();
    // sequence -> (started, outstanding, buffers); one table for both streams,
    // told apart by BG in the key.
    let mut live: HashMap<u64, (Instant, usize, Vec<Option<Buffer>>)> = HashMap::new();
    let mut seq = 0u64;
    let mut start = |free: &mut Vec<Vec<Buffer>>,
                     live: &mut HashMap<u64, (Instant, usize, Vec<Option<Buffer>>)>,
                     class: Class,
                     tag: u64,
                     rng: &mut Rng|
     -> Result<(), String> {
        let mut buffers = free.pop().expect("a free buffer set");
        let layer = layers[(rng.next() % layers.len() as u64) as usize];
        let expert = (rng.next() % n_experts as u64) as usize;
        let id = seq;
        seq += 1;
        for (m, (range, buffer)) in
            map.ranges(layer, expert).expect("valid").zip(buffers.drain(..)).enumerate()
        {
            reader
                .submit(Read {
                    key: tag | (id << 2) | m as u64,
                    file: FileId(range.file.0),
                    offset: range.offset,
                    len: range.len,
                    class,
                    buffer,
                })
                .map_err(|r| r.to_string())?;
        }
        live.insert(
            tag | id,
            (Instant::now(), sizes.len(), (0..sizes.len()).map(|_| None).collect()),
        );
        Ok(())
    };

    let t0 = Instant::now();
    let mut latencies = Vec::with_capacity(count);
    let mut bg_done = 0u64;
    for _ in 0..background {
        start(&mut free, &mut live, bg_class, BG, rng)?;
    }
    start(&mut free, &mut live, Class::Demand, 0, rng)?;
    while latencies.len() < count {
        let c = reader.wait_any(Duration::from_secs(60)).ok_or("a read did not finish")?;
        c.result.as_ref().map_err(|e| e.to_string())?;
        let tag = c.key & BG;
        let id = (c.key & !BG) >> 2;
        let entry = live.get_mut(&(tag | id)).expect("a live expert");
        entry.1 -= 1;
        entry.2[(c.key & 3) as usize] = Some(c.buffer);
        if entry.1 == 0 {
            let (started, _, buffers) = live.remove(&(tag | id)).expect("present");
            free.push(buffers.into_iter().map(|b| b.expect("all reads came back")).collect());
            if tag == 0 {
                latencies.push(started.elapsed());
                start(&mut free, &mut live, Class::Demand, 0, rng)?;
            } else {
                bg_done += 1;
                start(&mut free, &mut live, bg_class, BG, rng)?;
            }
        }
    }
    let elapsed = t0.elapsed();
    // Let the outstanding reads finish so the next run starts clean.
    let mut cancelled = 0;
    for key in live.keys().copied().collect::<Vec<_>>() {
        for m in 0..sizes.len() as u64 {
            let key = if key & BG != 0 { BG | ((key & !BG) << 2) | m } else { (key << 2) | m };
            if reader.cancel(key).is_ok() {
                cancelled += 1;
            }
        }
    }
    let _ = cancelled;
    while reader.stats().outstanding > 0 {
        if reader.wait_any(Duration::from_millis(200)).is_none() {
            break;
        }
    }
    latencies.sort();
    Ok((latencies, bg_done * map.expert_bytes(layers[0]).unwrap_or(0), elapsed))
}

/// The disk under `path`.
struct Disk {
    /// Name in `/proc/diskstats`, for example `nvme1n1`.
    name: String,
    /// What the benchmark prints about it.
    description: String,
}

impl Disk {
    /// Megabytes the disk has read since boot, from `/proc/diskstats`. The
    /// disk may be shared, so the difference between two readings includes
    /// everyone else's reads; it is a sanity check that direct reads reach
    /// the device (and that the page cache is not answering them).
    fn mib_read(&self) -> Option<f64> {
        let stats = std::fs::read_to_string("/proc/diskstats").ok()?;
        let line = stats.lines().find(|l| l.split_whitespace().nth(2) == Some(&self.name))?;
        let sectors: u64 = line.split_whitespace().nth(5)?.parse().ok()?;
        Some(sectors as f64 * 512.0 / (1 << 20) as f64)
    }
}

/// Finds the disk a file lives on through the mount table and sysfs.
fn locate_disk(path: &str) -> Option<Disk> {
    use std::os::unix::fs::MetadataExt;
    let path = std::fs::canonicalize(path).ok()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    // The mount with the longest mount point that is a prefix of the path.
    let (_, source) = mountinfo
        .lines()
        .filter_map(|line| {
            let mount_point = line.split(' ').nth(4)?;
            let (_, after) = line.split_once(" - ")?;
            let source = after.split(' ').nth(1)?;
            path.starts_with(mount_point).then(|| (mount_point.len(), source.to_owned()))
        })
        .max_by_key(|(len, _)| *len)?;
    let rdev = std::fs::metadata(&source).ok()?.rdev();
    let (rmaj, rmin) =
        (((rdev >> 8) & 0xfff) as u32, ((rdev & 0xff) | ((rdev >> 12) & !0xff)) as u32);
    let dir = std::fs::canonicalize(format!("/sys/dev/block/{rmaj}:{rmin}")).ok()?;
    // A partition's queue settings and statistics name are its disk's.
    let disk_dir = if dir.join("partition").exists() { dir.parent()?.to_owned() } else { dir };
    let read = |name: &str| {
        std::fs::read_to_string(disk_dir.join("queue").join(name)).ok().map(|s| s.trim().to_owned())
    };
    let model = std::fs::read_to_string(disk_dir.join("device/model")).unwrap_or_default();
    let name = disk_dir.file_name()?.to_string_lossy().into_owned();
    let description = format!(
        "{source} on {name} {}: logical block {} bytes, physical block {} bytes, max request {} KiB",
        model.trim(),
        read("logical_block_size")?,
        read("physical_block_size")?,
        read("max_sectors_kb")?,
    );
    Some(Disk { name, description })
}

/// Per matrix of an expert, the bytes a read buffer needs in the layer that stores it
/// largest, with `margin` bytes for the widening to blocks, rounded up to `align`.
fn largest_ranges(map: &ExpertMap, layers: &[usize], margin: usize, align: usize) -> Vec<usize> {
    let mut sizes: Vec<usize> = Vec::new();
    for &layer in layers {
        for (part, range) in map.ranges(layer, 0).expect("expert 0 exists").enumerate() {
            let size = (range.len + margin).next_multiple_of(align);
            if part == sizes.len() {
                sizes.push(size);
            } else {
                sizes[part] = sizes[part].max(size);
            }
        }
    }
    sizes
}
