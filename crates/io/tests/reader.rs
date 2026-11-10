//! The reader against real files: correctness of the bytes on both I/O paths,
//! the order rules, cancel and promote, errors, and many concurrent reads.
//!
//! Each test that depends on the I/O path runs on every setup of
//! `common::setups` (the build disk with direct I/O, the build disk with
//! direct I/O switched off, the system temp directory) and reports on stderr
//! how the file was actually read.

mod common;

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use ares_io::{
    Buffer, BufferedReason, Cancel, Class, Completion, Config, Error, FileId, IoMode, Promote,
    Read, Reader,
};
use common::*;

/// How long a read that must finish may take on a busy machine and disk.
const PATIENCE: Duration = Duration::from_secs(30);

/// Reads the ranges through `reader` in batches and compares them with `data`.
fn check_ranges(reader: &Reader, data: &[u8], ranges: impl Iterator<Item = (u64, usize)>) -> usize {
    let mut ranges = ranges.peekable();
    let mut key = 0;
    let mut checked = 0;
    while ranges.peek().is_some() {
        let batch: Vec<(u64, usize)> = ranges.by_ref().take(256).collect();
        let keys: Vec<u64> = (key..key + batch.len() as u64).collect();
        key += batch.len() as u64;
        for (k, (offset, len)) in keys.iter().zip(&batch) {
            reader.submit(read(reader, *k, Class::Demand, *offset, *len)).unwrap();
        }
        for (c, (offset, len)) in collect(reader, &keys).into_iter().zip(&batch) {
            let want = &data[*offset as usize..*offset as usize + len];
            assert!(c.result.is_ok(), "read of {len} at {offset}: {:?}", c.result);
            assert_eq!(c.data(), want, "bytes of {len} at offset {offset}");
            checked += 1;
        }
    }
    checked
}

#[test]
fn bytes_equal_the_file_at_every_alignment_and_across_the_end_of_file() {
    for setup in setups("align") {
        let size = 6 * 4096 + 1234; // the last block is partial
        let data = pattern(size, 1);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], setup.config(), "alignment");

        // Every start offset across two 4 KiB blocks, with ranges that end
        // inside a block, on a boundary and across one.
        let inside = (0..=2 * 4096 + 2u64).flat_map(|o| [1usize, 100, 4097].map(|l| (o, l)));
        let n = check_ranges(&reader, &data, inside);
        // Every start offset in the last 9100 bytes, reading to the very end
        // of the file and one byte: the end-of-file block is short.
        let tail = (size as u64 - 9100..=size as u64 - 1)
            .flat_map(|o| [size - o as usize, 1].map(|l| (o, l)));
        let m = check_ranges(&reader, &data, tail);
        report(&format!("alignment: {} checked ranges on {}", n + m, setup.name()));
    }
}

#[test]
fn files_are_opened_with_o_direct_exactly_when_the_reader_says_direct() {
    for setup in setups("odirect") {
        let path = setup.dir.file("data.bin", &pattern(10_000, 2));
        let reader =
            open_reporting(&setup, std::slice::from_ref(&path), setup.config(), "o_direct");
        let info = &reader.files()[0];
        let flags = open_flags_of(&path).expect("the reader keeps the file open");
        let has_o_direct = flags & libc::O_DIRECT != 0;
        assert_eq!(has_o_direct, info.mode == IoMode::Direct, "{}: flags {flags:#o}", setup.name());
        if info.mode == IoMode::Direct {
            assert!(info.block.is_power_of_two() && info.block >= 512, "block {}", info.block);
            assert_eq!(reader.buffer_alignment() as u64, info.block);
            assert!(reader.all_direct());
        } else {
            assert_eq!(info.block, 1);
            assert!(!reader.all_direct());
        }
        if setup.direct && info.mode != IoMode::Direct {
            report(&format!(
                "o_direct: {} has no direct I/O here ({})",
                setup.dir.label, info.mode
            ));
        }
    }
}

#[test]
fn direct_reads_leave_the_page_cache_alone_and_buffered_reads_fill_it() {
    for setup in setups("pagecache") {
        let size = 4 << 20;
        let path = setup.dir.file("data.bin", &pattern(size, 17));
        // Written through the cache: push it to disk and drop it from memory,
        // so that what is resident afterwards was put there by the reads.
        std::fs::File::open(&path).unwrap().sync_all().unwrap();
        evict_from_page_cache(&path);
        if resident_pages(&path) != 0 {
            // tmpfs keeps its files in the page cache, so there is nothing to
            // measure there.
            report(&format!(
                "pagecache: {} cannot be evicted from the page cache; skipped",
                setup.name()
            ));
            continue;
        }

        let reader =
            open_reporting(&setup, std::slice::from_ref(&path), setup.config(), "pagecache");
        let ranges: Vec<(u64, usize)> =
            (0..8).map(|i| (i * (512 << 10) + 700, 400 << 10)).collect();
        let keys: Vec<u64> = (0..ranges.len() as u64).collect();
        for (k, (offset, len)) in keys.iter().zip(&ranges) {
            reader.submit(read(&reader, *k, Class::Demand, *offset, *len)).unwrap();
        }
        assert!(collect(&reader, &keys).iter().all(|c| c.result.is_ok()));

        let resident = resident_pages(&path);
        match reader.files()[0].mode {
            IoMode::Direct => {
                assert_eq!(resident, 0, "direct reads must not fill the page cache");
            }
            IoMode::Buffered(_) => {
                assert!(resident > 0, "a buffered read leaves its pages in the cache");
            }
        }
        report(&format!(
            "pagecache: {} -> {}, {resident} of {} pages resident after reading",
            setup.name(),
            reader.files()[0].mode,
            size / 4096
        ));
    }
}

fn evict_from_page_cache(path: &std::path::Path) {
    use std::os::fd::AsRawFd;
    let file = std::fs::File::open(path).unwrap();
    // SAFETY: plain call on an open descriptor; the result only says whether
    // the advice was accepted.
    unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
}

/// How many pages of the file are in the page cache (`mincore`).
fn resident_pages(path: &std::path::Path) -> usize {
    use std::os::fd::AsRawFd;
    let file = std::fs::File::open(path).unwrap();
    let len = file.metadata().unwrap().len() as usize;
    // SAFETY: a private read-only mapping of the whole file; it is unmapped
    // before returning and `mincore` writes one byte per page into `vec`.
    unsafe {
        let map = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        );
        assert_ne!(map, libc::MAP_FAILED);
        let pages = len.div_ceil(4096);
        let mut vec = vec![0u8; pages];
        let rc = libc::mincore(map, len, vec.as_mut_ptr());
        assert_eq!(rc, 0);
        libc::munmap(map, len);
        vec.iter().filter(|b| *b & 1 != 0).count()
    }
}

/// The open flags of the descriptor this process holds on `path`, from
/// `/proc/self/fdinfo`.
fn open_flags_of(path: &std::path::Path) -> Option<i32> {
    for entry in std::fs::read_dir("/proc/self/fd").ok()? {
        let entry = entry.ok()?;
        if std::fs::read_link(entry.path()).ok().as_deref() == Some(path) {
            let info = std::fs::read_to_string(format!(
                "/proc/self/fdinfo/{}",
                entry.file_name().to_string_lossy()
            ))
            .ok()?;
            let flags = info.lines().find_map(|l| l.strip_prefix("flags:"))?;
            return i32::from_str_radix(flags.trim(), 8).ok();
        }
    }
    None
}

#[test]
fn the_fallback_gives_the_same_bytes_and_reports_itself() {
    let dir = &setups("fallback")[1]; // direct switched off on the build disk
    assert!(!dir.direct);
    let data = pattern(50_000, 3);
    let path = dir.dir.file("data.bin", &data);
    let buffered = open_reporting(dir, std::slice::from_ref(&path), dir.config(), "fallback");
    assert_eq!(buffered.files()[0].mode, IoMode::Buffered(BufferedReason::Disabled));
    assert!(!buffered.all_direct());

    let ranges: Vec<(u64, usize)> =
        vec![(0, 50_000), (1, 49_999), (4095, 2), (12_345, 1), (30_000, 20_000)];
    check_ranges(&buffered, &data, ranges.iter().copied());

    // The same ranges with direct I/O, where the filesystem has it.
    let direct = Reader::open(&[&path], Config::default()).unwrap();
    report(&format!(
        "fallback: the same file with direct I/O requested: {}",
        direct.files()[0].mode
    ));
    check_ranges(&direct, &data, ranges.iter().copied());
    if direct.all_direct() {
        // The widened span is what differs: the buffered path reads exactly.
        let s = buffered.span(FileId(0), 4095, 2).unwrap();
        assert_eq!((s.start, s.len, s.skip), (4095, 2, 0));
        let s = direct.span(FileId(0), 4095, 2).unwrap();
        assert_eq!(s.skip as u64, 4095 % direct.files()[0].block);
    }

    // On a filesystem without direct I/O the reason is `Unsupported`.
    let tmp = &setups("fallback-tmp")[2];
    let path = tmp.dir.file("data.bin", &data);
    let reader = open_reporting(tmp, &[path], tmp.config(), "fallback");
    match reader.files()[0].mode {
        IoMode::Direct => report("fallback: the system temp dir supports direct I/O here"),
        IoMode::Buffered(reason) => assert_eq!(reason, BufferedReason::Unsupported),
    }
    check_ranges(&reader, &data, ranges.iter().copied());
}

/// A reader whose chunks are one 4 KiB block each, so that a prefetch of a
/// few hundred KiB is hundreds of chunks, with every start recorded.
fn tracing_config(setup: &Setup) -> Config {
    Config {
        queue_depth: 4,
        prefetch_depth: 2,
        prefetch_chunk: 4096,
        record_starts: true,
        ..setup.config()
    }
}

/// Waits until `cond` holds.
fn until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_micros(200));
    }
}

/// Runs `attempt` until it returns true. A scenario that freezes a prefetch
/// half way returns false when the machine was too busy for the test thread to
/// get there before the prefetch ran to its end; a fresh attempt starts over.
fn retry(what: &str, mut attempt: impl FnMut() -> bool) {
    for _ in 0..25 {
        if attempt() {
            return;
        }
    }
    panic!("{what}: could not freeze a prefetch half way in 25 attempts");
}

/// Waits for a running prefetch of `total` chunks to have started a few
/// chunks, then holds the queue and waits for the device to drain. `None` when
/// the prefetch was already (almost) over.
fn freeze<'a>(reader: &'a Reader, total: usize) -> Option<(ares_io::Hold<'a>, usize)> {
    until("the prefetch to start", || reader.started().len() >= 4);
    let hold = reader.hold();
    until("the device to drain", || reader.stats().in_flight == 0);
    let before = reader.started().len();
    (before + 2 < total).then_some((hold, before))
}

#[test]
fn demand_reads_start_before_the_later_chunks_of_a_prefetch() {
    for setup in setups("order") {
        let data = pattern(2 << 20, 4);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], tracing_config(&setup), "order");

        // With the queue held, a prefetch of 256 chunks is submitted first
        // and two demand reads after it. Nothing has started, so the order in
        // which the chunks are handed to the ring is decided by the queues
        // alone.
        let hold = reader.hold();
        reader.submit(read(&reader, 1, Class::Prefetch, 0, 1 << 20)).unwrap();
        reader.submit(read(&reader, 2, Class::Demand, 1 << 20, 5000)).unwrap();
        reader.submit(read(&reader, 3, Class::Demand, (1 << 20) + 9000, 100)).unwrap();
        assert!(reader.started().is_empty());
        drop(hold);
        let done = collect(&reader, &[1, 2, 3]);

        let trace = reader.started();
        let mut want = vec![(2, 0), (3, 0)];
        want.extend((0..256).map(|c| (1, c)));
        assert_eq!(trace, want, "demand reads first, then the prefetch chunk by chunk");
        assert_eq!(done[0].data(), &data[..1 << 20]);
        assert_eq!(done[1].data(), &data[1 << 20..(1 << 20) + 5000]);
    }
}

#[test]
fn a_demand_read_submitted_while_a_prefetch_runs_goes_next() {
    for setup in setups("overtake") {
        let data = pattern(8 << 20, 5);
        let path = setup.dir.file("data.bin", &data);
        retry("overtake", || {
            let reader = open_reporting(
                &setup,
                std::slice::from_ref(&path),
                tracing_config(&setup),
                "overtake",
            );

            // A prefetch of 2048 chunks starts running and is frozen at a
            // known point, with nothing at the device.
            reader.submit(read(&reader, 1, Class::Prefetch, 0, 8 << 20)).unwrap();
            let Some((hold, before)) = freeze(&reader, 2048) else { return false };

            reader.submit(read(&reader, 2, Class::Demand, 3 << 20, 4096)).unwrap();
            drop(hold);
            let done = collect(&reader, &[2, 1]);

            let trace = reader.started();
            assert_eq!(trace[before], (2, 0), "the very next chunk started is the demand read's");
            assert_eq!(trace.len(), 2048 + 1);
            assert!(trace[..before].iter().chain(&trace[before + 1..]).all(|(k, _)| *k == 1));
            assert_eq!(done[0].data(), &data[3 << 20..(3 << 20) + 4096]);
            assert_eq!(done[1].data(), &data[..]);
            true
        });
    }
}

#[test]
fn cancelling_a_queued_prefetch_returns_its_buffer_and_starts_nothing() {
    for setup in setups("cancel-queued") {
        let path = setup.dir.file("data.bin", &pattern(2 << 20, 6));
        let reader = open_reporting(&setup, &[path], tracing_config(&setup), "cancel");

        let hold = reader.hold();
        let r = read(&reader, 1, Class::Prefetch, 0, 1 << 20);
        let (ptr, len) = (r.buffer.as_ptr(), r.buffer.len());
        reader.submit(r).unwrap();
        assert_eq!(reader.cancel(1).unwrap(), Cancel::Cancelled);
        let c = reader.wait(1, PATIENCE).unwrap().expect("a queued read cancels at once");
        assert!(matches!(c.result, Err(Error::Cancelled)), "{:?}", c.result);
        assert_eq!((c.buffer.as_ptr(), c.buffer.len()), (ptr, len), "the same buffer is back");
        assert!(c.data().is_empty());
        drop(hold);

        // The queue is not poisoned: a later read works, and the cancelled
        // prefetch never started a chunk.
        reader.submit(read(&reader, 2, Class::Demand, 0, 100)).unwrap();
        assert!(collect(&reader, &[2])[0].result.is_ok());
        assert_eq!(reader.started(), [(2, 0)]);
        assert_eq!(reader.stats().cancelled, 1);
        assert!(matches!(reader.cancel(1), Err(Error::UnknownKey(1))), "its completion was taken");
    }
}

#[test]
fn cancelling_a_running_prefetch_stops_its_later_chunks() {
    for setup in setups("cancel-running") {
        let path = setup.dir.file("data.bin", &pattern(8 << 20, 7));
        retry("cancel-running", || {
            let reader = open_reporting(
                &setup,
                std::slice::from_ref(&path),
                tracing_config(&setup),
                "cancel-running",
            );
            reader.submit(read(&reader, 1, Class::Prefetch, 0, 8 << 20)).unwrap();
            let Some((hold, before)) = freeze(&reader, 2048) else { return false };

            assert_eq!(reader.cancel(1).unwrap(), Cancel::Cancelled);
            let c =
                reader.wait(1, PATIENCE).unwrap().expect("nothing is at the device, so it is done");
            assert!(matches!(c.result, Err(Error::Cancelled)));
            assert_eq!(c.buffer.len(), 8 << 20);
            drop(hold);

            reader.submit(read(&reader, 2, Class::Demand, 0, 100)).unwrap();
            collect(&reader, &[2]);
            let trace = reader.started();
            assert_eq!(trace.len(), before + 1, "no chunk of the cancelled read started after it");
            assert_eq!(trace[before], (2, 0));
            let stats = reader.stats();
            assert_eq!(
                stats.prefetch.device_bytes,
                before as u64 * 4096,
                "only started chunks were read"
            );
            true
        });
    }
}

#[test]
fn promoting_a_prefetch_moves_its_chunks_ahead_of_other_prefetches() {
    for setup in setups("promote") {
        let data = pattern(2 << 20, 8);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], tracing_config(&setup), "promote");

        let hold = reader.hold();
        reader.submit(read(&reader, 1, Class::Prefetch, 0, 256 << 10)).unwrap(); // 64 chunks
        reader.submit(read(&reader, 2, Class::Prefetch, 256 << 10, 256 << 10)).unwrap();
        reader.submit(read(&reader, 3, Class::Demand, 512 << 10, 100)).unwrap();
        assert_eq!(reader.promote(2).unwrap(), Promote::Promoted);
        assert_eq!(reader.promote(2).unwrap(), Promote::AlreadyDemand);
        assert_eq!(reader.promote(3).unwrap(), Promote::AlreadyDemand);
        drop(hold);
        let done = collect(&reader, &[1, 2, 3]);

        let mut want = vec![(3, 0)];
        want.extend((0..64).map(|c| (2, c)));
        want.extend((0..64).map(|c| (1, c)));
        assert_eq!(
            reader.started(),
            want,
            "demand, then the promoted read, then the other prefetch"
        );
        assert!(done[1].promoted && !done[0].promoted);
        assert_eq!(done[1].class, Class::Prefetch, "the completion says what it was submitted as");
        assert_eq!(done[1].data(), &data[256 << 10..512 << 10]);
        assert_eq!(reader.stats().promoted, 1);
    }
}

#[test]
fn promoting_a_running_prefetch_reorders_what_is_left() {
    for setup in setups("promote-running") {
        let path = setup.dir.file("data.bin", &pattern(8 << 20, 9));
        retry("promote-running", || {
            let reader = open_reporting(
                &setup,
                std::slice::from_ref(&path),
                tracing_config(&setup),
                "promote-running",
            );
            reader.submit(read(&reader, 1, Class::Prefetch, 0, 4 << 20)).unwrap(); // 1024 chunks
            reader.submit(read(&reader, 2, Class::Prefetch, 4 << 20, 1 << 20)).unwrap(); // 256
            let Some((hold, before)) = freeze(&reader, 1024) else { return false };
            assert!(reader.started().iter().all(|(k, _)| *k == 1), "the second prefetch is behind");

            assert_eq!(reader.promote(2).unwrap(), Promote::Promoted);
            drop(hold);
            collect(&reader, &[2, 1]);

            let trace = reader.started();
            let rest = &trace[before..];
            assert!(
                rest[..256].iter().enumerate().all(|(i, t)| *t == (2, i)),
                "promoted chunks first"
            );
            assert!(
                rest[256..].iter().enumerate().all(|(i, t)| *t == (1, before + i)),
                "then the first prefetch carries on where it stopped"
            );
            assert_eq!(rest.len(), 256 + 1024 - before);
            true
        });
    }
}

#[test]
fn a_read_past_the_end_of_the_file_fails_alone() {
    for setup in setups("eof") {
        let size = 10_000usize;
        let data = pattern(size, 10);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], setup.config(), "eof");

        let cases: [(u64, usize); 6] = [
            (0, 100),                  // fine
            (size as u64 + 4096, 100), // entirely past the end
            (size as u64 - 50, 100),   // starts inside, ends past
            (size as u64 - 100, 100),  // ends exactly at the end
            (5000, 1000),              // fine
            (size as u64, 1),          // the first byte past the end
        ];
        for (key, (offset, len)) in cases.iter().enumerate() {
            reader.submit(read(&reader, key as u64, Class::Demand, *offset, *len)).unwrap();
        }
        let done = collect(&reader, &[0, 1, 2, 3, 4, 5]);
        for i in [0, 3, 4] {
            let (offset, len) = cases[i];
            assert_eq!(done[i].data(), &data[offset as usize..offset as usize + len], "case {i}");
        }
        for (i, got) in [(1, 0), (2, 50), (5, 0)] {
            match &done[i].result {
                Err(Error::ShortRead { got: g, wanted, size: s, .. }) => {
                    assert_eq!((*g, *wanted, *s), (got, cases[i].1, size as u64), "case {i}");
                }
                other => panic!("case {i}: expected a short read, got {other:?}"),
            }
            assert!(done[i].data().is_empty());
            assert_eq!(
                done[i].buffer.len(),
                reader.span(FileId(0), cases[i].0, cases[i].1).unwrap().len
            );
        }

        // The queue carries on, and the failures are counted.
        reader.submit(read(&reader, 10, Class::Demand, 100, 100)).unwrap();
        assert_eq!(collect(&reader, &[10])[0].data(), &data[100..200]);
        let stats = reader.stats();
        assert_eq!((stats.demand.requests, stats.demand.completed, stats.demand.failed), (7, 7, 3));
        let message = done[2].result.as_ref().unwrap_err().to_string();
        assert!(message.contains("data.bin") && message.contains("truncated"), "{message}");
    }
}

#[test]
fn refused_reads_come_back_whole_and_the_reader_stays_usable() {
    for setup in setups("refused") {
        let data = pattern(100_000, 11);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], setup.config(), "refused");

        // A buffer that is too small.
        let mut small = read(&reader, 1, Class::Demand, 100, 5000);
        small.buffer = Buffer::aligned(10, reader.buffer_alignment());
        let rejected = reader.submit(small).unwrap_err();
        assert!(matches!(rejected.error, Error::BufferTooSmall { have: 10, .. }));
        assert_eq!(rejected.read.buffer.len(), 10);
        assert_eq!((rejected.read.key, rejected.read.offset, rejected.read.len), (1, 100, 5000));

        // A file the reader does not have, and a range beyond the address space.
        let mut other = read(&reader, 2, Class::Demand, 0, 10);
        other.file = FileId(7);
        assert!(matches!(
            reader.submit(other).unwrap_err().error,
            Error::UnknownFile(FileId(7), 1)
        ));
        let huge =
            Read { offset: u64::MAX - 5, len: 100, ..read(&reader, 3, Class::Demand, 0, 10) };
        assert!(matches!(reader.submit(huge).unwrap_err().error, Error::RangeOverflow { .. }));

        // Misaligned memory is refused where direct I/O needs alignment.
        if reader.buffer_alignment() > 1 {
            let mut backing = vec![0u8; 16384];
            let align = reader.buffer_alignment();
            let aligned = backing.as_mut_ptr().align_offset(align);
            // SAFETY: `backing` outlives the buffer, which is only inspected
            // by the failed submit and dropped.
            let buffer = unsafe { Buffer::from_raw(backing.as_mut_ptr().add(aligned + 1), 8192) };
            let mut r = read(&reader, 4, Class::Demand, 0, 10);
            r.buffer = buffer;
            assert!(matches!(reader.submit(r).unwrap_err().error, Error::BufferMisaligned { .. }));
        }

        // A key in use is refused until its completion is taken.
        reader.submit(read(&reader, 5, Class::Demand, 0, 10)).unwrap();
        let again = reader.submit(read(&reader, 5, Class::Demand, 20, 10)).unwrap_err();
        assert!(matches!(again.error, Error::DuplicateKey(5)));
        let c = reader.wait(5, PATIENCE).unwrap().unwrap();
        assert_eq!(c.data(), &data[..10]);
        reader.submit(read(&reader, 5, Class::Demand, 20, 10)).unwrap();
        assert_eq!(reader.wait(5, PATIENCE).unwrap().unwrap().data(), &data[20..30]);
        assert_eq!(reader.stats().outstanding, 0);
    }
}

#[test]
fn thousands_of_concurrent_reads_complete_exactly_once() {
    const THREADS: u64 = 4;
    const PER_THREAD: u64 = 1500;
    for setup in setups("many") {
        let size = 1 << 20;
        let data = pattern(size, 12);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], setup.config(), "many");

        let mut seen = HashSet::new();
        let total = (THREADS * PER_THREAD) as usize;
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let (reader, data) = (&reader, &data);
                scope.spawn(move || {
                    let mut state = 0x1234_5678 + t;
                    for i in 0..PER_THREAD {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let len = 1 + (state >> 40) as usize % 3000;
                        let offset = (state >> 8) % (size as u64 - len as u64);
                        let class = if i % 3 == 0 { Class::Prefetch } else { Class::Demand };
                        let key = t * PER_THREAD + i;
                        reader.submit(read(reader, key, class, offset, len)).unwrap();
                        let _ = data;
                    }
                });
            }
            // Meanwhile this thread polls, as the engine's step loop does.
            let mut batch = Vec::new();
            let deadline = Instant::now() + Duration::from_secs(60);
            while seen.len() < total {
                assert!(Instant::now() < deadline, "only {} of {total} finished", seen.len());
                batch.clear();
                if reader.poll(&mut batch) == 0 {
                    std::thread::sleep(Duration::from_micros(100));
                }
                for c in batch.drain(..) {
                    assert!(c.result.is_ok(), "{:?}", c.result);
                    let (t, i) = (c.key / PER_THREAD, c.key % PER_THREAD);
                    // The request is reproducible from the key.
                    let mut state = 0x1234_5678 + t;
                    for _ in 0..=i {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                    }
                    let len = 1 + (state >> 40) as usize % 3000;
                    let offset = ((state >> 8) % (size as u64 - len as u64)) as usize;
                    assert_eq!(c.data(), &data[offset..offset + len], "key {}", c.key);
                    assert!(seen.insert(c.key), "key {} completed twice", c.key);
                }
            }
        });

        let mut extra = Vec::new();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(reader.poll(&mut extra), 0, "no completion is delivered twice");
        let stats = reader.stats();
        assert_eq!(stats.demand.requests + stats.prefetch.requests, total as u64);
        assert_eq!(stats.demand.completed + stats.prefetch.completed, total as u64);
        assert_eq!((stats.outstanding, stats.in_flight, stats.queued_demand), (0, 0, 0));
        assert!(stats.peak_in_flight <= Config::default().queue_depth);
        report(&format!(
            "many: {total} reads on {}, peak queue depth {}",
            setup.name(),
            stats.peak_in_flight
        ));
    }
}

#[test]
fn random_traffic_with_cancels_promotions_and_holds_keeps_every_invariant() {
    for setup in setups("random") {
        let size = 2 << 20;
        let data = pattern(size, 18);
        let path = setup.dir.file("data.bin", &data);
        let config =
            Config { queue_depth: 8, prefetch_depth: 3, prefetch_chunk: 4096, ..setup.config() };
        let reader = open_reporting(&setup, &[path], config, "random");

        let mut state = 0xDEAD_BEEF_u64;
        let mut rand = move |n: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        // key -> (offset, len, what cancel said, if called)
        let mut asked: HashMap<u64, (usize, usize, Option<Cancel>)> = HashMap::new();
        let mut live: Vec<u64> = Vec::new();
        let mut done: HashMap<u64, Completion> = HashMap::new();
        let mut hold = None;
        let total = 3000u64;
        let mut next = 0u64;
        let mut batch = Vec::new();
        while done.len() < total as usize {
            match rand(10) {
                0..=4 if next < total => {
                    let len = 1 + rand(60_000) as usize;
                    let offset = rand((size - len) as u64) as usize;
                    let class = if rand(2) == 0 { Class::Prefetch } else { Class::Demand };
                    reader.submit(read(&reader, next, class, offset as u64, len)).unwrap();
                    asked.insert(next, (offset, len, None));
                    live.push(next);
                    next += 1;
                }
                5 if !live.is_empty() => {
                    let key = live[rand(live.len() as u64) as usize];
                    if !done.contains_key(&key) {
                        let outcome = reader.cancel(key).unwrap();
                        asked.get_mut(&key).unwrap().2 = Some(outcome);
                    }
                }
                6 if !live.is_empty() => {
                    let key = live[rand(live.len() as u64) as usize];
                    if !done.contains_key(&key) {
                        let _ = reader.promote(key).unwrap();
                    }
                }
                7 => hold = if hold.is_some() { None } else { Some(reader.hold()) },
                _ => {}
            }
            batch.clear();
            reader.poll(&mut batch);
            for c in batch.drain(..) {
                assert!(done.insert(c.key, c).is_none(), "completed twice");
            }
            if next == total && hold.is_some() && rand(4) == 0 {
                hold = None;
            }
            if next == total && hold.is_none() {
                // Everything is submitted and running: wait for the rest.
                if let Some(c) = reader.wait_any(Duration::from_millis(20)) {
                    assert!(done.insert(c.key, c).is_none(), "completed twice");
                }
            }
        }
        drop(hold);

        let (mut ok, mut cancelled) = (0, 0);
        for (key, c) in &done {
            let (offset, len, cancel) = asked[key];
            match (&c.result, cancel) {
                (Ok(()), Some(Cancel::Cancelled)) => panic!("read {key}: cancelled, yet delivered"),
                (Ok(()), _) => {
                    assert_eq!(c.data(), &data[offset..offset + len], "read {key}");
                    ok += 1;
                }
                (Err(Error::Cancelled), Some(Cancel::Cancelled)) => {
                    assert!(c.data().is_empty());
                    cancelled += 1;
                }
                (Err(e), cancel) => panic!("read {key}: {e} (cancel said {cancel:?})"),
            }
            assert!(c.buffer.len() >= len, "the buffer came back");
        }
        let stats = reader.stats();
        assert_eq!(ok + cancelled, total);
        assert_eq!(stats.cancelled, cancelled);
        assert_eq!(stats.demand.requests + stats.prefetch.requests, total);
        assert_eq!(stats.demand.completed + stats.prefetch.completed, total);
        assert_eq!((stats.outstanding, stats.in_flight), (0, 0));
        assert!(stats.peak_in_flight <= 8);
        report(&format!(
            "random: {} -> {ok} delivered, {cancelled} cancelled, {} promoted",
            setup.name(),
            stats.promoted
        ));
    }
}

#[test]
fn waiting_times_out_and_names_unknown_keys() {
    let setup = &setups("wait")[0];
    let data = pattern(100_000, 13);
    let path = setup.dir.file("data.bin", &data);
    let reader = open_reporting(setup, &[path], setup.config(), "wait");

    assert!(matches!(reader.wait(42, PATIENCE), Err(Error::UnknownKey(42))));
    assert!(reader.wait_any(Duration::from_millis(5)).is_none());

    let hold = reader.hold();
    reader.submit(read(&reader, 1, Class::Demand, 0, 100)).unwrap();
    let t = Instant::now();
    assert!(reader.wait(1, Duration::from_millis(30)).unwrap().is_none());
    assert!(t.elapsed() >= Duration::from_millis(30));
    assert!(reader.wait_any(Duration::ZERO).is_none());
    let mut out = Vec::new();
    assert_eq!(reader.poll(&mut out), 0);

    // A waiter in another thread is woken by the completion, not by its timeout.
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| {
            let t = Instant::now();
            let c = reader.wait(1, Duration::from_secs(20)).unwrap().expect("finished");
            (t.elapsed(), c)
        });
        std::thread::sleep(Duration::from_millis(50));
        drop(hold);
        let (waited, c) = waiter.join().unwrap();
        assert!(waited < Duration::from_secs(10), "woken by the completion, waited {waited:?}");
        assert_eq!(c.data(), &data[..100]);
    });

    reader.submit(read(&reader, 2, Class::Prefetch, 0, 100)).unwrap();
    let c = reader.wait_any(PATIENCE).expect("any completion");
    assert_eq!(c.key, 2);
}

#[test]
fn statistics_describe_the_traffic() {
    let setup = &setups("stats")[0];
    let data = pattern(4 << 20, 14);
    let path = setup.dir.file("data.bin", &data);
    let reader = open_reporting(setup, &[path], setup.config(), "stats");
    assert_eq!(reader.stats(), Default::default());

    let mut keys = Vec::new();
    for i in 0..20u64 {
        reader.submit(read(&reader, i, Class::Demand, i * 10_000 + 7, 1000)).unwrap();
        keys.push(i);
    }
    for i in 20..24u64 {
        reader
            .submit(read(&reader, i, Class::Prefetch, (i - 20) * (512 << 10), 512 << 10))
            .unwrap();
        keys.push(i);
    }
    let mid = reader.stats();
    assert_eq!((mid.demand.requests, mid.prefetch.requests), (20, 4));
    collect(&reader, &keys);

    let stats = reader.stats();
    assert_eq!(stats.demand.requests, 20);
    assert_eq!(stats.demand.completed, 20);
    assert_eq!(stats.demand.bytes, 20_000);
    assert!(stats.demand.device_bytes >= 20_000);
    assert_eq!(stats.prefetch.bytes, 4 * (512 << 10));
    assert_eq!((stats.cancelled, stats.promoted, stats.outstanding), (0, 0, 0));
    assert_eq!((stats.queued_demand, stats.queued_prefetch, stats.in_flight), (0, 0, 0));
    let l = stats.demand_latency;
    assert_eq!((l.total, l.samples), (20, 20));
    assert!(Duration::ZERO < l.p50 && l.p50 <= l.p95 && l.p95 <= l.p99 && l.p99 <= l.max, "{l:?}");
    assert_eq!(stats.prefetch_latency.total, 4);
}

#[test]
fn memory_lent_to_the_reader_receives_the_bytes_in_place() {
    for setup in setups("lent") {
        let data = pattern(200_000, 15);
        let path = setup.dir.file("data.bin", &data);
        let reader = open_reporting(&setup, &[path], setup.config(), "lent");

        // A ring of slots in one allocation, as a pinned staging area is.
        let slot = 64 * 1024;
        let mut ring = Buffer::aligned(4 * slot, 4096.max(reader.buffer_alignment()));
        let base = ring.as_mut_ptr();
        let offsets = [1000u64, 70_001, 130_000, 5];
        for (i, offset) in offsets.iter().enumerate() {
            // SAFETY: slot `i` is a disjoint part of `ring`, which outlives
            // every read below and is not touched while they run.
            let buffer = unsafe { Buffer::from_raw(base.add(i * slot), slot) };
            let mut r = read(&reader, i as u64, Class::Demand, *offset, 50_000);
            r.buffer = buffer;
            reader.submit(r).unwrap();
        }
        for (i, c) in collect(&reader, &[0, 1, 2, 3]).into_iter().enumerate() {
            assert!(c.result.is_ok(), "{:?}", c.result);
            let offset = offsets[i] as usize;
            assert_eq!(c.data(), &data[offset..offset + 50_000]);
            // The data sits where the span says, inside the lent slot.
            let at = c.skip + i * slot;
            assert_eq!(&ring.as_slice()[at..at + 50_000], &data[offset..offset + 50_000]);
        }
    }
}

#[test]
fn dropping_the_reader_with_reads_outstanding_waits_for_the_device_and_frees_cleanly() {
    for setup in setups("drop") {
        let path = setup.dir.file("data.bin", &pattern(8 << 20, 16));
        let mut backing = Buffer::aligned(8 << 20, 4096);
        {
            let reader = open_reporting(&setup, &[path], setup.config(), "drop");
            // SAFETY: `backing` outlives the reader (declared before it) and
            // is not touched until the reader is gone.
            let lent = unsafe { Buffer::from_raw(backing.as_mut_ptr(), 8 << 20) };
            let mut r = read(&reader, 1, Class::Prefetch, 0, 8 << 20);
            r.buffer = lent;
            reader.submit(r).unwrap();
            for key in 2..20 {
                reader.submit(read(&reader, key, Class::Prefetch, 0, 1 << 20)).unwrap();
            }
            until("some chunks to run", || reader.stats().peak_in_flight > 0);
        }
        // Reaching this line means the I/O thread ended and no read is still
        // writing into `backing` (a write after this point would be a data
        // race the sanitizers would catch; the memory is ours again).
        assert_eq!(backing.len(), 8 << 20);
    }
}

#[test]
fn open_reports_files_that_cannot_be_opened_and_bad_settings() {
    let setup = &setups("open")[0];
    let missing = setup.dir.path.join("missing.bin");
    match Reader::open(&[&missing], Config::default()) {
        Err(e @ Error::Open { .. }) => assert!(e.to_string().contains("missing.bin"), "{e}"),
        other => panic!("{other:?}"),
    }
    let path = setup.dir.file("data.bin", b"x");
    for config in [
        Config { queue_depth: 0, ..Config::default() },
        Config { prefetch_depth: 99, ..Config::default() },
        Config { prefetch_chunk: 0, ..Config::default() },
    ] {
        assert!(matches!(Reader::open(&[&path], config), Err(Error::Config(_))));
    }
    // An empty list of files is fine: such a reader just has nothing to read.
    let empty: [&std::path::Path; 0] = [];
    let reader = Reader::open(&empty, Config::default()).unwrap();
    assert!(reader.files().is_empty());
    assert_eq!(reader.buffer_alignment(), 1);
}
