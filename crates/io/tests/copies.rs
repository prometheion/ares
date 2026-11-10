//! Files with several copies: what is checked at open, that reads return the
//! file's bytes whichever copy serves them, mirrors of a GGUF model, and, when
//! a directory on a second drive is given, that both drives carry the reads.
//!
//! ```sh
//! ARES_TEST_SECOND_DRIVE=/mnt/other/tmp cargo test -p ares-io --test copies -- --nocapture
//! ```
//!
//! The directory must be on another drive than the build directory. The test
//! writes 64 MiB there and removes it again.

mod common;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Class, Config, DriveId, Error, FileId, Reader};
use common::*;

fn scratch(tag: &str) -> TestDir {
    TestDir::new(Path::new(env!("CARGO_TARGET_TMPDIR")), "target dir", tag)
}

/// Takes the completions of `keys`, in that order.
fn finish(reader: &Reader, keys: &[u64]) -> Vec<ares_io::Completion> {
    let wait = |key: &u64| reader.wait(*key, Duration::from_secs(60)).unwrap().expect("in time");
    keys.iter().map(wait).collect()
}

#[test]
fn copies_on_one_drive_count_as_one_drive_and_reads_return_the_bytes() {
    let dir = scratch("same-drive");
    let data = pattern(3 * 1024 * 1024 + 777, 11);
    let a = dir.file("a.bin", &data);
    let b = dir.file("b.bin", &data);
    let other = dir.file("other.bin", &pattern(50_000, 12));

    let reader =
        Reader::open_copies(&[vec![a.clone(), b.clone()], vec![other]], Config::default()).unwrap();
    assert_eq!(reader.drives().len(), 1, "{:?}", reader.drives());
    let files = reader.files();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path, a);
    assert_eq!(files[0].size, data.len() as u64);
    let paths: Vec<&PathBuf> = files[0].copies.iter().map(|c| &c.path).collect();
    assert_eq!(paths, [&a, &b]);
    assert!(files[0].copies.iter().all(|c| c.drive == DriveId(0)));
    assert_eq!(files[1].copies.len(), 1);

    let ranges = [(0u64, 100usize), (4090, 9000), (1 << 20, 1 << 20), (data.len() as u64 - 5, 5)];
    for (key, (offset, len)) in ranges.iter().enumerate() {
        reader.submit(read(&reader, key as u64, Class::Demand, *offset, *len)).unwrap();
    }
    let keys: Vec<u64> = (0..ranges.len() as u64).collect();
    for (done, (offset, len)) in finish(&reader, &keys).iter().zip(ranges) {
        assert!(done.result.is_ok(), "{:?}", done.result);
        assert_eq!(done.data(), &data[offset as usize..offset as usize + len]);
    }
    let stats = reader.drive_stats();
    assert_eq!(stats.len(), 1);
    assert!(stats[0].chunks >= 4 && stats[0].bytes >= (1 << 20));
}

#[test]
fn copies_that_are_not_the_same_bytes_are_refused_at_open() {
    let dir = scratch("differ");
    // Larger than what is compared whole, so that the samples are spread.
    let data = pattern(3 * 1024 * 1024 + 4321, 21);
    let good = dir.file("good.bin", &data);

    let short = dir.file("short.bin", &data[..data.len() - 1]);
    match Reader::open_copies(&[[&good, &short]], Config::default()) {
        Err(Error::CopySize { path, size, other, other_size }) => {
            assert_eq!((path, other), (good.clone(), short));
            assert_eq!((size, other_size), (data.len() as u64, data.len() as u64 - 1));
        }
        other => panic!("expected the sizes to be refused, got {other:?}"),
    }

    // A flipped byte at the start, at the end, and in the middle sample.
    let middle = 8 * ((data.len() as u64 - 65536) / 15) + 100;
    for at in [0u64, data.len() as u64 - 1, middle] {
        let mut changed = data.clone();
        changed[at as usize] ^= 0x40;
        let bad = dir.file("bad.bin", &changed);
        match Reader::open_copies(&[[&good, &bad]], Config::default()) {
            Err(Error::CopyDiffers { path, other, offset }) => {
                assert_eq!((path, other, offset), (good.clone(), bad, at));
            }
            other => panic!("expected the byte at {at} to be found, got {other:?}"),
        }
    }

    // A small file is compared whole.
    let small = pattern(200_000, 22);
    let small_path = dir.file("small.bin", &small);
    let mut changed = small.clone();
    changed[123_456] ^= 1;
    let small_bad = dir.file("small-bad.bin", &changed);
    assert!(matches!(
        Reader::open_copies(&[[&small_path, &small_bad]], Config::default()),
        Err(Error::CopyDiffers { offset: 123_456, .. })
    ));

    // A file needs at least one copy, and a copy that does not exist is named.
    let none: [Vec<PathBuf>; 2] = [vec![good.clone()], vec![]];
    assert!(matches!(Reader::open_copies(&none, Config::default()), Err(Error::NoCopies(1))));
    let missing = dir.path.join("missing.bin");
    match Reader::open_copies(&[[&good, &missing]], Config::default()) {
        Err(Error::Open { path, .. }) => assert_eq!(path, missing),
        other => panic!("expected the missing copy to be named, got {other:?}"),
    }
}

/// A model of one or two files in `dir`; returns the path of its first part.
fn write_model(dir: &TestDir, split: bool) -> PathBuf {
    let head = |count: u16| {
        GgufBuilder::new().str("general.architecture", "test").u16("split.count", count)
    };
    let values: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    if !split {
        return dir.file("model.gguf", &head(1).tensor_f32("a.weight", &[64, 64], &values).build());
    }
    let first = head(2).tensor_f32("a.weight", &[64, 64], &values);
    let second = GgufBuilder::new().tensor_f32("b.weight", &[64, 64], &values);
    let path = dir.file("model-00001-of-00002.gguf", &first.build());
    dir.file("model-00002-of-00002.gguf", &second.build());
    path
}

fn copy_into(from: &Path, to: &Path) {
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

#[test]
fn a_gguf_model_is_opened_with_its_mirrors() {
    for split in [false, true] {
        let dir = scratch("gguf");
        let mirror = scratch("gguf-mirror");
        let model = write_model(&dir, split);
        copy_into(&dir.path, &mirror.path);
        let gguf = Gguf::open(&model).unwrap();
        let parts = if split { 2 } else { 1 };
        assert_eq!(gguf.n_parts(), parts);

        // Without mirrors every part has its one copy.
        let plain = Reader::open_gguf(&gguf, Config::default()).unwrap();
        assert!(plain.files().iter().all(|f| f.copies.len() == 1));
        drop(plain);

        // A mirror directory holds every part under its file name.
        let reader = Reader::open_gguf_mirrored(&gguf, &[&mirror.path], Config::default()).unwrap();
        assert_eq!(reader.files().len(), parts);
        for (part, file) in reader.files().iter().enumerate() {
            let original = gguf.path(part as u16);
            assert_eq!(file.path, original);
            assert_eq!(file.copies.len(), 2);
            assert_eq!(file.copies[1].path, mirror.path.join(original.file_name().unwrap()));
        }
        // A tensor of the last part reads back through the reader.
        let name = if split { "b.weight" } else { "a.weight" };
        let tensor = gguf.tensor(name).expect("the tensor exists");
        let file = FileId(tensor.part);
        let mut read = read(&reader, 1, Class::Demand, tensor.offset, tensor.size as usize);
        read.file = file;
        read.buffer = buffer_for(&reader, file, tensor.offset, tensor.size as usize);
        reader.submit(read).unwrap();
        let done = &finish(&reader, &[1])[0];
        assert!(done.result.is_ok(), "{:?}", done.result);
        assert_eq!(done.data(), gguf.data(tensor));
        drop(reader);

        // The path of a copy works as a mirror for a model of one file only.
        let copy = mirror.path.join(model.file_name().unwrap());
        match Reader::open_gguf_mirrored(&gguf, &[&copy], Config::default()) {
            Ok(reader) => {
                assert!(!split);
                assert_eq!(reader.files()[0].copies[1].path, copy);
            }
            Err(Error::Mirror { mirror, .. }) => {
                assert!(split);
                assert_eq!(mirror, copy);
            }
            Err(other) => panic!("unexpected {other:?}"),
        }

        // A mirror that lacks a part is refused, naming the missing file.
        let empty = scratch("gguf-empty");
        match Reader::open_gguf_mirrored(&gguf, &[&empty.path], Config::default()) {
            Err(Error::Open { path, .. }) => {
                assert_eq!(path, empty.path.join(model.file_name().unwrap()));
            }
            other => panic!("expected the missing part to be named, got {other:?}"),
        }
    }
}

#[test]
fn two_drives_both_carry_the_reads_of_a_mirrored_file() {
    let Some(second) = std::env::var_os("ARES_TEST_SECOND_DRIVE") else {
        report("skipped: set ARES_TEST_SECOND_DRIVE=<directory on another drive> to run");
        return;
    };
    let here = scratch("two-drives");
    let there = TestDir::new(Path::new(&second), "second drive", "two-drives");
    let size = 64 * 1024 * 1024;
    let data = pattern(size, 31);
    let a = here.file("mirrored.bin", &data);
    let b = there.file("mirrored.bin", &data);

    let reader = Reader::open_copies(&[[&a, &b]], Config::default()).unwrap();
    let drives = reader.drives();
    assert_eq!(
        drives.len(),
        2,
        "{} is on the same drive as the build directory: {drives:?}",
        there.path.display()
    );
    report(&format!("two drives: {} and {}", drives[0].name, drives[1].name));
    assert!(reader.all_direct(), "both copies should be read directly: {:?}", reader.files());

    // Reads of 256 KiB all over the file, three rounds. Each round is queued
    // whole before any of it starts, so both drives are filled to their depth.
    let len = 256 * 1024;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let started = Instant::now();
    let mut total = 0u64;
    for round in 0..3u64 {
        let offsets: Vec<u64> = (0..256).map(|_| next() % (size - len) as u64).collect();
        let keys: Vec<u64> = (0..offsets.len() as u64).map(|k| round * 1000 + k).collect();
        let hold = reader.hold();
        for (key, offset) in keys.iter().zip(&offsets) {
            reader.submit(read(&reader, *key, Class::Demand, *offset, len)).unwrap();
        }
        drop(hold);
        for (done, offset) in finish(&reader, &keys).iter().zip(&offsets) {
            assert!(done.result.is_ok(), "{:?}", done.result);
            assert_eq!(done.data(), &data[*offset as usize..*offset as usize + len]);
            total += len as u64;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let stats = reader.drive_stats();
    for (drive, s) in drives.iter().zip(&stats) {
        report(&format!(
            "  {}: {} chunks, {} MiB, peak {} in flight, {} us per MiB and chunk under load",
            drive.name,
            s.chunks,
            s.bytes >> 20,
            s.peak_in_flight,
            s.shared_ns_per_mib / 1000
        ));
    }
    report(&format!("  {:.0} MiB/s over both", total as f64 / seconds / (1 << 20) as f64));
    let delivered: u64 = stats.iter().map(|s| s.bytes).sum();
    assert!(delivered >= total);
    let depth = Config::default().queue_depth;
    for (drive, s) in drives.iter().zip(&stats) {
        assert_eq!(s.peak_in_flight, depth, "{} was not filled to its depth", drive.name);
        assert!(s.bytes > 0 && s.shared_ns_per_mib > 0, "{} delivered nothing", drive.name);
    }
}
