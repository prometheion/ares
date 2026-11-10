//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use ares_io::{Buffer, Class, Completion, Config, FileId, Key, Read, Reader};

/// Prints a line that the test harness does not swallow, so that a plain
/// `cargo test` shows which I/O path each test ran on.
pub fn report(line: &str) {
    let _ = writeln!(std::io::stderr(), "[ares-io] {line}");
}

/// A scratch directory, removed when dropped.
pub struct TestDir {
    pub path: PathBuf,
    /// Where it lives, for the report.
    pub label: &'static str,
}

impl TestDir {
    pub fn new(base: &Path, label: &'static str, tag: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = base.join(format!("ares-io-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self { path, label }
    }

    pub fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// One way to run a test: a directory and whether direct I/O is requested.
/// The directory under `target/` is on the build disk (a real filesystem
/// that normally supports `O_DIRECT`); the system temp directory is often
/// tmpfs, which does not. Requesting buffered reads on the first covers the
/// fallback on a filesystem that could do direct I/O.
pub struct Setup {
    pub dir: TestDir,
    pub direct: bool,
}

impl Setup {
    pub fn config(&self) -> Config {
        Config { direct: self.direct, ..Config::default() }
    }

    pub fn name(&self) -> String {
        format!(
            "{} ({})",
            self.dir.label,
            if self.direct { "direct requested" } else { "buffered requested" }
        )
    }
}

pub fn setups(tag: &str) -> Vec<Setup> {
    let target = Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(target).unwrap();
    vec![
        Setup { dir: TestDir::new(target, "target dir", tag), direct: true },
        Setup { dir: TestDir::new(target, "target dir", tag), direct: false },
        Setup { dir: TestDir::new(&std::env::temp_dir(), "system temp dir", tag), direct: true },
    ]
}

/// Opens `path` and says on stderr how it is read.
pub fn open_reporting(setup: &Setup, paths: &[PathBuf], config: Config, test: &str) -> Reader {
    let reader = Reader::open(paths, config).unwrap();
    let info = &reader.files()[0];
    report(&format!(
        "{test}: {} -> {} (block {}, kernel reports {:?})",
        setup.name(),
        info.mode,
        info.block,
        info.direct_align
    ));
    reader
}

/// Deterministic bytes that are not periodic at any block size.
pub fn pattern(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

pub fn buffer_for(reader: &Reader, file: FileId, offset: u64, len: usize) -> Buffer {
    let span = reader.span(file, offset, len).unwrap();
    Buffer::aligned(span.len, reader.buffer_alignment())
}

pub fn read(reader: &Reader, key: Key, class: Class, offset: u64, len: usize) -> Read {
    Read {
        key,
        file: FileId(0),
        offset,
        len,
        class,
        buffer: buffer_for(reader, FileId(0), offset, len),
    }
}

/// Waits for every key in `keys` and returns the completions in that order.
pub fn collect(reader: &Reader, keys: &[Key]) -> Vec<Completion> {
    keys.iter()
        .map(|key| {
            reader
                .wait(*key, Duration::from_secs(30))
                .unwrap()
                .unwrap_or_else(|| panic!("read {key} did not finish"))
        })
        .collect()
}
