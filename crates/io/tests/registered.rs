//! Registered files and registered buffers. This file holds a single test so
//! that the process has exactly one ring to look at in `/proc`.

mod common;

use std::collections::HashMap;

use ares_io::{Buffer, Class, Config, FileId, FixedRegion, Read};
use common::*;

/// The `key: value` lines of `/proc/self/fdinfo/<fd>` of every io_uring
/// descriptor of this process.
fn rings() -> Vec<HashMap<String, String>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir("/proc/self/fd").unwrap().flatten() {
        let Ok(target) = std::fs::read_link(entry.path()) else { continue };
        if target.to_string_lossy() != "anon_inode:[io_uring]" {
            continue;
        }
        let info = std::fs::read_to_string(format!(
            "/proc/self/fdinfo/{}",
            entry.file_name().to_string_lossy()
        ))
        .unwrap();
        found.push(
            info.lines()
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
                .collect(),
        );
    }
    found
}

#[test]
fn files_are_registered_and_buffers_inside_a_region_are_read_in_place() {
    let setup = &setups("registered")[0];
    let data_a = pattern(300_000, 21);
    let data_b = pattern(200_000, 22);
    let a = setup.dir.file("a.bin", &data_a);
    let b = setup.dir.file("b.bin", &data_b);

    // A region of 256 KiB (far below the locked-memory limit); the reads go
    // to slices of it and, for comparison, to an ordinary buffer.
    let mut region = Buffer::aligned(256 << 10, 4096);
    let base = region.as_mut_ptr();
    // SAFETY: `region` is declared before the reader and so outlives it, and
    // it is not moved or freed meanwhile.
    let fixed = unsafe { FixedRegion::new(base, region.len()) };
    let config = Config { fixed_regions: vec![fixed], ..setup.config() };
    let reader = open_reporting(setup, &[a, b], config, "registered");

    let rings = rings();
    assert_eq!(rings.len(), 1, "one ring for one reader");
    assert_eq!(rings[0].get("UserFiles").map(String::as_str), Some("2"), "{:?}", rings[0]);
    assert_eq!(rings[0].get("UserBufs").map(String::as_str), Some("1"), "{:?}", rings[0]);

    let slot = 64 << 10;
    let cases = [(0u16, 5_000u64, 40_000usize), (1, 100_000, 30_000), (0, 250_000, 49_000)];
    for (i, (file, offset, len)) in cases.iter().enumerate() {
        // SAFETY: slice `i` of the region is disjoint from the others and the
        // region outlives the read.
        let buffer = unsafe { Buffer::from_raw(base.add(i * slot), slot) };
        reader
            .submit(Read {
                key: i as u64,
                file: FileId(*file),
                offset: *offset,
                len: *len,
                class: Class::Demand,
                buffer,
            })
            .unwrap();
    }
    // One outside the region.
    reader
        .submit(Read {
            key: 9,
            file: FileId(1),
            offset: 1_234,
            len: 50_000,
            class: Class::Demand,
            buffer: Buffer::aligned(64 << 10, 4096),
        })
        .unwrap();
    for (i, c) in collect(&reader, &[0, 1, 2, 9]).iter().enumerate() {
        assert!(c.result.is_ok(), "{:?}", c.result);
        let (file, offset, len) = cases.get(i).copied().unwrap_or((1, 1_234, 50_000));
        let data = if file == 0 { &data_a } else { &data_b };
        assert_eq!(c.data(), &data[offset as usize..offset as usize + len], "read {i}");
        if i < 3 {
            // The bytes landed in the registered memory itself.
            let at = i * slot + c.skip;
            assert_eq!(&region.as_slice()[at..at + len], c.data());
        }
    }
}
