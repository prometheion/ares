//! Row reads of a table against files written by the GGUF builder.

mod common;

use ares_core::GgmlType;
use ares_gguf::{Gguf, GgufBuilder};
use ares_io::{Class, Error, LayoutError, Reader, RowGather, RowReader, RowTable};
use common::*;

const ROWS: u64 = 3000;
const ROW_BYTES: usize = 90; // 45 f16 values, as the hashed tables of qwen4exp

fn table_bytes() -> &'static [u8] {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| (0..ROWS).flat_map(|r| pattern(ROW_BYTES, r + 1)).collect())
}

/// A model with a small tensor in front, so that the table does not start on
/// a block, and the table itself.
fn write_model(dir: &TestDir) -> std::path::PathBuf {
    let b = GgufBuilder::new()
        .str("general.architecture", "testrows")
        .tensor_f32("front.weight", &[7], &[1.0; 7])
        .tensor("table.weight", &[45, ROWS], GgmlType::F16, table_bytes().to_vec());
    dir.file("model.gguf", &b.build())
}

fn gather(
    reader: &Reader,
    rows: &mut RowReader,
    ids: &[u64],
    key_base: u64,
    out: &mut RowGather,
) -> usize {
    rows.start(reader, ids, key_base, Class::Demand, out).unwrap();
    let n_reads = out.keys().count();
    while !out.is_done() {
        let c = reader.wait_any(std::time::Duration::from_secs(20)).expect("a read finishes");
        assert!(rows.accept(out, c).is_none(), "the completion is the gather's");
    }
    out.result().unwrap();
    n_reads
}

fn expect(out: &RowGather, ids: &[u64]) {
    assert_eq!(out.n_rows(), ids.len());
    assert_eq!(out.rows().len(), ids.len() * ROW_BYTES);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(out.row(i), &table_bytes()[*id as usize * ROW_BYTES..][..ROW_BYTES], "row {id}");
    }
}

#[test]
fn rows_come_back_right_for_duplicate_adjacent_and_far_apart_ids() {
    for setup in setups("rows") {
        for cache_budget in [0usize, 1 << 20] {
            let gguf = Gguf::open(write_model(&setup.dir)).unwrap();
            let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
            report(&format!(
                "rows: {} cache={cache_budget} -> {}",
                setup.name(),
                reader.files()[0].mode
            ));
            let table =
                RowTable::for_reader(&reader, gguf.tensor("table.weight").unwrap()).unwrap();
            assert_eq!((table.row_bytes(), table.n_rows()), (ROW_BYTES, ROWS));
            let mut rows = RowReader::new(&reader, table, cache_budget);
            let mut out = RowGather::new();

            // The same row three times: one read.
            let ids = [17, 17, 17];
            assert_eq!(gather(&reader, &mut rows, &ids, 100, &mut out), 1);
            expect(&out, &ids);

            // Adjacent rows (they share blocks and touch): one read.
            let ids = [100, 101, 102, 103];
            assert_eq!(gather(&reader, &mut rows, &ids, 200, &mut out), 1);
            expect(&out, &ids);

            // Rows far apart, in no order, one of them at the very end and
            // one at the very start of the table.
            let ids = [ROWS - 1, 0, 1500, 745, 1];
            let n_reads = gather(&reader, &mut rows, &ids, 300, &mut out);
            assert!((3..=ids.len()).contains(&n_reads), "{n_reads} reads");
            expect(&out, &ids);

            // Rows across a block boundary, in both directions.
            let ids = [44, 45, 46, 90, 91, 134, 135, 136];
            gather(&reader, &mut rows, &ids, 400, &mut out);
            expect(&out, &ids);

            // Everything in one gather, shuffled, with repeats.
            let mut ids: Vec<u64> = (0..ROWS).chain(0..50).collect();
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            for i in (1..ids.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ids.swap(i, (state % (i as u64 + 1)) as usize);
            }
            let n_reads = gather(&reader, &mut rows, &ids, 10_000, &mut out);
            // Without a cache the whole table is a couple of long reads; with
            // one, the rows cached by the gathers above split the runs.
            let most = if cache_budget == 0 { 3 } else { 24 };
            assert!(n_reads <= most, "{n_reads} reads for the whole table");
            expect(&out, &ids);

            // Nothing at all.
            assert_eq!(gather(&reader, &mut rows, &[], 20_000, &mut out), 0);
            assert!(out.is_done() && out.rows().is_empty());
        }
    }
}

#[test]
fn the_cache_serves_repeated_rows_without_reading() {
    let setup = &setups("rows-cache")[0];
    let gguf = Gguf::open(write_model(&setup.dir)).unwrap();
    let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
    let table = RowTable::for_reader(&reader, gguf.tensor("table.weight").unwrap()).unwrap();
    let mut rows = RowReader::new(&reader, table, 1 << 20);
    let mut out = RowGather::new();

    let ids = [5, 900, 2500, 901];
    gather(&reader, &mut rows, &ids, 0, &mut out);
    expect(&out, &ids);
    let requests = reader.stats().demand.requests;
    assert!(requests >= 3);
    assert_eq!(rows.cache().unwrap().len(), 4);

    // The same rows again, and a mixture of cached and new rows.
    assert_eq!(gather(&reader, &mut rows, &ids, 100, &mut out), 0, "all served from the cache");
    expect(&out, &ids);
    assert_eq!(reader.stats().demand.requests, requests);
    assert!(out.is_done());
    let ids = [900, 1234, 5, 1235];
    let n = gather(&reader, &mut rows, &ids, 200, &mut out);
    assert_eq!(n, 1, "only the two new, adjacent rows are read");
    expect(&out, &ids);
    let stats = rows.cache().unwrap().stats();
    assert_eq!(stats.hits, 4 + 2);
    assert_eq!(stats.evictions, 0);
}

#[test]
fn a_small_cache_evicts_and_still_returns_the_right_rows() {
    let setup = &setups("rows-small-cache")[0];
    let gguf = Gguf::open(write_model(&setup.dir)).unwrap();
    let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
    let table = RowTable::for_reader(&reader, gguf.tensor("table.weight").unwrap()).unwrap();
    // Room for 8 rows.
    let mut rows = RowReader::new(&reader, table, 8 * (ROW_BYTES + 25));
    assert_eq!(rows.cache().unwrap().capacity(), 8);
    let mut out = RowGather::new();
    for round in 0..20u64 {
        let ids: Vec<u64> = (0..12).map(|i| (round * 7 + i * 211) % ROWS).collect();
        gather(&reader, &mut rows, &ids, round * 100, &mut out);
        expect(&out, &ids);
    }
    let stats = rows.cache().unwrap().stats();
    assert!(stats.evictions > 0 && rows.cache().unwrap().len() == 8, "{stats:?}");
}

#[test]
fn two_gathers_can_be_in_flight_and_completions_find_their_owner() {
    let setup = &setups("rows-two")[0];
    let gguf = Gguf::open(write_model(&setup.dir)).unwrap();
    let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
    let table = RowTable::for_reader(&reader, gguf.tensor("table.weight").unwrap()).unwrap();
    let mut rows = RowReader::new(&reader, table, 0);
    let (mut a, mut b) = (RowGather::new(), RowGather::new());
    let ids_a = [10, 2000, 1000];
    let ids_b = [2999, 3, 1500, 1501];
    rows.start(&reader, &ids_a, 0, Class::Demand, &mut a).unwrap();
    rows.start(&reader, &ids_b, 1000, Class::Prefetch, &mut b).unwrap();
    while !(a.is_done() && b.is_done()) {
        let c = reader.wait_any(std::time::Duration::from_secs(20)).unwrap();
        // Offer the completion to one gather; if it is not theirs, to the other.
        if let Some(c) = rows.accept(&mut a, c) {
            assert!(rows.accept(&mut b, c).is_none(), "it is one of the two");
        }
    }
    expect(&a, &ids_a);
    expect(&b, &ids_b);
}

#[test]
fn bad_row_ids_and_failed_reads_are_reported_without_harm() {
    let setup = &setups("rows-bad")[0];
    let gguf = Gguf::open(write_model(&setup.dir)).unwrap();
    let reader = Reader::open_gguf(&gguf, setup.config()).unwrap();
    let table = RowTable::for_reader(&reader, gguf.tensor("table.weight").unwrap()).unwrap();
    let mut rows = RowReader::new(&reader, table, 1 << 16);
    let mut out = RowGather::new();

    let err = rows.start(&reader, &[1, ROWS], 0, Class::Demand, &mut out).unwrap_err();
    assert!(
        matches!(err, Error::Layout(LayoutError::RowOutOfRange { row: 3000, rows: 3000 })),
        "{err}"
    );
    assert_eq!(reader.stats().demand.requests, 0, "nothing was read for a bad request");

    // The gather is reusable afterwards.
    let ids = [1, 2, 3];
    gather(&reader, &mut rows, &ids, 0, &mut out);
    expect(&out, &ids);

    // A key that is already in use is refused and the reads queued before it
    // are cancelled; the gather can be started again.
    let hold = reader.hold();
    reader.submit(read(&reader, 51, Class::Demand, 0, 10)).unwrap();
    let err =
        rows.start(&reader, &[10, 2000, 1000, 2900], 50, Class::Demand, &mut out).unwrap_err();
    assert!(matches!(err, Error::DuplicateKey(51)), "{err}");
    drop(hold);
    // Every outstanding read comes back, the cancelled ones as such, and none
    // of them is the failed gather's.
    while reader.stats().outstanding > 0 {
        let c = reader.wait_any(std::time::Duration::from_secs(30)).expect("a read finishes");
        assert!(rows.accept(&mut out, c).is_some(), "the failed gather owns nothing");
    }
    let ids = [10, 2000];
    gather(&reader, &mut rows, &ids, 500, &mut out);
    expect(&out, &ids);
}

#[test]
fn a_tensor_that_is_not_a_table_is_refused() {
    let setup = &setups("rows-3d")[0];
    let b = GgufBuilder::new().tensor(
        "cube.weight",
        &[16, 2, 4],
        GgmlType::F32,
        vec![0; 16 * 2 * 4 * 4],
    );
    let gguf = Gguf::open(setup.dir.file("cube.gguf", &b.build())).unwrap();
    let err = RowTable::new(gguf.tensor("cube.weight").unwrap(), 512).unwrap_err();
    assert!(matches!(err, LayoutError::NotATable { dims: 3, .. }), "{err}");
}
