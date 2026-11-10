//! Tests of the stored KV sequence of one head and of attention over it
//! (`ares_quant::kvhead`), through the public API.
//!
//! Every test states one property and builds what it needs itself. The model of
//! what must be stored (`expected_*`) is written from the documentation of the
//! module and the policy, not from the implementation: which cells are exact,
//! which tile is converted at which length, from which values. Attention is
//! compared with `attend_plain` (everything decoded, `f64`), with an exact
//! attention on the unquantized rows written here, and with a deliberately
//! naive `f32` implementation that stands for a kernel.

// `&[0..n]` is a list of one visible range, as the API takes it
#![allow(clippy::single_range_in_vec_init)]

mod common;

use std::ops::Range;

use ares_quant::kvhead::{OUTPUT_TOLERANCE, WEIGHT_TOLERANCE};
use ares_quant::{
    AttnState, KV_TILE_CELLS, KvFormat, KvHead, KvHeadError, KvPolicy, decode_row_rotated,
    encode_row, encode_tile, f16_to_f32, f32_to_f16, kv_bytes_per_cell, rotate_groups,
};
use common::{Rng, offset_rows, realistic_row, rows};

const TILE: usize = KV_TILE_CELLS;
const HD: usize = 64;

// ---------------------------------------------------------------------------
// Data and models
// ---------------------------------------------------------------------------

fn fmt(name: &str) -> KvFormat {
    name.parse().unwrap()
}

fn policy(k: &str, v: &str, first: usize, newest: usize) -> KvPolicy {
    KvPolicy { k: fmt(k), v: fmt(v), exact_first_cells: first, exact_newest_cells: newest }
}

fn pairs() -> Vec<(KvFormat, KvFormat)> {
    KvFormat::ALL.iter().flat_map(|k| KvFormat::ALL.iter().map(move |v| (*k, *v))).collect()
}

/// K rows with a persistent offset per channel (as real keys have) and V rows
/// with outliers, `n` cells of `hd` values.
fn cells(seed: u64, n: usize, hd: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng(seed);
    let k = offset_rows(&mut rng, n, hd);
    let v = rows(&mut rng, n, hd, realistic_row);
    (k, v)
}

fn build(policy: KvPolicy, hd: usize, k: &[Vec<f32>], v: &[Vec<f32>]) -> KvHead {
    let mut head = KvHead::new(policy, hd).unwrap();
    for (k, v) in k.iter().zip(v) {
        head.append(k, v).unwrap();
    }
    head
}

fn through_f16(row: &[f32]) -> Vec<f32> {
    row.iter().map(|x| f16_to_f32(f32_to_f16(x.clamp(-65504.0, 65504.0)))).collect()
}

fn f16_bytes(row: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0; 2 * row.len()];
    encode_row(KvFormat::F16, row, &mut bytes).unwrap();
    bytes
}

fn halves(bytes: &[u8]) -> Vec<f32> {
    let (halves, _) = bytes.as_chunks::<2>();
    halves.iter().map(|h| f16_to_f32(u16::from_le_bytes(*h))).collect()
}

/// What `encode_tile` writes for `rows` (mean, rows).
fn tile_bytes(format: KvFormat, hd: usize, rows: &[Vec<f32>]) -> (Vec<u8>, Vec<u8>) {
    let values: Vec<f32> = rows.iter().flatten().copied().collect();
    let mut mean = vec![0; format.mean_bytes(hd).unwrap()];
    let mut out = vec![0; rows.len() * format.row_bytes(hd).unwrap()];
    encode_tile(format, hd, &values, &mut mean, &mut out).unwrap();
    (mean, out)
}

/// Whether a converted cell of the policy was never exact: no mean, so cells are
/// converted one by one, and no newest window, so a cell is converted when it is
/// appended.
fn never_exact(policy: &KvPolicy) -> bool {
    !policy.has_mean() && policy.exact_newest_cells == 0
}

/// The model of the stored state after `n` cells: panics unless every cell is
/// where the policy puts it, with the bytes the codec gives for the right
/// values.
fn check_state(head: &KvHead, k: &[Vec<f32>], v: &[Vec<f32>], what: &str) {
    let policy = *head.policy();
    let hd = head.head_dim();
    let n = head.len();
    let converted = policy.quantized_cells(n);
    assert_eq!(head.converted(), converted, "{what}: converted range at {n} cells");
    let unit_cells = if policy.has_mean() { TILE } else { 1 };
    for cell in 0..n {
        if converted.contains(&cell) {
            assert!(!head.is_exact(cell), "{what}: cell {cell} of {n} is converted");
            assert!(head.exact(cell).is_none());
            let (unit, i) = head.unit_of(cell).unwrap();
            assert_eq!(unit.first_cell + i, cell);
            assert_eq!(unit.cells, unit_cells);
            assert_eq!(unit.k_row(i).len(), policy.k.row_bytes(hd).unwrap());
            assert_eq!(unit.v_row(i).len(), policy.v.row_bytes(hd).unwrap());
        } else {
            assert!(head.is_exact(cell), "{what}: cell {cell} of {n} is exact");
            assert!(head.unit_of(cell).is_none());
            let (kb, vb) = head.exact(cell).unwrap();
            assert_eq!(kb, f16_bytes(&k[cell]), "{what}: exact K of cell {cell}");
            assert_eq!(vb, f16_bytes(&v[cell]), "{what}: exact V of cell {cell}");
        }
    }
    let mut at = converted.start;
    while at < converted.end {
        let (unit, i) = head.unit_of(at).unwrap();
        assert_eq!((unit.first_cell, i), (at, 0));
        let (kv, vv): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if never_exact(&policy) {
            (vec![k[at].clone()], vec![v[at].clone()])
        } else {
            (
                (at..at + unit_cells).map(|c| through_f16(&k[c])).collect(),
                (at..at + unit_cells).map(|c| through_f16(&v[c])).collect(),
            )
        };
        let (k_mean, k_rows) = tile_bytes(policy.k, hd, &kv);
        let (v_mean, v_rows) = tile_bytes(policy.v, hd, &vv);
        assert_eq!(unit.k_mean, k_mean, "{what}: K mean of the unit at {at}");
        assert_eq!(unit.k_rows, k_rows, "{what}: K rows of the unit at {at}");
        assert_eq!(unit.v_mean, v_mean, "{what}: V mean of the unit at {at}");
        assert_eq!(unit.v_rows, v_rows, "{what}: V rows of the unit at {at}");
        at += unit_cells;
    }
}

// ---------------------------------------------------------------------------
// What is stored
// ---------------------------------------------------------------------------

#[test]
fn every_cell_is_where_the_policy_puts_it_with_the_bytes_of_the_codec_for_every_pair() {
    let n = 700;
    let (k, v) = cells(1, n, HD);
    for (kf, vf) in pairs() {
        let policy = KvPolicy::new(kf, vf);
        let mut head = KvHead::new(policy, HD).unwrap();
        for (len, (kr, vr)) in k.iter().zip(&v).enumerate() {
            head.append(kr, vr).unwrap();
            let len = len + 1;
            if [1, 2, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 640, 700]
                .contains(&len)
            {
                check_state(&head, &k, &v, &format!("{kf}/{vf} default"));
            }
        }
    }
}

#[test]
fn every_cell_is_where_the_policy_puts_it_with_small_windows_and_none() {
    let n = 400;
    let (k, v) = cells(2, n, HD);
    for (kf, vf) in pairs() {
        for (first, newest) in [(0, 0), (5, 0), (0, 9), (3, 20), (130, 130)] {
            let policy =
                KvPolicy { k: kf, v: vf, exact_first_cells: first, exact_newest_cells: newest };
            let mut head = KvHead::new(policy, HD).unwrap();
            for (kr, vr) in k.iter().zip(&v) {
                head.append(kr, vr).unwrap();
                let len = head.len();
                if len.is_multiple_of(29) || len == 1 || len == 128 || len == 129 || len == n {
                    check_state(&head, &k, &v, &format!("{kf}/{vf} first {first} newest {newest}"));
                }
            }
        }
    }
}

#[test]
fn conversion_happens_at_exactly_the_lengths_quantized_cells_gives_also_for_mixed_pairs() {
    let n = 900;
    let (k, v) = cells(3, n, HD);
    for (kf, vf) in pairs() {
        let policy = KvPolicy::new(kf, vf);
        let mut head = KvHead::new(policy, HD).unwrap();
        let mut grew_at = Vec::new();
        let mut before = 0..0;
        for (kr, vr) in k.iter().zip(&v) {
            head.append(kr, vr).unwrap();
            let len = head.len();
            let range = policy.quantized_cells(len);
            assert_eq!(head.converted(), range, "{kf}/{vf} at {len}");
            assert_eq!(head.settled_len(), len);
            if range != before {
                grew_at.push((len, range.clone()));
                before = range;
            }
        }
        // the lengths at which the policy's range changes are the lengths at
        // which the head converted
        let mut expected = Vec::new();
        let mut before = 0..0;
        for len in 1..=n {
            let range = policy.quantized_cells(len);
            if range != before {
                expected.push((len, range.clone()));
                before = range;
            }
        }
        assert_eq!(grew_at, expected, "{kf}/{vf}");
        if policy.has_mean() {
            // the first conversion of a policy with a mean: one full tile past
            // the first window and the newest window
            let first = policy.exact_first_cells.next_multiple_of(TILE);
            let (len, range) = &grew_at[0];
            assert_eq!(range.clone(), first..first + TILE);
            assert_eq!(*len, first + TILE + policy.exact_newest_cells, "{kf}/{vf}");
        }
    }
}

#[test]
fn a_converted_tile_is_encode_tile_of_the_f16_values_and_not_of_the_originals() {
    let n = 3 * TILE + 5;
    let (k, v) = cells(4, n, HD);
    let mut any_differs = false;
    for (kf, vf) in pairs().into_iter().filter(|(a, b)| a.has_mean() || b.has_mean()) {
        let policy = KvPolicy { k: kf, v: vf, exact_first_cells: 0, exact_newest_cells: 0 };
        let head = build(policy, HD, &k, &v);
        assert_eq!(head.converted(), 0..3 * TILE, "{kf}/{vf}");
        assert_eq!(head.units().len(), 3);
        for (t, unit) in head.units().iter().enumerate() {
            let cells = t * TILE..(t + 1) * TILE;
            let from_f16 = |side: &[Vec<f32>]| -> Vec<Vec<f32>> {
                cells.clone().map(|c| through_f16(&side[c])).collect()
            };
            let from_f32 = |side: &[Vec<f32>]| -> Vec<Vec<f32>> {
                cells.clone().map(|c| side[c].clone()).collect()
            };
            let (km, kr) = tile_bytes(kf, HD, &from_f16(&k));
            let (vm, vr) = tile_bytes(vf, HD, &from_f16(&v));
            assert_eq!((&unit.k_mean, &unit.k_rows), (&km, &kr), "{kf}/{vf} tile {t} K");
            assert_eq!((&unit.v_mean, &unit.v_rows), (&vm, &vr), "{kf}/{vf} tile {t} V");
            let (km, kr) = tile_bytes(kf, HD, &from_f32(&k));
            any_differs |= (km, kr) != (unit.k_mean.clone(), unit.k_rows.clone());
        }
    }
    assert!(any_differs, "the originals and their f16 values must give different bytes somewhere");
}

#[test]
fn a_cell_that_the_policy_never_keeps_exact_is_encoded_from_the_f32_row() {
    // values that f16 rounds visibly
    let rows_k: Vec<Vec<f32>> = (0..40)
        .map(|i| (0..HD).map(|j| 1.0 + 0.000_13 * (i * HD + j) as f32 * 0.37).collect())
        .collect();
    let rows_v: Vec<Vec<f32>> =
        rows_k.iter().map(|r| r.iter().map(|x| x * -0.731).collect()).collect();
    for (kf, vf) in [("q8", "q8"), ("q8", "f16"), ("f16", "q8"), ("f16", "f16")] {
        let policy = policy(kf, vf, 0, 0);
        assert!(never_exact(&policy));
        let mut head = KvHead::new(policy, HD).unwrap();
        let mut differs = false;
        for (c, (k, v)) in rows_k.iter().zip(&rows_v).enumerate() {
            head.append(k, v).unwrap();
            assert!(!head.is_exact(c), "{kf}/{vf} cell {c} is never exact");
            assert_eq!(head.converted(), 0..c + 1);
            let (unit, i) = head.unit_of(c).unwrap();
            assert_eq!(i, 0);
            let (_, from_f32) = tile_bytes(policy.k, HD, std::slice::from_ref(k));
            assert_eq!(unit.k_rows, from_f32, "{kf}/{vf} K cell {c}");
            let (_, from_f32) = tile_bytes(policy.v, HD, std::slice::from_ref(v));
            assert_eq!(unit.v_rows, from_f32, "{kf}/{vf} V cell {c}");
            let (_, from_f16) = tile_bytes(policy.k, HD, &[through_f16(k)]);
            differs |= from_f16 != unit.k_rows;
        }
        if kf == "q8" {
            assert!(differs, "the f16 detour must change a q8 row somewhere");
        }
    }
}

#[test]
fn a_cell_of_a_windowed_policy_without_a_mean_is_converted_from_its_f16_values() {
    let n = 60;
    let (k, v) = cells(5, n, HD);
    for (kf, vf) in [("q8", "q8"), ("q8", "f16"), ("f16", "q8")] {
        let policy = policy(kf, vf, 3, 7);
        let head = build(policy, HD, &k, &v);
        assert_eq!(head.converted(), 3..n - 7);
        check_state(&head, &k, &v, &format!("{kf}/{vf}"));
        let (unit, _) = head.unit_of(10).unwrap();
        assert_eq!(unit.cells, 1);
        assert!(unit.k_mean.is_empty() && unit.v_mean.is_empty());
    }
}

#[test]
fn the_stored_bytes_are_the_arithmetic_of_the_policy() {
    let n = 800;
    let (k, v) = cells(6, n, HD);
    for (kf, vf) in pairs() {
        let policy = KvPolicy::new(kf, vf);
        let mut head = KvHead::new(policy, HD).unwrap();
        for (len, (kr, vr)) in k.iter().zip(&v).enumerate() {
            head.append(kr, vr).unwrap();
            let len = len + 1;
            if len % 64 == 0 || len == 129 || len == 512 {
                assert_eq!(
                    head.stored_bytes(),
                    policy.bytes_for_cells(len, 1, HD).unwrap(),
                    "{kf}/{vf} at {len}"
                );
            }
        }
    }
    // and the per-cell sizes it is made of
    assert_eq!(kv_bytes_per_cell(fmt("q4"), fmt("q4"), 1, HD).unwrap(), 2 * 36);
}

#[test]
fn appending_one_at_a_time_equals_pending_cells_settled_in_one_go_or_in_chunks() {
    let n = 700;
    let (k, v) = cells(7, n, HD);
    let cases = [
        KvPolicy::new(fmt("q4"), fmt("q4")),
        KvPolicy::new(fmt("q6"), fmt("q5")),
        KvPolicy::new(fmt("q8"), fmt("q4")),
        KvPolicy::new(fmt("f16"), fmt("q7")),
        policy("q8", "q8", 4, 11),
        policy("q4", "f16", 0, 0),
        policy("q5", "q5", 130, 3),
    ];
    for policy in cases {
        let one_by_one = build(policy, HD, &k, &v);
        let mut at_once = KvHead::new(policy, HD).unwrap();
        for (kr, vr) in k.iter().zip(&v) {
            at_once.append_pending(kr, vr).unwrap();
        }
        assert_eq!(at_once.settled_len(), 0);
        assert!(at_once.converted().is_empty());
        at_once.settle().unwrap();
        assert_eq!(at_once, one_by_one, "{policy:?}");

        let mut chunked = KvHead::new(policy, HD).unwrap();
        for (i, (kr, vr)) in k.iter().zip(&v).enumerate() {
            chunked.append_pending(kr, vr).unwrap();
            if i % 37 == 36 {
                chunked.settle().unwrap();
            }
        }
        chunked.settle().unwrap();
        assert_eq!(chunked, one_by_one, "{policy:?} chunked");
    }
}

#[test]
fn nothing_changes_when_an_append_is_refused() {
    let (k, v) = cells(8, 140, HD);
    let policy = policy("q4", "q5", 0, 0);
    let mut head = build(policy, HD, &k[..130], &v[..130]);
    let before = head.clone();
    let mut bad = k[130].clone();
    bad[7] = f32::NAN;
    assert!(matches!(head.append(&bad, &v[130]), Err(KvHeadError::Codec(_))));
    assert!(matches!(head.append(&k[130], &[0.0; HD + 32]), Err(KvHeadError::RowLength { .. })));
    assert!(matches!(
        head.append_pending(&k[130][..HD - 1], &v[130]),
        Err(KvHeadError::RowLength { .. })
    ));
    let mut bad_v = v[130].clone();
    bad_v[0] = f32::INFINITY;
    assert!(matches!(head.append(&k[130], &bad_v), Err(KvHeadError::Codec(_))));
    assert_eq!(head, before);
    // a q8 cell that is encoded straight from f32 refuses a rotation overflow
    let mut q8 = KvHead::new(policy_q8(), HD).unwrap();
    q8.append(&vec![1.0; HD], &vec![1.0; HD]).unwrap();
    let before = q8.clone();
    assert!(matches!(q8.append(&vec![3.0e38; HD], &vec![1.0; HD]), Err(KvHeadError::Codec(_))));
    assert_eq!(q8, before);
    // head sizes
    for bad in [0, 31, 33, 100] {
        assert_eq!(KvHead::new(policy_q8(), bad).unwrap_err(), KvHeadError::HeadDim(bad));
    }
}

fn policy_q8() -> KvPolicy {
    policy("q8", "q8", 0, 0)
}

#[test]
fn the_edge_lengths_of_a_policy_with_a_mean_and_no_windows() {
    let n = 5 * TILE + 3;
    let (k, v) = cells(9, n, HD);
    let policy = policy("q4", "q4", 0, 0);
    let mut head = KvHead::new(policy, HD).unwrap();
    assert_eq!((head.lead_cells(), head.unit_cells()), (0, TILE));
    assert!(head.is_empty());
    assert_eq!(head.converted(), 0..0);
    for (kr, vr) in k.iter().zip(&v) {
        head.append(kr, vr).unwrap();
        let len = head.len();
        let expected = len / TILE * TILE;
        assert_eq!(head.converted(), if expected == 0 { 0..0 } else { 0..expected }, "len {len}");
        if [1, 127, 128, 129, 255, 256, 257, 639, 640, 643].contains(&len) {
            check_state(&head, &k, &v, "edge");
        }
    }
    assert_eq!(head.units().len(), 5);
    assert_eq!((0..n).filter(|c| head.is_exact(*c)).count(), 3);
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

#[test]
fn cutting_inside_the_exact_cells_is_the_sequence_built_from_the_prefix() {
    let (k, v) = cells(10, 380, HD);
    let policy = KvPolicy::new(fmt("q4"), fmt("q4"));
    let mut head = build(policy, HD, &k, &v);
    assert!(head.converted().is_empty(), "128 first + 256 newest leave nothing to convert at 380");
    assert_eq!(head.shortest(), 0);
    for to in [379, 300, 129, 128, 17, 1, 0] {
        head.truncate(to).unwrap();
        assert_eq!(head, build(policy, HD, &k[..to], &v[..to]), "cut to {to}");
        assert_eq!(head.len(), to);
    }
    // and the sequence grows again from there
    for (kr, vr) in k.iter().zip(&v) {
        head.append(kr, vr).unwrap();
    }
    assert_eq!(head, build(policy, HD, &k, &v));
}

#[test]
fn cutting_a_converted_tile_is_refused_and_changes_nothing_cutting_the_room_above_it_is_not() {
    let n = 700;
    let (k, v) = cells(11, n, HD);
    for (kf, vf) in [("q4", "q4"), ("q6", "q5"), ("q8", "q4"), ("f16", "q7")] {
        let policy = KvPolicy::new(fmt(kf), fmt(vf));
        let head = build(policy, HD, &k, &v);
        let conv = head.converted();
        assert!(!conv.is_empty(), "{kf}/{vf}");
        let shortest = head.shortest();
        assert_eq!(shortest, conv.end + policy.exact_newest_cells);
        assert!(shortest <= n);

        let mut cut = head.clone();
        cut.truncate(shortest).unwrap();
        assert_eq!(
            cut,
            build(policy, HD, &k[..shortest], &v[..shortest]),
            "{kf}/{vf} cut to {shortest}"
        );

        for to in [shortest - 1, conv.end, conv.start + 1, 0] {
            let mut cut = head.clone();
            let error = cut.truncate(to).unwrap_err();
            assert_eq!(
                error,
                KvHeadError::TruncateConverted { to, len: n, shortest },
                "{kf}/{vf} to {to}"
            );
            assert_eq!(cut, head, "a refused cut changes nothing");
        }
        let mut cut = head.clone();
        assert_eq!(cut.truncate(n + 1), Err(KvHeadError::TruncateBeyond { to: n + 1, len: n }));
        assert_eq!(cut, head);
        // equal length is a no-op
        cut.truncate(n).unwrap();
        assert_eq!(cut, head);
    }
}

#[test]
fn right_after_a_conversion_nothing_can_be_cut_and_the_room_grows_with_the_sequence() {
    let (k, v) = cells(12, 800, HD);
    let policy = policy("q4", "q4", 0, 10);
    let mut head = KvHead::new(policy, HD).unwrap();
    let mut room_after_conversion = Vec::new();
    let mut before = 0..0;
    for (kr, vr) in k.iter().zip(&v) {
        head.append(kr, vr).unwrap();
        let room = head.len() - head.shortest();
        if head.converted() != before {
            before = head.converted();
            room_after_conversion.push(room);
        } else if !head.converted().is_empty() {
            assert!(room < TILE, "the room is the cells since the last conversion");
        }
    }
    assert!(room_after_conversion.len() >= 5);
    assert!(room_after_conversion.iter().all(|r| *r == 0));
}

#[test]
fn without_a_mean_a_converted_cell_cannot_be_cut_either() {
    let (k, v) = cells(13, 30, HD);
    let policy = policy("q8", "q8", 2, 4);
    let head = build(policy, HD, &k, &v);
    assert_eq!(head.converted(), 2..26);
    assert_eq!(head.shortest(), 30);
    let mut cut = head.clone();
    assert!(matches!(
        cut.truncate(29),
        Err(KvHeadError::TruncateConverted { to: 29, len: 30, shortest: 30 })
    ));
    cut.truncate(30).unwrap();
    // before anything is converted every cut is fine
    let mut early = build(policy, HD, &k[..5], &v[..5]);
    assert_eq!(early.converted(), 0..0);
    early.truncate(0).unwrap();
}

#[test]
fn draft_cells_appended_pending_can_always_be_cut_but_appended_normally_they_may_not() {
    let (k, v) = cells(14, 700, HD);
    let policy = KvPolicy::new(fmt("q4"), fmt("q4"));
    // the length at which the first tile is converted
    let first_conversion = (1..=700).find(|n| !policy.quantized_cells(*n).is_empty()).unwrap();
    let committed = first_conversion - 3;
    let drafts = 5;

    // pending: the drafts push the sequence past the conversion, but nothing is converted
    let mut head = build(policy, HD, &k[..committed], &v[..committed]);
    for c in committed..committed + drafts {
        head.append_pending(&k[c], &v[c]).unwrap();
    }
    assert_eq!(head.len(), committed + drafts);
    assert!(head.converted().is_empty());
    assert_eq!(head.settled_len(), committed);
    // two accepted, three rejected
    head.truncate(committed + 2).unwrap();
    head.settle().unwrap();
    assert_eq!(head, build(policy, HD, &k[..committed + 2], &v[..committed + 2]));

    // appended normally, the same drafts convert the tile, and cutting them is refused
    let mut head = build(policy, HD, &k[..committed], &v[..committed]);
    for c in committed..committed + drafts {
        head.append(&k[c], &v[c]).unwrap();
    }
    assert!(!head.converted().is_empty());
    assert!(matches!(head.truncate(committed), Err(KvHeadError::TruncateConverted { .. })));

    // accepting all of them by settling equals appending them
    let mut settled = build(policy, HD, &k[..committed], &v[..committed]);
    for c in committed..committed + drafts {
        settled.append_pending(&k[c], &v[c]).unwrap();
    }
    settled.settle().unwrap();
    assert_eq!(settled, head);
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

fn query(seed: u64, hd: usize) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..hd).map(|_| 0.7 * rng.gaussian()).collect()
}

fn scale(hd: usize) -> f32 {
    1.0 / (hd as f32).sqrt()
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max)
}

fn max_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).fold(0.0, f64::max)
}

fn l1_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).sum()
}

/// Attention on rows as they are (`f64`), the weights and the output.
fn exact_attention(
    q: &[f32],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    cells: &[usize],
    scale: f32,
) -> (Vec<f64>, Vec<f64>) {
    let scores: Vec<f64> = cells
        .iter()
        .map(|&c| {
            f64::from(scale)
                * q.iter().zip(&k[c]).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>()
        })
        .collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    let weights: Vec<f64> = exps.iter().map(|e| e / total).collect();
    let mut out = vec![0.0; v[0].len()];
    for (&c, w) in cells.iter().zip(&weights) {
        for (o, x) in out.iter_mut().zip(&v[c]) {
            *o += w * f64::from(*x);
        }
    }
    (weights, out)
}

fn cells_of(ranges: &[Range<usize>]) -> Vec<usize> {
    ranges.iter().flat_map(|r| r.clone()).collect()
}

#[test]
fn the_kernel_way_agrees_with_the_plain_way_for_every_pair_across_tile_and_window_boundaries() {
    let n = 5 * TILE + 40;
    let (k, v) = cells(20, n, HD);
    let q = query(21, HD);
    for (kf, vf) in pairs() {
        for (first, newest) in [(0, 0), (0, 5), (7, 3), (130, 20), (128, 256)] {
            let policy =
                KvPolicy { k: kf, v: vf, exact_first_cells: first, exact_newest_cells: newest };
            let head = build(policy, HD, &k, &v);
            for len in [1, 127, 128, 129, 400, n] {
                let mut head = head.clone();
                if head.truncate(len).is_err() {
                    // the cut is refused where tiles are converted: build the prefix
                    head = build(policy, HD, &k[..len], &v[..len]);
                }
                let selections: Vec<Vec<Range<usize>>> = vec![
                    vec![0..len],
                    vec![len - 1..len],
                    vec![0..len.min(3)],
                    vec![len.saturating_sub(130)..len],
                    vec![
                        0..len.min(5),
                        len.min(120)..len.min(131),
                        len.min(250)..len.min(262),
                        len.min(384)..len.min(385),
                        len.saturating_sub(4)..len,
                    ],
                ];
                for ranges in selections {
                    let ranges: Vec<Range<usize>> = {
                        // keep them sorted and disjoint after the clamping above
                        let mut kept: Vec<Range<usize>> = Vec::new();
                        for r in ranges {
                            if r.start >= r.end {
                                continue;
                            }
                            if kept.last().is_some_and(|p| r.start < p.end) {
                                continue;
                            }
                            kept.push(r);
                        }
                        kept
                    };
                    let kernel = head.attend(&q, scale(HD), &ranges).unwrap();
                    let plain = head.attend_plain(&q, scale(HD), &ranges).unwrap();
                    let context =
                        format!("{kf}/{vf} first {first} newest {newest} len {len} {ranges:?}");
                    assert_eq!(kernel.weights.len(), cells_of(&ranges).len());
                    let scale_out = max_abs(&plain.output).max(1e-9);
                    assert!(
                        max_diff(&kernel.output, &plain.output) <= 2e-6 * scale_out,
                        "{context}: output differs by {}",
                        max_diff(&kernel.output, &plain.output)
                    );
                    assert!(l1_diff(&kernel.weights, &plain.weights) <= 2e-6, "{context}: weights");
                    let total: f64 = kernel.weights.iter().map(|w| f64::from(*w)).sum();
                    assert!((total - 1.0).abs() < 1e-6, "{context}: weights sum to {total}");
                    for (a, b) in kernel.scores.iter().zip(&plain.scores) {
                        assert!(
                            (a - b).abs() <= 1e-6 * (1.0 + a.abs()),
                            "{context}: scores {a} {b}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn attention_over_exact_cells_is_attention_on_the_f16_rows() {
    // short sequences keep everything exact under the default policy
    let n = 120;
    let (k, v) = cells(22, n, HD);
    let q = query(23, HD);
    let k16: Vec<Vec<f32>> = k.iter().map(|r| through_f16(r)).collect();
    let v16: Vec<Vec<f32>> = v.iter().map(|r| through_f16(r)).collect();
    for (kf, vf) in pairs().into_iter().filter(|(a, b)| a.has_mean() || b.has_mean()) {
        let head = build(KvPolicy::new(kf, vf), HD, &k, &v);
        assert!(head.converted().is_empty());
        let got = head.attend(&q, scale(HD), &[0..n]).unwrap();
        let cells: Vec<usize> = (0..n).collect();
        let (weights, out) = exact_attention(&q, &k16, &v16, &cells, scale(HD));
        for (a, b) in got.weights.iter().zip(&weights) {
            assert!((f64::from(*a) - b).abs() < 1e-7, "{kf}/{vf}");
        }
        for (a, b) in got.output.iter().zip(&out) {
            assert!((f64::from(*a) - b).abs() < 1e-6 * (1.0 + b.abs()), "{kf}/{vf}");
        }
    }
}

#[test]
fn quantized_attention_stays_within_a_tolerance_of_exact_attention_per_format() {
    // (format, bound on the relative L2 error of the output, bound on the L1
    // error of the weights); measured values are well below, see the printout
    let bounds = [
        ("q8", 0.012, 0.008),
        ("q7", 0.016, 0.008),
        ("q6", 0.03, 0.015),
        ("q5", 0.06, 0.03),
        ("q4", 0.12, 0.055),
    ];
    let n = 700;
    let (k, v) = cells(24, n, 128);
    let cells_all: Vec<usize> = (0..n).collect();
    for (name, out_bound, weight_bound) in bounds {
        let format = fmt(name);
        // converted: everything but the newest 16 cells
        let policy =
            KvPolicy { k: format, v: format, exact_first_cells: 0, exact_newest_cells: 16 };
        let head = build(policy, 128, &k, &v);
        assert!(head.converted().len() >= 4 * TILE);
        let (mut worst_out, mut worst_weights) = (0.0f64, 0.0f64);
        for seed in 0..8 {
            let q = query(100 + seed, 128);
            let got = head.attend(&q, scale(128), &[0..n]).unwrap();
            let (weights, out) = exact_attention(&q, &k, &v, &cells_all, scale(128));
            let num: f64 =
                got.output.iter().zip(&out).map(|(a, b)| (f64::from(*a) - b).powi(2)).sum();
            let den: f64 = out.iter().map(|b| b * b).sum();
            worst_out = worst_out.max((num / den).sqrt());
            let l1: f64 =
                got.weights.iter().zip(&weights).map(|(a, b)| (f64::from(*a) - b).abs()).sum();
            worst_weights = worst_weights.max(l1);
        }
        println!(
            "{name}: relative L2 error of the output {worst_out:.4}, L1 error of the weights {worst_weights:.4}"
        );
        assert!(worst_out < out_bound, "{name}: output error {worst_out}");
        assert!(worst_weights < weight_bound, "{name}: weights error {worst_weights}");
    }
}

#[test]
fn a_selection_is_attention_over_the_selected_cells_and_dense_is_one_range() {
    let n = 3 * TILE + 70;
    let (k, v) = cells(25, n, HD);
    let q = query(26, HD);
    let policy = policy("q5", "q4", 6, 40);
    let head = build(policy, HD, &k, &v);
    let dense = head.attend(&q, scale(HD), &[0..n]).unwrap();
    assert_eq!(dense.weights.len(), n);
    // the same cells given as several ranges, empty ones in between
    let split = head.attend(&q, scale(HD), &[0..100, 100..100, 100..n]).unwrap();
    assert_eq!(split, dense);

    let selection = [3..9, 120..135, 250..256, 256..300, n - 3..n];
    let got = head.attend(&q, scale(HD), &selection).unwrap();
    let cells = cells_of(&selection);
    // a cell's score does not depend on which others are visible
    for (i, &c) in cells.iter().enumerate() {
        assert_eq!(got.scores[i], dense.scores[c], "score of cell {c}");
    }
    // the weights are the softmax of exactly those scores
    let max = got.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = got.scores.iter().map(|s| (s - max).exp()).sum();
    for (w, s) in got.weights.iter().zip(&got.scores) {
        assert!((f64::from(*w) - (s - max).exp() / total).abs() < 1e-7);
    }
    let plain = head.attend_plain(&q, scale(HD), &selection).unwrap();
    assert!(max_diff(&got.output, &plain.output) <= 2e-6 * max_abs(&plain.output));
}

#[test]
fn a_selection_that_cuts_tiles_adds_the_mean_of_each_tile_with_the_weight_of_its_visible_cells() {
    // every tile is a different constant row, so its rows are all mean and no
    // code: the output is the weights of the tiles times their rows, and a mean
    // added with another tile's weight, or the weight of cells that are not
    // visible, shows
    let tiles = 4;
    let n = tiles * TILE;
    let tile_row = |t: usize, scale: f32| -> Vec<f32> {
        (0..HD).map(|j| scale * (1.0 + t as f32 * 0.7 + (j % 7) as f32 * 0.1)).collect()
    };
    let k: Vec<Vec<f32>> = (0..n).map(|c| tile_row(c / TILE, 0.4)).collect();
    let v: Vec<Vec<f32>> = (0..n).map(|c| tile_row(c / TILE, -1.0)).collect();
    let q = query(27, HD);
    for (kf, vf) in [("q4", "q4"), ("q6", "q8"), ("q8", "q5"), ("q5", "f16")] {
        let head = build(policy(kf, vf, 0, 0), HD, &k, &v);
        assert_eq!(head.converted(), 0..n);
        for selection in [
            vec![5..9],
            vec![0..1, 130..131],
            vec![100..140],
            vec![60..70, 200..210, 300..303, 400..412],
        ] {
            let got = head.attend(&q, scale(HD), &selection).unwrap();
            let (_, expected) = exact_attention(&q, &k, &v, &cells_of(&selection), scale(HD));
            // a format without a mean (q8) puts the large mean of these rows in its
            // codes: a few per cent of the largest value, against steps of 0.7 for a
            // mean taken from the wrong tile
            let largest = expected.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            for (o, x) in got.output.iter().zip(&expected) {
                assert!(
                    (f64::from(*o) - x).abs() < 0.03 * largest,
                    "{kf}/{vf} {selection:?}: {o} against {x}"
                );
            }
        }
    }
}

#[test]
fn the_visible_ranges_must_be_sorted_disjoint_inside_the_sequence_and_not_all_empty() {
    let (k, v) = cells(28, 50, HD);
    let head = build(KvPolicy::new(fmt("q4"), fmt("q4")), HD, &k, &v);
    let q = query(29, HD);
    let s = scale(HD);
    let range_error = |ranges: &[Range<usize>]| match head.attend(&q, s, ranges) {
        Err(KvHeadError::VisibleRange { index, .. }) => index,
        other => panic!("{ranges:?}: {other:?}"),
    };
    #[allow(clippy::reversed_empty_ranges)]
    let inverted = [5..3];
    assert_eq!(range_error(&[10..20, 5..8]), 1);
    assert_eq!(range_error(&[0..10, 9..12]), 1);
    assert_eq!(range_error(&[0..51]), 0);
    assert_eq!(range_error(&[0..5, 40..60]), 1);
    assert_eq!(range_error(&inverted), 0);
    // touching ranges and empty ones are fine
    head.attend(&q, s, &[0..10, 10..20, 20..20, 30..31]).unwrap();
    for ranges in [vec![], vec![0..0], vec![7..7, 9..9]] {
        assert_eq!(head.attend(&q, s, &ranges), Err(KvHeadError::NoVisibleCells));
        assert_eq!(head.attend_plain(&q, s, &ranges), Err(KvHeadError::NoVisibleCells));
    }
    // the query and the scale
    assert!(matches!(head.attend(&q[..HD - 1], s, &[0..5]), Err(KvHeadError::RowLength { .. })));
    let mut bad = q.clone();
    bad[3] = f32::NAN;
    assert!(matches!(head.attend(&bad, s, &[0..5]), Err(KvHeadError::Codec(_))));
    assert_eq!(
        head.attend(&q, f32::INFINITY, &[0..5]),
        Err(KvHeadError::NotFinite { what: "scale" })
    );
    // an empty head has nothing to attend to
    let empty = KvHead::new(KvPolicy::new(fmt("q4"), fmt("q4")), HD).unwrap();
    assert_eq!(empty.attend(&q, s, &[0..0]), Err(KvHeadError::NoVisibleCells));
    assert!(matches!(empty.attend(&q, s, &[0..1]), Err(KvHeadError::VisibleRange { .. })));
}

#[test]
fn a_sequence_of_one_cell_attends_to_itself_with_weight_one() {
    let (k, v) = cells(30, 1, HD);
    let q = query(31, HD);
    for (kf, vf) in pairs() {
        for policy in [KvPolicy::new(kf, vf), policy_with(kf, vf)] {
            let head = build(policy, HD, &k, &v);
            let got = head.attend(&q, scale(HD), &[0..1]).unwrap();
            assert_eq!(got.weights, vec![1.0], "{policy:?}");
            let plain = head.attend_plain(&q, scale(HD), &[0..1]).unwrap();
            assert!(
                max_diff(&got.output, &plain.output) <= 2e-6 * max_abs(&plain.output).max(1e-9)
            );
            if head.is_exact(0) {
                // an exact cell: its f16 row
                for (o, x) in got.output.iter().zip(through_f16(&v[0])) {
                    assert!((o - x).abs() <= 1e-6 * (1.0 + x.abs()), "{policy:?}");
                }
            }
        }
    }
}

fn policy_with(k: KvFormat, v: KvFormat) -> KvPolicy {
    KvPolicy { k, v, exact_first_cells: 0, exact_newest_cells: 0 }
}

#[test]
fn rows_of_256_values_work_the_same_way() {
    let n = 2 * TILE + 30;
    let (k, v) = cells(32, n, 256);
    let q = query(33, 256);
    let head = build(policy("q4", "q5", 3, 10), 256, &k, &v);
    assert_eq!(head.converted(), TILE..2 * TILE);
    let kernel = head.attend(&q, scale(256), &[0..n]).unwrap();
    let plain = head.attend_plain(&q, scale(256), &[0..n]).unwrap();
    assert!(max_diff(&kernel.output, &plain.output) <= 2e-6 * max_abs(&plain.output));
    assert_eq!(head.stored_bytes(), head.policy().bytes_for_cells(n, 1, 256).unwrap());
}

/// The codes of a row of `q4`..`q8` as the specification lays them out: per
/// group of 32, an f16 scale and then the codes in planes (the low four bits two
/// per byte, then the planes of the high bits; one byte per code for `q8`), as
/// two's complement numbers. Returns (scale, codes) per group.
fn unpack_row(bytes: &[u8], bits: u32) -> Vec<(f32, Vec<i32>)> {
    let planes: &[(usize, usize)] = match bits {
        8 => &[(8, 0)],
        7 => &[(4, 0), (2, 4), (1, 6)],
        6 => &[(4, 0), (2, 4)],
        5 => &[(4, 0), (1, 4)],
        _ => &[(4, 0)],
    };
    let group = 2 + 4 * bits as usize;
    bytes
        .chunks_exact(group)
        .map(|g| {
            let scale = f16_to_f32(u16::from_le_bytes([g[0], g[1]]));
            let mut codes = vec![0i32; 32];
            let mut at = 2;
            for &(width, lowest) in planes {
                for (i, code) in codes.iter_mut().enumerate() {
                    for b in 0..width {
                        let n = i * width + b;
                        *code |= i32::from((g[at + n / 8] >> (n % 8)) & 1) << (lowest + b);
                    }
                }
                at += 4 * width;
            }
            let codes = codes.into_iter().map(|raw| (raw << (32 - bits)) >> (32 - bits)).collect();
            (scale, codes)
        })
        .collect()
}

#[test]
fn attention_follows_the_documented_steps_on_the_exposed_bytes() {
    // one group per row: the steps are written out here from the bytes alone
    let hd = 32;
    let n = 2 * TILE + 20;
    let (k, v) = cells(50, n, hd);
    let policy = policy("q5", "q4", 0, 10);
    let head = build(policy, hd, &k, &v);
    assert_eq!(head.converted(), 0..2 * TILE);
    let q = query(51, hd);
    let s = scale(hd);
    let selection = [2..40, 120..140, 250..256, 256..262];
    let cells = cells_of(&selection);

    let mut q_rot = q.clone();
    rotate_groups(&mut q_rot).unwrap();
    let mut scores = Vec::new();
    for &c in &cells {
        let dot = if let Some((kb, _)) = head.exact(c) {
            q.iter().zip(halves(kb)).map(|(a, b)| f64::from(*a) * f64::from(b)).sum::<f64>()
        } else {
            let (unit, i) = head.unit_of(c).unwrap();
            let groups = unpack_row(unit.k_row(i), 5);
            assert_eq!(groups.len(), 1);
            let (scale, codes) = &groups[0];
            let partial: f64 =
                q_rot.iter().zip(codes).map(|(a, c)| f64::from(*a) * f64::from(*c)).sum();
            let mean: f64 = q_rot
                .iter()
                .zip(halves(&unit.k_mean))
                .map(|(a, m)| f64::from(*a) * f64::from(m))
                .sum();
            f64::from(*scale) * partial + mean
        };
        scores.push(f64::from(s) * dot);
    }
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = scores.iter().map(|x| (x - max).exp()).sum();
    let weights: Vec<f64> = scores.iter().map(|x| (x - max).exp() / total).collect();

    let mut plain = vec![0.0f64; hd];
    let mut rotated = vec![0.0f64; hd];
    let mut tile_weight = std::collections::BTreeMap::<usize, f64>::new();
    for (&c, w) in cells.iter().zip(&weights) {
        if let Some((_, vb)) = head.exact(c) {
            for (acc, x) in plain.iter_mut().zip(halves(vb)) {
                *acc += w * f64::from(x);
            }
        } else {
            let (unit, i) = head.unit_of(c).unwrap();
            let (scale, codes) = &unpack_row(unit.v_row(i), 4)[0];
            for (acc, code) in rotated.iter_mut().zip(codes) {
                *acc += w * f64::from(*scale) * f64::from(*code);
            }
            *tile_weight.entry(unit.first_cell).or_default() += w;
        }
    }
    for (first, w) in tile_weight {
        let (unit, _) = head.unit_of(first).unwrap();
        for (acc, m) in rotated.iter_mut().zip(halves(&unit.v_mean)) {
            *acc += w * f64::from(m);
        }
    }
    let mut unrotated: Vec<f32> = rotated.iter().map(|x| *x as f32).collect();
    rotate_groups(&mut unrotated).unwrap();
    let expected: Vec<f32> = plain.iter().zip(&unrotated).map(|(a, b)| *a as f32 + b).collect();

    let got = head.attend(&q, s, &selection).unwrap();
    for (a, b) in got.scores.iter().zip(&scores) {
        assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "score {a} {b}");
    }
    for (a, b) in got.weights.iter().zip(&weights) {
        assert!((f64::from(*a) - b).abs() < 1e-7);
    }
    assert!(
        max_diff(&got.output, &expected) <= 1e-6 * max_abs(&expected),
        "{:?} {:?}",
        got.output,
        expected
    );
}

#[test]
fn an_append_after_pending_cells_settles_them_too() {
    let (k, v) = cells(52, 600, HD);
    let policy = KvPolicy::new(fmt("q6"), fmt("q6"));
    let mut head = KvHead::new(policy, HD).unwrap();
    for c in 0..560 {
        head.append_pending(&k[c], &v[c]).unwrap();
    }
    assert_eq!(head.settled_len(), 0);
    head.append(&k[560], &v[560]).unwrap();
    assert_eq!(head.settled_len(), 561);
    assert_eq!(head, build(policy, HD, &k[..561], &v[..561]));
}

// ---------------------------------------------------------------------------
// States and their merge
// ---------------------------------------------------------------------------

fn state_distance(a: &AttnState, b: &AttnState) -> (f64, f64) {
    let scale = b.v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    let v = a.v.iter().zip(&b.v).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale;
    ((a.lse - b.lse).abs() / (1.0 + b.lse.abs()), v)
}

/// The pieces for a split: consecutive ranges that cover `0..n`, from the
/// cut points.
fn pieces(n: usize, cuts: &[usize]) -> Vec<Range<usize>> {
    let mut at = 0;
    let mut out = Vec::new();
    for &cut in cuts.iter().chain([&n]) {
        if cut > at {
            out.push(at..cut);
            at = cut;
        }
    }
    out
}

/// The states of the pieces merged in four orders: forward, backward, as a
/// balanced tree, and in an order shuffled by `seed`.
fn merged_in_four_orders(
    head: &KvHead,
    q: &[f32],
    pieces: &[Range<usize>],
    seed: u64,
) -> Vec<AttnState> {
    let states: Vec<AttnState> =
        pieces.iter().map(|r| head.state(q, scale(head.head_dim()), r.clone()).unwrap()).collect();
    let n = head.head_dim();
    let forward = AttnState::merge_all(n, &states);
    let backward = AttnState::merge_all(n, states.iter().rev());
    fn tree(states: &[AttnState], n: usize) -> AttnState {
        match states.len() {
            0 => AttnState::empty(n),
            1 => states[0].clone(),
            len => tree(&states[..len / 2], n).merge(&tree(&states[len / 2..], n)),
        }
    }
    let balanced = tree(&states, n);
    let mut rng = Rng(seed);
    let mut order: Vec<usize> = (0..states.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let shuffled = AttnState::merge_all(n, order.iter().map(|&i| &states[i]));
    vec![forward, backward, balanced, shuffled]
}

#[test]
fn any_split_of_the_visible_cells_merges_to_the_state_of_the_whole() {
    let n = 3 * TILE + 70;
    let (k, v) = cells(60, n, HD);
    let q = query(61, HD);
    for (kf, vf) in
        [("q4", "q4"), ("q6", "q5"), ("q8", "q4"), ("f16", "q7"), ("q5", "f16"), ("q8", "q8")]
    {
        let policy = policy(kf, vf, 6, 40);
        let head = build(policy, HD, &k, &v);
        let conv = head.converted();
        if policy.has_mean() {
            assert!(!conv.is_empty(), "{kf}/{vf}");
        }
        let whole = head.state(&q, scale(HD), 0..n).unwrap();
        let tile_cuts: Vec<usize> = (1..=n / TILE).map(|t| t * TILE).collect();
        let page = |size: usize| -> Vec<usize> { (1..n / size + 1).map(|p| p * size).collect() };
        let splits: Vec<(&str, Vec<Range<usize>>)> = vec![
            ("per tile", pieces(n, &tile_cuts)),
            ("pages of 16", pieces(n, &page(16))),
            ("pages of 64", pieces(n, &page(64))),
            ("exact cells apart from the converted ones", pieces(n, &[conv.start, conv.end])),
            ("two halves", pieces(n, &[n / 2])),
            ("two halves cutting a tile", pieces(n, &[TILE + 37])),
            ("single cells at the ends", pieces(n, &[1, 2, n - 2, n - 1])),
            (
                "random",
                pieces(n, &{
                    let mut rng = Rng(62);
                    let mut cuts: Vec<usize> = (0..9).map(|_| 1 + rng.below(n - 1)).collect();
                    cuts.sort();
                    cuts.dedup();
                    cuts
                }),
            ),
        ];
        for (what, split) in splits {
            for (i, merged) in merged_in_four_orders(&head, &q, &split, 63).iter().enumerate() {
                let (lse, vv) = state_distance(merged, &whole);
                assert!(
                    lse < 1e-13 && vv < 1e-11,
                    "{kf}/{vf} {what} order {i}: lse {lse:e}, v {vv:e}"
                );
            }
        }
    }
}

#[test]
fn a_split_gives_the_same_output_as_the_plain_way_over_all_cells() {
    let n = 2 * TILE + 90;
    let (k, v) = cells(64, n, HD);
    let q = query(65, HD);
    for (kf, vf) in [("q4", "q4"), ("q5", "q6"), ("q8", "q5")] {
        let head = build(policy(kf, vf, 3, 17), HD, &k, &v);
        let plain = head.attend_plain(&q, scale(HD), &[0..n]).unwrap();
        for split in
            [pieces(n, &[100]), pieces(n, &[16, 32, 48, 64, 128, 130, 256]), pieces(n, &[n - 1])]
        {
            let merged = merged_in_four_orders(&head, &q, &split, 66);
            for state in merged {
                let out: Vec<f32> = state.v.iter().map(|x| *x as f32).collect();
                assert!(
                    max_diff(&out, &plain.output) <= 2e-6 * max_abs(&plain.output),
                    "{kf}/{vf} {split:?}"
                );
            }
        }
    }
}

#[test]
fn attend_is_the_merge_of_the_states_of_its_ranges() {
    let n = 3 * TILE + 20;
    let (k, v) = cells(67, n, HD);
    let q = query(68, HD);
    let head = build(policy("q5", "q4", 6, 25), HD, &k, &v);
    let ranges = [0..10, 100..140, 140..300, 301..n];
    let attention = head.attend(&q, scale(HD), &ranges).unwrap();
    let states: Vec<AttnState> =
        ranges.iter().map(|r| head.state(&q, scale(HD), r.clone()).unwrap()).collect();
    let merged = AttnState::merge_all(HD, &states);
    let expected: Vec<f32> = merged.v.iter().map(|x| *x as f32).collect();
    assert_eq!(attention.output, expected);
    // the weights follow from the merged lse and the scores
    for (w, s) in attention.weights.iter().zip(&attention.scores) {
        assert_eq!(*w, merged.weight(*s) as f32);
    }
    let total: f64 = attention.weights.iter().map(|w| f64::from(*w)).sum();
    assert!((total - 1.0).abs() < 1e-6);
    // a single range is its own state, bit for bit
    let one = head.attend(&q, scale(HD), &[0..n]).unwrap();
    let state = head.state(&q, scale(HD), 0..n).unwrap();
    let expected: Vec<f32> = state.v.iter().map(|x| *x as f32).collect();
    assert_eq!(one.output, expected);
}

#[test]
fn the_state_of_a_prefix_is_computed_once_and_merged_with_each_rows_own_cells() {
    // two rows of a step share the first cells and have different cells of their own
    let n = 3 * TILE + 50;
    let (k, v) = cells(69, n, HD);
    let head = build(policy("q4", "q5", 5, 30), HD, &k, &v);
    let prefix = 0..2 * TILE;
    for (seed, own) in [(70u64, vec![2 * TILE..2 * TILE + 40]), (71, vec![300..310, 400..n])] {
        let q = query(seed, HD);
        let prefix_state = head.state(&q, scale(HD), prefix.clone()).unwrap();
        let mut ranges = vec![prefix.clone()];
        ranges.extend(own.clone());
        let dense = head.attend(&q, scale(HD), &ranges).unwrap();
        let mut state = prefix_state;
        for r in &own {
            state = state.merge(&head.state(&q, scale(HD), r.clone()).unwrap());
        }
        let out: Vec<f32> = state.v.iter().map(|x| *x as f32).collect();
        assert!(max_diff(&out, &dense.output) <= 1e-9 * max_abs(&dense.output));
    }
}

#[test]
fn an_empty_piece_is_the_identity_and_two_empty_pieces_are_empty() {
    let n = 200;
    let (k, v) = cells(72, n, HD);
    let q = query(73, HD);
    let head = build(policy("q4", "q4", 0, 0), HD, &k, &v);
    let empty = head.state(&q, scale(HD), 5..5).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.lse, f64::NEG_INFINITY);
    assert!(empty.v.iter().all(|x| *x == 0.0));
    assert_eq!(empty, AttnState::empty(HD));
    let some = head.state(&q, scale(HD), 3..90).unwrap();
    assert!(!some.is_empty());
    assert_eq!(some.merge(&empty), some);
    assert_eq!(empty.merge(&some), some);
    assert!(empty.merge(&empty).is_empty());
    assert!(AttnState::merge_all(HD, std::iter::empty::<&AttnState>()).is_empty());
    // a range that is out of the sequence is an error, even an empty one past the end
    assert!(matches!(head.state(&q, scale(HD), 0..n + 1), Err(KvHeadError::VisibleRange { .. })));
}

#[test]
fn a_piece_far_below_the_others_adds_nothing_and_makes_no_nan() {
    // states built by hand
    let big = AttnState { lse: 0.0, v: vec![1.0; HD] };
    for far in [-800.0, -2000.0, -1.0e6, -1.0e300] {
        let small = AttnState { lse: far, v: vec![7.0; HD] };
        for merged in [big.merge(&small), small.merge(&big)] {
            assert_eq!(merged, big, "{far}");
        }
    }
    // large lse too
    let a = AttnState { lse: 1.0e6, v: vec![2.0; HD] };
    let b = AttnState { lse: 1.0e6 - 3000.0, v: vec![5.0; HD] };
    let merged = a.merge(&b);
    assert_eq!(merged, a);
    assert!(merged.v.iter().all(|x| x.is_finite()));

    // and from a sequence: a huge scale makes the scores of the pieces differ by thousands
    let n = 2 * TILE;
    let (k, v) = cells(74, n, HD);
    let q = query(75, HD);
    let head = build(policy("q4", "q4", 0, 0), HD, &k, &v);
    let huge = 5000.0f32;
    let first = head.state(&q, huge, 0..TILE).unwrap();
    let second = head.state(&q, huge, TILE..n).unwrap();
    assert!((first.lse - second.lse).abs() > 800.0, "{} {}", first.lse, second.lse);
    let merged = first.merge(&second);
    assert!(merged.lse.is_finite() && merged.v.iter().all(|x| x.is_finite()));
    let (winner, loser) =
        if first.lse > second.lse { (&first, &second) } else { (&second, &first) };
    assert_eq!(merged.lse, winner.lse);
    assert_eq!(merged.v, winner.v);
    assert!(loser.lse < winner.lse);
    // attend over everything with such a scale: finite, the weights sum to one
    let got = head.attend(&q, huge, &[0..n]).unwrap();
    assert!(got.output.iter().all(|x| x.is_finite()));
    assert!(got.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    let total: f64 = got.weights.iter().map(|w| f64::from(*w)).sum();
    assert!((total - 1.0).abs() < 1e-6, "{total}");
    // and it is the plain way's answer
    let plain = head.attend_plain(&q, huge, &[0..n]).unwrap();
    assert!(max_diff(&got.output, &plain.output) <= 1e-5 * max_abs(&plain.output));
}

#[test]
fn merging_is_commutative_exactly_and_associative_up_to_rounding() {
    let n = 4 * TILE;
    let (k, v) = cells(76, n, HD);
    let q = query(77, HD);
    let head = build(policy("q6", "q6", 0, 0), HD, &k, &v);
    let s: Vec<AttnState> = [0..90, 90..200, 200..400, 400..n]
        .iter()
        .map(|r| head.state(&q, scale(HD), r.clone()).unwrap())
        .collect();
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(s[i].merge(&s[j]), s[j].merge(&s[i]), "commutative {i} {j}");
        }
    }
    let left = s[0].merge(&s[1]).merge(&s[2]);
    let right = s[0].merge(&s[1].merge(&s[2]));
    let (lse, v) = state_distance(&left, &right);
    assert!(lse < 1e-14 && v < 1e-13, "{lse:e} {v:e}");
}

// ---------------------------------------------------------------------------
// The tolerance of an f32 kernel
// ---------------------------------------------------------------------------

/// What a kernel might do for one piece: every row decoded with the codec, the
/// dot products and the weighted sums in `f32`, a sequential or a blocked sum,
/// `f32` exp, the softmax within the piece. Returns the scores, the `lse` and
/// the normalised V sum of the piece.
fn naive_f32_piece(
    head: &KvHead,
    q: &[f32],
    scale: f32,
    range: Range<usize>,
    blocked: bool,
) -> (Vec<f32>, f32, Vec<f32>) {
    let hd = head.head_dim();
    let policy = *head.policy();
    let mut q_rot = q.to_vec();
    if policy.k.is_rotated() {
        rotate_groups(&mut q_rot).unwrap();
    }
    let sum = |terms: &mut dyn Iterator<Item = f32>| -> f32 {
        if blocked {
            let mut blocks = [0.0f32; 8];
            for (i, t) in terms.enumerate() {
                blocks[i % 8] += t;
            }
            blocks.iter().sum()
        } else {
            terms.fold(0.0, |a, t| a + t)
        }
    };
    let mut scores = Vec::new();
    let mut tile_mean_dot = (usize::MAX, 0.0f32);
    let mut row = vec![0.0f32; hd];
    for cell in range.clone() {
        let s = if let Some((kb, _)) = head.exact(cell) {
            let k = halves(kb);
            sum(&mut q.iter().zip(&k).map(|(a, b)| a * b))
        } else {
            let (unit, i) = head.unit_of(cell).unwrap();
            decode_row_rotated(policy.k, unit.k_row(i), &mut row).unwrap();
            let mut s = if policy.k.is_rotated() {
                sum(&mut q_rot.iter().zip(&row).map(|(a, b)| a * b))
            } else {
                sum(&mut q.iter().zip(&row).map(|(a, b)| a * b))
            };
            if policy.k.has_mean() {
                if tile_mean_dot.0 != unit.first_cell {
                    let mean = halves(&unit.k_mean);
                    tile_mean_dot =
                        (unit.first_cell, sum(&mut q_rot.iter().zip(&mean).map(|(a, b)| a * b)));
                }
                s += tile_mean_dot.1;
            }
            s
        };
        scores.push(s * scale);
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total = sum(&mut exps.iter().copied());
    let lse = max + total.ln();
    let weights: Vec<f32> = exps.iter().map(|e| e / total).collect();
    let mut plain = vec![0.0f32; hd];
    let mut rotated = vec![0.0f32; hd];
    let mut unit_weight = 0.0f32;
    let mut unit_first: Option<usize> = None;
    let flush = |rotated: &mut Vec<f32>, unit_first: Option<usize>, w: f32| {
        if let (Some(first), true) = (unit_first, policy.v.has_mean()) {
            let (unit, _) = head.unit_of(first).unwrap();
            for (acc, m) in rotated.iter_mut().zip(halves(&unit.v_mean)) {
                *acc += w * m;
            }
        }
    };
    for (cell, &w) in range.zip(&weights) {
        if let Some((_, vb)) = head.exact(cell) {
            for (acc, x) in plain.iter_mut().zip(halves(vb)) {
                *acc += w * x;
            }
        } else {
            let (unit, i) = head.unit_of(cell).unwrap();
            decode_row_rotated(policy.v, unit.v_row(i), &mut row).unwrap();
            let target = if policy.v.is_rotated() { &mut rotated } else { &mut plain };
            for (acc, x) in target.iter_mut().zip(&row) {
                *acc += w * x;
            }
            if policy.v.is_rotated() {
                if unit_first != Some(unit.first_cell) {
                    flush(&mut rotated, unit_first, unit_weight);
                    unit_weight = 0.0;
                }
                unit_first = Some(unit.first_cell);
                unit_weight += w;
            }
        }
    }
    flush(&mut rotated, unit_first, unit_weight);
    if policy.v.is_rotated() {
        rotate_groups(&mut rotated).unwrap();
    }
    let v = plain.iter().zip(&rotated).map(|(a, b)| a + b).collect();
    (scores, lse, v)
}

/// The merge of two `f32` states, as a kernel would do it.
fn merge_f32(a: &(f32, Vec<f32>), b: &(f32, Vec<f32>)) -> (f32, Vec<f32>) {
    let m = a.0.max(b.0);
    let (wa, wb) = ((a.0 - m).exp(), (b.0 - m).exp());
    let total = wa + wb;
    let v = a.1.iter().zip(&b.1).map(|(x, y)| (wa * x + wb * y) / total).collect();
    (m + total.ln(), v)
}

/// A naive `f32` kernel over cells `0..n` in pieces of `page` cells, merged
/// sequentially or as a tree; returns the output and the weights.
fn naive_f32_kernel(
    head: &KvHead,
    q: &[f32],
    scale: f32,
    n: usize,
    page: usize,
    blocked: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut scores = Vec::new();
    let mut states: Vec<(f32, Vec<f32>)> = Vec::new();
    for start in (0..n).step_by(page) {
        let (s, lse, v) = naive_f32_piece(head, q, scale, start..(start + page).min(n), blocked);
        scores.extend(s);
        states.push((lse, v));
    }
    let merged = if blocked {
        fn tree(states: &[(f32, Vec<f32>)]) -> (f32, Vec<f32>) {
            if states.len() == 1 {
                return states[0].clone();
            }
            merge_f32(&tree(&states[..states.len() / 2]), &tree(&states[states.len() / 2..]))
        }
        tree(&states)
    } else {
        states[1..].iter().fold(states[0].clone(), |acc, s| merge_f32(&acc, s))
    };
    let weights = scores.iter().map(|s| (s - merged.0).exp()).collect();
    (merged.1, weights)
}

#[test]
fn a_naive_f32_kernel_stays_within_the_documented_tolerances_with_room_to_spare() {
    let n = 2000;
    let (k, v) = cells(40, n, 256);
    let (mut worst_out, mut worst_weights) = (0.0f64, 0.0f64);
    for (kf, vf) in [("q4", "q4"), ("q6", "q5"), ("q8", "q8"), ("f16", "q4"), ("q5", "f16")] {
        let head = build(KvPolicy::new(fmt(kf), fmt(vf)), 256, &k, &v);
        assert!(!head.converted().is_empty());
        for seed in 0..4 {
            let q = query(200 + seed, 256);
            let reference = head.attend(&q, scale(256), &[0..n]).unwrap();
            // whole, in pieces of 128, 64 and 16 cells; sequential sums and a
            // sequential merge, blocked sums and a tree
            for page in [n, 128, 64, 16] {
                for blocked in [false, true] {
                    let (out, weights) = naive_f32_kernel(&head, &q, scale(256), n, page, blocked);
                    let rel_out = max_diff(&out, &reference.output) / max_abs(&reference.output);
                    let l1 = l1_diff(&weights, &reference.weights);
                    worst_out = worst_out.max(rel_out);
                    worst_weights = worst_weights.max(l1);
                    assert!(rel_out <= OUTPUT_TOLERANCE, "{kf}/{vf} page {page}: output {rel_out}");
                    assert!(l1 <= WEIGHT_TOLERANCE, "{kf}/{vf} page {page}: weights {l1}");
                }
            }
        }
    }
    println!(
        "naive f32 kernel, 2000 cells, whole and in pieces: output {worst_out:.2e} (tolerance {OUTPUT_TOLERANCE:.0e}), weights {worst_weights:.2e} (tolerance {WEIGHT_TOLERANCE:.0e})"
    );
    assert!(
        worst_out * 3.0 < OUTPUT_TOLERANCE,
        "the output tolerance must be several times what a naive kernel reaches"
    );
    assert!(
        worst_weights * 3.0 < WEIGHT_TOLERANCE,
        "the weight tolerance must be several times what a naive kernel reaches"
    );
}
