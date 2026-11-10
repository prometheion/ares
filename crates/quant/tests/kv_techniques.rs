//! Measurement of the techniques that were candidates for the KV formats `q7`
//! to `q4`, and of the codec as built.
//!
//! Run it with `cargo test -p ares-quant --test kv_techniques -- --nocapture`.
//! For every kind of row and every width it prints the signal-to-noise ratio of
//! encode + decode and three measures of what attention does with a row: the
//! error of the dot products `q.k`, the same with the part that all keys share
//! removed (a softmax does not see a shift of all logits), and the error of a
//! weighted sum of rows (what attention takes of V). It prints them for each
//! candidate technique alone and combined, then a summary of the gain of each
//! candidate over the codec the formats had before the techniques, which is what
//! the choice of the ladder in the module documentation of `kv` rests on.
//!
//! The numbers are not asserted. The tests check only what must hold whatever
//! the numbers are: that the experimental codec here and the library are the
//! same codec where they should be (to the bit), and that a search that includes
//! the old scale among its candidates cannot have a larger squared error than the
//! old scale.
//!
//! **There is no real model data yet.** The rows are synthetic (see
//! `tests/common/mod.rs`). They have the shapes that quantizers are known to
//! struggle with, but how often and how strongly the K and V rows of real models
//! show them is not known, so every choice made from these numbers must be
//! measured again on activations of the real models. Which kinds count as
//! "realistic" below is a judgement, marked in the `realistic` field of the
//! kinds, and the candidates are kept in one table (`candidates`), so that
//! revisiting is a matter of changing a line.
//!
//! The experimental codec is written separately from the library, with the
//! techniques switchable, and works on floats without packing bits. Its
//! arithmetic is the library's, so that the candidate that is the ladder gives
//! the same bits.

mod common;

use ares_quant::{KvFormat, decode_row, encode_row, f16_to_f32, f32_to_f16, rotate_groups};
use common::{
    Rng, channel_rows, heavy_tailed_row, mean_shifted_row, mildly_shifted_row, offset_rows,
    realistic_row, rows, spike_row,
};

const GROUP: usize = 32;
const WIDTHS: [u32; 5] = [8, 7, 6, 5, 4];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rotation {
    /// No rotation at all: the quantizer applied to the values as they are.
    None,
    /// The Walsh-Hadamard transform of each group of 32 (the formats).
    Group,
    /// One transform over each block of this many values; 256 is the whole row.
    Block(usize),
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Offset {
    None,
    /// The mean of each group of the original values, stored as f16 and added
    /// back after un-rotating. At read time: one term per group, the mean times
    /// the sum of the query over the group.
    GroupMean,
    /// The same for the whole row: one f16 per row.
    RowMean,
    /// The midrange of the rotated group, stored as f16: an asymmetric grid.
    /// At read time: one term per group, the offset times the sum of the
    /// rotated query over the group.
    GroupMidrange,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Sign {
    /// The scale is positive: the extra code (if any) is on the negative side.
    Positive,
    /// The scale is negative when the larger extreme of the group is positive,
    /// so that the extra code is on the side of the larger extreme.
    ByExtreme,
    /// Both signs are searched.
    Both,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Search {
    /// From the absolute maximum (or the two extremes), as the formats were.
    AbsMax,
    /// Over the scales for the ratios `(64 - j) / 64`, `j = 0 ..= 2^(8-B)`, of
    /// the largest magnitude, keeping the candidate with the least squared
    /// error; with `fit`, each candidate has its scale replaced by the one that
    /// fits its codes best in the least-squares sense.
    Squares { fit: bool },
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    name: &'static str,
    rotation: Rotation,
    /// Allow the code -2^(B-1), which the symmetric clamp of the old formats
    /// never writes.
    full_range: bool,
    sign: Sign,
    search: Search,
    offset: Offset,
    /// Extra bits per value this costs in a row of 256 values.
    extra_bits: f32,
    /// Not simulated: the library's `encode_row` and `decode_row`.
    library: bool,
}

const BASE: Candidate = Candidate {
    name: "before",
    rotation: Rotation::Group,
    full_range: false,
    sign: Sign::Positive,
    search: Search::AbsMax,
    offset: Offset::None,
    extra_bits: 0.0,
    library: false,
};

/// The techniques of the ladder, as the formats `q7` to `q4` have them.
const LADDER: Candidate = Candidate {
    name: "the ladder",
    full_range: true,
    sign: Sign::ByExtreme,
    search: Search::Squares { fit: true },
    ..BASE
};

const ROW_MEAN_BITS: f32 = 16.0 / 256.0;

fn candidates() -> Vec<Candidate> {
    let with = |name, c: Candidate| Candidate { name, ..c };
    let ladder = LADDER;
    vec![
        with("no rotation", Candidate { rotation: Rotation::None, ..BASE }),
        BASE,
        // Write-only, in the order of what each adds.
        with("full code range", Candidate { full_range: true, ..BASE }),
        with(
            "full range + sign of scale",
            Candidate { full_range: true, sign: Sign::ByExtreme, ..BASE },
        ),
        with("search by squares", Candidate { search: Search::Squares { fit: false }, ..BASE }),
        with("search + fitted scale", Candidate { search: Search::Squares { fit: true }, ..BASE }),
        with("... + full range", Candidate { sign: Sign::Positive, ..ladder }),
        ladder,
        with("the ladder, both signs", Candidate { sign: Sign::Both, ..ladder }),
        Candidate { name: "the ladder as built", library: true, ..ladder },
        // A different rotation.
        with("row rotation", Candidate { rotation: Rotation::Block(256), ..BASE }),
        with("block 64 + ladder", Candidate { rotation: Rotation::Block(64), ..ladder }),
        with("block 128 + ladder", Candidate { rotation: Rotation::Block(128), ..ladder }),
        with("block 256 + ladder", Candidate { rotation: Rotation::Block(256), ..ladder }),
        // One extra term at read time, with the formats' scale.
        with("group mean", Candidate { offset: Offset::GroupMean, extra_bits: 0.5, ..BASE }),
        with("row mean", Candidate { offset: Offset::RowMean, extra_bits: ROW_MEAN_BITS, ..BASE }),
        with(
            "midrange offset",
            Candidate { offset: Offset::GroupMidrange, extra_bits: 0.5, ..BASE },
        ),
        // And with the ladder.
        with(
            "ladder + group mean",
            Candidate { offset: Offset::GroupMean, extra_bits: 0.5, ..ladder },
        ),
        with(
            "ladder + row mean",
            Candidate { offset: Offset::RowMean, extra_bits: ROW_MEAN_BITS, ..ladder },
        ),
        with(
            "ladder + midrange",
            Candidate { offset: Offset::GroupMidrange, extra_bits: 0.5, ..ladder },
        ),
    ]
}

// ---------------------------------------------------------------------------
// The experimental codec
// ---------------------------------------------------------------------------

fn through_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// The Walsh-Hadamard transform of a block of a power-of-two length, in place,
/// orthonormal (its own inverse).
fn wht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let c = 1.0 / (n as f32).sqrt();
    for v in x {
        *v *= c;
    }
}

fn rotate(rotation: Rotation, x: &mut [f32]) {
    match rotation {
        Rotation::None => {}
        Rotation::Group => rotate_groups(x).unwrap(),
        Rotation::Block(block) => x.chunks_mut(block).for_each(wht),
    }
}

/// The sum of 32 values, in the order of the specification: distance 16 first,
/// then 8, 4, 2, 1.
fn sum32(v: &[f32]) -> f32 {
    let mut v: [f32; GROUP] = v.try_into().unwrap();
    let mut half = GROUP / 2;
    while half > 0 {
        for i in 0..half {
            v[i] += v[i + half];
        }
        half /= 2;
    }
    v[0]
}

/// The smallest positive f16 not below `q`, saturating at the largest finite one
/// (the scale rule of the specification).
fn scale_up(q: f32) -> f32 {
    let nearest = f32_to_f16(q);
    if nearest >= 0x7c00 {
        return f16_to_f32(0x7bff);
    }
    let up = if nearest == 0 || f16_to_f32(nearest) < q { nearest + 1 } else { nearest };
    f16_to_f32(up.min(0x7bff))
}

/// Quantizes one group (already rotated, offset removed) and returns the values
/// it decodes to.
fn quantize_group(t: &[f32], bits: u32, c: &Candidate) -> Vec<f32> {
    let qmax = ((1 << (bits - 1)) - 1) as f32;
    let low = if c.full_range { -qmax - 1.0 } else { -qmax };
    let pos = t.iter().fold(0.0f32, |m, v| m.max(*v));
    let neg = t.iter().fold(0.0f32, |m, v| m.max(-v));
    let amax = pos.max(neg);
    if amax == 0.0 {
        return vec![0.0; t.len()];
    }
    // The signs of the scale to try. With a positive scale the extra code is on
    // the negative side; with a negative one on the positive side.
    let signs: &[f32] = match (c.sign, c.full_range) {
        (Sign::Positive, _) | (_, false) => &[1.0],
        (Sign::ByExtreme, true) => {
            if neg >= pos {
                &[1.0]
            } else {
                &[-1.0]
            }
        }
        (Sign::Both, true) => &[1.0, -1.0],
    };
    let codes_for = |g: f32| -> Vec<f32> {
        t.iter().map(|v| (v / g).round_ties_even().clamp(low, qmax)).collect()
    };
    let decode = |codes: &[f32], s: f32| -> Vec<f32> { codes.iter().map(|c| c * s).collect() };
    let error = |decoded: &[f32]| -> f32 {
        let squares: Vec<f32> = t.iter().zip(decoded).map(|(a, b)| (a - b) * (a - b)).collect();
        sum32(&squares)
    };

    let mut best: Vec<f32> = Vec::new();
    let mut best_error = f32::INFINITY;
    let mut first = true;
    for &sign in signs {
        match c.search {
            Search::AbsMax => {
                let q = if c.full_range {
                    if sign > 0.0 {
                        (pos / qmax).max(neg / (qmax + 1.0))
                    } else {
                        (pos / (qmax + 1.0)).max(neg / qmax)
                    }
                } else {
                    amax / qmax
                };
                let s = sign * scale_up(q);
                let decoded = decode(&codes_for(s), s);
                let e = error(&decoded);
                if first || e < best_error {
                    (best, best_error, first) = (decoded, e, false);
                }
            }
            Search::Squares { fit } => {
                for j in 0..=1u32 << (8 - bits) {
                    let candidate = amax * ((64 - j) as f32 / 64.0);
                    let base = scale_up(candidate / qmax);
                    let codes = codes_for(sign * base);
                    let mut s = sign * base;
                    if fit {
                        let products: Vec<f32> = t.iter().zip(&codes).map(|(v, c)| v * c).collect();
                        let n: f32 = codes.iter().map(|c| c * c).sum();
                        let p = sum32(&products);
                        if n > 0.0 {
                            // Never rounded to zero (which a code other than
                            // zero rules out anyway).
                            let fitted = through_f16((p.abs() / n).min(65504.0));
                            s = sign * fitted.max(f16_to_f32(1));
                        } else {
                            s = base;
                        }
                    }
                    let decoded = decode(&codes, s);
                    let e = error(&decoded);
                    if first || e < best_error {
                        (best, best_error, first) = (decoded, e, false);
                    }
                }
            }
        }
    }
    best
}

/// Encode and decode `x`, as the candidate does.
fn round_trip(c: &Candidate, bits: u32, x: &[f32]) -> Vec<f32> {
    if c.library {
        return library_round_trip(bits, x);
    }
    let n = x.len();
    let mut y = x.to_vec();
    let mut offsets = vec![0.0f32; n / GROUP];
    match c.offset {
        Offset::GroupMean => {
            for (g, group) in y.chunks_mut(GROUP).enumerate() {
                let m = through_f16(group.iter().sum::<f32>() / GROUP as f32);
                group.iter_mut().for_each(|v| *v -= m);
                offsets[g] = m;
            }
        }
        Offset::RowMean => {
            let m = through_f16(y.iter().sum::<f32>() / n as f32);
            y.iter_mut().for_each(|v| *v -= m);
            offsets.fill(m);
        }
        Offset::None | Offset::GroupMidrange => {}
    }
    rotate(c.rotation, &mut y);
    let mut decoded = Vec::with_capacity(n);
    for group in y.chunks(GROUP) {
        if c.offset == Offset::GroupMidrange {
            let max = group.iter().fold(f32::MIN, |m, v| m.max(*v));
            let min = group.iter().fold(f32::MAX, |m, v| m.min(*v));
            let z = through_f16((max + min) / 2.0);
            let shifted: Vec<f32> = group.iter().map(|v| v - z).collect();
            decoded.extend(quantize_group(&shifted, bits, c).into_iter().map(|v| v + z));
        } else {
            decoded.extend(quantize_group(group, bits, c));
        }
    }
    rotate(c.rotation, &mut decoded);
    for (group, m) in decoded.chunks_mut(GROUP).zip(offsets) {
        group.iter_mut().for_each(|v| *v += m);
    }
    decoded
}

fn library_round_trip(bits: u32, x: &[f32]) -> Vec<f32> {
    let format = KvFormat::quantized(bits).unwrap();
    let mut bytes = vec![0; format.row_bytes(x.len()).unwrap()];
    encode_row(format, x, &mut bytes).unwrap();
    let mut y = vec![0.0; x.len()];
    decode_row(format, &bytes, &mut y).unwrap();
    y
}

// ---------------------------------------------------------------------------
// Data and metrics
// ---------------------------------------------------------------------------

/// A kind of row: rows to store and rows to use as queries, from the same
/// distribution (so a query is loud where the keys are loud).
struct Kind {
    name: &'static str,
    /// Counts for the averages "on the realistic rows": a judgement. The first
    /// five are what real K and V rows are known or expected to look like; a
    /// single large value among small ones, and a large mean in every group, are
    /// stress cases that real rows show rarely if at all.
    realistic: bool,
    keys: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

const KEYS: usize = 192;
const QUERIES: usize = 24;

fn kinds(n: usize) -> Vec<Kind> {
    let make = |seed: u64, name, realistic, generate: &dyn Fn(&mut Rng, usize) -> Vec<Vec<f32>>| {
        let mut rng = Rng(seed);
        let mut all = generate(&mut rng, KEYS + QUERIES);
        let queries = all.split_off(KEYS);
        Kind { name, realistic, keys: all, queries }
    };
    vec![
        make(101, "Gaussian with outliers", true, &|r, k| rows(r, k, n, realistic_row)),
        make(102, "a few loud channels", true, &|r, k| channel_rows(r, k, n)),
        make(103, "heavy-tailed", true, &|r, k| rows(r, k, n, heavy_tailed_row)),
        make(106, "fixed channel offsets", true, &|r, k| offset_rows(r, k, n)),
        make(107, "small non-zero mean", true, &|r, k| rows(r, k, n, mildly_shifted_row)),
        make(104, "single large value", false, &|r, k| rows(r, k, n, spike_row)),
        make(105, "large non-zero mean", false, &|r, k| rows(r, k, n, mean_shifted_row)),
    ]
}

/// 10 log10 of signal power over error power.
fn db(signal: f64, noise: f64) -> f64 {
    10.0 * (signal / noise.max(1e-300)).log10()
}

struct Measured {
    /// Signal-to-noise ratio of the stored rows.
    snr: f64,
    /// Error of the dot products `q.k` over all queries and keys, in dB below
    /// the dot products themselves.
    dot: f64,
    /// The same with the common part removed per query: for each query the mean
    /// over the keys of the dot product and of its error is subtracted first,
    /// because a softmax over the keys does not see a shift of all logits.
    dot_centered: f64,
    /// Error of a weighted sum of stored rows (weights summing to 1, the rows
    /// attention takes of V), in dB below the weighted sums. Errors the rows
    /// have in common add up in it instead of cancelling.
    weighted: f64,
}

fn measure(kind: &Kind, decoded: &[Vec<f32>]) -> Measured {
    let (mut signal, mut noise) = (0.0f64, 0.0f64);
    for (x, y) in kind.keys.iter().zip(decoded) {
        for (a, b) in x.iter().zip(y) {
            signal += f64::from(*a).powi(2);
            noise += (f64::from(*a) - f64::from(*b)).powi(2);
        }
    }

    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter().zip(b).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()
    };
    let (mut dot_signal, mut dot_noise) = (0.0f64, 0.0f64);
    let (mut centered_signal, mut centered_noise) = (0.0f64, 0.0f64);
    for q in &kind.queries {
        let logits: Vec<f64> = kind.keys.iter().map(|x| dot(q, x)).collect();
        let errors: Vec<f64> =
            kind.keys.iter().zip(decoded).map(|(x, y)| dot(q, y) - dot(q, x)).collect();
        let (mean_logit, mean_error) = (
            logits.iter().sum::<f64>() / logits.len() as f64,
            errors.iter().sum::<f64>() / errors.len() as f64,
        );
        for (l, e) in logits.iter().zip(&errors) {
            dot_signal += l * l;
            dot_noise += e * e;
            centered_signal += (l - mean_logit).powi(2);
            centered_noise += (e - mean_error).powi(2);
        }
    }

    // Weighted sums of 24 rows with peaked random weights, 48 of them.
    let mut rng = Rng(0x57ee7);
    let (mut sum_signal, mut sum_noise) = (0.0f64, 0.0f64);
    for _ in 0..48 {
        let picks: Vec<(usize, f64)> = (0..24)
            .map(|_| {
                let u = f64::from(rng.uniform());
                (rng.below(kind.keys.len()), u * u + 1e-3)
            })
            .collect();
        let total: f64 = picks.iter().map(|(_, w)| w).sum();
        let len = kind.keys[0].len();
        let (mut exact, mut approx) = (vec![0.0f64; len], vec![0.0f64; len]);
        for &(row, w) in &picks {
            for (e, v) in exact.iter_mut().zip(&kind.keys[row]) {
                *e += w / total * f64::from(*v);
            }
            for (a, v) in approx.iter_mut().zip(&decoded[row]) {
                *a += w / total * f64::from(*v);
            }
        }
        for (e, a) in exact.iter().zip(&approx) {
            sum_signal += e * e;
            sum_noise += (a - e).powi(2);
        }
    }
    Measured {
        snr: db(signal, noise),
        dot: db(dot_signal, dot_noise),
        dot_centered: db(centered_signal, centered_noise),
        weighted: db(sum_signal, sum_noise),
    }
}

fn mse(x: &[Vec<f32>], y: &[Vec<f32>]) -> f64 {
    let n: usize = x.iter().map(Vec::len).sum();
    let sum: f64 = x
        .iter()
        .zip(y)
        .flat_map(|(a, b)| a.iter().zip(b))
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum();
    sum / n as f64
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

/// One measured candidate on one kind of row, at every width.
struct Cell {
    measured: Vec<Measured>,
    /// Mean squared error per width.
    mse: Vec<f64>,
}

/// All numbers of one row length, `[kind][candidate]`.
fn table(kinds: &[Kind], candidates: &[Candidate]) -> Vec<Vec<Cell>> {
    kinds
        .iter()
        .map(|kind| {
            candidates
                .iter()
                .map(|c| {
                    let mut cell = Cell { measured: Vec::new(), mse: Vec::new() };
                    for bits in WIDTHS {
                        let decoded: Vec<Vec<f32>> =
                            kind.keys.iter().map(|x| round_trip(c, bits, x)).collect();
                        cell.mse.push(mse(&kind.keys, &decoded));
                        cell.measured.push(measure(kind, &decoded));
                    }
                    cell
                })
                .collect()
        })
        .collect()
}

fn widths_header() -> String {
    WIDTHS.map(|b| format!("{:>6}", format!("q{b}"))).join(" ")
}

fn row_of(cell: &Cell, f: impl Fn(&Measured) -> f64) -> String {
    cell.measured.iter().map(|m| format!("{:>6.1}", f(m))).collect::<Vec<_>>().join(" ")
}

fn print_tables(n: usize, kinds: &[Kind], candidates: &[Candidate], cells: &[Vec<Cell>]) {
    println!();
    println!("rows of {n} values, {KEYS} rows per kind, {QUERIES} queries of the same kind");
    for (kind, per_candidate) in kinds.iter().zip(cells) {
        println!();
        println!("{} ({})", kind.name, if kind.realistic { "realistic" } else { "stress" });
        println!(
            "  {:<28} {:>5} | SNR, dB                          | q.k error, dB below q.k          | q.k, common part removed         | weighted sum of rows, dB",
            "", "+bits"
        );
        println!(
            "  {:<28} {:>5} | {} | {} | {} | {}",
            "",
            "",
            widths_header(),
            widths_header(),
            widths_header(),
            widths_header()
        );
        for (c, cell) in candidates.iter().zip(per_candidate) {
            println!(
                "  {:<28} {:>5.2} | {} | {} | {} | {}",
                c.name,
                c.extra_bits,
                row_of(cell, |m| m.snr),
                row_of(cell, |m| m.dot),
                row_of(cell, |m| m.dot_centered),
                row_of(cell, |m| m.weighted),
            );
        }
    }
}

/// Gain over the codec before the techniques, in dB: per candidate and width,
/// the mean over the realistic kinds, the worst of the realistic kinds and the
/// worst of all kinds, for four measures.
fn print_summary(n: usize, kinds: &[Kind], candidates: &[Candidate], cells: &[Vec<Cell>]) {
    let base = candidates.iter().position(|c| c.name == BASE.name).unwrap();
    let realistic: Vec<usize> =
        kinds.iter().enumerate().filter(|(_, k)| k.realistic).map(|(i, _)| i).collect();
    type Pick = fn(&Measured) -> f64;
    let measures: [(&str, Pick); 4] = [
        ("SNR", |m| m.snr),
        ("q.k", |m| m.dot),
        ("q.k, common part removed", |m| m.dot_centered),
        ("weighted sum of rows", |m| m.weighted),
    ];
    for (title, f) in measures {
        println!();
        println!("gain over the codec before the techniques in {title}, rows of {n} values (dB)");
        println!(
            "  {:<28} {:>5} | mean on the realistic kinds | worst of the realistic kinds | worst of all kinds",
            "", "+bits"
        );
        println!(
            "  {:<28} {:>5} | {} | {} | {}",
            "",
            "",
            widths_header(),
            widths_header(),
            widths_header()
        );
        for (ci, c) in candidates.iter().enumerate() {
            let (mut mean, mut worst_realistic, mut worst) = (Vec::new(), Vec::new(), Vec::new());
            for w in 0..WIDTHS.len() {
                let gain = |k: usize| f(&cells[k][ci].measured[w]) - f(&cells[k][base].measured[w]);
                mean.push(realistic.iter().map(|&k| gain(k)).sum::<f64>() / realistic.len() as f64);
                worst_realistic.push(realistic.iter().map(|&k| gain(k)).fold(f64::MAX, f64::min));
                worst.push((0..kinds.len()).map(gain).fold(f64::MAX, f64::min));
            }
            let fmt =
                |v: &[f64]| v.iter().map(|g| format!("{g:>6.1}")).collect::<Vec<_>>().join(" ");
            println!(
                "  {:<28} {:>5.2} | {} | {} | {}",
                c.name,
                c.extra_bits,
                fmt(&mean),
                fmt(&worst_realistic),
                fmt(&worst)
            );
        }
    }
    println!();
    println!("  (one more bit of width is worth about 6 dB: a technique that adds 0.5 bit per");
    println!("   value must gain about 3 dB to pay for itself against the next width)");
}

#[test]
fn measure_the_candidate_techniques() {
    for n in [256, 128] {
        let kinds = kinds(n);
        let candidates = candidates();
        let cells = table(&kinds, &candidates);
        if n == 256 {
            print_tables(n, &kinds, &candidates, &cells);
        }
        print_summary(n, &kinds, &candidates, &cells);

        // What must hold whatever the numbers are: the search has the old scale
        // among its candidates, so it cannot have a larger squared error than
        // the old scale (the rotation preserves that error up to rounding). The
        // search with a fitted scale and the ladder, which only add to what the
        // first candidate is, must not be worse than the old codec either.
        let index = |name: &str| candidates.iter().position(|c| c.name == name).unwrap();
        for (k, kind) in kinds.iter().enumerate() {
            for (w, bits) in WIDTHS.iter().enumerate().filter(|(_, b)| **b < 8) {
                let before = cells[k][index("before")].mse[w];
                for name in ["search by squares", "search + fitted scale", "the ladder"] {
                    let after = cells[k][index(name)].mse[w];
                    assert!(
                        after <= before * (1.0 + 1e-5),
                        "{} q{bits}: {name} {after:e}, before {before:e}",
                        kind.name,
                    );
                }
            }
        }
    }
}

#[test]
fn the_ladder_candidate_is_the_library_to_the_bit_and_the_old_codec_is_q8() {
    // The comparison with the experimental variants is a comparison with the
    // real codec only if the one is the other: the candidate that is the ladder
    // gives the bits of `encode_row` and `decode_row` at q7 to q4, and the old
    // codec those of q8.
    let bits32 = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for kind in &kinds(256) {
        for bits in WIDTHS {
            let reference = if bits == 8 { BASE } else { LADDER };
            for x in &kind.keys {
                let ours = round_trip(&reference, bits, x);
                let theirs = library_round_trip(bits, x);
                assert_eq!(bits32(&ours), bits32(&theirs), "{} q{bits}", kind.name);
            }
        }
    }
}
