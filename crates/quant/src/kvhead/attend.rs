//! Attention over a [`KvHead`] as composable states: the state of a piece of
//! the sequence ([`KvHead::state`]), the merge of states ([`AttnState::merge`]),
//! the attention over a list of ranges as the merge of the states of its ranges
//! ([`KvHead::attend`]), and the plain way ([`KvHead::attend_plain`]). The
//! arithmetic is described in the module documentation of
//! [`kvhead`](crate::kvhead).

use std::ops::Range;

use super::{ExactCell, KvHead, KvHeadError, KvUnit};
use crate::error::Error;
use crate::half::f16_to_f32;
use crate::kv::{GROUP_VALUES, decode_row_rotated, decode_tile, rotate_groups};

/// How far an `f32` kernel may differ from [`KvHead::attend`], per element of
/// the output, relative to the largest magnitude in the output:
/// `|kernel - reference| <= OUTPUT_TOLERANCE * max |reference|`.
///
/// Calibrated on 2 000 visible cells and rows of 256 values: a naive `f32`
/// implementation (sequential or blocked sums, `f32` exp), whole or in pieces
/// of 16 to 2 000 cells merged in `f32`, reaches 2e-6, which is a tenth of this
/// (see the module documentation, "Arithmetic").
pub const OUTPUT_TOLERANCE: f64 = 2e-5;

/// How far the weights of an `f32` kernel may differ from those of
/// [`KvHead::attend`] in total: `sum |kernel - reference| <= WEIGHT_TOLERANCE`
/// (the weights sum to 1).
pub const WEIGHT_TOLERANCE: f64 = 2e-5;

/// What one query row of one head attends to.
#[derive(Clone, Debug, PartialEq)]
pub struct Attention {
    /// The output row, `head_dim` values.
    pub output: Vec<f32>,
    /// The softmax weights of the visible cells, in the order of the visible
    /// ranges.
    pub weights: Vec<f32>,
    /// The scores before the softmax (`scale * dot`), `f64`, in the same order.
    pub scores: Vec<f64>,
}

/// The attention of one query over a set of cells, as a state that can be
/// merged with the state of another set: the logarithm of the sum of the
/// exponentials of the scores of the cells, and the sum of their V rows weighted
/// by `exp(score - lse)`, that is by their softmax weights within the set. The
/// state of the union of two disjoint sets is [`merge`](Self::merge) of their
/// states, and the output of the attention over a set is the `v` of its state.
///
/// `v` is in the original domain (unrotated, whatever the format of V).
/// A set without cells has `lse = -infinity` and `v = 0`: the identity of the
/// merge.
#[derive(Clone, Debug, PartialEq)]
pub struct AttnState {
    /// `ln(sum(exp(score)))` over the cells of the set; `-infinity` for none.
    pub lse: f64,
    /// `sum(exp(score - lse) * V)`, `head_dim` values.
    pub v: Vec<f64>,
}

impl AttnState {
    /// The state of no cells, for rows of `head_dim` values.
    pub fn empty(head_dim: usize) -> Self {
        AttnState { lse: f64::NEG_INFINITY, v: vec![0.0; head_dim] }
    }

    /// Whether the set has no cell.
    pub fn is_empty(&self) -> bool {
        self.lse == f64::NEG_INFINITY
    }

    /// The state of the union of the two sets of cells, which must be disjoint
    /// and states of the same head size (it panics on two sizes).
    ///
    /// With `m = max(lse_a, lse_b)`, `a = exp(lse_a - m)` and
    /// `b = exp(lse_b - m)`: `lse = m + ln(a + b)` and
    /// `v = (a * v_a + b * v_b) / (a + b)`. One of `a`, `b` is 1, so nothing
    /// overflows; a set whose scores are so far below the other's that its
    /// weight underflows contributes exactly nothing and produces no NaN; an
    /// empty set is skipped, and the merge of one with an empty set is the
    /// other state itself, bit for bit.
    ///
    /// The merge is commutative exactly and associative up to rounding (a few
    /// ulps of `f64` per merge in this reference), so the states of any pieces
    /// can be merged in any order and any tree.
    pub fn merge(&self, other: &AttnState) -> AttnState {
        assert_eq!(self.v.len(), other.v.len(), "states of two head sizes cannot be merged");
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let m = self.lse.max(other.lse);
        let a = (self.lse - m).exp();
        let b = (other.lse - m).exp();
        let sum = a + b;
        let (wa, wb) = (a / sum, b / sum);
        let v = self.v.iter().zip(&other.v).map(|(x, y)| wa * x + wb * y).collect();
        AttnState { lse: m + sum.ln(), v }
    }

    /// The merge of all the states, in the order given (any order gives the
    /// same state up to rounding).
    pub fn merge_all<'a>(
        head_dim: usize,
        states: impl IntoIterator<Item = &'a AttnState>,
    ) -> AttnState {
        states.into_iter().fold(AttnState::empty(head_dim), |acc, s| acc.merge(s))
    }

    /// The softmax weight of a cell of the set that has the score `score`, once
    /// this state is the state of all visible cells.
    pub fn weight(&self, score: f64) -> f64 {
        (score - self.lse).exp()
    }
}

/// The constant of the 32-point rotation of the specification, as `f64`.
const ROTATION_C: f64 = f32::from_bits(0x3e35_04f3) as f64;

/// The rotation of the specification in `f64`, group by group, in place.
fn rotate_groups_f64(values: &mut [f64]) {
    let (groups, _) = values.as_chunks_mut::<GROUP_VALUES>();
    for group in groups {
        let mut h = 1;
        while h < GROUP_VALUES {
            for i in (0..GROUP_VALUES).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (group[j], group[j + h]);
                    group[j] = a + b;
                    group[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for v in group {
            *v *= ROTATION_C;
        }
    }
}

/// The halves of an `f16` row as `f32`.
fn widen(bytes: &[u8]) -> Vec<f32> {
    let (halves, _) = bytes.as_chunks::<2>();
    halves.iter().map(|h| f16_to_f32(u16::from_le_bytes(*h))).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// Where a cell is stored.
enum Place<'a> {
    Exact(&'a ExactCell),
    Packed(&'a KvUnit, usize),
}

/// The query as every piece needs it: as given, rotated once, and the scale.
struct Query<'a> {
    plain: &'a [f32],
    rotated: Vec<f32>,
    scale: f64,
}

impl KvHead {
    fn place(&self, cell: usize) -> Place<'_> {
        match &self.cells[cell] {
            Some(exact) => Place::Exact(exact),
            None => {
                let (unit, i) = self.unit_of(cell).expect("a cell that is not exact is converted");
                Place::Packed(unit, i)
            }
        }
    }

    /// Checks the query and the scale, and prepares the query: rotated once when
    /// K is a rotated format.
    fn query<'a>(&self, query: &'a [f32], scale: f32) -> Result<Query<'a>, KvHeadError> {
        if query.len() != self.head_dim {
            return Err(KvHeadError::RowLength {
                what: "query",
                expected: self.head_dim,
                actual: query.len(),
            });
        }
        if let Some(position) = query.iter().position(|v| !v.is_finite()) {
            let kind = if query[position].is_nan() {
                "NaN"
            } else if query[position] > 0.0 {
                "+infinity"
            } else {
                "-infinity"
            };
            return Err(Error::KvNonFinite { position, kind }.into());
        }
        if !scale.is_finite() {
            return Err(KvHeadError::NotFinite { what: "scale" });
        }
        let mut rotated = query.to_vec();
        if self.policy.k.is_rotated() {
            rotate_groups(&mut rotated)?;
        }
        Ok(Query { plain: query, rotated, scale: f64::from(scale) })
    }

    /// Checks one range against the sequence.
    fn check_range(&self, index: usize, range: &Range<usize>) -> Result<(), KvHeadError> {
        if range.start > range.end || range.end > self.len() {
            return Err(KvHeadError::VisibleRange {
                index,
                start: range.start,
                end: range.end,
                len: self.len(),
            });
        }
        Ok(())
    }

    /// Checks the visible ranges and returns the ones that hold cells.
    fn visible_ranges(&self, visible: &[Range<usize>]) -> Result<Vec<Range<usize>>, KvHeadError> {
        let mut previous_end = 0;
        let mut ranges = Vec::new();
        for (index, range) in visible.iter().enumerate() {
            self.check_range(index, range)?;
            if range.start < previous_end {
                return Err(KvHeadError::VisibleRange {
                    index,
                    start: range.start,
                    end: range.end,
                    len: self.len(),
                });
            }
            previous_end = range.end;
            if !range.is_empty() {
                ranges.push(range.clone());
            }
        }
        if ranges.is_empty() {
            return Err(KvHeadError::NoVisibleCells);
        }
        Ok(ranges)
    }

    /// The scores and the state of one piece, a range of cells: the steps of
    /// the module documentation on those cells alone.
    fn piece(
        &self,
        q: &Query<'_>,
        range: Range<usize>,
    ) -> Result<(Vec<f64>, AttnState), KvHeadError> {
        let n = self.head_dim;
        if range.is_empty() {
            return Ok((Vec::new(), AttnState::empty(n)));
        }
        let (k_format, v_format) = (self.policy.k, self.policy.v);
        let mut scratch = vec![0.0f32; n];

        // scores
        let mut mean_dot: Option<(usize, f64)> = None;
        let mut scores = Vec::with_capacity(range.len());
        for cell in range.clone() {
            let sum = match self.place(cell) {
                Place::Exact(exact) => dot(q.plain, &widen(&exact.k)),
                Place::Packed(unit, i) => {
                    decode_row_rotated(k_format, unit.k_row(i), &mut scratch)?;
                    if k_format.is_rotated() {
                        // scale * sum(q_rot * code) for each group, summed over groups
                        let mut sum = 0.0;
                        let (q_groups, _) = q.rotated.as_chunks::<GROUP_VALUES>();
                        let (row_groups, _) = scratch.as_chunks::<GROUP_VALUES>();
                        for (qg, row) in q_groups.iter().zip(row_groups) {
                            sum += dot(qg, row);
                        }
                        if k_format.has_mean() {
                            // once per tile and piece, however many of its cells are in it
                            let value = match mean_dot {
                                Some((first, value)) if first == unit.first_cell => value,
                                _ => {
                                    let value = dot(&q.rotated, &widen(&unit.k_mean));
                                    mean_dot = Some((unit.first_cell, value));
                                    value
                                }
                            };
                            sum += value;
                        }
                        sum
                    } else {
                        dot(q.plain, &scratch)
                    }
                }
            };
            scores.push(q.scale * sum);
        }

        // the softmax within the piece
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let lse = max + sum.ln();
        let weights: Vec<f64> = exps.iter().map(|e| e / sum).collect();

        // V in the stored domain
        let mut plain = vec![0.0f64; n];
        let mut rotated = vec![0.0f64; n];
        // the unit being summed and the weights of its cells in the piece so far
        let mut current: Option<(&KvUnit, f64)> = None;
        let add_mean = |rotated: &mut [f64], unit: &KvUnit, sum: f64| {
            if v_format.has_mean() {
                for (acc, m) in rotated.iter_mut().zip(widen(&unit.v_mean)) {
                    *acc += sum * f64::from(m);
                }
            }
        };
        for (cell, &weight) in range.clone().zip(&weights) {
            match self.place(cell) {
                Place::Exact(exact) => {
                    for (acc, v) in plain.iter_mut().zip(widen(&exact.v)) {
                        *acc += weight * f64::from(v);
                    }
                }
                Place::Packed(unit, i) => {
                    decode_row_rotated(v_format, unit.v_row(i), &mut scratch)?;
                    let target = if v_format.is_rotated() { &mut rotated } else { &mut plain };
                    for (acc, v) in target.iter_mut().zip(&scratch) {
                        *acc += weight * f64::from(*v);
                    }
                    if v_format.is_rotated() {
                        let same =
                            current.as_ref().is_some_and(|(u, _)| u.first_cell == unit.first_cell);
                        if same {
                            if let Some((_, sum)) = current.as_mut() {
                                *sum += weight;
                            }
                        } else {
                            if let Some((u, sum)) = current.take() {
                                add_mean(&mut rotated, u, sum);
                            }
                            current = Some((unit, weight));
                        }
                    }
                }
            }
        }
        if let Some((u, sum)) = current.take() {
            add_mean(&mut rotated, u, sum);
        }
        if v_format.is_rotated() {
            rotate_groups_f64(&mut rotated);
        }
        let v = plain.iter().zip(&rotated).map(|(a, b)| a + b).collect();
        Ok((scores, AttnState { lse, v }))
    }

    /// The state of one piece of the sequence, a range of cells, for one query:
    /// the attention over just those cells, ready to be merged with the states
    /// of other pieces ([`AttnState::merge`]). The range may be any range of
    /// cells, exact cells and tiles mixed, whole tiles or a part of one; an
    /// empty range gives the empty state. See the [module
    /// documentation](crate::kvhead).
    pub fn state(
        &self,
        query: &[f32],
        scale: f32,
        range: Range<usize>,
    ) -> Result<AttnState, KvHeadError> {
        let q = self.query(query, scale)?;
        self.check_range(0, &range)?;
        Ok(self.piece(&q, range)?.1)
    }

    /// Attention of one query row over the visible cells, defined as the merge
    /// of the states of the visible ranges ([`state`](Self::state)), in the
    /// order of the ranges: the query rotated once, exact cells with the plain dot
    /// product, cells of a tile with their codes and the tile's `q_rot . mean`,
    /// V accumulated in the stored domain and un-rotated, per piece. See the
    /// [module documentation](crate::kvhead) for the steps and the arithmetic.
    ///
    /// `query` is `head_dim` finite `f32` values, `scale` multiplies the dot
    /// products (typically `1 / sqrt(head_dim)`), `visible` is a sorted list of
    /// disjoint ranges of cells, which may include empty ranges but must contain
    /// a cell. The weights and scores come in the order of the ranges.
    pub fn attend(
        &self,
        query: &[f32],
        scale: f32,
        visible: &[Range<usize>],
    ) -> Result<Attention, KvHeadError> {
        let ranges = self.visible_ranges(visible)?;
        let q = self.query(query, scale)?;
        let mut state = AttnState::empty(self.head_dim);
        let mut scores = Vec::new();
        for range in ranges {
            let (piece_scores, piece) = self.piece(&q, range)?;
            scores.extend(piece_scores);
            state = state.merge(&piece);
        }
        let weights = scores.iter().map(|s| state.weight(*s) as f32).collect();
        let output = state.v.iter().map(|x| *x as f32).collect();
        Ok(Attention { output, weights, scores })
    }

    /// Every cell decoded to `f32` in the original domain: `[len][head_dim]` for
    /// K and for V.
    fn decode_all(&self) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let n = self.head_dim;
        let mut k = vec![0.0f32; self.len() * n];
        let mut v = vec![0.0f32; self.len() * n];
        for (cell, exact) in self.cells.iter().enumerate() {
            if let Some(exact) = exact {
                k[cell * n..(cell + 1) * n].copy_from_slice(&widen(&exact.k));
                v[cell * n..(cell + 1) * n].copy_from_slice(&widen(&exact.v));
            }
        }
        for unit in &self.units {
            let at = unit.first_cell * n..(unit.first_cell + unit.cells) * n;
            decode_tile(self.policy.k, n, &unit.k_mean, &unit.k_rows, &mut k[at.clone()])?;
            decode_tile(self.policy.v, n, &unit.v_mean, &unit.v_rows, &mut v[at])?;
        }
        Ok((k, v))
    }

    /// The same attention the plain way: every cell decoded to `f32` with
    /// [`decode_tile`] and the `f16` rows, the dot products, the softmax and the
    /// weighted sum of V in `f64` over all visible cells at once. It shares no
    /// step with [`attend`](Self::attend) except the validation of its
    /// arguments, and the two agree to rounding.
    pub fn attend_plain(
        &self,
        query: &[f32],
        scale: f32,
        visible: &[Range<usize>],
    ) -> Result<Attention, KvHeadError> {
        let ranges = self.visible_ranges(visible)?;
        self.query(query, scale)?;
        let cells: Vec<usize> = ranges.into_iter().flatten().collect();
        let n = self.head_dim;
        let (k, v) = self.decode_all()?;
        let scores: Vec<f64> = cells
            .iter()
            .map(|&cell| f64::from(scale) * dot(query, &k[cell * n..(cell + 1) * n]))
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let weights: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        let mut output = vec![0.0f64; n];
        for (&cell, &weight) in cells.iter().zip(&weights) {
            for (acc, x) in output.iter_mut().zip(&v[cell * n..(cell + 1) * n]) {
                *acc += weight * f64::from(*x);
            }
        }
        Ok(Attention {
            output: output.iter().map(|x| *x as f32).collect(),
            weights: weights.iter().map(|w| *w as f32).collect(),
            scores,
        })
    }
}
