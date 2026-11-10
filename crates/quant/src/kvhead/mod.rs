//! The stored KV sequence of one head, and attention over it the way a kernel
//! computes it: the reference that the attention kernels of both backends are
//! held to.
//!
//! [`kv`](crate::kv) is the codec: how one row, or the rows of one tile, become
//! bytes. This module is what the engine keeps in memory for one KV head of one
//! attention layer under a [`KvPolicy`], and what attention does with it. A
//! kernel that writes K and V must produce the bytes [`KvHead`] exposes; a
//! kernel that attends must produce what [`KvHead::attend`] returns, within the
//! tolerance given under "Arithmetic". Nothing here is fast. It is meant to be
//! read.
//!
//! # The stored sequence
//!
//! A [`KvHead`] holds `len()` cells of one head. A cell is the K row and the V
//! row of one position, `head_dim` values each (a multiple of 32). Cells are
//! appended one at a time as `f32` ([`KvHead::append`]), and each is in one of
//! two states:
//!
//! - **exact**: the K row and the V row as `f16`, unrotated, `2 * head_dim`
//!   bytes each. This is the [`KvFormat::F16`] encoding of the row (little
//!   endian halves, saturating at 65504, never an infinity), so a value is
//!   rounded to `f16` once, when the cell is appended;
//! - **converted**: stored in the formats `policy.k` and `policy.v`, a unit at a
//!   time ([`KvUnit`]): a tile of [`TILE_CELLS`] cells when K or V is `q7` to
//!   `q4` (the rows of the tile and, where the format has one, the mean of its
//!   rotated rows), a single cell otherwise. The bytes are those of
//!   [`encode_tile`].
//!
//! Which cells are converted is decided by the policy alone:
//! `policy.quantized_cells(len)` is the range of converted cells. It starts at
//! [`KvHead::lead_cells`] and grows, always by whole units, as the sequence
//! grows; every other cell is exact. After every call that returns, the stored
//! state is exactly that (see "Pending cells" for the one exception, asked for
//! by the caller).
//!
//! A unit is converted from the **stored** `f16` values of its cells, not from
//! the original `f32`: when a tile is converted the originals are gone. So a
//! converted tile is `encode_tile` of the `f16` values, which the tests check
//! bit by bit. The one exception follows from the same rule: a cell that the
//! policy never keeps exact (`q8`/`f16` with no first and no newest window, the
//! default of `q8`) is encoded straight from the `f32` row when it is appended,
//! because it is never stored as `f16`. A policy with a mean always keeps a cell
//! exact until its tile is full and out of the newest window.
//!
//! The bytes are the contract between this reference and the kernels that write
//! the cache: [`KvHead::exact`] for an exact cell, [`KvHead::unit_of`] for a
//! converted one ([`KvUnit::k_rows`], [`KvUnit::k_mean`] and the same for V).
//! [`KvHead::stored_bytes`] is [`KvPolicy::bytes_for_cells`] for one head,
//! which the tests check.
//!
//! # Truncation
//!
//! Rejected draft tokens are removed with [`KvHead::truncate`]. Cutting inside
//! the exact cells is always fine. A conversion cannot be undone (the `f16`
//! values are gone and a tile's rows depend on all of its cells through the
//! mean), so cutting to a length at which the policy would have fewer converted
//! cells than are stored is an error ([`KvHeadError::TruncateConverted`]) and
//! changes nothing. The stored state is therefore always the policy's state for
//! the current length, not a function of the history, and a sequence cut to `m`
//! cells is bit for bit the sequence built from its first `m` cells.
//!
//! How far a sequence can be cut: with a mean, a tile is converted at the length
//! at which it is full and `exact_newest_cells` cells newer than it exist, so
//! right after a conversion nothing can be cut at all and the room grows by one
//! per appended cell until the next conversion. [`KvHead::shortest`] is the
//! shortest length `truncate` accepts. The condition "the newest window is at
//! least the draft length" is not enough with [`KvHead::append`], because the
//! draft cells themselves can be the ones that push a tile out of the window.
//! The engine therefore appends draft cells with [`KvHead::append_pending`],
//! which converts nothing, verifies, cuts the rejected cells (always allowed:
//! they were never converted) and calls [`KvHead::settle`] for the accepted
//! ones. Settling converts exactly what `append` would have converted. With this
//! protocol the only bound on the policy is the ordinary one: rolling back
//! committed cells is not possible, rolling back pending ones always is.
//!
//! # Attention as states
//!
//! The attention of one query over a set of cells is a **state**
//! ([`AttnState`]): `lse`, the logarithm of the sum of the exponentials of the
//! scores of the cells, and `v`, the sum of their V rows weighted by
//! `exp(score - lse)`, that is by their softmax weights within the set. The
//! output of the attention over the set is its `v`. The state of the union of
//! two disjoint sets is the [`merge`](AttnState::merge) of their states: with
//! `m = max(lse_a, lse_b)`, `lse = m + ln(e^(lse_a - m) + e^(lse_b - m))` and `v`
//! the average of `v_a` and `v_b` weighted by those two exponentials. This is
//! the formulation of FlashInfer (the kernel library under SGLang), and it is
//! what this reference is built on: [`KvHead::state`] computes the state of a
//! range of cells, and [`KvHead::attend`] over a list of ranges is by definition
//! the merge of the states of its ranges, in order. Dense causal attention is
//! one range from 0, block-sparse attention the selected blocks.
//!
//! What a kernel may rely on:
//!
//! - **The merge is commutative and associative up to rounding, and a piece can
//!   be any range of cells**: a page of 16 or 64 cells, a tile of 128, the
//!   exact cells apart from the tiles, a block, half of the sequence. Pieces can
//!   be computed in any order, on any thread block or device, and merged in any
//!   order and any tree. The tests split the same sequence per tile, per page, by
//!   exactness and in halves, and merge forward, backward and as a tree.
//! - **An empty piece is the identity** (`lse = -infinity`, `v = 0`) and a piece
//!   whose scores are far below the others' contributes nothing and produces no
//!   NaN (its weight underflows to 0 against the larger `lse`).
//! - **A piece that cuts a tile** computes the tile's `q_rot . mean` itself, and
//!   adds the tile's mean to V with the weights of its own cells only; the
//!   pieces add up because `code * scale + mean` is what a row stands for.
//! - **The state is in the original domain.** A piece un-rotates its V
//!   accumulator once before it is a state. Since the rotation is linear and the
//!   weights are scalars, a kernel may merge tile pieces among themselves in the
//!   rotated domain and un-rotate once; the reference does not, so that states
//!   from different kinds of pieces can always be merged.
//!
//! This is what lets the kernels:
//!
//! 1. **compute pages independently**, one thread block per page or tile, and
//!    combine the partial states at the end (or in a second kernel);
//! 2. **split K and V of a sequence over two devices** under a tensor split:
//!    each device attends over the cells it holds and only states (`lse` and
//!    `head_dim` values per query) cross;
//! 3. **attend to a prefix shared by many rows of a step once**: the shared
//!    cells are read once for all rows that share them and each row merges its
//!    state over the prefix with the state over its own cells.
//!
//! ## The steps of a piece
//!
//! For one query and one range of cells:
//!
//! 1. If K is a rotated format, the query is rotated once, group by group
//!    ([`rotate_groups`](crate::rotate_groups), f32, the bits of the
//!    specification) into `q_rot`.
//! 2. The score of a cell is `scale * dot`. For an exact cell `dot` is the plain
//!    dot product of the unrotated query with the `f16` K row. For a cell of a
//!    unit, `dot` is, for each group, the f16 scale of the group times the sum of
//!    `q_rot * code` over its 32 values, summed over the groups, plus (for a
//!    format with a mean) `q_rot . mean` of the unit, computed once per unit
//!    and piece however many of its cells the piece holds.
//! 3. The softmax of the scores within the piece, and `lse`.
//! 4. V is accumulated in the stored domain. Exact cells add `weight * v` to a
//!    plain accumulator. Cells of a unit add `weight * code * scale` to a rotated
//!    accumulator, and at the end of each unit's cells in the piece the weight
//!    sum of those cells times the unit's mean. At the end the rotated
//!    accumulator is un-rotated, group by group, and added to the plain one.
//!
//! [`KvHead::attend_plain`] is the check: every cell decoded to `f32` with
//! [`decode_tile`](crate::decode_tile) and the `f16` rows, attention in `f64` on
//! the decoded rows over all visible cells at once, with no states and no
//! merge. The two agree to rounding on every pair of formats.
//!
//! ## Arithmetic
//!
//! The reference is meant to be the exact value of the expression above for the
//! stored bytes, so everything after the rotation of the query is `f64`:
//!
//! - `q_rot` is `f32` and bit-exact (the rotation of the specification). It is
//!   the one place where the reference is not "exact": a kernel must produce
//!   these bits too, or the comparison is about the rotation.
//! - Decoding a row ([`decode_row_rotated`](crate::decode_row_rotated)) is exact
//!   in `f32`, and widening to `f64` is exact. Products of two such numbers are
//!   exact in `f64` (24 + 24 bits), so the only roundings are in the sums:
//!   sequential `f64` sums (the 32 products of a group, then the groups, then
//!   the mean term), the softmax of a piece (`exp` of `f64`, max subtracted, sum
//!   in cell order, one division), the weighted sums of the values in cell
//!   order, and the merges.
//! - The un-rotation of the V accumulator is the butterfly network of the
//!   specification with its constant `C = 0x3e3504f3` (as `f64`, so it is the same
//!   linear map as the kernel's), in `f64`.
//! - The output and the weights are rounded once to `f32`; [`Attention::scores`]
//!   keeps the `f64` scores and the states are `f64`.
//!
//! Splitting the visible cells into pieces and merging changes the result by
//! `f64` rounding, about 1e-14 relative in the tests, so the reference does not
//! depend on how a caller cuts the sequence.
//!
//! A kernel is free to sum in `f32`, in any order, to run an online softmax, to
//! merge in `f32`, to keep the weights in `f32` and to fuse multiply and add. All
//! of that differs from the reference by rounding only. The size of that
//! difference for `f32` kernels is [`OUTPUT_TOLERANCE`] (per output element,
//! relative to the largest output magnitude) and [`WEIGHT_TOLERANCE`] (the sum of
//! the absolute differences of the weights). They were calibrated with a
//! deliberately naive `f32` implementation (sequential and blocked sums, `f32`
//! exp, the codec's decoding) on 2 000 visible cells and 256 values per row, whole
//! and in pieces of 16, 64 and 128 cells merged in `f32`; it reaches 2e-6 and
//! 1.8e-6, and the tolerances are 10 and 11 times that. The error of a sum grows
//! with the square root of its length for a blockwise kernel, so a test with 64k
//! visible cells should relax them accordingly.
//!
//! Where the reference is *not* a bound for a kernel: a kernel that keeps scores
//! or weights in a narrower type than `f32` (an `f16` P matrix for a tensor-core
//! product, say) differs by more, as the narrower type's rounding says, and has
//! to state and test its own tolerance against [`KvHead::attend_plain`] on real
//! data. This reference does not decide that.
//!
//! # Pending cells
//!
//! [`KvHead::append_pending`] stores a cell as exact and converts nothing, even
//! if the policy would convert at the new length; [`KvHead::settle`] converts to
//! what the policy says for the current length. Attention over pending cells
//! reads them as exact (more precise than the policy's state, never less).

mod attend;

use std::ops::Range;

use thiserror::Error as ThisError;

use crate::error::Error;
use crate::half::f16_to_f32;
use crate::kv::{KvFormat, KvPolicy, TILE_CELLS, encode_row, encode_tile};

pub use attend::{Attention, AttnState, OUTPUT_TOLERANCE, WEIGHT_TOLERANCE};

/// What can go wrong storing a cell, cutting a sequence, or attending.
#[derive(Debug, ThisError, Clone, PartialEq, Eq)]
pub enum KvHeadError {
    /// The codec refused: a value that is NaN or infinite, or a rotation that
    /// overflows (see [`encode_row`]).
    #[error(transparent)]
    Codec(#[from] Error),
    /// The head size is not a positive multiple of 32.
    #[error("a head size of {0} values is not a positive multiple of 32")]
    HeadDim(usize),
    /// A K row, V row or query does not have `head_dim` values.
    #[error("the {what} has {actual} values, the head has {expected}")]
    RowLength { what: &'static str, expected: usize, actual: usize },
    /// The query or the scale is NaN or infinite.
    #[error("the {what} is not finite")]
    NotFinite { what: &'static str },
    /// `truncate` was asked for more cells than there are.
    #[error("cannot cut to {to} cells, the sequence has {len}")]
    TruncateBeyond { to: usize, len: usize },
    /// `truncate` would have to turn converted cells back into exact ones.
    #[error(
        "cannot cut a sequence of {len} cells to {to}: cells that the policy keeps exact at {to} cells are already stored in the formats; the shortest length is {shortest}"
    )]
    TruncateConverted { to: usize, len: usize, shortest: usize },
    /// A visible range is out of order, overlaps the one before it, or ends
    /// beyond the sequence.
    #[error(
        "visible range {index} ({start}..{end}) is out of order, overlaps the previous one, or ends beyond the {len} cells"
    )]
    VisibleRange { index: usize, start: usize, end: usize, len: usize },
    /// The ranges contain no cell, so there is nothing to attend to.
    #[error("no cell is visible")]
    NoVisibleCells,
}

/// The K row and V row of an exact cell, as the `f16` format stores them.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExactCell {
    k: Box<[u8]>,
    v: Box<[u8]>,
}

/// A unit of converted cells: the rows of one tile (or of one cell, when the
/// policy has no mean) of K and of V as `encode_tile` writes them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvUnit {
    /// The position of the first cell of the unit.
    pub first_cell: usize,
    /// The number of cells: [`TILE_CELLS`] or 1.
    pub cells: usize,
    /// The mean of the rotated K rows, f16 halves; empty for `f16` and `q8`.
    pub k_mean: Vec<u8>,
    /// The K rows one after the other, `format.row_bytes(head_dim)` each.
    pub k_rows: Vec<u8>,
    /// The mean of the rotated V rows; empty for `f16` and `q8`.
    pub v_mean: Vec<u8>,
    /// The V rows one after the other.
    pub v_rows: Vec<u8>,
    k_row_bytes: usize,
    v_row_bytes: usize,
}

impl KvUnit {
    /// The bytes of the K row of the `i`-th cell of the unit.
    pub fn k_row(&self, i: usize) -> &[u8] {
        &self.k_rows[i * self.k_row_bytes..(i + 1) * self.k_row_bytes]
    }

    /// The bytes of the V row of the `i`-th cell of the unit.
    pub fn v_row(&self, i: usize) -> &[u8] {
        &self.v_rows[i * self.v_row_bytes..(i + 1) * self.v_row_bytes]
    }

    /// Bytes the unit takes: rows and means.
    pub fn bytes(&self) -> usize {
        self.k_rows.len() + self.k_mean.len() + self.v_rows.len() + self.v_mean.len()
    }

    /// Encodes `cells` rows of K and of V (`f32`, `cells * head_dim` values each)
    /// that start at position `first_cell`.
    fn encode(
        policy: &KvPolicy,
        head_dim: usize,
        first_cell: usize,
        cells: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<KvUnit, Error> {
        let side = |format: KvFormat, values: &[f32]| -> Result<(Vec<u8>, Vec<u8>, usize), Error> {
            let row_bytes = format.row_bytes(head_dim)?;
            let mut mean = vec![0; format.mean_bytes(head_dim)?];
            let mut rows = vec![0; cells * row_bytes];
            encode_tile(format, head_dim, values, &mut mean, &mut rows)?;
            Ok((mean, rows, row_bytes))
        };
        let (k_mean, k_rows, k_row_bytes) = side(policy.k, k)?;
        let (v_mean, v_rows, v_row_bytes) = side(policy.v, v)?;
        Ok(KvUnit { first_cell, cells, k_mean, k_rows, v_mean, v_rows, k_row_bytes, v_row_bytes })
    }
}

/// The stored sequence of one KV head under a [`KvPolicy`]. See the
/// [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct KvHead {
    policy: KvPolicy,
    head_dim: usize,
    /// The length at which the conversions were last brought up to date; the
    /// cells from here on are pending.
    settled: usize,
    /// One entry per cell: its `f16` rows if it is exact, nothing if converted.
    cells: Vec<Option<ExactCell>>,
    /// The converted units, in order, from `lead_cells()` on.
    units: Vec<KvUnit>,
}

impl KvHead {
    /// An empty sequence of a head of `head_dim` values under `policy`.
    pub fn new(policy: KvPolicy, head_dim: usize) -> Result<Self, KvHeadError> {
        if head_dim == 0 || !head_dim.is_multiple_of(crate::kv::GROUP_VALUES) {
            return Err(KvHeadError::HeadDim(head_dim));
        }
        Ok(KvHead { policy, head_dim, settled: 0, cells: Vec::new(), units: Vec::new() })
    }

    pub fn policy(&self) -> &KvPolicy {
        &self.policy
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The number of cells, pending ones included.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The length at which the conversions are up to date; equal to `len()`
    /// unless cells were added with [`append_pending`](Self::append_pending).
    pub fn settled_len(&self) -> usize {
        self.settled
    }

    /// Cells per converted unit: [`TILE_CELLS`] when K or V has a mean, else 1.
    pub fn unit_cells(&self) -> usize {
        if self.policy.has_mean() { TILE_CELLS } else { 1 }
    }

    /// The cells that are always exact because they come first: the first window
    /// of the policy, rounded up to a whole tile when the policy has a mean.
    /// Converted units start here.
    pub fn lead_cells(&self) -> usize {
        let first = self.policy.exact_first_cells;
        if self.policy.has_mean() {
            first.checked_next_multiple_of(TILE_CELLS).unwrap_or(usize::MAX)
        } else {
            first
        }
    }

    /// The cells stored in the formats, `0..0` if there are none; at the settled
    /// length this is `policy.quantized_cells(len())`.
    pub fn converted(&self) -> Range<usize> {
        if self.units.is_empty() {
            0..0
        } else {
            let lead = self.lead_cells();
            lead..lead + self.units.len() * self.unit_cells()
        }
    }

    /// Whether the cell at `cell` is stored as `f16`. A cell beyond the sequence
    /// is not.
    pub fn is_exact(&self, cell: usize) -> bool {
        self.cells.get(cell).is_some_and(Option::is_some)
    }

    /// The `f16` bytes of the K row and of the V row of an exact cell.
    pub fn exact(&self, cell: usize) -> Option<(&[u8], &[u8])> {
        self.cells.get(cell)?.as_ref().map(|c| (&*c.k, &*c.v))
    }

    /// The unit that holds a converted cell, and the cell's index in it.
    pub fn unit_of(&self, cell: usize) -> Option<(&KvUnit, usize)> {
        let conv = self.converted();
        if !conv.contains(&cell) {
            return None;
        }
        let unit = self.unit_cells();
        let at = (cell - self.lead_cells()) / unit;
        Some((&self.units[at], (cell - self.lead_cells()) % unit))
    }

    /// The converted units in order.
    pub fn units(&self) -> &[KvUnit] {
        &self.units
    }

    /// Bytes the sequence takes: exact cells at `4 * head_dim` each (K and V as
    /// `f16`) and the units with their means. For a settled sequence this is
    /// `policy.bytes_for_cells(len(), 1, head_dim)`.
    pub fn stored_bytes(&self) -> usize {
        let exact = self.cells.iter().flatten().count() * 4 * self.head_dim;
        exact + self.units.iter().map(KvUnit::bytes).sum::<usize>()
    }

    fn check_row(&self, what: &'static str, row: &[f32]) -> Result<(), KvHeadError> {
        if row.len() != self.head_dim {
            return Err(KvHeadError::RowLength {
                what,
                expected: self.head_dim,
                actual: row.len(),
            });
        }
        Ok(())
    }

    /// Appends a cell and converts what the policy converts at the new length.
    /// On an error nothing changes.
    ///
    /// The rows are `f32`; a value that is NaN or infinite is refused. The cell
    /// is stored as `f16` (rounded once), except that a cell the policy never
    /// keeps exact is encoded straight from the `f32` rows.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), KvHeadError> {
        self.check_row("K row", k)?;
        self.check_row("V row", v)?;
        let at = self.len();
        let never_exact = self.settled == at
            && !self.policy.has_mean()
            && self.policy.quantized_cells(at + 1).contains(&at);
        if never_exact {
            // all cells before it from the first window on are converted too
            let unit = KvUnit::encode(&self.policy, self.head_dim, at, 1, k, v)?;
            debug_assert_eq!(self.converted().end.max(self.lead_cells()), at);
            self.units.push(unit);
            self.cells.push(None);
            self.settled = at + 1;
            return Ok(());
        }
        self.append_pending(k, v)?;
        if let Err(error) = self.settle() {
            self.cells.pop();
            return Err(error);
        }
        Ok(())
    }

    /// Appends a cell as exact and converts nothing: for draft cells that may
    /// still be cut. Call [`settle`](Self::settle) when they are accepted. On an
    /// error nothing changes.
    pub fn append_pending(&mut self, k: &[f32], v: &[f32]) -> Result<(), KvHeadError> {
        self.check_row("K row", k)?;
        self.check_row("V row", v)?;
        let half = |row: &[f32]| -> Result<Box<[u8]>, KvHeadError> {
            let mut bytes = vec![0; 2 * self.head_dim];
            encode_row(KvFormat::F16, row, &mut bytes)?;
            Ok(bytes.into_boxed_slice())
        };
        let cell = ExactCell { k: half(k)?, v: half(v)? };
        self.cells.push(Some(cell));
        Ok(())
    }

    /// Converts every unit that `policy.quantized_cells(len())` stores in the
    /// formats and that is not yet, from the stored `f16` values of its cells.
    /// On an error nothing changes.
    pub fn settle(&mut self) -> Result<(), KvHeadError> {
        let len = self.len();
        let range = self.policy.quantized_cells(len);
        let mut new_units = Vec::new();
        if !range.is_empty() {
            debug_assert_eq!(range.start, self.lead_cells());
            let unit = self.unit_cells();
            let mut at = self.lead_cells() + self.units.len() * unit;
            while at + unit <= range.end {
                new_units.push(self.convert(at, unit)?);
                at += unit;
            }
        }
        let from = self.lead_cells() + self.units.len() * self.unit_cells();
        let to = from + new_units.len() * self.unit_cells();
        for cell in &mut self.cells[from.min(len)..to.min(len)] {
            *cell = None;
        }
        self.units.extend(new_units);
        self.settled = len;
        Ok(())
    }

    /// Converts the `cells` exact cells from `at` on into a unit.
    fn convert(&self, at: usize, cells: usize) -> Result<KvUnit, KvHeadError> {
        let n = self.head_dim;
        let (mut k, mut v) = (Vec::with_capacity(cells * n), Vec::with_capacity(cells * n));
        let widen = |bytes: &[u8], out: &mut Vec<f32>| {
            let (halves, _) = bytes.as_chunks::<2>();
            out.extend(halves.iter().map(|h| f16_to_f32(u16::from_le_bytes(*h))));
        };
        for cell in at..at + cells {
            let exact = self.cells[cell].as_ref().expect("a cell to convert is exact");
            widen(&exact.k, &mut k);
            widen(&exact.v, &mut v);
        }
        Ok(KvUnit::encode(&self.policy, n, at, cells, &k, &v)?)
    }

    /// Cuts the sequence to its first `new_len` cells.
    ///
    /// Pending cells and exact cells can always be cut. A cut that would leave
    /// fewer converted cells than are stored (the policy at `new_len` cells would
    /// keep some of them exact) is an error, [`KvHeadError::TruncateConverted`];
    /// so is a `new_len` beyond the sequence. On an error nothing changes. The
    /// result is the sequence built from the first `new_len` cells.
    pub fn truncate(&mut self, new_len: usize) -> Result<(), KvHeadError> {
        let len = self.len();
        if new_len > len {
            return Err(KvHeadError::TruncateBeyond { to: new_len, len });
        }
        if new_len < self.settled {
            if self.policy.quantized_cells(new_len) != self.converted() {
                return Err(KvHeadError::TruncateConverted {
                    to: new_len,
                    len,
                    shortest: self.shortest(),
                });
            }
            self.settled = new_len;
        }
        self.cells.truncate(new_len);
        Ok(())
    }

    /// The shortest length [`truncate`](Self::truncate) accepts now: 0 while no
    /// cell is converted, otherwise the end of the converted cells plus the
    /// newest window of the policy.
    pub fn shortest(&self) -> usize {
        let conv = self.converted();
        if conv.is_empty() {
            0
        } else {
            (conv.end + self.policy.exact_newest_cells).min(self.settled)
        }
    }
}
