//! The plan of a step's attention: which blocks run, and what each reads.
//!
//! The host owns the page tables, so it knows for every row which entries are
//! exact and which converted, and it cuts the work into **items** accordingly.
//! An item is one thread block of an attention kernel:
//!
//! - some rows of one sequence (the query heads of one KV head of these rows
//!   are the vectors of the product: `rows * q_per_kv` of them, at most 8, 16, 48
//!   or 64 by [`AttnShape`]),
//! - a list of ranges of cells (one range for dense attention),
//! - all of them in entries of one kind, exact or converted, so that a piece has
//!   one domain (rotated or not) and its formats are known to the kernel.
//!
//! A row covered by one item is finished by it (`FINAL`: the output with the
//! gate applied); a row covered by several gets a partial state (the log2 of the
//! sum of the exponentials and the weighted V, in the original domain) per item
//! in a state slot, and the merge kernel combines them. That is the composability
//! of the reference: pieces are independent, any cut of the cells is valid, and a
//! piece can be read by several rows at once: the rows of one window do, and so do
//! the rows of several sequences over a prefix of entries they have in common
//! ([`PlanConfig::min_shared_cells`]), where that is cheaper than each on its own.
//!
//! **Rows that must not depend on the step** ([`Planner::window`]). A row of a
//! draft window has to give the bits it gives as the only row of a step, or a
//! reply would depend on what was drafted. Three things decide the bits of a
//! row, and such a window fixes all three by the row alone: the pieces (the
//! cut of the row's cells: the pieces merge as rounded sums), which are those
//! of a step that holds nothing but the row; the kernel, which is the
//! tensor-core kernel in its decode shapes for one row and for several (a
//! vector has the same arithmetic in each of them, whatever shares the item);
//! and the entries, which the caller names per window. Rows whose pieces are
//! the same share their items, so the cells are still read once per window.

use std::ops::Range;

use super::{AttnRow, CacheGeometry, ENTRY_CELLS};

pub const ITEM_CONVERTED: u32 = 1;
pub const ITEM_CAUSAL: u32 = 2;
pub const ITEM_FINAL: u32 = 4;
/// The item runs on the tensor-core kernels whatever its vectors and formats.
pub const ITEM_TENSOR: u32 = 8;
const CONVERTED_BIT: u32 = 0x8000_0000;

/// One block of work, as the kernels read it (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttnItem {
    /// Index of the first entry of the sequence's table in the table array.
    pub table: u32,
    /// Index in the row-id list of the item's first row.
    pub row0: u32,
    pub n_rows: u32,
    /// The KV head among those in the pools.
    pub kv_head: u32,
    /// Index in the range array of the item's `(start, end, cells before)` triples.
    pub ranges_off: u32,
    pub n_ranges: u32,
    /// Cells in all of the ranges.
    pub total: u32,
    pub flags: u32,
    /// Which piece of its rows the item is.
    pub slot: u32,
    /// One more than the tile whose entry the item takes from `swap_entry`
    /// instead of the table; 0 for none.
    pub swap_tile: u32,
    pub swap_entry: u32,
    pub pad: u32,
}

impl AttnItem {
    /// Whether the item may run on the direct kernel, as far as the item
    /// says: it reads converted entries and is not bound to the tensor cores.
    pub fn direct(&self) -> bool {
        self.flags & ITEM_CONVERTED != 0 && self.flags & ITEM_TENSOR == 0
    }

    pub fn to_bytes(items: &[AttnItem]) -> Vec<u8> {
        items
            .iter()
            .flat_map(|i| {
                [
                    i.table,
                    i.row0,
                    i.n_rows,
                    i.kv_head,
                    i.ranges_off,
                    i.n_ranges,
                    i.total,
                    i.flags,
                    i.slot,
                    i.swap_tile,
                    i.swap_entry,
                    0,
                ]
            })
            .flat_map(u32::to_le_bytes)
            .collect()
    }
}

/// The compiled shapes of the attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttnShape {
    /// Up to 8 vectors (one row of a decode step). Over quantized K and V in converted
    /// entries such an item runs on the direct kernel and this shape's layout is not used;
    /// otherwise the cells are the rows of the product, the vectors one tile of 8 columns,
    /// 2 warps do the product and 6 load and decode.
    Decode8,
    /// Up to 16 vectors, the same with two tiles of columns.
    Decode16,
    /// Up to 48 vectors (eight rows of 6 heads): three warps share the vectors.
    Decode48,
    /// Up to 64 vectors, for wide steps: the vectors are the rows, 4 warps, 16 cells per chunk.
    /// Items of a list of ranges; an item of one range runs on [`AttnShape::Prompt128`].
    Wide64,
    /// Up to 128 vectors over one range of cells that starts at a multiple of 32, for the rows
    /// of a prompt: 8 warps of 16 vectors, 32 cells per chunk (`kernels/attn_prompt.cuh`).
    Prompt128,
}

impl AttnShape {
    pub const ALL: [AttnShape; 5] = [
        AttnShape::Decode8,
        AttnShape::Decode16,
        AttnShape::Decode48,
        AttnShape::Wide64,
        AttnShape::Prompt128,
    ];

    pub const fn max_vecs(self) -> u32 {
        match self {
            AttnShape::Decode8 => 8,
            AttnShape::Decode16 => 16,
            AttnShape::Decode48 => 48,
            AttnShape::Wide64 => 64,
            AttnShape::Prompt128 => 128,
        }
    }

    /// The smallest shape that holds `vecs` vectors.
    pub fn for_vecs(vecs: u32) -> AttnShape {
        AttnShape::ALL
            .into_iter()
            .find(|s| vecs <= s.max_vecs())
            .expect("at most 64 vectors per item")
    }

    /// (warps sharing the vectors, cell groups, cells per chunk, helper warps, tiles of 8
    /// vectors per warp or 0 for the orientation with the vectors as rows).
    pub const fn layout(self) -> (u32, u32, u32, u32, u32) {
        match self {
            AttnShape::Decode8 => (1, 2, 32, 6, 1),
            AttnShape::Decode16 => (1, 2, 32, 6, 2),
            AttnShape::Decode48 => (3, 2, 32, 2, 2),
            AttnShape::Wide64 => (4, 1, 16, 4, 0),
            AttnShape::Prompt128 => (8, 1, 32, 0, 0),
        }
    }

    pub const fn threads(self) -> u32 {
        let (mt, cg, _, xw, _) = self.layout();
        32 * (mt * cg + xw)
    }

    pub fn kernel_name(self, head_dim: u32) -> String {
        let suffix = match self {
            AttnShape::Decode8 => "s8",
            AttnShape::Decode16 => "s16",
            AttnShape::Decode48 => "s48",
            AttnShape::Wide64 => "w64",
            AttnShape::Prompt128 => "p128",
        };
        format!("ares_attn_d{head_dim}_{suffix}")
    }

    /// Bytes of shared memory per raw stage and side: the larger of a chunk of
    /// q8 rows and 16 rows of f16.
    pub const fn rawcap(self, head_dim: u32) -> u32 {
        let (_, _, chunk, _, _) = self.layout();
        let q8 = chunk * (head_dim / 32 * 34);
        let f16 = 16 * 2 * head_dim;
        (if q8 > f16 { q8 } else { f16 }).div_ceil(16) * 16
    }

    /// Dynamic shared memory of a launch: the layout of `attn_body` in
    /// `kernels/attn_kernel.cuh`.
    pub const fn smem_bytes(self, head_dim: u32) -> u32 {
        let (mt, cg, chunk, _, nvw) = self.layout();
        if let AttnShape::Prompt128 = self {
            // the queries, the K and V tiles of a chunk (f16), the limits and the mean dots
            return (128 + 2 * chunk) * head_dim * 2 + 128 * 4 * 2;
        }
        let rows = if nvw > 0 {
            let r = mt * nvw * 8;
            if r < 16 { 16 } else { r }
        } else {
            16 * mt
        };
        let me = if rows <= 16 { 32 } else { 4 };
        (rows + 2 * chunk) * (head_dim + 8) * 2
            + 4 * self.rawcap(head_dim)
            + 3 * chunk * 3 * 4
            + 3 * (me + 1) * 4
            + rows * 4
            + rows * me * 4
            + 16
            + mt * cg * nvw * 8 * 24 * 2
            + 64
    }
}

/// How the planner cuts the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanConfig {
    /// Blocks to aim for in a launch; a decode step is cut into pieces until
    /// there are about this many (two to four per multiprocessor).
    pub target_blocks: u32,
    /// The fewest cells a piece of a range list is cut into.
    pub min_piece_cells: u32,
    /// The most pieces a row is cut into: bounds the state scratch.
    pub max_pieces: u32,
    /// The fewest cells of a prefix that several sequences share for which their rows
    /// read it together as one item.
    pub min_shared_cells: u32,
}

impl PlanConfig {
    pub fn for_multiprocessors(sm: u32) -> Self {
        Self { target_blocks: 3 * sm, min_piece_cells: 256, max_pieces: 64, min_shared_cells: 512 }
    }

    /// The most state slots ([`Plan::state_slots`]) a plan of [`Planner::finish`] takes for
    /// `kv_heads` KV heads of `q_per_kv` query heads each, in steps of at most `rows` rows
    /// of at most `sequences` sequences, of which at most `alone` rows are planned as alone
    /// ([`Planner::window`]: a generated row, the rows of draft windows). `mean` says that
    /// the formats keep exact and converted entries (a tile mean): a piece is of one kind,
    /// and a row reads the exact first tile, converted tiles and the exact newest ones.
    ///
    /// A row with more than one piece takes a slot per piece. The rows of a prompt are cut
    /// into tiles of `per` rows (64 vectors of queries), and a tile gets pieces in proportion
    /// to its cells, at least one: over all tiles at most the budget (`target_blocks` over
    /// the heads) and one per tile. A tile whose cells are of two or three runs of kinds
    /// has a piece more for every run after the first (one more for the cells of a prefix
    /// that sequences read together, whose tiles are full and whose newest are converted),
    /// and a row whose sequence reads a prefix with others has the pieces of both its
    /// tiles: so `per` times the budget and two tiles per sequence, and two (with a tile
    /// mean five) slots a row. A row alone has the budget (at most `max_pieces`) and two
    /// more with a tile mean. Checked against random steps
    /// (`bound_tests::the_state_slots_stay_in_their_bound`).
    pub fn state_slots_bound(
        &self,
        kv_heads: u32,
        q_per_kv: u32,
        rows: u32,
        alone: u32,
        sequences: u32,
        mean: bool,
    ) -> u64 {
        let budget = u64::from((self.target_blocks / kv_heads.max(1)).max(1));
        let per = u64::from((128 / q_per_kv.max(1)).max(1)).min(u64::from(rows.max(1)));
        let (row_pieces, extra) = if mean { (5, 2) } else { (2, 0) };
        let prompts = per * (budget + 2 * u64::from(sequences)) + row_pieces * u64::from(rows);
        let alone_pieces = budget.min(u64::from(self.max_pieces)) + extra;
        prompts + u64::from(alone) * alone_pieces
    }
}

/// A window of consecutive rows of one sequence that attend causally: row `i`
/// is the cell `pos0 + i` and sees the cells `0..=pos0 + i`.
#[derive(Debug)]
struct Dense {
    table: u32,
    entries: Vec<u32>,
    kinds: Vec<bool>,
    first_row: u32,
    n_rows: u32,
    pos0: u32,
    /// Planned row by row as each row alone would be ([`Planner::window`]).
    alone: bool,
    /// The tile whose entry the rows take from here, not from the table.
    swap: Option<(u32, u32)>,
}

/// One row that sees a list of ranges of cells.
#[derive(Debug)]
struct Sparse {
    table: u32,
    kinds: Vec<bool>,
    row: u32,
    pos: u32,
    ranges: Vec<Range<u32>>,
}

/// A piece: cells in entries of one kind, as ranges.
struct Piece {
    converted: bool,
    ranges: Vec<Range<u32>>,
}

/// Dense windows that read their common prefix together.
struct Share {
    /// Indices of the windows.
    members: Vec<usize>,
    /// The rows of all members.
    rows: Vec<u32>,
    /// Cells of the common prefix: whole entries, all before every member's first row.
    cells: u32,
}

/// The finished plan of a step.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// All items, grouped by shape (heaviest first within a group).
    pub items: Vec<AttnItem>,
    /// Per launch: the shape and the range of `items`. The shape of up to 8 vectors has
    /// two launches, one per kind of entry (converted, then exact), where the kinds run
    /// on different kernels.
    pub groups: Vec<(AttnShape, Range<usize>)>,
    pub ranges: Vec<u32>,
    pub rows: Vec<AttnRow>,
    pub row_ids: Vec<u32>,
    /// State slots used (each is `q_heads_on_device * (head_dim + 1)` floats).
    pub state_slots: u32,
    /// Whether any row has more than one piece, so that the merge kernel runs.
    pub needs_merge: bool,
}

impl Plan {
    /// Bytes of state scratch the plan needs: partial outputs and their log sums.
    pub fn state_bytes(&self, geo: &CacheGeometry) -> u64 {
        let hq = u64::from(geo.kv_heads * geo.q_per_kv);
        u64::from(self.state_slots) * hq * (u64::from(geo.head_dim) + 1) * 4
    }
}

#[derive(Debug)]
pub struct Planner {
    geo: CacheGeometry,
    cfg: PlanConfig,
    dense: Vec<Dense>,
    sparse: Vec<Sparse>,
    n_rows: u32,
}

impl Planner {
    /// A planner for a step of `n_rows` rows.
    pub fn new(geo: CacheGeometry, cfg: PlanConfig, n_rows: u32) -> Self {
        Self { geo, cfg, dense: Vec::new(), sparse: Vec::new(), n_rows }
    }

    fn kinds(entries: &[u32]) -> Vec<bool> {
        entries.iter().map(|e| e & CONVERTED_BIT != 0).collect()
    }

    /// Rows `first_row..first_row + n_rows` of the step are consecutive cells of a
    /// sequence, starting at `pos0`; each sees everything before it and itself.
    /// `entries` is the host copy of the sequence's table, which is at `table`
    /// in the device array, and covers the cells up to the last row.
    pub fn dense(&mut self, table: u32, entries: &[u32], first_row: u32, n_rows: u32, pos0: u32) {
        let need = (pos0 + n_rows).div_ceil(ENTRY_CELLS) as usize;
        assert!(
            entries.len() >= need,
            "the table has {} entries, {need} are needed",
            entries.len()
        );
        self.dense.push(Dense {
            table,
            entries: entries[..need].to_vec(),
            kinds: Self::kinds(entries),
            first_row,
            n_rows,
            pos0,
            alone: false,
            swap: None,
        });
    }

    /// As [`Planner::dense`], for rows whose results must not depend on what
    /// else the step holds (the rows of a draft window, a row that is
    /// generated without drafts): every row gets the pieces and the kernel it
    /// has as the only row of a step, and so the same bits. The window reads
    /// nothing together with other sequences.
    ///
    /// `swap` names one tile and the entry these rows read it from instead of
    /// the table's: a tile that the rows before this window still read in
    /// another form (a tile that is converted for the rows from some position
    /// on, while the table has it as the rows before that position read it).
    pub fn window(
        &mut self,
        table: u32,
        entries: &[u32],
        first_row: u32,
        n_rows: u32,
        pos0: u32,
        swap: Option<(u32, u32)>,
    ) {
        let need = (pos0 + n_rows).div_ceil(ENTRY_CELLS) as usize;
        assert!(
            entries.len() >= need,
            "the table has {} entries, {need} are needed",
            entries.len()
        );
        let mut entries = entries[..need].to_vec();
        if let Some((tile, entry)) = swap {
            assert!((tile as usize) < need, "tile {tile} is past the window's cells");
            entries[tile as usize] = entry;
        }
        self.dense.push(Dense {
            table,
            kinds: Self::kinds(&entries),
            entries,
            first_row,
            n_rows,
            pos0,
            alone: true,
            swap,
        });
    }

    /// Row `row` of the step is the cell `pos` of a sequence and sees the cells in
    /// `ranges` (sorted, disjoint) and nothing else.
    pub fn sparse(
        &mut self,
        table: u32,
        entries: &[u32],
        row: u32,
        pos: u32,
        ranges: &[Range<u32>],
    ) {
        self.sparse.push(Sparse {
            table,
            kinds: Self::kinds(entries),
            row,
            pos,
            ranges: ranges.to_vec(),
        });
    }

    fn pieces_of_entries(kinds: &[bool], cells: Range<u32>, pieces_wanted: u32) -> Vec<Piece> {
        // Runs of entries of one kind inside the cell range, cut into pieces of whole
        // entries; the pieces share `pieces_wanted` in proportion to their length.
        let n_entries = cells.end.div_ceil(ENTRY_CELLS);
        let first = cells.start / ENTRY_CELLS;
        let mut runs: Vec<(bool, u32, u32)> = Vec::new();
        for e in first..n_entries {
            let kind = kinds[e as usize];
            match runs.last_mut() {
                Some(run) if run.0 == kind => run.2 = e + 1,
                _ => runs.push((kind, e, e + 1)),
            }
        }
        let total_entries = n_entries - first;
        let mut out = Vec::new();
        for (converted, e0, e1) in runs {
            let n = e1 - e0;
            let want = ((u64::from(pieces_wanted) * u64::from(n) + u64::from(total_entries) / 2)
                / u64::from(total_entries))
            .clamp(1, u64::from(n)) as u32;
            // `want` pieces of whole entries, the lengths differing by at most one entry.
            let (base, extra) = (n / want, n % want);
            let mut e = e0;
            for i in 0..want {
                let next = e + base + u32::from(i < extra);
                let start = (e * ENTRY_CELLS).max(cells.start);
                let end = (next * ENTRY_CELLS).min(cells.end);
                out.push(Piece { converted, ranges: std::iter::once(start..end).collect() });
                e = next;
            }
        }
        out
    }

    fn pieces_of_ranges(
        kinds: &[bool],
        ranges: &[Range<u32>],
        pieces_wanted: u32,
        min_cells: u32,
    ) -> Vec<Piece> {
        // Cut at entry boundaries (the kind can change there), then group runs of
        // one kind into pieces of about total / pieces_wanted cells.
        let total: u32 = ranges.iter().map(|r| r.end - r.start).sum();
        let target = total.div_ceil(pieces_wanted.max(1)).max(min_cells);
        let mut out: Vec<Piece> = Vec::new();
        let mut cells_in_piece = 0;
        for r in ranges {
            let mut at = r.start;
            while at < r.end {
                let entry_end = (at / ENTRY_CELLS + 1) * ENTRY_CELLS;
                let end = entry_end.min(r.end);
                let converted = kinds[(at / ENTRY_CELLS) as usize];
                let open = matches!(out.last(), Some(p) if p.converted == converted)
                    && cells_in_piece < target;
                if !open {
                    out.push(Piece { converted, ranges: Vec::new() });
                    cells_in_piece = 0;
                }
                let piece = out.last_mut().unwrap();
                match piece.ranges.last_mut() {
                    Some(last) if last.end == at => last.end = end,
                    _ => piece.ranges.push(at..end),
                }
                cells_in_piece += end - at;
                at = end;
            }
        }
        out
    }

    /// The groups of dense windows that read a common prefix together. A window joins the
    /// group it shares the most entries with. A group reads its prefix together if that
    /// is cheaper than apart, by the cost of a cell in the kernel each tile runs on: a
    /// tile of up to 8 vectors over quantized K and V runs on the direct kernel, every
    /// other on the tensor-core kernels, which take 1.6 to 3 times as long per cell
    /// (measured at 262144 cells: `q8` 1.6 to 1.75, `q6`/`q5` 2.6, `q4` 2.3 to 3.0).
    /// Counted as 2.5: two sequences of one row share with an `f16` format and stay
    /// apart with quantized ones, three or more always share.
    fn shared_prefixes(&self, g: u32) -> Vec<Share> {
        let quantized = self.geo.k.id() != 0 && self.geo.v.id() != 0;
        // cost of a cell of a tile of `rows` rows, in halves of the direct kernel's
        let cost = |rows: u32| -> u32 {
            if quantized && rows * g <= AttnShape::Decode8.max_vecs() { 2 } else { 5 }
        };
        let min_entries = self.cfg.min_shared_cells.div_ceil(ENTRY_CELLS).max(1);
        let mut groups: Vec<(Vec<usize>, u32)> = Vec::new();
        for (i, d) in self.dense.iter().enumerate() {
            // A window whose rows are planned as alone shares nothing.
            if d.alone {
                continue;
            }
            // Only entries that end before the window's first row can be shared: every row
            // of the item sees all of its cells.
            let own = d.pos0 / ENTRY_CELLS;
            let mut best: Option<(usize, u32, u64)> = None;
            for (gi, (members, shared)) in groups.iter().enumerate() {
                let first = &self.dense[members[0]];
                let common =
                    first.entries.iter().zip(&d.entries).take_while(|(a, b)| a == b).count() as u32;
                let common = common.min(*shared).min(own);
                let n = members.len() as u64;
                // cells read once instead of once per member, before and after joining
                let (before, after) = ((n - 1) * u64::from(*shared), n * u64::from(common));
                if common >= min_entries
                    && after > before
                    && best.is_none_or(|(_, _, gain)| after - before > gain)
                {
                    best = Some((gi, common, after - before));
                }
            }
            match best {
                Some((gi, common, _)) => {
                    groups[gi].0.push(i);
                    groups[gi].1 = common;
                }
                None => groups.push((vec![i], own)),
            }
        }
        groups
            .into_iter()
            .filter_map(|(members, shared)| {
                let rows: Vec<u32> = members
                    .iter()
                    .flat_map(|&m| {
                        let d = &self.dense[m];
                        d.first_row..d.first_row + d.n_rows
                    })
                    .collect();
                let tiles = |n: u32| -> u32 {
                    let per = rows_per_tile(n, g);
                    n.div_ceil(per) * cost(per.min(n))
                };
                let apart: u32 = members.iter().map(|&m| tiles(self.dense[m].n_rows)).sum();
                let together = tiles(rows.len() as u32);
                (members.len() > 1 && together < apart).then_some(Share {
                    members,
                    rows,
                    cells: shared * ENTRY_CELLS,
                })
            })
            .collect()
    }

    /// Cuts the step into items.
    pub fn finish(self) -> Plan {
        let geo = self.geo;
        let (g, h) = (geo.q_per_kv, geo.kv_heads);
        let mut b = Builder {
            g,
            h,
            plan: Plan {
                rows: vec![AttnRow::default(); self.n_rows as usize],
                row_ids: (0..self.n_rows).collect(),
                ..Plan::default()
            },
            groups: Default::default(),
            next_slot: 0,
        };
        // Windows whose sequences share a prefix of entries read it together: one item with
        // the rows of all of them, so the prefix is read once and not once per sequence.
        let shares = self.shared_prefixes(g);
        // Tiles: (cells, kind of tile). A tile gets pieces in proportion to its cells, rounded
        // down: a launch with a few blocks more than the device holds at once waits for a
        // second wave that runs almost alone.
        let mut work: u64 = 0;
        for share in &shares {
            let chunks =
                share.rows.len().div_ceil(rows_per_tile(share.rows.len() as u32, g) as usize);
            work += u64::from(share.cells) * chunks as u64;
        }
        for (i, d) in self.dense.iter().enumerate().filter(|(_, d)| !d.alone) {
            let from = shares.iter().find(|s| s.members.contains(&i)).map_or(0, |s| s.cells);
            let per = rows_per_tile(d.n_rows, g);
            let mut r = 0;
            while r < d.n_rows {
                let n = per.min(d.n_rows - r);
                work += u64::from(d.pos0 + r + n - from);
                r += n;
            }
        }
        work += self
            .sparse
            .iter()
            .map(|s| s.ranges.iter().map(|r| u64::from(r.end - r.start)).sum::<u64>())
            .sum::<u64>();
        let budget = u64::from((self.cfg.target_blocks / h.max(1)).max(1));
        let max_pieces = self.cfg.max_pieces;
        let pieces_for =
            |cells: u64| -> u32 { ((budget * cells / work.max(1)) as u32).clamp(1, max_pieces) };

        // The shared prefixes: the rows of all members, in tiles of rows.
        let mut shared_pieces: Vec<usize> = vec![0; self.dense.len()];
        for share in &shares {
            let first = &self.dense[share.members[0]];
            let per = rows_per_tile(share.rows.len() as u32, g) as usize;
            let pieces = Self::pieces_of_entries(
                &first.kinds,
                0..share.cells,
                pieces_for(u64::from(share.cells)),
            );
            for &m in &share.members {
                shared_pieces[m] = pieces.len();
            }
            for chunk in share.rows.chunks(per) {
                // The rows of a tile are a list in `row_ids`; consecutive rows of the step
                // are already one (the identity at its start).
                let consecutive = chunk.windows(2).all(|w| w[1] == w[0] + 1);
                let row0 = if consecutive {
                    chunk[0]
                } else {
                    let at = b.plan.row_ids.len() as u32;
                    b.plan.row_ids.extend_from_slice(chunk);
                    at
                };
                b.items(first.table, row0, chunk.len() as u32, &pieces, 0, true, false, None);
            }
        }
        // The pieces of a row that is alone in its step: all of the budget.
        let alone_pieces = (budget as u32).clamp(1, max_pieces);
        for d in self.dense.iter().filter(|d| d.alone) {
            // Rows share a tile while they have the same pieces, which are a
            // function of the entries a row reads, and while their vectors fit
            // the decode shapes of the kernel.
            let per = (AttnShape::Decode48.max_vecs() / g).max(1);
            let mut r = 0;
            while r < d.n_rows {
                let pos = d.pos0 + r;
                let entry_end = (pos / ENTRY_CELLS + 1) * ENTRY_CELLS;
                let n = per.min(d.n_rows - r).min(entry_end - pos);
                let pieces = Self::pieces_of_entries(&d.kinds, 0..pos + n, alone_pieces);
                b.tile(d.table, d.first_row + r, n, pos, pieces, true, 0, Some(d.swap));
                r += n;
            }
        }
        for (i, d) in self.dense.iter().enumerate().filter(|(_, d)| !d.alone) {
            let from = shares.iter().find(|s| s.members.contains(&i)).map_or(0, |s| s.cells);
            let per = rows_per_tile(d.n_rows, g);
            let mut r = 0;
            while r < d.n_rows {
                let n = per.min(d.n_rows - r);
                let last = d.pos0 + r + n - 1;
                let pieces = Self::pieces_of_entries(
                    &d.kinds,
                    from..last + 1,
                    pieces_for(u64::from(last + 1 - from)),
                );
                b.tile(
                    d.table,
                    d.first_row + r,
                    n,
                    d.pos0 + r,
                    pieces,
                    true,
                    shared_pieces[i] as u32,
                    None,
                );
                r += n;
            }
        }
        for s in &self.sparse {
            let cells: u64 = s.ranges.iter().map(|r| u64::from(r.end - r.start)).sum();
            let pieces = Self::pieces_of_ranges(
                &s.kinds,
                &s.ranges,
                pieces_for(cells),
                self.cfg.min_piece_cells,
            );
            b.tile(s.table, s.row, 1, s.pos, pieces, false, 0, None);
        }
        let Builder { mut plan, groups, next_slot, .. } = b;
        for (shape, mut items) in AttnShape::ALL.into_iter().zip(groups) {
            if items.is_empty() {
                continue;
            }
            // The heaviest items first. Where the two kinds of entry run on different
            // kernels (up to 8 vectors over quantized K and V: converted entries on the
            // direct kernel, exact ones on the tensor cores) each kind is a launch of its
            // own, the converted items first; otherwise one launch takes both.
            // Items that must run on the tensor cores go with the exact ones.
            let split = shape == AttnShape::Decode8 && geo.k.id() != 0 && geo.v.id() != 0;
            items.sort_by_key(|i| (split && !i.direct(), std::cmp::Reverse(i.total)));
            let first =
                if split { items.iter().filter(|i| i.direct()).count() } else { items.len() };
            let start = plan.items.len();
            plan.items.extend(items);
            let end = plan.items.len();
            for part in [start..start + first, start + first..end] {
                if !part.is_empty() {
                    plan.groups.push((shape, part));
                }
            }
        }
        plan.state_slots = next_slot;
        plan.needs_merge = plan.rows.iter().any(|r| r.n_slots > 0);
        plan
    }
}

struct Builder {
    g: u32,
    h: u32,
    plan: Plan,
    groups: [Vec<AttnItem>; 5],
    next_slot: u32,
}

impl Builder {
    /// Emits the items of `n_rows` consecutive rows starting at `first_row`, whose
    /// first row is the cell `pos0`, cut into `pieces`. The rows have `before` more
    /// pieces from items emitted elsewhere (a prefix shared with other sequences),
    /// which take the first of the rows' state slots.
    #[allow(clippy::too_many_arguments)]
    fn tile(
        &mut self,
        table: u32,
        first_row: u32,
        n_rows: u32,
        pos0: u32,
        pieces: Vec<Piece>,
        causal: bool,
        before: u32,
        alone: Option<Option<(u32, u32)>>,
    ) {
        let n_pieces = before + pieces.len() as u32;
        let single = n_pieces == 1;
        for i in 0..n_rows {
            let row = &mut self.plan.rows[(first_row + i) as usize];
            row.table = table;
            row.pos = pos0 + i;
            if !single {
                row.slot_base = self.next_slot + i * n_pieces;
                row.n_slots = n_pieces;
            }
        }
        if !single {
            self.next_slot += n_rows * n_pieces;
        }
        self.items(table, first_row, n_rows, &pieces, before, causal, single, alone);
    }

    /// Emits one item per piece and KV head for the `n_rows` rows listed in `row_ids`
    /// from `row0` on. The pieces are the rows' pieces `slot0` and following. `alone`
    /// is given for rows planned as alone ([`Planner::window`]), with their swapped
    /// tile: their items are bound to the tensor-core kernels.
    #[allow(clippy::too_many_arguments)]
    fn items(
        &mut self,
        table: u32,
        row0: u32,
        n_rows: u32,
        pieces: &[Piece],
        slot0: u32,
        causal: bool,
        single: bool,
        alone: Option<Option<(u32, u32)>>,
    ) {
        let (swap_tile, swap_entry) =
            alone.flatten().map_or((0, 0), |(tile, entry)| (tile + 1, entry));
        let vecs = n_rows * self.g;
        let shape = AttnShape::ALL
            .iter()
            .position(|s| vecs <= s.max_vecs())
            .expect("at most 128 vectors per item");
        for (slot, piece) in pieces.iter().enumerate() {
            // More vectors than the decode shapes hold: the prompt kernel takes an item of
            // one range that starts at a multiple of its chunk, the wide kernel any other.
            let prompt = piece.ranges.len() == 1 && piece.ranges[0].start.is_multiple_of(32);
            let shape = if vecs <= AttnShape::Decode48.max_vecs() {
                shape
            } else if prompt {
                4
            } else {
                assert!(vecs <= AttnShape::Wide64.max_vecs(), "at most 64 vectors per item of ranges");
                3
            };
            let ranges_off = self.plan.ranges.len() as u32;
            let mut before = 0;
            for r in &piece.ranges {
                self.plan.ranges.extend([r.start, r.end, before]);
                before += r.end - r.start;
            }
            for kv_head in 0..self.h {
                self.groups[shape].push(AttnItem {
                    table,
                    row0,
                    n_rows,
                    kv_head,
                    ranges_off,
                    n_ranges: piece.ranges.len() as u32,
                    total: before,
                    flags: (if piece.converted { ITEM_CONVERTED } else { 0 })
                        | (if causal { ITEM_CAUSAL } else { 0 })
                        | (if single { ITEM_FINAL } else { 0 })
                        | (if alone.is_some() { ITEM_TENSOR } else { 0 }),
                    slot: slot0 + slot as u32,
                    swap_tile,
                    swap_entry,
                    pad: 0,
                });
            }
        }
    }
}

/// Rows per tile of a window: all of them while the vectors fit a decode shape (up to 48),
/// otherwise tiles of up to 128 vectors (the prompt kernel), as equal as they can be.
fn rows_per_tile(n_rows: u32, g: u32) -> u32 {
    if n_rows * g <= 48 {
        return n_rows;
    }
    let most = (128 / g).max(1);
    n_rows.div_ceil(n_rows.div_ceil(most))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::CacheFormat;

    fn geometry() -> CacheGeometry {
        CacheGeometry {
            k: CacheFormat::Q8,
            v: CacheFormat::Q8,
            head_dim: 256,
            kv_heads: 4,
            kv_total: 4,
            first_kv_head: 0,
            q_per_kv: 6,
        }
    }

    /// What a row is to see: its table, its position, and its ranges of cells (a causal
    /// window sees `0..=pos`).
    type Want = (u32, u32, Vec<Range<u32>>);

    fn window(table: u32, pos: u32) -> Want {
        (table, pos, std::iter::once(0..pos + 1).collect())
    }

    /// Every row must see each of its cells in exactly one item of every KV head, in
    /// entries of the item's kind, and its pieces must fill its state slots.
    fn check(plan: &Plan, geo: &CacheGeometry, tables: &[u32], want: &[Want]) {
        assert_eq!(plan.rows.len(), want.len());
        for (row, (table, pos, cells)) in want.iter().enumerate() {
            let (table, pos) = (*table, *pos);
            assert_eq!((plan.rows[row].table, plan.rows[row].pos), (table, pos), "row {row}");
            let mut expected = vec![0u32; pos as usize + 1];
            for cell in cells.iter().flat_map(|r| r.clone()) {
                expected[cell as usize] = 1;
            }
            for head in 0..geo.kv_heads {
                let mut seen = vec![0u32; pos as usize + 1];
                let mut slots = Vec::new();
                for item in plan.items.iter().filter(|i| i.kv_head == head) {
                    let list =
                        &plan.row_ids[item.row0 as usize..(item.row0 + item.n_rows) as usize];
                    if !list.contains(&(row as u32)) {
                        continue;
                    }
                    assert!(item.n_rows * geo.q_per_kv <= 64);
                    slots.push(item.slot);
                    assert_eq!(item.flags & ITEM_FINAL != 0, plan.rows[row].n_slots == 0);
                    let ranges = &plan.ranges[item.ranges_off as usize..];
                    let mut total = 0;
                    for r in 0..item.n_ranges as usize {
                        let (start, end, before) =
                            (ranges[3 * r], ranges[3 * r + 1], ranges[3 * r + 2]);
                        assert_eq!(before, total);
                        total += end - start;
                        for cell in start..end {
                            let entry = tables[(item.table + cell / ENTRY_CELLS) as usize];
                            assert_eq!(entry, tables[(table + cell / ENTRY_CELLS) as usize]);
                            assert_eq!(
                                entry & CONVERTED_BIT != 0,
                                item.flags & ITEM_CONVERTED != 0
                            );
                            if cell <= pos {
                                seen[cell as usize] += 1;
                            } else {
                                assert!(
                                    item.flags & ITEM_CAUSAL != 0,
                                    "cell {cell} beyond row {row}"
                                );
                            }
                        }
                    }
                    assert_eq!(total, item.total);
                }
                assert_eq!(seen, expected, "row {row} head {head}");
                slots.sort_unstable();
                let n = plan.rows[row].n_slots.max(1);
                assert_eq!(slots, (0..n).collect::<Vec<_>>(), "row {row} head {head}");
            }
        }
        let groups: usize = plan.groups.iter().map(|(_, r)| r.len()).sum();
        assert_eq!(groups, plan.items.len());
        for (shape, range) in &plan.groups {
            let items = &plan.items[range.clone()];
            assert!(items.iter().all(|i| i.n_rows * geo.q_per_kv <= shape.max_vecs()));
            // A launch is of one kernel: the direct one, or the tensor cores.
            if *shape == AttnShape::Decode8 && geo.k.id() != 0 && geo.v.id() != 0 {
                let direct = items[0].direct();
                assert!(items.iter().all(|i| i.direct() == direct));
            }
        }
    }

    #[test]
    fn rows_of_sequences_with_a_common_prefix_read_it_as_one_item() {
        let geo = geometry();
        let cfg = PlanConfig::for_multiprocessors(46);
        // Three sequences: the first 40 entries are the same slots, then each has its own;
        // a fourth shares nothing. One row each, at different lengths.
        let mut tables = Vec::new();
        let mut want = Vec::new();
        let mut planner = Planner::new(geo, cfg, 4);
        for (s, len) in [(0u32, 6000u32), (1, 5500), (2, 7000), (3, 6100)] {
            let entries: Vec<u32> = (0..len.div_ceil(ENTRY_CELLS))
                .map(|e| {
                    let slot = if e < 40 && s < 3 { e } else { 1000 * (s + 1) + e };
                    // the tile that is still filling is exact
                    if e + 1 == len.div_ceil(ENTRY_CELLS) { slot } else { slot | CONVERTED_BIT }
                })
                .collect();
            let table = tables.len() as u32;
            planner.dense(table, &entries, s, 1, len - 1);
            want.push(window(table, len - 1));
            tables.extend(entries);
        }
        let plan = planner.finish();
        check(&plan, &geo, &tables, &want);
        // The prefix of 40 entries is one tile of 3 rows (18 vectors), read once.
        let shared: Vec<&AttnItem> = plan.items.iter().filter(|i| i.n_rows == 3).collect();
        assert!(!shared.is_empty());
        let cells: u32 = shared.iter().filter(|i| i.kv_head == 0).map(|i| i.total).sum();
        assert_eq!(cells, 40 * ENTRY_CELLS);
        // No more blocks than the device holds at once, apart from the small tails.
        let large = plan.items.iter().filter(|i| i.total > 1024).count() as u32;
        assert!(large <= cfg.target_blocks, "{large} blocks");
    }

    /// What a plan has row `row` (the cell `pos`) read: per item of the row its KV head,
    /// its piece, its flags, and its ranges cut at the row's own cell.
    fn seen(plan: &Plan, row: u32, pos: u32) -> Vec<(u32, u32, u32, Vec<Range<u32>>)> {
        let mut items: Vec<_> = plan
            .items
            .iter()
            .filter(|i| plan.row_ids[i.row0 as usize..(i.row0 + i.n_rows) as usize].contains(&row))
            .map(|i| {
                let ranges = &plan.ranges[i.ranges_off as usize..][..3 * i.n_ranges as usize];
                let cut = ranges.chunks(3).map(|r| r[0]..r[1].min(pos + 1)).collect();
                (i.kv_head, i.slot, i.flags, cut)
            })
            .collect();
        items.sort_by_key(|i| (i.0, i.1));
        items
    }

    #[test]
    fn a_window_cuts_every_row_as_the_row_alone_whatever_else_the_step_holds() {
        let geo = geometry();
        // 21 pieces for a row; the sequence has 80 entries, exact at both ends.
        let cfg = PlanConfig::for_multiprocessors(28);
        let entries: Vec<u32> = (0..80u32)
            .map(|e| if e == 0 || e >= 77 { 500 + e } else { e | CONVERTED_BIT })
            .collect();
        // 11 rows across the entry that begins at 9984, where the pieces of a row
        // change, and more vectors than one tile holds; next to them the rows of a
        // prompt and a row of a third sequence.
        let (pos0, n) = (9980u32, 11u32);
        let other: Vec<u32> = (0..40u32).map(|e| (2000 + e) | CONVERTED_BIT).collect();
        let mut planner = Planner::new(geo, cfg, n + 31);
        planner.dense(80, &other, 0, 30, 4000);
        planner.window(0, &entries, 30, n, pos0, None);
        planner.window(120, &other[..20], 30 + n, 1, 2500, None);
        let together = planner.finish();
        let mut tables = entries.clone();
        tables.extend(&other);
        tables.extend(&other[..20]);
        let mut want: Vec<Want> = (0..30).map(|i| window(80, 4000 + i)).collect();
        want.extend((0..n).map(|i| window(0, pos0 + i)));
        want.push(window(120, 2500));
        check(&together, &geo, &tables, &want);
        for r in 0..n {
            let mut planner = Planner::new(geo, cfg, 1);
            planner.window(0, &entries, 0, 1, pos0 + r, None);
            let alone = planner.finish();
            check(&alone, &geo, &entries, &[window(0, pos0 + r)]);
            assert_eq!(seen(&together, 30 + r, pos0 + r), seen(&alone, 0, pos0 + r), "row {r}");
            assert_eq!(together.rows[(30 + r) as usize].n_slots, alone.rows[0].n_slots);
        }
        // Rows share items where their pieces are the same: the four rows before the
        // new entry, and the seven after it (42 vectors, one tile).
        let rows_of = |row: u32| {
            let item = together.items.iter().find(|i| {
                together.row_ids[i.row0 as usize..(i.row0 + i.n_rows) as usize].contains(&row)
            });
            item.map(|i| i.n_rows)
        };
        assert_eq!((rows_of(30), rows_of(34), rows_of(40)), (Some(4), Some(7), Some(7)));
        // None of the window's items may run on the direct kernel, one row or many.
        let windows = together.items.iter().filter(|i| i.table != 80);
        assert!(windows.clone().count() > 0 && windows.clone().all(|i| !i.direct()));
    }

    #[test]
    fn a_window_names_the_tile_it_reads_from_another_entry() {
        let geo = CacheGeometry { k: CacheFormat::Q4, v: CacheFormat::Q4, ..geometry() };
        let cfg = PlanConfig::for_multiprocessors(46);
        // Five entries, all exact; the rows read tile 1 from the converted entry 9.
        let entries: Vec<u32> = (0..5u32).map(|e| 100 + e).collect();
        let mut planner = Planner::new(geo, cfg, 4);
        planner.window(0, &entries, 0, 2, 510, None);
        planner.window(0, &entries, 2, 2, 512, Some((1, 9 | CONVERTED_BIT)));
        let plan = planner.finish();
        let swapped: Vec<&AttnItem> = plan.items.iter().filter(|i| i.swap_tile != 0).collect();
        assert!(swapped.iter().all(|i| (i.swap_tile, i.swap_entry) == (2, 9 | CONVERTED_BIT)));
        // A piece per entry (few entries): the rows from the position on read tile 1
        // converted, the rows before it read exact entries only.
        let kinds = |row: u32| -> Vec<bool> {
            let mut items: Vec<&AttnItem> = plan
                .items
                .iter()
                .filter(|i| i.kv_head == 0 && (i.row0..i.row0 + i.n_rows).contains(&row))
                .collect();
            items.sort_by_key(|i| i.slot);
            items.iter().map(|i| i.flags & ITEM_CONVERTED != 0).collect()
        };
        assert_eq!(kinds(0), [false; 4]);
        assert_eq!(kinds(2), [false, true, false, false, false]);
        assert!(plan.items.iter().filter(|i| i.row0 == 0).all(|i| i.swap_tile == 0));
    }

    #[test]
    fn two_rows_share_a_prefix_only_where_reading_it_together_is_cheaper() {
        // Two sequences of one row with the same 30 entries. Quantized, each row runs on
        // the direct kernel and a tile of two rows would not: they stay apart. With an f16
        // format every tile runs on the tensor cores: together.
        for (k, shared) in [(CacheFormat::Q8, false), (CacheFormat::F16, true)] {
            let geo = CacheGeometry { k, ..geometry() };
            let entries: Vec<u32> = (0..32u32).map(|e| e | CONVERTED_BIT).collect();
            let mut planner = Planner::new(geo, PlanConfig::for_multiprocessors(46), 2);
            planner.dense(0, &entries[..31], 0, 1, 3900);
            planner.dense(0, &entries, 1, 1, 4000);
            let plan = planner.finish();
            check(&plan, &geo, &entries, &[window(0, 3900), window(0, 4000)]);
            assert_eq!(plan.items.iter().any(|i| i.n_rows == 2), shared, "{k:?}");
        }
    }

    #[test]
    fn windows_without_a_common_prefix_and_lists_of_ranges_are_cut_as_before() {
        let geo = geometry();
        let cfg = PlanConfig::for_multiprocessors(28);
        let mut tables = Vec::new();
        let mut want = Vec::new();
        let mut planner = Planner::new(geo, cfg, 9);
        // a window of 8 rows on a sequence with exact entries at both ends
        let entries: Vec<u32> = (0..80u32)
            .map(|e| if e == 0 || e >= 77 { 500 + e } else { e | CONVERTED_BIT })
            .collect();
        planner.dense(0, &entries, 0, 8, 10_000);
        want.extend((0..8).map(|i| window(0, 10_000 + i)));
        tables.extend(&entries);
        // one row of another sequence that sees a list of ranges
        let other: Vec<u32> = (0..20u32).map(|e| (2000 + e) | CONVERTED_BIT).collect();
        let ranges: Vec<Range<u32>> = (0..60u32).map(|i| i * 40..i * 40 + 4).collect();
        planner.sparse(80, &other, 8, 2500, &ranges);
        want.push((80, 2500, ranges));
        tables.extend(&other);
        let plan = planner.finish();
        check(&plan, &geo, &tables, &want);
        // the sparse row: its ranges exactly once
        let cells: u32 =
            plan.items.iter().filter(|i| i.table == 80 && i.kv_head == 0).map(|i| i.total).sum();
        assert_eq!(cells, 240);
        assert_eq!(plan.rows[8].pos, 2500);
    }
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    use crate::attention::CacheFormat;

    /// The entries of a sequence of `cells` cells: page ids from `base`, the first `shared`
    /// of them from a prefix other sequences have too; with a tile mean the first tile and
    /// the tiles from `exact_from` on are exact.
    fn entries(cells: u32, base: u32, shared: u32, mean: bool, exact_from: u32) -> Vec<u32> {
        (0..cells.div_ceil(ENTRY_CELLS))
            .map(|e| {
                let id = if e < shared { 5_000_000 + e } else { base + e };
                let exact = mean && (e == 0 || e * ENTRY_CELLS >= exact_from);
                if exact { id } else { id | CONVERTED_BIT }
            })
            .collect()
    }

    /// Random steps of prompts of one or two sequences (apart or reading a common
    /// prefix), with and without rows alone next to them, and of rows alone only, for
    /// devices of several sizes and head counts, with and without a tile mean: no plan
    /// takes more state slots than [`PlanConfig::state_slots_bound`]. (The bound of
    /// before, twenty tiles of the budget, was passed by a prompt that read its prefix
    /// with another sequence, and by any prompt past its first tile with a tile mean:
    /// three pieces a row, the step of 256 rows that the 27B failed on in `q5`.)
    #[test]
    fn the_state_slots_stay_in_their_bound() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut random = |n: u32| -> u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % u64::from(n.max(1))) as u32
        };
        let (max_rows, sequences, alone_rows) = (512u32, 2u32, 16u32);
        for (kv_heads, q_per_kv, sms) in
            [(1, 6, 28), (3, 6, 28), (4, 6, 46), (2, 8, 34), (1, 4, 46)]
        {
            for mean in [false, true] {
                let format = if mean { CacheFormat::Q5 } else { CacheFormat::Q8 };
                let geo = CacheGeometry {
                    k: format,
                    v: format,
                    head_dim: 256,
                    kv_heads,
                    kv_total: kv_heads,
                    first_kv_head: 0,
                    q_per_kv,
                };
                let cfg = PlanConfig::for_multiprocessors(sms);
                let bound = cfg
                    .state_slots_bound(kv_heads, q_per_kv, max_rows, alone_rows, sequences, mean);
                let mut worst = 0;
                for _ in 0..1500 {
                    // 0: prompts; 1: prompts and rows alone; 2: rows alone.
                    let kind = random(3);
                    let alone = if kind == 0 { 0 } else { 1 + random(alone_rows / 2) };
                    let mut dense = Vec::new();
                    let mut row = 0;
                    let shared = random(2) == 1;
                    let prefix = 1 + random(40);
                    for s in 0..if kind == 2 { 0 } else { 1 + random(sequences) } {
                        let left = max_rows - 2 * alone - row;
                        if left < 2 {
                            break;
                        }
                        let n = 2 + random(left - 1);
                        let pos0 = if shared {
                            prefix * ENTRY_CELLS + random(4000)
                        } else {
                            random(60000)
                        };
                        let newest = pos0.saturating_sub(ENTRY_CELLS) / ENTRY_CELLS * ENTRY_CELLS;
                        let cells = entries(
                            pos0 + n,
                            100_000 * (s + 1),
                            if shared { prefix } else { 0 },
                            mean,
                            newest,
                        );
                        dense.push((1000 * s, cells, row, n, pos0));
                        row += n;
                    }
                    let mut windows = Vec::new();
                    for w in 0..if kind == 0 { 0 } else { 2 } {
                        let n = 1 + random(alone_rows / 2);
                        let pos = 8 + random(60000);
                        let newest = pos.saturating_sub(ENTRY_CELLS) / ENTRY_CELLS * ENTRY_CELLS;
                        windows.push((
                            10_000 + 1000 * w,
                            entries(pos + n, 300_000 * (w + 1), 0, mean, newest),
                            row,
                            n,
                            pos,
                        ));
                        row += n;
                    }
                    let mut planner = Planner::new(geo, cfg, row.max(1));
                    for (table, cells, first, n, pos0) in &dense {
                        planner.dense(*table, cells, *first, *n, *pos0);
                    }
                    for (table, cells, first, n, pos) in &windows {
                        planner.window(*table, cells, *first, *n, *pos, None);
                    }
                    worst = worst.max(planner.finish().state_slots);
                }
                eprintln!(
                    "{kv_heads} KV heads of {q_per_kv}, {sms} SMs, tile mean {mean}: worst \
                     {worst} of {bound}"
                );
                assert!(
                    u64::from(worst) <= bound,
                    "{kv_heads} heads, mean {mean}: {worst} > {bound}"
                );
            }
        }
    }
}
