//! The plan of a step's attention: which blocks run, and what each reads.
//!
//! The host owns the page tables, so it knows for every row which entries are
//! exact and which converted, and it cuts the work into **items** accordingly.
//! An item is one thread block of an attention kernel:
//!
//! - some rows of one sequence (the query heads of one KV head of these rows
//!   are the vectors of the product: `rows * q_per_kv` of them, at most 16, 48
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
//! piece can be read by several rows at once (rows of one sequence do; a prefix
//! shared by the rows of several sequences could, see the notes on what is open).

use std::ops::Range;

use super::{AttnRow, CacheGeometry, ENTRY_CELLS};

pub const ITEM_CONVERTED: u32 = 1;
pub const ITEM_CAUSAL: u32 = 2;
pub const ITEM_FINAL: u32 = 4;
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
    pub pad: [u32; 3],
}

impl AttnItem {
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
                    0,
                    0,
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
    /// Up to 16 vectors (one row of a decode step): 2 warps, 32 cells per chunk.
    Decode16,
    /// Up to 48 vectors (up to eight rows): 6 warps.
    Decode48,
    /// Up to 64 vectors, for wide steps: 4 warps, 16 cells per chunk.
    Wide64,
}

impl AttnShape {
    pub const ALL: [AttnShape; 3] = [AttnShape::Decode16, AttnShape::Decode48, AttnShape::Wide64];

    pub const fn max_vecs(self) -> u32 {
        match self {
            AttnShape::Decode16 => 16,
            AttnShape::Decode48 => 48,
            AttnShape::Wide64 => 64,
        }
    }

    /// m-tiles of 16 vectors, cell groups, cells per chunk.
    pub const fn mt_cg_chunk(self) -> (u32, u32, u32) {
        match self {
            AttnShape::Decode16 => (1, 2, 32),
            AttnShape::Decode48 => (3, 2, 32),
            AttnShape::Wide64 => (4, 1, 16),
        }
    }

    /// Warps beyond the `mt * cg` that run the product: they load and decode.
    pub const fn extra_warps(self) -> u32 {
        match self {
            AttnShape::Decode16 => 6,
            AttnShape::Decode48 => 2,
            AttnShape::Wide64 => 4,
        }
    }

    pub const fn threads(self) -> u32 {
        let (mt, cg, _) = self.mt_cg_chunk();
        32 * (mt * cg + self.extra_warps())
    }

    pub fn kernel_name(self, head_dim: u32) -> String {
        let (mt, cg, chunk) = self.mt_cg_chunk();
        format!("ares_attn_d{head_dim}_m{mt}c{cg}k{chunk}x{}", self.extra_warps())
    }

    /// Bytes of shared memory per raw stage and side: the larger of a chunk of
    /// q8 rows and 16 rows of f16.
    pub const fn rawcap(self, head_dim: u32) -> u32 {
        let (_, _, chunk) = self.mt_cg_chunk();
        let q8 = chunk * (head_dim / 32 * 34);
        let f16 = 16 * 2 * head_dim;
        (if q8 > f16 { q8 } else { f16 }).div_ceil(16) * 16
    }

    /// Dynamic shared memory of a launch (the formula of `attn_smem_bytes` in
    /// `kernels/attn_kernel.cuh`).
    pub const fn smem_bytes(self, head_dim: u32) -> u32 {
        let (mt, _, chunk) = self.mt_cg_chunk();
        let me = if mt == 1 { 32 } else { 4 };
        (16 * mt + 2 * chunk) * (head_dim + 8) * 2
            + 4 * self.rawcap(head_dim)
            + 3 * chunk * 3 * 4
            + 3 * (me + 1) * 4
            + 16 * mt * 4
            + 16 * mt * me * 4
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
}

impl PlanConfig {
    pub fn for_multiprocessors(sm: u32) -> Self {
        Self { target_blocks: 3 * sm, min_piece_cells: 256, max_pieces: 64 }
    }
}

/// A window of consecutive rows of one sequence that attend causally: row `i`
/// is the cell `pos0 + i` and sees the cells `0..=pos0 + i`.
#[derive(Debug)]
struct Dense {
    table: u32,
    kinds: Vec<bool>,
    first_row: u32,
    n_rows: u32,
    pos0: u32,
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

/// The finished plan of a step.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// All items, grouped by shape (heaviest first within a group).
    pub items: Vec<AttnItem>,
    /// Per shape: the range of `items` it launches.
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
        self.dense.push(Dense { table, kinds: Self::kinds(entries), first_row, n_rows, pos0 });
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
            let per = n.div_ceil(want);
            let mut e = e0;
            while e < e1 {
                let next = (e + per).min(e1);
                let start = (e * ENTRY_CELLS).max(cells.start);
                let end = (next * ENTRY_CELLS).min(cells.end);
                out.push(Piece { converted, ranges: vec![start..end] });
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
        // How many pieces each tile aims for: enough blocks to fill the device.
        let tiles: u32 =
            self.dense.iter().map(|d| d.n_rows.div_ceil(rows_per_tile(d.n_rows, g))).sum::<u32>()
                + self.sparse.len() as u32;
        let per_tile =
            self.cfg.target_blocks.div_ceil((tiles * h).max(1)).clamp(1, self.cfg.max_pieces);
        for d in &self.dense {
            let per = rows_per_tile(d.n_rows, g);
            let mut r = 0;
            while r < d.n_rows {
                let n = per.min(d.n_rows - r);
                let last = d.pos0 + r + n - 1;
                let pieces = Self::pieces_of_entries(&d.kinds, 0..last + 1, per_tile);
                b.tile(d.table, d.first_row + r, n, d.pos0 + r, pieces, true);
                r += n;
            }
        }
        for s in &self.sparse {
            let pieces =
                Self::pieces_of_ranges(&s.kinds, &s.ranges, per_tile, self.cfg.min_piece_cells);
            b.tile(s.table, s.row, 1, s.pos, pieces, false);
        }
        let Builder { mut plan, groups, next_slot, .. } = b;
        for (shape, mut items) in AttnShape::ALL.into_iter().zip(groups) {
            if items.is_empty() {
                continue;
            }
            items.sort_by_key(|i| std::cmp::Reverse(i.total));
            let start = plan.items.len();
            plan.items.extend(items);
            plan.groups.push((shape, start..plan.items.len()));
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
    groups: [Vec<AttnItem>; 3],
    next_slot: u32,
}

impl Builder {
    /// Emits the items of `n_rows` consecutive rows starting at `first_row`, whose
    /// first row is the cell `pos0`, cut into `pieces`.
    fn tile(
        &mut self,
        table: u32,
        first_row: u32,
        n_rows: u32,
        pos0: u32,
        pieces: Vec<Piece>,
        causal: bool,
    ) {
        let vecs = n_rows * self.g;
        let shape = if vecs <= 16 {
            0
        } else if vecs <= 48 {
            1
        } else {
            2
        };
        let single = pieces.len() == 1;
        for i in 0..n_rows {
            let row = &mut self.plan.rows[(first_row + i) as usize];
            row.table = table;
            row.pos = pos0 + i;
            if !single {
                row.slot_base = self.next_slot + i * pieces.len() as u32;
                row.n_slots = pieces.len() as u32;
            }
        }
        if !single {
            self.next_slot += n_rows * pieces.len() as u32;
        }
        for (slot, piece) in pieces.iter().enumerate() {
            let ranges_off = self.plan.ranges.len() as u32;
            let mut before = 0;
            for r in &piece.ranges {
                self.plan.ranges.extend([r.start, r.end, before]);
                before += r.end - r.start;
            }
            for kv_head in 0..self.h {
                self.groups[shape].push(AttnItem {
                    table,
                    row0: first_row,
                    n_rows,
                    kv_head,
                    ranges_off,
                    n_ranges: piece.ranges.len() as u32,
                    total: before,
                    flags: (if piece.converted { ITEM_CONVERTED } else { 0 })
                        | (if causal { ITEM_CAUSAL } else { 0 })
                        | (if single { ITEM_FINAL } else { 0 }),
                    slot: slot as u32,
                    pad: [0; 3],
                });
            }
        }
    }
}

/// Rows per tile of a window: all of them while the vectors fit one launch shape
/// (up to 48), otherwise as many as fit 64 vectors.
fn rows_per_tile(n_rows: u32, g: u32) -> u32 {
    if n_rows * g <= 48 { n_rows } else { (64 / g).max(1) }
}
