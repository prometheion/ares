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
    Wide64,
}

impl AttnShape {
    pub const ALL: [AttnShape; 4] =
        [AttnShape::Decode8, AttnShape::Decode16, AttnShape::Decode48, AttnShape::Wide64];

    pub const fn max_vecs(self) -> u32 {
        match self {
            AttnShape::Decode8 => 8,
            AttnShape::Decode16 => 16,
            AttnShape::Decode48 => 48,
            AttnShape::Wide64 => 64,
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
        for (i, d) in self.dense.iter().enumerate() {
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
                b.items(first.table, row0, chunk.len() as u32, &pieces, 0, true, false);
            }
        }
        for (i, d) in self.dense.iter().enumerate() {
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
            b.tile(s.table, s.row, 1, s.pos, pieces, false, 0);
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
            let split = shape == AttnShape::Decode8 && geo.k.id() != 0 && geo.v.id() != 0;
            items.sort_by_key(|i| {
                (split && i.flags & ITEM_CONVERTED == 0, std::cmp::Reverse(i.total))
            });
            let first = if split {
                items.iter().filter(|i| i.flags & ITEM_CONVERTED != 0).count()
            } else {
                items.len()
            };
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
    groups: [Vec<AttnItem>; 4],
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
        self.items(table, first_row, n_rows, &pieces, before, causal, single);
    }

    /// Emits one item per piece and KV head for the `n_rows` rows listed in `row_ids`
    /// from `row0` on. The pieces are the rows' pieces `slot0` and following.
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
    ) {
        let vecs = n_rows * self.g;
        let shape = AttnShape::ALL
            .iter()
            .position(|s| vecs <= s.max_vecs())
            .expect("at most 64 vectors per item");
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
                    row0,
                    n_rows,
                    kv_head,
                    ranges_off,
                    n_ranges: piece.ranges.len() as u32,
                    total: before,
                    flags: (if piece.converted { ITEM_CONVERTED } else { 0 })
                        | (if causal { ITEM_CAUSAL } else { 0 })
                        | (if single { ITEM_FINAL } else { 0 }),
                    slot: slot0 + slot as u32,
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
            if *shape == AttnShape::Decode8 && geo.k.id() != 0 && geo.v.id() != 0 {
                let kind = items[0].flags & ITEM_CONVERTED;
                assert!(items.iter().all(|i| i.flags & ITEM_CONVERTED == kind));
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
