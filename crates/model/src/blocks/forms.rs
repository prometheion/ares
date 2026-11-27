//! Which tiles of the K/V pages are exact, and when the others are converted.
//!
//! With `q7` to `q4` for K or V a sequence is stored a tile of 128 cells (one
//! page) at a time, relative to the tile's mean. The first tile and the newest
//! cells stay `f16` (`ares_quant::KvPolicy`): a tile is an **exact** entry
//! until it is full and has left the newest window, and then it is converted
//! into the page's **converted** entry. The converted entry of page `p` is slot
//! `p` of the converted pool, which the engine's page pool sizes; an exact
//! entry is a slot of a small second pool, held by the page for as long as the
//! page is exact. This is the part of the state that a device keeps for that
//! second pool; the conductor knows nothing of it, apart from telling the
//! device which pages were freed.
//!
//! - A page gets an exact slot with its first write.
//! - At the start of every step, for every sequence in it, the tiles that left
//!   the sequence's window are converted. A shared page is converted for every
//!   holder, and every holder's table follows the next time its sequence runs.
//! - A row reads a tile converted from the position at which the tile has
//!   left the newest cells, whatever step computes the row. A draft window
//!   that reaches across such a position has the tile both ways: converted
//!   at the start of the step for its rows from that position on
//!   ([`Ahead`]), and still an exact entry, which its rows before the
//!   position read and which a step that starts past the position gives up.
//!   So a row of a window reads what it would read in a step of its own.
//! - A page that the cache holds and no sequence needs stays exact until the
//!   pool is short of slots; then the page used least recently among the full
//!   ones is converted. A partly used page (a tail of the prefix cache) never
//!   is, so the cache limits its tails (see [`exact_slots`]).
//! - A page the pool frees gives its slot back.
//!
//! Everything here is plain data; the device part launches the conversions and
//! writes the tables.

use std::ops::Range;

/// "No exact slot".
pub(crate) const NONE: u32 = u32::MAX;

/// Cells of a tile.
const TILE: u32 = ares_cuda::ENTRY_CELLS;

/// A tile that goes from its exact slot into the converted entry of its page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Convert {
    pub exact: u32,
    pub page: u32,
}

/// What one sequence does in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reach {
    pub slot: u32,
    /// The position of the first row: the cells before it are settled.
    pub pos0: u32,
    pub rows: u32,
    /// Whether the rows write cells (a replay does not).
    pub writes: bool,
    /// Whether all rows are certain to be accepted.
    pub settles: bool,
}

/// A tile that the later rows of a draft window read converted while the
/// table still names its exact entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ahead {
    pub slot: u32,
    /// The first position whose row reads the tile converted.
    pub from: u32,
    pub tile: u32,
    pub page: u32,
}

/// The most rows of a draft window for a group that keeps its `newest` cells
/// exact: the tile that leaves the newest cells inside the window must have
/// been full before the window (so no more rows than `newest`), and only one
/// tile may leave (so no more rows than a tile).
pub(crate) fn window_limit(newest: u32) -> u32 {
    newest.min(TILE)
}

impl Reach {
    /// The tiles the rows read or write.
    pub(crate) fn tiles(&self) -> Range<u32> {
        0..(self.pos0 + self.rows).div_ceil(TILE)
    }
}

/// The exact pool has no slot to give.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Exhausted {
    pub capacity: u32,
    /// Exact pages that are not full (tails and the ends of sequences).
    pub partial: u32,
    /// Exact pages some sequence of the step reads.
    pub in_step: u32,
}

impl std::fmt::Display for Exhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "all {} exact K/V slots are taken ({} by pages that are not full, {} by sequences of \
             this step)",
            self.capacity, self.partial, self.in_step
        )
    }
}

/// The exact slots a group of layers needs with `newest` exact cells at the
/// end of a sequence, for `slots` sequences in steps of `rows` rows at most and
/// `tails` partly used pages that the prefix cache holds.
///
/// A sequence between steps has its first tile exact and, in its window, at
/// most one more tile than `ceil(newest / 128)`; a step adds the tiles its
/// rows begin, at most `ceil(rows / 128)` over all sequences and one more for
/// every sequence that starts in the middle of a tile.
pub(crate) fn exact_slots(newest: u32, slots: u32, rows: u32, tails: u32) -> u32 {
    slots * (2 + newest.div_ceil(TILE)) + rows.div_ceil(TILE) + slots + tails
}

/// The state of the exact pool of one group of layers.
#[derive(Debug)]
pub(crate) struct Forms {
    newest: u32,
    capacity: u32,
    /// Per page: its exact slot, or [`NONE`].
    exact_of: Vec<u32>,
    /// Per page: whether all its cells are settled.
    full: Vec<bool>,
    /// Per page: the step that read or wrote it last.
    used: Vec<u64>,
    free: Vec<u32>,
    clock: u64,
    jobs: Vec<Convert>,
    /// The tiles the draft windows of the step read both ways.
    ahead: Vec<Ahead>,
}

impl Forms {
    /// An empty pool of `capacity` slots for `pages` pages whose groups keep
    /// their newest `newest` cells exact.
    pub(crate) fn new(capacity: u32, pages: u32, newest: u32) -> Self {
        Self {
            newest,
            capacity,
            exact_of: vec![NONE; pages as usize],
            full: vec![false; pages as usize],
            used: vec![0; pages as usize],
            free: (0..capacity).rev().collect(),
            clock: 0,
            jobs: Vec::new(),
            ahead: Vec::new(),
        }
    }

    /// The exact slots of the pool.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The exact slots in use.
    pub(crate) fn in_use(&self) -> u32 {
        self.capacity - self.free.len() as u32
    }

    /// The exact slot of `page`.
    pub(crate) fn slot_of(&self, page: u32) -> Option<u32> {
        Some(self.exact_of[page as usize]).filter(|slot| *slot != NONE)
    }

    /// The table entry of `page` in a group: its exact slot, or the converted
    /// entry (marked by `converted`).
    pub(crate) fn entry(&self, page: u32, converted: u32) -> u32 {
        self.slot_of(page).unwrap_or(page | converted)
    }

    /// The tiles of a sequence of `cells` settled cells that are converted:
    /// every full tile between the first and the window of the newest cells.
    pub(crate) fn converted_tiles(&self, cells: u32) -> Range<u32> {
        1..cells.saturating_sub(self.newest) / TILE
    }

    /// The pool freed `page`.
    pub(crate) fn release(&mut self, page: u32) {
        let p = page as usize;
        if let Some(slot) = self.slot_of(page) {
            self.free.push(slot);
            self.exact_of[p] = NONE;
        }
        self.full[p] = false;
        self.used[p] = 0;
    }

    /// Notes that `page` holds all its cells, settled.
    pub(crate) fn settle_page(&mut self, page: u32) {
        self.full[page as usize] = true;
    }

    /// The pages `row` names, up to the settled `cells`, are full.
    pub(crate) fn settle(&mut self, row: impl Fn(u32) -> u32, cells: u32) {
        for tile in 0..cells / TILE {
            self.settle_page(row(tile));
        }
    }

    fn convert(&mut self, page: u32) {
        let slot = self.exact_of[page as usize];
        debug_assert_ne!(slot, NONE, "only an exact page is converted");
        self.jobs.push(Convert { exact: slot, page });
        self.free.push(slot);
        self.exact_of[page as usize] = NONE;
    }

    /// Gives `page` an exact slot if it has none. A full page that no sequence
    /// of this step reads is converted to make room when the pool is empty.
    pub(crate) fn ensure(&mut self, page: u32, protect: Option<u32>) -> Result<u32, Exhausted> {
        if let Some(slot) = self.slot_of(page) {
            return Ok(slot);
        }
        if self.free.is_empty() {
            let victim = (0..self.exact_of.len() as u32)
                .filter(|p| {
                    self.exact_of[*p as usize] != NONE
                        && self.full[*p as usize]
                        && self.used[*p as usize] < self.clock
                        && Some(*p) != protect
                })
                .min_by_key(|p| self.used[*p as usize]);
            match victim {
                Some(victim) => self.convert(victim),
                None => {
                    let exact = (0..self.exact_of.len()).filter(|p| self.exact_of[*p] != NONE);
                    return Err(Exhausted {
                        capacity: self.capacity,
                        partial: exact.clone().filter(|p| !self.full[*p]).count() as u32,
                        in_step: exact.filter(|p| self.used[*p] == self.clock).count() as u32,
                    });
                }
            }
        }
        let slot = self.free.pop().expect("a slot was freed above");
        self.exact_of[page as usize] = slot;
        // A page that gets its slot now starts a tile.
        self.full[page as usize] = false;
        Ok(slot)
    }

    /// Prepares a step: the tiles that left their sequence's window are
    /// converted, and every page the rows write has an exact slot.
    /// `page(slot, tile)` names the pages of a sequence's table. The
    /// conversions wait in [`Forms::take_jobs`]. With `windows`, the tiles
    /// that leave the newest cells inside a draft window are converted too
    /// and keep their exact entries ([`Forms::ahead`]); a group whose rows
    /// attend one at a time has no use for that.
    pub(crate) fn prepare(
        &mut self,
        reaches: &[Reach],
        page: impl Fn(u32, u32) -> u32,
        windows: bool,
    ) -> Result<(), Exhausted> {
        self.clock += 1;
        self.ahead.clear();
        // The pages of the step are not given up for room.
        for r in reaches {
            for tile in r.tiles() {
                self.used[page(r.slot, tile) as usize] = self.clock;
            }
            self.settle(|tile| page(r.slot, tile), r.pos0);
        }
        for r in reaches {
            for tile in self.converted_tiles(r.pos0) {
                let p = page(r.slot, tile);
                if self.exact_of[p as usize] != NONE {
                    self.convert(p);
                }
            }
        }
        for r in reaches.iter().filter(|r| windows && !r.settles && r.rows > 1) {
            let last = r.pos0 + r.rows - 1;
            let leaving = self.converted_tiles(r.pos0).end.max(1)..self.converted_tiles(last).end;
            // Only a tile that was full before the window: its cells are
            // settled, so what is converted now is what a later step would
            // convert ([`window_limit`] keeps a window from reaching further).
            for tile in leaving.filter(|tile| (tile + 1) * TILE <= r.pos0) {
                let p = page(r.slot, tile);
                let slot = self.exact_of[p as usize];
                if slot != NONE {
                    self.jobs.push(Convert { exact: slot, page: p });
                    let from = self.newest + (tile + 1) * TILE;
                    self.ahead.push(Ahead { slot: r.slot, from, tile, page: p });
                }
            }
        }
        for r in reaches.iter().filter(|r| r.writes) {
            for tile in r.pos0 / TILE..r.tiles().end {
                let p = page(r.slot, tile);
                self.ensure(p, None)?;
                // A tile that begins in this step is a new tile, whatever the
                // page held before it was handed out again.
                if tile * TILE >= r.pos0 {
                    self.full[p as usize] = false;
                }
            }
        }
        // After a step whose rows are all accepted the tiles they filled are
        // settled; the others wait for the commit.
        for r in reaches.iter().filter(|r| r.settles) {
            self.settle(|tile| page(r.slot, tile), r.pos0 + r.rows);
        }
        Ok(())
    }

    /// Gives `page` an exact slot for a page that comes in exact, whatever the
    /// pool kept for its number before. A full page is settled, so it may be
    /// converted to make room like any full page the cache holds; a `tail` (a
    /// partly used page of the prefix cache) is not: it stays exact until the
    /// pool frees it, as a tail that never left the device does.
    pub(crate) fn adopt(&mut self, page: u32, tail: bool) -> Result<u32, Exhausted> {
        self.release(page);
        let slot = self.ensure(page, None)?;
        if !tail {
            self.settle_page(page);
        }
        Ok(slot)
    }

    /// Converts `page` now when it is exact, as when the window of a sequence
    /// has left it: the job waits in [`Forms::take_jobs`].
    pub(crate) fn convert_page(&mut self, page: u32) {
        if self.exact_of[page as usize] != NONE {
            self.convert(page);
        }
    }

    /// The tile that the rows of the draft window of `slot` read converted
    /// from a position inside the window on, in the step last prepared.
    pub(crate) fn ahead(&self, slot: u32) -> Option<Ahead> {
        self.ahead.iter().copied().find(|a| a.slot == slot)
    }

    /// The conversions that wait, in the order they must run.
    pub(crate) fn take_jobs(&mut self) -> Vec<Convert> {
        std::mem::take(&mut self.jobs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAG: u32 = 1 << 31;

    /// One sequence in slot 0 whose tile `t` is page `t`.
    fn reach(pos0: u32, rows: u32) -> Reach {
        Reach { slot: 0, pos0, rows, writes: true, settles: true }
    }

    fn page(_slot: u32, tile: u32) -> u32 {
        tile
    }

    /// The pages of slot 1 in the tests that need two sequences.
    fn other(_slot: u32, tile: u32) -> u32 {
        10 + tile
    }

    fn second(pos0: u32, rows: u32) -> Reach {
        Reach { slot: 1, ..reach(pos0, rows) }
    }

    #[test]
    fn a_tile_is_converted_when_it_leaves_the_window_of_the_newest_cells() {
        // 256 newest cells: at n cells the tiles below (n - 256) / 128 are converted.
        let forms = Forms::new(8, 32, 256);
        assert!(forms.converted_tiles(0).is_empty());
        assert!(
            forms.converted_tiles(383).is_empty(),
            "tile 1 ends at 256, the window starts at 127"
        );
        assert!(forms.converted_tiles(511).is_empty());
        assert_eq!(forms.converted_tiles(512), 1..2);
        assert_eq!(forms.converted_tiles(639), 1..2);
        assert_eq!(forms.converted_tiles(640), 1..3);

        // A prompt of 768 cells in steps of 128.
        let mut forms = Forms::new(8, 32, 256);
        for step in 0..6 {
            forms.prepare(&[reach(step * 128, 128)], page, true).unwrap();
        }
        let jobs = forms.take_jobs();
        assert_eq!(
            jobs.iter().map(|j| j.page).collect::<Vec<_>>(),
            [1, 2],
            "tile 1 goes at 512 cells, tile 2 at 640"
        );
        assert!(forms.slot_of(1).is_none() && forms.slot_of(2).is_none());
        assert!((3..6).all(|p| forms.slot_of(p).is_some()), "the window stays exact");
        assert!(forms.slot_of(0).is_some(), "the first tile stays exact");
        assert_eq!(forms.in_use(), 4);
        assert_eq!(forms.entry(1, FLAG), 1 | FLAG);
        assert_eq!(forms.entry(3, FLAG), forms.slot_of(3).unwrap());
    }

    #[test]
    fn sequences_never_need_more_slots_than_the_bound() {
        let (slots, rows) = (2, 256);
        let capacity = exact_slots(256, slots, rows, 0);
        let mut forms = Forms::new(capacity, 400, 256);
        let mut pos = [0u32; 2];
        // Two sequences with pages of their own (slot 1 from page 200), a
        // prompt of 3 and 5 steps and then one row at a time.
        let page = |slot: u32, tile: u32| slot * 200 + tile;
        let mut high = 0;
        for step in 0..1500u32 {
            let reaches: Vec<Reach> = (0..2)
                .map(|slot| {
                    let rows = if step < 3 + 2 * slot { 256 } else { 1 };
                    Reach { slot, pos0: pos[slot as usize], rows, writes: true, settles: true }
                })
                .collect();
            forms.prepare(&reaches, page, true).unwrap();
            for r in &reaches {
                pos[r.slot as usize] += r.rows;
            }
            high = high.max(forms.in_use());
            forms.take_jobs();
        }
        assert!(high <= capacity, "{high} of {capacity}");
    }

    #[test]
    fn a_full_page_that_nobody_reads_is_converted_when_the_pool_is_short() {
        let mut forms = Forms::new(2, 16, 256);
        forms.prepare(&[reach(0, 256)], page, true).unwrap();
        assert_eq!(forms.in_use(), 2);
        let (slot_of_0, slot_of_1) = (forms.slot_of(0).unwrap(), forms.slot_of(1).unwrap());
        // Another sequence writes a page of its own: the page used longest ago
        // makes room.
        forms.prepare(&[second(0, 1)], other, true).unwrap();
        assert_eq!(forms.take_jobs(), [Convert { exact: slot_of_0, page: 0 }]);
        assert_eq!(forms.slot_of(0), None);
        assert_eq!(forms.slot_of(1), Some(slot_of_1));
        assert_eq!(forms.slot_of(10), Some(slot_of_0), "the slot went to the new page");
    }

    #[test]
    fn a_partly_used_page_is_never_converted_for_room() {
        let mut forms = Forms::new(1, 16, 256);
        forms.prepare(&[reach(0, 5)], page, true).unwrap();
        let error = forms.prepare(&[second(0, 1)], other, true).unwrap_err();
        assert_eq!(error, Exhausted { capacity: 1, partial: 1, in_step: 0 });
        assert!(forms.take_jobs().is_empty());
    }

    #[test]
    fn a_tail_that_comes_back_exact_is_never_converted_for_room() {
        // Page 3 comes back from RAM as a tail of the prefix cache, page 4 as
        // a full page; later another sequence needs a slot.
        let mut forms = Forms::new(2, 16, 256);
        let tail = forms.adopt(3, true).unwrap();
        let full = forms.adopt(4, false).unwrap();
        forms.prepare(&[second(0, 1)], other, true).unwrap();
        assert_eq!(
            forms.take_jobs(),
            [Convert { exact: full, page: 4 }],
            "the full page makes room"
        );
        assert_eq!(forms.slot_of(3), Some(tail), "the tail stays exact");
        // With the tail alone in the pool there is no room: it is not given up.
        let mut forms = Forms::new(1, 16, 256);
        forms.adopt(3, true).unwrap();
        let error = forms.prepare(&[second(0, 1)], other, true).unwrap_err();
        assert_eq!(error, Exhausted { capacity: 1, partial: 1, in_step: 0 });
        assert!(forms.take_jobs().is_empty());
    }

    #[test]
    fn the_pages_of_the_step_are_not_given_up() {
        let mut forms = Forms::new(2, 16, 256);
        forms.prepare(&[reach(0, 256)], page, true).unwrap();
        // The same sequence goes on and needs a third page: its own pages are
        // read in this step, so none is converted for it.
        let error = forms.prepare(&[reach(256, 1)], page, true).unwrap_err();
        assert_eq!(error, Exhausted { capacity: 2, partial: 0, in_step: 2 });
    }

    #[test]
    fn a_freed_page_gives_its_slot_back() {
        let mut forms = Forms::new(1, 16, 256);
        forms.prepare(&[reach(0, 5)], page, true).unwrap();
        assert_eq!(forms.in_use(), 1);
        forms.release(0);
        assert_eq!(forms.in_use(), 0);
        assert_eq!(forms.slot_of(0), None);
        forms.release(0);
        assert_eq!(forms.in_use(), 0, "releasing twice changes nothing");
        forms.prepare(&[reach(0, 5)], page, true).unwrap();
        assert_eq!(forms.slot_of(0), Some(0));
    }

    #[test]
    fn a_page_is_converted_for_every_sequence_that_holds_it() {
        let mut forms = Forms::new(8, 32, 128);
        // Slot 0 wrote six pages; with 128 newest cells it keeps tiles 0, 4 and 5.
        for step in 0..6 {
            forms.prepare(&[reach(step * 128, 128)], page, true).unwrap();
        }
        forms.prepare(&[reach(768, 1)], page, true).unwrap();
        forms.take_jobs();
        assert!((1..5).all(|p| forms.slot_of(p).is_none()));
        // Slot 1 reads the same pages from a shorter sequence, whose window
        // would hold pages 2 and 3 exact: it finds them converted.
        let shared = |_slot: u32, tile: u32| tile;
        let reader = Reach { slot: 1, pos0: 384, rows: 1, writes: false, settles: false };
        forms.prepare(&[reader], shared, true).unwrap();
        assert!(forms.take_jobs().is_empty());
        assert_eq!(forms.entry(2, FLAG), 2 | FLAG);
        assert_eq!(forms.entry(3, FLAG), 3 | FLAG);
    }

    #[test]
    fn a_page_handed_out_again_is_not_full_until_its_new_tile_is() {
        let mut forms = Forms::new(2, 16, 256);
        forms.prepare(&[reach(0, 128)], page, true).unwrap();
        assert!(forms.slot_of(0).is_some());
        // The pool freed page 0 and gave it to another sequence between two
        // records without the device being told: it starts a tile in this step.
        forms.prepare(&[second(0, 5)], |_slot, _tile| 0, true).unwrap();
        // Another sequence needs the second slot and finds none to give up: the
        // page holds 5 cells of its new tile.
        forms.prepare(&[Reach { slot: 2, ..reach(0, 1) }], |_slot, tile| 5 + tile, true).unwrap();
        let error = forms.prepare(&[Reach { slot: 3, ..reach(0, 1) }], |_s, tile| 9 + tile, true);
        assert_eq!(error.unwrap_err().partial, 2, "neither page is full");
    }

    #[test]
    fn a_draft_window_across_a_conversion_has_the_tile_both_ways() {
        // 128 newest cells: tile 1 is converted for rows from position 384 on.
        let mut forms = Forms::new(8, 32, 128);
        forms.prepare(&[reach(0, 380)], page, true).unwrap();
        assert!(forms.take_jobs().is_empty());
        let slot_of_1 = forms.slot_of(1).unwrap();
        // A window of 8 rows from 380: rows 380 to 383 read tile 1 exact, rows
        // 384 to 387 converted. The tile is converted now and keeps its slot.
        let window = Reach { settles: false, ..reach(380, 8) };
        forms.prepare(&[window], page, true).unwrap();
        assert_eq!(forms.take_jobs(), [Convert { exact: slot_of_1, page: 1 }]);
        assert_eq!(forms.ahead(0), Some(Ahead { slot: 0, from: 384, tile: 1, page: 1 }));
        assert_eq!(forms.slot_of(1), Some(slot_of_1), "the rows before 384 read the exact entry");
        // Two rows are accepted: the next window starts at 382 and has the
        // tile both ways again; a step that starts at 384 gives the slot up.
        forms.prepare(&[Reach { settles: false, ..reach(382, 8) }], page, true).unwrap();
        assert_eq!(forms.ahead(0).map(|a| a.from), Some(384));
        assert_eq!(forms.take_jobs().len(), 1);
        forms.prepare(&[reach(384, 1)], page, true).unwrap();
        assert_eq!(forms.ahead(0), None);
        assert_eq!(forms.take_jobs(), [Convert { exact: slot_of_1, page: 1 }]);
        assert_eq!(forms.slot_of(1), None);
        // A window inside one stretch between two conversions, a row alone and
        // the rows of a prompt have no such tile, and neither has a group that
        // does not ask for it.
        let mut forms = Forms::new(8, 32, 128);
        forms.prepare(&[reach(0, 380)], page, true).unwrap();
        forms.prepare(&[Reach { settles: false, ..reach(380, 4) }], page, true).unwrap();
        assert_eq!(forms.ahead(0), None);
        forms.prepare(&[Reach { settles: false, ..reach(380, 8) }], page, false).unwrap();
        assert_eq!(forms.ahead(0), None);
        forms.prepare(&[reach(380, 8)], page, true).unwrap();
        assert_eq!(forms.ahead(0), None);
        assert!(forms.take_jobs().is_empty());
        assert_eq!(window_limit(32), 32);
        assert_eq!(window_limit(256), 128);
    }

    #[test]
    fn a_replay_writes_nothing() {
        let mut forms = Forms::new(4, 16, 256);
        forms.prepare(&[Reach { writes: false, ..reach(0, 300) }], page, true).unwrap();
        assert_eq!(forms.in_use(), 0);
    }

    #[test]
    fn a_page_is_settled_when_the_rows_that_filled_it_are_certain() {
        let mut forms = Forms::new(1, 16, 256);
        // A window with drafts is not certain: its page is not given up until a
        // step starts past it.
        forms.prepare(&[Reach { settles: false, ..reach(0, 128) }], page, true).unwrap();
        assert_eq!(forms.prepare(&[second(0, 1)], other, true).unwrap_err().partial, 1);
        let commit = Reach { slot: 0, pos0: 128, rows: 0, writes: false, settles: false };
        forms.prepare(&[commit], page, true).unwrap();
        forms.prepare(&[second(0, 1)], other, true).unwrap();
        assert_eq!(forms.take_jobs().len(), 1);

        // The rows of a prompt are certain: the page they fill can go at once.
        let mut forms = Forms::new(1, 16, 256);
        forms.prepare(&[reach(0, 128)], page, true).unwrap();
        forms.prepare(&[second(0, 1)], other, true).unwrap();
        assert_eq!(forms.take_jobs().len(), 1);
    }
}

#[cfg(test)]
mod model;
