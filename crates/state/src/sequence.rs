//! A sequence's page table.

use std::ops::Range;

use crate::pool::{PageId, PagePool};

/// A page copy the devices must perform before the sequence writes: the first
/// `cells` cells of page `from` go to page `to`, for every part of the state
/// that lives in pages.
///
/// It is issued when a sequence is about to write into a page that others
/// read too. The copy must be enqueued on the devices before any later write
/// to either page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOnWrite {
    pub from: PageId,
    pub to: PageId,
    pub cells: u32,
}

/// What [`Sequence::append`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grown {
    /// The position of the first new cell.
    pub first_cell: u64,
    /// The copy to perform before the new cells are written, if the write
    /// starts inside a page that was shared.
    pub copy: Option<CopyOnWrite>,
}

/// The pool cannot supply the pages an operation needs. Nothing was changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{needed} pages are needed and {free} are free")]
pub struct OutOfPages {
    pub needed: u32,
    pub free: u32,
}

/// The pages of one sequence, in position order, and its length in cells.
///
/// A sequence holds one reference to each of its pages. It must be given back
/// to the pool with [`Sequence::release`]; dropping one that still holds pages
/// is a bug that debug builds report.
#[derive(Debug, Default)]
pub struct Sequence {
    pages: Vec<PageId>,
    len: u64,
    /// The device copies of the page table are current for `pages[..clean]`.
    clean: usize,
}

impl Sequence {
    /// An empty sequence.
    pub fn new() -> Self {
        Self::default()
    }

    /// A sequence of `len` cells over existing pages, which gain a holder
    /// each. This is how a request starts from a cached prefix.
    ///
    /// # Panics
    ///
    /// Panics when `pages` is not exactly the number of pages `len` cells
    /// take.
    pub fn from_shared(pool: &mut PagePool, pages: &[PageId], len: u64) -> Self {
        assert_eq!(pages.len() as u64, len.div_ceil(u64::from(pool.cells_per_page())));
        for page in pages {
            pool.retain(*page);
        }
        Self { pages: pages.to_vec(), len, clean: 0 }
    }

    /// The length in cells.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The page table: entry `i` holds cells `i * cells_per_page ..`.
    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }

    /// The page and the index within it of cell `cell`.
    pub fn locate(&self, pool: &PagePool, cell: u64) -> Option<(PageId, u32)> {
        if cell >= self.len {
            return None;
        }
        let per_page = u64::from(pool.cells_per_page());
        Some((self.pages[(cell / per_page) as usize], (cell % per_page) as u32))
    }

    /// What appending `n` cells takes: the new pages, and whether the write
    /// starts inside a partly used page that others hold too, which this
    /// sequence then copies.
    fn growth(&self, pool: &PagePool, n: u64) -> (u64, bool) {
        if n == 0 {
            return (0, false);
        }
        let per_page = u64::from(pool.cells_per_page());
        let used_in_last = self.len % per_page;
        let new_pages = (self.len + n).div_ceil(per_page) - self.pages.len() as u64;
        let shared_tail =
            used_in_last != 0 && pool.refcount(*self.pages.last().expect("a partly used page")) > 1;
        (new_pages, shared_tail)
    }

    /// How many pages [`Sequence::append`] of `n` cells would take from the
    /// pool: what a step claims for this sequence.
    pub fn pages_to_append(&self, pool: &PagePool, n: u64) -> u32 {
        let (new_pages, shared_tail) = self.growth(pool, n);
        u32::try_from(new_pages + u64::from(shared_tail)).unwrap_or(u32::MAX)
    }

    /// How many of this sequence's pages no other sequence holds: the pages
    /// that become free, or evictable where the prefix index holds them too,
    /// when this sequence lets go of everything.
    ///
    /// The count is what eviction can really free as long as sequences enter
    /// their pages through [`PrefixIndex::publish`](crate::PrefixIndex::publish):
    /// then a sequence that holds a cached page holds the cached pages before
    /// it as well, and no cached page it leaves behind has a holder below it.
    pub fn releasable(&self, pool: &PagePool) -> u32 {
        self.pages
            .iter()
            .filter(|page| pool.refcount(**page) == 1 + u32::from(pool.is_cached(**page)))
            .count() as u32
    }

    /// Replaces page-table entry `index` by `page`, which holds the same
    /// content and gains a holder; the page that was there loses one. This is
    /// how a sequence gives up its own copy of a page that turned out to be
    /// cached already.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not a full page of this sequence.
    pub fn adopt(&mut self, pool: &mut PagePool, index: usize, page: PageId) {
        let full = (self.len / u64::from(pool.cells_per_page())) as usize;
        assert!(index < full, "page {index} of a sequence with {full} full pages");
        if self.pages[index] == page {
            return;
        }
        pool.retain(page);
        pool.release(std::mem::replace(&mut self.pages[index], page));
        self.clean = self.clean.min(index);
    }

    /// Adds `n` cells at the end. Either everything needed is allocated or
    /// nothing changes.
    pub fn append(&mut self, pool: &mut PagePool, n: u64) -> Result<Grown, OutOfPages> {
        let first_cell = self.len;
        if n == 0 {
            return Ok(Grown { first_cell, copy: None });
        }
        let per_page = u64::from(pool.cells_per_page());
        let used_in_last = (self.len % per_page) as u32;
        // Writing starts inside the last page when it is partly used. If
        // others hold that page too, this sequence gets its own copy.
        let (new_pages, shared_tail) = self.growth(pool, n);

        let needed = new_pages + u64::from(shared_tail);
        if needed > u64::from(pool.free_pages()) {
            return Err(OutOfPages {
                needed: u32::try_from(needed).unwrap_or(u32::MAX),
                free: pool.free_pages(),
            });
        }

        let mut copy = None;
        if shared_tail {
            let index = self.pages.len() - 1;
            let from = self.pages[index];
            let to = pool.alloc().expect("checked above");
            self.pages[index] = to;
            pool.release(from);
            self.clean = self.clean.min(index);
            copy = Some(CopyOnWrite { from, to, cells: used_in_last });
        }
        for _ in 0..new_pages {
            self.pages.push(pool.alloc().expect("checked above"));
        }
        self.len += n;
        Ok(Grown { first_cell, copy })
    }

    /// Cuts the sequence to `len` cells and lets go of the pages past it.
    /// Rejected draft tokens are removed this way. A longer `len` is ignored.
    pub fn truncate(&mut self, pool: &mut PagePool, len: u64) {
        if len >= self.len {
            return;
        }
        let keep = len.div_ceil(u64::from(pool.cells_per_page())) as usize;
        for page in self.pages.drain(keep..) {
            pool.release(page);
        }
        self.len = len;
        self.clean = self.clean.min(keep);
    }

    /// A second sequence over the same pages. Neither copies anything until
    /// one of them writes into the page they both end in.
    pub fn fork(&self, pool: &mut PagePool) -> Sequence {
        Self::from_shared(pool, &self.pages, self.len)
    }

    /// A second sequence over the first `len` cells.
    ///
    /// # Panics
    ///
    /// Panics when `len` is past the end.
    pub fn fork_prefix(&self, pool: &mut PagePool, len: u64) -> Sequence {
        assert!(len <= self.len, "prefix of {len} cells of a sequence of {}", self.len);
        let pages = len.div_ceil(u64::from(pool.cells_per_page())) as usize;
        Self::from_shared(pool, &self.pages[..pages], len)
    }

    /// The range of page-table entries that changed since the last call, for
    /// the upload of the table to the devices.
    pub fn take_dirty(&mut self) -> Range<usize> {
        let dirty = self.clean..self.pages.len();
        self.clean = self.pages.len();
        dirty
    }

    /// Gives every page back to the pool.
    pub fn release(mut self, pool: &mut PagePool) {
        for page in self.pages.drain(..) {
            pool.release(page);
        }
        self.len = 0;
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        debug_assert!(
            self.pages.is_empty() || std::thread::panicking(),
            "a sequence holding {} pages was dropped without Sequence::release",
            self.pages.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_page_by_page_and_refuses_atomically() {
        let mut pool = PagePool::new(3, 4);
        let mut seq = Sequence::new();
        assert_eq!(seq.append(&mut pool, 5), Ok(Grown { first_cell: 0, copy: None }));
        assert_eq!(seq.pages().len(), 2);
        assert_eq!(seq.append(&mut pool, 3), Ok(Grown { first_cell: 5, copy: None }));
        assert_eq!(seq.pages().len(), 2, "the second page had room");
        assert_eq!(seq.append(&mut pool, 9), Err(OutOfPages { needed: 3, free: 1 }));
        assert_eq!((seq.len(), seq.pages().len(), pool.free_pages()), (8, 2, 1));
        assert_eq!(seq.locate(&pool, 6), Some((seq.pages()[1], 2)));
        assert_eq!(seq.locate(&pool, 8), None);
        seq.release(&mut pool);
        assert_eq!(pool.free_pages(), 3);
    }

    #[test]
    fn a_fork_copies_only_the_page_it_writes_into() {
        let mut pool = PagePool::new(8, 4);
        let mut a = Sequence::new();
        a.append(&mut pool, 6).unwrap();
        let mut b = a.fork(&mut pool);
        assert_eq!(pool.free_pages(), 6, "a fork allocates nothing");

        // b writes into the shared, partly used last page: it gets a copy of
        // the 2 cells in use. The full first page stays shared.
        let grown = b.append(&mut pool, 1).unwrap();
        let copy = grown.copy.expect("the last page was shared");
        assert_eq!((copy.from, copy.cells), (a.pages()[1], 2));
        assert_eq!(copy.to, b.pages()[1]);
        assert_eq!(a.pages()[0], b.pages()[0]);
        assert_ne!(a.pages()[1], b.pages()[1]);

        // a is now the only holder of its last page and writes in place.
        assert_eq!(a.append(&mut pool, 1).unwrap().copy, None);

        a.release(&mut pool);
        b.release(&mut pool);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn a_fork_at_a_page_boundary_needs_no_copy() {
        let mut pool = PagePool::new(8, 4);
        let mut a = Sequence::new();
        a.append(&mut pool, 8).unwrap();
        let mut b = a.fork(&mut pool);
        assert_eq!(b.append(&mut pool, 3).unwrap().copy, None);
        assert_eq!(b.pages().len(), 3);
        a.release(&mut pool);
        b.release(&mut pool);
    }

    #[test]
    fn truncating_into_a_shared_page_copies_on_the_next_write() {
        let mut pool = PagePool::new(8, 4);
        let mut a = Sequence::new();
        a.append(&mut pool, 8).unwrap();
        let mut b = a.fork(&mut pool);
        // b rolls back three cells: its last page is full for a, partial for b.
        b.truncate(&mut pool, 5);
        assert_eq!(b.pages().len(), 2);
        let grown = b.append(&mut pool, 2).unwrap();
        assert_eq!(grown.first_cell, 5);
        assert_eq!(grown.copy.map(|c| (c.from, c.cells)), Some((a.pages()[1], 1)));
        // a still reads its 8 cells from untouched pages.
        assert_eq!(a.len(), 8);
        a.release(&mut pool);
        b.release(&mut pool);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn a_claim_is_what_an_append_takes() {
        let mut pool = PagePool::new(8, 4);
        let mut a = Sequence::new();
        assert_eq!(a.pages_to_append(&pool, 0), 0);
        assert_eq!(a.pages_to_append(&pool, 5), 2);
        a.append(&mut pool, 5).unwrap();
        assert_eq!(a.pages_to_append(&pool, 3), 0, "the second page has room");
        assert_eq!(a.pages_to_append(&pool, 4), 1);
        // A fork writes into a copy of the shared, partly used page.
        let mut b = a.fork(&mut pool);
        assert_eq!(b.pages_to_append(&pool, 1), 1);
        assert_eq!(b.pages_to_append(&pool, 4), 2);
        for n in [1u64, 3, 4, 9] {
            let before = pool.free_pages();
            let claim = b.pages_to_append(&pool, n);
            b.append(&mut pool, n).unwrap();
            assert_eq!(before - pool.free_pages(), claim, "{n} cells");
        }
        // What each would free: a shares its first page with b and no other.
        assert_eq!(a.releasable(&pool), 1);
        a.release(&mut pool);
        assert_eq!(b.releasable(&pool), b.pages().len() as u32);
        b.release(&mut pool);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn adopting_a_page_swaps_one_table_entry() {
        let mut pool = PagePool::new(8, 4);
        let (mut a, mut b) = (Sequence::new(), Sequence::new());
        a.append(&mut pool, 9).unwrap();
        b.append(&mut pool, 9).unwrap();
        b.take_dirty();
        // b computed the same first page as a: it reads a's from now on.
        let own = b.pages()[0];
        b.adopt(&mut pool, 0, a.pages()[0]);
        assert_eq!(b.pages()[0], a.pages()[0]);
        assert_eq!((pool.refcount(a.pages()[0]), pool.refcount(own)), (2, 0));
        assert_eq!(b.take_dirty(), 0..3, "the table is uploaded again from the swapped entry");
        b.adopt(&mut pool, 0, a.pages()[0]);
        assert_eq!(pool.refcount(a.pages()[0]), 2, "adopting the page it has changes nothing");
        a.release(&mut pool);
        b.release(&mut pool);
        assert_eq!(pool.free_pages(), 8);
    }

    #[test]
    fn dirty_ranges_cover_what_changed() {
        let mut pool = PagePool::new(8, 4);
        let mut a = Sequence::new();
        a.append(&mut pool, 9).unwrap();
        assert_eq!(a.take_dirty(), 0..3);
        assert_eq!(a.take_dirty(), 3..3);
        a.append(&mut pool, 4).unwrap();
        assert_eq!(a.take_dirty(), 3..4);
        a.truncate(&mut pool, 2);
        a.append(&mut pool, 4).unwrap();
        assert_eq!(a.take_dirty(), 1..2);

        // Replacing a shared last page dirties its entry.
        let mut b = a.fork(&mut pool);
        assert_eq!(b.take_dirty(), 0..2);
        b.append(&mut pool, 1).unwrap();
        assert_eq!(b.take_dirty(), 1..2);
        a.release(&mut pool);
        b.release(&mut pool);
    }
}
