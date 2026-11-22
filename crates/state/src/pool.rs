//! The pool of pages.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// The number of a page. Devices index their state arrays with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageId(pub u32);

/// A fixed set of pages with reference counts.
///
/// The capacity is decided once, from the memory budget, when the model is
/// loaded. Running out of pages is an expected condition that the scheduler
/// handles (evict cached prefixes, park a sequence, refuse a request); it is
/// never an allocation failure on a device.
#[derive(Debug)]
pub struct PagePool {
    cells_per_page: u32,
    refcount: Vec<u32>,
    /// Whether the prefix index is one of the holders of a page.
    cached: Vec<bool>,
    /// Free pages; the lowest is handed out next, so a pool that is mostly
    /// empty stays compact at the front. A model may keep its first pages in
    /// faster memory than the rest (the K and V of a context beyond VRAM): a
    /// sequence that starts gets the first free pages whatever was freed last.
    free: BinaryHeap<Reverse<u32>>,
    /// Pages that were freed since the devices were last told, each once.
    freed: Vec<u32>,
    /// Whether a page is in `freed`.
    unreported: Vec<bool>,
}

impl PagePool {
    /// A pool of `n_pages` pages of `cells_per_page` cells each.
    ///
    /// # Panics
    ///
    /// Panics when `cells_per_page` is 0.
    pub fn new(n_pages: u32, cells_per_page: u32) -> Self {
        assert!(cells_per_page > 0, "a page holds at least one cell");
        Self {
            cells_per_page,
            refcount: vec![0; n_pages as usize],
            cached: vec![false; n_pages as usize],
            free: (0..n_pages).map(Reverse).collect(),
            freed: Vec::new(),
            unreported: vec![false; n_pages as usize],
        }
    }

    pub fn cells_per_page(&self) -> u32 {
        self.cells_per_page
    }

    pub fn capacity(&self) -> u32 {
        self.refcount.len() as u32
    }

    pub fn free_pages(&self) -> u32 {
        self.free.len() as u32
    }

    /// How many holders page `id` has.
    pub fn refcount(&self, id: PageId) -> u32 {
        self.refcount[id.0 as usize]
    }

    /// Whether the prefix index holds page `id` (as a cached page or as a
    /// tail).
    pub fn is_cached(&self, id: PageId) -> bool {
        self.cached[id.0 as usize]
    }

    /// Records whether the prefix index is among the holders of page `id`.
    pub(crate) fn set_cached(&mut self, id: PageId, cached: bool) {
        debug_assert!(self.refcount[id.0 as usize] > 0, "a free page is not cached");
        self.cached[id.0 as usize] = cached;
    }

    /// Takes a free page with one holder, or `None` when the pool is empty.
    pub(crate) fn alloc(&mut self) -> Option<PageId> {
        let Reverse(id) = self.free.pop()?;
        debug_assert_eq!(self.refcount[id as usize], 0);
        self.refcount[id as usize] = 1;
        Some(PageId(id))
    }

    /// Adds a holder to a page that has at least one.
    pub(crate) fn retain(&mut self, id: PageId) {
        let count = &mut self.refcount[id.0 as usize];
        assert!(*count > 0, "page {} is free and cannot be shared", id.0);
        *count += 1;
    }

    /// Removes a holder; the page becomes free when the last one is gone.
    pub(crate) fn release(&mut self, id: PageId) {
        let count = &mut self.refcount[id.0 as usize];
        assert!(*count > 0, "page {} was released more often than held", id.0);
        *count -= 1;
        if *count == 0 {
            debug_assert!(!self.cached[id.0 as usize], "the index let go without saying so");
            self.free.push(Reverse(id.0));
            if !std::mem::replace(&mut self.unreported[id.0 as usize], true) {
                self.freed.push(id.0);
            }
        }
    }

    /// Takes up to `max` pages that were freed since the last call and are
    /// still free, for the devices to give back what they keep per page. A
    /// page that has a holder again is not named: whatever a device kept for it
    /// belongs to the new holder. The pages beyond `max` stay for the next call.
    pub fn take_freed(&mut self, max: usize, out: &mut Vec<u32>) {
        let mut taken = 0;
        while taken < self.freed.len() && out.len() < max {
            let id = self.freed[taken];
            taken += 1;
            self.unreported[id as usize] = false;
            if self.refcount[id as usize] == 0 {
                out.push(id);
            }
        }
        self.freed.drain(..taken);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lowest_free_page_goes_out_first() {
        let mut pool = PagePool::new(8, 4);
        let pages: Vec<PageId> = (0..6).map(|_| pool.alloc().unwrap()).collect();
        pool.release(pages[1]);
        pool.release(pages[4]);
        assert_eq!(pool.alloc(), Some(PageId(1)), "page 4 was freed last, page 1 is lower");
        assert_eq!(pool.alloc(), Some(PageId(4)));
        assert_eq!(pool.alloc(), Some(PageId(6)));
    }

    #[test]
    fn pages_are_free_again_when_the_last_holder_lets_go() {
        let mut pool = PagePool::new(2, 4);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        assert_eq!((a, b), (PageId(0), PageId(1)));
        assert_eq!(pool.alloc(), None);

        pool.retain(a);
        pool.release(a);
        assert_eq!(pool.free_pages(), 0, "one holder is left");
        pool.release(a);
        assert_eq!(pool.free_pages(), 1);
        assert_eq!(pool.alloc(), Some(a));
    }

    #[test]
    fn pages_are_named_as_freed_once_and_only_while_they_are_free() {
        let mut pool = PagePool::new(4, 4);
        let (a, b, c) = (pool.alloc().unwrap(), pool.alloc().unwrap(), pool.alloc().unwrap());
        let mut out = Vec::new();
        pool.take_freed(8, &mut out);
        assert!(out.is_empty(), "nothing was freed yet");

        pool.release(a);
        pool.release(b);
        pool.release(c);
        // `a` has an owner again before the devices are told.
        assert_eq!(pool.alloc(), Some(a), "the lowest free page is handed out first");
        pool.take_freed(8, &mut out);
        assert_eq!(out, [b.0, c.0]);
        out.clear();
        pool.take_freed(8, &mut out);
        assert!(out.is_empty(), "each page is named once");

        // Freed again after being named: named again.
        pool.release(a);
        pool.take_freed(8, &mut out);
        assert_eq!(out, [a.0]);
    }

    #[test]
    fn what_does_not_fit_in_one_record_waits_for_the_next() {
        let mut pool = PagePool::new(8, 4);
        let ids: Vec<PageId> = (0..5).map(|_| pool.alloc().unwrap()).collect();
        for id in &ids {
            pool.release(*id);
        }
        let mut out = Vec::new();
        pool.take_freed(3, &mut out);
        assert_eq!(out.len(), 3);
        let mut rest = Vec::new();
        pool.take_freed(3, &mut rest);
        assert_eq!(rest.len(), 2);
        out.extend(rest);
        out.sort_unstable();
        assert_eq!(out, [0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "released more often")]
    fn releasing_a_free_page_is_a_bug() {
        let mut pool = PagePool::new(1, 4);
        let a = pool.alloc().unwrap();
        pool.release(a);
        pool.release(a);
    }
}
