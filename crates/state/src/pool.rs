//! The pool of pages.

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
    /// Free pages; the last entry is handed out next. Low numbers go out
    /// first, so a pool that is mostly empty stays compact at the front.
    free: Vec<u32>,
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
            free: (0..n_pages).rev().collect(),
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

    /// Takes a free page with one holder, or `None` when the pool is empty.
    pub(crate) fn alloc(&mut self) -> Option<PageId> {
        let id = self.free.pop()?;
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
            self.free.push(id.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[should_panic(expected = "released more often")]
    fn releasing_a_free_page_is_a_bug() {
        let mut pool = PagePool::new(1, 4);
        let a = pool.alloc().unwrap();
        pool.release(a);
        pool.release(a);
    }
}
