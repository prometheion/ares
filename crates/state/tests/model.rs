//! A randomised check of the whole bookkeeping against a simulated device.
//!
//! The simulated model is the simplest one with the property real models
//! have: the state of a cell is a function of all tokens up to it. The test
//! drives sequences through random appends, rollbacks, forks, prefix-cache
//! hits and evictions, performs the page copies the bookkeeping asks for on a
//! simulated device array, and after every operation checks that
//!
//! - every sequence reads, through its page table, exactly the state its own
//!   tokens produce (so sharing and copy-on-write never mix up sequences),
//! - nothing is ever written into a page that someone else also holds,
//! - the pool's free count equals the pages nobody references (no leaks).

use std::collections::HashSet;

use ares_state::{ChainHash, PageId, PagePool, PrefixIndex, Sequence};

const PER_PAGE: u32 = 4;
const PAGES: u32 = 96;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The state of the cell that follows `previous` and holds `token`.
fn cell_state(previous: u64, token: u32) -> u64 {
    (previous ^ u64::from(token)).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23) ^ 0x51
}

struct Live {
    tokens: Vec<u32>,
    seq: Sequence,
}

struct World {
    pool: PagePool,
    index: PrefixIndex,
    root: ChainHash,
    /// The device array: one value per cell of every page.
    device: Vec<u64>,
    live: Vec<Live>,
}

impl World {
    fn write(&mut self, page: PageId, offset: u32, value: u64) {
        assert_eq!(self.pool.refcount(page), 1, "write into a shared page");
        self.device[(page.0 * PER_PAGE + offset) as usize] = value;
    }

    /// Appends `tokens` to sequence `i` and computes their cells. Returns
    /// false, with nothing changed, when the pool is too small.
    fn append(&mut self, i: usize, tokens: &[u32]) -> bool {
        let Ok(grown) = self.live[i].seq.append(&mut self.pool, tokens.len() as u64) else {
            return false;
        };
        if let Some(copy) = grown.copy {
            for cell in 0..copy.cells {
                let value = self.device[(copy.from.0 * PER_PAGE + cell) as usize];
                self.write(copy.to, cell, value);
            }
        }
        let mut state = self.state_before(i, grown.first_cell);
        for (k, token) in tokens.iter().enumerate() {
            state = cell_state(state, *token);
            let cell = grown.first_cell + k as u64;
            let (page, offset) = self.live[i].seq.locate(&self.pool, cell).unwrap();
            self.write(page, offset, state);
        }
        self.live[i].tokens.extend_from_slice(tokens);
        true
    }

    /// The state of the cell before `cell` of sequence `i`, as the device
    /// holds it (0 before the first cell).
    fn state_before(&self, i: usize, cell: u64) -> u64 {
        if cell == 0 {
            return 0;
        }
        let (page, offset) = self.live[i].seq.locate(&self.pool, cell - 1).unwrap();
        self.device[(page.0 * PER_PAGE + offset) as usize]
    }

    /// Registers every full page of sequence `i` in the prefix index.
    fn cache(&mut self, i: usize) {
        let mut hash = self.root;
        let live = &self.live[i];
        let per_page = self.pool.cells_per_page() as usize;
        for (p, page_tokens) in live.tokens.chunks_exact(per_page).enumerate() {
            let parent = hash;
            hash = hash.extend(page_tokens);
            self.index.insert(&mut self.pool, parent, hash, live.seq.pages()[p]);
        }
    }

    fn check(&self) {
        let mut referenced = HashSet::new();
        for live in &self.live {
            assert_eq!(live.seq.len(), live.tokens.len() as u64);
            let mut state = 0;
            for (cell, token) in live.tokens.iter().enumerate() {
                state = cell_state(state, *token);
                let (page, offset) = live.seq.locate(&self.pool, cell as u64).unwrap();
                assert_eq!(
                    self.device[(page.0 * PER_PAGE + offset) as usize],
                    state,
                    "cell {cell} of a sequence of {} cells",
                    live.tokens.len()
                );
            }
            referenced.extend(live.seq.pages().iter().copied());
        }
        // Pages only the index holds are in use too; count them by refcount.
        let in_use = (0..PAGES).filter(|p| self.pool.refcount(PageId(*p)) > 0).count();
        assert_eq!(self.pool.free_pages() as usize, PAGES as usize - in_use);
        assert!(referenced.len() <= in_use);
        assert!(in_use <= referenced.len() + self.index.len());
    }
}

#[test]
fn sequences_never_see_each_others_state() {
    for seed in [1u64, 0x9e3779b97f4a7c15, 0xdeadbeefcafe] {
        let mut rng = Rng(seed);
        let mut world = World {
            pool: PagePool::new(PAGES, PER_PAGE),
            index: PrefixIndex::new(),
            root: ChainHash::root(b"test model"),
            device: vec![u64::MAX; (PAGES * PER_PAGE) as usize],
            live: Vec::new(),
        };
        // A small alphabet makes prompts share prefixes by chance as well.
        let token = |rng: &mut Rng| rng.below(3) as u32;

        for _ in 0..4000 {
            let n_live = world.live.len();
            match rng.below(100) {
                // A new prompt: often the start of a live sequence plus a new
                // ending, resumed from whatever the index has cached.
                0..=14 if n_live < 12 => {
                    let mut tokens = Vec::new();
                    if n_live > 0 && rng.below(4) != 0 {
                        let from = &world.live[rng.below(n_live as u64) as usize].tokens;
                        let keep = rng.below(from.len() as u64 + 1) as usize;
                        tokens.extend_from_slice(&from[..keep]);
                    }
                    for _ in 0..rng.below(14) {
                        tokens.push(token(&mut rng));
                    }
                    let found = world.index.lookup(world.root, &tokens, PER_PAGE);
                    let cached = found.pages.len() * PER_PAGE as usize;
                    let seq = Sequence::from_shared(&mut world.pool, &found.pages, cached as u64);
                    world.live.push(Live { tokens: tokens[..cached].to_vec(), seq });
                    let i = world.live.len() - 1;
                    if !world.append(i, &tokens[cached..]) {
                        let live = world.live.pop().unwrap();
                        live.seq.release(&mut world.pool);
                    }
                }
                // Generation: a few more tokens.
                15..=49 if n_live > 0 => {
                    let i = rng.below(n_live as u64) as usize;
                    let tokens: Vec<u32> = (0..1 + rng.below(6)).map(|_| token(&mut rng)).collect();
                    world.append(i, &tokens);
                }
                // Rejected drafts: roll back a few cells.
                50..=64 if n_live > 0 => {
                    let i = rng.below(n_live as u64) as usize;
                    let live = &mut world.live[i];
                    let len = live.tokens.len() as u64;
                    let keep = len - rng.below(len.min(6) + 1);
                    live.seq.truncate(&mut world.pool, keep);
                    live.tokens.truncate(keep as usize);
                }
                // A fork, whole or of a prefix (branches of one prompt).
                65..=74 if n_live > 0 && n_live < 12 => {
                    let i = rng.below(n_live as u64) as usize;
                    let live = &world.live[i];
                    let keep = rng.below(live.tokens.len() as u64 + 1);
                    let seq = live.seq.fork_prefix(&mut world.pool, keep);
                    let tokens = live.tokens[..keep as usize].to_vec();
                    world.live.push(Live { tokens, seq });
                }
                // A request ends; its full pages stay cached.
                75..=84 if n_live > 0 => {
                    let i = rng.below(n_live as u64) as usize;
                    if rng.below(3) != 0 {
                        world.cache(i);
                    }
                    let live = world.live.swap_remove(i);
                    live.seq.release(&mut world.pool);
                }
                // Caching in the middle of a request.
                85..=92 if n_live > 0 => world.cache(rng.below(n_live as u64) as usize),
                // Memory pressure.
                _ => {
                    for _ in 0..rng.below(5) {
                        world.index.evict(&mut world.pool);
                    }
                }
            }
            world.check();
        }

        for live in world.live.drain(..) {
            live.seq.release(&mut world.pool);
        }
        world.index.clear(&mut world.pool);
        assert_eq!(world.pool.free_pages(), PAGES, "pages leaked (seed {seed:#x})");
    }
}
