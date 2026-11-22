//! A randomised check of the prefix cache as a running engine uses it: pages
//! published when they fill, tails, prompts that wait for a page another
//! sequence computes, classes, and checkpoints of recurrent state placed and
//! dropped by the policy.
//!
//! The simulated model has both kinds of state. The state of a cell is a
//! function of all tokens up to it (what K and V are), written to a simulated
//! device array through the page tables. The recurrent state is one value
//! that every token rewrites; a checkpoint is a copy of it, kept by the test
//! under the id the index holds. Sequences start from whatever the cache has,
//! restore the checkpoint the policy names, replay the rows it names without
//! writing cells, and compute the rest. After every operation the test checks
//! that
//!
//! - every sequence reads, through its page table, exactly the cells its own
//!   tokens produce, also after it gave up its own copy of a cached page or
//!   continued in a copy of a tail;
//! - the recurrent state after restore and replay is the state of the cached
//!   text: the policy named the right checkpoint and the right rows;
//! - nothing is written into a page someone else holds, no page leaks, and no
//!   checkpoint leaks: the test stores exactly the checkpoints the index has;
//! - checkpoints never exceed the budget, and each one's rank is taken from
//!   its real position and a real checkpoint above it;
//! - `evictable` is exactly what `evict` then gives up;
//! - a prompt waits only for a page that a running sequence announced, and
//!   every waiting prompt starts in the end.

use std::collections::{HashMap, HashSet};

use ares_state::{
    Chain, ChainHash, CheckpointId, CheckpointPolicy, Owner, PageId, PagePool, PageState,
    PrefixIndex, Priority, Sequence,
};

const PER_PAGE: u32 = 4;
const PAGES: u32 = 160;

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
fn cell_state(previous: u64, token: u64) -> u64 {
    (previous ^ u64::from(token)).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23) ^ 0x51
}

/// The recurrent state after `token`, from the state before it.
fn recurrent(previous: u64, token: u64) -> u64 {
    (previous.rotate_left(11) ^ u64::from(token)).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0x1f
}

fn recurrent_of(tokens: &[u64]) -> u64 {
    tokens.iter().fold(0, |state, token| recurrent(state, *token))
}

struct Live {
    owner: Owner,
    /// The prompt and what was generated. The first `seq.len()` are computed.
    tokens: Vec<u64>,
    prompt_len: usize,
    seq: Sequence,
    chain: Chain,
    class: Priority,
    /// The recurrent state, and the number of cells it is the state after.
    state: u64,
    state_at: u64,
    /// The position of the last checkpoint on this sequence's path.
    last_checkpoint: u64,
}

/// A prompt that waits for a page another sequence computes.
struct Held {
    tokens: Vec<u64>,
    chain: Chain,
    awaited: ChainHash,
}

/// What the test stores for a checkpoint: the text it is the state of, the
/// state, and who published it.
struct Stored {
    text: Vec<u64>,
    state: u64,
    by: Owner,
}

struct World {
    pool: PagePool,
    index: PrefixIndex,
    policy: CheckpointPolicy,
    root: ChainHash,
    /// The device array: one value per cell of every page.
    device: Vec<u64>,
    live: Vec<Live>,
    held: Vec<Held>,
    /// Texts of sequences that ended, to continue as conversations.
    ended: Vec<Vec<u64>>,
    checkpoints: HashMap<CheckpointId, Stored>,
    next_owner: u64,
    next_checkpoint: u64,
    stats: Stats,
}

#[derive(Default, Debug)]
struct Stats {
    started: u64,
    held: u64,
    tail_hits: u64,
    page_hits: u64,
    replayed_rows: u64,
    restored: u64,
    adopted: u64,
    captured: u64,
    dropped: u64,
    evicted: u64,
}

impl World {
    fn write(&mut self, page: PageId, offset: u32, value: u64) {
        assert_eq!(self.pool.refcount(page), 1, "write into a shared page");
        self.device[(page.0 * PER_PAGE + offset) as usize] = value;
    }

    fn cell(&self, seq: &Sequence, cell: u64) -> u64 {
        let (page, offset) = seq.locate(&self.pool, cell).expect("a computed cell");
        self.device[(page.0 * PER_PAGE + offset) as usize]
    }

    fn forget(&mut self, id: Option<CheckpointId>) {
        if let Some(id) = id {
            assert!(self.checkpoints.remove(&id).is_some(), "a checkpoint was handed back twice");
            self.stats.dropped += 1;
        }
    }

    /// Keeps `state`, the recurrent state after `text`, at the entry or tail
    /// `hash`, unless there is a checkpoint there already.
    fn capture(&mut self, hash: ChainHash, text: &[u64], state: u64, by: Owner) {
        if self.index.checkpoint(hash).is_some() || self.policy.capacity() == 0 {
            return;
        }
        for id in self.index.make_room_for_checkpoints(&self.policy, 1) {
            self.forget(Some(id));
        }
        self.next_checkpoint += 1;
        let id = CheckpointId(self.next_checkpoint);
        let before = self.index.attach_checkpoint(hash, id, by).expect("captured at a known entry");
        assert_eq!(before, None);
        self.checkpoints.insert(id, Stored { text: text.to_vec(), state, by });
        self.stats.captured += 1;
        assert!(self.index.checkpoints() <= self.policy.capacity(), "over the budget");
    }

    /// Advances the recurrent state of sequence `i` to `to` cells, taking the
    /// checkpoints the policy asks for on the way. The cells up to `to` exist.
    fn advance(&mut self, i: usize, to: u64) {
        let prompt_len = self.live[i].prompt_len as u64;
        loop {
            let live = &mut self.live[i];
            let next =
                self.policy.next_capture(prompt_len, &[], live.last_checkpoint, live.state_at, to);
            let stop = next.unwrap_or(to);
            for cell in live.state_at..stop {
                live.state = recurrent(live.state, live.tokens[cell as usize]);
            }
            live.state_at = stop;
            let Some(position) = next else { break };
            assert!(position % u64::from(PER_PAGE) == 0 && position <= to);
            live.last_checkpoint = position;
            let hash = live.chain.hashes()[(position / u64::from(PER_PAGE)) as usize - 1];
            let (text, state, owner) =
                (live.tokens[..position as usize].to_vec(), live.state, live.owner);
            self.capture(hash, &text, state, owner);
        }
    }

    /// Starts `tokens` as a prompt, or holds it back when the page it misses
    /// first is being computed. `named` is the prompt's chain when it was
    /// named before (a prompt that waited).
    fn submit(&mut self, tokens: Vec<u64>, named: Option<Chain>, class: Priority) {
        let named = named.unwrap_or_else(|| {
            let mut chain = Chain::new(self.root, PER_PAGE);
            chain.name(&tokens);
            chain
        });
        // The last token is always computed.
        let found = self.index.lookup_chain(&named, &tokens[..tokens.len() - 1]);
        if let Some((awaited, owner)) = found.computing {
            assert!(
                self.live.iter().any(|l| l.owner == owner),
                "a prompt waits for a sequence that is gone"
            );
            self.held.push(Held { tokens, chain: named, awaited });
            self.stats.held += 1;
            return;
        }
        let resume = self.policy.resume(&found, tokens.len() as u64);
        assert_eq!(resume.cached_cells, found.cells(PER_PAGE));
        assert!(resume.cached_cells < tokens.len() as u64, "nothing is left to compute");
        assert_eq!(resume.replay.end, resume.cached_cells);
        assert_eq!(resume.compute, resume.cached_cells..tokens.len() as u64);
        self.stats.started += 1;
        self.stats.page_hits += found.pages.len() as u64;
        self.stats.tail_hits += u64::from(found.tail.is_some());
        self.stats.replayed_rows += resume.replay.end - resume.replay.start;

        // The recurrent state: the checkpoint, then the replay.
        let state = match resume.restore {
            Some(id) => {
                let stored = &self.checkpoints[&id];
                assert_eq!(
                    stored.text,
                    tokens[..resume.replay.start as usize],
                    "the checkpoint is not the state where the replay starts"
                );
                self.stats.restored += 1;
                stored.state
            }
            None => {
                assert_eq!(resume.replay.start, 0);
                0
            }
        };
        let seq = Sequence::from_shared(&mut self.pool, &found.table(), resume.cached_cells);
        // The chain knows the names of the whole prompt and which of them
        // are published; the rest is what this sequence will compute.
        let mut chain = Chain::resumed(self.root, PER_PAGE, &found);
        chain.name(&tokens);
        assert_eq!(chain.hashes(), named.hashes());
        self.next_owner += 1;
        let owner = Owner(self.next_owner);
        self.index.announce(owner, chain.unpublished());
        self.live.push(Live {
            owner,
            prompt_len: tokens.len(),
            tokens,
            seq,
            chain,
            class,
            state,
            state_at: resume.replay.start,
            last_checkpoint: resume.replay.start,
        });
        // The replay reads the cached cells and writes none; it passes the
        // places where checkpoints are due.
        let i = self.live.len() - 1;
        self.advance(i, resume.cached_cells);
        let live = &self.live[i];
        assert_eq!(live.state, recurrent_of(&live.tokens[..resume.cached_cells as usize]));
    }

    /// Sequence `i` computes `rows` more cells: rows of its prompt, or tokens
    /// it generates, of which the last `drafts` are rejected again. Returns
    /// false, with nothing changed, when the pool is too small.
    fn step(&mut self, i: usize, rows: u64, drafts: u64, rng: &mut Rng) -> bool {
        let before = self.live[i].seq.len();
        let prompt_len = self.live[i].prompt_len;
        while (self.live[i].tokens.len() as u64) < before + rows {
            self.live[i].tokens.push(rng.below(3));
        }
        let Ok(grown) = self.live[i].seq.append(&mut self.pool, rows) else {
            self.live[i].tokens.truncate(prompt_len.max(before as usize));
            return false;
        };
        assert_eq!(grown.first_cell, before);
        if let Some(copy) = grown.copy {
            for cell in 0..copy.cells {
                let value = self.device[(copy.from.0 * PER_PAGE + cell) as usize];
                self.write(copy.to, cell, value);
            }
        }
        let mut state = if before == 0 { 0 } else { self.cell(&self.live[i].seq, before - 1) };
        for cell in before..before + rows {
            state = cell_state(state, self.live[i].tokens[cell as usize]);
            let (page, offset) = self.live[i].seq.locate(&self.pool, cell).expect("appended");
            self.write(page, offset, state);
        }
        // Rejected drafts are cut before anything is published.
        let keep = before + rows - drafts;
        let live = &mut self.live[i];
        live.seq.truncate(&mut self.pool, keep);
        live.tokens.truncate((keep as usize).max(prompt_len));

        // The commit: publish what filled, take the checkpoints that are due.
        let pages_before: Vec<PageId> = live.seq.pages().to_vec();
        let (class, settled) = (live.class, live.seq.len());
        self.index.publish(
            &mut self.pool,
            &mut live.seq,
            &mut live.chain,
            &live.tokens,
            settled,
            class,
        );
        self.stats.adopted +=
            pages_before.iter().zip(live.seq.pages()).filter(|(a, b)| a != b).count() as u64;
        self.advance(i, settled);
        true
    }

    /// Sequence `i` ends. Its partly used last page becomes a tail with the
    /// end state as its checkpoint.
    fn end(&mut self, i: usize, publish_tail: bool) {
        let live = self.live.swap_remove(i);
        self.index.withdraw(live.owner);
        let len = live.seq.len() as usize;
        let full = len / PER_PAGE as usize;
        if publish_tail && !len.is_multiple_of(PER_PAGE as usize) {
            let rest = &live.tokens[full * PER_PAGE as usize..len];
            let prior = live.chain.before(full);
            let dropped = self.index.insert_tail(
                &mut self.pool,
                prior,
                rest,
                live.seq.pages()[full],
                live.class,
            );
            self.forget(dropped.and_then(|d| d.checkpoint));
            let tail = PrefixIndex::tail_hash(prior, rest);
            assert_eq!(live.state_at, len as u64);
            self.capture(tail, &live.tokens[..len], live.state, live.owner);
            self.ended.push(live.tokens[..len].to_vec());
        }
        live.seq.release(&mut self.pool);
    }

    /// Prompts that waited look again.
    fn wake(&mut self) {
        for held in std::mem::take(&mut self.held) {
            match self.index.state(held.awaited) {
                PageState::Computing(owner) => {
                    assert!(self.live.iter().any(|l| l.owner == owner), "a stale announcement");
                    self.held.push(held);
                }
                PageState::Cached | PageState::Unknown => {
                    self.submit(held.tokens, Some(held.chain), Priority::NORMAL);
                }
            }
        }
    }

    fn check(&self) {
        let mut referenced = HashSet::new();
        for live in &self.live {
            let len = live.seq.len() as usize;
            assert!(len <= live.tokens.len());
            let mut state = 0;
            for (cell, token) in live.tokens[..len].iter().enumerate() {
                state = cell_state(state, *token);
                assert_eq!(self.cell(&live.seq, cell as u64), state, "cell {cell} of {len}");
            }
            assert_eq!(live.state_at, len as u64);
            assert_eq!(live.state, recurrent_of(&live.tokens[..len]), "the recurrent state");
            referenced.extend(live.seq.pages().iter().copied());
            // Every full page of a running sequence is the cached one.
            for (page, hash) in live.seq.pages().iter().zip(live.chain.hashes()).take(len / 4) {
                assert_eq!(self.index.state(*hash), PageState::Cached);
                assert!(self.pool.is_cached(*page), "a full page that is not the cached one");
            }
        }
        // Every announcement is a page that a running sequence still has to
        // compute.
        let to_compute: usize = self.live.iter().map(|l| l.chain.unpublished().len()).sum();
        assert!(self.index.announced() <= to_compute, "an announcement outlived its page");
        let in_use = (0..PAGES).filter(|p| self.pool.refcount(PageId(*p)) > 0).count();
        assert_eq!(self.pool.free_pages() as usize, PAGES as usize - in_use);
        assert!(referenced.len() <= in_use);
        assert!(in_use <= referenced.len() + self.index.len());

        // The checkpoints the test stores are the ones the index has.
        assert_eq!(self.index.checkpoints(), self.checkpoints.len());
        assert!(self.index.checkpoints() <= self.policy.capacity());
        let ranks = self.index.checkpoint_ranks(&self.policy);
        assert_eq!(ranks.len(), self.checkpoints.len());
        for (k, rank) in ranks.iter().enumerate() {
            let stored = &self.checkpoints[&rank.checkpoint];
            assert_eq!(self.index.checkpoint(rank.hash), Some(rank.checkpoint));
            assert_eq!(rank.position, stored.text.len() as u64, "the position of a checkpoint");
            assert!(rank.rows_saved > 0 && rank.rows_saved <= rank.position);
            // The checkpoint it is measured against is a real one above it.
            let above = (rank.position - rank.rows_saved) as usize;
            if above > 0 {
                assert!(
                    self.checkpoints.values().any(|c| c.text == stored.text[..above]),
                    "no checkpoint {} rows above the one at {}",
                    rank.rows_saved,
                    rank.position
                );
            }
            // No checkpoint at a page boundary above it is nearer than the
            // one it is measured against. (A tail's checkpoint can be: the
            // index knows that a page continues a tail only when the page
            // came after the tail.)
            let nearest_page = self
                .checkpoints
                .values()
                .filter(|c| c.text.len() % PER_PAGE as usize == 0)
                .filter(|c| c.text.len() < stored.text.len() && stored.text.starts_with(&c.text))
                .map(|c| stored.text.len() - c.text.len())
                .min();
            if let Some(nearest) = nearest_page {
                assert!(rank.rows_saved <= nearest as u64, "a nearer checkpoint was overlooked");
            }
            // Below an aged one there is a later request's checkpoint, or as
            // many checkpoints as stay fresh; one with nothing below it is not
            // aged.
            let below =
                |c: &&Stored| c.text.len() > stored.text.len() && c.text.starts_with(&stored.text);
            if rank.aged {
                let mut under = self.checkpoints.values().filter(below);
                let count = self.checkpoints.values().filter(below).count();
                assert!(
                    under.any(|c| c.by != stored.by)
                        || count >= self.policy.fresh_checkpoints as usize
                );
            }
            if !self.checkpoints.values().any(|c| below(&c)) {
                assert!(!rank.aged, "a checkpoint with nothing below it is aged");
            }
            if k > 0 {
                let previous = &ranks[k - 1];
                assert!(
                    (!previous.aged, previous.rows_saved, previous.last_used)
                        <= (!rank.aged, rank.rows_saved, rank.last_used),
                    "the ranks are out of order"
                );
            }
        }
    }
}

fn run(seed: u64, budget: u64, stats: &mut Stats) {
    let mut rng = Rng(seed);
    let mut world = World {
        pool: PagePool::new(PAGES, PER_PAGE),
        index: PrefixIndex::new(),
        policy: CheckpointPolicy {
            cells_per_page: PER_PAGE,
            prompt_interval: 16,
            generation_interval: 8,
            boundary_interval: 8,
            boundary_reach: 0,
            volatile_rows: 1,
            bytes_per_checkpoint: 10,
            budget_bytes: budget * 10,
            fresh_requests: 0,
            fresh_checkpoints: 3,
        },
        root: ChainHash::root(b"test model"),
        device: vec![u64::MAX; (PAGES * PER_PAGE) as usize],
        live: Vec::new(),
        held: Vec::new(),
        ended: Vec::new(),
        checkpoints: HashMap::new(),
        next_owner: 0,
        next_checkpoint: 0,
        stats: Stats::default(),
    };
    // A small alphabet makes texts share prefixes by chance as well.
    let token = |rng: &mut Rng| rng.below(3);

    for _ in 0..5000 {
        let n_live = world.live.len();
        match rng.below(100) {
            // A new prompt: the next turn of a conversation that ended, the
            // start of a running one with another ending, or new text.
            0..=17 if n_live + world.held.len() < 10 => {
                let mut tokens = Vec::new();
                match rng.below(4) {
                    0 | 1 if !world.ended.is_empty() => {
                        let at = rng.below(world.ended.len() as u64) as usize;
                        tokens.extend_from_slice(&world.ended[at]);
                        if world.ended.len() > 12 {
                            world.ended.swap_remove(at);
                        }
                    }
                    2 if n_live > 0 => {
                        let from = &world.live[rng.below(n_live as u64) as usize].tokens;
                        let keep = rng.below(from.len() as u64 + 1) as usize;
                        tokens.extend_from_slice(&from[..keep]);
                    }
                    _ => {}
                }
                for _ in 0..1 + rng.below(22) {
                    tokens.push(token(&mut rng));
                }
                let class = if rng.below(6) == 0 { Priority(200) } else { Priority::NORMAL };
                world.submit(tokens, None, class);
            }
            // A step: rows of the prompt, or generated tokens with drafts.
            18..=59 if n_live > 0 => {
                let i = rng.below(n_live as u64) as usize;
                let live = &world.live[i];
                let left = live.prompt_len as u64 - live.seq.len().min(live.prompt_len as u64);
                let (rows, drafts) = if left > 0 {
                    (1 + rng.below(left.min(9)), 0)
                } else {
                    let rows = 1 + rng.below(6);
                    (rows, rng.below(rows))
                };
                world.step(i, rows, drafts, &mut rng);
            }
            // A request ends and leaves its tail, or is cancelled.
            60..=71 if n_live > 0 => {
                let i = rng.below(n_live as u64) as usize;
                world.end(i, rng.below(5) != 0);
            }
            // Memory pressure: what `evictable` says is what `evict` gives.
            72..=79 => {
                let evictable = world.index.evictable(&world.pool);
                let all = rng.below(4) == 0;
                let n = if all { evictable } else { rng.below(6).min(u64::from(evictable)) as u32 };
                for _ in 0..n {
                    let evicted = world.index.evict(&mut world.pool).expect("counted as evictable");
                    world.forget(evicted.checkpoint);
                    world.stats.evicted += 1;
                }
                if all {
                    assert_eq!(world.index.evict(&mut world.pool), None, "more than was counted");
                }
            }
            // A cached page is asked to be kept, or not to be.
            80..=83 if n_live > 0 => {
                let live = &world.live[rng.below(n_live as u64) as usize];
                if let Some(hash) = live.chain.hashes().first() {
                    let class = Priority(rng.below(255) as u8);
                    world.index.set_priority(*hash, class);
                }
            }
            _ => {}
        }
        world.wake();
        world.check();
    }

    // Everything ends; the prompts that waited start and end too.
    let mut rounds = 0;
    while !world.live.is_empty() || !world.held.is_empty() {
        rounds += 1;
        assert!(rounds < 10_000, "seed {seed:#x}: a prompt waits for ever");
        if !world.live.is_empty() {
            let i = rng.below(world.live.len() as u64) as usize;
            let live = &world.live[i];
            let left = (live.prompt_len as u64).saturating_sub(live.seq.len());
            if left == 0 {
                world.end(i, true);
            } else if !world.step(i, left, 0, &mut rng) {
                // The pool is full of cached pages: make room, or give up.
                match world.index.evict(&mut world.pool) {
                    Some(evicted) => world.forget(evicted.checkpoint),
                    None => world.end(i, false),
                }
            }
        }
        world.wake();
        world.check();
    }
    let returned = world.index.clear(&mut world.pool);
    assert_eq!(returned.len(), world.checkpoints.len(), "checkpoints leaked (seed {seed:#x})");
    assert_eq!(world.pool.free_pages(), PAGES, "pages leaked (seed {seed:#x})");

    stats.started += world.stats.started;
    stats.held += world.stats.held;
    stats.tail_hits += world.stats.tail_hits;
    stats.page_hits += world.stats.page_hits;
    stats.replayed_rows += world.stats.replayed_rows;
    stats.restored += world.stats.restored;
    stats.adopted += world.stats.adopted;
    stats.captured += world.stats.captured;
    stats.dropped += world.stats.dropped;
    stats.evicted += world.stats.evicted;
}

#[test]
fn the_cache_serves_running_sequences_and_keeps_its_books() {
    let mut stats = Stats::default();
    for (seed, budget) in
        [(1u64, 6u64), (0x9e3779b97f4a7c15, 12), (0xdeadbeefcafe, 3), (42, 40), (7, 0)]
    {
        run(seed, budget, &mut stats);
    }
    // The runs went through what they are meant to check.
    assert!(stats.started > 1500, "{stats:?}");
    assert!(stats.held > 20, "{stats:?}");
    assert!(stats.tail_hits > 100, "{stats:?}");
    assert!(stats.page_hits > 1000, "{stats:?}");
    assert!(stats.restored > 200, "{stats:?}");
    assert!(stats.replayed_rows > 500, "{stats:?}");
    assert!(stats.adopted > 5, "{stats:?}");
    assert!(stats.captured > 500, "{stats:?}");
    assert!(stats.dropped > 200, "{stats:?}");
    assert!(stats.evicted > 500, "{stats:?}");
}
