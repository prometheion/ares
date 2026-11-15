//! A model check of the exact pool against the pool of pages and the prefix
//! index, the way the conductor drives them: sequences start from the cache,
//! read prompts in chunks, decode, verify windows of drafts and roll some of
//! them back, end and leave their pages and a tail in the cache. After every
//! step the exact pool must hold what the policy and the bound promise: no
//! step runs short of slots at the size `exact_slots` gives, every partly used
//! page of a sequence has its slot, every slot belongs to one page, and the
//! pages the pool names as freed are not read by anyone.

use ares_state::{Chain, ChainHash, PagePool, PrefixIndex, Priority, Sequence};

use super::{Forms, NONE, Reach, TILE, exact_slots};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// A running sequence.
struct Live {
    cells: Sequence,
    tokens: Vec<u32>,
    chain: Chain,
    prompt: usize,
    generate: usize,
}

impl Forms {
    /// Every slot is free or belongs to one page.
    fn assert_consistent(&self) {
        let mut seen = vec![false; self.capacity as usize];
        for slot in &self.free {
            assert!(!std::mem::replace(&mut seen[*slot as usize], true), "slot {slot} twice");
        }
        for (page, slot) in self.exact_of.iter().enumerate() {
            if *slot != NONE {
                assert!(
                    !std::mem::replace(&mut seen[*slot as usize], true),
                    "slot {slot} of page {page} is also free or another page's"
                );
            }
        }
        assert!(seen.iter().all(|s| *s), "a slot is neither free nor a page's");
    }
}

fn text(rng: &mut Rng, documents: &[Vec<u32>]) -> Vec<u32> {
    let doc = &documents[rng.below(documents.len() as u64) as usize];
    let len = 20 + rng.below(1400) as usize;
    doc[..len.min(doc.len())].to_vec()
}

/// What a run did, so that the test can see it went through the cases.
#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    started: u32,
    from_cache: u32,
    windows: u32,
    rolled_back: u32,
    tails: u32,
    converted: u32,
    copies: u32,
    high: u32,
}

fn run(seed: u64, newest: u32) -> Stats {
    let mut stats = Stats::default();
    let (slots, max_rows) = (3u32, 300u32);
    let stride = 24u32;
    let pages = 220u32;
    let capacity = exact_slots(newest, slots, max_rows, slots);
    let mut rng = Rng(seed);
    let mut pool = PagePool::new(pages, TILE);
    let mut index = PrefixIndex::new().with_tail_limit(slots as usize);
    let root = ChainHash::root(b"model");
    let mut forms = Forms::new(capacity, pages, newest);
    let mut page_tab = vec![0u32; (slots * stride) as usize];
    // Four documents; two pairs share their first 300 tokens.
    let documents: Vec<Vec<u32>> = (0..4u32)
        .map(|d| {
            (0..1500u32)
                .map(|i| if i < 300 { 1000 * (d / 2) + i } else { 100_000 * d + i })
                .collect()
        })
        .collect();
    let mut running: Vec<Option<Live>> = (0..slots).map(|_| None).collect();
    let mut high = 0;
    for _ in 0..400 {
        // A request starts in a free slot.
        if let Some(slot) = running.iter().position(Option::is_none)
            && rng.below(3) == 0
        {
            let prompt = text(&mut rng, &documents);
            let found = index.lookup(root, &prompt[..prompt.len() - 1], TILE);
            let cells = found.cells(TILE);
            stats.started += 1;
            stats.from_cache += u32::from(cells > 0);
            let seq = Sequence::from_shared(&mut pool, &found.table(), cells);
            let mut chain = Chain::resumed(root, TILE, &found);
            chain.name(&prompt);
            running[slot] = Some(Live {
                cells: seq,
                chain,
                prompt: prompt.len(),
                tokens: prompt,
                generate: 1 + rng.below(300) as usize,
            });
        }
        // Each running sequence takes a step: pages are claimed (cached pages
        // go when the pool is short), then the record is written.
        let mut reaches = Vec::new();
        let mut windows = Vec::new();
        for (slot, entry) in running.iter_mut().enumerate() {
            let Some(live) = entry else { continue };
            let done = live.cells.len() as usize;
            // The prompt is read up to its last token, which a step of its own
            // computes with the first generated one.
            let (rows, direct) = if done + 1 < live.prompt {
                let left = live.prompt - 1 - done;
                (left.min(1 + rng.below(u64::from(max_rows)) as usize), true)
            } else if rng.below(2) == 0 {
                (1, true)
            } else {
                (2 + rng.below(4) as usize, false)
            };
            let need = live.cells.pages_to_append(&pool, rows as u64);
            while pool.free_pages() < need && index.evict(&mut pool).is_some() {}
            let grown = live.cells.append(&mut pool, rows as u64).expect("room in the pool");
            if let Some(copy) = grown.copy {
                stats.copies += 1;
                // The copy of a partly used page: its source is exact.
                assert!(
                    forms.slot_of(copy.from.0).is_some(),
                    "seed {seed}: a copied page is exact"
                );
            }
            let range = live.cells.take_dirty();
            for i in range {
                page_tab[slot * stride as usize + i] = live.cells.pages()[i].0;
            }
            if live.tokens.len() < done + rows {
                let tail = live.tokens.len();
                live.tokens.extend((tail..done + rows).map(|i| 7_000_000 + i as u32));
            }
            reaches.push(Reach {
                slot: slot as u32,
                pos0: done as u32,
                rows: rows as u32,
                writes: true,
                settles: direct,
            });
            if !direct {
                stats.windows += 1;
                windows.push((slot, done, rows));
            }
        }
        if reaches.is_empty() {
            continue;
        }
        // The record: what the pool freed, then the step.
        let mut freed = Vec::new();
        pool.take_freed(1024, &mut freed);
        for r in &reaches {
            let live = running[r.slot as usize].as_ref().unwrap();
            for page in live.cells.pages().iter().take(r.tiles().end as usize) {
                assert!(!freed.contains(&page.0), "seed {seed}: a freed page is read");
            }
        }
        for page in &freed {
            forms.release(*page);
        }
        forms
            .prepare(&reaches, |slot, tile| page_tab[(slot * stride + tile) as usize], false)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        stats.converted += forms.take_jobs().len() as u32;
        forms.assert_consistent();
        high = high.max(forms.in_use());
        assert!(forms.in_use() <= capacity);
        // The commit: windows keep some of their rows, pages that are full
        // enter the index, a sequence that is done leaves its pages and a tail.
        for (slot, done, rows) in windows {
            let live = running[slot].as_mut().unwrap();
            let kept = 1 + rng.below(rows as u64) as usize;
            stats.rolled_back += u32::from(kept < rows);
            live.cells.truncate(&mut pool, (done + kept) as u64);
            live.tokens.truncate(done + kept);
            live.chain.truncate((done + kept) / TILE as usize);
            forms.settle(
                |tile| page_tab[slot * stride as usize + tile as usize],
                (done + kept) as u32,
            );
        }
        for entry in running.iter_mut() {
            let Some(live) = entry.as_mut() else { continue };
            let settled = live.cells.len();
            index.publish(
                &mut pool,
                &mut live.cells,
                &mut live.chain,
                &live.tokens,
                settled,
                Priority::NORMAL,
            );
            let over = live.cells.len() as usize >= live.prompt + live.generate;
            if over {
                let live = entry.take().unwrap();
                let cells = live.cells.len();
                let full = (cells / u64::from(TILE)) as usize;
                let rest = &live.tokens[full * TILE as usize..cells as usize];
                if !rest.is_empty() {
                    stats.tails += 1;
                    let prior = live.chain.before(full);
                    let page = live.cells.pages()[full];
                    index.insert_tail(&mut pool, prior, rest, page, Priority::NORMAL);
                }
                live.cells.release(&mut pool);
            }
        }
        // Every partly used page of a running sequence has its slot.
        for (slot, entry) in running.iter().enumerate() {
            let Some(live) = entry else { continue };
            let len = live.cells.len();
            if len % u64::from(TILE) != 0 {
                let tile = (len / u64::from(TILE)) as usize;
                let page = live.cells.pages()[tile].0;
                assert!(
                    forms.slot_of(page).is_some(),
                    "seed {seed}: the partly used page {page} of slot {slot} has no slot"
                );
            }
        }
    }
    for entry in running.iter_mut() {
        if let Some(live) = entry.take() {
            live.cells.release(&mut pool);
        }
    }
    index.clear(&mut pool);
    let mut freed = Vec::new();
    pool.take_freed(4096, &mut freed);
    for page in freed {
        forms.release(page);
    }
    assert_eq!(forms.in_use(), 0, "seed {seed}: nothing is left");
    stats.high = high;
    stats
}

#[test]
fn sequences_that_start_from_the_cache_end_and_roll_back_never_run_the_pool_short() {
    let mut all = Stats::default();
    for seed in 1..=40u64 {
        // The windows of the formats from q7 to q4 and a wide one.
        for newest in [32, 64, 128, 256, 384] {
            let s = run(seed * 0x9e37_79b9 + u64::from(newest), newest);
            all.started += s.started;
            all.from_cache += s.from_cache;
            all.windows += s.windows;
            all.rolled_back += s.rolled_back;
            all.tails += s.tails;
            all.converted += s.converted;
            all.copies += s.copies;
            all.high = all.high.max(s.high);
        }
    }
    assert!(all.started > 500 && all.from_cache > 100, "{all:?}");
    assert!(all.windows > 500 && all.rolled_back > 100, "{all:?}");
    assert!(all.tails > 100 && all.copies > 20 && all.converted > 1000, "{all:?}");
}
