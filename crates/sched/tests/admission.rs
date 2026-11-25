//! A randomised check of admission against a simulated pool.
//!
//! The simulation is the simplest pool with the properties the real one has:
//! a fixed number of pages; running requests that hold pages of their own and
//! pages of cached prefixes; cached prefixes that nobody holds, which can be
//! evicted. Requests arrive at random with random prompts, limits and true
//! output lengths, hit cached prefixes or not, run in steps (chunks of their
//! prompt, then a token and some drafts per step), finish, are cancelled.
//! Every decision of [`Admission`] is carried out literally, and after every
//! pass the test checks that
//!
//! - nothing was granted that the pool cannot give: the evictions asked for
//!   exist, and every granted claim finds a free page
//!   (so the pool never holds more pages than it has);
//! - a request that is alone is admitted at once, whatever it may grow to,
//!   and the only refusal ever is a prompt larger than the pool;
//! - a request that is admitted next to others fits by the stated rule;
//! - nobody is spilled unless the step's required claims exceed the free and
//!   evictable pages, the victims are the ones with the fewest generated
//!   tokens, and none of them could have stayed;
//! - a waiting request is overtaken by later arrivals of its own or a worse
//!   priority in at most `starvation_passes` passes, and when arrivals stop,
//!   every request is admitted and ends;
//! - with several priorities: a shortage spills the worst priority first, a
//!   running request gives way to a newcomer only for a better priority, and
//!   a spilled request that stays in line holds back its own and every worse
//!   priority.

use std::collections::BTreeMap;

use ares_sched::{Admission, AdmissionConfig, Arrival, Outcome, PoolView, SeqId};

const PER_PAGE: u32 = 16;
const STARVATION: u32 = 4;

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

fn pages_for(cells: u32) -> u32 {
    cells.div_ceil(PER_PAGE)
}

/// A cached prefix: `pages` pages, held by `holders` running requests.
#[derive(Clone, Copy, Debug)]
struct Doc {
    pages: u32,
    holders: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Waiting,
    Running,
    Spilled,
}

#[derive(Clone, Debug)]
struct Req {
    order: u64,
    phase: Phase,
    prompt: u32,
    max_new: u32,
    /// Tokens it generates before it stops by itself.
    true_len: u32,
    /// The cached prefix its prompt starts with, and how many of its pages.
    hit: Option<(usize, u32)>,
    /// Cells computed.
    cells: u32,
    /// The cached prefixes it holds and how many pages of each it uses.
    docs: Vec<(usize, u32)>,
    rows_left: u32,
    generated: u32,
    /// For a spilled request: the prefixes it needs again.
    wanted: Vec<(usize, u32)>,
    /// Passes in which a later arrival of its own or a worse priority was
    /// admitted while this one waited.
    overtaken: u32,
    /// A lower number first.
    priority: i32,
}

impl Req {
    fn shared(&self) -> u32 {
        self.docs.iter().map(|d| d.1).sum()
    }

    fn own(&self) -> u32 {
        pages_for(self.cells) - self.shared()
    }
}

struct World {
    capacity: u32,
    /// Cached prefixes by number. One that lost all its pages is gone.
    docs: BTreeMap<usize, Doc>,
    next_doc: usize,
    reqs: BTreeMap<u32, Req>,
}

impl World {
    fn doc(&self, doc: usize) -> Doc {
        self.docs.get(&doc).copied().unwrap_or(Doc { pages: 0, holders: 0 })
    }

    fn hold(&mut self, doc: usize) {
        self.docs.get_mut(&doc).expect("a held prefix is cached").holders += 1;
    }

    fn used(&self) -> u32 {
        let own: u32 = self.reqs.values().filter(|r| r.phase == Phase::Running).map(Req::own).sum();
        own + self.docs.values().map(|d| d.pages).sum::<u32>()
    }

    fn free(&self) -> u32 {
        self.capacity.checked_sub(self.used()).expect("the pool holds more pages than it has")
    }

    fn evictable(&self) -> u32 {
        self.docs.values().filter(|d| d.holders == 0).map(|d| d.pages).sum()
    }

    fn releasable(&self, seq: u32) -> u32 {
        let r = &self.reqs[&seq];
        r.own()
            + r.docs
                .iter()
                .map(|d| self.doc(d.0))
                .filter(|d| d.holders == 1)
                .map(|d| d.pages)
                .sum::<u32>()
    }

    /// What the cache has for a waiting request now: (pages, of them idle).
    fn cached(&self, r: &Req) -> (u32, u32) {
        match r.hit {
            Some((doc, want)) => {
                // Holding a prefix holds all of it here, so all of its pages
                // stop being evictable, not only the ones the request uses.
                let cached = self.doc(doc);
                let pages = want.min(cached.pages);
                (pages, if cached.holders == 0 && pages > 0 { cached.pages } else { 0 })
            }
            None => (0, 0),
        }
    }

    /// What a spilled request computes again to come back (its text that
    /// the cache does not hold now, which it gave up when it was spilled),
    /// and the idle cached pages it takes again.
    fn recompute(&self, r: &Req) -> (u32, u32) {
        let (mut kept, mut idle) = (0, 0);
        for (doc, used) in &r.wanted {
            let cached = self.doc(*doc);
            let keep = (*used).min(cached.pages);
            kept += keep;
            if cached.holders == 0 && keep > 0 {
                idle += cached.pages;
            }
        }
        (r.cells.saturating_sub(kept * PER_PAGE) + r.rows_left, idle)
    }

    fn evict(&mut self, mut pages: u32) {
        for doc in self.docs.values_mut().filter(|d| d.holders == 0) {
            let take = pages.min(doc.pages);
            doc.pages -= take;
            pages -= take;
        }
        assert_eq!(pages, 0, "more evictions were asked for than the cache can give");
        self.docs.retain(|_, d| d.pages > 0);
    }

    /// The request lets go of everything; what it computed stays cached.
    fn let_go(&mut self, seq: u32, keep_cached: bool) -> Vec<(usize, u32)> {
        let own = self.reqs[&seq].own();
        let r = self.reqs.get_mut(&seq).expect("known request");
        let mut wanted = std::mem::take(&mut r.docs);
        for (doc, _) in &wanted {
            self.docs.get_mut(doc).expect("a held prefix is cached").holders -= 1;
        }
        if keep_cached && own > 0 {
            self.next_doc += 1;
            self.docs.insert(self.next_doc, Doc { pages: own, holders: 0 });
            wanted.push((self.next_doc, own));
        }
        wanted
    }
}

struct View<'a>(&'a World);

impl PoolView for View<'_> {
    fn evictable(&mut self) -> u32 {
        self.0.evictable()
    }

    fn releasable(&mut self, seq: SeqId) -> u32 {
        self.0.releasable(seq.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct Work {
    seq: u32,
    required: u32,
    optional: u32,
    prompt_rows: u32,
    drafts: u32,
}

fn output_pages(tokens_left: u32, clip: u32, ratio: f64) -> u32 {
    if tokens_left == 0 {
        return 0;
    }
    ((f64::from(tokens_left.min(clip)) * ratio / f64::from(PER_PAGE)).ceil() as u32).max(1)
}

/// Who gives way, written the plain way: the worst priority first, among those
/// the fewest generated tokens first (ties: the longest prompt, then the
/// newest) until the required claims fit,
/// never the last sequence for itself; then everyone who can stay stays, the
/// last chosen first.
fn reference_victims(world: &World, work: &[Work], free: u32) -> (Vec<u32>, Option<u32>, u32) {
    let claim = |seq: u32| work.iter().filter(|w| w.seq == seq).map(|w| w.required).sum::<u32>();
    let mut required: u32 = work.iter().map(|w| w.required).sum();
    let mut avail = free + world.evictable();
    let mut order: Vec<(i32, u32, u32, u32)> = world
        .reqs
        .iter()
        .filter(|(_, r)| r.phase == Phase::Running)
        .map(|(seq, r)| (r.priority, r.generated, r.prompt, *seq))
        .collect();
    order.sort_by_key(|(priority, generated, prompt, seq)| {
        use std::cmp::Reverse;
        (Reverse(*priority), *generated, Reverse(*prompt), Reverse(*seq))
    });
    let (mut victims, mut exhausted) = (Vec::new(), None);
    for (k, (_, _, _, seq)) in order.iter().enumerate() {
        if required <= avail {
            break;
        }
        required -= claim(*seq);
        if k + 1 == order.len() {
            exhausted = Some(*seq);
            break;
        }
        avail += world.releasable(*seq);
        victims.push(*seq);
    }
    let mut returned = 0;
    for seq in victims.clone().into_iter().rev() {
        let freed = world.releasable(seq);
        if avail >= freed && required + claim(seq) <= avail - freed {
            avail -= freed;
            required += claim(seq);
            victims.retain(|v| *v != seq);
            returned += 1;
        }
    }
    (victims, exhausted, returned)
}

struct Totals {
    admitted: u64,
    spilled: u64,
    waited: u64,
    refused: u64,
    cut: u64,
    exhausted: u64,
    evicted: u64,
    /// Victims that could stay after all.
    returned: u64,
    /// Running requests that gave way to a newcomer of a better priority.
    outranked: u64,
    overtaken_max: u32,
}

/// The spills of an outcome: the ones a shortage made, then the ones that gave
/// way to a better priority.
fn split_spills(out: &Outcome) -> (Vec<u32>, Vec<u32>) {
    let shortage = out.spill.len() - out.outranked as usize;
    let mut short: Vec<u32> = out.spill[..shortage].iter().map(|s| s.0).collect();
    short.sort_unstable();
    (short, out.spill[shortage..].iter().map(|s| s.0).collect())
}

/// One run. `clip` is the output a request is charged for at most: with a
/// small one the reservation is thin, many requests run next to each other
/// and the pool runs short often, which is what exercises the spills.
/// Arrivals take one of `priorities` priorities at random (from -1 up).
fn run(seed: u64, capacity: u32, clip: u32, priorities: u64, passes: u32, totals: &mut Totals) {
    let mut rng = Rng(seed);
    let mut config = AdmissionConfig::new(capacity, PER_PAGE);
    config.output_clip = clip;
    config.starvation_passes = STARVATION;
    let mut a = Admission::new(config);
    let mut world = World { capacity, docs: BTreeMap::new(), next_doc: 0, reqs: BTreeMap::new() };
    let mut next_seq = 0u32;
    let mut arrivals = 0u64;

    let mut pass_no = 0u32;
    loop {
        pass_no += 1;
        let draining = pass_no > passes;
        if draining && world.reqs.is_empty() {
            break;
        }
        assert!(pass_no < passes + 200_000, "seed {seed}: the line never emptied");

        // Arrivals: a steady trickle, and now and then a burst that fills
        // the line.
        let arriving = match rng.below(400) {
            _ if draining => 0,
            0 => u64::from(capacity / 6).min(12),
            n if n < u64::from(capacity / 3).min(40) => 1,
            _ => 0,
        };
        for _ in 0..arriving {
            let prompt = match rng.below(40) {
                0 => 1 + rng.below(u64::from(capacity * PER_PAGE * 3 / 2)) as u32,
                1..=5 => 1 + rng.below(u64::from(capacity * PER_PAGE / 3)) as u32,
                _ => 1 + rng.below(8 * u64::from(PER_PAGE)) as u32,
            };
            let max_new = if rng.below(3) == 0 { u32::MAX } else { 1 + rng.below(600) as u32 };
            // Most answers are short; now and then one runs far past what
            // the others taught the ratio to expect.
            let true_len = match rng.below(12) {
                0 => 300 + rng.below(1500) as u32,
                _ => (rng.below(24) * rng.below(12)) as u32,
            };
            let hit = if !world.docs.is_empty() && rng.below(3) != 0 {
                // Half of the hits are on the largest cached prefix, so that
                // requests share pages (a system prompt, a document).
                let doc = if rng.below(2) == 0 {
                    *world.docs.iter().max_by_key(|(_, d)| d.pages).expect("not empty").0
                } else {
                    let nth = rng.below(world.docs.len() as u64) as usize;
                    *world.docs.keys().nth(nth).expect("in range")
                };
                // A whole-prompt hit keeps one row to compute.
                let want = world.doc(doc).pages.min((prompt - 1) / PER_PAGE);
                (want > 0).then_some((doc, want))
            } else {
                None
            };
            // Drawn only with several, so that a run of one priority plays
            // the same requests as before priorities were played.
            let priority = if priorities > 1 { rng.below(priorities) as i32 - 1 } else { 0 };
            next_seq += 1;
            arrivals += 1;
            let mut r = Req {
                order: arrivals,
                phase: Phase::Waiting,
                prompt,
                max_new,
                true_len,
                hit,
                cells: 0,
                docs: Vec::new(),
                rows_left: prompt,
                generated: 0,
                wanted: Vec::new(),
                overtaken: 0,
                priority,
            };
            let (cached, idle) = world.cached(&r);
            let arrival = Arrival {
                seq: SeqId(next_seq),
                prompt_cells: prompt,
                cached_cells: cached * PER_PAGE,
                idle_cached_pages: idle,
                max_new_tokens: max_new,
                priority,
            };
            let fits_at_all = pages_for(prompt + 1) <= capacity;
            match a.enqueue(arrival) {
                Ok(()) => {
                    assert!(fits_at_all, "seed {seed}: a prompt larger than the pool got in line");
                    r.rows_left = prompt - cached * PER_PAGE;
                    world.reqs.insert(next_seq, r);
                }
                Err(_) => {
                    assert!(!fits_at_all, "seed {seed}: refused for load");
                    totals.refused += 1;
                }
            }
        }

        // What the cache has for the waiting ones may have changed.
        for (seq, r) in &world.reqs {
            match r.phase {
                Phase::Waiting => {
                    let (cached, idle) = world.cached(r);
                    a.refresh(SeqId(*seq), cached * PER_PAGE, idle);
                }
                Phase::Spilled => {
                    let (rows, idle) = world.recompute(r);
                    a.set_recompute(SeqId(*seq), rows, idle);
                }
                Phase::Running => {}
            }
        }

        // The step: what every running request would compute.
        let mut work: Vec<Work> = Vec::new();
        for (seq, r) in &world.reqs {
            // Not every running request is in every step.
            if r.phase != Phase::Running || rng.below(4) == 0 {
                continue;
            }
            let (rows, drafts, prompt_rows) = if r.rows_left > 0 {
                let rows = r.rows_left.min(1 + rng.below(300) as u32);
                (rows, 0, rows)
            } else {
                (1, if rng.below(3) == 0 { rng.below(9) as u32 } else { 0 }, 0)
            };
            let required = pages_for(r.cells + rows) - pages_for(r.cells);
            let optional = pages_for(r.cells + rows + drafts) - pages_for(r.cells + rows);
            work.push(Work { seq: *seq, required, optional, prompt_rows, drafts });
        }

        let free = world.free();
        let evictable = world.evictable();
        let ratio = a.ratio();
        let running_before: Vec<u32> = world
            .reqs
            .iter()
            .filter(|(_, r)| r.phase == Phase::Running)
            .map(|(seq, _)| *seq)
            .collect();
        let someone_waits = world.reqs.values().any(|r| r.phase != Phase::Running);
        let (want_spilled, want_exhausted, _) = reference_victims(&world, &work, free);

        // A step that never runs: on a copy of the accounting, claims far
        // larger than real steps make. Real shortages are a page or two and
        // one victim settles them; these need several, and victims that turn
        // out not to be needed.
        if !running_before.is_empty() && rng.below(4) == 0 {
            let mut probe = a.clone();
            let mut fake: Vec<Work> = Vec::new();
            for seq in &running_before {
                if rng.below(2) == 0 {
                    let required = rng.below(u64::from(capacity / 2) + 1) as u32;
                    fake.push(Work { seq: *seq, required, optional: 0, prompt_rows: 0, drafts: 0 });
                }
            }
            let (mut want, exhausted, returned) = reference_victims(&world, &fake, free);
            totals.returned += u64::from(returned);
            let mut claims = probe.begin(free);
            for w in &fake {
                claims.claim(SeqId(w.seq), w.required, 0, 0);
            }
            let out = claims.settle(&mut View(&world));
            // A probe that fits admits, and a newcomer may outrank someone.
            let (spilled, _) = split_spills(&out);
            want.sort_unstable();
            assert_eq!(spilled, want, "seed {seed} pass {pass_no}: {fake:?}");
            assert_eq!(out.exhausted.map(|s| s.0), exhausted, "seed {seed} pass {pass_no}");
            // What is granted fits in what is free, evictable or let go.
            let kept: u32 = fake
                .iter()
                .filter(|w| !spilled.contains(&w.seq) && exhausted != Some(w.seq))
                .map(|w| w.required)
                .sum();
            let freed: u32 = spilled.iter().map(|seq| world.releasable(*seq)).sum();
            assert!(kept <= free + evictable + freed, "seed {seed} pass {pass_no}: {out:?}");
        }

        let mut claims = a.begin(free);
        for w in &work {
            claims.claim(SeqId(w.seq), w.required, w.optional, w.prompt_rows);
        }
        let out: Outcome = claims.settle(&mut View(&world));
        let context = format!("seed {seed} pass {pass_no}: {out:?}");

        // Who gives way.
        let required: u32 = work.iter().map(|w| w.required).sum();
        let (spilled, outranked) = split_spills(&out);
        let mut want = want_spilled.clone();
        want.sort_unstable();
        assert_eq!(spilled, want, "{context}");
        assert_eq!(out.exhausted.map(|s| s.0), want_exhausted, "{context}");
        if required <= free + evictable {
            assert!(spilled.is_empty() && out.exhausted.is_none(), "{context}");
        }
        if !spilled.is_empty() || out.exhausted.is_some() {
            assert!(out.admit.is_empty() && out.resume.is_empty(), "{context}");
        }
        // A running request gives way to a newcomer of a better priority only.
        let best_seated = out
            .resume
            .iter()
            .chain(&out.admit)
            .map(|seq| world.reqs[&seq.0].priority)
            .min()
            .unwrap_or(i32::MAX);
        for seq in &outranked {
            assert!(world.reqs[seq].priority > best_seated, "{context}: {seq} outranked");
        }
        // What the outranked let go of is room for the newcomers.
        let released: u32 = outranked.iter().map(|seq| world.releasable(*seq)).sum();
        // A request spilled in an earlier pass that stays in line holds back
        // the newcomers of its priority and everyone of a worse one.
        for (seq, r) in world.reqs.iter().filter(|(_, r)| r.phase == Phase::Spilled) {
            if out.resume.contains(&SeqId(*seq)) {
                continue;
            }
            for other in &out.admit {
                let p = world.reqs[&other.0].priority;
                assert!(p < r.priority, "{context}: {} went ahead of spilled {seq}", other.0);
            }
            for other in &out.resume {
                let p = world.reqs[&other.0].priority;
                assert!(p <= r.priority, "{context}: {} went ahead of spilled {seq}", other.0);
            }
        }
        totals.spilled += spilled.len() as u64;
        totals.outranked += outranked.len() as u64;
        totals.cut += out.cut.len() as u64;
        totals.evicted += u64::from(out.evict);

        // What the newcomers' prefixes were before the outranked let go: the
        // idle pages the accounting was told of, and the pages of a prefix an
        // outranked one held, which the newcomer takes again of what it let go
        // (the pages it uses, as the real pool holds them; this world holds
        // all of a prefix).
        let before: BTreeMap<u32, (u32, u32)> = out
            .admit
            .iter()
            .map(|seq| {
                let r = &world.reqs[&seq.0];
                let victims_held = match r.hit {
                    Some((doc, want))
                        if outranked
                            .iter()
                            .any(|v| world.reqs[v].docs.iter().any(|d| d.0 == doc)) =>
                    {
                        want.min(world.doc(doc).pages)
                    }
                    _ => 0,
                };
                (seq.0, (world.cached(r).1, victims_held))
            })
            .collect();
        // Carry it out: spills.
        for seq in &out.spill {
            let wanted = world.let_go(seq.0, true);
            let r = world.reqs.get_mut(&seq.0).expect("spilled requests exist");
            assert_eq!(r.phase, Phase::Running, "{context}");
            r.phase = Phase::Spilled;
            r.wanted = wanted;
        }

        // Resumed and admitted requests take what the cache has for them.
        let alone = running_before.is_empty();
        if alone && someone_waits {
            assert!(
                !out.admit.is_empty() || !out.resume.is_empty(),
                "{context}: alone and waiting"
            );
        }
        // The pages held back for the requests that ran before this pass,
        // after this step's claims: the rest of their prompts, and their
        // expected output.
        let mut held_back: u32 = running_before
            .iter()
            .filter(|seq| world.reqs[seq].phase == Phase::Running)
            .map(|seq| {
                let r = &world.reqs[seq];
                let stepped: u32 =
                    work.iter().filter(|w| w.seq == *seq).map(|w| w.prompt_rows).sum();
                pages_for(r.rows_left - stepped)
                    + output_pages(r.max_new.saturating_sub(r.generated), clip, ratio)
            })
            .sum();
        let step_pages: u32 = work
            .iter()
            .filter(|w| {
                world.reqs[&w.seq].phase == Phase::Running && out.exhausted != Some(SeqId(w.seq))
            })
            .map(|w| w.required + if out.cut.contains(&SeqId(w.seq)) { 0 } else { w.optional })
            .sum();
        let granted = step_pages;
        let mut taken_idle = 0u32;
        let mut taken_released = 0u32;
        let mut first = alone;
        // In the order they were seated: by priority, and in a priority the
        // resumed ones first.
        let mut seated: Vec<SeqId> = out.resume.iter().chain(&out.admit).copied().collect();
        seated.sort_by_key(|seq| world.reqs[&seq.0].priority);
        for seq in &seated {
            let r = world.reqs[&seq.0].clone();
            let resumed = r.phase == Phase::Spilled;
            assert_eq!(resumed, out.resume.contains(seq), "{context}");
            assert_ne!(r.phase, Phase::Running, "{context}");
            let (charge, rows, idle) = if resumed {
                let (rows, idle) = world.recompute(&r);
                let reserve = pages_for(rows)
                    + output_pages(r.max_new.saturating_sub(r.generated), clip, ratio);
                (reserve, rows, idle)
            } else {
                let cached = world.cached(&r).0;
                let (idle, victims_held) = before[&seq.0];
                taken_released += victims_held;
                (pages_for(r.prompt) - cached + output_pages(r.max_new, clip, 1.0), 0, idle)
            };
            if !first {
                let usable = evictable.saturating_sub(taken_idle + idle);
                let released = released.saturating_sub(taken_released);
                assert!(
                    granted + held_back + charge <= free + released + usable,
                    "{context}: {} admitted against the rule ({granted} + {held_back} + {charge} \
                     > {free} + {released} + {usable})",
                    seq.0
                );
            }
            first = false;
            taken_idle += idle;
            held_back += charge;

            // Take the cached pages.
            let mut docs = Vec::new();
            let mut cells = r.cells;
            if resumed {
                // What the cache kept comes back; the rest is computed again.
                let mut kept = 0;
                for (doc, used) in r.wanted.clone() {
                    let keep = used.min(world.doc(doc).pages);
                    if keep > 0 {
                        world.hold(doc);
                        docs.push((doc, keep));
                        kept += keep;
                    }
                }
                cells = cells.min(kept * PER_PAGE);
            } else {
                if let Some((doc, want)) = r.hit {
                    let keep = want.min(world.doc(doc).pages);
                    if keep > 0 {
                        world.hold(doc);
                        docs.push((doc, keep));
                    }
                    cells = keep * PER_PAGE;
                }
                totals.admitted += 1;
            }
            let r = world.reqs.get_mut(&seq.0).expect("admitted requests exist");
            r.docs = docs;
            r.wanted.clear();
            r.cells = cells;
            r.rows_left = if resumed { rows } else { r.prompt - cells };
            r.phase = Phase::Running;
        }
        assert_eq!(out.evict, granted.saturating_sub(free), "{context}");

        // The eviction, then the step, then the pages brought back. `free`
        // panics if the pool ever holds more than it has.
        assert!(out.evict <= world.evictable(), "{context}: evicts more than the cache has");
        world.evict(out.evict);
        assert!(granted <= world.free(), "{context}: what was granted does not fit");
        let mut finished: Vec<u32> = Vec::new();
        for w in &work {
            let r = world.reqs.get_mut(&w.seq).expect("requests with work exist");
            if r.phase != Phase::Running {
                assert!(out.spill.contains(&SeqId(w.seq)), "{context}");
                continue;
            }
            if out.exhausted == Some(SeqId(w.seq)) {
                // Ended by the caller: the pool is as large as this request.
                a.remove(SeqId(w.seq));
                finished.push(w.seq);
                totals.exhausted += 1;
                continue;
            }
            if w.prompt_rows > 0 {
                r.cells += w.prompt_rows;
                r.rows_left -= w.prompt_rows;
            } else {
                let drafts = if out.cut.contains(&SeqId(w.seq)) { 0 } else { w.drafts };
                let accepted = 1 + rng.below(u64::from(drafts) + 1) as u32;
                r.cells += accepted;
                r.generated += accepted;
                a.generated(SeqId(w.seq), accepted);
                if r.generated >= r.true_len.min(r.max_new) {
                    a.finished(SeqId(w.seq));
                    finished.push(w.seq);
                }
            }
        }
        let _ = world.free();
        for seq in finished {
            let keep = rng.below(3) != 0;
            world.let_go(seq, keep);
            world.reqs.remove(&seq);
        }

        // A cancellation now and then, of a request in any phase.
        if !draining && rng.below(40) == 0 && !world.reqs.is_empty() {
            let nth = rng.below(world.reqs.len() as u64) as usize;
            let seq = *world.reqs.keys().nth(nth).expect("in range");
            a.remove(SeqId(seq));
            if world.reqs[&seq].phase == Phase::Running {
                world.let_go(seq, false);
            }
            world.reqs.remove(&seq);
        }

        // Overtaking: a later arrival of the same or a worse priority was
        // admitted while an earlier one, never spilled, still waits.
        let admitted: Vec<(u64, i32)> = out
            .admit
            .iter()
            .filter_map(|s| world.reqs.get(&s.0).map(|r| (r.order, r.priority)))
            .collect();
        for r in world.reqs.values_mut().filter(|r| r.phase == Phase::Waiting) {
            if admitted.iter().any(|&(order, priority)| order > r.order && priority >= r.priority) {
                r.overtaken += 1;
                totals.overtaken_max = totals.overtaken_max.max(r.overtaken);
                assert!(
                    r.overtaken <= STARVATION,
                    "seed {seed} pass {pass_no}: overtaken in {} passes",
                    r.overtaken
                );
            }
        }
        totals.waited += world.reqs.values().filter(|r| r.phase != Phase::Running).count() as u64;
        // The accounting agrees with the world about who runs.
        let running = world.reqs.values().filter(|r| r.phase == Phase::Running).count();
        assert_eq!(a.running(), running, "{context}");
        assert_eq!(a.waiting(), world.reqs.len() - running, "{context}");
    }
    assert_eq!((a.running(), a.waiting()), (0, 0));
}

#[test]
fn admission_never_overcommits_and_never_starves() {
    let mut totals = Totals {
        admitted: 0,
        spilled: 0,
        waited: 0,
        refused: 0,
        cut: 0,
        exhausted: 0,
        evicted: 0,
        returned: 0,
        outranked: 0,
        overtaken_max: 0,
    };
    for (seed, capacity, clip, priorities) in [
        (1u64, 48u32, 256u32, 1u64),
        (0x9e3779b97f4a7c15, 96, 256, 1),
        (0xdeadbeef, 24, 256, 1),
        (77, 200, 256, 1),
        (4242, 12, 256, 1),
        (9, 64, 256, 1),
        (5, 64, 16, 1),
        (6, 160, 16, 1),
        (7, 32, 16, 1),
        (11, 48, 256, 3),
        (12, 96, 16, 3),
        (13, 24, 256, 2),
        (14, 160, 16, 3),
    ] {
        run(seed, capacity, clip, priorities, 8000, &mut totals);
    }
    // The run exercised every path it is meant to check.
    assert!(totals.admitted > 2000, "admitted {}", totals.admitted);
    assert!(totals.spilled >= 50, "spilled {}", totals.spilled);
    assert!(totals.returned > 100, "returned {}", totals.returned);
    assert!(totals.waited > 1000, "waited {}", totals.waited);
    assert!(totals.refused > 10, "refused {}", totals.refused);
    assert!(totals.cut > 10, "cut {}", totals.cut);
    assert!(totals.exhausted > 5, "exhausted {}", totals.exhausted);
    assert!(totals.evicted > 1000, "evicted {}", totals.evicted);
    assert!(totals.outranked > 50, "outranked {}", totals.outranked);
    assert_eq!(totals.overtaken_max, STARVATION, "the starvation bound was never reached");
}
