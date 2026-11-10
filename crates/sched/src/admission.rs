//! Admission: which requests run, which wait, and who gives way when the
//! page pool is short.
//!
//! Nothing is reserved for a request's longest possible length. The rules,
//! each taken from the engine that has the better one:
//!
//! - **Pages are claimed per step** (vLLM). A sequence takes from the pool
//!   exactly the pages the rows of this step write.
//! - **A reservation covers the output that is expected**, not the output
//!   that is possible (SGLang). Every running request counts for the pages of
//!   its prompt that are not computed yet, in full, and for `min(tokens it may
//!   still generate, clip) * ratio` tokens of output. The ratio starts
//!   cautious and corrects itself: every finished request moves it toward the
//!   share of its allowance that it really produced, and a shortage raises it
//!   to what the running requests have produced already.
//! - **A new request is charged for what it adds** (TGI): the pages of its
//!   prompt after the prefix match, and its clipped allowance once, in full
//!   (SGLang charges the newcomer in full and discounts only requests that
//!   run; a request let in on a discount would be the first to be spilled).
//!   It is admitted when that fits in the free and evictable pages that the
//!   reservation leaves.
//! - **A request that does not fit waits** (TGI). Nothing is refused for
//!   load. Waiting requests are tried with the fewest uncached rows first, so
//!   cache hits go ahead, as [`plan`](crate::plan) orders prompts; a request
//!   that was passed over [`AdmissionConfig::starvation_passes`] times goes
//!   first and holds back everything behind it until it is in.
//! - **A request alone is always admitted.** With nothing running there is no
//!   one to protect, so the first request in line starts whatever it may grow
//!   to.
//! - **A step's claims are planned against a copy of the counters and
//!   committed whole** (LMDeploy, TokenSpeed). [`Claims`] collects the claims
//!   of a step, decides on its copy what is evicted, who is cut and who is
//!   spilled, returns the victims that turned out not to be needed, and hands
//!   back one [`Outcome`]. Everything the outcome grants fits.
//! - **When the pool is short, the sequence with the fewest generated tokens
//!   gives way** (SGLang's order; ties: the longest prompt). Here it is
//!   spilled, not recomputed: its pages go to RAM and it waits at the front of
//!   the line to come back.
//!
//! This object decides and reports. It moves no page, and it reads the pool
//! only through the numbers it is given: the free pages at the start of a
//! pass, and, when those do not cover the pass, a [`PoolView`].

use crate::plan::SeqId;

/// The numbers admission works with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmissionConfig {
    /// The capacity of the page pool.
    pub pages: u32,
    /// Cells per page.
    pub cells_per_page: u32,
    /// The most output tokens a request is charged or reserved for. A request
    /// that names no limit may generate until the context is full; nobody
    /// expects it to (SGLang clips at 4096 as well).
    pub output_clip: u32,
    /// The ratio before any request has finished.
    pub initial_ratio: f64,
    /// The ratio never falls below this.
    pub min_ratio: f64,
    /// How far one finished request moves the ratio toward its own share.
    pub ratio_step: f64,
    /// After a shortage every running request is assumed to produce at least
    /// this many tokens more (SGLang: 20).
    pub shortage_margin: u32,
    /// A waiting request that was passed over this many times goes first and
    /// blocks the requests behind it.
    pub starvation_passes: u32,
}

impl AdmissionConfig {
    /// The defaults for a pool of `pages` pages of `cells_per_page` cells.
    ///
    /// # Panics
    ///
    /// Panics when `cells_per_page` is 0.
    pub fn new(pages: u32, cells_per_page: u32) -> Self {
        assert!(cells_per_page > 0, "a page holds at least one cell");
        Self {
            pages,
            cells_per_page,
            output_clip: 4096,
            initial_ratio: 0.7,
            min_ratio: 0.05,
            ratio_step: 0.1,
            shortage_margin: 20,
            starvation_passes: 32,
        }
    }
}

/// A request as it arrives, in the numbers admission needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arrival {
    /// The request. It keeps this id while it waits and while it runs.
    pub seq: SeqId,
    /// The length of the prompt in cells.
    pub prompt_cells: u32,
    /// How many of them the prefix index has (full pages and a tail).
    pub cached_cells: u32,
    /// How many of the cached pages it would take no sequence holds now.
    /// Taking them makes them unavailable for eviction.
    pub idle_cached_pages: u32,
    /// The most tokens it may generate: its own limit, or the room the
    /// context leaves after the prompt.
    pub max_new_tokens: u32,
}

/// The prompt alone takes more pages than the pool has. This is the one
/// refusal, and it does not depend on load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExceedsPool {
    /// The pages the prompt and one generated token take.
    pub pages_needed: u32,
    /// The capacity of the pool.
    pub pages: u32,
}

impl std::fmt::Display for ExceedsPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the prompt takes {} pages and the pool has {}", self.pages_needed, self.pages)
    }
}

impl std::error::Error for ExceedsPool {}

/// What a pass may ask about the pool. Both answers cost a walk, so they are
/// asked for only when the free pages do not cover the pass.
pub trait PoolView {
    /// Cached pages that no sequence holds: what eviction can free.
    fn evictable(&mut self) -> u32;
    /// The pages that become free or evictable when `seq` lets go of all of
    /// its pages (the ones no other sequence holds).
    fn releasable(&mut self, seq: SeqId) -> u32;
}

/// What a pass decided. The caller carries it out in this order: the spills;
/// the resumed and admitted sequences take what the prefix index has for them
/// (so that the eviction cannot take it); the eviction; the step with the
/// claims that were not cut; the pages the resumed sequences bring back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// Cached pages to give up so that the step's claims fit.
    pub evict: u32,
    /// Sequences to suspend: their pages leave the pool and they get nothing
    /// in this step. They wait at the front of the line.
    pub spill: Vec<SeqId>,
    /// Sequences whose optional pages (draft rows) are not granted; their
    /// required pages are.
    pub cut: Vec<SeqId>,
    /// A sequence that cannot grow although nothing else is left to give way:
    /// the pool is as large as this one sequence. Its claim is not granted;
    /// the caller ends it or moves it out of the pool.
    pub exhausted: Option<SeqId>,
    /// Spilled sequences to bring back, in order.
    pub resume: Vec<SeqId>,
    /// Waiting requests to start, in order.
    pub admit: Vec<SeqId>,
}

#[derive(Clone, Copy, Debug)]
struct Running {
    seq: SeqId,
    prompt_cells: u32,
    prompt_rows_left: u32,
    generated: u32,
    max_new_tokens: u32,
}

#[derive(Clone, Copy, Debug)]
struct Waiting {
    arrival: Arrival,
    /// Arrival order.
    order: u64,
    /// Passes in which it could have been admitted and was not.
    passed: u32,
    /// Waiting for a page another sequence computes; not a candidate.
    held: bool,
    /// For a spilled sequence: what it was when it was suspended and the
    /// pages it needs back.
    spilled: Option<(Running, u32)>,
}

#[derive(Clone, Copy, Debug)]
struct Claim {
    seq: SeqId,
    required: u32,
    optional: u32,
    prompt_rows: u32,
}

/// The admission accounting. Plain data, owned by the step loop.
#[derive(Clone, Debug)]
pub struct Admission {
    config: AdmissionConfig,
    ratio: f64,
    running: Vec<Running>,
    waiting: Vec<Waiting>,
    arrivals: u64,
    claims: Vec<Claim>,
    order: Vec<usize>,
    shortages: u64,
}

impl Admission {
    /// An accounting with nothing running and nobody waiting.
    ///
    /// # Panics
    ///
    /// Panics when `config.cells_per_page` is 0.
    pub fn new(config: AdmissionConfig) -> Self {
        assert!(config.cells_per_page > 0, "a page holds at least one cell");
        Self {
            config,
            ratio: config.initial_ratio.clamp(config.min_ratio, 1.0),
            running: Vec::new(),
            waiting: Vec::new(),
            arrivals: 0,
            claims: Vec::new(),
            order: Vec::new(),
            shortages: 0,
        }
    }

    /// The share of its remaining (clipped) allowance that a running request
    /// is expected to produce.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The number of requests that wait, spilled ones included.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// The number of requests that run.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Whether `seq` was admitted and neither ended nor was spilled.
    pub fn is_running(&self, seq: SeqId) -> bool {
        self.running.iter().any(|r| r.seq == seq)
    }

    /// How many passes found the pool too small for the running requests.
    pub fn shortages(&self) -> u64 {
        self.shortages
    }

    /// The pages held back for the running requests: the rest of their
    /// prompts and the output they are expected to produce.
    pub fn reserved_pages(&self) -> u32 {
        self.running.iter().map(|r| self.reserve(r)).sum()
    }

    fn pages_for(&self, cells: u32) -> u32 {
        cells.div_ceil(self.config.cells_per_page)
    }

    /// Pages for the output a request with `tokens_left` of allowance is
    /// counted with: at least one while it may produce anything.
    fn output_pages(&self, tokens_left: u32, ratio: f64) -> u32 {
        if tokens_left == 0 {
            return 0;
        }
        let tokens = f64::from(tokens_left.min(self.config.output_clip)) * ratio;
        ((tokens / f64::from(self.config.cells_per_page)).ceil() as u32).max(1)
    }

    fn reserve(&self, r: &Running) -> u32 {
        self.pages_for(r.prompt_rows_left)
            + self.output_pages(r.max_new_tokens.saturating_sub(r.generated), self.ratio)
    }

    /// What admitting `w` adds to the demand on the pool.
    fn charge(&self, w: &Waiting) -> u32 {
        match &w.spilled {
            Some((was, restore)) => restore + self.reserve(was),
            None => {
                let a = &w.arrival;
                let cached = a.cached_cells.min(a.prompt_cells);
                // A cached tail shares a page that the request must copy
                // before it writes, so only whole cached pages are free.
                self.pages_for(a.prompt_cells) - cached / self.config.cells_per_page
                    + self.output_pages(a.max_new_tokens, 1.0)
            }
        }
    }

    /// Puts a new request in line. The only refusal is a prompt that is
    /// larger than the whole pool.
    pub fn enqueue(&mut self, arrival: Arrival) -> Result<(), ExceedsPool> {
        // The prompt, and a cell for the first token it generates.
        let pages_needed = self.pages_for(arrival.prompt_cells.saturating_add(1));
        if pages_needed > self.config.pages {
            return Err(ExceedsPool { pages_needed, pages: self.config.pages });
        }
        self.arrivals += 1;
        self.waiting.push(Waiting {
            arrival,
            order: self.arrivals,
            passed: 0,
            held: false,
            spilled: None,
        });
        Ok(())
    }

    /// Replaces what the prefix index has for a waiting request, after the
    /// index changed (a page it waited for was published, pages were
    /// evicted), and ends a hold. Unknown and running requests are ignored.
    pub fn refresh(&mut self, seq: SeqId, cached_cells: u32, idle_cached_pages: u32) {
        if let Some(w) = self.waiting.iter_mut().find(|w| w.arrival.seq == seq) {
            w.arrival.cached_cells = cached_cells;
            w.arrival.idle_cached_pages = idle_cached_pages;
            w.held = false;
        }
    }

    /// Holds a waiting request back: another sequence is computing the page
    /// its prompt misses first. It is skipped, and blocks nobody, until
    /// [`Admission::refresh`].
    pub fn hold(&mut self, seq: SeqId) {
        if let Some(w) = self.waiting.iter_mut().find(|w| w.arrival.seq == seq) {
            w.held = true;
        }
    }

    /// Says what a spilled sequence needs to come back, when that changed
    /// since it was spilled (pages it shared were evicted, or are cached
    /// again): `pages` it must take from the pool, and `idle_cached_pages`
    /// cached pages it takes again that no sequence holds now.
    pub fn set_restore(&mut self, seq: SeqId, pages: u32, idle_cached_pages: u32) {
        if let Some(w) = self.waiting.iter_mut().find(|w| w.arrival.seq == seq)
            && let Some((_, restore)) = &mut w.spilled
        {
            *restore = pages;
            w.arrival.idle_cached_pages = idle_cached_pages;
        }
    }

    /// Records tokens a running request generated (accepted, after a commit).
    pub fn generated(&mut self, seq: SeqId, tokens: u32) {
        if let Some(r) = self.running.iter_mut().find(|r| r.seq == seq) {
            r.generated = r.generated.saturating_add(tokens);
        }
    }

    /// A running request ended by itself (end token, stop string, limit).
    /// The ratio learns from what it produced.
    pub fn finished(&mut self, seq: SeqId) {
        let Some(i) = self.running.iter().position(|r| r.seq == seq) else { return };
        let r = self.running.swap_remove(i);
        let allowance = r.max_new_tokens.min(self.config.output_clip);
        if allowance > 0 {
            let share = f64::from(r.generated.min(allowance)) / f64::from(allowance);
            self.ratio += self.config.ratio_step * (share - self.ratio);
            self.ratio = self.ratio.clamp(self.config.min_ratio, 1.0);
        }
    }

    /// Removes a request wherever it is (cancelled, failed, ended by the
    /// caller after [`Outcome::exhausted`]). The ratio learns nothing from it.
    pub fn remove(&mut self, seq: SeqId) {
        self.running.retain(|r| r.seq != seq);
        self.waiting.retain(|w| w.arrival.seq != seq);
    }

    /// Starts the accounting of one step. `free_pages` is the pool's free
    /// count now.
    pub fn begin(&mut self, free_pages: u32) -> Claims<'_> {
        self.claims.clear();
        Claims { admission: self, free: free_pages }
    }

    /// After a shortage the ratio is at least the share that the running
    /// requests have produced already, with a margin (SGLang's estimate after
    /// a retraction).
    fn correct_after_shortage(&mut self) {
        self.shortages += 1;
        let clip = self.config.output_clip;
        let (mut produced, mut allowed) = (0u64, 0u64);
        for r in &self.running {
            produced += u64::from(r.generated.min(clip)) + u64::from(self.config.shortage_margin);
            allowed += u64::from(r.max_new_tokens.min(clip));
        }
        if allowed > 0 {
            let seen = (produced as f64 / allowed as f64).min(1.0);
            self.ratio = self.ratio.max(seen).clamp(self.config.min_ratio, 1.0);
        }
    }

    /// Tries to admit the waiting request at index `i`: it fits when it is
    /// alone, or when its charge fits in the free and evictable pages next
    /// to this step's claims and everything held back for the others.
    fn try_admit(
        &mut self,
        i: usize,
        tally: &mut Tally,
        view: &mut dyn PoolView,
        out: &mut Outcome,
    ) -> bool {
        let w = self.waiting[i];
        let charge = self.charge(&w);
        let mut fits = self.running.is_empty();
        if !fits {
            // Free pages first; the walk over the cache only if needed.
            let need = u64::from(tally.granted) + u64::from(tally.held_back) + u64::from(charge);
            fits = need <= u64::from(tally.free);
            if !fits {
                let n = *tally.evictable.get_or_insert_with(|| view.evictable());
                let usable =
                    n.saturating_sub(tally.taken).saturating_sub(w.arrival.idle_cached_pages);
                fits = need <= u64::from(tally.free) + u64::from(usable);
            }
        }
        if !fits {
            self.waiting[i].passed = w.passed.saturating_add(1);
            return false;
        }
        tally.taken = tally.taken.saturating_add(w.arrival.idle_cached_pages);
        match w.spilled {
            Some((was, restore)) => {
                self.running.push(was);
                out.resume.push(was.seq);
                // The pages it brings back are taken in this pass.
                tally.granted = tally.granted.saturating_add(restore);
                tally.held_back = tally.held_back.saturating_add(charge - restore);
            }
            None => {
                let a = &w.arrival;
                self.running.push(Running {
                    seq: a.seq,
                    prompt_cells: a.prompt_cells,
                    prompt_rows_left: a.prompt_cells.saturating_sub(a.cached_cells),
                    generated: 0,
                    max_new_tokens: a.max_new_tokens,
                });
                out.admit.push(a.seq);
                tally.held_back = tally.held_back.saturating_add(charge);
            }
        }
        // Gone from the line; removed when the pass ends, so that indices
        // stay valid until then.
        self.waiting[i].order = 0;
        true
    }

    /// The admissions of a pass. Spilled requests come back first (first
    /// out, first back), then starved ones (the longest wait first); the
    /// first of these that does not fit ends the pass. Then the others, the
    /// fewest uncached rows first, each on its own merits.
    fn admit(&mut self, tally: &mut Tally, view: &mut dyn PoolView, out: &mut Outcome) {
        let starvation = self.config.starvation_passes;
        let uncached = |w: &Waiting| w.arrival.prompt_cells.saturating_sub(w.arrival.cached_cells);
        loop {
            let first = self
                .waiting
                .iter()
                .enumerate()
                .filter(|(_, w)| w.order != 0 && !w.held)
                .filter(|(_, w)| w.spilled.is_some() || w.passed >= starvation)
                .min_by_key(|(_, w)| (w.spilled.is_none(), w.order))
                .map(|(i, _)| i);
            let Some(i) = first else { break };
            if !self.try_admit(i, tally, view, out) {
                self.waiting.retain(|w| w.order != 0);
                return;
            }
        }

        // With nothing running the first in line starts, whatever it needs.
        if self.running.is_empty() {
            let first = self
                .waiting
                .iter()
                .enumerate()
                .filter(|(_, w)| w.order != 0 && !w.held)
                .min_by_key(|(_, w)| (uncached(w), w.order))
                .map(|(i, _)| i);
            if let Some(i) = first {
                self.try_admit(i, tally, view, out);
            }
        }
        // What is held back only grows during a pass, so a request that does
        // not fit now does not fit later in it either: those are passed over
        // without being put in order.
        self.order.clear();
        for i in 0..self.waiting.len() {
            let w = self.waiting[i];
            if w.order == 0 || w.held {
                continue;
            }
            let need =
                u64::from(tally.granted) + u64::from(tally.held_back) + u64::from(self.charge(&w));
            let mut room = u64::from(tally.free);
            if need > room {
                let n = *tally.evictable.get_or_insert_with(|| view.evictable());
                room += u64::from(n.saturating_sub(tally.taken));
            }
            if need <= room {
                self.order.push(i);
            } else {
                self.waiting[i].passed = w.passed.saturating_add(1);
            }
        }
        let waiting = &self.waiting;
        self.order.sort_by_key(|i| (uncached(&waiting[*i]), waiting[*i].order));
        for k in 0..self.order.len() {
            let i = self.order[k];
            self.try_admit(i, tally, view, out);
        }
        self.waiting.retain(|w| w.order != 0);
    }
}

/// The counters of a pass while admissions are decided.
struct Tally {
    free: u32,
    /// Pages this pass takes from the pool: the step's claims and the pages
    /// resumed sequences bring back.
    granted: u32,
    /// Pages committed to running requests beyond this step: the
    /// reservation, and in full what this pass admits.
    held_back: u32,
    /// Idle cached pages that the requests admitted in this pass take.
    taken: u32,
    /// The evictable pages, once they were asked for.
    evictable: Option<u32>,
}

/// The accounting of one step: the claims of the running sequences, then the
/// decisions. Nothing changes until [`Claims::settle`].
#[derive(Debug)]
pub struct Claims<'a> {
    admission: &'a mut Admission,
    free: u32,
}

impl Claims<'_> {
    /// A running sequence's claim for this step: `required` pages for the
    /// rows it must get (its newest token, decision rows, prompt rows) and
    /// `optional` pages more for rows it can do without (drafts), of which
    /// `prompt_rows` rows are rows of its prompt. Claims of sequences that do
    /// not run are ignored.
    pub fn claim(&mut self, seq: SeqId, required: u32, optional: u32, prompt_rows: u32) {
        if self.admission.is_running(seq) {
            self.admission.claims.push(Claim { seq, required, optional, prompt_rows });
        }
    }

    /// Decides the step and the admissions after it.
    pub fn settle(self, view: &mut dyn PoolView) -> Outcome {
        let a = self.admission;
        let free = self.free;
        let mut out = Outcome::default();
        let mut evictable: Option<u32> = None;
        // Free pages, plus the evictable ones once they were asked for.
        let mut avail = free;
        let mut required: u32 = a.claims.iter().map(|c| c.required).sum();

        if required > avail {
            let n = view.evictable();
            evictable = Some(n);
            avail = free.saturating_add(n);
        }
        if required > avail {
            a.correct_after_shortage();
            // (victim, what its leaving frees, its own required claim)
            let mut victims: Vec<(SeqId, u32, u32)> = Vec::new();
            while required > avail {
                let victim = a
                    .running
                    .iter()
                    .filter(|r| !victims.iter().any(|v| v.0 == r.seq))
                    .min_by_key(|r| {
                        (r.generated, std::cmp::Reverse(r.prompt_cells), std::cmp::Reverse(r.seq))
                    })
                    .map(|r| r.seq);
                let Some(victim) = victim else { break };
                let claim: u32 =
                    a.claims.iter().filter(|c| c.seq == victim).map(|c| c.required).sum();
                if victims.len() + 1 == a.running.len() {
                    // Nobody else is left to give way for this one.
                    out.exhausted = Some(victim);
                    required -= claim;
                    break;
                }
                let freed = view.releasable(victim);
                avail = avail.saturating_add(freed);
                required -= claim;
                victims.push((victim, freed, claim));
            }
            // Return the victims that were not needed, the last chosen first:
            // a later victim may have freed enough for an earlier one to stay.
            let mut i = victims.len();
            while i > 0 {
                i -= 1;
                let (_, freed, claim) = victims[i];
                if avail >= freed && required + claim <= avail - freed {
                    avail -= freed;
                    required += claim;
                    victims.remove(i);
                }
            }
            for (victim, freed, _) in victims {
                let Some(i) = a.running.iter().position(|r| r.seq == victim) else { continue };
                let was = a.running.swap_remove(i);
                a.arrivals += 1;
                a.waiting.push(Waiting {
                    arrival: Arrival {
                        seq: victim,
                        prompt_cells: was.prompt_cells,
                        cached_cells: was.prompt_cells - was.prompt_rows_left,
                        idle_cached_pages: 0,
                        max_new_tokens: was.max_new_tokens,
                    },
                    order: a.arrivals,
                    passed: 0,
                    held: false,
                    spilled: Some((was, freed)),
                });
                out.spill.push(victim);
            }
        }

        // The claims that stand. Optional pages are granted while they fit
        // and are never a reason to spill.
        let mut granted = 0u32;
        for i in 0..a.claims.len() {
            let claim = a.claims[i];
            if out.exhausted == Some(claim.seq) {
                continue;
            }
            let Some(r) = a.running.iter_mut().find(|r| r.seq == claim.seq) else { continue };
            r.prompt_rows_left = r.prompt_rows_left.saturating_sub(claim.prompt_rows);
            granted += claim.required;
        }
        debug_assert_eq!(granted, required);
        for i in 0..a.claims.len() {
            let claim = a.claims[i];
            if claim.optional == 0 || out.exhausted == Some(claim.seq) || !a.is_running(claim.seq) {
                continue;
            }
            if granted + claim.optional > avail && evictable.is_none() {
                let n = view.evictable();
                evictable = Some(n);
                avail = free.saturating_add(n);
            }
            if granted + claim.optional <= avail {
                granted += claim.optional;
            } else {
                out.cut.push(claim.seq);
            }
        }
        out.evict = granted.saturating_sub(free);

        // Admissions, unless this step already had to make room.
        if out.spill.is_empty() && out.exhausted.is_none() && !a.waiting.is_empty() {
            let mut tally =
                Tally { free, granted, held_back: a.reserved_pages(), taken: 0, evictable };
            a.admit(&mut tally, view, &mut out);
            out.evict = tally.granted.saturating_sub(free);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pool that answers from fixed numbers.
    struct View {
        evictable: u32,
        releasable: Vec<(u32, u32)>,
        asked: u32,
    }

    impl View {
        fn new(evictable: u32) -> Self {
            Self { evictable, releasable: Vec::new(), asked: 0 }
        }
    }

    impl PoolView for View {
        fn evictable(&mut self) -> u32 {
            self.asked += 1;
            self.evictable
        }

        fn releasable(&mut self, seq: SeqId) -> u32 {
            self.releasable.iter().find(|(s, _)| *s == seq.0).map_or(0, |(_, pages)| *pages)
        }
    }

    const PER_PAGE: u32 = 128;

    fn config(pages: u32) -> AdmissionConfig {
        AdmissionConfig::new(pages, PER_PAGE)
    }

    fn arrival(seq: u32, prompt: u32, cached: u32, max_new: u32) -> Arrival {
        Arrival {
            seq: SeqId(seq),
            prompt_cells: prompt,
            cached_cells: cached,
            idle_cached_pages: 0,
            max_new_tokens: max_new,
        }
    }

    fn pass(a: &mut Admission, free: u32, evictable: u32) -> Outcome {
        a.begin(free).settle(&mut View::new(evictable))
    }

    #[test]
    fn a_request_alone_is_admitted_however_long_it_may_get() {
        // A pool of 10 pages. The request may generate until a context of
        // 262 144 tokens is full, which is 2 048 pages.
        let mut a = Admission::new(config(10));
        a.enqueue(arrival(1, 1000, 0, 261_144)).unwrap();
        let out = pass(&mut a, 10, 0);
        assert_eq!(out.admit, [SeqId(1)]);
        assert_eq!(a.running(), 1);

        // The same with the pool full of cached pages and nothing free.
        let mut a = Admission::new(config(10));
        a.enqueue(arrival(1, 1000, 0, u32::MAX)).unwrap();
        assert_eq!(pass(&mut a, 0, 10).admit, [SeqId(1)]);

        // A prompt that the whole pool cannot hold is the one refusal.
        assert_eq!(
            a.enqueue(arrival(2, 10 * PER_PAGE, 0, 16)),
            Err(ExceedsPool { pages_needed: 11, pages: 10 })
        );
        assert!(a.enqueue(arrival(2, 10 * PER_PAGE - 1, 0, 16)).is_ok());
    }

    #[test]
    fn a_second_request_waits_instead_of_being_refused_and_starts_when_there_is_room() {
        // 64 pages. The first request takes a prompt of 20 pages and may
        // generate without limit: 20 pages of prompt and, at the first
        // ratio of 0.7, 23 pages of expected output are held back.
        let mut a = Admission::new(config(64));
        a.enqueue(arrival(1, 20 * PER_PAGE, 0, u32::MAX)).unwrap();
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(1)]);
        assert_eq!(a.reserved_pages(), 20 + 23);

        // The second needs 10 pages of prompt and its clipped allowance in
        // full, 32 pages: 42 with 43 held back do not fit in 64. It waits.
        a.enqueue(arrival(2, 10 * PER_PAGE, 0, u32::MAX)).unwrap();
        let out = pass(&mut a, 64, 0);
        assert!(out.admit.is_empty());
        assert_eq!((a.running(), a.waiting()), (1, 1));

        // With a limit of 200 tokens it would take 10 + 2 pages and fit.
        a.enqueue(arrival(3, 10 * PER_PAGE, 0, 200)).unwrap();
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(3)]);

        // The first one finishes after 300 tokens. The line moves.
        a.generated(SeqId(1), 300);
        a.finished(SeqId(1));
        a.finished(SeqId(3));
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(2)]);
    }

    #[test]
    fn a_request_is_charged_only_for_the_rows_after_its_prefix_match() {
        let mut a = Admission::new(config(64));
        a.enqueue(arrival(1, 20 * PER_PAGE, 0, 256)).unwrap();
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(1)]);
        // 22 pages are held back. A prompt of 41 pages with a limit of 256
        // tokens needs 43 more: too much. The same prompt with 30 pages in
        // the cache needs 13.
        a.enqueue(arrival(2, 41 * PER_PAGE, 0, 256)).unwrap();
        a.enqueue(arrival(3, 41 * PER_PAGE, 30 * PER_PAGE, 256)).unwrap();
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(3)]);

        // A cached tail does not save a page: the request copies that page
        // before it writes into it. 43 pages of prompt with 2 pages and a
        // tail cached are 41 + 2 with 22 held back: 65 of 64.
        let mut b = Admission::new(config(64));
        b.enqueue(arrival(1, 20 * PER_PAGE, 0, 256)).unwrap();
        pass(&mut b, 64, 0);
        b.enqueue(arrival(2, 43 * PER_PAGE, 2 * PER_PAGE + 100, 256)).unwrap();
        assert!(pass(&mut b, 64, 0).admit.is_empty());
        // With the third page whole in the cache it is 64 of 64.
        b.refresh(SeqId(2), 3 * PER_PAGE, 0);
        assert_eq!(pass(&mut b, 64, 0).admit, [SeqId(2)]);
    }

    #[test]
    fn cached_pages_count_as_room_except_the_ones_the_request_takes_itself() {
        let mut a = Admission::new(config(64));
        a.enqueue(arrival(1, 20 * PER_PAGE, 0, 256)).unwrap();
        pass(&mut a, 64, 0);
        // 4 pages free, 40 cached and idle. A prompt of 30 pages with 20 of
        // them cached: 10 + 2 pages against 22 held back needs 34.
        let mut hit = arrival(2, 30 * PER_PAGE, 20 * PER_PAGE, 256);
        hit.idle_cached_pages = 20;
        a.enqueue(hit).unwrap();
        // Of the 40 evictable pages 20 are its own prefix: 4 + 20 < 34.
        let mut view = View::new(40);
        assert!(a.begin(4).settle(&mut view).admit.is_empty());
        assert_eq!(view.asked, 1, "the cache is walked once per pass");
        // With 14 pages free it fits: 14 + 20 = 34.
        assert_eq!(pass(&mut a, 14, 40).admit, [SeqId(2)]);
        // Enough free pages: the cache is not asked at all.
        a.enqueue(arrival(3, PER_PAGE, 0, 16)).unwrap();
        let mut view = View::new(40);
        assert_eq!(a.begin(60).settle(&mut view).admit, [SeqId(3)]);
        assert_eq!(view.asked, 0);
    }

    #[test]
    fn cache_hits_go_ahead_until_a_request_has_starved() {
        let mut cfg = config(64);
        cfg.starvation_passes = 3;
        let mut a = Admission::new(cfg);
        a.enqueue(arrival(1, 30 * PER_PAGE, 0, 256)).unwrap();
        pass(&mut a, 64, 0);
        // Held back: 30 + 2. A long miss arrives first, then short hits.
        a.enqueue(arrival(2, 40 * PER_PAGE, 0, 256)).unwrap();
        a.enqueue(arrival(3, 40 * PER_PAGE, 39 * PER_PAGE, 256)).unwrap();
        // The hit (1 + 2 pages) goes ahead of the miss (40 + 2).
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(3)]);
        a.finished(SeqId(3));
        for seq in 4..6 {
            a.enqueue(arrival(seq, 40 * PER_PAGE, 39 * PER_PAGE, 256)).unwrap();
            assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(seq)]);
            a.finished(SeqId(seq));
        }
        // The miss was passed over three times. It now blocks: a hit that
        // would fit is not admitted before it.
        a.enqueue(arrival(6, 40 * PER_PAGE, 39 * PER_PAGE, 256)).unwrap();
        assert!(pass(&mut a, 64, 0).admit.is_empty());
        // When the first request is done the starved one goes first, and the
        // hit after it in the same pass if it still fits.
        a.finished(SeqId(1));
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(2), SeqId(6)]);
    }

    #[test]
    fn a_held_request_is_skipped_and_blocks_nobody() {
        let mut cfg = config(64);
        cfg.starvation_passes = 0;
        let mut a = Admission::new(cfg);
        a.enqueue(arrival(1, PER_PAGE, 0, 16)).unwrap();
        a.enqueue(arrival(2, PER_PAGE, 0, 16)).unwrap();
        a.hold(SeqId(1));
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(2)]);
        // The page it waited for is published: it comes back as a hit.
        a.refresh(SeqId(1), PER_PAGE - 1, 0);
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(1)]);
    }

    #[test]
    fn a_steps_claims_evict_cached_pages_and_drafts_are_cut_before_anyone_is_spilled() {
        let mut a = Admission::new(config(64));
        for seq in 1..=2 {
            a.enqueue(arrival(seq, PER_PAGE, 0, 64)).unwrap();
        }
        assert_eq!(pass(&mut a, 64, 0).admit, [SeqId(1), SeqId(2)]);

        // Each needs a page for its newest token; the first would like one
        // more for drafts. One page is free and two are cached.
        let mut p = a.begin(1);
        p.claim(SeqId(1), 1, 1, 0);
        p.claim(SeqId(2), 1, 0, 0);
        let out = p.settle(&mut View::new(2));
        assert_eq!((out.evict, out.cut.len(), out.spill.len()), (2, 0, 0));

        // Only one cached page: the required pages fit, the drafts do not.
        let mut p = a.begin(1);
        p.claim(SeqId(1), 1, 1, 0);
        p.claim(SeqId(2), 1, 0, 0);
        let out = p.settle(&mut View::new(1));
        assert_eq!(out.evict, 1);
        assert_eq!(out.cut, [SeqId(1)]);
        assert!(out.spill.is_empty());
        assert_eq!(a.shortages(), 0);
    }

    #[test]
    fn the_sequence_with_the_fewest_generated_tokens_is_spilled() {
        let mut a = Admission::new(config(64));
        for (seq, prompt) in [(1, 10), (2, 12), (3, 12)] {
            a.enqueue(arrival(seq, prompt * PER_PAGE, 0, 64)).unwrap();
        }
        pass(&mut a, 64, 0);
        assert_eq!(a.running(), 3);
        a.generated(SeqId(1), 5);
        a.generated(SeqId(2), 9);
        a.generated(SeqId(3), 40);

        // The pool is empty and the third sequence needs a page.
        let mut view = View::new(0);
        view.releasable = vec![(1, 10), (2, 12), (3, 12)];
        let mut p = a.begin(0);
        p.claim(SeqId(3), 1, 0, 0);
        let out = p.settle(&mut view);
        assert_eq!(out.spill, [SeqId(1)], "5 tokens are the fewest");
        assert!(out.admit.is_empty() && out.resume.is_empty());
        assert_eq!((a.running(), a.waiting(), a.shortages()), (2, 1, 1));

        // Equal tokens: the longer prompt goes (SGLang's tie).
        let mut b = Admission::new(config(64));
        for (seq, prompt) in [(1, 10), (2, 12)] {
            b.enqueue(arrival(seq, prompt * PER_PAGE, 0, 64)).unwrap();
        }
        pass(&mut b, 64, 0);
        let mut p = b.begin(0);
        p.claim(SeqId(1), 1, 0, 0);
        assert_eq!(p.settle(&mut view).spill, [SeqId(2)]);
    }

    #[test]
    fn a_spilled_sequence_comes_back_first_and_takes_its_pages_again() {
        let mut a = Admission::new(config(40));
        a.enqueue(arrival(1, 10 * PER_PAGE, 0, 64)).unwrap();
        a.enqueue(arrival(2, 10 * PER_PAGE, 0, 64)).unwrap();
        pass(&mut a, 40, 0);
        a.generated(SeqId(2), 50);
        let mut view = View::new(0);
        view.releasable = vec![(1, 10), (2, 11)];
        // Both prompts are computed (20 pages); the pool has nothing left.
        let mut p = a.begin(20);
        p.claim(SeqId(1), 10, 0, 10 * PER_PAGE);
        p.claim(SeqId(2), 10, 0, 10 * PER_PAGE);
        assert_eq!(p.settle(&mut view), Outcome::default());
        let mut p = a.begin(0);
        p.claim(SeqId(2), 1, 0, 0);
        assert_eq!(p.settle(&mut view).spill, [SeqId(1)]);

        // A new request arrives while the spilled one waits. The spilled one
        // needs its 10 pages back and a page of output; 29 pages are free and
        // the running one holds one back: it returns, and takes 10 pages.
        a.enqueue(arrival(3, PER_PAGE, 0, 64)).unwrap();
        let out = pass(&mut a, 29, 0);
        assert_eq!(out.resume, [SeqId(1)]);
        assert_eq!(out.admit, [SeqId(3)], "and the new one fits behind it");
        assert_eq!(out.evict, 0);
        assert!(a.is_running(SeqId(1)));
    }

    #[test]
    fn a_victim_that_was_not_needed_is_returned() {
        let mut a = Admission::new(config(64));
        for seq in 1..=3 {
            a.enqueue(arrival(seq, 4 * PER_PAGE, 0, 64)).unwrap();
        }
        pass(&mut a, 64, 0);
        a.generated(SeqId(2), 3);
        a.generated(SeqId(3), 9);
        // The third needs 6 pages. The first (no tokens yet) would free only
        // 1 page (the rest of its pages are shared); the second frees 8.
        let mut view = View::new(0);
        view.releasable = vec![(1, 1), (2, 8), (3, 8)];
        let mut p = a.begin(0);
        p.claim(SeqId(3), 6, 0, 0);
        let out = p.settle(&mut view);
        assert_eq!(out.spill, [SeqId(2)], "the first one stays: its page was not needed");
        assert!(a.is_running(SeqId(1)));
    }

    #[test]
    fn the_last_sequence_is_not_spilled_for_itself() {
        let mut a = Admission::new(config(8));
        a.enqueue(arrival(1, PER_PAGE, 0, u32::MAX)).unwrap();
        pass(&mut a, 8, 0);
        let mut p = a.begin(0);
        p.claim(SeqId(1), 1, 0, 0);
        let out = p.settle(&mut View::new(0));
        assert_eq!(out.exhausted, Some(SeqId(1)));
        assert!(out.spill.is_empty());
        assert_eq!(out.evict, 0);
        assert!(a.is_running(SeqId(1)), "the caller decides what becomes of it");
        a.remove(SeqId(1));
        assert_eq!(a.running(), 0);
    }

    #[test]
    fn the_ratio_follows_what_requests_produce_and_a_shortage_raises_it() {
        let mut a = Admission::new(config(1000));
        assert_eq!(a.ratio(), 0.7);
        // Requests without a limit that stop after about 400 tokens: a tenth
        // of the clip.
        for seq in 0..60 {
            a.enqueue(arrival(seq, PER_PAGE, 0, u32::MAX)).unwrap();
            pass(&mut a, 1000, 0);
            a.generated(SeqId(seq), 410);
            a.finished(SeqId(seq));
        }
        assert!((a.ratio() - 0.1).abs() < 0.01, "{}", a.ratio());
        // A running request is now held 4 pages of output, not 23.
        a.enqueue(arrival(100, PER_PAGE, 0, u32::MAX)).unwrap();
        pass(&mut a, 1000, 0);
        assert_eq!(a.reserved_pages(), 1 + 4);

        // Requests that use their whole limit pull it up again.
        for seq in 200..230 {
            a.enqueue(arrival(seq, PER_PAGE, 0, 300)).unwrap();
            pass(&mut a, 1000, 0);
            a.generated(SeqId(seq), 300);
            a.finished(SeqId(seq));
        }
        assert!(a.ratio() > 0.9, "{}", a.ratio());

        // A shortage: two requests have produced 3000 of 4096 each and the
        // ratio says a tenth. It is raised to what they have produced.
        let mut c = Admission::new(AdmissionConfig { initial_ratio: 0.1, ..config(200) });
        for seq in 1..=2 {
            c.enqueue(arrival(seq, PER_PAGE, 0, u32::MAX)).unwrap();
        }
        assert_eq!(pass(&mut c, 200, 0).admit.len(), 2);
        c.generated(SeqId(1), 3000);
        c.generated(SeqId(2), 3000);
        let mut p = c.begin(0);
        p.claim(SeqId(1), 1, 0, 0);
        let mut view = View::new(0);
        view.releasable = vec![(1, 24), (2, 24)];
        let out = p.settle(&mut view);
        assert_eq!(out.spill.len(), 1);
        let expected = (3000.0 + 3000.0 + 40.0) / 8192.0;
        assert!((c.ratio() - expected).abs() < 1e-9, "{}", c.ratio());
    }
}
