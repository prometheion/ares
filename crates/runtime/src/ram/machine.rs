//! The machine's memory and how the engine follows it.
//!
//! Page-locked memory is the one kind the kernel cannot take back: it is not
//! swapped and not reclaimed. An engine that locks what is free must therefore
//! watch the machine itself and unlock in time when something else grows. A
//! [`Machine`] does that for every engine of a process: it reads what the
//! machine has available ([`MemorySource`]), decides by a [`Rule`] how much
//! the elastic users of page-locked memory may hold together, and divides that
//! among them by rank: the tiers of experts first, the prompt caches with what
//! is left.
//!
//! The rule is described with its numbers ([`Rule`]).
//!
//! The policy ([`Follow`]) is a pure function of the times and numbers it is
//! shown, so that tests drive it with a clock and a memory of their own.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use ares_core::units::{GIB, MIB};

use super::Shared;

/// What the machine has available for programs to take without swapping.
pub trait MemorySource: Send + Sync {
    /// Bytes available now, or `None` when the machine does not say.
    fn available(&self) -> Option<u64>;

    /// Bytes of the machine's memory this process holds now, locked or not,
    /// when the source knows: what the process itself takes while it loads
    /// a model is then not taken for the growth of other programs. Without
    /// it only the page-locked memory in the engines' books counts as the
    /// process's own.
    fn own(&self) -> Option<u64> {
        None
    }
}

impl<F: Fn() -> Option<u64> + Send + Sync> MemorySource for F {
    fn available(&self) -> Option<u64> {
        self()
    }
}

/// `MemAvailable` of `/proc/meminfo`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meminfo;

impl Meminfo {
    fn field(name: &str) -> Option<u64> {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find_map(|line| line.strip_prefix(name))?;
        let kib: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
        kib.checked_mul(1024)
    }

    /// All the RAM the machine has (`MemTotal`).
    pub fn total() -> Option<u64> {
        Self::field("MemTotal:")
    }
}

impl MemorySource for Meminfo {
    fn available(&self) -> Option<u64> {
        Self::field("MemAvailable:")
    }

    /// The resident memory of this process (`/proc/self/statm`).
    fn own(&self) -> Option<u64> {
        let text = std::fs::read_to_string("/proc/self/statm").ok()?;
        text.split_whitespace().nth(1)?.parse::<u64>().ok()?.checked_mul(4096)
    }
}

/// The free memory the kernel keeps for itself before it stops reclaiming:
/// the high watermarks of all zones in the text of `/proc/zoneinfo`, in
/// bytes of pages of `page` bytes.
pub(crate) fn high_watermarks(zoneinfo: &str, page: u64) -> u64 {
    zoneinfo
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("high "))
        .filter_map(|pages| pages.trim().parse::<u64>().ok())
        .sum::<u64>()
        * page
}

/// The numbers of the policy by which the engine follows the machine's
/// memory.
///
/// - **The headroom** is what the engine leaves available. It is the sum of
///   two things: what the kernel itself keeps free before it stops reclaiming
///   (the high watermarks of its zones), and what other programs can take
///   while the engine is still releasing: the fastest growth of the others
///   that was seen, times the time the engine needs to notice and to release
///   (the period of its looks plus the slowest release that was seen). Both
///   measurements start from an assumption and are forgotten slowly, so the
///   headroom follows the machine it runs on. It has a most, so that one
///   jump of the machine's numbers does not cost the engine its memory.
/// - **Giving back.** When less than the headroom is available, the elastic
///   users together may hold that much less than they hold, at once.
/// - **Taking.** When more than the headroom and a band is available, and has
///   been for a while, they may hold the surplus down to the middle of the
///   band. The wait starts short and doubles every time memory had to be
///   given back soon after it was taken, so a program that allocates and
///   frees is followed down quickly and up slowly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rule {
    /// How often the machine is looked at.
    pub period: Duration,
    /// What the kernel keeps free for itself (the high watermarks of its
    /// zones).
    pub reserve: u64,
    /// The growth of other programs, in bytes per second, that is assumed
    /// until a faster one was seen.
    pub speed: u64,
    /// The time a release is assumed to take until a slower one was seen.
    pub latency: Duration,
    /// The size memory is locked and released in.
    pub piece: u64,
    /// How long memory must have been free before it is taken, at first and
    /// at most.
    pub calm: Duration,
    pub calm_max: Duration,
    /// Memory that is given back within this time of a take was taken too
    /// early: the wait doubles.
    pub settle: Duration,
    /// The time after which half of what was seen above the assumptions is
    /// forgotten.
    pub forget: Duration,
    /// The most the headroom may be, whatever was seen: one jump of the
    /// machine's numbers must not cost the engine its memory for an hour.
    pub most: u64,
}

impl Rule {
    /// The rule with the kernel's own reserve of this machine, read from
    /// `/proc/zoneinfo` (the default when it cannot be read).
    pub fn of_this_machine() -> Self {
        let reserve = std::fs::read_to_string("/proc/zoneinfo")
            .ok()
            .map(|text| high_watermarks(&text, 4096))
            .filter(|bytes| *bytes > 0);
        Self {
            reserve: reserve.unwrap_or(Self::default().reserve),
            // A sixteenth of the machine's RAM.
            most: Meminfo::total().map_or(Self::default().most, |total| total / 16),
            ..Self::default()
        }
    }
}

impl Default for Rule {
    fn default() -> Self {
        Self {
            period: Duration::from_millis(50),
            reserve: 128 * MIB,
            // One thread that writes to new pages; see "Host memory" in
            // docs/dev/experiments/host-memory.md for the measurement.
            speed: 2 * GIB,
            latency: Duration::from_millis(100),
            piece: crate::expert_store::PIECE as u64,
            calm: Duration::from_secs(3),
            calm_max: Duration::from_secs(300),
            settle: Duration::from_secs(60),
            forget: Duration::from_secs(600),
            most: 2 * GIB,
        }
    }
}

/// What a look at the machine came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Less than the headroom is available: this much is missing.
    Give(u64),
    /// Memory has been free long enough: this much may be taken.
    Take(u64),
    Stay,
}

/// The policy: when to give memory back and when to take it. It keeps no
/// clock and reads no memory; [`Rule`] says what it decides by.
#[derive(Clone, Debug)]
pub(crate) struct Follow {
    rule: Rule,
    /// The fastest growth of the others seen, bytes per second.
    speed: f64,
    /// The slowest release seen, seconds.
    latency: f64,
    /// The last looks: when, and what would be available if the engine held
    /// nothing.
    seen: VecDeque<(Duration, u64)>,
    last: Option<Duration>,
    /// Since when more than the headroom and the band has been available.
    calm_since: Option<Duration>,
    wait: Duration,
    last_take: Option<Duration>,
    giving: bool,
}

/// Looks the growth of the others is measured over: one piece that is booked
/// a look before or after it is locked must not pass for a fast program.
const WINDOW: usize = 4;

impl Follow {
    pub fn new(rule: Rule) -> Self {
        Self {
            rule,
            speed: rule.speed as f64,
            latency: rule.latency.as_secs_f64(),
            seen: VecDeque::with_capacity(WINDOW + 1),
            last: None,
            calm_since: None,
            wait: rule.calm,
            last_take: None,
            giving: false,
        }
    }

    pub fn rule(&self) -> &Rule {
        &self.rule
    }

    /// Sets what is kept free whatever was seen, and the most the headroom
    /// may be: the watch over a device's memory follows the reserve its state
    /// asks for at every look (`crate::vram`).
    pub fn set_reserve(&mut self, reserve: u64, most: u64) {
        self.rule.reserve = reserve;
        self.rule.most = most;
    }

    /// What the engine leaves available now.
    pub fn headroom(&self) -> u64 {
        let reaction = self.rule.period.as_secs_f64() + self.latency;
        let headroom = self.rule.reserve + (self.speed * reaction) as u64;
        headroom.min(self.rule.most.max(self.rule.reserve))
    }

    /// The width of the range of available memory in which nothing is given
    /// and nothing is taken.
    pub fn band(&self) -> u64 {
        (2 * self.rule.piece).max(self.headroom() / 4)
    }

    /// The wait before free memory is taken, as it stands.
    pub fn wait(&self) -> Duration {
        self.wait
    }

    /// The fastest growth of other programs seen, in bytes per second, and
    /// the slowest release.
    pub fn seen(&self) -> (u64, Duration) {
        (self.speed as u64, Duration::from_secs_f64(self.latency))
    }

    /// A release that was asked for took `took` until the memory was back.
    pub fn released(&mut self, took: Duration) {
        self.latency = self.latency.max(took.as_secs_f64());
    }

    /// One look: at `now` the machine has `available` bytes available and the
    /// process holds `locked` bytes of its own (its page-locked memory, or
    /// all its memory when the source says).
    pub fn look(&mut self, now: Duration, available: u64, locked: u64) -> Verdict {
        // What was seen above the assumptions fades.
        if let Some(last) = self.last {
            let half_lives =
                now.saturating_sub(last).as_secs_f64() / self.rule.forget.as_secs_f64().max(1e-9);
            let keep = 0.5f64.powf(half_lives);
            let (speed, latency) = (self.rule.speed as f64, self.rule.latency.as_secs_f64());
            self.speed = speed + (self.speed - speed).max(0.0) * keep;
            self.latency = latency + (self.latency - latency).max(0.0) * keep;
        }
        self.last = Some(now);
        // How fast the others grew: what the engine holds is not theirs.
        let outside = available.saturating_add(locked);
        if let Some((then, was)) = self.seen.front().copied() {
            let seconds = now.saturating_sub(then).as_secs_f64();
            if seconds > 1e-3 && was > outside {
                self.speed = self.speed.max((was - outside) as f64 / seconds);
            }
        }
        self.seen.push_back((now, outside));
        if self.seen.len() > WINDOW {
            self.seen.pop_front();
        }

        let headroom = self.headroom();
        let band = self.band();
        if available < headroom {
            if !std::mem::replace(&mut self.giving, true)
                && let Some(taken) = self.last_take.take()
            {
                // Taken too early doubles the wait, once for every take; a
                // give long after the last take starts it afresh.
                self.wait = if now.saturating_sub(taken) < self.rule.settle {
                    (self.wait * 2).min(self.rule.calm_max)
                } else {
                    self.rule.calm
                };
            }
            self.calm_since = None;
            return Verdict::Give(headroom - available);
        }
        self.giving = false;
        if available < headroom + band {
            self.calm_since = None;
            return Verdict::Stay;
        }
        let since = *self.calm_since.get_or_insert(now);
        if now.saturating_sub(since) < self.wait {
            return Verdict::Stay;
        }
        self.last_take = Some(now);
        Verdict::Take(available - headroom - band / 2)
    }
}

/// Divides `total` among users that can use `wants[i]` each: everyone gets
/// the same, as far as they can use it, and what one cannot use goes to the
/// others.
pub fn share(total: u64, wants: &[u64]) -> Vec<u64> {
    let mut given = vec![0u64; wants.len()];
    let mut left = total;
    loop {
        let open: Vec<usize> = (0..wants.len()).filter(|i| given[*i] < wants[*i]).collect();
        if open.is_empty() || left == 0 {
            return given;
        }
        let each = (left / open.len() as u64).max(1);
        for i in open {
            let more = each.min(wants[i] - given[i]).min(left);
            given[i] += more;
            left -= more;
        }
    }
}

struct State {
    follow: Follow,
    /// What the elastic users may hold together, above their floors.
    target: u64,
    members: Vec<Weak<Shared>>,
    available: Option<u64>,
}

/// What one look found and decided, for logs and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Look {
    pub available: u64,
    pub headroom: u64,
    pub verdict: Verdict,
    /// What the elastic users hold above their floors, and what they may.
    pub held: u64,
    pub target: u64,
    /// What they must still release.
    pub owed: u64,
}

/// The watch over a machine's memory, for all engines of a process.
pub struct Machine {
    source: Box<dyn MemorySource>,
    clock: Box<dyn Fn() -> Duration + Send + Sync>,
    state: Mutex<State>,
    watching: AtomicBool,
}

impl std::fmt::Debug for Machine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state();
        f.debug_struct("Machine")
            .field("headroom", &state.follow.headroom())
            .field("target", &state.target)
            .field("members", &state.members.len())
            .finish()
    }
}

impl Machine {
    /// A watch that reads `source` and the time from `clock` (the time since
    /// any fixed moment). Nothing looks until [`Machine::look`] is called or
    /// [`Machine::watch`] starts a thread that does.
    pub fn new(
        rule: Rule,
        source: impl MemorySource + 'static,
        clock: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            source: Box::new(source),
            clock: Box::new(clock),
            state: Mutex::new(State {
                follow: Follow::new(rule),
                target: 0,
                members: Vec::new(),
                available: None,
            }),
            watching: AtomicBool::new(false),
        })
    }

    /// The watch over this machine, one for the process: `/proc/meminfo`,
    /// the kernel's reserve of this machine, and a thread that looks.
    pub fn system() -> Arc<Self> {
        static SYSTEM: OnceLock<Arc<Machine>> = OnceLock::new();
        SYSTEM
            .get_or_init(|| {
                let started = Instant::now();
                let machine =
                    Machine::new(Rule::of_this_machine(), Meminfo, move || started.elapsed());
                machine.watch();
                machine
            })
            .clone()
    }

    /// Starts the thread that looks, once. It ends with the last reference
    /// to the machine.
    pub fn watch(self: &Arc<Self>) {
        if self.watching.swap(true, Ordering::AcqRel) {
            return;
        }
        let weak = Arc::downgrade(self);
        let spawned = std::thread::Builder::new().name("ares-ram".into()).spawn(move || {
            loop {
                let Some(machine) = weak.upgrade() else { return };
                let owed = machine.look().is_some_and(|look| look.owed > 0);
                let period = machine.state().follow.rule().period;
                drop(machine);
                // A release that is under way is followed closely.
                std::thread::sleep(if owed { period / 10 } else { period });
            }
        });
        if spawned.is_err() {
            self.watching.store(false, Ordering::Release);
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the engine leaves available now.
    pub fn headroom(&self) -> u64 {
        self.state().follow.headroom()
    }

    /// What the machine had available at the last look.
    pub fn available(&self) -> Option<u64> {
        self.state().available.or_else(|| self.source.available())
    }

    /// The fastest growth of other programs seen (bytes per second), the
    /// slowest release, and the wait before free memory is taken.
    pub fn seen(&self) -> (u64, Duration, Duration) {
        let state = self.state();
        let (speed, latency) = state.follow.seen();
        (speed, latency, state.follow.wait())
    }

    pub(super) fn admit(&self, member: &Arc<Shared>) {
        self.state().members.push(Arc::downgrade(member));
    }

    /// One look at the machine: decides, divides, and tells the users of
    /// every engine what they may hold. `None` when the machine does not say
    /// what it has.
    pub fn look(&self) -> Option<Look> {
        self.look_as(false)
    }

    /// A look for a user that has just said what it can use: what is free
    /// now is its to take without the wait, which is for memory that others
    /// have just given up.
    pub(super) fn claim(&self) -> Option<Look> {
        self.look_as(true)
    }

    fn look_as(&self, claim: bool) -> Option<Look> {
        let available = self.source.available()?;
        let now = (self.clock)();
        let (look, members) = {
            let mut state = self.state();
            state.available = Some(available);
            let members: Vec<Arc<Shared>> =
                state.members.iter().filter_map(Weak::upgrade).collect();
            state.members.retain(|member| member.strong_count() > 0);
            let wants: Vec<super::Wants> = members.iter().map(|m| m.wants()).collect();
            let locked: u64 = wants.iter().map(|w| w.locked).sum();
            let held: u64 = wants.iter().map(|w| w.experts_held + w.cache_held).sum();
            let own = self.source.own().unwrap_or(locked);
            let verdict = state.follow.look(now, available, own);
            let headroom = state.follow.headroom();
            let band = state.follow.band();
            let spare = available.saturating_sub(headroom);
            state.target = match verdict {
                Verdict::Give(missing) => held.saturating_sub(missing),
                Verdict::Take(more) => held + more,
                // An allowance that is not used is not kept when the memory
                // behind it has gone to others.
                Verdict::Stay if claim => held + spare.saturating_sub(band / 2),
                Verdict::Stay => state.target.min(held + spare),
            };
            // The tiers of experts first, the prompt caches with the rest.
            let experts =
                share(state.target, &wants.iter().map(|w| w.experts_want).collect::<Vec<_>>());
            let rest = state.target - experts.iter().sum::<u64>();
            let caches = share(rest, &wants.iter().map(|w| w.cache_want).collect::<Vec<_>>());
            // How long a release takes is what the headroom is sized by:
            // every engine times its own, from the look that asked for it to
            // the look that finds it holding that much less.
            let mut owed = 0;
            for (i, member) in members.iter().enumerate() {
                let over = wants[i].experts_held.saturating_sub(experts[i])
                    + wants[i].cache_held.saturating_sub(caches[i]);
                owed += over;
                let holds = wants[i].experts_held + wants[i].cache_held;
                if let Some(took) = member.allow(experts[i], caches[i], now, holds, over) {
                    state.follow.released(took);
                }
            }
            let look = Look { available, headroom, verdict, held, target: state.target, owed };
            (look, members)
        };
        // The users act outside the machine's lock: a release takes time.
        for member in members {
            member.follow();
        }
        Some(look)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    fn rule() -> Rule {
        Rule {
            period: 50 * MS,
            reserve: 100 * MIB,
            speed: GIB,
            latency: 50 * MS,
            piece: 64 * MIB,
            calm: Duration::from_secs(3),
            calm_max: Duration::from_secs(48),
            settle: Duration::from_secs(60),
            forget: Duration::from_secs(600),
            most: 64 * GIB,
        }
    }

    #[test]
    fn the_headroom_is_the_kernels_reserve_and_what_others_take_while_the_engine_reacts() {
        let follow = Follow::new(rule());
        // 100 MiB and a GiB per second for 50 + 50 ms.
        assert_eq!(follow.headroom(), 100 * MIB + GIB / 10);
        assert_eq!(follow.band(), 128 * MIB, "two pieces: more than a quarter of the headroom");
    }

    #[test]
    fn the_watermarks_of_the_zones_are_read_from_the_kernels_text() {
        let text = "Node 0, zone      DMA\n  pages free     3840\n        min      7\n        \
                    low      10\n        high     13\n        spanned  4095\nNode 0, zone   \
                    Normal\n        min      16586\n        low      24551\n        high     \
                    32516\n      nr_dirty_threshold 12\n        high:    63\n";
        assert_eq!(high_watermarks(text, 4096), (13 + 32516) * 4096);
        assert_eq!(high_watermarks("no zones here", 4096), 0);
    }

    #[test]
    fn less_than_the_headroom_is_given_back_at_once_and_free_memory_is_taken_after_the_wait() {
        let mut follow = Follow::new(rule());
        let h = follow.headroom();
        let band = follow.band();
        let mut now = Duration::ZERO;
        assert_eq!(follow.look(now, h + band - 1, 0), Verdict::Stay, "inside the band");
        now += Duration::from_secs(1);
        // 30 MiB below, by a program that grew more slowly than is assumed.
        assert_eq!(follow.look(now, h - 30 * MIB, 0), Verdict::Give(30 * MIB));
        // Free again: nothing is taken before the memory has been free for
        // the wait.
        let free = h + band + 512 * MIB;
        now += 50 * MS;
        assert_eq!(follow.look(now, free, 0), Verdict::Stay);
        now += Duration::from_secs(3) - MS;
        assert_eq!(follow.look(now, free, 0), Verdict::Stay);
        now += MS;
        let h = follow.headroom();
        assert_eq!(follow.look(now, free, 0), Verdict::Take(free - h - follow.band() / 2));
    }

    #[test]
    fn a_program_that_allocates_and_frees_is_followed_up_more_and_more_slowly() {
        let mut follow = Follow::new(rule());
        let mut now = Duration::ZERO;
        let mut waits = Vec::new();
        for _ in 0..6 {
            // The program frees: the engine waits, then takes.
            let free = follow.headroom() + follow.band() + GIB;
            let since = now;
            loop {
                now += 50 * MS;
                if let Verdict::Take(_) = follow.look(now, free, 0) {
                    break;
                }
            }
            waits.push((now - since).as_secs());
            // Ten seconds later it allocates again.
            now += Duration::from_secs(10);
            let short = follow.headroom() - MIB;
            assert!(matches!(follow.look(now, short, 0), Verdict::Give(_)));
        }
        assert_eq!(waits, [3, 6, 12, 24, 48, 48], "doubled up to the most");
        // A give long after the last take starts the wait afresh.
        let free = follow.headroom() + follow.band() + GIB;
        while follow.look(now, free, 0) == Verdict::Stay {
            now += 50 * MS;
        }
        now += Duration::from_secs(61);
        assert!(matches!(follow.look(now, 0, 0), Verdict::Give(_)));
        assert_eq!(follow.wait(), Duration::from_secs(3));
    }

    #[test]
    fn the_headroom_grows_with_the_speed_of_the_others_and_with_a_slow_release() {
        let mut follow = Follow::new(rule());
        let before = follow.headroom();
        // Others take 4 GiB in a second; the engine's own 2 GiB are not theirs.
        let mut now = Duration::ZERO;
        follow.look(now, 20 * GIB, GIB);
        now += Duration::from_secs(1);
        follow.look(now, 14 * GIB, 3 * GIB);
        assert_eq!(follow.seen().0, 4 * GIB);
        assert_eq!(follow.headroom(), 100 * MIB + 4 * GIB / 10);
        // A release that took 400 ms.
        follow.released(400 * MS);
        let part = (follow.headroom() - 100 * MIB) as f64 / GIB as f64;
        assert!((part - 4.0 * 0.45).abs() < 1e-6, "{part}");
        // Half of what was seen above the assumptions is forgotten in ten
        // minutes, nearly all of it in a day.
        now += Duration::from_secs(600);
        follow.look(now, 14 * GIB, 3 * GIB);
        let (speed, latency) = follow.seen();
        assert!((speed as f64 / GIB as f64 - 2.5).abs() < 0.01, "{speed}");
        assert!((latency.as_secs_f64() - 0.225).abs() < 0.001, "{latency:?}");
        now += Duration::from_secs(86_400);
        follow.look(now, 14 * GIB, 3 * GIB);
        assert!(follow.headroom() - before < MIB);
        // Whatever was seen, the headroom has a most.
        let mut follow = Follow::new(Rule { most: GIB, ..rule() });
        follow.look(Duration::ZERO, 60 * GIB, 0);
        follow.look(Duration::from_millis(50), 0, 0);
        assert_eq!(follow.seen().0, 1200 * GIB, "60 GiB in 50 ms");
        assert_eq!(follow.headroom(), GIB);
        // The engine's own locking is not growth of the others.
        let mut follow = Follow::new(rule());
        follow.look(Duration::ZERO, 20 * GIB, 0);
        follow.look(Duration::from_secs(1), 10 * GIB, 10 * GIB);
        assert_eq!(follow.seen().0, GIB);
    }

    #[test]
    fn memory_is_shared_evenly_as_far_as_each_can_use_it() {
        assert_eq!(share(90, &[100, 100, 100]), [30, 30, 30]);
        assert_eq!(share(90, &[10, 100, 100]), [10, 40, 40]);
        assert_eq!(share(500, &[10, 100, 100]), [10, 100, 100]);
        assert_eq!(share(0, &[10, 100]), [0, 0]);
        assert_eq!(share(7, &[]), Vec::<u64>::new());
        assert_eq!(share(5, &[3, 3, 3]).iter().sum::<u64>(), 5);
    }
}
