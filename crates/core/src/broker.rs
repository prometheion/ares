//! One limit of host RAM for memory that moves.
//!
//! A [`Ledger`](crate::Ledger) answers "does it fit" once, for memory that is
//! planned at load time and then stays put. Host RAM is not like that. It
//! serves users whose share changes while the engine runs: the K/V pages of
//! running sequences that do not fit in VRAM, the RAM tier of MoE experts, the
//! prompt cache (pages of finished conversations, which also exist on SSD),
//! row caches and staging buffers. The person running the engine wants one
//! setting for all of it, an upper limit of RAM, and within it the users have
//! to make room for each other without anyone planning that by hand. The
//! failure to design out is the one the ledger exists for, seen from the other
//! side: a running request must never fail to offload its K/V pages because a
//! cache is sitting on memory it could do without.
//!
//! # Model
//!
//! A [`Broker`] has a fixed capacity. [`Consumer`]s register with a name and a
//! [`Class`]:
//!
//! - [`Class::Required`]: memory a running request cannot do without (K/V
//!   offload of live sequences). It is never taken away, so it has no floor:
//!   every byte it holds is its floor.
//! - [`Class::Elastic`]: a cache. It has a `priority` (higher keeps its memory
//!   longer) and a `floor`, the least it needs to be useful (it may be 0). It
//!   can be asked to give memory back, down to its floor and no further.
//!
//! # Protocol
//!
//! The broker owns no memory and moves none; it keeps the books. A consumer
//! that wants more calls [`Broker::grow`]:
//!
//! 1. If the bytes are free, they are granted and recorded in the same call
//!    ([`Growth::Granted`]). Only then may the consumer allocate them. This is
//!    the one way usage goes up, so the books cannot hold memory that was not
//!    granted.
//! 2. If not, the answer is a [`Plan`] ([`Growth::Reclaim`]): who must give
//!    back how much so that the request then fits. Nothing was recorded. The
//!    caller frees that memory for real (evicts cache entries, drops a tier),
//!    reports each release with [`Broker::shrink`], and asks again.
//! 3. If even a full plan would not make room, the request is refused with
//!    [`BrokerError::Exhausted`], which carries how much was asked, how much
//!    is free, how much could be reclaimed, and every consumer with its class,
//!    priority, usage and floor, so whoever reads it sees who holds the memory.
//!
//! The retry in step 2 asks again from scratch instead of trusting the plan,
//! so a plan that is only partly carried out, or memory that changed hands in
//! between, can never produce a grant that does not fit.
//!
//! # Who may take from whom
//!
//! - A required consumer may take from any elastic consumer, lowest priority
//!   first.
//! - An elastic consumer may take from elastic consumers of strictly lower
//!   priority only: never from an equal (two caches of one rank would evict
//!   each other forever), never from a higher one, never from a required one.
//! - Among consumers of one priority the one **furthest above its floor** is
//!   taken from first, and it gives all it can (down to its floor) before the
//!   next is asked. The order is fixed from the state before the plan. When
//!   two are equally far above their floors, the one **registered first** goes
//!   first, so a plan is a pure function of the books.
//! - A plan never takes a consumer below its floor and asks for exactly what
//!   is missing, `requested - free`, and no more. Consumers that are not
//!   needed are not in the plan.
//!
//! A consumer may always shrink by itself, also below its floor: the floor
//! limits what others may take, not what the owner may free. Its floor can be
//! changed with [`Broker::set_floor`]; it can be lowered at any time and
//! raised only as far as the consumer holds plus what is free (a floor above
//! that could not be met by anyone and is more likely a mistake in units than
//! a wish). [`Broker::unregister`] frees everything the consumer holds.
//!
//! [`Broker::obtainable`] answers "how much could this consumer get right
//! now, counting what could be reclaimed for it" without changing anything.
//! Caches size themselves by it.
//!
//! # Why handles
//!
//! [`Broker::register`] returns a [`Consumer`], which is neither `Clone` nor
//! `Copy`. Growing, shrinking and changing the floor need the handle, and
//! [`Broker::unregister`] consumes it. So only the owner of a consumer can
//! change its books, a consumer cannot be used after it was unregistered, and
//! a plan, which names consumers by [`ConsumerId`], cannot be turned into a
//! change of someone else's books by mistake: an id identifies, a handle
//! allows. Ids are never reused within a broker. A handle belongs to the
//! broker that issued it; offering it to another broker is an error
//! ([`BrokerError::Unknown`]) as long as that broker has no consumer of the
//! same number, and that is the only protection this crate can give without
//! shared state.
//!
//! Dropping a handle without [`Broker::unregister`] leaves its bytes counted
//! as used. Like everything here, the broker is plain data owned by one
//! thread, with no locks and no interior mutability, so it cannot notice.
//!
//! # Example
//!
//! A prompt cache has grown into all the RAM it was given. A running request
//! then needs 6 GiB of K/V pages in RAM, and the cache makes room:
//!
//! ```
//! use ares_core::units::GIB;
//! use ares_core::{Broker, Class, Growth};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut broker = Broker::new("host RAM", 32 * GIB);
//! let kv = broker.register("kv offload", Class::Required)?;
//! let cache = broker.register("prompt cache", Class::Elastic { priority: 1, floor: GIB })?;
//!
//! // Nobody else wants memory, so the cache takes what is there.
//! assert_eq!(broker.grow(&cache, 32 * GIB)?, Growth::Granted);
//!
//! // The request cannot be granted as it is, but the broker says who can help.
//! let Growth::Reclaim(plan) = broker.grow(&kv, 6 * GIB)? else {
//!     unreachable!("the cache holds everything");
//! };
//! for release in plan.releases() {
//!     assert_eq!(release.consumer, cache.id());
//!     assert_eq!(release.bytes, 6 * GIB);
//!     // Evict 6 GiB of cache pages for real (they are on SSD as well), then say so.
//!     broker.shrink(&cache, release.bytes)?;
//! }
//!
//! // Asking again now succeeds. Only after this may the pages be allocated.
//! assert_eq!(broker.grow(&kv, 6 * GIB)?, Growth::Granted);
//! assert_eq!(broker.free(), 0);
//! # Ok(())
//! # }
//! ```

use std::cmp::Reverse;
use std::fmt;

use crate::units::Bytes;

/// What a consumer is, which decides whether and by whom memory is taken from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Memory a running request cannot do without. Never taken away.
    Required,
    /// A cache that gives memory back when someone who ranks above it needs
    /// it. A higher `priority` keeps its memory longer. Memory is taken from it
    /// down to `floor`, the least it needs to be useful.
    Elastic { priority: u32, floor: u64 },
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Class::Required => f.write_str("required"),
            Class::Elastic { priority, floor } => {
                write!(f, "elastic, priority {priority}, floor {}", Bytes(*floor))
            }
        }
    }
}

/// The name of a consumer in plans, errors and reports. It identifies and
/// allows nothing: changing a consumer's books takes its [`Consumer`] handle.
/// Ids are never reused within a broker.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConsumerId(u64);

impl fmt::Display for ConsumerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// The owner's handle to a registered consumer.
///
/// It is not `Clone`: whoever holds it is the only one who can grow, shrink or
/// unregister the consumer, and unregistering consumes it.
#[derive(Debug)]
#[must_use = "dropping a Consumer without unregistering it leaves its bytes counted as used"]
pub struct Consumer {
    id: ConsumerId,
}

impl Consumer {
    pub fn id(&self) -> ConsumerId {
        self.id
    }
}

/// What the broker knows about one consumer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerInfo {
    pub id: ConsumerId,
    pub name: String,
    pub class: Class,
    /// Bytes granted and not yet reported as released.
    pub usage: u64,
}

impl ConsumerInfo {
    /// Bytes others may take from this consumer: what it holds above its
    /// floor. Always 0 for a required consumer.
    pub fn reclaimable(&self) -> u64 {
        match self.class {
            Class::Required => 0,
            Class::Elastic { floor, .. } => self.usage.saturating_sub(floor),
        }
    }
}

/// One line of a report: usage, name, class.
impl fmt::Display for ConsumerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:>12}  {}  [{}]", Bytes(self.usage).to_string(), self.name, self.class)
    }
}

/// The consumers as indented lines without a final newline, for error messages.
struct Rows<'a>(&'a [ConsumerInfo]);

impl fmt::Display for Rows<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "  {row}")?;
        }
        Ok(())
    }
}

/// Why the broker refused something. Nothing was changed.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum BrokerError {
    #[error(
        "{broker}: cannot grow '{name}' by {requested}: {free} free and {reclaimable} that \
         others can give back leave it {} short (limit {capacity}); the limit would have to be \
         that much higher, or some floors lower. Who holds the memory:\n{}",
        Bytes(.requested.0 - .free.0 - .reclaimable.0),
        Rows(.consumers)
    )]
    Exhausted {
        broker: String,
        name: String,
        requested: Bytes,
        free: Bytes,
        /// What the plan could take for this consumer: above-floor memory of
        /// the consumers it may take from, not what everyone holds above floor.
        reclaimable: Bytes,
        capacity: Bytes,
        /// Every consumer, in registration order.
        consumers: Vec<ConsumerInfo>,
    },
    #[error("{broker}: a consumer named '{name}' exists already")]
    Duplicate { broker: String, name: String },
    #[error(
        "{broker}: consumer {id} is not registered here; a handle only works with the broker \
         that issued it, and not after it was unregistered"
    )]
    Unknown { broker: String, id: ConsumerId },
    #[error(
        "{broker}: cannot set the floor of '{name}' to {floor}: it holds {usage} and {free} are \
         free, so at most {} could ever be met; free memory first or choose a lower floor",
        Bytes(.usage.0 + .free.0)
    )]
    FloorTooHigh { broker: String, name: String, floor: Bytes, usage: Bytes, free: Bytes },
    #[error("{broker}: '{name}' is a required consumer and has no floor: nothing is taken from it")]
    NotElastic { broker: String, name: String },
    #[error(
        "{broker}: '{name}' reports {released} released but {held} are recorded for it; the \
         consumer's own accounting and the broker's are out of step"
    )]
    ReleasedTooMuch { broker: String, name: String, released: Bytes, held: Bytes },
}

/// One consumer's part of a [`Plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Release {
    pub consumer: ConsumerId,
    pub name: String,
    /// Bytes to free, more than 0 and at most what the consumer holds above
    /// its floor.
    pub bytes: u64,
}

/// What must be freed so that a request fits.
///
/// A plan is advice about a state of the books, not a reservation. It lists
/// the consumers in the order they were chosen, which is also a sensible order
/// to carry it out in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    consumer: ConsumerId,
    requested: u64,
    releases: Vec<Release>,
}

impl Plan {
    /// The consumer the memory is for.
    pub fn consumer(&self) -> ConsumerId {
        self.consumer
    }

    /// How much that consumer asked for.
    pub fn requested(&self) -> u64 {
        self.requested
    }

    /// Who gives back how much.
    pub fn releases(&self) -> &[Release] {
        &self.releases
    }

    /// The sum of the releases: `requested` minus what was free.
    pub fn total(&self) -> u64 {
        self.releases.iter().map(|r| r.bytes).sum()
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "free {} for {} by releasing", Bytes(self.total()), self.consumer)?;
        for (i, release) in self.releases.iter().enumerate() {
            let separator = if i == 0 { " " } else { ", " };
            write!(f, "{separator}{} from '{}'", Bytes(release.bytes), release.name)?;
        }
        Ok(())
    }
}

/// The answer to [`Broker::grow`] when the request is not refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Growth {
    /// The bytes were free. They are recorded for the consumer, which may now
    /// allocate them.
    Granted,
    /// Not granted and nothing recorded. Carry out the plan, report each
    /// release with [`Broker::shrink`], and ask again.
    Reclaim(Plan),
}

/// A fixed budget of host RAM shared by consumers that make room for each
/// other. See the [module documentation](self) for the rules.
#[derive(Clone, Debug)]
pub struct Broker {
    name: String,
    capacity: u64,
    used: u64,
    next_id: u64,
    /// In registration order, so ids ascend.
    entries: Vec<ConsumerInfo>,
}

/// A consumer that a request may take from, and how much it could give.
struct Source {
    index: usize,
    reclaimable: u64,
}

impl Broker {
    /// A broker over `capacity` bytes. `name` identifies it in errors and
    /// reports (for example `"host RAM"`).
    pub fn new(name: impl Into<String>, capacity: u64) -> Self {
        Self { name: name.into(), capacity, used: 0, next_id: 0, entries: Vec::new() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes granted to consumers.
    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn free(&self) -> u64 {
        self.capacity - self.used
    }

    /// The consumers in the order they registered.
    pub fn consumers(&self) -> &[ConsumerInfo] {
        &self.entries
    }

    /// What the broker knows about `id`, or `None` when it is not registered.
    pub fn info(&self, id: ConsumerId) -> Option<&ConsumerInfo> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Registers a consumer that holds nothing yet. Names are unique within a
    /// broker. An elastic floor is held to the rule of [`Broker::set_floor`],
    /// with no usage: it may not exceed what is free now.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        class: Class,
    ) -> Result<Consumer, BrokerError> {
        let name = name.into();
        if self.entries.iter().any(|e| e.name == name) {
            return Err(BrokerError::Duplicate { broker: self.name.clone(), name });
        }
        if let Class::Elastic { floor, .. } = class
            && floor > self.free()
        {
            return Err(self.floor_too_high(name, floor, 0));
        }
        let id = ConsumerId(self.next_id);
        self.next_id += 1;
        self.entries.push(ConsumerInfo { id, name, class, usage: 0 });
        Ok(Consumer { id })
    }

    /// Removes the consumer and frees everything it holds. Returns the bytes
    /// freed. The memory itself must be released by the caller.
    pub fn unregister(&mut self, consumer: Consumer) -> Result<u64, BrokerError> {
        let index = self.index_of(consumer.id)?;
        let entry = self.entries.remove(index);
        self.used -= entry.usage;
        Ok(entry.usage)
    }

    /// Asks to grow `consumer` by `bytes`.
    ///
    /// If the bytes are free they are granted and recorded here and now. If
    /// not, the answer is a [`Plan`] and nothing is recorded; see the module
    /// documentation for who the plan may take from. If even a full plan would
    /// not make room the request is refused with the full picture. A request
    /// of 0 bytes is always granted.
    pub fn grow(&mut self, consumer: &Consumer, bytes: u64) -> Result<Growth, BrokerError> {
        let index = self.index_of(consumer.id)?;
        if bytes <= self.free() {
            self.entries[index].usage += bytes;
            self.used += bytes;
            return Ok(Growth::Granted);
        }
        let need = bytes - self.free();
        let sources = self.sources(index);
        let reclaimable: u64 = sources.iter().map(|s| s.reclaimable).sum();
        if reclaimable < need {
            return Err(BrokerError::Exhausted {
                broker: self.name.clone(),
                name: self.entries[index].name.clone(),
                requested: Bytes(bytes),
                free: Bytes(self.free()),
                reclaimable: Bytes(reclaimable),
                capacity: Bytes(self.capacity),
                consumers: self.entries.clone(),
            });
        }
        let mut missing = need;
        let mut releases = Vec::new();
        for source in sources {
            if missing == 0 {
                break;
            }
            let take = source.reclaimable.min(missing);
            let entry = &self.entries[source.index];
            releases.push(Release { consumer: entry.id, name: entry.name.clone(), bytes: take });
            missing -= take;
        }
        Ok(Growth::Reclaim(Plan { consumer: consumer.id, requested: bytes, releases }))
    }

    /// Reports that `consumer` freed `bytes` of its memory, whether a plan
    /// asked for it or the consumer decided to. Shrinking below the floor is
    /// allowed; shrinking by more than the consumer holds is an error and
    /// changes nothing.
    pub fn shrink(&mut self, consumer: &Consumer, bytes: u64) -> Result<(), BrokerError> {
        let index = self.index_of(consumer.id)?;
        let entry = &mut self.entries[index];
        if bytes > entry.usage {
            return Err(BrokerError::ReleasedTooMuch {
                broker: self.name.clone(),
                name: entry.name.clone(),
                released: Bytes(bytes),
                held: Bytes(entry.usage),
            });
        }
        entry.usage -= bytes;
        self.used -= bytes;
        Ok(())
    }

    /// Changes the floor of an elastic consumer.
    ///
    /// A floor can always be lowered, or set to what it is. It can be raised
    /// only up to what the consumer holds plus what is free: a floor above
    /// that could not be met without taking memory from others, which the
    /// consumer has no right to.
    pub fn set_floor(&mut self, consumer: &Consumer, floor: u64) -> Result<(), BrokerError> {
        let index = self.index_of(consumer.id)?;
        let free = self.free();
        let entry = &mut self.entries[index];
        let Class::Elastic { floor: current, .. } = &mut entry.class else {
            return Err(BrokerError::NotElastic {
                broker: self.name.clone(),
                name: entry.name.clone(),
            });
        };
        if floor > *current && floor > entry.usage + free {
            let (name, usage) = (entry.name.clone(), entry.usage);
            return Err(self.floor_too_high(name, floor, usage));
        }
        *current = floor;
        Ok(())
    }

    /// How many more bytes `consumer` could get right now: what is free plus
    /// what a plan could take for it. A request of exactly this size is never
    /// refused, one byte more always is. Changes nothing.
    pub fn obtainable(&self, consumer: &Consumer) -> Result<u64, BrokerError> {
        let index = self.index_of(consumer.id)?;
        Ok(self.free() + self.sources(index).iter().map(|s| s.reclaimable).sum::<u64>())
    }

    fn index_of(&self, id: ConsumerId) -> Result<usize, BrokerError> {
        self.entries
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| BrokerError::Unknown { broker: self.name.clone(), id })
    }

    fn floor_too_high(&self, name: String, floor: u64, usage: u64) -> BrokerError {
        BrokerError::FloorTooHigh {
            broker: self.name.clone(),
            name,
            floor: Bytes(floor),
            usage: Bytes(usage),
            free: Bytes(self.free()),
        }
    }

    /// The consumers the one at `requester` may take from, in the order a plan
    /// takes from them: lowest priority first, then furthest above the floor,
    /// then registered first. Consumers that have nothing above their floor
    /// are left out.
    fn sources(&self, requester: usize) -> Vec<Source> {
        // A required consumer ranks above every cache; a cache only above
        // lower priorities.
        let ceiling = match self.entries[requester].class {
            Class::Required => None,
            Class::Elastic { priority, .. } => Some(priority),
        };
        let mut ranked: Vec<(u32, Source)> = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            let Class::Elastic { priority, .. } = entry.class else { continue };
            let reclaimable = entry.reclaimable();
            if reclaimable > 0 && ceiling.is_none_or(|ceiling| priority < ceiling) {
                ranked.push((priority, Source { index, reclaimable }));
            }
        }
        ranked.sort_by_key(|(priority, s)| {
            (*priority, Reverse(s.reclaimable), self.entries[s.index].id)
        });
        ranked.into_iter().map(|(_, source)| source).collect()
    }
}

impl fmt::Display for Broker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {} of {} used, {} free",
            self.name,
            Bytes(self.used),
            Bytes(self.capacity),
            Bytes(self.free())
        )?;
        for entry in &self.entries {
            writeln!(f, "  {entry}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::GIB;

    fn elastic(priority: u32, floor: u64) -> Class {
        Class::Elastic { priority, floor }
    }

    /// Registers `name` and grows it by `bytes`, which must fit as it is.
    fn holding(broker: &mut Broker, name: &str, class: Class, bytes: u64) -> Consumer {
        let consumer = broker.register(name, class).unwrap();
        assert_eq!(broker.grow(&consumer, bytes), Ok(Growth::Granted), "{name}");
        consumer
    }

    /// Like `holding`, for a consumer the test never touches again.
    fn fill(broker: &mut Broker, name: &str, class: Class, bytes: u64) {
        drop(holding(broker, name, class, bytes));
    }

    /// The plan of a request that must not be granted as it is.
    fn plan_for(broker: &mut Broker, consumer: &Consumer, bytes: u64) -> Plan {
        match broker.grow(consumer, bytes) {
            Ok(Growth::Reclaim(plan)) => plan,
            other => panic!("expected a plan, got {other:?}"),
        }
    }

    /// `(name, bytes)` of each release, in plan order.
    fn releases(plan: &Plan) -> Vec<(&str, u64)> {
        plan.releases().iter().map(|r| (r.name.as_str(), r.bytes)).collect()
    }

    /// Carries a plan out the way a caller does: frees, then reports.
    fn carry_out(broker: &mut Broker, plan: &Plan, owners: &[&Consumer]) {
        for release in plan.releases() {
            let owner = owners.iter().find(|c| c.id() == release.consumer).expect("an owner");
            broker.shrink(owner, release.bytes).unwrap();
        }
    }

    fn usage(broker: &Broker, consumer: &Consumer) -> u64 {
        broker.info(consumer.id()).unwrap().usage
    }

    /// Three caches of ranks 1, 2 and 3 and a required consumer fill 100 bytes.
    fn full_broker() -> (Broker, Consumer, [Consumer; 3]) {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        let a = holding(&mut broker, "a", elastic(1, 0), 30);
        let b = holding(&mut broker, "b", elastic(2, 0), 30);
        let c = holding(&mut broker, "c", elastic(3, 0), 30);
        assert_eq!(broker.free(), 0);
        (broker, kv, [a, b, c])
    }

    #[test]
    fn grants_what_is_free_and_records_it() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = broker.register("kv", Class::Required).unwrap();
        assert_eq!(broker.grow(&kv, 60), Ok(Growth::Granted));
        assert_eq!((usage(&broker, &kv), broker.used(), broker.free()), (60, 60, 40));
        assert_eq!(broker.grow(&kv, 40), Ok(Growth::Granted), "exactly what is left fits");
        assert_eq!((broker.used(), broker.free()), (100, 0));
        assert_eq!(broker.grow(&kv, 0), Ok(Growth::Granted), "nothing always fits");
    }

    #[test]
    fn a_plan_records_nothing_and_a_grant_needs_the_retry() {
        let (mut broker, kv, [a, ..]) = full_broker();
        let before = broker.to_string();
        let plan = plan_for(&mut broker, &kv, 5);
        assert_eq!(releases(&plan), [("a", 5)]);
        assert_eq!(broker.to_string(), before, "asking for a plan changes no books");
        carry_out(&mut broker, &plan, &[&a]);
        assert_eq!(usage(&broker, &kv), 10, "still not granted: the caller must ask again");
        assert_eq!(broker.grow(&kv, 5), Ok(Growth::Granted));
        assert_eq!((usage(&broker, &kv), usage(&broker, &a), broker.free()), (15, 25, 0));
    }

    #[test]
    fn a_required_request_takes_from_the_lowest_priority_cache_first() {
        let (mut broker, kv, [a, b, c]) = full_broker();
        let plan = plan_for(&mut broker, &kv, 25);
        assert_eq!((plan.consumer(), plan.requested(), plan.total()), (kv.id(), 25, 25));
        assert_eq!(releases(&plan), [("a", 25)], "b and c are not touched");

        // More than the lowest holds: it gives all, the next one the rest.
        let plan = plan_for(&mut broker, &kv, 35);
        assert_eq!(releases(&plan), [("a", 30), ("b", 5)]);
        let plan = plan_for(&mut broker, &kv, 65);
        assert_eq!(releases(&plan), [("a", 30), ("b", 30), ("c", 5)]);
        carry_out(&mut broker, &plan, &[&a, &b, &c]);
        assert_eq!(broker.grow(&kv, 65), Ok(Growth::Granted));
    }

    #[test]
    fn a_required_request_reaches_a_higher_cache_only_when_the_lower_is_at_its_floor() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        fill(&mut broker, "low", elastic(1, 25), 30);
        fill(&mut broker, "mid", elastic(2, 0), 30);
        fill(&mut broker, "high", elastic(3, 0), 30);

        // `low` can give 5 and no more, so `mid` supplies the rest.
        let plan = plan_for(&mut broker, &kv, 20);
        assert_eq!(releases(&plan), [("low", 5), ("mid", 15)]);

        // A cache exactly at its floor does not appear at all.
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        fill(&mut broker, "low", elastic(1, 30), 30);
        fill(&mut broker, "mid", elastic(2, 0), 30);
        fill(&mut broker, "high", elastic(3, 0), 30);
        let plan = plan_for(&mut broker, &kv, 10);
        assert_eq!(releases(&plan), [("mid", 10)]);
    }

    #[test]
    fn an_elastic_request_takes_only_from_strictly_lower_priorities() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        let low = holding(&mut broker, "low", elastic(1, 0), 30);
        let mid = holding(&mut broker, "mid", elastic(2, 0), 20);
        let twin = holding(&mut broker, "twin", elastic(2, 0), 20);
        let high = holding(&mut broker, "high", elastic(3, 0), 20);

        // `mid` may take from `low`, never from `twin` (equal) or `high`.
        let plan = plan_for(&mut broker, &mid, 10);
        assert_eq!(releases(&plan), [("low", 10)]);

        // `high` may take from all three below it, still not from `kv`.
        let plan = plan_for(&mut broker, &high, 70);
        assert_eq!(releases(&plan), [("low", 30), ("mid", 20), ("twin", 20)]);

        // What the lower ones hold is not enough: refused, not taken from equals.
        let refused = broker.grow(&mid, 31).unwrap_err();
        let BrokerError::Exhausted { reclaimable, .. } = refused else { panic!("{refused:?}") };
        assert_eq!(reclaimable, Bytes(30), "only `low` counts for `mid`");
        // The lowest cache has nobody to take from at all.
        let refused = broker.grow(&low, 1).unwrap_err();
        let BrokerError::Exhausted { reclaimable, .. } = refused else { panic!("{refused:?}") };
        assert_eq!(reclaimable, Bytes(0));
        // Nothing was disturbed by any of it.
        let held: Vec<u64> =
            [&kv, &low, &mid, &twin, &high].iter().map(|c| usage(&broker, c)).collect();
        assert_eq!(held, [10, 30, 20, 20, 20]);
    }

    #[test]
    fn an_elastic_request_never_takes_from_a_required_consumer() {
        let mut broker = Broker::new("host RAM", 100);
        fill(&mut broker, "kv", Class::Required, 90);
        let cache = holding(&mut broker, "cache", elastic(9, 0), 10);
        let refused = broker.grow(&cache, 1).unwrap_err();
        let BrokerError::Exhausted { free, reclaimable, .. } = refused else {
            panic!("{refused:?}")
        };
        assert_eq!((free, reclaimable), (Bytes(0), Bytes(0)));
    }

    #[test]
    fn floors_hold() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 20);
        let a = holding(&mut broker, "a", elastic(1, 15), 40);
        let b = holding(&mut broker, "b", elastic(2, 40), 40);
        // 20 + 40 + 40: `a` can give 25, `b` nothing.
        let plan = plan_for(&mut broker, &kv, 25);
        assert_eq!(releases(&plan), [("a", 25)]);
        carry_out(&mut broker, &plan, &[&a, &b]);
        assert_eq!(broker.grow(&kv, 25), Ok(Growth::Granted));
        assert_eq!((usage(&broker, &a), usage(&broker, &b)), (15, 40), "both sit on their floors");
        assert!(matches!(broker.grow(&kv, 1), Err(BrokerError::Exhausted { .. })));
    }

    #[test]
    fn a_plan_asks_for_exactly_what_is_missing() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        fill(&mut broker, "a", elastic(1, 0), 20);
        fill(&mut broker, "b", elastic(2, 0), 20);
        fill(&mut broker, "c", elastic(3, 0), 20);
        // 70 used, 30 free. Asking for 42 misses 12, and only 12 is released.
        let plan = plan_for(&mut broker, &kv, 42);
        assert_eq!((plan.total(), releases(&plan)), (12, vec![("a", 12)]));
        // Asking for 55 misses 25: all of `a` and a part of `b`.
        let plan = plan_for(&mut broker, &kv, 55);
        assert_eq!((plan.total(), releases(&plan)), (25, vec![("a", 20), ("b", 5)]));
        // Asking for what is free is granted, no plan.
        assert_eq!(broker.grow(&kv, 30), Ok(Growth::Granted));
    }

    #[test]
    fn equals_give_in_order_of_distance_above_floor_then_registration() {
        let mut broker = Broker::new("host RAM", 200);
        let kv = holding(&mut broker, "kv", Class::Required, 10);
        // 20 above the floor, registered first, name sorts last.
        fill(&mut broker, "zeta", elastic(1, 10), 30);
        // 10 above the floor although it holds the most.
        fill(&mut broker, "big", elastic(1, 30), 40);
        // 20 above the floor, like `zeta`, registered later.
        fill(&mut broker, "alpha", elastic(1, 5), 25);
        fill(&mut broker, "other", elastic(2, 0), 95);

        let plan = plan_for(&mut broker, &kv, 85);
        assert_eq!(
            releases(&plan),
            [("zeta", 20), ("alpha", 20), ("big", 10), ("other", 35)],
            "furthest above floor first, a tie goes to the earlier registration, then up a rank"
        );
        // Taking less stops inside the first.
        let plan = plan_for(&mut broker, &kv, 25);
        assert_eq!(releases(&plan), [("zeta", 20), ("alpha", 5)]);
    }

    #[test]
    fn a_refusal_carries_the_full_picture_and_changes_nothing() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 20);
        fill(&mut broker, "a", elastic(1, 15), 40);
        fill(&mut broker, "b", elastic(2, 40), 30);
        let before = broker.to_string();

        let refused = broker.grow(&kv, 50).unwrap_err();
        let BrokerError::Exhausted {
            broker: name,
            name: asker,
            requested,
            free,
            reclaimable,
            capacity,
            consumers,
        } = &refused
        else {
            panic!("{refused:?}")
        };
        assert_eq!(name, "host RAM");
        assert_eq!(asker, "kv");
        assert_eq!(
            (*requested, *free, *reclaimable, *capacity),
            (Bytes(50), Bytes(10), Bytes(25), Bytes(100))
        );
        assert_eq!(consumers.as_slice(), broker.consumers());

        let message = refused.to_string();
        assert_eq!(
            message,
            "host RAM: cannot grow 'kv' by 50 B: 10 B free and 25 B that others can give back \
             leave it 15 B short (limit 100 B); the limit would have to be that much higher, or \
             some floors lower. Who holds the memory:\n\
             \x20         20 B  kv  [required]\n\
             \x20         40 B  a  [elastic, priority 1, floor 15 B]\n\
             \x20         30 B  b  [elastic, priority 2, floor 40 B]"
        );
        assert_eq!(broker.to_string(), before, "a refusal changes no books");

        // The message is true: 15 B more room and the request goes through.
        let mut roomier = Broker::new("host RAM", 115);
        let kv = holding(&mut roomier, "kv", Class::Required, 20);
        fill(&mut roomier, "a", elastic(1, 15), 40);
        fill(&mut roomier, "b", elastic(2, 40), 30);
        assert!(matches!(roomier.grow(&kv, 50), Ok(Growth::Reclaim(_))));
    }

    #[test]
    fn a_request_beyond_any_size_is_refused_without_overflow() {
        let (mut broker, kv, [..]) = full_broker();
        let refused = broker.grow(&kv, u64::MAX).unwrap_err();
        let BrokerError::Exhausted { requested, .. } = refused else { panic!("{refused:?}") };
        assert_eq!(requested, Bytes(u64::MAX));
        assert_eq!(broker.used(), 100);
        assert_eq!(broker.obtainable(&kv), Ok(90), "10 of its own are not for taking");
    }

    #[test]
    fn unregistering_frees_what_the_consumer_held() {
        let (mut broker, kv, [a, b, c]) = full_broker();
        // All 90 bytes of the caches together are not enough for 91.
        let refused = broker.grow(&kv, 91).unwrap_err();
        assert!(matches!(refused, BrokerError::Exhausted { .. }));

        let b_id = b.id();
        assert_eq!(broker.unregister(b), Ok(30));
        assert_eq!((broker.used(), broker.free()), (70, 30));
        assert!(broker.info(b_id).is_none());
        assert_eq!(broker.consumers().len(), 3);
        assert_eq!(broker.grow(&kv, 30), Ok(Growth::Granted), "the freed bytes are usable");

        // Ids are not reused, so nothing that still names `b` can hit a newcomer.
        let newcomer = broker.register("b", elastic(2, 0)).unwrap();
        assert_ne!(newcomer.id(), b_id);
        assert!(broker.info(b_id).is_none());
        assert_eq!(usage(&broker, &newcomer), 0);
        let _ = (a, c);
    }

    #[test]
    fn a_consumer_may_shrink_by_itself_but_not_below_zero() {
        let mut broker = Broker::new("host RAM", 100);
        let cache = holding(&mut broker, "cache", elastic(1, 30), 50);
        let other = holding(&mut broker, "other", Class::Required, 20);

        broker.shrink(&cache, 15).unwrap();
        assert_eq!((usage(&broker, &cache), broker.used()), (35, 55));
        // Below the floor is the owner's choice: the floor only protects from others.
        broker.shrink(&cache, 10).unwrap();
        assert_eq!(usage(&broker, &cache), 25);
        assert_eq!(broker.info(cache.id()).unwrap().reclaimable(), 0);

        let refused = broker.shrink(&cache, 26).unwrap_err();
        assert_eq!(
            refused,
            BrokerError::ReleasedTooMuch {
                broker: "host RAM".into(),
                name: "cache".into(),
                released: Bytes(26),
                held: Bytes(25),
            }
        );
        assert!(refused.to_string().contains("26 B released but 25 B are recorded"));
        assert_eq!((usage(&broker, &cache), broker.used()), (25, 45), "refused means unchanged");
        broker.shrink(&cache, 25).unwrap();
        broker.shrink(&other, 0).unwrap();
        assert_eq!((usage(&broker, &cache), broker.used()), (0, 20));
    }

    #[test]
    fn a_floor_can_always_be_lowered_and_raised_up_to_usage_plus_free() {
        let mut broker = Broker::new("host RAM", 100);
        let a = holding(&mut broker, "a", elastic(1, 10), 30);
        let b = holding(&mut broker, "b", elastic(2, 0), 50);
        // `a` holds 30 and 20 are free: up to 50 is allowed, 51 is not.
        broker.set_floor(&a, 50).unwrap();
        assert_eq!(broker.info(a.id()).unwrap().class, elastic(1, 50));
        let refused = broker.set_floor(&a, 51).unwrap_err();
        assert_eq!(
            refused,
            BrokerError::FloorTooHigh {
                broker: "host RAM".into(),
                name: "a".into(),
                floor: Bytes(51),
                usage: Bytes(30),
                free: Bytes(20),
            }
        );
        assert!(refused.to_string().contains("at most 50 B could ever be met"));
        assert_eq!(broker.info(a.id()).unwrap().class, elastic(1, 50), "refused means unchanged");

        // `b` takes the free memory: `a`'s floor of 50 is now beyond what it could
        // reach. It can stay, and it can go down, but not up.
        assert_eq!(broker.grow(&b, 20), Ok(Growth::Granted));
        broker.set_floor(&a, 50).unwrap();
        broker.set_floor(&a, 45).unwrap();
        assert!(matches!(broker.set_floor(&a, 46), Err(BrokerError::FloorTooHigh { .. })));
        broker.set_floor(&a, 0).unwrap();
        assert_eq!(broker.info(a.id()).unwrap().reclaimable(), 30, "a lower floor frees memory");
        assert_eq!(broker.set_floor(&a, 30), Ok(()), "a floor at the usage is within the rule");
    }

    #[test]
    fn a_floor_is_held_to_the_same_rule_at_registration() {
        let mut broker = Broker::new("host RAM", 100);
        fill(&mut broker, "kv", Class::Required, 70);
        let refused = broker.register("cache", elastic(1, 31)).unwrap_err();
        assert!(matches!(refused, BrokerError::FloorTooHigh { floor: Bytes(31), .. }));
        assert!(broker.info(ConsumerId(1)).is_none(), "a refused registration leaves no trace");
        drop(broker.register("cache", elastic(1, 30)).unwrap());
    }

    #[test]
    fn a_required_consumer_has_no_floor_to_change() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = broker.register("kv", Class::Required).unwrap();
        assert_eq!(
            broker.set_floor(&kv, 1),
            Err(BrokerError::NotElastic { broker: "host RAM".into(), name: "kv".into() })
        );
    }

    #[test]
    fn names_are_unique() {
        let mut broker = Broker::new("host RAM", 100);
        let _cache = broker.register("cache", elastic(1, 0)).unwrap();
        let refused = broker.register("cache", Class::Required).unwrap_err();
        assert_eq!(
            refused,
            BrokerError::Duplicate { broker: "host RAM".into(), name: "cache".into() }
        );
        assert_eq!(broker.consumers().len(), 1);
    }

    #[test]
    fn a_handle_of_another_broker_is_an_error_not_a_panic() {
        let mut here = Broker::new("here", 100);
        let mut there = Broker::new("there", 100);
        let _first = here.register("first", Class::Required).unwrap();
        let stranger = there.register("a", Class::Required).unwrap();
        let second = there.register("b", Class::Required).unwrap();
        // Id 1 does not exist in `here`.
        let refused = here.grow(&second, 1).unwrap_err();
        assert_eq!(refused, BrokerError::Unknown { broker: "here".into(), id: second.id() });
        assert!(refused.to_string().contains("consumer #1 is not registered here"));
        assert!(matches!(here.shrink(&second, 0), Err(BrokerError::Unknown { .. })));
        assert!(matches!(here.set_floor(&second, 0), Err(BrokerError::Unknown { .. })));
        assert!(matches!(here.obtainable(&second), Err(BrokerError::Unknown { .. })));
        assert!(matches!(here.unregister(second), Err(BrokerError::Unknown { .. })));
        let _ = stranger;
    }

    #[test]
    fn obtainable_counts_what_can_be_reclaimed_for_that_consumer_and_changes_nothing() {
        let mut broker = Broker::new("host RAM", 100);
        let kv = holding(&mut broker, "kv", Class::Required, 20);
        let a = holding(&mut broker, "a", elastic(1, 10), 30);
        let b = holding(&mut broker, "b", elastic(2, 0), 30);
        let c = holding(&mut broker, "c", elastic(3, 5), 10);
        let before = broker.to_string();
        // 10 free. `a` gives 20, `b` 30, `c` 5.
        assert_eq!(broker.obtainable(&kv), Ok(10 + 20 + 30 + 5));
        assert_eq!(broker.obtainable(&c), Ok(10 + 20 + 30), "not from itself");
        assert_eq!(broker.obtainable(&b), Ok(10 + 20), "not from equals or higher");
        assert_eq!(broker.obtainable(&a), Ok(10), "the lowest has only what is free");
        assert_eq!(broker.to_string(), before);

        // The number is exact: that much goes through, one byte more is refused.
        for consumer in [&kv, &a, &b, &c] {
            let most = broker.obtainable(consumer).unwrap();
            assert!(broker.clone().grow(consumer, most).is_ok());
            assert!(matches!(
                broker.clone().grow(consumer, most + 1),
                Err(BrokerError::Exhausted { .. })
            ));
        }
    }

    #[test]
    fn the_report_shows_capacity_used_free_and_every_consumer() {
        let mut broker = Broker::new("host RAM", 100);
        fill(&mut broker, "kv", Class::Required, 20);
        fill(&mut broker, "prompt cache", elastic(1, 10), 30);
        let _rows = broker.register("row cache", elastic(2, 0)).unwrap();
        assert_eq!(
            broker.to_string(),
            "host RAM: 50 B of 100 B used, 50 B free\n\
             \x20         20 B  kv  [required]\n\
             \x20         30 B  prompt cache  [elastic, priority 1, floor 10 B]\n\
             \x20          0 B  row cache  [elastic, priority 2, floor 0 B]\n"
        );
    }

    #[test]
    fn a_plan_prints_who_gives_how_much() {
        let (mut broker, kv, _caches) = full_broker();
        let plan = plan_for(&mut broker, &kv, 35);
        assert_eq!(plan.to_string(), "free 35 B for #0 by releasing 30 B from 'a', 5 B from 'b'");
    }

    /// The numbers of a real machine: 32 GiB for the whole engine, shared by
    /// the prompt cache, the RAM tier of the experts, a row cache and the K/V
    /// offload of running requests.
    #[test]
    fn a_running_request_displaces_caches_down_to_their_floors() {
        let mut broker = Broker::new("host RAM", 32 * GIB);
        let kv = broker.register("kv offload", Class::Required).unwrap();
        // Prompt pages are on SSD as well, so this cache is the first to go.
        let prompt = broker.register("prompt cache", elastic(1, GIB)).unwrap();
        let rows = broker.register("row cache", elastic(2, 2 * GIB)).unwrap();
        // Below 8 GiB the expert tier is too small to help.
        let experts = broker.register("expert tier", elastic(3, 8 * GIB)).unwrap();

        // The caches grow until the limit is reached. The caches have no one
        // to take from but each other, and they start with everything free.
        for (consumer, bytes) in [(&experts, 14 * GIB), (&rows, 6 * GIB), (&prompt, 12 * GIB)] {
            assert_eq!(broker.grow(consumer, bytes), Ok(Growth::Granted));
        }
        assert_eq!(broker.free(), 0);

        // The prompt cache wants more: nothing ranks below it.
        assert!(matches!(broker.grow(&prompt, GIB), Err(BrokerError::Exhausted { .. })));
        // The expert tier wants more: it may take from the other two, and the
        // plan names them. Nothing is carried out here.
        let plan = plan_for(&mut broker, &experts, GIB);
        assert_eq!(releases(&plan), [("prompt cache", GIB)]);

        // A running request needs 6 GiB of K/V pages in RAM. The prompt cache,
        // the lowest, pays for all of it and keeps 6 GiB.
        let plan = plan_for(&mut broker, &kv, 6 * GIB);
        assert_eq!(releases(&plan), [("prompt cache", 6 * GIB)]);
        carry_out(&mut broker, &plan, &[&prompt, &rows, &experts]);
        assert_eq!(broker.grow(&kv, 6 * GIB), Ok(Growth::Granted));
        assert_eq!([&kv, &prompt, &rows, &experts].map(|c| usage(&broker, c) / GIB), [6, 6, 6, 14]);

        // Then another 20 GiB. All the caches could give is what they hold
        // above their floors: 5 + 4 + 6 = 15 GiB. The floors keep 11 GiB.
        let refused = broker.grow(&kv, 20 * GIB).unwrap_err();
        let BrokerError::Exhausted { free, reclaimable, requested, .. } = &refused else {
            panic!("{refused:?}")
        };
        assert_eq!((*requested, *free, *reclaimable), (Bytes(20 * GIB), Bytes(0), Bytes(15 * GIB)));
        let message = refused.to_string();
        assert!(message.contains("cannot grow 'kv offload' by 20.00 GiB"), "{message}");
        assert!(message.contains("0 B free and 15.00 GiB that others can give back"), "{message}");
        assert!(message.contains("5.00 GiB short"), "{message}");
        for line in [
            "6.00 GiB  kv offload  [required]",
            "6.00 GiB  prompt cache  [elastic, priority 1, floor 1.00 GiB]",
            "6.00 GiB  row cache  [elastic, priority 2, floor 2.00 GiB]",
            "14.00 GiB  expert tier  [elastic, priority 3, floor 8.00 GiB]",
        ] {
            assert!(message.contains(line), "{line} is missing from\n{message}");
        }

        // The caller can size by what is possible, and gets exactly that.
        let most = broker.obtainable(&kv).unwrap();
        assert_eq!(most, 15 * GIB);
        let plan = plan_for(&mut broker, &kv, most);
        assert_eq!(
            releases(&plan),
            [("prompt cache", 5 * GIB), ("row cache", 4 * GIB), ("expert tier", 6 * GIB)]
        );
        carry_out(&mut broker, &plan, &[&prompt, &rows, &experts]);
        assert_eq!(broker.grow(&kv, most), Ok(Growth::Granted));
        assert_eq!(
            [&kv, &prompt, &rows, &experts].map(|c| usage(&broker, c) / GIB),
            [21, 1, 2, 8],
            "every cache sits on its floor"
        );
        assert_eq!((broker.used(), broker.free()), (32 * GIB, 0));
        // The expert tier cannot win memory back from the caches below it
        // either: they are at their floors.
        assert!(matches!(broker.grow(&experts, GIB), Err(BrokerError::Exhausted { .. })));

        // When the request ends, its memory is free for the caches again.
        assert_eq!(broker.unregister(kv), Ok(21 * GIB));
        assert_eq!(broker.grow(&prompt, 21 * GIB), Ok(Growth::Granted));
    }

    struct Rng(u64);

    impl Rng {
        /// A number in `0..n`.
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// What the test believes a consumer holds, kept separately from the broker.
    struct Held {
        handle: Consumer,
        class: Class,
        usage: u64,
    }

    fn rank(info: &ConsumerInfo) -> (u32, Reverse<u64>, ConsumerId) {
        let Class::Elastic { priority, .. } = info.class else {
            panic!("{} is required", info.name)
        };
        (priority, Reverse(info.reclaimable()), info.id)
    }

    /// Checks a plan against the rules, from the books as they were when it was made.
    fn check_plan(before: &Broker, plan: &Plan, bytes: u64) {
        let asker = before.info(plan.consumer()).expect("the asker is registered");
        assert_eq!(plan.requested(), bytes);
        assert_eq!(plan.total(), bytes - before.free(), "a plan frees exactly what is missing");
        let mut taken = Vec::new();
        for release in plan.releases() {
            let source = before.info(release.consumer).expect("a source is registered");
            assert_eq!(source.name, release.name);
            let Class::Elastic { priority, floor } = source.class else {
                panic!("a required consumer is in a plan: {plan}")
            };
            assert!(release.bytes > 0, "{plan}");
            assert!(source.usage - release.bytes >= floor, "below the floor: {plan}");
            match asker.class {
                Class::Required => {}
                Class::Elastic { priority: own, .. } => {
                    assert!(priority < own, "took from an equal or higher rank: {plan}")
                }
            }
            assert!(!taken.contains(&release.consumer), "listed twice: {plan}");
            taken.push(release.consumer);
        }
        // The order is the rule's order, and a consumer is emptied to its floor
        // before the next one is asked.
        let ranks: Vec<_> =
            plan.releases().iter().map(|r| rank(before.info(r.consumer).unwrap())).collect();
        assert!(ranks.windows(2).all(|w| w[0] < w[1]), "{plan}");
        let (last, earlier) = plan.releases().split_last().expect("something to release");
        for release in earlier {
            assert_eq!(release.bytes, before.info(release.consumer).unwrap().reclaimable());
        }
        // Nobody who ranks before the last release and could give was skipped.
        let last_rank = rank(before.info(last.consumer).unwrap());
        for info in before.consumers() {
            let may = match (asker.class, info.class) {
                (_, Class::Required) => false,
                (Class::Required, _) => true,
                (Class::Elastic { priority: own, .. }, Class::Elastic { priority, .. }) => {
                    priority < own
                }
            };
            if may && info.reclaimable() > 0 && rank(info) < last_rank {
                assert!(taken.contains(&info.id), "{} was skipped: {plan}", info.name);
            }
        }
    }

    /// What the rules let `asker` take in total, worked out from the classes alone.
    fn takeable(broker: &Broker, asker: ConsumerId) -> u64 {
        let asker = broker.info(asker).unwrap().class;
        broker
            .consumers()
            .iter()
            .filter(|info| match (asker, info.class) {
                (_, Class::Required) => false,
                (Class::Required, _) => true,
                (Class::Elastic { priority: own, .. }, Class::Elastic { priority, .. }) => {
                    priority < own
                }
            })
            .map(ConsumerInfo::reclaimable)
            .sum()
    }

    fn check_books(broker: &Broker, held: &[Held]) {
        assert!(broker.used() <= broker.capacity(), "usage exceeds the capacity");
        assert_eq!(broker.used() + broker.free(), broker.capacity());
        assert_eq!(broker.used(), broker.consumers().iter().map(|c| c.usage).sum::<u64>());
        assert_eq!(broker.used(), held.iter().map(|h| h.usage).sum::<u64>());
        assert_eq!(broker.consumers().len(), held.len());
        for (info, held) in broker.consumers().iter().zip(held) {
            assert_eq!(info.id, held.handle.id());
            assert_eq!((info.usage, info.class), (held.usage, held.class), "{}", info.name);
        }
        let report = broker.to_string();
        assert_eq!(report.lines().count(), held.len() + 1);
        for info in broker.consumers() {
            assert!(report.contains(&info.name));
        }
    }

    /// Drives the broker with random registrations, grows (carrying out every plan
    /// the way a caller does), shrinks, floor changes and unregistrations, against a
    /// second set of books kept by the test.
    fn drive(seed: u64, capacity: u64, steps: usize) {
        let mut rng = Rng(seed);
        let mut broker = Broker::new("host RAM", capacity);
        let mut held: Vec<Held> = Vec::new();
        let mut serial = 0;
        let (mut plans, mut refusals) = (0, 0);
        for _ in 0..steps {
            let pick = rng.below(100);
            if pick < 12 && held.len() < 10 {
                serial += 1;
                let class = if rng.below(4) == 0 {
                    Class::Required
                } else {
                    elastic(rng.below(4) as u32, rng.below(capacity / 3 + 1))
                };
                let fits = match class {
                    Class::Elastic { floor, .. } => floor <= broker.free(),
                    Class::Required => true,
                };
                match broker.register(format!("c{serial}"), class) {
                    Ok(handle) => {
                        assert!(fits);
                        held.push(Held { handle, class, usage: 0 });
                    }
                    Err(BrokerError::FloorTooHigh { .. }) => assert!(!fits),
                    Err(other) => panic!("{other}"),
                }
            } else if held.is_empty() {
                continue;
            } else if pick < 55 {
                let i = rng.below(held.len() as u64) as usize;
                let bytes = match rng.below(10) {
                    0 => rng.below(capacity * 2),
                    _ => rng.below(capacity / 4 + 1),
                };
                let before = broker.clone();
                match broker.grow(&held[i].handle, bytes) {
                    Ok(Growth::Granted) => {
                        assert!(bytes <= before.free());
                        held[i].usage += bytes;
                    }
                    Err(BrokerError::Exhausted {
                        requested, free, reclaimable, consumers, ..
                    }) => {
                        refusals += 1;
                        let possible = takeable(&before, held[i].handle.id());
                        assert_eq!((requested, free), (Bytes(bytes), Bytes(before.free())));
                        assert_eq!(reclaimable, Bytes(possible));
                        assert!(possible < bytes - before.free(), "refused although it fits");
                        assert_eq!(consumers.as_slice(), before.consumers());
                        assert_eq!(broker.to_string(), before.to_string());
                    }
                    Err(other) => panic!("{other}"),
                    Ok(Growth::Reclaim(plan)) => {
                        plans += 1;
                        assert!(bytes > before.free());
                        check_plan(&before, &plan, bytes);
                        check_books(&broker, &held);
                        assert_eq!(broker.to_string(), before.to_string());
                        for release in plan.releases() {
                            let j = held.iter().position(|h| h.handle.id() == release.consumer);
                            let j = j.unwrap();
                            // A real cache may free more than asked, a page at a time.
                            let extra = if rng.below(4) == 0 {
                                rng.below(held[j].usage - release.bytes + 1)
                            } else {
                                0
                            };
                            let freed = release.bytes + extra;
                            broker.shrink(&held[j].handle, freed).unwrap();
                            held[j].usage -= freed;
                        }
                        assert_eq!(broker.grow(&held[i].handle, bytes), Ok(Growth::Granted));
                        held[i].usage += bytes;
                    }
                }
            } else if pick < 70 {
                let i = rng.below(held.len() as u64) as usize;
                let bytes = rng.below(held[i].usage + 3);
                let result = broker.shrink(&held[i].handle, bytes);
                if bytes <= held[i].usage {
                    result.unwrap();
                    held[i].usage -= bytes;
                } else {
                    assert!(matches!(result, Err(BrokerError::ReleasedTooMuch { .. })));
                }
            } else if pick < 82 {
                let i = rng.below(held.len() as u64) as usize;
                let floor = rng.below(capacity / 2 + 1);
                let result = broker.set_floor(&held[i].handle, floor);
                match held[i].class {
                    Class::Required => {
                        assert!(matches!(result, Err(BrokerError::NotElastic { .. })))
                    }
                    Class::Elastic { priority, floor: current } => {
                        if floor <= current || floor <= held[i].usage + broker.free() {
                            result.unwrap();
                            held[i].class = elastic(priority, floor);
                        } else {
                            assert!(matches!(result, Err(BrokerError::FloorTooHigh { .. })));
                        }
                    }
                }
            } else if pick < 90 {
                let i = rng.below(held.len() as u64) as usize;
                let probe = &held[i].handle;
                let most = broker.obtainable(probe).unwrap();
                assert_eq!(most, broker.free() + takeable(&broker, probe.id()));
                assert!(broker.clone().grow(probe, most).is_ok());
                if most < u64::MAX {
                    refusals += 1;
                    assert!(matches!(
                        broker.clone().grow(probe, most + 1),
                        Err(BrokerError::Exhausted { .. })
                    ));
                }
            } else {
                let removed = held.remove(rng.below(held.len() as u64) as usize);
                let id = removed.handle.id();
                assert_eq!(broker.unregister(removed.handle), Ok(removed.usage));
                assert!(broker.info(id).is_none());
            }
            check_books(&broker, &held);
        }
        // The run must have exercised the interesting paths.
        assert!(plans > 0 && refusals > 0, "seed {seed}: {plans} plans, {refusals} refusals");
    }

    #[test]
    fn random_use_keeps_the_rules_and_the_books() {
        for seed in 1..=60 {
            // A small capacity makes exact fits and ties common, a large one
            // makes long plans.
            drive(0x9e37_79b9_7f4a_7c15 ^ seed, 40, 1500);
            drive(0x2545_f491_4f6c_dd1d ^ seed, 1000, 1500);
        }
    }
}
