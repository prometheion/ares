//! The level of the prefix cache in RAM: what it holds and in which order it
//! gives things up.
//!
//! A page that leaves the devices is copied into host memory here (a buffer
//! per device share, page-locked and booked in the prompt cache's part of the
//! engine's RAM, [`PinnedRam::cache_buffer`]), under the name the index gave
//! it, with the checkpoint of recurrent state that hung on it. A prompt that
//! finds its start here has the pages copied back to the devices; the copy
//! in RAM stays, so a page that leaves the devices again costs no copy.
//! Checkpoints lie in RAM anyway (their places), so for a checkpoint leaving
//! the devices means only that it hangs on a page in RAM now and not on one
//! in the index.
//!
//! This module keeps the books of that and says what goes first; the cache
//! (`cache.rs`) moves the bytes and decides when. What goes first when room
//! is needed, in [`Class`] order and within a class the least recently used:
//!
//! 1. pages that the devices hold as well: they go for nothing;
//! 2. what the drive holds already, and what is not worth writing (a text
//!    shorter than the level on SSD's least, a tail, a checkpoint that later
//!    requests have left behind at deeper page boundaries), or anything when
//!    there is no level on SSD:
//!    it goes for nothing;
//! 3. the rest, which is written to the drive before it goes.
//!
//! The devices give pages up from the end of a text, and each page is the
//! most recently used of what left them when it arrives here
//! ([`RamLevel::left_the_cards`]), so a text gives up its end here before its
//! start too.

use std::collections::{BTreeSet, HashMap};

use ares_state::ChainHash;

use crate::exec::{HostBytes, TransitPart};
use crate::ram::PinnedRam;

/// How soon something the level holds goes when room is needed: the lower
/// first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Class {
    /// A page the devices hold as well.
    Copy,
    /// What goes without a write.
    Free,
    /// What is written to the drive before it goes.
    Dirty,
}

/// What an entry of the order stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Item {
    Page,
    Checkpoint,
}

/// A checkpoint of recurrent state that hangs on a page or a tail in RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Held {
    /// Its place.
    pub place: u32,
    /// Written to the drive when it goes, unless the drive has it: it lies at
    /// a page boundary, deep enough, and no later request has left it behind.
    pub worth: bool,
    /// The drive has it.
    pub stored: bool,
    /// It lies at a root: the end of a system prompt or a tool list.
    pub root: bool,
}

impl Held {
    fn class(&self, writes: bool) -> Class {
        if writes && self.worth && !self.stored { Class::Dirty } else { Class::Free }
    }
}

/// A page or a tail in RAM.
#[derive(Debug)]
pub(super) struct Entry {
    /// The hash of the page before it (or the root); for a tail, the page it
    /// hangs under.
    pub parent: ChainHash,
    /// Its number in its chain (1 for a first page); for a tail, the number
    /// of the page it hangs under.
    pub depth: u32,
    /// The tokens of a tail; `None` for a full page.
    pub tail: Option<Vec<u64>>,
    /// The bytes, a part per device; `None` while they are in a transfer or
    /// with the writer.
    pub parts: Option<Vec<TransitPart>>,
    pub checkpoint: Option<Held>,
    /// The devices hold the page too.
    pub on_cards: bool,
    /// The drive has the page.
    pub stored: bool,
    /// The page's text was long enough to be written.
    pub worth: bool,
    /// When it was last used, and where it stands in the order now.
    stamp: u64,
    keys: [Option<Key>; 2],
}

type Key = (Class, u64, Item, ChainHash);

impl Entry {
    /// A page or tail with nothing decided yet about its place in the order.
    pub fn new(parent: ChainHash, depth: u32, tail: Option<Vec<u64>>) -> Self {
        Self {
            parent,
            depth,
            tail,
            parts: None,
            checkpoint: None,
            on_cards: false,
            stored: false,
            worth: false,
            stamp: 0,
            keys: [None, None],
        }
    }

    fn page_class(&self, writes: bool) -> Class {
        if self.on_cards {
            Class::Copy
        } else if writes && self.worth && !self.stored && self.tail.is_none() {
            Class::Dirty
        } else {
            Class::Free
        }
    }
}

/// What the level has done since the engine started, and what it holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RamCounters {
    /// The level is in use: the model moves its pages to the host.
    pub enabled: bool,
    /// Pages (and tails) it holds now, and the bytes of their buffers, the
    /// buffers that hold nothing at the moment included.
    pub pages: u64,
    pub bytes: u64,
    /// Checkpoints that hang on its pages now.
    pub checkpoints: u64,
    /// Pages that left the devices and were copied here.
    pub pages_in: u64,
    /// Pages that a prompt took back to the devices from here.
    pub pages_read: u64,
    /// Pages given up without being written: they left the devices when
    /// there was no room here, or went from here unwritten although they
    /// were worth writing. Copies of pages the devices hold, pages the drive
    /// has, and pages of texts too short for the drive are not counted.
    pub pages_dropped: u64,
    /// Transfers between the devices and the host for the levels below the
    /// devices: calls, and the nanoseconds the conductor waited for them.
    pub transfers: u64,
    pub transfer_ns: u64,
}

/// The pages and tails in RAM. See the module's documentation.
pub(super) struct RamLevel {
    entries: HashMap<ChainHash, Entry>,
    /// The tails under each page (or the root).
    tails: HashMap<ChainHash, Vec<ChainHash>>,
    /// Everything that can go, in the order it goes, and the checkpoints
    /// among it.
    order: BTreeSet<Key>,
    checkpoint_order: BTreeSet<Key>,
    /// Pages and checkpoints of the order that are written before they go.
    dirty_pages: u64,
    dirty_checkpoints: u64,
    /// Buffers that hold nothing now.
    free: Vec<Vec<TransitPart>>,
    /// Buffers the level has made and not dropped: in entries, free, in a
    /// transfer or with the writer.
    made: u64,
    /// Entries whose bytes are away: in a transfer or with the writer.
    away: u64,
    /// The bytes of a page's share on each device, as the books count them.
    part_bytes: Vec<u64>,
    books: PinnedRam,
    /// There is a level on SSD: what is worth writing is written before it
    /// goes.
    writes: bool,
    clock: u64,
    checkpoints: u64,
    pub counters: RamCounters,
}

impl std::fmt::Debug for RamLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RamLevel")
            .field("entries", &self.entries.len())
            .field("free", &self.free.len())
            .field("made", &self.made)
            .finish_non_exhaustive()
    }
}

impl RamLevel {
    /// A level for pages of `page_bytes` per device at most, whose buffers
    /// are booked in `books`; `writes` when a level on SSD lies below it.
    pub fn new(page_bytes: &[u64], books: PinnedRam, writes: bool) -> Self {
        Self {
            entries: HashMap::new(),
            tails: HashMap::new(),
            order: BTreeSet::new(),
            checkpoint_order: BTreeSet::new(),
            dirty_pages: 0,
            dirty_checkpoints: 0,
            free: Vec::new(),
            made: 0,
            away: 0,
            part_bytes: page_bytes.iter().map(|b| ares_store::padded(*b as usize) as u64).collect(),
            books,
            writes,
            clock: 0,
            checkpoints: 0,
            counters: RamCounters { enabled: true, ..RamCounters::default() },
        }
    }

    /// The bytes of one page's buffers on all devices.
    pub fn page_bytes(&self) -> u64 {
        self.part_bytes.iter().sum()
    }

    /// The bytes of all buffers the level holds: in entries, free, and on
    /// their way somewhere.
    pub fn bytes(&self) -> u64 {
        self.made * self.page_bytes()
    }

    /// The bytes of the buffers that hold nothing.
    pub fn free_bytes(&self) -> u64 {
        self.free.len() as u64 * self.page_bytes()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The checkpoints that hang on entries.
    pub fn checkpoints(&self) -> u64 {
        self.checkpoints
    }

    /// The bytes of the entries whose bytes are away: with the writer,
    /// between steps.
    pub fn away_bytes(&self) -> u64 {
        self.away * self.page_bytes()
    }

    pub fn get(&self, hash: ChainHash) -> Option<&Entry> {
        self.entries.get(&hash)
    }

    /// The entry `hash` has its bytes here, not in a transfer or with the
    /// writer.
    pub fn ready(&self, hash: ChainHash) -> bool {
        self.entries.get(&hash).is_some_and(|entry| entry.parts.is_some())
    }

    /// The tails under `prior` whose bytes are here.
    pub fn tails_under(&self, prior: ChainHash) -> impl Iterator<Item = (ChainHash, &Entry)> {
        self.tails.get(&prior).into_iter().flatten().filter_map(|hash| {
            let entry = self.entries.get(hash)?;
            entry.parts.as_ref().map(|_| (*hash, entry))
        })
    }

    /// Every entry, the first to go first: for writing everything out.
    pub fn in_order(&self) -> Vec<ChainHash> {
        let mut hashes: Vec<(u64, ChainHash)> =
            self.entries.iter().map(|(hash, entry)| (entry.stamp, *hash)).collect();
        hashes.sort_unstable();
        hashes.into_iter().map(|(_, hash)| hash).collect()
    }

    /// The first entry of the order: what goes next, and its class.
    pub fn first(&self) -> Option<(Class, Item, ChainHash)> {
        self.order.first().map(|(class, _, item, hash)| (*class, *item, *hash))
    }

    /// The checkpoint that goes first.
    pub fn first_checkpoint(&self) -> Option<ChainHash> {
        self.checkpoint_order.first().map(|key| key.3)
    }

    /// What goes next of `class` and above, in the order.
    pub fn first_from(&self, class: Class) -> Option<(Class, Item, ChainHash)> {
        let from = (class, 0, Item::Page, ChainHash::from_bytes([0; 16]));
        self.order.range(from..).next().map(|(class, _, item, hash)| (*class, *item, *hash))
    }

    /// The bytes of what would be written before it goes: dirty pages and
    /// checkpoints whose bytes are here.
    pub fn dirty_bytes(&self, checkpoint_bytes: u64) -> u64 {
        self.dirty_pages * self.page_bytes() + self.dirty_checkpoints * checkpoint_bytes
    }

    fn count(&mut self, key: &Key, by: i64) {
        if key.0 == Class::Dirty {
            let count = match key.2 {
                Item::Page => &mut self.dirty_pages,
                Item::Checkpoint => &mut self.dirty_checkpoints,
            };
            *count = count.checked_add_signed(by).expect("counted when ordered");
        }
    }

    fn unorder(&mut self, hash: ChainHash) {
        let Some(entry) = self.entries.get_mut(&hash) else { return };
        let keys: Vec<Key> = entry.keys.iter_mut().filter_map(Option::take).collect();
        for key in keys {
            self.order.remove(&key);
            self.checkpoint_order.remove(&key);
            self.count(&key, -1);
        }
    }

    /// Puts the entry where its class and its stamp say. An entry whose
    /// bytes are away is not in the order, and neither is its checkpoint.
    fn reorder(&mut self, hash: ChainHash) {
        self.unorder(hash);
        let writes = self.writes;
        let Some(entry) = self.entries.get_mut(&hash) else { return };
        if entry.parts.is_none() {
            return;
        }
        let page = (entry.page_class(writes), entry.stamp, Item::Page, hash);
        entry.keys[0] = Some(page);
        let checkpoint =
            entry.checkpoint.map(|held| (held.class(writes), entry.stamp, Item::Checkpoint, hash));
        entry.keys[1] = checkpoint;
        self.order.insert(page);
        self.count(&page, 1);
        if let Some(key) = checkpoint {
            self.order.insert(key);
            self.checkpoint_order.insert(key);
            self.count(&key, 1);
        }
    }

    /// Takes in a page or tail whose bytes are in `parts`. It is the least
    /// recently used of all until it is touched: pages leave the devices
    /// from the end of a text, and one text's later pages go first.
    pub fn insert(&mut self, hash: ChainHash, mut entry: Entry, parts: Vec<TransitPart>) {
        if self.entries.contains_key(&hash) {
            // The level has it already: the new copy is not needed.
            self.recycle(parts);
            return;
        }
        self.clock += 1;
        entry.stamp = self.clock;
        entry.parts = Some(parts);
        entry.keys = [None, None];
        self.checkpoints += u64::from(entry.checkpoint.is_some());
        if entry.tail.is_some() {
            self.tails.entry(entry.parent).or_default().push(hash);
        }
        self.entries.insert(hash, entry);
        self.counters.pages += 1;
        self.reorder(hash);
    }

    /// Changes what is known of entry `hash` and puts it where that says.
    pub fn update(&mut self, hash: ChainHash, change: impl FnOnce(&mut Entry)) {
        let Some(entry) = self.entries.get_mut(&hash) else { return };
        let had = entry.checkpoint.is_some();
        change(entry);
        let has = entry.checkpoint.is_some();
        self.checkpoints = self.checkpoints + u64::from(has) - u64::from(had);
        self.reorder(hash);
    }

    /// The page leaves the devices: it is the next to be used again of all
    /// that left (`stamp` anew), and keeps what is known of it.
    pub fn left_the_cards(&mut self, hash: ChainHash, change: impl FnOnce(&mut Entry)) {
        self.clock += 1;
        let stamp = self.clock;
        self.update(hash, |entry| {
            entry.on_cards = false;
            entry.stamp = stamp;
            change(entry);
        });
    }

    /// Takes the bytes of `hash` out for a transfer or a write; the entry
    /// stays and is out of the order until they come back.
    pub fn take(&mut self, hash: ChainHash) -> Option<Vec<TransitPart>> {
        self.unorder(hash);
        let parts = self.entries.get_mut(&hash)?.parts.take()?;
        self.away += 1;
        Some(parts)
    }

    /// The bytes of `hash` come back.
    pub fn give(&mut self, hash: ChainHash, parts: Vec<TransitPart>) {
        match self.entries.get_mut(&hash) {
            Some(entry) if entry.parts.is_none() => {
                entry.parts = Some(parts);
                self.away -= 1;
                self.reorder(hash);
            }
            _ => self.recycle(parts),
        }
    }

    /// Takes the entry `hash` out of the level and returns it, its bytes
    /// with it when they are here. Its checkpoint is the caller's.
    pub fn remove(&mut self, hash: ChainHash) -> Option<Entry> {
        self.unorder(hash);
        let entry = self.entries.remove(&hash)?;
        self.checkpoints -= u64::from(entry.checkpoint.is_some());
        self.away -= u64::from(entry.parts.is_none());
        self.counters.pages -= 1;
        if entry.tail.is_some()
            && let Some(tails) = self.tails.get_mut(&entry.parent)
        {
            tails.retain(|tail| *tail != hash);
            if tails.is_empty() {
                self.tails.remove(&entry.parent);
            }
        }
        Some(entry)
    }

    /// A buffer that holds nothing, if there is one.
    pub fn spare(&mut self) -> Option<Vec<TransitPart>> {
        self.free.pop()
    }

    /// A new buffer, booked. `Err` says why there is none.
    pub fn make(&mut self) -> Result<Vec<TransitPart>, String> {
        let mut parts = Vec::with_capacity(self.part_bytes.len());
        for (part, bytes) in self.part_bytes.iter().enumerate() {
            let buffer = self.books.cache_buffer(part, *bytes as usize)?;
            parts.push(TransitPart { mem: HostBytes::Cache(buffer), len: 0, tag: 0 });
        }
        self.made += 1;
        Ok(parts)
    }

    /// A buffer is not needed now: it is kept for the next page.
    pub fn recycle(&mut self, parts: Vec<TransitPart>) {
        // Memory in transit that is not the level's own is not kept.
        if let Some(HostBytes::Cache(_)) = parts.first().map(|part| &part.mem) {
            self.free.push(parts);
        }
    }

    /// A buffer goes back to the system.
    pub fn discard(&mut self, parts: Vec<TransitPart>) {
        if matches!(parts.first().map(|part| &part.mem), Some(HostBytes::Cache(_))) {
            self.made -= 1;
        }
    }

    /// Gives the buffers that hold nothing back to the system, down to
    /// `keep` of them. Returns the bytes given back.
    pub fn shed(&mut self, keep: usize) -> u64 {
        let mut given = 0;
        while self.free.len() > keep {
            let parts = self.free.pop().expect("more than `keep` are free");
            self.discard(parts);
            given += self.page_bytes();
        }
        given
    }

    /// What the level holds now, for the counters.
    pub fn snapshot(&self) -> RamCounters {
        RamCounters { bytes: self.bytes(), checkpoints: self.checkpoints, ..self.counters }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(i: u8) -> ChainHash {
        ChainHash::root(&[i])
    }

    fn level(writes: bool) -> RamLevel {
        RamLevel::new(&[100, 8], PinnedRam::new(1 << 20), writes)
    }

    fn page(level: &mut RamLevel, i: u8, change: impl FnOnce(&mut Entry)) {
        let mut entry = Entry::new(hash(0), u32::from(i), None);
        change(&mut entry);
        let parts = level.make().unwrap();
        level.insert(hash(i), entry, parts);
    }

    #[test]
    fn copies_go_before_what_the_drive_has_and_that_before_what_it_has_not() {
        let mut level = level(true);
        page(&mut level, 1, |e| e.worth = true);
        page(&mut level, 2, |e| e.on_cards = true);
        page(&mut level, 3, |e| {
            e.worth = true;
            e.stored = true;
        });
        page(&mut level, 4, |_| {});
        let order: Vec<_> = std::iter::from_fn(|| {
            let (class, _, hash) = level.first()?;
            level.remove(hash);
            Some((class, hash))
        })
        .collect();
        assert_eq!(
            order,
            [
                (Class::Copy, hash(2)),
                (Class::Free, hash(3)),
                (Class::Free, hash(4)),
                (Class::Dirty, hash(1))
            ]
        );
        // Without a level on SSD nothing is dirty.
        let mut level = super::tests::level(false);
        page(&mut level, 1, |e| e.worth = true);
        assert_eq!(level.first().map(|(class, ..)| class), Some(Class::Free));
    }

    #[test]
    fn a_text_gives_up_its_end_first_and_a_text_that_left_the_devices_later_goes_later() {
        let mut level = level(false);
        // The devices give pages up from the end of a text.
        for i in [6, 5, 4] {
            page(&mut level, i, |e| e.on_cards = true);
        }
        for i in [3, 2, 1] {
            page(&mut level, i, |_| {});
        }
        assert_eq!(level.first().map(|(.., hash)| hash), Some(hash(6)), "copies first");
        // The first text was used again and leaves the devices now.
        for i in [6, 5, 4] {
            level.left_the_cards(hash(i), |_| {});
        }
        let order: Vec<ChainHash> = std::iter::from_fn(|| {
            let (.., hash) = level.first()?;
            level.remove(hash);
            Some(hash)
        })
        .collect();
        assert_eq!(order, [3, 2, 1, 6, 5, 4].map(hash));
    }

    #[test]
    fn a_checkpoint_goes_on_its_own_and_bytes_that_are_away_hold_their_place() {
        let mut level = level(true);
        page(&mut level, 1, |e| {
            e.stored = true;
            e.checkpoint = Some(Held { place: 7, worth: true, stored: false, root: false });
        });
        assert_eq!(level.checkpoints(), 1);
        // The page goes for nothing; its checkpoint must be written.
        assert_eq!(level.first(), Some((Class::Free, Item::Page, hash(1))));
        assert_eq!(level.first_from(Class::Dirty), Some((Class::Dirty, Item::Checkpoint, hash(1))));
        assert_eq!(level.dirty_bytes(1000), 1000);
        // While the bytes are away neither is in the order.
        let parts = level.take(hash(1)).unwrap();
        assert_eq!((level.first(), level.ready(hash(1))), (None, false));
        level.give(hash(1), parts);
        level.update(hash(1), |e| e.checkpoint = None);
        assert_eq!((level.checkpoints(), level.dirty_bytes(1000)), (0, 0));
    }

    #[test]
    fn buffers_are_booked_kept_for_the_next_page_and_given_back() {
        let books = PinnedRam::new(1 << 20);
        let mut level = RamLevel::new(&[100, 8], books.clone(), false);
        let page = 2 * ares_store::BLOCK as u64;
        assert_eq!(level.page_bytes(), page);
        let parts = level.make().unwrap();
        assert_eq!((books.cache_pages(), level.bytes()), (page, page));
        level.recycle(parts);
        assert_eq!(level.free_bytes(), page);
        assert!(level.spare().is_some_and(|parts| {
            level.recycle(parts);
            true
        }));
        assert_eq!(level.shed(0), page);
        assert_eq!((books.cache_pages(), level.bytes()), (0, 0));
        // The limit of the books holds.
        let tight = RamLevel::new(&[1 << 20], PinnedRam::new(1 << 19), false).make();
        assert!(tight.is_err());
    }
}
