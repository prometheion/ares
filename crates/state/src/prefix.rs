//! Finding cached state for the start of a prompt.
//!
//! Every full page of a sequence can be named by a hash of everything that
//! determines its content: the context it was computed in and the names of
//! all cells up to its end. A cell's name is 64 bits: a text cell's is its
//! token, an image cell's has the top bit set and the rest from what the
//! image is and which of its rows the cell holds (the runtime makes them).
//! The "tokens" of this module are these names. The hashes form a chain (each page's hash extends the previous
//! one), so the longest cached prefix of a new prompt is found by walking its
//! pages from the start until the index has no entry.
//!
//! The index holds one reference to each page it knows, so cached pages
//! outlive the sequence that computed them and any number of sequences can
//! start from them.
//!
//! **A page enters the index when it fills**, not when its sequence ends
//! ([`PrefixIndex::publish`], called from the commit): two requests that
//! start with the same prompt at the same time share from the first full
//! page, as in vLLM and SGLang. A sequence that finds its page cached already
//! gives up its own copy.
//!
//! **A prompt waits for a page that is being computed.** A sequence
//! announces the pages its prompt will fill ([`PrefixIndex::announce`]). A
//! second prompt whose first missing page is announced is told so
//! ([`PrefixMatch::computing`]) and is held back until the page is published
//! or the announcement is withdrawn, instead of computing the same rows next
//! to the first (LMDeploy's producer mark).
//!
//! **Tails** let a prompt resume at the real end of the cached text instead
//! of at the last page boundary (TokenSpeed's tail table). A tail is the
//! partly used last page of a sequence, published under the hash of the page
//! before it together with its tokens; up to [`MAX_TAILS`] are kept per page,
//! least recently used first. A prompt that continues a tail takes the tail's
//! page and copies it before its first write, which the page table does
//! anyway for a shared partly used page. See [`PrefixIndex::insert_tail`] for
//! what the cells of a tail must be.
//!
//! **Eviction.** Pages nobody else holds are given up from the ends of the
//! chains, so a chain shrinks from its tail and stays usable. Among the ends,
//! a lower [`Priority`] goes before a higher one whatever their age
//! (TensorRT-LLM), and within a class the least recently used goes first.
//!
//! State that is not addressable by position (recurrent layers) cannot be
//! resumed from pages alone. For models that have it, a checkpoint can be
//! attached to an entry or to a tail; where checkpoints go and which are
//! dropped is the [`CheckpointPolicy`](crate::CheckpointPolicy).

use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

use crate::pool::{PageId, PagePool};
use crate::sequence::Sequence;

/// The name of a page's content: a 128-bit hash of the context and of every
/// token up to the end of the page.
///
/// The hash is not cryptographic. Two different prefixes that collide would
/// share state, so the width is chosen to make an accidental collision
/// implausible; a server that takes prompts from parties that must not see
/// each other's state needs a keyed hash here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChainHash(u128);

/// 64x64 -> 128-bit multiply, folded.
fn fold(a: u64, b: u64) -> u64 {
    let product = u128::from(a) * u128::from(b);
    (product as u64) ^ ((product >> 64) as u64)
}

const K: [u64; 6] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
    0x2545_f491_4f6c_dd1d,
    0xd6e8_feb8_6659_fd93,
    0xa076_1d64_78bd_642f,
];

impl ChainHash {
    /// The start of a chain. `context` must cover everything besides the
    /// tokens that changes the state: the model, its quantization, the state
    /// formats, and anything injected into the prompt that is not a token.
    pub fn root(context: &[u8]) -> Self {
        let (mut a, mut b) = (K[0], K[1]);
        for chunk in context.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            let word = u64::from_le_bytes(word);
            a = fold(a ^ word, K[2]);
            b = fold(b ^ word.rotate_left(29), K[3]);
        }
        let len = context.len() as u64;
        Self(u128::from(fold(a ^ len, K[4])) | u128::from(fold(b ^ len, K[5])) << 64)
    }

    /// The hash as 16 bytes, the form in which it names a page on disk.
    pub fn to_bytes(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }

    /// The hash from its 16 bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(u128::from_le_bytes(bytes))
    }

    /// The hash of the page that follows `self` and holds `tokens`.
    pub fn extend(self, tokens: &[u64]) -> Self {
        let (mut a, mut b) = (self.0 as u64 ^ K[0], (self.0 >> 64) as u64 ^ K[1]);
        for &token in tokens {
            a = fold(a ^ token ^ K[4], K[2]);
            b = fold(b ^ token.rotate_left(32) ^ K[5], K[3]);
        }
        let len = tokens.len() as u64;
        Self(u128::from(fold(a ^ len, K[4])) | u128::from(fold(b ^ len, K[5])) << 64)
    }
}

/// An opaque handle to a checkpoint of non-addressable state, owned by
/// whoever stores checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub u64);

/// Who computes a page: an opaque number of the caller's, one per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Owner(pub u64);

/// How much a cached page is worth keeping. When the pool runs short, every
/// chain end of a lower class is given up before any of a higher class; the
/// age of a page decides only within its class.
///
/// The class is also what a residency layer can read off an evicted page
/// ([`Evicted::class`]) to decide whether it is worth moving to a slower tier
/// or is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    /// The class of pages nobody asked to keep or to drop.
    pub const NORMAL: Priority = Priority(128);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// The most tails kept under one page (TokenSpeed keeps 8 as well). A tail
/// holds a page of the pool for less than a page of cells, so the number is
/// small; the least recently used one makes room.
pub const MAX_TAILS: usize = 8;

/// What the index knows about the page named by a hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    /// It is cached.
    Cached,
    /// A sequence announced that it is computing it.
    Computing(Owner),
    /// Neither.
    Unknown,
}

pub(crate) struct Entry {
    pub(crate) page: PageId,
    pub(crate) parent: ChainHash,
    /// Cached pages and tails that follow this page.
    children: u32,
    /// The number of pages from the start of the chain to the end of this
    /// one. A page inserted under a hash the index does not know counts as
    /// the first.
    pub(crate) depth: u32,
    pub(crate) last_used: u64,
    class: Priority,
    pub(crate) checkpoint: Option<CheckpointId>,
    /// Who published the checkpoint, when that is known.
    pub(crate) checkpoint_by: Option<Owner>,
    /// The checkpoint is at a root: the end of a system prompt or a tool list.
    pub(crate) checkpoint_root: bool,
    /// Scratch for walks: the epoch of the walk that last marked the entry.
    mark: u64,
}

pub(crate) struct Tail {
    pub(crate) hash: ChainHash,
    pub(crate) tokens: Vec<u64>,
    page: PageId,
    pub(crate) last_used: u64,
    class: Priority,
    pub(crate) checkpoint: Option<CheckpointId>,
    pub(crate) checkpoint_by: Option<Owner>,
    /// Cached pages under the same page whose tokens begin with this tail's:
    /// the text went on past the tail.
    pub(crate) extended_by: Vec<ChainHash>,
}

/// The part of a tail a prompt can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailMatch {
    /// The tail's page. The prompt takes it as its last page.
    pub page: PageId,
    /// How many of its cells the prompt shares, from the start of the page.
    pub cells: u32,
    /// The tail's own name, for [`PrefixIndex::set_checkpoint`].
    pub hash: ChainHash,
    /// The checkpoint at the end of the tail, when the prompt shares all of
    /// the tail's cells: the state is the one after `cells` cells then.
    pub checkpoint: Option<CheckpointId>,
}

/// The cached start of a prompt: one entry per matched page, in order, and
/// what follows the last matched page.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PrefixMatch {
    /// The cached pages, the first page of the prompt first.
    pub pages: Vec<PageId>,
    /// The name of each of them.
    pub hashes: Vec<ChainHash>,
    /// The checkpoint at the end of each of them, where there is one.
    pub checkpoints: Vec<Option<CheckpointId>>,
    /// A tail under the last matched page that the prompt continues.
    pub tail: Option<TailMatch>,
    /// The prompt's first page that is not cached is being computed by
    /// another sequence: its hash and who computes it. The prompt should
    /// wait for it ([`PrefixIndex::state`]) and look again.
    pub computing: Option<(ChainHash, Owner)>,
}

impl PrefixMatch {
    /// The number of matched pages that can be resumed from when the model
    /// needs a checkpoint to resume, and that checkpoint.
    pub fn last_checkpoint(&self) -> Option<(usize, CheckpointId)> {
        self.checkpoints.iter().enumerate().rev().find_map(|(i, c)| Some((i + 1, (*c)?)))
    }

    /// The number of cells the match covers: the pages and the tail.
    pub fn cells(&self, cells_per_page: u32) -> u64 {
        self.pages.len() as u64 * u64::from(cells_per_page)
            + self.tail.map_or(0, |tail| u64::from(tail.cells))
    }

    /// The page table of a sequence that starts from this match, for
    /// [`Sequence::from_shared`] with [`PrefixMatch::cells`] cells.
    pub fn table(&self) -> Vec<PageId> {
        self.pages.iter().copied().chain(self.tail.map(|tail| tail.page)).collect()
    }
}

/// What the index gave up, with its name, so that a slower level can keep it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evicted {
    /// The page, which is free again unless a sequence still holds it.
    pub page: PageId,
    /// The checkpoint that was attached to the entry and is now unreferenced.
    pub checkpoint: Option<CheckpointId>,
    /// That checkpoint was at a root ([`PrefixIndex::mark_root`]).
    pub root: bool,
    /// The class the page had.
    pub class: Priority,
    /// The name of what went: a page's hash, or a tail's.
    pub hash: ChainHash,
    /// What it follows: the hash of the page before it (or the root). For a
    /// tail, the page it hangs under.
    pub parent: ChainHash,
    /// A page's number in its chain (1 for a first page). For a tail, the
    /// number of the page it hangs under (0 under the root).
    pub depth: u32,
    /// A tail's tokens; `None` for a full page.
    pub tail: Option<Vec<u64>>,
}

/// A cached page as [`PrefixIndex::cached_pages`] lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachedPage {
    pub hash: ChainHash,
    pub page: PageId,
    /// The hash of the page before it, or the root.
    pub parent: ChainHash,
    /// Its number in its chain (1 for a first page).
    pub depth: u32,
    pub checkpoint: Option<CheckpointId>,
}

/// The names of a sequence's full pages, kept next to the sequence so that
/// pages can be published as they fill without hashing the text again.
#[derive(Clone, Debug)]
pub struct Chain {
    root: ChainHash,
    cells_per_page: u32,
    hashes: Vec<ChainHash>,
    /// How many of the named pages are in the index as this sequence's page.
    published: usize,
}

impl Chain {
    /// The chain of a sequence that starts from nothing in the context
    /// `root`.
    pub fn new(root: ChainHash, cells_per_page: u32) -> Self {
        assert!(cells_per_page > 0, "a page holds at least one cell");
        Self { root, cells_per_page, hashes: Vec::new(), published: 0 }
    }

    /// The chain of a sequence that starts from the pages of `found`.
    pub fn resumed(root: ChainHash, cells_per_page: u32, found: &PrefixMatch) -> Self {
        let mut chain = Self::new(root, cells_per_page);
        chain.hashes.extend_from_slice(&found.hashes);
        chain.published = found.hashes.len();
        chain
    }

    /// Names every full page of `tokens` that has no name yet. `tokens` is
    /// the sequence's text from its start and must begin with the text named
    /// so far. Naming the whole prompt ahead is what [`PrefixIndex::announce`]
    /// needs.
    pub fn name(&mut self, tokens: &[u64]) {
        let per_page = self.cells_per_page as usize;
        let mut hash = self.hashes.last().copied().unwrap_or(self.root);
        for page in tokens.chunks_exact(per_page).skip(self.hashes.len()) {
            hash = hash.extend(page);
            self.hashes.push(hash);
        }
    }

    /// The names of the full pages, in order.
    pub fn hashes(&self) -> &[ChainHash] {
        &self.hashes
    }

    /// The names of the pages that are named and not published: what the
    /// sequence will compute.
    pub fn unpublished(&self) -> &[ChainHash] {
        &self.hashes[self.published..]
    }

    /// The hash that page `page` extends: the previous page's, or the root.
    pub fn before(&self, page: usize) -> ChainHash {
        if page == 0 { self.root } else { self.hashes[page - 1] }
    }

    /// Forgets the names past the first `pages` pages, after the sequence was
    /// cut back or its text changed from there.
    pub fn truncate(&mut self, pages: usize) {
        self.hashes.truncate(pages);
        self.published = self.published.min(pages);
    }
}

/// The index of cached pages by content.
#[derive(Default)]
pub struct PrefixIndex {
    pub(crate) entries: HashMap<ChainHash, Entry>,
    /// The tails under each page (or root), in no order.
    pub(crate) tails: HashMap<ChainHash, Vec<Tail>>,
    /// For every tail, the hash it hangs under.
    pub(crate) tail_prior: HashMap<ChainHash, ChainHash>,
    /// Entries without children and all tails, by class and then oldest use
    /// first. Only these can be given up: an entry in the middle of a chain
    /// is what makes its successors reachable.
    leaves: BTreeSet<(Priority, u64, ChainHash)>,
    /// Pages that are being computed.
    computing: HashMap<ChainHash, Owner>,
    /// How many entries and tails carry a checkpoint.
    pub(crate) checkpoints: usize,
    pub(crate) clock: u64,
    epoch: u64,
    /// The most tails the index keeps in all.
    tail_limit: usize,
}

impl std::fmt::Debug for PrefixIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixIndex")
            .field("entries", &self.entries.len())
            .field("tails", &self.tail_prior.len())
            .field("leaves", &self.leaves.len())
            .field("computing", &self.computing.len())
            .finish()
    }
}

impl PrefixIndex {
    /// An index that knows nothing.
    pub fn new() -> Self {
        Self { tail_limit: usize::MAX, ..Self::default() }
    }

    /// Limits the tails the index keeps in all (apart from the [`MAX_TAILS`]
    /// under one page) to `limit`: the least recently used goes when a new one
    /// would be the `limit + 1`-th. A device that keeps something per partly
    /// used page, such as exact K/V entries, sizes it for this.
    pub fn with_tail_limit(mut self, limit: usize) -> Self {
        self.tail_limit = limit;
        self
    }

    /// The number of pages the index holds: cached pages and tails.
    pub fn len(&self) -> usize {
        self.entries.len() + self.tail_prior.len()
    }

    /// Every cached page (the tails not), the least recently used first.
    pub fn cached_pages(&self) -> Vec<CachedPage> {
        let mut pages: Vec<(u64, CachedPage)> = self
            .entries
            .iter()
            .map(|(hash, entry)| {
                let page = CachedPage {
                    hash: *hash,
                    page: entry.page,
                    parent: entry.parent,
                    depth: entry.depth,
                    checkpoint: entry.checkpoint,
                };
                (entry.last_used, page)
            })
            .collect();
        pages.sort_unstable_by_key(|(used, page)| (*used, page.hash));
        pages.into_iter().map(|(_, page)| page).collect()
    }

    /// Whether the index holds a page or a tail of this name.
    pub fn holds(&self, hash: ChainHash) -> bool {
        self.entries.contains_key(&hash) || self.tail_prior.contains_key(&hash)
    }

    /// Whether the index holds no page at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.tail_prior.is_empty()
    }

    /// The number of tails.
    pub fn tails(&self) -> usize {
        self.tail_prior.len()
    }

    fn touch(&mut self, hash: ChainHash) {
        self.clock += 1;
        let entry = self.entries.get_mut(&hash).expect("touched entries exist");
        if entry.children == 0 {
            self.leaves.remove(&(entry.class, entry.last_used, hash));
            self.leaves.insert((entry.class, self.clock, hash));
        }
        entry.last_used = self.clock;
    }

    fn tail_mut(&mut self, hash: ChainHash) -> Option<&mut Tail> {
        let prior = self.tail_prior.get(&hash)?;
        self.tails.get_mut(prior)?.iter_mut().find(|tail| tail.hash == hash)
    }

    fn touch_tail(&mut self, hash: ChainHash) {
        self.clock += 1;
        let clock = self.clock;
        let tail = self.tail_mut(hash).expect("touched tails exist");
        let old = (tail.class, tail.last_used, hash);
        tail.last_used = clock;
        let new = (tail.class, clock, hash);
        self.leaves.remove(&old);
        self.leaves.insert(new);
    }

    /// The longest cached prefix of `tokens`, starting from `root`: whole
    /// pages, then as much of a tail as the tokens share. The matched entries
    /// count as used now.
    ///
    /// The caller takes what it resumes from with
    /// [`Sequence::from_shared`](crate::Sequence::from_shared)
    /// ([`PrefixMatch::table`], [`PrefixMatch::cells`]).
    ///
    /// A request needs the logits after its last prompt token, so that row is
    /// always computed: look up the prompt without its last token, and the
    /// match can never cover it.
    pub fn lookup(&mut self, root: ChainHash, tokens: &[u64], cells_per_page: u32) -> PrefixMatch {
        let per_page = cells_per_page as usize;
        let mut found = PrefixMatch::default();
        let mut hash = root;
        for page_tokens in tokens.chunks_exact(per_page) {
            let next = hash.extend(page_tokens);
            let Some(entry) = self.entries.get(&next) else {
                found.computing = self.computing.get(&next).map(|owner| (next, *owner));
                break;
            };
            found.pages.push(entry.page);
            found.hashes.push(next);
            found.checkpoints.push(entry.checkpoint);
            self.touch(next);
            hash = next;
        }
        found.tail = self.match_tail(hash, &tokens[found.pages.len() * per_page..]);
        found
    }

    /// [`PrefixIndex::lookup`] for a prompt whose pages are named already
    /// (`chain` was given the prompt with [`Chain::name`]), so that looking
    /// again costs no hashing. `tokens` is the same text as for `lookup`.
    pub fn lookup_chain(&mut self, chain: &Chain, tokens: &[u64]) -> PrefixMatch {
        let per_page = chain.cells_per_page as usize;
        let mut found = PrefixMatch::default();
        let mut hash = chain.root;
        for next in chain.hashes.iter().take(tokens.len() / per_page) {
            let Some(entry) = self.entries.get(next) else {
                found.computing = self.computing.get(next).map(|owner| (*next, *owner));
                break;
            };
            found.pages.push(entry.page);
            found.hashes.push(*next);
            found.checkpoints.push(entry.checkpoint);
            self.touch(*next);
            hash = *next;
        }
        found.tail = self.match_tail(hash, &tokens[found.pages.len() * per_page..]);
        found
    }

    /// The tail under `prior` that shares the most leading tokens with
    /// `rest`, if any shares one.
    fn match_tail(&mut self, prior: ChainHash, rest: &[u64]) -> Option<TailMatch> {
        let tails = self.tails.get(&prior)?;
        let (best, cells) = tails
            .iter()
            .map(|tail| {
                let shared = tail.tokens.iter().zip(rest).take_while(|(a, b)| a == b).count();
                (tail, shared)
            })
            // The most cells; among equals the tail that ends there, whose
            // checkpoint can be used.
            .max_by_key(|(tail, shared)| (*shared, *shared == tail.tokens.len()))?;
        if cells == 0 {
            return None;
        }
        let found = TailMatch {
            page: best.page,
            cells: cells as u32,
            hash: best.hash,
            checkpoint: if cells == best.tokens.len() { best.checkpoint } else { None },
        };
        self.touch_tail(found.hash);
        Some(found)
    }

    /// What the index knows about the page named `hash`.
    pub fn state(&self, hash: ChainHash) -> PageState {
        if self.entries.contains_key(&hash) {
            PageState::Cached
        } else if let Some(owner) = self.computing.get(&hash) {
            PageState::Computing(*owner)
        } else {
            PageState::Unknown
        }
    }

    /// Announces that `owner` is going to compute the pages named `hashes`
    /// (the pages of its prompt that are not cached, from
    /// [`Chain::unpublished`]). Pages that are cached or announced by someone
    /// else are left alone. An announcement ends when the page is published
    /// or with [`PrefixIndex::withdraw`].
    ///
    /// Only a sequence that is running announces, never one that waits, so
    /// two prompts cannot wait for each other.
    pub fn announce(&mut self, owner: Owner, hashes: &[ChainHash]) {
        for hash in hashes {
            if !self.entries.contains_key(hash) {
                self.computing.entry(*hash).or_insert(owner);
            }
        }
    }

    /// The number of pages that are announced and not published yet.
    pub fn announced(&self) -> usize {
        self.computing.len()
    }

    /// Ends every announcement of `owner`: its sequence ended, was cancelled
    /// or was suspended before it computed the pages. Prompts that waited for
    /// them find them [`PageState::Unknown`] and compute them themselves.
    pub fn withdraw(&mut self, owner: Owner) {
        self.computing.retain(|_, who| *who != owner);
    }

    fn attach(&mut self, parent: ChainHash) -> u32 {
        match self.entries.get_mut(&parent) {
            Some(entry) => {
                if entry.children == 0 {
                    self.leaves.remove(&(entry.class, entry.last_used, parent));
                }
                entry.children += 1;
                entry.depth
            }
            None => 0,
        }
    }

    fn detach(&mut self, parent: ChainHash) {
        if let Some(entry) = self.entries.get_mut(&parent) {
            entry.children -= 1;
            if entry.children == 0 {
                self.leaves.insert((entry.class, entry.last_used, parent));
            }
        }
    }

    /// Records that `page` holds the content named `hash`, which follows
    /// `parent` (the previous page's hash, or the root for the first page).
    /// The index takes a reference to the page.
    ///
    /// Returns `false` and changes nothing when that content is cached
    /// already.
    ///
    /// This is the plain form for a caller that names pages itself. A
    /// sequence that runs publishes through [`PrefixIndex::publish`], which
    /// also drops the sequence's own copy of a page that is cached already.
    pub fn insert(
        &mut self,
        pool: &mut PagePool,
        parent: ChainHash,
        hash: ChainHash,
        page: PageId,
    ) -> bool {
        self.insert_page(pool, parent, hash, page, Priority::NORMAL, &[]).is_none()
    }

    /// Inserts a page, or returns the page that is cached under `hash`
    /// already. `tokens` are the page's tokens when the caller has them.
    fn insert_page(
        &mut self,
        pool: &mut PagePool,
        parent: ChainHash,
        hash: ChainHash,
        page: PageId,
        class: Priority,
        tokens: &[u64],
    ) -> Option<PageId> {
        if let Some(entry) = self.entries.get_mut(&hash) {
            let known = entry.page;
            if class > entry.class {
                if entry.children == 0 {
                    self.leaves.remove(&(entry.class, entry.last_used, hash));
                    self.leaves.insert((class, entry.last_used, hash));
                }
                entry.class = class;
            }
            self.touch(hash);
            return Some(known);
        }
        pool.retain(page);
        pool.set_cached(page, true);
        self.computing.remove(&hash);
        let depth = self.attach(parent) + 1;
        self.clock += 1;
        self.entries.insert(
            hash,
            Entry {
                page,
                parent,
                children: 0,
                depth,
                last_used: self.clock,
                class,
                checkpoint: None,
                checkpoint_by: None,
                checkpoint_root: false,
                mark: 0,
            },
        );
        self.leaves.insert((class, self.clock, hash));
        // The text went on past the tails that this page begins with.
        if let Some(tails) = self.tails.get_mut(&parent) {
            for tail in tails.iter_mut().filter(|tail| tokens.starts_with(&tail.tokens)) {
                tail.extended_by.push(hash);
            }
        }
        None
    }

    /// Publishes the pages of `seq` that are full and not in the index yet,
    /// up to `settled` cells (the cells that are committed: draft cells of a
    /// step in flight are not). `tokens` is the sequence's text from its
    /// start, at least `settled` long, and `chain` its chain, which is named
    /// further as needed. Called from the commit, this is what makes a page
    /// enter the index when it fills.
    ///
    /// Where a page is cached already (another sequence computed the same
    /// text), the sequence takes the cached page and lets go of its own, so
    /// the content is held once; its page table changes at that entry
    /// ([`Sequence::take_dirty`]). Either way the sequence afterwards holds
    /// the index's page for each of its full pages.
    ///
    /// Returns the range of page-table entries that were published, for
    /// whoever writes full pages to disk or attaches checkpoints.
    pub fn publish(
        &mut self,
        pool: &mut PagePool,
        seq: &mut Sequence,
        chain: &mut Chain,
        tokens: &[u64],
        settled: u64,
        class: Priority,
    ) -> Range<usize> {
        let per_page = chain.cells_per_page as usize;
        let settled = settled.min(seq.len()).min(tokens.len() as u64) as usize;
        let full = settled / per_page;
        chain.name(&tokens[..full * per_page]);
        let first = chain.published.min(full);
        for i in first..full {
            let page_tokens = &tokens[i * per_page..(i + 1) * per_page];
            let known = self.insert_page(
                pool,
                chain.before(i),
                chain.hashes[i],
                seq.pages()[i],
                class,
                page_tokens,
            );
            if let Some(known) = known {
                seq.adopt(pool, i, known);
            }
        }
        chain.published = chain.published.max(full);
        first..full
    }

    /// Publishes a tail: the first `tokens.len()` cells of `page`, which hold
    /// the state of `tokens` after the page named `prior` (or after the root).
    /// The index takes a reference to the page, so a sequence that goes on
    /// writing copies it first and the tail stays as it is.
    ///
    /// **What a tail may contain.** A prompt may take any leading part of a
    /// tail and continues it in a copy, so every cell of a tail must be
    /// stored in a form that depends on no later cell:
    ///
    /// - With `f16` or `q8` for K and V every row is encoded by itself. Any
    ///   partly used page qualifies, at any time.
    /// - With `q7` to `q4` the rows of a tile of 128 cells are stored
    ///   relative to the tile's mean and depend on all of its cells; a tile
    ///   is converted only when it is full and has left the window of newest
    ///   cells, and until then its cells are exact `f16` cells. A tail may
    ///   hold exact cells only. A page is one tile, so the partly used last
    ///   page of a sequence is always an exact entry and qualifies; the tail
    ///   must be published while that is so, which it is at the commit that
    ///   ends there. A page that is full is never a tail: once converted, its
    ///   leading cells cannot be continued.
    ///
    /// At most [`MAX_TAILS`] tails are kept under one page, and at most the
    /// limit of [`PrefixIndex::with_tail_limit`] in all. Returns the tail that
    /// was dropped to make room, if any. Publishing a tail that is known
    /// changes nothing but its age.
    ///
    /// # Panics
    ///
    /// Panics when `tokens` is empty.
    pub fn insert_tail(
        &mut self,
        pool: &mut PagePool,
        prior: ChainHash,
        tokens: &[u64],
        page: PageId,
        class: Priority,
    ) -> Option<Evicted> {
        assert!(!tokens.is_empty(), "a tail holds at least one cell");
        let hash = prior.extend(tokens);
        if self.tail_prior.contains_key(&hash) {
            self.touch_tail(hash);
            return None;
        }
        pool.retain(page);
        pool.set_cached(page, true);
        self.attach(prior);
        self.clock += 1;
        let tails = self.tails.entry(prior).or_default();
        tails.push(Tail {
            hash,
            tokens: tokens.to_vec(),
            page,
            last_used: self.clock,
            class,
            checkpoint: None,
            checkpoint_by: None,
            extended_by: Vec::new(),
        });
        // One tail goes at most: a page with too many, or the index with too
        // many (a new tail is one more of either).
        let oldest = if tails.len() > MAX_TAILS {
            tails.iter().min_by_key(|tail| tail.last_used).map(|tail| tail.hash)
        } else if self.tail_prior.len() >= self.tail_limit {
            self.tails.values().flatten().min_by_key(|tail| tail.last_used).map(|tail| tail.hash)
        } else {
            None
        };
        self.tail_prior.insert(hash, prior);
        self.leaves.insert((class, self.clock, hash));
        oldest.map(|oldest| self.remove_tail(pool, oldest))
    }

    /// The name of the tail that holds `tokens` after `prior`.
    pub fn tail_hash(prior: ChainHash, tokens: &[u64]) -> ChainHash {
        prior.extend(tokens)
    }

    fn remove_tail(&mut self, pool: &mut PagePool, hash: ChainHash) -> Evicted {
        let prior = self.tail_prior.remove(&hash).expect("removed tails exist");
        let tails = self.tails.get_mut(&prior).expect("a tail hangs under its prior");
        let at = tails.iter().position(|tail| tail.hash == hash).expect("listed under its prior");
        let tail = tails.swap_remove(at);
        if tails.is_empty() {
            self.tails.remove(&prior);
        }
        self.leaves.remove(&(tail.class, tail.last_used, hash));
        pool.set_cached(tail.page, false);
        pool.release(tail.page);
        self.detach(prior);
        self.checkpoints -= usize::from(tail.checkpoint.is_some());
        Evicted {
            page: tail.page,
            checkpoint: tail.checkpoint,
            root: false,
            class: tail.class,
            hash,
            parent: prior,
            depth: self.entries.get(&prior).map_or(0, |entry| entry.depth),
            tail: Some(tail.tokens),
        }
    }

    /// Sets the class of the cached page or tail named `hash`. Returns
    /// `false` when there is none.
    pub fn set_priority(&mut self, hash: ChainHash, class: Priority) -> bool {
        if let Some(entry) = self.entries.get_mut(&hash) {
            if entry.children == 0 {
                self.leaves.remove(&(entry.class, entry.last_used, hash));
                self.leaves.insert((class, entry.last_used, hash));
            }
            entry.class = class;
            return true;
        }
        let Some(tail) = self.tail_mut(hash) else { return false };
        let old = (tail.class, tail.last_used, hash);
        tail.class = class;
        let new = (class, tail.last_used, hash);
        self.leaves.remove(&old);
        self.leaves.insert(new);
        true
    }

    /// Attaches a checkpoint to the entry or tail `hash`, or removes it with
    /// `None`. Returns the checkpoint that was attached before, which the
    /// caller now owns again. `Err` gives the checkpoint back when there is
    /// no such entry.
    ///
    /// A checkpoint attached this way has no known publisher; a running
    /// sequence uses [`PrefixIndex::attach_checkpoint`].
    pub fn set_checkpoint(
        &mut self,
        hash: ChainHash,
        checkpoint: Option<CheckpointId>,
    ) -> Result<Option<CheckpointId>, Option<CheckpointId>> {
        self.replace_checkpoint(hash, checkpoint, None)
    }

    pub(crate) fn replace_checkpoint(
        &mut self,
        hash: ChainHash,
        checkpoint: Option<CheckpointId>,
        by: Option<Owner>,
    ) -> Result<Option<CheckpointId>, Option<CheckpointId>> {
        let (slot, slot_by) = if let Some(entry) = self.entries.get_mut(&hash) {
            entry.checkpoint_root = false;
            (&mut entry.checkpoint, &mut entry.checkpoint_by)
        } else {
            let Some(prior) = self.tail_prior.get(&hash) else { return Err(checkpoint) };
            let tails = self.tails.get_mut(prior).expect("a tail hangs under its prior");
            let tail = tails.iter_mut().find(|t| t.hash == hash).expect("listed under its prior");
            (&mut tail.checkpoint, &mut tail.checkpoint_by)
        };
        let before = std::mem::replace(slot, checkpoint);
        *slot_by = if checkpoint.is_some() { by } else { None };
        self.checkpoints =
            self.checkpoints + usize::from(checkpoint.is_some()) - usize::from(before.is_some());
        Ok(before)
    }

    /// Marks the checkpoint of the cached page `hash` as one at a root: the
    /// end of a system prompt or a tool list, which every conversation of an
    /// agent starts from. It is dropped after every other checkpoint and is
    /// never left behind. `false` when the page is not cached or has no
    /// checkpoint; a tail's checkpoint is never a root.
    pub fn mark_root(&mut self, hash: ChainHash) -> bool {
        match self.entries.get_mut(&hash) {
            Some(entry) if entry.checkpoint.is_some() => {
                entry.checkpoint_root = true;
                true
            }
            _ => false,
        }
    }

    /// Whether the checkpoint of the cached page `hash` is at a root.
    pub fn is_root(&self, hash: ChainHash) -> bool {
        self.entries.get(&hash).is_some_and(|entry| entry.checkpoint_root)
    }

    /// The cached page named `hash`: the page, the hash it follows and its
    /// number in its chain (1 for a first page). `None` for a tail and for a
    /// page the index does not hold.
    pub fn page(&self, hash: ChainHash) -> Option<(PageId, ChainHash, u32)> {
        self.entries.get(&hash).map(|entry| (entry.page, entry.parent, entry.depth))
    }

    /// The checkpoint attached to the entry or tail `hash`.
    pub fn checkpoint(&self, hash: ChainHash) -> Option<CheckpointId> {
        if let Some(entry) = self.entries.get(&hash) {
            return entry.checkpoint;
        }
        let prior = self.tail_prior.get(&hash)?;
        self.tails.get(prior)?.iter().find(|tail| tail.hash == hash)?.checkpoint
    }

    /// Gives up one page that no sequence holds, from the end of a chain,
    /// and returns it to the pool: of the lowest class there is, and within
    /// it the least recently used. A leaf goes whole, with its checkpoint.
    /// `None` when every cached page is in use by a sequence.
    pub fn evict(&mut self, pool: &mut PagePool) -> Option<Evicted> {
        let key = *self.leaves.iter().find(|(_, _, hash)| match self.entries.get(hash) {
            Some(entry) => pool.refcount(entry.page) == 1,
            None => self.tail_page(*hash).is_some_and(|page| pool.refcount(page) == 1),
        })?;
        let hash = key.2;
        let Some(entry) = self.entries.remove(&hash) else {
            return Some(self.remove_tail(pool, hash));
        };
        self.leaves.remove(&key);
        pool.set_cached(entry.page, false);
        pool.release(entry.page);
        if let Some(tails) = self.tails.get_mut(&entry.parent) {
            for tail in tails {
                tail.extended_by.retain(|page| *page != hash);
            }
        }
        self.detach(entry.parent);
        self.checkpoints -= usize::from(entry.checkpoint.is_some());
        Some(Evicted {
            page: entry.page,
            checkpoint: entry.checkpoint,
            root: entry.checkpoint_root,
            class: entry.class,
            hash,
            parent: entry.parent,
            depth: entry.depth,
            tail: None,
        })
    }

    fn tail_page(&self, hash: ChainHash) -> Option<PageId> {
        let prior = self.tail_prior.get(&hash)?;
        Some(self.tails.get(prior)?.iter().find(|tail| tail.hash == hash)?.page)
    }

    /// How many pages [`PrefixIndex::evict`] can give up, one after the
    /// other: the cached pages and tails that no sequence holds and that
    /// have nothing below them that a sequence holds.
    ///
    /// This walks the whole index. Admission asks for it only when a step's
    /// claims do not fit in the free pages.
    pub fn evictable(&mut self, pool: &PagePool) -> u32 {
        self.epoch += 1;
        let epoch = self.epoch;
        let mut held: Vec<ChainHash> = Vec::new();
        for (hash, entry) in &self.entries {
            if pool.refcount(entry.page) > 1 {
                held.push(*hash);
            }
        }
        let mut idle_tails = 0;
        for (prior, tails) in &self.tails {
            for tail in tails {
                if pool.refcount(tail.page) > 1 {
                    held.push(*prior);
                } else {
                    idle_tails += 1;
                }
            }
        }
        // Everything above a held page stays.
        let mut pinned = 0u32;
        for mut hash in held {
            while let Some(entry) = self.entries.get_mut(&hash) {
                if entry.mark == epoch {
                    break;
                }
                entry.mark = epoch;
                pinned += 1;
                hash = entry.parent;
            }
        }
        self.entries.len() as u32 - pinned + idle_tails
    }

    /// Gives up every cached page and returns the checkpoints that were
    /// attached. Announcements stay: they belong to running sequences.
    pub fn clear(&mut self, pool: &mut PagePool) -> Vec<CheckpointId> {
        self.leaves.clear();
        self.tail_prior.clear();
        self.checkpoints = 0;
        let mut checkpoints = Vec::new();
        for (_, entry) in self.entries.drain() {
            pool.set_cached(entry.page, false);
            pool.release(entry.page);
            checkpoints.extend(entry.checkpoint);
        }
        for (_, tails) in self.tails.drain() {
            for tail in tails {
                pool.set_cached(tail.page, false);
                pool.release(tail.page);
                checkpoints.extend(tail.checkpoint);
            }
        }
        checkpoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sequence;

    const PER_PAGE: u32 = 4;

    /// Appends `tokens` to `seq` and registers every page they complete.
    fn cache(
        pool: &mut PagePool,
        index: &mut PrefixIndex,
        root: ChainHash,
        seq: &mut Sequence,
        tokens: &[u64],
    ) {
        seq.append(pool, tokens.len() as u64).unwrap();
        let per_page = pool.cells_per_page() as usize;
        let mut hash = root;
        for (i, page_tokens) in tokens.chunks_exact(per_page).enumerate() {
            let parent = hash;
            hash = hash.extend(page_tokens);
            index.insert(pool, parent, hash, seq.pages()[i]);
        }
    }

    #[test]
    fn hashes_depend_on_context_order_and_page_boundaries() {
        let root = ChainHash::root(b"model-a");
        assert_ne!(root, ChainHash::root(b"model-b"));
        assert_ne!(ChainHash::root(b""), ChainHash::root(b"\0"));
        let ab = root.extend(&[1, 2]).extend(&[3, 4]);
        assert_eq!(ab, root.extend(&[1, 2]).extend(&[3, 4]));
        assert_ne!(ab, root.extend(&[3, 4]).extend(&[1, 2]));
        assert_ne!(ab, root.extend(&[1, 2, 3, 4]));
        assert_ne!(root.extend(&[0]), root.extend(&[0, 0]));
        assert_ne!(ab, ChainHash::root(b"model-b").extend(&[1, 2]).extend(&[3, 4]));
    }

    #[test]
    fn a_second_prompt_starts_from_the_pages_of_the_first() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let system: Vec<u64> = (100..110).collect();

        let mut first = Sequence::new();
        cache(&mut pool, &mut index, root, &mut first, &system);
        assert_eq!(index.len(), 2, "10 tokens complete 2 pages of 4");

        // Same 8 leading tokens, then a different continuation.
        let mut prompt = system[..8].to_vec();
        prompt.extend([7, 7, 7, 7, 7]);
        let found = index.lookup(root, &prompt, PER_PAGE);
        assert_eq!(found.pages, first.pages()[..2]);
        let mut second = Sequence::from_shared(&mut pool, &found.pages, 8);
        assert_eq!(pool.free_pages(), 13, "the second prompt allocated nothing yet");
        let grown = second.append(&mut pool, 5).unwrap();
        assert_eq!((grown.first_cell, grown.copy), (8, None));

        // Another context finds nothing, and so does a prompt that differs in
        // the first page.
        assert!(index.lookup(ChainHash::root(b"other"), &prompt, PER_PAGE).pages.is_empty());
        let mut other = prompt.clone();
        other[1] = 0;
        assert!(index.lookup(root, &other, PER_PAGE).pages.is_empty());

        first.release(&mut pool);
        second.release(&mut pool);
        assert_eq!(pool.free_pages(), 14, "the index still holds the 2 cached pages");
        assert!(index.clear(&mut pool).is_empty());
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn eviction_takes_unused_chain_ends_oldest_first() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");

        let old: Vec<u64> = (0..8).collect();
        let new: Vec<u64> = (50..58).collect();
        let (mut a, mut b) = (Sequence::new(), Sequence::new());
        cache(&mut pool, &mut index, root, &mut a, &old);
        cache(&mut pool, &mut index, root, &mut b, &new);
        let (a_pages, b_pages) = (a.pages().to_vec(), b.pages().to_vec());

        // While the sequences hold their pages nothing can be evicted.
        assert_eq!(index.evict(&mut pool), None);
        a.release(&mut pool);
        b.release(&mut pool);
        assert_eq!(pool.free_pages(), 12);

        // Using the old chain makes the other one the oldest.
        assert_eq!(index.lookup(root, &old, PER_PAGE).pages.len(), 2);
        // Ends before starts: a chain shrinks from its tail and stays usable.
        assert_eq!(index.evict(&mut pool).map(|e| e.page), Some(b_pages[1]));
        assert_eq!(index.evict(&mut pool).map(|e| e.page), Some(b_pages[0]));
        assert_eq!(index.evict(&mut pool).map(|e| e.page), Some(a_pages[1]));
        assert_eq!(index.lookup(root, &old, PER_PAGE).pages, a_pages[..1]);
        assert_eq!(index.evict(&mut pool).map(|e| e.page), Some(a_pages[0]));
        assert_eq!(index.evict(&mut pool), None);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn checkpoints_say_where_a_recurrent_model_can_resume() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let tokens: Vec<u64> = (0..12).collect();
        let mut seq = Sequence::new();
        cache(&mut pool, &mut index, root, &mut seq, &tokens);

        let found = index.lookup(root, &tokens, PER_PAGE);
        assert_eq!(found.pages.len(), 3);
        assert_eq!(found.last_checkpoint(), None, "pages alone are not enough");

        // A checkpoint after the second page.
        assert_eq!(index.set_checkpoint(found.hashes[1], Some(CheckpointId(7))), Ok(None));
        let found = index.lookup(root, &tokens, PER_PAGE);
        assert_eq!(found.last_checkpoint(), Some((2, CheckpointId(7))));

        // Unknown entries hand the checkpoint back.
        let unknown = root.extend(&[9, 9, 9, 9]);
        assert_eq!(
            index.set_checkpoint(unknown, Some(CheckpointId(8))),
            Err(Some(CheckpointId(8)))
        );

        // Evicting an entry returns its checkpoint to the caller.
        seq.release(&mut pool);
        assert_eq!(index.evict(&mut pool).and_then(|e| e.checkpoint), None);
        assert_eq!(index.evict(&mut pool).and_then(|e| e.checkpoint), Some(CheckpointId(7)));
        assert!(index.clear(&mut pool).is_empty());
    }

    #[test]
    fn inserting_known_content_keeps_the_first_page() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let tokens: Vec<u64> = (0..4).collect();
        let (mut a, mut b) = (Sequence::new(), Sequence::new());
        cache(&mut pool, &mut index, root, &mut a, &tokens);
        // A second sequence computed the same page on its own.
        cache(&mut pool, &mut index, root, &mut b, &tokens);
        assert_eq!(index.len(), 1);
        assert_eq!(index.lookup(root, &tokens, PER_PAGE).pages, a.pages());
        assert_eq!(pool.refcount(b.pages()[0]), 1, "the duplicate is not held by the index");
        a.release(&mut pool);
        b.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    /// A running sequence with its text and chain, as the conductor keeps it.
    struct Running {
        tokens: Vec<u64>,
        seq: Sequence,
        chain: Chain,
    }

    impl Running {
        /// Starts a prompt from whatever the index has for it. The last
        /// token is never taken from the cache.
        fn start(
            pool: &mut PagePool,
            index: &mut PrefixIndex,
            root: ChainHash,
            prompt: &[u64],
        ) -> Self {
            let found = index.lookup(root, &prompt[..prompt.len() - 1], PER_PAGE);
            let cells = found.cells(PER_PAGE);
            let seq = Sequence::from_shared(pool, &found.table(), cells);
            let chain = Chain::resumed(root, PER_PAGE, &found);
            Self { tokens: prompt[..cells as usize].to_vec(), seq, chain }
        }

        /// Computes `tokens` and publishes the pages they fill.
        fn step(&mut self, pool: &mut PagePool, index: &mut PrefixIndex, tokens: &[u64]) -> Grown {
            let grown = self.seq.append(pool, tokens.len() as u64).unwrap();
            self.tokens.extend_from_slice(tokens);
            let settled = self.tokens.len() as u64;
            index.publish(
                pool,
                &mut self.seq,
                &mut self.chain,
                &self.tokens,
                settled,
                Priority::NORMAL,
            );
            grown
        }

        /// Publishes the partly used last page as a tail.
        fn publish_tail(&self, pool: &mut PagePool, index: &mut PrefixIndex) -> ChainHash {
            let full = self.tokens.len() / PER_PAGE as usize;
            let rest = &self.tokens[full * PER_PAGE as usize..];
            let prior = self.chain.before(full);
            index.insert_tail(pool, prior, rest, self.seq.pages()[full], Priority::NORMAL);
            PrefixIndex::tail_hash(prior, rest)
        }
    }

    use crate::sequence::Grown;

    #[test]
    fn a_page_enters_the_index_when_it_fills_and_a_second_request_shares_it_at_once() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let prompt: Vec<u64> = (0..11).collect();

        // The first request has read 5 tokens of its prompt and is running.
        let mut first = Running::start(&mut pool, &mut index, root, &prompt);
        first.step(&mut pool, &mut index, &prompt[..3]);
        assert_eq!(index.len(), 0, "no page is full yet");
        first.step(&mut pool, &mut index, &prompt[3..5]);
        assert_eq!(index.len(), 1, "the first page filled and is in the index");

        // A second request with the same prompt arrives now and starts from
        // that page, while the first is still at work.
        let second = Running::start(&mut pool, &mut index, root, &prompt);
        assert_eq!(second.seq.pages(), &first.seq.pages()[..1]);
        assert_eq!(second.seq.len(), 4);

        // Cells of a step in flight (drafts) are not published: of 8 cells
        // only 7 are settled, so the second page stays out.
        first.seq.append(&mut pool, 3).unwrap();
        first.tokens.extend_from_slice(&prompt[5..8]);
        let published = index.publish(
            &mut pool,
            &mut first.seq,
            &mut first.chain,
            &first.tokens,
            7,
            Priority::NORMAL,
        );
        assert_eq!((published, index.len()), (1..1, 1));
        let published = index.publish(
            &mut pool,
            &mut first.seq,
            &mut first.chain,
            &first.tokens,
            8,
            Priority::NORMAL,
        );
        assert_eq!((published, index.len()), (1..2, 2));

        first.seq.release(&mut pool);
        second.seq.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn a_sequence_gives_up_its_own_copy_of_a_page_that_is_cached() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let text: Vec<u64> = (0..9).collect();
        // Two requests started before either had filled a page, so both
        // compute the first two pages.
        let mut a = Running::start(&mut pool, &mut index, root, &text);
        let mut b = Running::start(&mut pool, &mut index, root, &text);
        b.seq.append(&mut pool, 9).unwrap();
        b.tokens.extend_from_slice(&text);
        a.step(&mut pool, &mut index, &text);
        assert_eq!(pool.free_pages(), 16 - 6);
        b.seq.take_dirty();

        let published =
            index.publish(&mut pool, &mut b.seq, &mut b.chain, &b.tokens, 9, Priority::NORMAL);
        assert_eq!(published, 0..2);
        assert_eq!(b.seq.pages()[..2], a.seq.pages()[..2], "b reads a's pages now");
        assert_eq!(pool.free_pages(), 16 - 4, "and b's two copies are free");
        assert_eq!(b.seq.take_dirty(), 0..3, "its table changed from the first entry");
        assert_eq!(index.len(), 2);

        a.seq.release(&mut pool);
        b.seq.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn a_prompt_resumes_at_the_end_of_the_cached_text_through_a_tail() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        // A conversation's first turn: 6 tokens, so one full page and two
        // cells in the next.
        let turn: Vec<u64> = (0..6).collect();
        let mut first = Running::start(&mut pool, &mut index, root, &turn);
        first.step(&mut pool, &mut index, &turn);
        first.publish_tail(&mut pool, &mut index);
        let tail_page = first.seq.pages()[1];
        first.seq.release(&mut pool);
        assert_eq!((index.len(), index.tails()), (2, 1));
        assert_eq!(pool.free_pages(), 14, "the index holds the page and the tail");

        // The second turn continues the text. Without the tail it would
        // start at cell 4; with it, at cell 6.
        let mut prompt = turn.clone();
        prompt.extend([50, 51, 52]);
        let mut second = Running::start(&mut pool, &mut index, root, &prompt);
        assert_eq!(second.seq.len(), 6);
        assert_eq!(second.seq.pages()[1], tail_page);
        // Its first write copies the two cells out of the tail's page.
        let grown = second.step(&mut pool, &mut index, &prompt[6..]);
        let copy = grown.copy.expect("the tail's page is shared");
        assert_eq!((copy.from, copy.cells, grown.first_cell), (tail_page, 2, 6));
        assert_ne!(second.seq.pages()[1], tail_page);

        // A prompt that leaves the text inside the tail takes the cell it
        // shares; one that differs at once takes the full page only.
        let found = index.lookup(root, &[0, 1, 2, 3, 4, 99, 99], PER_PAGE);
        assert_eq!(found.tail.map(|t| (t.page, t.cells)), Some((tail_page, 1)));
        assert_eq!(found.cells(PER_PAGE), 5);
        let found = index.lookup(root, &[0, 1, 2, 3, 99, 99], PER_PAGE);
        assert_eq!((found.pages.len(), found.tail), (1, None));
        // The same text again: the last token is computed, so of the tail's
        // two cells one is taken.
        let again = Running::start(&mut pool, &mut index, root, &turn);
        assert_eq!(again.seq.len(), 5);

        again.seq.release(&mut pool);
        second.seq.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn a_tail_stays_as_it_was_when_its_sequence_goes_on() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let prompt: Vec<u64> = (0..6).collect();
        // A tail at the end of the prompt, then the request generates.
        let mut a = Running::start(&mut pool, &mut index, root, &prompt);
        a.step(&mut pool, &mut index, &prompt);
        let tail = a.publish_tail(&mut pool, &mut index);
        let tail_page = a.seq.pages()[1];
        let grown = a.step(&mut pool, &mut index, &[70, 71, 72]);
        // The sequence writes into a copy; the tail's page keeps the two
        // cells and nothing else.
        assert_eq!(grown.copy.map(|c| (c.from, c.cells)), Some((tail_page, 2)));
        assert_ne!(a.seq.pages()[1], tail_page);
        assert_eq!(pool.refcount(tail_page), 1);
        // The page the sequence filled is cached next to the tail, and the
        // tail still serves a prompt that branches after it.
        assert_eq!((index.len(), index.tails()), (3, 1));
        let found = index.lookup(root, &[0, 1, 2, 3, 4, 5, 9, 9], PER_PAGE);
        assert_eq!(found.tail.map(|t| (t.hash, t.cells)), Some((tail, 2)));
        // Publishing the same tail again changes nothing.
        assert_eq!(
            index.insert_tail(&mut pool, a.chain.before(1), &[4, 5], tail_page, Priority::NORMAL),
            None
        );
        assert_eq!(index.tails(), 1);
        a.seq.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn at_most_eight_tails_are_kept_under_a_page_and_the_least_recently_used_goes() {
        let mut pool = PagePool::new(32, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let mut pages = Vec::new();
        for i in 0..MAX_TAILS as u32 {
            // Nine answers to the same question, each ending inside the page.
            let mut seq = Sequence::new();
            seq.append(&mut pool, 2).unwrap();
            assert_eq!(
                index.insert_tail(&mut pool, root, &[1, 100 + u64::from(i)], seq.pages()[0], Priority::NORMAL),
                None
            );
            pages.push(seq.pages()[0]);
            seq.release(&mut pool);
        }
        let first = PrefixIndex::tail_hash(root, &[1, 100]);
        index.set_checkpoint(first, Some(CheckpointId(3))).unwrap();
        // The second tail is used; the first is now the least recently used.
        assert_eq!(index.lookup(root, &[1, 101, 7], PER_PAGE).tail.map(|t| t.cells), Some(2));
        let mut seq = Sequence::new();
        seq.append(&mut pool, 3).unwrap();
        let dropped =
            index.insert_tail(&mut pool, root, &[1, 200, 201], seq.pages()[0], Priority::NORMAL);
        assert_eq!(
            dropped,
            Some(Evicted {
                page: pages[0],
                checkpoint: Some(CheckpointId(3)),
                root: false,
                class: Priority::NORMAL,
                hash: first,
                parent: root,
                depth: 0,
                tail: Some(vec![1, 100]),
            })
        );
        seq.release(&mut pool);
        assert_eq!((index.tails(), index.checkpoints()), (MAX_TAILS, 0));
        assert_eq!(pool.free_pages(), 32 - MAX_TAILS as u32);
        // The longest shared start wins among tails.
        assert_eq!(index.lookup(root, &[1, 200, 201, 5], PER_PAGE).tail.map(|t| t.cells), Some(3));
        // Tails are chain ends: eviction takes them, oldest first.
        assert_eq!(index.evict(&mut pool).map(|e| e.page), Some(pages[2]));
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 32);
    }

    #[test]
    fn the_index_keeps_no_more_tails_than_its_limit_whatever_pages_they_hang_under() {
        let mut pool = PagePool::new(32, PER_PAGE);
        let mut index = PrefixIndex::new().with_tail_limit(2);
        let mut pages = Vec::new();
        // Three tails under three different roots.
        for i in 0..3u32 {
            let root = ChainHash::root(&[i as u8]);
            let mut seq = Sequence::new();
            seq.append(&mut pool, 2).unwrap();
            let dropped =
                index.insert_tail(&mut pool, root, &[1, 2], seq.pages()[0], Priority::NORMAL);
            pages.push(seq.pages()[0]);
            seq.release(&mut pool);
            // The third one makes the first go.
            assert_eq!(dropped.map(|e| e.page), (i == 2).then_some(pages[0]), "tail {i}");
        }
        assert_eq!(index.tails(), 2);
        assert_eq!(pool.free_pages(), 32 - 2);
        // A tail that is used is not the one to go.
        let first = ChainHash::root(&[1]);
        assert_eq!(index.lookup(first, &[1, 2, 9], PER_PAGE).tail.map(|t| t.cells), Some(2));
        let root = ChainHash::root(&[3]);
        let mut seq = Sequence::new();
        seq.append(&mut pool, 2).unwrap();
        let dropped = index.insert_tail(&mut pool, root, &[7, 8], seq.pages()[0], Priority::NORMAL);
        seq.release(&mut pool);
        assert_eq!(dropped.map(|e| e.page), Some(pages[2]), "the tail unused longest");
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 32);
    }

    #[test]
    fn a_prompt_is_told_when_the_page_it_misses_is_being_computed() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let prompt: Vec<u64> = (0..13).collect();

        // The first request names its prompt's pages and announces them.
        let mut first = Running::start(&mut pool, &mut index, root, &prompt);
        first.chain.name(&prompt);
        assert_eq!(first.chain.unpublished().len(), 3);
        index.announce(Owner(1), first.chain.unpublished());

        // A second request with the same start misses the first page, which
        // is announced: it waits instead of computing the same rows.
        let mut other = prompt[..8].to_vec();
        other.extend([90, 91, 92, 93, 94]);
        let mut chain = Chain::new(root, PER_PAGE);
        chain.name(&other);
        let found = index.lookup_chain(&chain, &other[..other.len() - 1]);
        let (waits_for, owner) = found.computing.expect("the first page is announced");
        assert_eq!((found.pages.len(), owner), (0, Owner(1)));
        assert_eq!(index.state(waits_for), PageState::Computing(Owner(1)));
        // Its own announcement of the same pages changes nothing.
        index.announce(Owner(2), chain.unpublished());
        assert_eq!(index.state(waits_for), PageState::Computing(Owner(1)));

        // The page is published: the announcement ends and the wait with it.
        assert_eq!(index.announced(), 4, "three of the first, and the second's own third page");
        first.step(&mut pool, &mut index, &prompt[..5]);
        assert_eq!(index.state(waits_for), PageState::Cached);
        assert_eq!(index.announced(), 3);
        let found = index.lookup_chain(&chain, &other[..other.len() - 1]);
        // It has the first page now and waits for the second.
        assert_eq!(found.pages.len(), 1);
        assert_eq!(found.computing, Some((chain.hashes()[1], Owner(1))));

        // The first request is cancelled before it gets there. Its
        // announcements are withdrawn; the page that only the second request
        // announced (its own third) stays announced.
        index.withdraw(Owner(1));
        let found = index.lookup_chain(&chain, &other[..other.len() - 1]);
        assert_eq!((found.pages.len(), found.computing), (1, None));
        assert_eq!(index.state(chain.hashes()[1]), PageState::Unknown);
        assert_eq!(index.announced(), 1);
        index.withdraw(Owner(2));
        assert_eq!(index.announced(), 0);
        first.seq.release(&mut pool);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn a_lower_class_is_given_up_before_a_higher_one_whatever_its_age() {
        let mut pool = PagePool::new(16, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let keep = Priority(200);
        // An old system prompt that was asked to be kept, then two newer
        // prompts of the normal class.
        let system: Vec<u64> = (0..8).collect();
        let mut seq = Sequence::new();
        let mut chain = Chain::new(root, PER_PAGE);
        seq.append(&mut pool, 8).unwrap();
        index.publish(&mut pool, &mut seq, &mut chain, &system, 8, keep);
        let system_pages = seq.pages().to_vec();
        seq.release(&mut pool);
        let mut newer = Vec::new();
        for start in [100u64, 200] {
            let tokens: Vec<u64> = (start..start + 4).collect();
            let mut run = Running::start(&mut pool, &mut index, root, &tokens);
            run.step(&mut pool, &mut index, &tokens);
            newer.push(run.seq.pages()[0]);
            run.seq.release(&mut pool);
        }
        // The normal ones go first though they are newer; then the kept
        // chain, from its end.
        let order: Vec<PageId> =
            std::iter::from_fn(|| index.evict(&mut pool)).map(|e| e.page).collect();
        assert_eq!(order, [newer[0], newer[1], system_pages[1], system_pages[0]]);

        // A class can be set later, and publishing known content in a higher
        // class raises it.
        let mut a = Running::start(&mut pool, &mut index, root, &system);
        a.step(&mut pool, &mut index, &system);
        assert!(index.set_priority(a.chain.hashes()[1], Priority(1)));
        assert!(!index.set_priority(root, Priority(1)));
        let other: Vec<u64> = (300..304).collect();
        let mut b = Running::start(&mut pool, &mut index, root, &other);
        b.step(&mut pool, &mut index, &other);
        // A second sequence computed the same page and publishes it in the
        // higher class: the cached page is kept in that class from now on.
        let mut c = Sequence::new();
        c.append(&mut pool, 4).unwrap();
        index.publish(&mut pool, &mut c, &mut Chain::new(root, PER_PAGE), &other, 4, keep);
        assert_eq!(c.pages(), b.seq.pages());
        let (pages_of_a, page_of_b) = (a.seq.pages().to_vec(), b.seq.pages()[0]);
        a.seq.release(&mut pool);
        b.seq.release(&mut pool);
        c.release(&mut pool);
        let order: Vec<(PageId, Priority)> =
            std::iter::from_fn(|| index.evict(&mut pool)).map(|e| (e.page, e.class)).collect();
        assert_eq!(
            order,
            [(pages_of_a[1], Priority(1)), (pages_of_a[0], Priority::NORMAL), (page_of_b, keep)]
        );
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn evictable_is_what_evict_can_give_up() {
        let mut pool = PagePool::new(32, PER_PAGE);
        let mut index = PrefixIndex::new();
        let root = ChainHash::root(b"m");
        let long: Vec<u64> = (0..18).collect();
        let mut a = Running::start(&mut pool, &mut index, root, &long);
        a.step(&mut pool, &mut index, &long);
        a.publish_tail(&mut pool, &mut index);
        assert_eq!(index.evictable(&pool), 0, "the sequence holds all of it");
        // A fork holds the first two pages.
        let fork = a.seq.fork_prefix(&mut pool, 8);
        a.seq.release(&mut pool);
        assert_eq!(index.evictable(&pool), 3, "two pages and the tail below what the fork holds");
        // A page inserted by hand under a page that a sequence does not hold:
        // the sequence holds the child and not the parent, and the parent is
        // still not evictable.
        let other: Vec<u64> = (500..508).collect();
        let (mut b, mut c) = (Sequence::new(), Sequence::new());
        b.append(&mut pool, 8).unwrap();
        c.append(&mut pool, 8).unwrap();
        let h0 = root.extend(&other[..4]);
        let h1 = h0.extend(&other[4..]);
        index.insert(&mut pool, root, h0, b.pages()[0]);
        index.insert(&mut pool, h0, h1, c.pages()[1]);
        b.release(&mut pool);
        assert_eq!(pool.refcount(index.lookup(root, &other[..4], PER_PAGE).pages[0]), 1);
        assert_eq!(index.evictable(&pool), 3);
        c.release(&mut pool);
        assert_eq!(index.evictable(&pool), 5);

        let mut given_up = 0;
        while index.evict(&mut pool).is_some() {
            given_up += 1;
        }
        assert_eq!(given_up, 5);
        assert_eq!(index.evictable(&pool), 0);
        fork.release(&mut pool);
        assert_eq!(index.evictable(&pool), 2);
        index.clear(&mut pool);
        assert_eq!(pool.free_pages(), 32);
    }
}
