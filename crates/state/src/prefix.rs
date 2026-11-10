//! Finding cached state for the start of a prompt.
//!
//! Every full page of a sequence can be named by a hash of everything that
//! determines its content: the context it was computed in and all tokens up
//! to its end. The hashes form a chain (each page's hash extends the previous
//! one), so the longest cached prefix of a new prompt is found by walking its
//! pages from the start until the index has no entry.
//!
//! The index holds one reference to each page it knows, so cached pages
//! outlive the sequence that computed them and any number of sequences can
//! start from them. Pages nobody else holds are given up oldest-first, from
//! the ends of the chains, when the pool runs short.
//!
//! State that is not addressable by position (recurrent layers) cannot be
//! resumed from pages alone. For models that have it, a checkpoint can be
//! attached to an entry; a prompt can then resume at the last matched entry
//! that carries one.

use std::collections::{BTreeSet, HashMap};

use crate::pool::{PageId, PagePool};

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
    pub fn extend(self, tokens: &[u32]) -> Self {
        let (mut a, mut b) = (self.0 as u64 ^ K[0], (self.0 >> 64) as u64 ^ K[1]);
        for token in tokens {
            let token = u64::from(*token);
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

struct Entry {
    page: PageId,
    parent: ChainHash,
    children: u32,
    last_used: u64,
    checkpoint: Option<CheckpointId>,
}

/// The cached start of a prompt: one entry per matched page, in order.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PrefixMatch {
    pub pages: Vec<PageId>,
    pub hashes: Vec<ChainHash>,
    pub checkpoints: Vec<Option<CheckpointId>>,
}

impl PrefixMatch {
    /// The number of matched pages that can be resumed from when the model
    /// needs a checkpoint to resume, and that checkpoint.
    pub fn last_checkpoint(&self) -> Option<(usize, CheckpointId)> {
        self.checkpoints.iter().enumerate().rev().find_map(|(i, c)| Some((i + 1, (*c)?)))
    }
}

/// What the index gave up to free a page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evicted {
    pub page: PageId,
    /// The checkpoint that was attached to the entry and is now unreferenced.
    pub checkpoint: Option<CheckpointId>,
}

/// The index of cached pages by content.
#[derive(Default)]
pub struct PrefixIndex {
    entries: HashMap<ChainHash, Entry>,
    /// Entries without children, oldest use first. Only these can be given
    /// up: an entry in the middle of a chain is what makes its successors
    /// reachable.
    leaves: BTreeSet<(u64, ChainHash)>,
    clock: u64,
}

impl std::fmt::Debug for PrefixIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixIndex")
            .field("entries", &self.entries.len())
            .field("leaves", &self.leaves.len())
            .finish()
    }
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of cached pages.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn touch(&mut self, hash: ChainHash) {
        self.clock += 1;
        let entry = self.entries.get_mut(&hash).expect("touched entries exist");
        if entry.children == 0 {
            self.leaves.remove(&(entry.last_used, hash));
            self.leaves.insert((self.clock, hash));
        }
        entry.last_used = self.clock;
    }

    /// The longest cached prefix of `tokens` in whole pages, starting from
    /// `root`. The matched entries count as used now.
    ///
    /// The caller takes the pages it resumes from with
    /// [`Sequence::from_shared`](crate::Sequence::from_shared).
    pub fn lookup(&mut self, root: ChainHash, tokens: &[u32], cells_per_page: u32) -> PrefixMatch {
        let mut found = PrefixMatch::default();
        let mut hash = root;
        for page_tokens in tokens.chunks_exact(cells_per_page as usize) {
            hash = hash.extend(page_tokens);
            let Some(entry) = self.entries.get(&hash) else { break };
            found.pages.push(entry.page);
            found.hashes.push(hash);
            found.checkpoints.push(entry.checkpoint);
            self.touch(hash);
        }
        found
    }

    /// Records that `page` holds the content named `hash`, which follows
    /// `parent` (the previous page's hash, or the root for the first page).
    /// The index takes a reference to the page.
    ///
    /// Returns `false` and changes nothing when that content is cached
    /// already.
    pub fn insert(
        &mut self,
        pool: &mut PagePool,
        parent: ChainHash,
        hash: ChainHash,
        page: PageId,
    ) -> bool {
        if self.entries.contains_key(&hash) {
            self.touch(hash);
            return false;
        }
        pool.retain(page);
        self.clock += 1;
        if let Some(parent_entry) = self.entries.get_mut(&parent) {
            if parent_entry.children == 0 {
                self.leaves.remove(&(parent_entry.last_used, parent));
            }
            parent_entry.children += 1;
        }
        self.entries.insert(
            hash,
            Entry { page, parent, children: 0, last_used: self.clock, checkpoint: None },
        );
        self.leaves.insert((self.clock, hash));
        true
    }

    /// Attaches a checkpoint to the entry `hash`, or removes it with `None`.
    /// Returns the checkpoint that was attached before, which the caller now
    /// owns again. `Err` gives the checkpoint back when there is no such
    /// entry.
    pub fn set_checkpoint(
        &mut self,
        hash: ChainHash,
        checkpoint: Option<CheckpointId>,
    ) -> Result<Option<CheckpointId>, Option<CheckpointId>> {
        match self.entries.get_mut(&hash) {
            Some(entry) => Ok(std::mem::replace(&mut entry.checkpoint, checkpoint)),
            None => Err(checkpoint),
        }
    }

    /// Gives up the least recently used cached page that no sequence holds,
    /// from the end of a chain, and returns it to the pool. `None` when every
    /// cached page is in use by a sequence.
    pub fn evict(&mut self, pool: &mut PagePool) -> Option<Evicted> {
        let key =
            *self.leaves.iter().find(|(_, hash)| pool.refcount(self.entries[hash].page) == 1)?;
        self.leaves.remove(&key);
        let entry = self.entries.remove(&key.1).expect("leaves are entries");
        pool.release(entry.page);
        if let Some(parent) = self.entries.get_mut(&entry.parent) {
            parent.children -= 1;
            if parent.children == 0 {
                self.leaves.insert((parent.last_used, entry.parent));
            }
        }
        Some(Evicted { page: entry.page, checkpoint: entry.checkpoint })
    }

    /// Gives up every cached page and returns the checkpoints that were
    /// attached.
    pub fn clear(&mut self, pool: &mut PagePool) -> Vec<CheckpointId> {
        self.leaves.clear();
        self.entries
            .drain()
            .filter_map(|(_, entry)| {
                pool.release(entry.page);
                entry.checkpoint
            })
            .collect()
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
        tokens: &[u32],
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
        let system: Vec<u32> = (100..110).collect();

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

        let old: Vec<u32> = (0..8).collect();
        let new: Vec<u32> = (50..58).collect();
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
        let tokens: Vec<u32> = (0..12).collect();
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
        let tokens: Vec<u32> = (0..4).collect();
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
}
