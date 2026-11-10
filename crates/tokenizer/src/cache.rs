//! A cache of the tokens of words that came up before.
//!
//! The tokens of a word depend on the word and the merges, nothing else, and
//! text repeats its words: a few thousand words make up most of English and
//! most of any source language. Looking a word up costs a hash and a compare;
//! merging it costs a dozen lookups in a table of several megabytes.

/// Words longer than this are not cached.
pub(crate) const MAX_WORD: usize = 24;
/// Words with more tokens than this are not cached.
const MAX_TOKENS: usize = 8;
/// A power of two. 4096 entries of 64 bytes stay in the second-level cache.
const ENTRIES: usize = 4096;

#[derive(Clone, Copy)]
struct Entry {
    len: u8,
    n_tokens: u8,
    word: [u8; MAX_WORD],
    tokens: [u32; MAX_TOKENS],
}

const EMPTY: Entry = Entry { len: 0, n_tokens: 0, word: [0; MAX_WORD], tokens: [0; MAX_TOKENS] };

/// A direct-mapped cache: a word has one place, and a newer word with the same
/// place replaces it.
pub(crate) struct WordCache {
    /// Which tokenizer the entries belong to.
    owner: u64,
    entries: Box<[Entry]>,
}

impl WordCache {
    pub(crate) fn new(owner: u64) -> Self {
        Self { owner, entries: vec![EMPTY; ENTRIES].into() }
    }

    /// Empties the cache if it was filled by another tokenizer.
    pub(crate) fn belong_to(&mut self, owner: u64) {
        if self.owner != owner {
            self.owner = owner;
            self.entries.fill(EMPTY);
        }
    }

    #[inline(always)]
    fn slot(word: &[u8]) -> usize {
        let mut hash = word.len() as u64;
        for chunk in word.chunks(8) {
            let mut bytes = [0u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            hash = (hash.rotate_left(5) ^ u64::from_le_bytes(bytes))
                .wrapping_mul(0x517C_C1B7_2722_0A95);
        }
        (hash >> (64 - ENTRIES.trailing_zeros())) as usize
    }

    /// The tokens of `word`, if they are in the cache. `word` has at most
    /// [`MAX_WORD`] bytes.
    #[inline(always)]
    pub(crate) fn get(&self, word: &[u8]) -> Option<&[u32]> {
        let entry = &self.entries[Self::slot(word)];
        (usize::from(entry.len) == word.len() && entry.word[..word.len()] == *word)
            .then(|| &entry.tokens[..usize::from(entry.n_tokens)])
    }

    /// Remembers the tokens of `word`, unless they are too many to keep.
    pub(crate) fn put(&mut self, word: &[u8], tokens: &[u32]) {
        if tokens.len() > MAX_TOKENS {
            return;
        }
        let entry = &mut self.entries[Self::slot(word)];
        entry.len = word.len() as u8;
        entry.n_tokens = tokens.len() as u8;
        entry.word[..word.len()].copy_from_slice(word);
        entry.tokens[..tokens.len()].copy_from_slice(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_what_was_put_and_nothing_else() {
        let mut cache = WordCache::new(1);
        assert_eq!(cache.get(b"hello"), None);
        cache.put(b"hello", &[7, 8]);
        assert_eq!(cache.get(b"hello"), Some(&[7, 8][..]));
        assert_eq!(cache.get(b"hell"), None, "a prefix is another word");
        assert_eq!(cache.get(b"hello!"), None);
        cache.put(b"x", &[]);
        assert_eq!(cache.get(b"x"), Some(&[][..]));
    }

    #[test]
    fn keeps_a_word_of_full_length_and_refuses_long_token_lists() {
        let mut cache = WordCache::new(1);
        let word = [b'a'; MAX_WORD];
        cache.put(&word, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(cache.get(&word), Some(&[1, 2, 3, 4, 5, 6, 7, 8][..]));
        cache.put(b"many", &[0; 9]);
        assert_eq!(cache.get(b"many"), None);
    }

    #[test]
    fn a_new_owner_starts_empty() {
        let mut cache = WordCache::new(1);
        cache.put(b"a", &[1]);
        cache.belong_to(1);
        assert!(cache.get(b"a").is_some());
        cache.belong_to(2);
        assert!(cache.get(b"a").is_none());
    }

    #[test]
    fn a_word_replaces_another_that_has_the_same_place() {
        let mut cache = WordCache::new(1);
        // Find two words that collide, by trying.
        let first = b"w0".to_vec();
        let second = (1..100_000u32)
            .map(|i| format!("w{i}").into_bytes())
            .find(|w| WordCache::slot(w) == WordCache::slot(&first))
            .expect("a collision among 100000 words in 4096 places");
        cache.put(&first, &[1]);
        cache.put(&second, &[2]);
        assert_eq!(cache.get(&second), Some(&[2][..]));
        assert_eq!(cache.get(&first), None);
    }
}
