//! Drafting by lookup.

use std::collections::HashMap;

/// Proposes drafts by finding where the last few tokens occurred before and
/// continuing from there.
///
/// Models copy a lot: they quote the prompt, rewrite a file with one line
/// changed, repeat a function signature. Whenever the text being generated
/// has an earlier occurrence in the same sequence, the tokens that followed
/// it are a draft that costs a hash lookup and is usually right for a long
/// stretch.
///
/// The drafter keeps the sequence's tokens (prompt and output) and, for every
/// run of `n` tokens, where that run last ended.
#[derive(Clone, Debug)]
pub struct NgramDrafter {
    n: usize,
    tokens: Vec<u32>,
    /// Hash of a run of `n` tokens -> index of its last token, for the most
    /// recent run that ended before the current end. Entries can be stale
    /// after a rollback and are checked against the tokens when used.
    last_end: HashMap<u64, u32>,
}

impl NgramDrafter {
    /// A drafter that matches on the last `n` tokens. Short runs match often
    /// and wrongly; 3 or more is sensible for subword tokens.
    ///
    /// # Panics
    ///
    /// Panics when `n` is 0.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "a match needs at least one token");
        Self { n, tokens: Vec::new(), last_end: HashMap::new() }
    }

    /// The number of tokens seen.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    fn key(run: &[u32]) -> u64 {
        run.iter().fold(0xcbf2_9ce4_8422_2325, |hash, token| {
            (hash ^ u64::from(*token)).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(17)
        })
    }

    /// Adds a token of the sequence (prompt or accepted output).
    pub fn push(&mut self, token: u32) {
        // The run that ended at the previous token becomes findable now that
        // it is no longer the end of the sequence. Indexing it earlier would
        // make the current end find itself.
        let len = self.tokens.len();
        if len >= self.n {
            let key = Self::key(&self.tokens[len - self.n..]);
            self.last_end.insert(key, (len - 1) as u32);
        }
        self.tokens.push(token);
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        for token in tokens {
            self.push(*token);
        }
    }

    /// Forgets the tokens past `len`, after a rollback.
    pub fn truncate(&mut self, len: usize) {
        self.tokens.truncate(len);
    }

    /// Up to `max` tokens that followed the most recent earlier occurrence of
    /// the last `n` tokens. Empty when there is none.
    pub fn draft(&self, max: usize) -> &[u32] {
        let len = self.tokens.len();
        if len < self.n {
            return &[];
        }
        let suffix = &self.tokens[len - self.n..];
        let Some(&end) = self.last_end.get(&Self::key(suffix)) else {
            return &[];
        };
        let end = end as usize;
        // A stale entry (from before a rollback) or a hash collision: the
        // tokens decide.
        if end + 1 >= len || end + 1 < self.n || self.tokens[end + 1 - self.n..=end] != *suffix {
            return &[];
        }
        &self.tokens[end + 1..len.min(end + 1 + max)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continues_an_earlier_occurrence() {
        let mut drafter = NgramDrafter::new(3);
        // A "file" in the prompt, then the model starts to write it again.
        drafter.extend(&[10, 11, 12, 13, 14, 15, 16, 99, 98]);
        assert!(drafter.draft(8).is_empty(), "nothing repeats yet");
        drafter.extend(&[10, 11]);
        assert!(drafter.draft(8).is_empty(), "two tokens are not a match of three");
        drafter.push(12);
        assert_eq!(drafter.draft(3), [13, 14, 15]);
        assert_eq!(drafter.draft(100), [13, 14, 15, 16, 99, 98, 10, 11, 12]);

        // The drafts are accepted: the next lookup continues further.
        drafter.extend(&[13, 14]);
        assert_eq!(drafter.draft(2), [15, 16]);
    }

    #[test]
    fn prefers_the_most_recent_occurrence() {
        let mut drafter = NgramDrafter::new(2);
        drafter.extend(&[1, 2, 30, 0, 1, 2, 40, 0, 1, 2]);
        assert_eq!(drafter.draft(1), [40]);
    }

    #[test]
    fn a_repeating_pattern_drafts_up_to_the_end_of_what_exists() {
        let mut drafter = NgramDrafter::new(2);
        drafter.extend(&[7, 8, 9, 7, 8, 9, 7, 8]);
        // The run "7 8" last ended at index 4; what followed is "9 7 8".
        assert_eq!(drafter.draft(10), [9, 7, 8]);
    }

    #[test]
    fn a_rollback_leaves_no_drafts_from_removed_text() {
        let mut drafter = NgramDrafter::new(2);
        drafter.extend(&[1, 2, 3, 4, 5, 6, 1, 2]);
        assert_eq!(drafter.draft(2), [3, 4]);

        // Rejected drafts are cut; the sequence goes on differently.
        drafter.truncate(4);
        assert_eq!(drafter.len(), 4);
        drafter.extend(&[9, 9, 5, 6]);
        // "5 6" was only in the removed text.
        assert!(drafter.draft(2).is_empty());
        drafter.extend(&[1, 2]);
        assert_eq!(drafter.draft(3), [3, 4, 9]);
    }

    #[test]
    fn short_sequences_have_no_drafts() {
        let mut drafter = NgramDrafter::new(4);
        assert!(drafter.draft(4).is_empty());
        drafter.extend(&[1, 1, 1]);
        assert!(drafter.draft(4).is_empty());
        drafter.push(1);
        assert!(drafter.draft(4).is_empty(), "the only occurrence is the end itself");
        drafter.push(1);
        assert_eq!(drafter.draft(4), [1]);
    }
}
