//! The set of tokens a row may pick, as one bit per token of the vocabulary.

/// A bit per token: set means the token may be picked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenMask {
    words: Vec<u32>,
    tokens: u32,
}

impl TokenMask {
    /// The words a mask of a vocabulary of `tokens` tokens takes.
    pub fn words_for(tokens: u32) -> usize {
        (tokens as usize).div_ceil(32)
    }

    /// A mask that allows every token.
    pub fn all(tokens: u32) -> Self {
        let mut mask = Self { words: vec![u32::MAX; Self::words_for(tokens)], tokens };
        mask.trim();
        mask
    }

    /// A mask that allows no token.
    pub fn none(tokens: u32) -> Self {
        Self { words: vec![0; Self::words_for(tokens)], tokens }
    }

    /// A mask that allows exactly `allowed`.
    pub fn only(tokens: u32, allowed: impl IntoIterator<Item = u32>) -> Self {
        let mut mask = Self::none(tokens);
        for token in allowed {
            mask.set(token, true);
        }
        mask
    }

    fn trim(&mut self) {
        let rest = self.tokens % 32;
        if rest != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u32 << rest) - 1;
        }
    }

    pub fn tokens(&self) -> u32 {
        self.tokens
    }

    pub fn allowed(&self, token: u32) -> bool {
        token < self.tokens && self.words[(token / 32) as usize] >> (token % 32) & 1 != 0
    }

    pub fn set(&mut self, token: u32, allowed: bool) {
        assert!(token < self.tokens, "token {token} of {}", self.tokens);
        let (word, bit) = ((token / 32) as usize, token % 32);
        if allowed {
            self.words[word] |= 1 << bit;
        } else {
            self.words[word] &= !(1 << bit);
        }
    }

    /// The mask as the words the device reads.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Allows every token and returns the mask for reuse.
    pub fn fill(&mut self, allowed: bool) {
        self.words.fill(if allowed { u32::MAX } else { 0 });
        self.trim();
    }

    /// How many tokens are allowed.
    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

/// Whether the mask `words` allows `token`; a token beyond the words is not
/// allowed.
pub fn allows(words: &[u32], token: u32) -> bool {
    words.get((token / 32) as usize).is_some_and(|w| w >> (token % 32) & 1 != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mask_allows_what_was_set() {
        let mut mask = TokenMask::none(70);
        assert_eq!((mask.words().len(), mask.count()), (3, 0));
        mask.set(0, true);
        mask.set(69, true);
        assert!(mask.allowed(0) && mask.allowed(69) && !mask.allowed(68) && !mask.allowed(70));
        mask.set(0, false);
        assert_eq!(mask.count(), 1);
        let all = TokenMask::all(70);
        assert_eq!(all.count(), 70, "the bits past the end stay clear");
        assert!(all.allowed(69) && !all.allowed(70));
        assert_eq!(TokenMask::only(70, [3, 5]).count(), 2);
        assert!(allows(TokenMask::only(70, [3]).words(), 3) && !allows(&[], 3));
        let mut m = TokenMask::none(33);
        m.fill(true);
        assert_eq!(m.count(), 33);
    }
}
