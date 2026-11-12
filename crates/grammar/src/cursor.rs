//! A position in a grammar, moved along the tokens of a reply and its drafts.

use llguidance::Matcher;

use crate::{Error, Vocabulary};

/// A position in a compiled grammar.
///
/// The cursor knows the tokens it has taken. [`Cursor::goto`] puts it after a
/// given sequence of tokens by taking back the ones it took that the sequence
/// does not start with and taking the new ones, so a caller that asks about
/// the reply and a few drafts after it, and then about the same reply with
/// other drafts, pays for what changed.
pub struct Cursor {
    matcher: Matcher,
    vocab: Vocabulary,
    path: Vec<u32>,
}

impl std::fmt::Debug for Cursor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cursor").field("tokens", &self.path.len()).finish_non_exhaustive()
    }
}

impl Cursor {
    pub(crate) fn new(matcher: Matcher, vocab: Vocabulary) -> Self {
        Self { matcher, vocab, path: Vec::new() }
    }

    /// Puts the cursor after `tokens`, counted from the start of the grammar.
    /// Returns how many of them the grammar takes: fewer than `tokens.len()`
    /// means the token at that index is not allowed there, and the cursor
    /// stands after the ones before it.
    pub fn goto(&mut self, tokens: &[u32]) -> usize {
        let same = self.path.iter().zip(tokens).take_while(|(a, b)| a == b).count();
        if same < self.path.len() {
            // The engine keeps every position it has been in; going back
            // cannot fail for a count it has taken.
            let back = self.path.len() - same;
            if self.matcher.rollback(back).is_err() {
                self.path.clear();
                return self.restart(tokens);
            }
            self.path.truncate(same);
        }
        match self.matcher.try_consume_tokens(&tokens[same..]) {
            Ok(taken) => {
                self.path.extend_from_slice(&tokens[same..same + taken]);
                same + taken
            }
            Err(_) => same,
        }
    }

    /// The cursor cannot go back (the engine refused): start again from the
    /// beginning of the grammar.
    fn restart(&mut self, tokens: &[u32]) -> usize {
        if self.matcher.reset().is_err() {
            return 0;
        }
        match self.matcher.try_consume_tokens(tokens) {
            Ok(taken) => {
                self.path.extend_from_slice(&tokens[..taken]);
                taken
            }
            Err(_) => 0,
        }
    }

    /// The tokens the grammar allows next, as the words of a mask over the
    /// vocabulary (bit `t % 32` of word `t / 32` is token `t`). The end tokens
    /// are in it where the grammar can end and nowhere else.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] when the engine fails in the middle of the grammar
    /// (it ran out of the fuel it gives a grammar, for one).
    pub fn mask(&mut self) -> Result<Vec<u32>, Error> {
        let set = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| Error::Compile(format!("the grammar cannot be followed further: {e}")))?;
        let mut words = set.as_slice().to_vec();
        words.resize(self.vocab.mask_words(), 0);
        Ok(words)
    }

    /// Whether the reply so far is a whole sentence of the grammar: an end
    /// token may come now.
    pub fn accepting(&mut self) -> bool {
        self.matcher.is_accepting().unwrap_or(false)
    }

    /// How many tokens the cursor has taken.
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Whether the cursor is at the start.
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}
