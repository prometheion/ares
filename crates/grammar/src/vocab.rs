//! The vocabulary the engine matches a grammar against: the bytes every token
//! stands for, and which tokens end a reply.

use std::sync::Arc;

use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

use crate::Error;

/// The tokens of a model as the grammar engine sees them. Cheap to clone: the
/// tables behind it are shared.
#[derive(Clone)]
pub struct Vocabulary {
    env: TokEnv,
    size: u32,
    ends: Arc<[u32]>,
}

impl std::fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocabulary").field("size", &self.size).field("ends", &self.ends).finish()
    }
}

impl Vocabulary {
    /// Builds the vocabulary from the bytes of each token: `pieces[i]` is the text
    /// of token `i`. A token that stands for no text (a control token, or one
    /// the vocabulary never uses) has an empty piece, and no grammar allows it.
    /// `ends` are the tokens that end a reply: a grammar allows exactly them
    /// where it can end, and nowhere else (an end token is never text, whatever
    /// its piece).
    ///
    /// # Errors
    ///
    /// [`Error::Vocabulary`] when there are no tokens, or an end token is not
    /// one of them.
    pub fn new(pieces: &[Vec<u8>], ends: &[u32]) -> Result<Self, Error> {
        let size = u32::try_from(pieces.len())
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Vocabulary("the vocabulary has no tokens".into()))?;
        if let Some(bad) = ends.iter().find(|&&t| t >= size) {
            return Err(Error::Vocabulary(format!(
                "end token {bad} is not in a vocabulary of {size} tokens"
            )));
        }
        // The trie wants one end token; the others are named next.
        let info = TokRxInfo::new(size, ends.first().copied().unwrap_or(toktrie::INVALID_TOKEN));
        let mut words = pieces.to_vec();
        for &end in ends {
            words[end as usize].clear();
        }
        let mut trie = TokTrie::from(&info, &words);
        if !ends.is_empty() {
            trie = trie.with_eos_tokens(ends);
        }
        Ok(Self { env: Arc::new(ApproximateTokEnv::new(trie)), size, ends: ends.into() })
    }

    /// How many tokens there are.
    pub fn len(&self) -> u32 {
        self.size
    }

    /// Whether there are no tokens (never: [`Vocabulary::new`] refuses it).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The tokens that end a reply.
    pub fn ends(&self) -> &[u32] {
        &self.ends
    }

    /// The bytes of token `token`.
    pub fn piece(&self, token: u32) -> &[u8] {
        self.env.tok_trie().token(token)
    }

    /// The words of a mask over this vocabulary.
    pub fn mask_words(&self) -> usize {
        (self.size as usize).div_ceil(32)
    }

    pub(crate) fn env(&self) -> &TokEnv {
        &self.env
    }
}

impl Vocabulary {
    /// The token whose text is exactly `text`, if there is one.
    pub fn find(&self, text: &[u8]) -> Option<u32> {
        self.env.tok_trie().token_id(text)
    }
}
