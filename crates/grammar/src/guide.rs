//! A reply's walk through a structure: where the grammar starts to bind, and
//! the tokens it allows from there on.

use std::collections::HashMap;

use ares_api::{Start, Structure};

use crate::{Compiler, Cursor, Error, Source, Vocabulary};

/// Where a reply is relative to the text its grammar binds.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    /// No token of the grammar yet; the reply so far is this many bytes of the
    /// tag that opens the reasoning.
    Lead(usize),
    /// In the reasoning; these are the last bytes of it, as many as can still
    /// become the closing tag.
    Reasoning(Vec<u8>),
    /// The grammar's first token is the token at this index of the reply.
    Bound(usize),
}

/// A structure being followed by one reply.
///
/// [`mask`](Guide::mask) and [`admitted`](Guide::admitted) each take the tokens of the reply so far and the
/// drafts that follow them, and answer for the position after the last draft.
/// The guide moves its cursor to that position and keeps what it worked out
/// about the reply, so asking again for the same reply with other drafts costs
/// what changed.
pub struct Guide {
    cursor: Cursor,
    vocab: Vocabulary,
    open: Option<Vec<u8>>,
    close: Vec<u8>,
    /// What the guide knows of the reply it has been given: the phase after
    /// `seen` tokens.
    seen: usize,
    phase: Phase,
    /// The tokens that may come while the reply is in the lead, by how much of
    /// the opening tag it has: the tokens that continue the tag, and for the
    /// start also the first tokens of the grammar.
    lead: HashMap<usize, Vec<u32>>,
    /// The tokens the grammar allows first.
    start: Option<Vec<u32>>,
    path: Vec<u32>,
    /// The last mask worked out and the path (the whole reply and drafts) it is
    /// for.
    last: Option<(Vec<u32>, Option<Vec<u32>>)>,
}

impl std::fmt::Debug for Guide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Guide").field("seen", &self.seen).field("phase", &self.phase).finish()
    }
}

impl Compiler {
    /// The guide for a reply that follows `structure`.
    ///
    /// # Errors
    ///
    /// The errors of [`Compiler::compile`].
    pub fn guide(&self, structure: &Structure) -> Result<Guide, Error> {
        let source = match &structure.shape {
            ares_api::Shape::Gbnf(text) => Source::Gbnf(text),
            ares_api::Shape::JsonSchema(text) => Source::JsonSchema(text),
        };
        let mut grammar = self.compile(source)?;
        let (open, close, phase) = match &structure.start {
            Start::Reply => (None, Vec::new(), Phase::Bound(0)),
            Start::AfterReasoning { open: None, close } => {
                (None, close.as_bytes().to_vec(), Phase::Reasoning(Vec::new()))
            }
            Start::AfterReasoning { open: Some(open), close } => {
                (Some(open.as_bytes().to_vec()), close.as_bytes().to_vec(), Phase::Lead(0))
            }
        };
        if matches!(phase, Phase::Reasoning(_) | Phase::Lead(_)) && close.is_empty() {
            return Err(Error::Compile("the tag that closes the reasoning is empty".into()));
        }
        Ok(Guide {
            cursor: grammar.start(),
            vocab: self.vocabulary().clone(),
            open,
            close,
            seen: 0,
            phase,
            lead: HashMap::new(),
            start: None,
            path: Vec::new(),
            last: None,
        })
    }
}

impl Guide {
    /// The phase after `token`, which is the token at `index` of the reply.
    fn fold(&mut self, phase: Phase, index: usize, token: u32) -> Result<Phase, Error> {
        Ok(match phase {
            Phase::Bound(from) => Phase::Bound(from),
            Phase::Reasoning(tail) => {
                let mut bytes = tail;
                bytes.extend_from_slice(self.vocab.piece(token));
                if find(&bytes, &self.close).is_some() {
                    // Text after the tag in the same token is not checked
                    // against the grammar: tokenizers write the tag as a token
                    // of its own.
                    Phase::Bound(index + 1)
                } else {
                    let keep = self.close.len() - 1;
                    bytes.drain(..bytes.len().saturating_sub(keep));
                    Phase::Reasoning(bytes)
                }
            }
            Phase::Lead(matched) => {
                if matched == 0 && self.starts_grammar(token)? {
                    return Ok(Phase::Bound(index));
                }
                let open = self.open.as_deref().expect("a lead has a tag");
                let piece = self.vocab.piece(token);
                if !piece.is_empty() && open[matched..].starts_with(piece) {
                    let matched = matched + piece.len();
                    if matched == open.len() {
                        Phase::Reasoning(Vec::new())
                    } else {
                        Phase::Lead(matched)
                    }
                } else {
                    // Neither the tag nor the grammar: the grammar will refuse it.
                    Phase::Bound(index)
                }
            }
        })
    }

    /// Whether the grammar allows `token` as its first.
    fn starts_grammar(&mut self, token: u32) -> Result<bool, Error> {
        if self.start.is_none() {
            self.cursor.goto(&[]);
            self.start = Some(self.cursor.mask()?);
        }
        Ok(self.start.as_deref().is_some_and(|words| crate::mask_allows(words, token)))
    }

    /// The tokens a reply may continue with when it has `matched` bytes of the
    /// opening tag: the tokens that continue the tag, and at the start the
    /// first tokens of the grammar.
    fn lead_words(&mut self, matched: usize) -> Result<Vec<u32>, Error> {
        if let Some(words) = self.lead.get(&matched) {
            return Ok(words.clone());
        }
        let mut words = vec![0u32; self.vocab.mask_words()];
        let rest = &self.open.as_deref().expect("a lead has a tag")[matched..];
        for token in 0..self.vocab.len() {
            let piece = self.vocab.piece(token);
            if !piece.is_empty() && rest.starts_with(piece) {
                words[(token / 32) as usize] |= 1 << (token % 32);
            }
        }
        if matched == 0 {
            self.starts_grammar(0)?;
            for (word, grammar) in words.iter_mut().zip(self.start.iter().flatten()) {
                *word |= grammar;
            }
        }
        self.lead.insert(matched, words.clone());
        Ok(words)
    }

    /// Takes the phase through the tokens of `reply` the guide has not seen.
    fn see(&mut self, reply: &[u32]) -> Result<(), Error> {
        if reply.len() < self.seen {
            // A different reply from the one the guide followed: start again.
            self.seen = 0;
            self.phase = self.initial();
        }
        for (index, &token) in reply.iter().enumerate().skip(self.seen) {
            let phase = std::mem::replace(&mut self.phase, Phase::Bound(0));
            self.phase = self.fold(phase, index, token)?;
        }
        self.seen = reply.len();
        Ok(())
    }

    fn initial(&self) -> Phase {
        match (&self.open, self.close.is_empty()) {
            (_, true) => Phase::Bound(0),
            (Some(_), false) => Phase::Lead(0),
            (None, false) => Phase::Reasoning(Vec::new()),
        }
    }

    /// The phase after `reply` and then `drafts`, and for each draft whether it
    /// is a token the reply may have there while it is not yet in the grammar.
    fn walk(&mut self, reply: &[u32], drafts: &[u32]) -> Result<(Phase, usize), Error> {
        self.see(reply)?;
        let mut phase = self.phase.clone();
        let mut free = drafts.len();
        for (i, &token) in drafts.iter().enumerate() {
            if let Phase::Lead(matched) = phase {
                let words = self.lead_words(matched)?;
                if !crate::mask_allows(&words, token) && free == drafts.len() {
                    free = i;
                }
            }
            phase = self.fold(phase, reply.len() + i, token)?;
        }
        Ok((phase, free))
    }

    /// The tokens the grammar allows after `reply` and `drafts`, as the words
    /// of a mask over the vocabulary; `None` when the reply is not bound by the
    /// grammar there (it is thinking), or when a draft the grammar refuses
    /// comes before: that position is never used.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] when the engine fails to follow the grammar.
    pub fn mask(&mut self, reply: &[u32], drafts: &[u32]) -> Result<Option<Vec<u32>>, Error> {
        self.path.clear();
        self.path.extend_from_slice(reply);
        self.path.extend_from_slice(drafts);
        if let Some((path, mask)) = &self.last
            && *path == self.path
        {
            return Ok(mask.clone());
        }
        let (phase, free) = self.walk(reply, drafts)?;
        let mask = match phase {
            Phase::Reasoning(_) => None,
            Phase::Lead(matched) if free == drafts.len() => Some(self.lead_words(matched)?),
            Phase::Lead(_) => None,
            Phase::Bound(from) => {
                let path = std::mem::take(&mut self.path);
                let grammar_part = &path[from.min(path.len())..];
                let mask = if free == drafts.len()
                    && self.cursor.goto(grammar_part) == grammar_part.len()
                {
                    Some(self.cursor.mask()?)
                } else {
                    None
                };
                self.path = path;
                mask
            }
        };
        self.last = Some((self.path.clone(), mask.clone()));
        Ok(mask)
    }

    /// How many of the leading `drafts` may follow `reply`: the drafts up to the
    /// first one the grammar refuses.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] when the engine fails to follow the grammar.
    pub fn admitted(&mut self, reply: &[u32], drafts: &[u32]) -> Result<usize, Error> {
        let (phase, free) = self.walk(reply, drafts)?;
        if free < drafts.len() {
            return Ok(free);
        }
        let Phase::Bound(from) = phase else { return Ok(drafts.len()) };
        self.path.clear();
        self.path.extend_from_slice(reply);
        self.path.extend_from_slice(drafts);
        let path = std::mem::take(&mut self.path);
        let grammar_part = &path[from.min(path.len())..];
        let taken = self.cursor.goto(grammar_part);
        self.path = path;
        // The tokens of the grammar before the drafts are the reply's.
        let before = reply.len().saturating_sub(from);
        let leading = from.saturating_sub(reply.len());
        Ok((leading + taken.saturating_sub(before)).min(drafts.len()))
    }
}

/// The position of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
