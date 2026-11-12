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
    /// Past the reasoning and free, until the text has the trigger; these are
    /// the last bytes of it, as many as can still become the trigger.
    Waiting(Vec<u8>),
    /// The grammar's first token is the token at this index of the reply.
    Bound(usize),
}

/// A structure being followed by one reply.
///
/// A structure with a trigger leaves the answer free until its text has the
/// trigger, and binds what follows it; a reply that may open its reasoning
/// itself is then free from its first token.
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
    /// The text the grammar waits for; empty when it waits for none.
    trigger: Vec<u8>,
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
            ares_api::Shape::Lark(text) => Source::Lark(text),
        };
        let mut grammar = self.compile(source)?;
        let (open, close) = match &structure.start {
            Start::Reply => (None, Vec::new()),
            Start::AfterReasoning { open, close } => {
                if close.is_empty() {
                    return Err(Error::Compile(
                        "the tag that closes the reasoning is empty".into(),
                    ));
                }
                (open.as_ref().map(|open| open.as_bytes().to_vec()), close.as_bytes().to_vec())
            }
        };
        let trigger = match &structure.trigger {
            Some(trigger) if trigger.is_empty() => {
                return Err(Error::Compile("the trigger of the grammar is empty".into()));
            }
            Some(trigger) => trigger.as_bytes().to_vec(),
            None => Vec::new(),
        };
        let mut guide = Guide {
            cursor: grammar.start(),
            vocab: self.vocabulary().clone(),
            open,
            close,
            trigger,
            seen: 0,
            phase: Phase::Bound(0),
            lead: HashMap::new(),
            start: None,
            path: Vec::new(),
            last: None,
        };
        guide.phase = guide.initial();
        Ok(guide)
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
                    self.answer(index + 1)
                } else {
                    let keep = self.close.len() - 1;
                    bytes.drain(..bytes.len().saturating_sub(keep));
                    Phase::Reasoning(bytes)
                }
            }
            Phase::Waiting(tail) => self.wait(tail, index, token),
            Phase::Lead(matched) => {
                let waits = !self.trigger.is_empty();
                if !waits && matched == 0 && self.starts_grammar(token)? {
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
                } else if waits {
                    // Not the tag after all: the answer, which began with what
                    // looked like the tag.
                    let tail = open[..matched].to_vec();
                    self.wait(tail, index, token)
                } else {
                    // Neither the tag nor the grammar: the grammar will refuse it.
                    Phase::Bound(index)
                }
            }
        })
    }

    /// The phase of the answer whose first token is the one at `index`: bound,
    /// or free until the trigger.
    fn answer(&self, index: usize) -> Phase {
        if self.trigger.is_empty() { Phase::Bound(index) } else { Phase::Waiting(Vec::new()) }
    }

    /// The phase after `token` at `index` of an answer that waits for the
    /// trigger and ends with `tail`.
    fn wait(&self, tail: Vec<u8>, index: usize, token: u32) -> Phase {
        let mut bytes = tail;
        bytes.extend_from_slice(self.vocab.piece(token));
        if find(&bytes, &self.trigger).is_some() {
            // As for the tag that closes the reasoning: the grammar is
            // checked from the next token on.
            Phase::Bound(index + 1)
        } else {
            let keep = self.trigger.len() - 1;
            bytes.drain(..bytes.len().saturating_sub(keep));
            Phase::Waiting(bytes)
        }
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
            (_, true) => self.answer(0),
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
            if let Phase::Lead(matched) = phase
                && self.trigger.is_empty()
            {
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
            Phase::Reasoning(_) | Phase::Waiting(_) => None,
            Phase::Lead(matched) if free == drafts.len() && self.trigger.is_empty() => {
                Some(self.lead_words(matched)?)
            }
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

    /// How many rows of a window the structure leaves free, counted from the
    /// first: row `k` of the window follows `reply` and the first `k` of
    /// `tokens`, and it is free when [`mask`](Guide::mask) has no mask for it
    /// because the grammar does not bind there (the reply is thinking, or
    /// waits for its trigger). The count stops at the first row that is not
    /// free, so it is at most one more than `tokens` has.
    ///
    /// # Errors
    ///
    /// [`Error::Compile`] when the engine fails to follow the grammar.
    pub fn free_rows(&mut self, reply: &[u32], tokens: &[u32]) -> Result<usize, Error> {
        self.see(reply)?;
        let mut phase = self.phase.clone();
        let mut rows = 0;
        while self.free(&phase) {
            let Some(&token) = tokens.get(rows) else { return Ok(rows + 1) };
            phase = self.fold(phase, reply.len() + rows, token)?;
            rows += 1;
        }
        Ok(rows)
    }

    /// Whether a reply in `phase` may go on with any token.
    fn free(&self, phase: &Phase) -> bool {
        match phase {
            Phase::Reasoning(_) | Phase::Waiting(_) => true,
            // A reply that may still open its reasoning is held to the tag or
            // to the grammar's first tokens, unless the grammar waits for a
            // trigger: then the answer it may turn out to be is free as well.
            Phase::Lead(_) => !self.trigger.is_empty(),
            Phase::Bound(_) => false,
        }
    }
}

/// The position of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
