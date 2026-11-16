//! The rows of the n-gram table that a step needs, and the least that is
//! known of each sequence for them.
//!
//! The rows of the table that a token needs depend on the token and the ones
//! before it, so they are asked for the moment the token is known
//! ([`NgramFeed::expect`], for a driver that knows better than the next step
//! does), not when the step reaches the layer that reads them.

use ares_gguf::Gguf;
use ares_io::{Class, Completion, Reader, RowGather, RowReader, RowTable};

use super::Error;
use crate::qwen4exp::Config;
use crate::qwen4exp::config::Ngram;
use crate::qwen4exp::reference::pieces;

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

/// The bytes a checkpoint keeps of what the feed knows of a sequence, for a
/// model with the n-gram block `ngram` ([`NgramFeed::save`]).
pub(super) fn saved_bytes(ngram: Option<&Ngram>) -> u64 {
    ngram.map_or(0, |g| 8 + 4 * g.ngram_size.saturating_sub(1) as u64)
}

struct Sequence {
    position: u32,
    /// The last tokens of the sequence, as many as the n-gram hash reads.
    recent: Vec<u32>,
}

struct Window {
    slot: u32,
    tokens: Vec<u32>,
}

/// Rows of the n-gram table that were asked for before the step that needs
/// them.
struct Ahead {
    rows: Vec<u64>,
    gather: RowGather,
}

/// The reader of the table and the sequences it reads for.
pub(super) struct NgramFeed {
    ngram: Option<Ngram>,
    /// The table is read through a reader of its own: the expert store takes
    /// every completion of its reader for itself.
    rows: Option<(Reader, RowReader)>,
    gather: RowGather,
    /// The rows asked for ahead of the next step, and gathers that were given
    /// up with reads still in flight.
    ahead: Option<Ahead>,
    stale: Vec<RowGather>,
    reads: u64,
    completions: Vec<Completion>,
    sequences: Vec<Sequence>,
    /// The draft windows of the last step.
    windows: Vec<Window>,
}

impl NgramFeed {
    /// A feed for `slots` sequences with a cache of `cache` bytes of rows.
    pub(super) fn open(gguf: &Gguf, c: &Config, cache: usize, slots: u32) -> Result<Self, Error> {
        let rows = match &c.ngram {
            Some(_) => {
                let info = gguf
                    .tensor("per_layer_token_embd.weight")
                    .ok_or_else(|| invalid("the model has no n-gram table"))?;
                let reader = Reader::open_gguf(gguf, ares_io::Config::default())?;
                let table = RowTable::for_reader(&reader, info)?;
                let row_reader = RowReader::new(&reader, table, cache);
                Some((reader, row_reader))
            }
            None => None,
        };
        Ok(Self {
            ngram: c.ngram.clone(),
            rows,
            gather: RowGather::new(),
            ahead: None,
            stale: Vec::new(),
            reads: 0,
            completions: Vec::new(),
            sequences: (0..slots).map(|_| Sequence { position: 0, recent: Vec::new() }).collect(),
            windows: Vec::new(),
        })
    }

    /// The cell the next row of the sequence in `slot` takes.
    pub(super) fn position(&self, slot: u32) -> u32 {
        self.sequences.get(slot as usize).map_or(0, |s| s.position)
    }

    /// Starts the sequence in `slot` over: its next row is cell 0.
    pub(super) fn restart(&mut self, slot: u32) -> Result<(), Error> {
        let sequence = self
            .sequences
            .get_mut(slot as usize)
            .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
        sequence.position = 0;
        sequence.recent.clear();
        Ok(())
    }

    /// The rows of `slot` start at cell `position`: a sequence that starts at
    /// cell 0 starts over, any other must be where the sequence is.
    pub(super) fn begin_sequence(&mut self, slot: u32, position: u32) -> Result<(), Error> {
        if position == 0 {
            return self.restart(slot);
        }
        match self.sequences.get(slot as usize) {
            Some(sequence) if sequence.position == position => Ok(()),
            Some(sequence) => Err(invalid(format!(
                "the rows of slot {slot} start at cell {position} and its sequence is at cell {}",
                sequence.position
            ))),
            None => Err(invalid(format!("slot {slot} of the session"))),
        }
    }

    /// The bytes of what the feed knows of one sequence, as a checkpoint
    /// keeps it ([`NgramFeed::save`]): its position, the number of its last
    /// tokens, and room for as many of them as the hash reads. 0 for a model
    /// without the n-gram block.
    pub(super) fn saved_bytes(&self) -> u64 {
        saved_bytes(self.ngram.as_ref())
    }

    /// What the feed knows of the sequence in `slot`: [`NgramFeed::saved_bytes`]
    /// bytes, for a checkpoint of the sequence.
    pub(super) fn save(&self, slot: u32) -> Result<Vec<u8>, Error> {
        let sequence = self
            .sequences
            .get(slot as usize)
            .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
        let mut out = vec![0u8; self.saved_bytes() as usize];
        if out.is_empty() {
            return Ok(out);
        }
        out[..4].copy_from_slice(&sequence.position.to_le_bytes());
        out[4..8].copy_from_slice(&(sequence.recent.len() as u32).to_le_bytes());
        for (word, token) in out[8..].chunks_exact_mut(4).zip(&sequence.recent) {
            word.copy_from_slice(&token.to_le_bytes());
        }
        Ok(out)
    }

    /// The sequence in `slot` goes on from what [`NgramFeed::save`] gave of
    /// another (a checkpoint the slot was restored from): its rows start
    /// where that sequence was, after the same tokens.
    pub(super) fn restore(&mut self, slot: u32, saved: &[u8]) -> Result<(), Error> {
        if saved.len() as u64 != self.saved_bytes() {
            return Err(invalid(format!(
                "{} bytes of an n-gram sequence; the feed keeps {}",
                saved.len(),
                self.saved_bytes()
            )));
        }
        if saved.is_empty() {
            return Ok(());
        }
        let word = |at: usize| {
            u32::from_le_bytes([saved[at], saved[at + 1], saved[at + 2], saved[at + 3]])
        };
        let count = word(4) as usize;
        if 8 + 4 * count > saved.len() {
            return Err(invalid(format!("{count} tokens of an n-gram sequence")));
        }
        let sequence = self
            .sequences
            .get_mut(slot as usize)
            .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
        sequence.position = word(0);
        sequence.recent = (0..count).map(|i| word(8 + 4 * i)).collect();
        Ok(())
    }

    /// Whether draft windows wait to be settled.
    pub(super) fn has_windows(&self) -> bool {
        !self.windows.is_empty()
    }

    /// The rows of the table that `tokens` need, one after the other, after
    /// the sequence in `slot` as it is now.
    fn table_rows(&self, slot: u32, tokens: &[u32], out: &mut Vec<u64>) -> Result<(), Error> {
        let Some(ngram) = &self.ngram else { return Ok(()) };
        let sequence = self
            .sequences
            .get(slot as usize)
            .ok_or_else(|| invalid(format!("slot {slot} of the session")))?;
        let mut before = sequence.recent.clone();
        for &token in tokens {
            out.extend(pieces::ngram_rows(ngram, token, &before));
            before.push(token);
        }
        Ok(())
    }

    /// Asks for the rows a step with `parts` (slot and tokens) needs, unless
    /// they were asked for already.
    pub(super) fn ask(&mut self, parts: &[(u32, &[u32])]) -> Result<(), Error> {
        let mut table_rows = Vec::new();
        for (slot, tokens) in parts {
            self.table_rows(*slot, tokens, &mut table_rows)?;
        }
        match self.ahead.take() {
            Some(ahead) if ahead.rows == table_rows => {
                let last = std::mem::replace(&mut self.gather, ahead.gather);
                self.retire(last);
            }
            ahead => {
                if let Some(ahead) = ahead {
                    self.retire(ahead.gather);
                }
                let last = std::mem::replace(&mut self.gather, RowGather::new());
                self.retire(last);
                if let Some((reader, row_reader)) = &mut self.rows {
                    self.reads += 1;
                    row_reader.start(
                        reader,
                        &table_rows,
                        self.reads << 20,
                        Class::Demand,
                        &mut self.gather,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Says what the next step will most likely be: for each sequence its
    /// slot and its next tokens, in the order the step will have them. The
    /// rows those tokens need are read now, so the step finds them there; a
    /// step with other tokens reads its own.
    pub(super) fn expect(&mut self, next: &[(u32, &[u32])]) -> Result<(), Error> {
        if self.ngram.is_none() {
            return Ok(());
        }
        let mut table_rows = Vec::new();
        for (slot, tokens) in next {
            self.table_rows(*slot, tokens, &mut table_rows)?;
        }
        if self.ahead.as_ref().is_some_and(|ahead| ahead.rows == table_rows) {
            return Ok(());
        }
        if let Some(ahead) = self.ahead.take() {
            self.retire(ahead.gather);
        }
        let Some((reader, row_reader)) = &mut self.rows else { return Ok(()) };
        let mut gather = RowGather::new();
        self.reads += 1;
        row_reader.start(reader, &table_rows, self.reads << 20, Class::Demand, &mut gather)?;
        self.ahead = Some(Ahead { rows: table_rows, gather });
        Ok(())
    }

    /// Keeps a gather that is given up until its reads have come back: their
    /// buffers return to the reader's pool and their rows go to the cache.
    fn retire(&mut self, gather: RowGather) {
        if gather.pending() > 0 {
            self.stale.push(gather);
        }
    }

    /// Hands the reads of table rows that have completed to their gathers.
    /// Never waits.
    pub(super) fn poll(&mut self) {
        let Some((reader, row_reader)) = &mut self.rows else { return };
        reader.poll(&mut self.completions);
        for completion in self.completions.drain(..) {
            let mut left = row_reader.accept(&mut self.gather, completion);
            if let (Some(completion), Some(ahead)) = (left.take(), &mut self.ahead) {
                left = row_reader.accept(&mut ahead.gather, completion);
            }
            for stale in &mut self.stale {
                let Some(completion) = left.take() else { break };
                left = row_reader.accept(stale, completion);
            }
        }
        self.stale.retain(|gather| gather.pending() > 0);
    }

    /// Whether the rows of the step have arrived.
    pub(super) fn is_done(&self) -> bool {
        self.gather.is_done()
    }

    /// The rows of the step, in head order, once they have arrived.
    pub(super) fn rows(&self) -> Result<&[u8], Error> {
        match self.gather.result() {
            Ok(()) => Ok(self.gather.rows()),
            Err(error) => Err(invalid(format!("reading rows of the n-gram table: {error}"))),
        }
    }

    /// The rows of a sequence that are certain are now behind it.
    pub(super) fn advance(&mut self, slot: u32, tokens: &[u32]) {
        let keep = self.ngram.as_ref().map_or(0, |g| g.ngram_size.saturating_sub(1));
        let sequence = &mut self.sequences[slot as usize];
        sequence.position += tokens.len() as u32;
        sequence.recent.extend_from_slice(tokens);
        let surplus = sequence.recent.len().saturating_sub(keep);
        sequence.recent.drain(..surplus);
    }

    /// The rows of `slot` are a draft window: how many count is said by
    /// [`NgramFeed::settle`].
    pub(super) fn hold(&mut self, slot: u32, tokens: &[u32]) {
        self.windows.push(Window { slot, tokens: tokens.to_vec() });
    }

    /// Settles the draft windows held: `accepted[i]` rows of the `i`-th count.
    pub(super) fn settle(&mut self, accepted: &[u32]) -> Result<(), Error> {
        if accepted.len() != self.windows.len() {
            return Err(invalid(format!(
                "{} outcomes for {} draft windows",
                accepted.len(),
                self.windows.len()
            )));
        }
        for (window, count) in self.windows.iter().zip(accepted) {
            if *count as usize > window.tokens.len() {
                return Err(invalid(format!(
                    "{count} accepted rows of a window of {}",
                    window.tokens.len()
                )));
            }
        }
        let windows = std::mem::take(&mut self.windows);
        for (window, count) in windows.iter().zip(accepted) {
            self.advance(window.slot, &window.tokens[..*count as usize]);
        }
        Ok(())
    }
}
